#!/usr/bin/env python3
"""
Training script for KG-LFM with wandb logging, checkpointing, and better validation.
"""

import logging
import argparse
import os
import time
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from datetime import datetime
from collections import defaultdict
import gc

import psutil

import torch
import torch.nn as nn
from torch.utils.data import DataLoader 
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import set_seed
import wandb
from tqdm.auto import tqdm

# Assuming these are defined in your project structure
from KG_LFM.configuration import load_yaml_config, ProjectConfig
from KG_LFM.model.KG_LFM_arch import KG_LFM, KG_LFMConfig, set_KGLM_model_args
from KG_LFM.utils.Dataloaders.pretrain_data import create_dataloader

# import abstract class abc
from abc import ABC, abstractmethod

class MetricsTracker(ABC):

    @abstractmethod
    def update(self, metrics: Dict[str, float], count: int = 1):
        pass

    @abstractmethod
    def get_averages(self) -> Dict[str, float]:
        pass

    @abstractmethod
    def reset(self):
        pass
    

class DefaultMetricsTracker(MetricsTracker):
    """Track and compute running averages of metrics using defaultdict."""

    def __init__(self):
        self.values = defaultdict(float)
        self.counts = defaultdict(int)

    def reset(self):
        """Resets the tracker."""
        self.values.clear()
        self.counts.clear()

    def update(self, metrics: Dict[str, float], count: int = 1):
        """Update metrics."""
        for key, value in metrics.items():
            self.values[key] += value * count
            self.counts[key] += count

    def get_averages(self) -> Dict[str, float]:
        """Get the current average of all tracked metrics."""
        return {key: self.values[key] / self.counts[key]
                for key in self.values if self.counts[key] > 0}

class KG_LFM_Trainer:
    """
    Advanced trainer for KG-LFM with comprehensive features for robust and reproducible training.
    """

    def __init__(self, 
                 config: ProjectConfig, 
                 run_name: Optional[str] = None, 
                 resume: Optional[bool] = False, 
                 enable_wandb: bool = True,
                 save_checkpoints: bool = True,
                 metrics_tracker: Optional[MetricsTracker] = DefaultMetricsTracker()
                ):
        self.config = config
        self.run_name = run_name or f"kg_lfm_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.resume = resume
        self.enable_wandb = enable_wandb
        self.save_checkpoints = save_checkpoints

        self.metrics_tracker : MetricsTracker = metrics_tracker

        # Initialize Accelerator and set seed for reproducibility
        kwargs = DistributedDataParallelKwargs(
            find_unused_parameters=False,
        )
        self.accelerator = Accelerator(
            log_with="wandb" if enable_wandb else None,
            # gradient_accumulation_steps=self.config.train_conf.gradient_accumulation_steps,
            kwargs_handlers=[kwargs]
        )
        set_seed(self.config.seed)

        self.device = self.accelerator.device

        # Setup logging
        self.logger = logging.getLogger(__name__)

        # Initialize state
        self.model : KG_LFM = None
        
        self.optimizer = None
        self.scheduler = None
        
        self.skip_train_dataloader = None  # Used to skip initial batches in the training dataloader
        self.train_dataloader: DataLoader = None
        self.val_dataloader: DataLoader = None
        self.test_dataloader: DataLoader = None
        self.dataloader_factory = None  # Store factory for epoch setting
        
        self.wandb_run_id = None

        self.global_step = 0
        self.start_epoch = 0
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0
        self.last_best_model_save_step = -1  # Track when we last saved a best model
        
        self.steps_train = self.config.train_conf.steps_train
        self.clip_grad_norm = self.config.train_conf.clip_grad_norm
        self.grad_accumulation_steps = self.config.train_conf.gradient_accumulation_steps
        
        self.train_iter = None  # Iterator for training dataloader
        
        self.percentage_eval = self.config.train_conf.eval_perc # Percentage of evaluation dataset to use for validation after each training step
        
        # Checkpoint frequency to reduce I/O overhead
        self.checkpoint_frequency = self.config.train_conf.checkpoint_frequency
        
        self.checkpoint_dir = Path(self.config.train_conf.checkpoint_dir + f"/{self.run_name}")

        self.accelerator.print(f"Trainer initialized on device: {self.device}")
        self.accelerator.print(f"World size: {self.accelerator.num_processes}")
        self.accelerator.print(f"Local rank: {self.accelerator.local_process_index}")
        self.accelerator.print(f"Is main process: {self.accelerator.is_main_process}")
        self.accelerator.print(f"Checkpoint frequency: every {self.checkpoint_frequency} epoch(s)")
        self.accelerator.print(f"Training steps per evaluation: {self.steps_train}")
        
        # Calculate and display effective batch size
        micro_batch_size = self.config.train_conf.dataloader.batch_size
        self.accelerator.print(f"Micro batch size per GPU: {micro_batch_size}")
        
        # Add safety check for distributed training
        if self.accelerator.num_processes > 1:
            self.accelerator.print(f"Distributed training with {self.accelerator.num_processes} processes")

    def setup_wandb(self):
        """Initialize wandb, potentially resuming a previous run."""
        if self.accelerator.is_main_process and self.enable_wandb:
            # if already initialized, just resume
            if wandb.run is not None:
                self.accelerator.print(f"Resuming existing wandb run: {wandb.run.id}")
            
            wandb_config = self.config.to_dict() # Assuming a method to convert config to dict
            
            run_id = self.wandb_run_id
            
            self.accelerator.init_trackers(
                project_name="kg-lfm-pretraining",
                config=wandb_config,
                init_kwargs={"wandb": {
                    "name": self.run_name,
                    "resume": "auto", 
                    "id": run_id
                    }}
            )
            self.logger.info("Wandb initialized successfully")
        
        # For non-main processes, just wait
        self.accelerator.wait_for_everyone()

    def setup_model(self):
        """Initialize the model with proper configuration."""
        self.accelerator.print("Setting up model...")
        
        # Profile model initialization
        model_config = KG_LFMConfig.from_pretrained(
            self.config.model.llm_model_name,
            trust_remote_code=True,
        )
        model_config = set_KGLM_model_args(model_config, self.config.model)
        self.model = KG_LFM(model_config)
        
        # Freeze layers if configured
        if not self.config.model.tune_language_model:
            self.accelerator.print("Freezing language model parameters.")
            for param in self.model.llm.parameters():
                param.requires_grad = False

        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        self.logger.info(f"Total parameters: {total_params:,}")
        self.logger.info(f"Trainable parameters: {trainable_params:,}")
        
        # Only log from main process to avoid duplicate logs
        if self.accelerator.is_main_process:
            self.accelerator.log({
                "model/total_parameters": total_params,
                "model/trainable_parameters": trainable_params,
            }, step=0)

    def setup_data(self):
        """Setup data loaders with distributed support."""
        self.accelerator.print("Setting up data loaders...")
        
        train, val, test = create_dataloader(
            self.config.dataset,
            self.config.train_conf.dataloader,
            tokenizer=self.model.tokenizer,
            split="all",
        )
        
        self.train_dataloader = train
        self.val_dataloader = val
        self.test_dataloader = test
        
        # if steps_train is float, convert to int
        if isinstance(self.steps_train, float):
            self.steps_train = int(len(self.train_dataloader) * self.steps_train)
            self.accelerator.print(f"Converted steps_train to {self.steps_train} based on dataset size.")
             
        
    def setup_optimizer_and_scheduler(self, num_epochs: int):
        """Setup optimizer and learning rate scheduler."""
        self.accelerator.print("Setting up optimizer and scheduler...")
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        
        self.optimizer = AdamW(
            trainable_params,
            lr=self.config.train_conf.learning_rate,
            weight_decay=self.config.train_conf.weight_decay
        )

        total_steps = (len(self.train_dataloader) * num_epochs) // self.accelerator.num_processes

        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=0.1,
            patience=self.config.train_conf.early_stopping_patience // 2,
            verbose=True
        )

        self.target_lr = self.config.train_conf.learning_rate

    def prepare_for_training(self):
        """Prepare all components with accelerator."""
        self.accelerator.print("Preparing components with Accelerator...")
        
        (self.model, self.optimizer, self.scheduler,
         self.train_dataloader,self.val_dataloader, self.test_dataloader, 
        ) = self.accelerator.prepare(
                self.model, self.optimizer,  self.scheduler,
                self.train_dataloader, self.val_dataloader, self.test_dataloader
            )

    def clear_memory(self):
        """Clear GPU memory cache and run garbage collection."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    def compute_train_metrics(self, loss: torch.Tensor) -> Dict[str, float]:
        """Compute metrics from loss."""
        # Ensure loss is a scalar tensor and safely convert to float
        if hasattr(loss, 'item'):
            loss_val = loss.item()
        else:
            loss_val = float(loss)
            
        metrics = {
            'training_loss': loss_val,
            'learning_rate': self.optimizer.param_groups[0]['lr']
        }
        
        return metrics
    
    def compute_val_metrics(self, loss: torch.Tensor) -> Dict[str, float]:
        """Compute validation metrics from loss."""
        # Ensure loss is a scalar tensor and safely convert to float
        if hasattr(loss, 'item'):
            loss_val = loss.item()
        else:
            loss_val = float(loss)

        metrics = {
            'validation_loss': loss_val,
            'learning_rate': self.optimizer.param_groups[0]['lr']
        }

        return metrics

    def _evaluation_loop(self, dataloader: torch.utils.data.DataLoader, description: str, percentage_eval: float) -> Dict[str, float]:
        """Generic evaluation loop for validation or testing with distributed support."""
        self.model.eval()
        
        # Calculate number of batches to process instead of samples
        total_batches = len(dataloader)
        num_batches = max(1, int(total_batches * percentage_eval))
        
        self.accelerator.print(f"Evaluating {num_batches}/{total_batches} batches ({num_batches/total_batches*100:.1f}%)")
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(dataloader, desc=description, disable=not self.accelerator.is_local_main_process, total=num_batches)):
                if batch_idx >= num_batches:
                    break
                    
                try:
                    # DeepSpeed handles mixed precision automatically 
                    outputs = self.model(
                        input_ids=batch['input_ids'],
                        graphs=batch['graphs'],
                        attention_mask=batch['attention_mask'],
                        labels=batch['input_ids'],
                        use_cache=False,
                    )
                    
                    # Gather loss across all devices
                    loss = self.accelerator.gather(outputs.loss).mean()
                    metrics = self.compute_val_metrics(loss)
                    self.metrics_tracker.update(metrics, batch['input_ids'].size(0))
                    
                    # Delete outputs to free memory immediately
                    del outputs, loss
                    
                except Exception as e:
                    self.accelerator.print(f"Error in evaluation batch {batch_idx}: {e}")
                    # Clear cache on error
                    self.clear_memory()
                    # Ensure synchronization even on error
                    self.accelerator.wait_for_everyone()
                    continue
        
        # Final cleanup
        self.clear_memory()
        
        # Wait for all processes to complete evaluation before returning
        self.accelerator.wait_for_everyone()
        
        averages = self.metrics_tracker.get_averages()
        self.metrics_tracker.reset()  # Reset tracker for next evaluation
        
        return averages

    def train_step(self):
        """Train for one step."""
        # Ensure model is in training mode and gradients are enabled
        self.model.train()
        
        # Clear any leftover gradients
        self.optimizer.zero_grad()
        
        # Create an iterator from the dataloader
        if self.train_iter is None:
            if self.skip_train_dataloader is not None:
                self.train_iter = iter(self.skip_train_dataloader)
            else:
                self.train_iter = iter(self.train_dataloader)
        
        progress_bar = tqdm(
            desc=f"Training Step {self.global_step // self.steps_train}",
            disable=not self.accelerator.is_local_main_process,
            total=self.steps_train
        )

        total_loss = torch.tensor(0.0, device=self.device, requires_grad=False)

        step = 0
        while step < self.steps_train:
            # get batch, reset iterator if needed
            try:
                batch = next(self.train_iter)
            except StopIteration:
                self.logger.info(f"Resetting dataloader iterator at step {step + 1}")
                self.train_iter.close()
                self.train_iter = iter(self.train_dataloader)
                self.skip_train_dataloader = None  # Reset skip dataloader
                batch = next(self.train_iter)

            # self.accelerator.print(f"Processing step {step + 1}/{self.steps_train} on process {self.accelerator.local_process_index}")
            
            # Standard forward pass
            outputs = self.model(
                input_ids=batch['input_ids'],
                graphs=batch['graphs'],
                attention_mask=batch['attention_mask'],
                labels=batch['input_ids'],
                use_cache=False,
            )
            language_loss = outputs.loss
            RVQ_loss = outputs.RVQ_loss
            loss = language_loss + RVQ_loss
            
            # print all processes' losses for debugging
            self.logger.debug(f"Step {step + 1}/{self.steps_train} - Language Loss: {language_loss.item():.4f}, RVQ Loss: {RVQ_loss.item():.4f}, Total Loss: {loss.item():.4f}")
            
            # create a tensor to accumulate from all processes to skip all backward passes if loss is NaN or Inf
            is_valid_loss = torch.tensor(not (torch.isnan(loss) or torch.isinf(loss)), device=self.device, dtype=torch.bool)
            
            # Gather across all processes to check for NaN/Inf
            is_valid_loss = self.accelerator.gather(is_valid_loss).all()
            # Skip invalid losses
            if not is_valid_loss:
                self.logger.warning(f"NaN or Inf detected at step {step + 1}, skipping backward.")
            else:
                loss /= self.grad_accumulation_steps  # Scale loss for gradient accumulation
                total_loss += loss
                self.accelerator.backward(loss)
            
            self.logger.debug(f"Step {step + 1}/{self.steps_train} - Final Loss {loss}")

            if (step + 1) % self.grad_accumulation_steps == 0 or (step + 1) == self.steps_train:
                # Clip gradients if configured
                if self.clip_grad_norm is not None:
                    self.accelerator.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
                    
                self.optimizer.step()
                # self.scheduler.step()
                self.optimizer.zero_grad()
                
                # update global step and logging
                metrics = self.compute_train_metrics(total_loss)
                total_loss = torch.tensor(0.0, device=self.device, requires_grad=False)
                self.metrics_tracker.update(metrics, batch['input_ids'].size(0))
                log_metrics = {f"train/{k}": v for k, v in metrics.items()}
                self.accelerator.log(log_metrics, step=self.global_step)

            self.global_step += 1
            progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})
            progress_bar.update(1)
            step += 1
            
        progress_bar.close()  # Close progress bar properly
        
        # Clear memory after training step
        self.clear_memory()

        averages = self.metrics_tracker.get_averages()
        self.metrics_tracker.reset()  # Reset tracker for next step
        
        return averages

    def save_model(self, sub_path: Optional[Path] = None):
        """Save the model and training state.
        Args:
            sub_path (Optional[Path]): Subdirectory to save the model in.
        """
        if self.accelerator.is_main_process:
            self.accelerator.print(f"Saving model to {self.checkpoint_dir}")
        
        # Ensure checkpoint directory exists
        model_dir = self.checkpoint_dir if sub_path is None else self.checkpoint_dir / sub_path
        model_dir.mkdir(parents=True, exist_ok=True)

        try:
            unwrapped_model: KG_LFM = self.accelerator.unwrap_model(self.model)
            unwrapped_model.save_pretrained(model_dir)
            self.logger.info(f"Model saved successfully to {model_dir}")
        except Exception as e:
            self.logger.error(f"Error saving model: {e}")
            raise
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint with proper distributed coordination."""
        checkpoint_dir = self.checkpoint_dir
        
        # Ensure all processes wait at the same point before saving
        self.accelerator.wait_for_everyone()
        
        # Create checkpoint directory on main process
        if self.accelerator.is_main_process:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Wait for directory creation to complete
        self.accelerator.wait_for_everyone()
        
        self.logger.info(f"Saving checkpoint for epoch {epoch + 1}...")
        
        # Clear memory before saving checkpoint to prevent OOM
        self.clear_memory()
        
        # show RAM usage before saving
        cpu_ram = psutil.virtual_memory().percent
        self.logger.info(f"CPU RAM usage before saving checkpoint: {cpu_ram}%")
        
        try:
            self.accelerator.save_state(checkpoint_dir / "latest_checkpoint")
            self.logger.info(f"Latest training state saved at epoch {epoch + 1}")
        except Exception as e:
            self.logger.error(f"Error saving checkpoint: {e}")
            raise
        
        # Additional state saving only on main process
        if self.accelerator.is_main_process:
            # save the current epoch
            other_state = {
                'epoch': epoch + 1,
                'global_step': self.global_step,
                'best_val_loss': self.best_val_loss,
                'epochs_without_improvement': self.epochs_without_improvement,
                'last_best_model_save_step': self.last_best_model_save_step,
                'wandb_run_id': wandb.run.id if wandb.run else None,
            }
            torch.save(other_state, checkpoint_dir / "training_state.pth")
            self.logger.info(f"Training state saved: {other_state}")

            if is_best:
                self.last_best_model_save_step = self.global_step
                best_path = checkpoint_dir / "best_model"
                
                # Clear memory before saving best model to prevent OOM
                self.clear_memory()
                
                try:
                    unwrapped_model : KG_LFM = self.accelerator.unwrap_model(self.model)
                    unwrapped_model.save_pretrained(best_path)
                    self.logger.info(f"New best model saved to {best_path}")
                except Exception as e:
                    self.logger.error(f"Error saving best model: {e}")
                    raise

                # Log the best model as a wandb Artifact
                # artifact = wandb.Artifact(f"{self.run_name}-best-model", type="model")
                # artifact.add_dir(str(best_path))
                # wandb.log_artifact(artifact)
        
        # Final synchronization to ensure all processes complete before continuing
        self.accelerator.wait_for_everyone()
        
        # Memory cleanup after checkpoint
        self.clear_memory()
        
    def train(self):
        """Main training loop."""
        num_epochs = self.config.train_conf.epochs
        patience = self.config.train_conf.early_stopping_patience

        self.setup_model()
        self.setup_data()
        self.setup_optimizer_and_scheduler(num_epochs)
        self.prepare_for_training()

        if self.resume and os.path.exists(self.checkpoint_dir / "latest_checkpoint"):
            self.accelerator.print(f"Resuming training from: {self.checkpoint_dir}")
            self.accelerator.load_state(self.checkpoint_dir / "latest_checkpoint")
            
            state_path = self.checkpoint_dir / "training_state.pth"
            state = torch.load(state_path, map_location=self.device)
            
            self.start_epoch = state.get('epoch', 0)
            self.global_step = state.get('global_step', 0)
            self.best_val_loss = state.get('best_val_loss', float('inf'))
            self.epochs_without_improvement = state.get('epochs_without_improvement', 0)
            self.last_best_model_save_step = state.get('last_best_model_save_step', -1)
            self.accelerator.print(f"Resuming from epoch {self.start_epoch + 1}, global step {self.global_step}")
            self.wandb_run_id = state.get('wandb_run_id', None)
            
            # skip to the correct batch in the dataloader
            batches_seen_in_ep = self.global_step % len(self.train_dataloader)

            # tell the sampler which epoch we're in (DDP needs this)
            if hasattr(self.train_dataloader.sampler, 'set_epoch'):
                self.train_dataloader.sampler.set_epoch(self.start_epoch)

            # efficient, deterministic skip
            self.skip_train_dataloader = self.accelerator.skip_first_batches(
                self.train_dataloader, batches_seen_in_ep
            )

        self.setup_wandb()
        
        self.accelerator.print(f"Starting training for {num_epochs} epochs with patience {patience}")
        
        # Calculate total training steps needed
        steps_per_epoch = len(self.train_dataloader) // self.steps_train
        if len(self.train_dataloader) % self.steps_train != 0:
            steps_per_epoch += 1  # Account for partial step at end of epoch
        
        total_training_steps = num_epochs * steps_per_epoch
        steps_completed = self.global_step // self.steps_train
        
        self.accelerator.print(f"Total training steps: {total_training_steps}, Steps per epoch: {steps_per_epoch}")
        
        for step_idx in range(steps_completed, total_training_steps):
            current_epoch = step_idx // steps_per_epoch
            step_in_epoch = step_idx % steps_per_epoch
            self.accelerator.print(f"--- Training step {step_idx + 1}/{total_training_steps} (Epoch {current_epoch + 1}/{num_epochs}, Step {step_in_epoch + 1}/{steps_per_epoch}) ---")
            
            # Add timeout protection and better error handling
            train_metrics = self.train_step()
            
            # Add explicit synchronization before validation
            self.accelerator.wait_for_everyone()
            
            val_metrics = self._evaluation_loop(self.val_dataloader, "Validation", self.percentage_eval)
            
            self.scheduler.step(val_metrics['validation_loss'])  # Step scheduler based on validation loss
            
            # Another synchronization after validation
            self.accelerator.wait_for_everyone()
            
            log_metrics = {
                "epoch": current_epoch + 1,
                "step": step_idx + 1,
                **{f"train_epoch/{k}": v for k, v in train_metrics.items()},
                **{f"val_epoch/{k}": v for k, v in val_metrics.items()},
            }
            
            self.accelerator.log(log_metrics, step=self.global_step)
            
            self.accelerator.print(f"Step {step_idx + 1}/{total_training_steps} | Train Loss: {train_metrics['training_loss']:.4f} | Val Loss: {val_metrics['validation_loss']:.4f} | LR: {train_metrics['learning_rate']:.6f}")

            is_best = val_metrics['validation_loss'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics['validation_loss']
                self.epochs_without_improvement = 0
                self.accelerator.print(f"New best validation loss: {self.best_val_loss:.4f}")
            else:
                self.epochs_without_improvement += 1

            # Save checkpoint with frequency control
            should_checkpoint = (
                is_best or  # Always save best models
                (step_idx + 1) % self.checkpoint_frequency == 0 or  # Regular frequency
                step_idx == total_training_steps - 1 or  # Last step
                self.epochs_without_improvement >= patience  # Before early stopping
            )

            if should_checkpoint and self.save_checkpoints:
                self.save_checkpoint(current_epoch, is_best)

            if self.epochs_without_improvement >= patience:
                self.accelerator.print(f"Early stopping triggered after {patience} train loops without improvement.")
                break
        
        self.accelerator.print("Training finished. Evaluating on the test set with the best model.")
        # Wait for all processes before test evaluation
        self.accelerator.wait_for_everyone()
        self.run_test_evaluation()
        self.accelerator.wait_for_everyone()

        self.accelerator.end_training()
        self.logger.info("Training process completed.")

    def run_test_evaluation(self):
        """Load the best model and evaluate on the test set."""
        if self.test_dataloader:
            best_model_path = self.checkpoint_dir / "best_model"
            if best_model_path.exists():
                self.logger.info(f"Loading best model from {best_model_path} for test evaluation.")
                
                try:
                    self.model.load_pretrained(best_model_path)
                    self.logger.info("Best model loaded successfully.")
                except Exception as e:
                    self.logger.error(f"Error loading best model: {e}")
                    raise

                test_metrics = self._evaluation_loop(self.test_dataloader, "Testing", 1)

                
                self.logger.info(f"Test metrics: {test_metrics}")
                
                final_log = {
                    "test/final_loss": test_metrics['validation_loss'],
                }
                
                self.metrics_tracker.update(final_log, 1)
                
                if 'perplexity' in test_metrics:
                    final_log["test/final_perplexity"] = test_metrics['perplexity']
                    self.logger.info(f"Perplexity: {test_metrics['perplexity']:.2f}")
                
                self.accelerator.log(final_log)
            else:
                self.logger.warning("No best model found to evaluate on the test set.")
        else:
            self.logger.warning("Skipping test evaluation.")
