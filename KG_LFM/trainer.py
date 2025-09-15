#!/usr/bin/env python3
"""
Training script for KG-LFM with wandb logging, checkpointing, and better validation.
"""

import logging
import os
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from collections import defaultdict
import gc
from copy import deepcopy
import psutil
import uuid

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
from KG_LFM.configuration import ProjectConfig
from KG_LFM.model.KG_LFM_arch import KG_LFM, KG_LFMConfig, set_KGLM_model_args
from KG_LFM.utils.Dataloader import create_dataloader
from KG_LFM.evaluator import compute_hit_k

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

    def __init__(self, accelerator: Accelerator):
        self.values = defaultdict(float)
        self.counts = defaultdict(int)
        self.accelerator = accelerator

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
        """Get the current average of all tracked metrics. Aggregating from all processes."""
        result = {}
        for key, value in self.values.items():
            all_counts = self.accelerator.gather(torch.tensor(self.counts[key], device=self.accelerator.device))
            all_value = self.accelerator.gather(torch.tensor(value, device=self.accelerator.device))
            
            counts = all_counts.sum().item()
            value = all_value.sum().item()

            result[key] = value / counts if counts > 0 else 0.0
        return result

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
                 metrics_tracker: Optional[type] = DefaultMetricsTracker,
                 accelerator: Optional[Accelerator] = None
                ):
        self.config = config
        self.run_name = run_name or f"kg_lfm_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.resume = resume
        self.enable_wandb = enable_wandb
        self.save_checkpoints = save_checkpoints
        set_seed(self.config.seed)

        # Initialize Accelerator and set seed for reproducibility
        if accelerator is not None:
            self.accelerator = accelerator
        else:
            kwargs = DistributedDataParallelKwargs(
                find_unused_parameters=False,
            )
            self.accelerator = Accelerator(
                log_with="wandb" if enable_wandb else None,
                gradient_accumulation_steps=self.config.train_conf.gradient_accumulation_steps,
                kwargs_handlers=[kwargs],
                step_scheduler_with_optimizer=False,  # Let us handle this manually
            )
        
        self.metrics_tracker : MetricsTracker = metrics_tracker(self.accelerator)
        
        self.device = self.accelerator.device

        # Setup logging
        self.logger = logging.getLogger()

        # Initialize state
        self.model : KG_LFM = None
        
        self.optimizer = None
        self.scheduler = None
        
        self.skip_train_dataloader = None  # Used to skip initial batches in the training dataloader
        self.train_dataloader: DataLoader = None
        self.val_dataloader: DataLoader = None
        self.test_dataloader: DataLoader = None
        self.dataloader_factory = None  # Store factory for epoch setting
        
        # Generate unique wandb run ID to prevent collisions in simultaneous runs
        # Since run names are guaranteed to be unique, we'll use the run name as the wandb ID
        self.wandb_run_id = None

        self.global_step = 0
        self.start_epoch = 0
        self.best_val_metric = float('inf') if self.config.train_conf.scheduler_mode == "min" else float('-inf')
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

        self.logger.info(f"Trainer initialized on device: {self.device}")
        self.logger.info(f"World size: {self.accelerator.num_processes}")
        self.logger.info(f"Local rank: {self.accelerator.local_process_index}")
        self.logger.info(f"Is main process: {self.accelerator.is_main_process}")
        self.logger.info(f"Checkpoint frequency: every {self.checkpoint_frequency} epoch(s)")
        self.logger.info(f"Training steps per evaluation: {self.steps_train}")
        
        # Calculate and display effective batch size
        micro_batch_size = self.config.train_conf.dataloader.batch_size
        self.logger.info(f"Micro batch size per GPU: {micro_batch_size}")
        
        # Add safety check for distributed training
        if self.accelerator.num_processes > 1:
            self.logger.info(f"Distributed training with {self.accelerator.num_processes} processes")

    def setup_wandb(self):
        """Initialize wandb, potentially resuming a previous run."""
        if self.accelerator.is_main_process and self.enable_wandb:
            # if already initialized, just resume
            if wandb.run is not None:
                self.logger.info(f"Resuming existing wandb run: {wandb.run.id}")
                return
            
            wandb_config = deepcopy(self.config).to_dict() # Assuming a method to convert config to dict
            
            # For resuming: use stored wandb_run_id, for new runs: generate unique prefix
            if self.wandb_run_id:
                # Resuming from checkpoint - use exact same ID
                run_id = self.wandb_run_id
                resume_mode = "must"  # Must resume this exact run
                self.logger.info(f"Resuming wandb run with ID: {run_id}")
            else:
                # New run - generate unique prefix to prevent ID collisions
                unique_prefix = f"{self.run_name}_{uuid.uuid4().hex[:8]}"
                run_id = unique_prefix
                resume_mode = "never"  # Never resume, always create new
                self.wandb_run_id = run_id  # Store for checkpointing
                self.logger.info(f"Creating new wandb run with unique ID: {run_id}")
            
            self.accelerator.init_trackers(
                project_name="kg-lfm-pretraining",
                config=wandb_config,
                init_kwargs={"wandb": {
                    "name": self.run_name,
                    "resume": resume_mode,
                    "id": run_id
                    }}
            )
            
            self.logger.info("Wandb initialized successfully")
        
        # For non-main processes, just wait
        self.accelerator.wait_for_everyone()

    def setup_model(self):
        """Initialize the model with proper configuration."""
        self.logger.info("Setting up model...")
        
        # Profile model initialization
        if not self.config.train_conf.start_from_checkpoint:
            self.logger.info("Loading fresh model configuration from pretrained model.")
            model_config = KG_LFMConfig.from_pretrained(
                self.config.model.llm_model_name,
                trust_remote_code=True,
            )
            model_config = set_KGLM_model_args(model_config, self.config.model)
            self.model = KG_LFM(model_config)
        else:
            self.logger.info(f"Loading model from checkpoint: {self.config.train_conf.start_from_checkpoint}")
            self.model = KG_LFM.from_pretrained(self.config.train_conf.start_from_checkpoint)

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
        self.logger.info("Setting up data loaders...")
        
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
            self.steps_train = int(len(self.train_dataloader) * self.steps_train / self.accelerator.num_processes)
            self.logger.info(f"Converted steps_train to {self.steps_train} based on dataset size.")
             
        
    def setup_optimizer_and_scheduler(self, num_epochs: int):
        """Setup optimizer and learning rate scheduler."""
        self.logger.info("Setting up optimizer and scheduler...")
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        
        # Enhanced optimizer with different learning rates for different components
        kg_encoder_params = [p for n, p in self.model.named_parameters() if 'kg_encoder' in n and p.requires_grad]
        llm_params = [p for n, p in self.model.named_parameters() if 'kg_encoder' not in n and p.requires_grad]

        param_groups = []
        if kg_encoder_params:
            param_groups.append({
                'params': kg_encoder_params, 
                'lr': self.config.train_conf.KG_learning_rate,
                'weight_decay': self.config.train_conf.weight_decay
            })
        if llm_params:
            param_groups.append({
                'params': llm_params, 
                'lr': self.config.train_conf.LLM_learning_rate,  # Lower LR for LLM
                'weight_decay': self.config.train_conf.weight_decay
            })
        
        self.optimizer = AdamW(
            param_groups if param_groups else trainable_params,
            lr=self.config.train_conf.KG_learning_rate,
            weight_decay=self.config.train_conf.weight_decay,
            eps=1e-8,
            betas=(0.9, 0.95)  # More stable betas for large models
        )

        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode=self.config.train_conf.scheduler_mode,
            factor=0.5,
            patience=self.config.train_conf.scheduler_patience,
            verbose=True
        )

    def prepare_for_training(self):
        """Prepare all components with accelerator."""
        self.logger.info("Preparing components with Accelerator...")
        
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

    def compute_train_metrics(self, RVQ_loss: torch.Tensor, CE_loss: torch.Tensor, decoder_loss: Optional[torch.Tensor]) -> Dict[str, float]:
        """Compute metrics from loss."""
        # Ensure loss is a scalar tensor and safely convert to float
        CE_loss_val = CE_loss.flatten().mean().item()
        if RVQ_loss is None:
            RVQ_loss_val = torch.tensor(0.0, device=self.device)
        else:
            RVQ_loss_val = RVQ_loss.flatten().mean().item()
        if decoder_loss is not None:
            decoder_loss_val = decoder_loss.flatten().mean().item()
        else:
            decoder_loss_val = 0.0

        metrics = {
            'language_loss': CE_loss_val,
            'RVQ_loss': RVQ_loss_val,
            'decoder_loss': decoder_loss_val,
            'encoder_learning_rate': self.optimizer.param_groups[0]['lr'],
            'llm_learning_rate': self.optimizer.param_groups[1]['lr'] if len(self.optimizer.param_groups) > 1 else 0
        }
        
        return metrics

    def compute_val_metrics(
            self, 
            language_loss: torch.Tensor, RVQ_loss: torch.Tensor, decoder_loss: Optional[torch.Tensor],
            logits: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor,
            object_boundaries: List[Tuple[int]]
        ) -> Dict[str, float]:
        """Compute validation metrics from loss."""
        # Ensure loss is a scalar tensor and safely convert to float
        language_loss_val = language_loss.flatten()
        if RVQ_loss is None:
            RVQ_loss_val = torch.tensor(0.0, device=self.device)
        else:
            RVQ_loss_val = RVQ_loss.flatten()

        if torch.isnan(language_loss_val).any() or torch.isnan(RVQ_loss_val).any():
            self.logger.warning("NaN detected in validation losses")

        language_loss_val = language_loss_val.mean().item()
        RVQ_loss_val = RVQ_loss_val.mean().item()
        if decoder_loss is not None:
            decoder_loss_val = decoder_loss.flatten().mean().item()
        else:
            decoder_loss_val = 0.0
        
        hit_k_correct, average_num_tokens, total_objects = compute_hit_k(
            logits, input_ids, [1,3,5,10],
            object_boundaries, 
            (self.config.model.num_quantizers + (2 if self.config.model.bounding_tokens else 0) if self.config.model.use_kg_encoder else 0),
            attention_mask, special_token=self.model.special_kg_token, 
            tokenizer=self.model.tokenizer
        )

        average_num_tokens = average_num_tokens / total_objects if total_objects > 0 else 0

        hit_k_correct = {
            f"Hit@{k}": v / total_objects if total_objects > 0 else 0
            for k, v in hit_k_correct.items()
        }
        
        metrics = {
            'language_loss': language_loss_val,
            'RVQ_loss': RVQ_loss_val,
            'decoder_loss': decoder_loss_val,
            **hit_k_correct,
            'average_num_tokens': average_num_tokens
        }

        return metrics

    def _evaluation_loop(self, dataloader: torch.utils.data.DataLoader, description: str, percentage_eval: float) -> Dict[str, float]:
        """Generic evaluation loop for validation or testing with distributed support."""
        self.model.eval()
        
        # Calculate number of batches to process instead of samples
        total_batches = len(dataloader)
        num_batches = max(1, int(total_batches * percentage_eval))
        codebook_indices_seen = [set() for _ in range(self.config.model.num_quantizers)]

        self.logger.info(f"Evaluating {num_batches}/{total_batches} batches ({num_batches/total_batches*100:.1f}%)")
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(dataloader, desc=description, disable=not self.accelerator.is_local_main_process, total=num_batches)):
                if batch_idx >= num_batches:
                    break
                    
                try:
                    outputs = self.model(
                        input_ids=batch['input_ids'],
                        graphs=batch['graphs'],
                        attention_mask=batch['attention_mask'],
                        labels=batch['labels'],
                        use_cache=False,
                    )
                    
                    metrics = self.compute_val_metrics(
                        outputs.loss, outputs.RVQ_loss, outputs.decoder_loss,
                        outputs.logits, batch['input_ids'],
                        batch['attention_mask'],
                        [batch['objects'][i]['token_boundaries'] for i in range(len(batch['objects']))]
                    )
                    self.metrics_tracker.update(metrics, batch['input_ids'].size(0))

                    if outputs.RVQ_indices is not None:
                        for i in range(self.config.model.num_quantizers):
                            codebook_indices_seen[i].update(outputs.RVQ_indices[:, i].tolist())

                except Exception as e:
                    self.logger.info(f"Error in evaluation batch {batch_idx}: {e}")
                    # Clear cache on error
                    self.clear_memory()
                    # Ensure synchronization even on error
                    self.accelerator.wait_for_everyone()
                    continue
        
        # Final cleanup
        self.clear_memory()
        self.accelerator.wait_for_everyone()
        
        averages = self.metrics_tracker.get_averages()
        self.metrics_tracker.reset()  # Reset tracker for next evaluation

        vocab_size = self.config.model.codebook_size
        codebook_utilization = {
            f"codebook/codebook_utilization_{i}": len(codebook_indices_seen[i]) / vocab_size
            for i in range(self.config.model.num_quantizers)
        }
        if not self.config.model.shared_codebook:
            overall_codebook_utilization = sum(codebook_utilization.values()) / self.config.model.num_quantizers
        else:
            seen_idx = set.union(*codebook_indices_seen)
            overall_codebook_utilization = len(seen_idx) / vocab_size

        averages.update({
            "codebook/overall_codebook_utilization": overall_codebook_utilization,
            **codebook_utilization
        })

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

        # Initialize accumulation variables properly
        total_RVQ_loss = 0.0
        total_CE_loss = 0.0
        total_decoder_loss = 0.0

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

            # self.logger.info(f"Processing step {step + 1}/{self.steps_train} on process {self.accelerator.local_process_index}")
            with self.accelerator.accumulate(self.model):
                # Standard forward pass
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    graphs=batch['graphs'],
                    attention_mask=batch['attention_mask'],
                    labels=batch['labels'],
                    use_cache=False,
                )
                language_loss = outputs.loss
                RVQ_loss = outputs.RVQ_loss
                decoder_loss = outputs.decoder_loss
                
                decoder_loss = decoder_loss if decoder_loss is not None else torch.tensor(0.0, device=self.device)
                RVQ_loss = RVQ_loss if RVQ_loss is not None else torch.tensor(0.0, device=self.device)
                
                # Weighted loss combination with configurable weights
                rvq_weight = getattr(self.config.train_conf, 'rvq_loss_weight', 1.0)
                loss = language_loss + rvq_weight * RVQ_loss + decoder_loss
                
                # print all processes' losses for debugging
                self.logger.debug(
                    "Step %d/%d - Language Loss: %.4f, RVQ Loss: %.4f",
                    step + 1, self.steps_train,
                    language_loss.item(), RVQ_loss.item()
                )
                if decoder_loss is not None:
                    self.logger.debug("Decoder Loss: %.4f", decoder_loss.item())
                
                # create a tensor to accumulate from all processes to skip all backward passes if loss is NaN or Inf
                is_valid_loss = torch.tensor(not (torch.isnan(loss) or torch.isinf(loss)), device=self.device, dtype=torch.bool)
                
                # Gather across all processes to check for NaN/Inf
                is_valid_loss = self.accelerator.gather(is_valid_loss).all()
                # Skip invalid losses
                if not is_valid_loss:
                    self.logger.warning(f"NaN or Inf detected at step {step + 1}, skipping backward. Language Loss: {language_loss.item()}, RVQ Loss: {RVQ_loss.item()}")
                else:
                    # Accumulate losses correctly - just add the raw values
                    total_RVQ_loss += RVQ_loss.item()
                    total_CE_loss += language_loss.item()
                    total_decoder_loss += decoder_loss.item() if decoder_loss is not None else 0.0
                    self.accelerator.backward(loss)
                
                if self.accelerator.sync_gradients: # TRUE when gradient accumulation is complete

                    if self.clip_grad_norm > 0:
                        norm = self.accelerator.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)

                    # Compute average losses over accumulation steps
                    avg_RVQ_loss = total_RVQ_loss / self.grad_accumulation_steps
                    avg_CE_loss = total_CE_loss / self.grad_accumulation_steps
                    avg_decoder_loss = total_decoder_loss / self.grad_accumulation_steps

                    metrics = self.compute_train_metrics(
                        torch.tensor(avg_RVQ_loss, device=self.device), 
                        torch.tensor(avg_CE_loss, device=self.device),
                        torch.tensor(avg_decoder_loss, device=self.device)
                    )
                    self.metrics_tracker.update(metrics, batch['input_ids'].size(0))
                    
                    log_metrics = {
                        f"train/{k}": v for k, v in metrics.items()
                    }
                    log_metrics["train/clip_grad_norm"] = norm if self.clip_grad_norm > 0 else 0.0
                    self.accelerator.log(log_metrics, step=self.global_step)
                    
                    # Reset accumulation variables
                    total_CE_loss = 0.0
                    total_RVQ_loss = 0.0
                    total_decoder_loss = 0.0
                    
                # does NOTHING when using deepspeed
                self.optimizer.step() 
                self.optimizer.zero_grad()
                
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
            self.logger.info(f"Saving model to {self.checkpoint_dir}")
        
            # Ensure checkpoint directory exists
            model_dir = self.checkpoint_dir if sub_path is None else self.checkpoint_dir / sub_path
            model_dir.mkdir(parents=True, exist_ok=True)

            try:
                unwrapped_model: KG_LFM = self.accelerator.unwrap_model(self.model)
                unwrapped_model.save_pretrained(model_dir)
                self.logger.info(f"Model saved successfully to {model_dir}")
            except Exception as e:
                self.logger.error(f"Error saving model: {e}")
                
        self.accelerator.wait_for_everyone()  # Ensure all processes complete before continuing
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint with proper distributed coordination."""
        checkpoint_dir = self.checkpoint_dir        
        
        # Create checkpoint directory on main process
        if self.accelerator.is_main_process:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Saving checkpoint for epoch {epoch + 1}...")
        
        self.clear_memory()
        # Ensure all processes wait at the same point before saving
        self.accelerator.wait_for_everyone()
        
        # show RAM usage before saving
        cpu_ram = psutil.virtual_memory().percent
        self.logger.info(f"CPU RAM usage before saving checkpoint: {cpu_ram}%")
        
        # Save the model
        self.accelerator.save_state(checkpoint_dir / "latest_checkpoint")
        self.logger.info(f"Latest training state saved at epoch {epoch + 1}")
        
        # Additional state saving only on main process
        if self.accelerator.is_main_process:
            # save the current epoch
            other_state = {
                'epoch': epoch + 1,
                'global_step': self.global_step,
                'best_val_metric': self.best_val_metric,
                'epochs_without_improvement': self.epochs_without_improvement,
                'last_best_model_save_step': self.last_best_model_save_step,
                'wandb_run_id': wandb.run.id if wandb.run else None,
            }
            torch.save(other_state, checkpoint_dir / "training_state.pth")
            self.logger.info(f"Training state saved: {other_state}")

        if is_best:
            self.last_best_model_save_step = self.global_step
            best_path = "best_model"
            
            self.save_model(sub_path=best_path)
        
        # Final synchronization to ensure all processes complete before continuing
        self.accelerator.wait_for_everyone()
        
        # Memory cleanup after checkpoint
        self.clear_memory()
        
    def train(self, time_budget_s: Optional[int] = None):
        """Main training loop.
        
        Args:
            time_budget_s (Optional[int]): Time budget for training in seconds.
        """
        self.logger.info("Starting training...")
        num_epochs = self.config.train_conf.epochs
        patience = self.config.train_conf.early_stopping_patience

        self.setup_model()
        self.setup_data()
        self.setup_optimizer_and_scheduler(num_epochs)
        self.prepare_for_training()

        start_time = time.time()

        if self.resume and os.path.exists(self.checkpoint_dir / "latest_checkpoint"):
            self.logger.info(f"Resuming training from: {self.checkpoint_dir}")
            self.accelerator.load_state(self.checkpoint_dir / "latest_checkpoint")
            
            state_path = self.checkpoint_dir / "training_state.pth"
            state = torch.load(state_path, map_location=self.device)
            
            self.start_epoch = state.get('epoch', 0)
            self.global_step = state.get('global_step', 0)
            self.best_val_metric = state.get('best_val_metric', float('inf') if self.config.train_conf.scheduler_mode == "min" else float('-inf'))
            self.epochs_without_improvement = state.get('epochs_without_improvement', 0)
            self.last_best_model_save_step = state.get('last_best_model_save_step', -1)
            self.logger.info(f"Resuming from epoch {self.start_epoch + 1}, global step {self.global_step}")
            
            # Restore wandb run ID from checkpoint to maintain consistency
            restored_wandb_id = state.get('wandb_run_id', None)
            if restored_wandb_id:
                self.wandb_run_id = restored_wandb_id
                self.logger.info(f"Restored wandb run ID from checkpoint: {restored_wandb_id}")
            
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
        
        self.logger.info(f"Starting training for {num_epochs} epochs with patience {patience}")
        
        # Calculate total training steps needed
        steps_per_epoch = len(self.train_dataloader) // self.steps_train
        if len(self.train_dataloader) % self.steps_train != 0:
            steps_per_epoch += 1  # Account for partial step at end of epoch
        
        total_training_steps = num_epochs * steps_per_epoch
        steps_completed = self.global_step // self.steps_train
        
        time_one_step = 0
        
        self.logger.info(f"Total training steps: {total_training_steps}, Steps per epoch: {steps_per_epoch}")
        
        for step_idx in range(steps_completed, total_training_steps):
            
            # TIMEOUT PROTECTION
            should_stop = torch.tensor([False], device=self.device)
            if time_budget_s is not None:
                step_end = time.time() - start_time + time_one_step
                if step_end >= time_budget_s:
                    should_stop = torch.tensor([True], device=self.device)
            start_step_time = time.time()
            
            should_stop = self.accelerator.gather(should_stop).all().item()
            
            if should_stop:
                # If time budget is reached, stop training
                self.logger.info(f"Time budget of {time_budget_s} seconds reached. Stopping training.")
                break
            
            
            # ACTIVATE TRAINING STEP
            current_epoch = step_idx // steps_per_epoch
            step_in_epoch = step_idx % steps_per_epoch
            self.accelerator.print(f"--- Training step {step_idx + 1}/{total_training_steps} (Epoch {current_epoch + 1}/{num_epochs}, Step {step_in_epoch + 1}/{steps_per_epoch}) ---")

            train_metrics = self.train_step()
            self.accelerator.wait_for_everyone()
            
            val_metrics = self._evaluation_loop(self.val_dataloader, "Validation", self.percentage_eval)
            self.accelerator.wait_for_everyone()

            self.scheduler.step(val_metrics[self.config.train_conf.scheduler_metric])  # Step scheduler based on validation loss

            log_metrics = {
                "epoch": current_epoch + 1,
                "step": step_idx + 1,
                **{f"train_step/{k}": v for k, v in train_metrics.items()},
                **{f"val_step/{k}": v for k, v in val_metrics.items()},
            }
            self.accelerator.log(log_metrics, step=self.global_step)
            
            if self.logger.isEnabledFor(logging.DEBUG):
                self.logger.debug(f"Training step {step_idx + 1}/{total_training_steps} (Epoch {current_epoch + 1}/{num_epochs}, Step {step_in_epoch + 1}/{steps_per_epoch}) - Train Metrics: {train_metrics} - Val Metrics: {val_metrics}")
                time.sleep(60)

            is_best = (
                (
                    val_metrics[self.config.train_conf.scheduler_metric] < self.best_val_metric 
                    and self.config.train_conf.scheduler_mode == "min" 
                ) 
                or
                (
                    val_metrics[self.config.train_conf.scheduler_metric] > self.best_val_metric 
                    and self.config.train_conf.scheduler_mode == "max" 
                )
            )
            if is_best:
                self.best_val_metric = val_metrics[self.config.train_conf.scheduler_metric]
                self.epochs_without_improvement = 0
                self.logger.info(f"New best validation loss: {self.best_val_metric:.4f}")
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

            if self.epochs_without_improvement > patience:
                self.logger.info(f"Early stopping triggered after {patience} train loops without improvement.")
                break
            
            # UPDATE ESTIMATE OF TIME PER STEP
            end_step_time = time.time() - start_step_time
            if time_one_step == 0:
                time_one_step = end_step_time
            else:
                time_one_step = 0.7 * time_one_step + 0.3 * end_step_time

        self.logger.info("Training finished.")
        
    def close(self):
        """Cleanup resources."""
        self.accelerator.end_training()
        self.logger.info("Training resources cleaned up.")
