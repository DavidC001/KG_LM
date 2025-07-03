#!/usr/bin/env python3
"""
Training script for KG-LFM with wandb logging, checkpointing, and better validation.
"""

import logging
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict
import gc

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from accelerate import Accelerator
from accelerate.utils import set_seed
import wandb
from tqdm.auto import tqdm

# Assuming these are defined in your project structure
from configuration import load_yaml_config, ProjectConfig
from model.KG_LFM_arch import KG_LFM, KG_LFMConfig, set_KGLM_model_args
from utils.Dataloaders.pretrain_data import create_dataloader

# silence warnings
import warnings
warnings.filterwarnings("ignore")

class MetricsTracker:
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

    def __init__(self, config: ProjectConfig, run_name: Optional[str] = None, resume: Optional[bool] = False):
        self.config = config
        self.run_name = run_name or f"kg_lfm_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.resume = resume

        # Initialize Accelerator and set seed for reproducibility
        self.accelerator = Accelerator(log_with="wandb")
        set_seed(self.config.seed)

        self.device = self.accelerator.device

        # Setup logging
        self.setup_logging()

        # Initialize state
        self.model = None
        
        self.optimizer = None
        self.scheduler = None
        
        self.grad_accumulation_steps = self.config.pretrain_conf.gradient_accumulation_steps
        
        self.train_dataloader = None
        self.val_dataloader = None
        self.test_dataloader = None
        self.dataloader_factory = None  # Store factory for epoch setting
        
        self.wandb_run_id = None

        self.global_step = 0
        self.start_epoch = 0
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0
        
        # Checkpoint frequency to reduce I/O overhead
        self.checkpoint_frequency = self.config.pretrain_conf.checkpoint_frequency
        
        self.checkpoint_dir = Path(self.config.pretrain_conf.checkpoint_dir + f"/{self.run_name}")

        self.accelerator.print(f"Trainer initialized on device: {self.device}")
        self.accelerator.print(f"Checkpoint frequency: every {self.checkpoint_frequency} epoch(s)")

    def setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f'training_{self.run_name}.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def setup_wandb(self):
        """Initialize wandb, potentially resuming a previous run."""
        wandb_config = self.config.to_dict() # Assuming a method to convert config to dict
        
        run_id = self.wandb_run_id
        
        self.accelerator.init_trackers(
            project_name="kg-lfm-training",
            config=wandb_config,
            init_kwargs={"wandb": {"name": self.run_name, "resume": "auto", "id": run_id}}
        )
        self.logger.info("Wandb initialized successfully")

    def setup_model(self):
        """Initialize the model with proper configuration."""
        self.accelerator.print("Setting up model...")
        
        # Profile model initialization
        def _setup_model_internal():
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

        _setup_model_internal()

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
        
        # Check if we're using distributed training
        distributed = self.accelerator.num_processes > 1
        
        train, val, test = create_dataloader(
            self.config.dataset,
            self.config.pretrain_conf.dataloader,
            tokenizer=self.model.tokenizer,
            distributed=distributed,
            split="all",
        )
        
        self.train_dataloader = train
        self.val_dataloader = val
        self.test_dataloader = test
        
        
    def setup_optimizer_and_scheduler(self, num_epochs: int):
        """Setup optimizer and learning rate scheduler with warmup."""
        self.accelerator.print("Setting up optimizer and scheduler...")
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        
        # Start with a lower learning rate for warmup
        initial_lr = self.config.pretrain_conf.learning_rate * 0.1
        
        self.optimizer = AdamW(
            trainable_params,
            lr=initial_lr,
            weight_decay=self.config.pretrain_conf.weight_decay,
            eps=1e-8,  # Slightly higher epsilon for stability
            betas=(0.9, 0.999)
        )

        total_steps = len(self.train_dataloader) * num_epochs
        warmup_steps = min(500, total_steps // 10)  # 10% of total steps or 500 steps max
        
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps,
            eta_min=self.config.pretrain_conf.scheduler_eta_min
        )
        
        self.warmup_steps = warmup_steps
        self.target_lr = self.config.pretrain_conf.learning_rate
        self.initial_lr = initial_lr
        
        self.logger.info(f"Warmup steps: {warmup_steps}, Target LR: {self.target_lr}")

    def update_learning_rate(self, step):
        """Update learning rate with warmup."""
        if step < self.warmup_steps:
            # Linear warmup
            lr = self.initial_lr + (self.target_lr - self.initial_lr) * (step / self.warmup_steps)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        else:
            # Use cosine annealing after warmup
            self.scheduler.step()

    def prepare_for_training(self):
        """Prepare all components with accelerator."""
        self.accelerator.print("Preparing components with Accelerator...")
        
        def _prepare_components():
            return self.accelerator.prepare(
                self.model, self.optimizer, self.train_dataloader, 
                self.val_dataloader, self.test_dataloader, self.scheduler
            )
        
        prepared_components = _prepare_components()
            
        (self.model, self.optimizer, self.train_dataloader, 
         self.val_dataloader, self.test_dataloader, self.scheduler) = prepared_components

    def compute_metrics(self, loss: torch.Tensor) -> Dict[str, float]:
        """Compute metrics from loss."""
        metrics = {
            'loss': loss.item(),
            'learning_rate': self.optimizer.param_groups[0]['lr']
        }
        
        # Add memory metrics
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            metrics.update({
                'gpu_memory_allocated_gb': allocated,
                'gpu_memory_reserved_gb': reserved
            })
        
        return metrics

    def _evaluation_loop(self, dataloader: torch.utils.data.DataLoader, description: str) -> Dict[str, float]:
        """Generic evaluation loop for validation or testing with distributed support."""
        self.model.eval()
        metrics_tracker = MetricsTracker()
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(dataloader, desc=description, disable=not self.accelerator.is_local_main_process)):
                try:
                    outputs = self.model(
                        input_ids=batch['input_ids'],
                        graphs=batch['graphs'],
                        attention_mask=batch['attention_mask'],
                        labels=batch['input_ids'],
                        use_cache=False,
                    )
                    
                    # Gather loss across all devices
                    loss = self.accelerator.gather(outputs.loss).mean()
                    metrics = self.compute_metrics(loss)
                    metrics_tracker.update(metrics, batch['input_ids'].size(0))
                    
                except Exception as e:
                    self.logger.error(f"Error in evaluation step {batch_idx + 1}: {e}")
                    continue
        
        # Wait for all processes to complete evaluation before returning
        self.accelerator.wait_for_everyone()
        
        return metrics_tracker.get_averages()

    def train_epoch(self, epoch: int):
        """Train for one epoch."""
        self.model.train()
        
        # Set epoch for distributed samplers to ensure proper shuffling
        if hasattr(self, 'dataloader_factory') and self.dataloader_factory is not None:
            self.dataloader_factory.set_epoch(epoch)
        
        metrics_tracker = MetricsTracker()

        progress_bar = tqdm(
            self.train_dataloader,
            desc=f"Epoch {epoch + 1}",
            disable=not self.accelerator.is_local_main_process
        )

        loss = torch.tensor(0.0, device=self.device)
        
        # Add periodic synchronization to prevent hanging
        sync_frequency = max(1, len(self.train_dataloader) // 5)  # Sync every 20% of epoch

        for step, batch in enumerate(progress_bar):
            try:
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    graphs=batch['graphs'],
                    attention_mask=batch['attention_mask'],
                    labels=batch['input_ids'],
                    use_cache=False,
                )
                
                language_loss = outputs.loss
                RVQ_loss = outputs.RVQ_loss
                
                # Check if any of the two is nan
                if torch.isnan(language_loss) or torch.isnan(RVQ_loss) or torch.isinf(language_loss) or torch.isinf(RVQ_loss):
                    self.accelerator.print(f"NaN or inf detected in losses at step {step + 1}. Skipping this batch.")
                    continue
                
                loss += (language_loss + RVQ_loss) / self.grad_accumulation_steps
                
                if (step + 1) % self.grad_accumulation_steps == 0:
                    self.accelerator.backward(loss)
                    
                    # Clip gradients to prevent exploding gradients
                    if self.config.pretrain_conf.clip_grad_norm is not None:
                        nn.utils.clip_grad_norm_(self.model.parameters(), self.config.pretrain_conf.clip_grad_norm)
                    
                    self.optimizer.step()
                    self.update_learning_rate(self.global_step)
                    self.optimizer.zero_grad()
                    
                    if self.accelerator.sync_gradients:
                        self.global_step += 1
                        metrics = self.compute_metrics(loss)
                        metrics_tracker.update(metrics, batch['input_ids'].size(0))
                        
                        log_metrics = {f"train/{k}": v for k, v in metrics.items()}
                        self.accelerator.log(log_metrics, step=self.global_step)
                        
                        progress_bar.set_postfix({
                            'loss': f"{metrics['loss']:.4f}",
                            'lr': f"{metrics['learning_rate']:.2e}",
                        })

                    loss = torch.tensor(0.0, device=self.device)
                
                # Periodic synchronization to prevent deadlocks
                if (step + 1) % sync_frequency == 0:
                    self.accelerator.wait_for_everyone()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        
            except Exception as e:
                self.logger.error(f"Error in training step {step + 1}: {e}")
                self.accelerator.print(f"Skipping batch {step + 1} due to error: {e}")
                
                # Reset loss accumulation
                loss = torch.tensor(0.0, device=self.device)
                
                # Clear any partially accumulated gradients
                self.optimizer.zero_grad()
                
                # Force synchronization after error
                try:
                    self.accelerator.wait_for_everyone()
                except:
                    pass
                
                continue
        
        return metrics_tracker.get_averages()
    
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
        
        # Save the full training state with accelerator (handles distributed coordination)
        self.logger.info(f"Saving checkpoint for epoch {epoch + 1}...")
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
                'wandb_run_id': wandb.run.id if wandb.run else None,
            }
            torch.save(other_state, checkpoint_dir / "training_state.pth")
            self.logger.info(f"Training state saved: {other_state}")

            if is_best:
                best_path = checkpoint_dir / "best_model"
                
                unwrapped_model : KG_LFM = self.accelerator.unwrap_model(self.model)
                unwrapped_model.save_pretrained(best_path)
                self.logger.info(f"New best model saved to {best_path}")

                # Log the best model as a wandb Artifact
                # artifact = wandb.Artifact(f"{self.run_name}-best-model", type="model")
                # artifact.add_dir(str(best_path))
                # wandb.log_artifact(artifact)
        
        # Final synchronization to ensure all processes complete before continuing
        self.accelerator.wait_for_everyone()
        
        # Memory cleanup after checkpoint
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    def handle_nccl_error(self, error: Exception, step: int) -> bool:
        """
        Handle NCCL communication errors gracefully.
        Returns True if training should continue, False if it should stop.
        """
        error_str = str(error).lower()
        
        if 'nccl' in error_str or 'timeout' in error_str or 'collective' in error_str:
            self.logger.error(f"NCCL communication error at step {step}: {error}")
            self.accelerator.print(f"NCCL error detected at step {step}. Attempting recovery...")
            
            try:
                # Clear any pending operations
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                
                # Force cleanup of any hanging collective operations
                if hasattr(torch.distributed, 'destroy_process_group'):
                    try:
                        torch.distributed.barrier(timeout=timedelta(seconds=30))
                    except:
                        pass
                
                # Reset optimizer state to clear any accumulated gradients
                self.optimizer.zero_grad()
                
                self.accelerator.print("NCCL error recovery attempted. Continuing training...")
                return True
                
            except Exception as recovery_error:
                self.logger.error(f"Failed to recover from NCCL error: {recovery_error}")
                self.accelerator.print("NCCL error recovery failed. Stopping training.")
                return False
        
        # For non-NCCL errors, re-raise
        raise error

    def train(self):
        """Main training loop."""
        num_epochs = self.config.pretrain_conf.epochs
        patience = self.config.pretrain_conf.early_stopping_patience

        self.setup_model()
        self.setup_data()
        self.setup_optimizer_and_scheduler(num_epochs)
        self.prepare_for_training()

        if self.resume:
            self.accelerator.print(f"Resuming training from: {self.checkpoint_dir}")
            self.accelerator.load_state(self.checkpoint_dir / "latest_checkpoint")
            
            state_path = self.checkpoint_dir / "training_state.pth"
            state = torch.load(state_path, map_location=self.device)
            
            self.start_epoch = state.get('epoch', 0)
            self.global_step = state.get('global_step', 0)
            self.best_val_loss = state.get('best_val_loss', float('inf'))
            self.epochs_without_improvement = state.get('epochs_without_improvement', 0)
            self.accelerator.print(f"Resuming from epoch {self.start_epoch + 1}, global step {self.global_step}")
            self.wandb_run_id = state.get('wandb_run_id', None)

        self.setup_wandb()
        
        self.accelerator.print(f"Starting training for {num_epochs} epochs with patience {patience}")
        
        for epoch in range(self.start_epoch, num_epochs):
            self.accelerator.print(f"--- Starting epoch {epoch + 1}/{num_epochs} ---")
            
            
            train_metrics = self.train_epoch(epoch)
            val_metrics = self._evaluation_loop(self.val_dataloader, "Validation")
            
            log_metrics = {
                "epoch": epoch + 1,
                **{f"train_epoch/{k}": v for k, v in train_metrics.items()},
                **{f"val_epoch/{k}": v for k, v in val_metrics.items()},
            }
            self.accelerator.log(log_metrics, step=self.global_step)
            
            self.accelerator.print(f"Epoch {epoch + 1} | Train Loss: {train_metrics['loss']:.4f} | Val Loss: {val_metrics['loss']:.4f}")

            is_best = val_metrics['loss'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics['loss']
                self.epochs_without_improvement = 0
                self.accelerator.print(f"New best validation loss: {self.best_val_loss:.4f}")
            else:
                self.epochs_without_improvement += 1

            # Save checkpoint with frequency control
            should_checkpoint = (
                is_best or  # Always save best models
                (epoch + 1) % self.checkpoint_frequency == 0 or  # Regular frequency
                epoch == num_epochs - 1 or  # Last epoch
                self.epochs_without_improvement >= patience  # Before early stopping
            )
            
            if should_checkpoint:
                try:
                    self.save_checkpoint(epoch, is_best)
                except Exception as e:
                    self.logger.error(f"Failed to save checkpoint: {e}")
                    self.accelerator.print(f"Warning: Checkpoint saving failed at epoch {epoch + 1}")
                    # Continue training despite checkpoint failure

            if self.epochs_without_improvement >= patience:
                self.accelerator.print(f"Early stopping triggered after {patience} epochs.")
                break
        
        self.accelerator.print("Training finished. Evaluating on the test set with the best model.")
        # Wait for all processes before test evaluation
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            self.run_test_evaluation()
        self.accelerator.wait_for_everyone()

        self.accelerator.end_training()
        self.logger.info("Training process completed.")

    def run_test_evaluation(self):
        """Load the best model and evaluate on the test set."""
        if self.test_dataloader:
            best_model_path = self.checkpoint_dir / "best_model"
            if best_model_path.exists():
                self.accelerator.print(f"Loading best model from {best_model_path} for test evaluation.")
                
                def _load_model():
                    cpu_model = KG_LFM.from_pretrained(best_model_path)
                    cpu_model.to(self.device)
                    return self.accelerator.prepare(cpu_model)
                
                self.model = _load_model()

                test_metrics = self._evaluation_loop(self.test_dataloader, "Testing")

                
                self.accelerator.print(f"Test Set Evaluation | Loss: {test_metrics['loss']:.4f}")
                
                final_log = {
                    "test/final_loss": test_metrics['loss'],
                }
                
                if 'perplexity' in test_metrics:
                    final_log["test/final_perplexity"] = test_metrics['perplexity']
                    self.accelerator.print(f"Perplexity: {test_metrics['perplexity']:.2f}")
                
                self.accelerator.log(final_log)
            else:
                self.accelerator.print("No best model found to evaluate on the test set.")
        else:
            self.accelerator.print("Skipping test evaluation.")


def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(description="Train KG-LFM model")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to configuration file.")
    
    args = parser.parse_args()
    
    config = load_yaml_config(args.config)
    
    trainer = KG_LFM_Trainer(config, run_name=config.pretrain_conf.run_name, resume=config.pretrain_conf.resume)
    
    try:
        trainer.train()
    except Exception as e:
        logging.error(f"Training failed with an unexpected error: {e}", exc_info=True)
        # Ensure wandb run is closed on error
        if wandb.run:
            wandb.finish()
        raise

if __name__ == "__main__":
    main()