from KG_LFM.configuration import load_yaml_config
from KG_LFM.trainer import KG_LFM_Trainer, DefaultMetricsTracker
import argparse
import logging
import copy
import os
from filelock import FileLock

import ray
from ray import air, tune
from ray.air import session
from ray.tune.schedulers import AsyncHyperBandScheduler

from typing import Dict

from ray.tune.stopper import MaximumIterationStopper, TimeoutStopper

class RayTuneMetricsTracker(DefaultMetricsTracker):
    """Metrics tracker for Ray Tune that reports metrics to the session."""

    def __init__(self):
        super().__init__()

    def reset(self):
        """Resets the tracker."""
         # Only report to Ray Tune if we have validation_loss
        # This prevents premature reporting of only training_loss
        if 'validation_loss' in self.values:
            # Get current averages and report them
            averages = self.get_averages()
            session.report(averages)
        
        super().reset()

    def update(self, metrics: Dict[str, float], count: int = 1):
        """Update metrics."""
        super().update(metrics, count)

    def get_averages(self) -> Dict[str, float]:
        """Get the current average of all tracked metrics."""
        return super().get_averages()


def train_kg_lfm(sweep_config):
    """Training function for Ray Tune hyperparameter optimization."""
    # get the base configuration
    config = load_yaml_config(sweep_config["base_config"])
    
    # Update the configuration with the sweep parameters
    config.train_conf.learning_rate = sweep_config["learning_rate"]
    config.model.num_heads = sweep_config["num_heads"]
    config.model.num_quantizers = sweep_config["num_quantizers"]
    config.model.codebook_size = sweep_config["codebook_size"]
    config.model.use_lora = sweep_config["use_lora"]

    # Ensure the run name is unique for each trial
    config.train_conf.run_name = f"{config.train_conf.run_name}_{session.get_trial_id()}"
    
    logging.info(f"Starting training for trial {session.get_trial_id()} with config: {sweep_config}")
    
    # Initialize the trainer with the updated config
    trainer = KG_LFM_Trainer(
        config=config, 
        run_name=session.get_trial_id(),
        resume=False,
        enable_wandb=False,
        save_checkpoints=False,
        metrics_tracker=RayTuneMetricsTracker()
    )
    
    trainer.train()

    # Print the results
    logging.info(f"Training completed for trial {session.get_trial_id()}.")
    logging.info(f"Final metrics: {trainer.metrics_tracker.get_averages()} for config: {sweep_config}")

def main():
    """Main function with Ray Tune hyperparameter optimization."""
    parser = argparse.ArgumentParser(description="Ray Tune hyperparameter sweep for KG-LFM model")
    parser.add_argument("--base_config", type=str, default="configs/sweep_base_config.yaml", 
                       help="Path to base configuration file.")
    parser.add_argument("--time_budget", type=int, default=3600*23,
                       help="Time budget for the sweep in seconds.")

    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    )
    
    base_config_path = args.base_config
    time_budget = args.time_budget
    
    ray.init()
    
    sched = AsyncHyperBandScheduler()
    
    resources_per_trial = {"cpu": 8, "gpu": 1}  # Adjust based on your total resources
    tuner = tune.Tuner(
        tune.with_resources(
            train_kg_lfm,
            resources=resources_per_trial
        ),
        tune_config=tune.TuneConfig(
            metric="validation_loss",
            mode="min",
            num_samples=20,
            scheduler=sched,
            time_budget_s=time_budget, 
        ),
        run_config=air.RunConfig(
            name="kg_lfm_hyperparameter_sweep",
            stop=TimeoutStopper(time_budget//4),  # Stop after time_budget
        ),
        param_space={
            "learning_rate": tune.loguniform(1e-4, 1e-2),
            "num_heads": tune.choice([4, 8, 16]),
            "num_quantizers": tune.choice([4, 8, 10, 16]),
            "codebook_size": tune.choice([128, 256, 512]),
            "use_lora": tune.choice([True, False]),
            "base_config": tune.grid_search([base_config_path]),
        }
    )
    results = tuner.fit()
    
    print("Best hyperparameters found were: ", results.get_best_result().config)
    
if __name__ == "__main__":
    main()  