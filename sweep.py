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

from ray.tune.search.bayesopt import BayesOptSearch

class RayTuneMetricsTracker(DefaultMetricsTracker):
    """Metrics tracker for Ray Tune that reports metrics to the session."""

    def __init__(self):
        super().__init__()

    def reset(self):
        """Resets the tracker."""
        super().reset()

    def update(self, metrics: Dict[str, float], count: int = 1):
        """Update metrics."""
        super().update(metrics, count)
        
        # Report metrics to Ray Tune session
        for key, value in metrics.items():
            session.report({key: value})

    def get_averages(self) -> Dict[str, float]:
        """Get the current average of all tracked metrics."""
        super().get_averages()


def train_kg_lfm(sweep_config):
    """Training function for Ray Tune hyperparameter optimization."""
    # get the base configuration
    config = load_yaml_config(sweep_config["base_config"])
    
    # Update the configuration with the sweep parameters
    config.update({
        "learning_rate": sweep_config["learning_rate"],
        "gradient_accumulation_steps": sweep_config["gradient_accumulation_steps"],
        "num_heads": sweep_config["num_heads"],
        "num_quantizers": sweep_config["num_quantizers"],
        "codebook_size": sweep_config["codebook_size"],
        "shared_codebook": sweep_config["shared_codebook"]
    })
    
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
    
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    
    base_config_path = args.base_config
    
    ray.init()
    
    sched = AsyncHyperBandScheduler()
    
    algo = BayesOptSearch(
        random_search_steps=4,
        points_to_evaluate=[
            {
                "learning_rate": 1e-3,
                "gradient_accumulation_steps": 16,
                "num_heads": 8,
                "num_quantizers": 4,
                "codebook_size": 128,
                "shared_codebook": False
            }
        ]
    )
    
    resources_per_trial = {"cpu": 32, "gpu": 4}
    tuner = tune.Tuner(
        tune.with_resources(
            train_kg_lfm,
            resources=resources_per_trial
        ),
        tune_config=tune.TuneConfig(
            metric="validation_loss",
            mode="min",
            search_alg=algo,
            scheduler=sched
        ),
        run_config=air.RunConfig(
            name="kg_lfm_hyperparameter_sweep",
            stop={
                "validation_loss": 0.01,
                "training_iteration": 1000
            },
        ),
        param_space={
            "learning_rate": tune.loguniform(1e-4, 1e-2),
            "gradient_accumulation_steps": tune.choice([4, 8, 16]),
            "num_heads": tune.choice([4, 8, 16]),
            "num_quantizers": tune.choice([4, 8, 16]),
            "codebook_size": tune.choice([128, 256, 512]),
            "shared_codebook": tune.choice([True, False]),
            "base_config": tune.grid_search([base_config_path])
        }
    )
    results = tuner.fit()
    
    print("Best hyperparameters found were: ", results.get_best_result().config)
    
if __name__ == "__main__":
    main()  