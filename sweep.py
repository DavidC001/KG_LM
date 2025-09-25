import torch
from KG_LM.configuration import load_yaml_config
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs, DeepSpeedPlugin
from KG_LM.trainer import KG_LM_Trainer, DefaultMetricsTracker
import logging
import argparse
import os

import ray
from ray import air, tune, train
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.stopper import TimeoutStopper
from ray.train.torch import TorchTrainer 
from ray.air.config import ScalingConfig
# NEW: Import advanced search algorithms
from ray.tune.search.optuna import OptunaSearch
from ray.tune.search import ConcurrencyLimiter
from ray.tune.integration.ray_train import TuneReportCallback
import optuna

import uuid

class RayTuneMetricsTracker(DefaultMetricsTracker):
    """Metrics tracker for Ray Tune that reports metrics to the session."""

    def reset(self):
        """Resets the tracker."""
        # Only report to Ray Tune if we have validation_loss
        # This prevents premature reporting of only training_loss
        if 'validation_loss' in self.values:
            print("Reporting metrics to Ray...")
            # Get current averages and report them
            averages = self.get_averages()
            train.report(averages)
        
        super().reset()

def train_KG_LM(config):
    """Training function for Ray Tune hyperparameter optimization with Accelerate."""
    try:
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)
        
        # Get the base configuration
        base_config_path = config["base_config"]
        config_obj = load_yaml_config(base_config_path)
        
        # Update the configuration with the sweep parameters
        config_obj.model.num_heads = config["num_heads"]
        config_obj.model.num_quantizers = config["num_quantizers"]
        config_obj.model.codebook_size = config["codebook_size"]

        # Ensure the run name is unique for each trial
        trial_id = str(uuid.uuid4())
        config_obj.train_conf.run_name = f"RAY_{config_obj.train_conf.run_name}_{trial_id}_{config_obj.model.num_heads}_{config_obj.model.num_quantizers}_{config_obj.model.codebook_size}"

        logger.info(f"Starting training for trial {trial_id} with config: {config}")
        
        # Initialize Accelerator using config file
        deepspeed_config_path = "/leonardo/home/userexternal/dcavicch/projects/KG_LM/configs/deepspeed_config.json"
        
        # Load accelerate config
        kwargs = DistributedDataParallelKwargs(
            find_unused_parameters=False,
        )
        accelerator = Accelerator(
            mixed_precision="bf16",  # From deepspeed_config.json bf16.enabled: true
            step_scheduler_with_optimizer=False,
            gradient_accumulation_steps=config_obj.train_conf.gradient_accumulation_steps,
            deepspeed_plugin=DeepSpeedPlugin(deepspeed_config_path),
            kwargs_handlers=[kwargs],
            cpu=False,
        )
        
        # Initialize the trainer with the updated config AND the accelerator
        trainer = KG_LM_Trainer(
            config=config_obj, 
            run_name=config_obj.train_conf.run_name,
            resume=False,
            enable_wandb=True,
            save_checkpoints=True,
            metrics_tracker=RayTuneMetricsTracker,
            accelerator=accelerator  # Pass the accelerator we created
        )
        
        # Run training and report metrics
        trainer.train(config["time_budget_s"])  # Pass the time budget if specified
        # I do not close the trainer here, as Ray Tune expects to handle the process group
        
        logger.info(f"Training completed for trial {trial_id}.")
        
        # Report final metrics to Ray Tune
        final_metrics = trainer.best_val_loss
        train.report({"validation_loss": final_metrics})
        
    except Exception as e:
        logger.error(f"Error during training: {e}")
        
        # Report failure to Ray Tune
        train.report({"validation_loss": 1e10})  # Report a high loss to indicate failure

def launch_torch_trainer(config):
    """Launches the TorchTrainer with the given configuration."""
    scaling = ScalingConfig(num_workers=4, use_gpu=True, resources_per_worker={"GPU": 1, "CPU": 3})
    
    torch_trainer = TorchTrainer(
        train_loop_per_worker=train_KG_LM,
        train_loop_config=config,
        scaling_config=scaling,
        run_config=train.RunConfig(
            name="KG_LM_hyperparameter_sweep",
            callbacks=[TuneReportCallback()],
        ),
    )
    
    torch_trainer.fit()

def main():
    """Main function with Ray Tune hyperparameter optimization."""
    parser = argparse.ArgumentParser(description="Ray Tune hyperparameter sweep for KG-LFM model")
    parser.add_argument("--base_config", type=str, default="/leonardo/home/userexternal/dcavicch/projects/KG_LM/configs/sweep_base_config.yaml", 
                       help="Path to base configuration file.")
    parser.add_argument("--time_budget", type=int, default=3600*23,
                       help="Time budget for the sweep in seconds.")
    parser.add_argument("--num_concurrent_trials", type=int, default=1,
                       help="Number of concurrent trials to run.")
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        filename="logs/sweep.log",
    )
    
    base_config_path = args.base_config
    time_budget = args.time_budget
    time_budget = max(time_budget//4, 8*3600-5*60)
    num_concurrent_trials = args.num_concurrent_trials
    
    ray.init(
        include_dashboard=False,
    )
    
    storage_path = "~/ray_results"
    exp_name = "KG_LM_hyperparameter_sweep"
    path = os.path.join(storage_path, exp_name)
    
    param_space = {
        "num_heads": tune.choice([8, 16, 32]),
        "num_quantizers": tune.choice([5, 10, 20]),
        "codebook_size": tune.choice([64, 128, 256]),
        "base_config": base_config_path,
        "time_budget_s": time_budget  # Ensure at least 8 hours for each trial
    }
    
    if tune.Tuner.can_restore(path):
        tuner = tune.Tuner.restore(path, trainable=launch_torch_trainer, restart_errored=True, resume_unfinished=False, param_space=param_space)
    else:
        algo = OptunaSearch(
            sampler=optuna.samplers.TPESampler(
                n_startup_trials=num_concurrent_trials,  # Random trials before TPE
                n_ei_candidates=num_concurrent_trials,   # Number of candidates per iteration
                seed=42
            ),
            points_to_evaluate=[
                {
                    "num_heads": 16,
                    "num_quantizers": 10,
                    "codebook_size": 256,
                },
            ],
            metric="hit@10",
            mode="min"
        )
        
        # Limit concurrency for better performance
        algo = ConcurrencyLimiter(algo, max_concurrent=num_concurrent_trials)
        
        tuner = tune.Tuner(
            trainable=launch_torch_trainer,
            tune_config=tune.TuneConfig(
                metric="validation_loss",
                mode="min",
                num_samples=-1,
                # scheduler=scheduler,
                # time_budget_s=time_budget,
                search_alg=algo
            ),
            run_config=air.RunConfig(
                name="KG_LM_hyperparameter_sweep",
            ),
            param_space=param_space
        )
    
    results = tuner.fit()
    
    print("Best hyperparameters found were: ", results.get_best_result().config)
    
    # for testing call directly
    # launch_torch_trainer({
    #     "learning_rate": 0.001,
    #     "weight_decay": 0.0001,
    #     "num_heads": 8,
    #     "num_quantizers": 4,
    #     "codebook_size": 256,
    #     "base_config": base_config_path
    # })
    
if __name__ == "__main__":
    main()  