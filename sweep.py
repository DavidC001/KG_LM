from KG_LFM.configuration import load_yaml_config
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs, DeepSpeedPlugin
from KG_LFM.trainer import KG_LFM_Trainer, DefaultMetricsTracker
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
import optuna

import uuid

class RayTuneMetricsTracker(DefaultMetricsTracker):
    """Metrics tracker for Ray Tune that reports metrics to the session."""

    def reset(self):
        """Resets the tracker."""
        # Only report to Ray Tune if we have validation_loss
        # This prevents premature reporting of only training_loss
        if 'validation_loss' in self.values and train.get_context().get_world_rank() == 0:
            # Get current averages and report them
            averages = self.get_averages()
            train.report(averages)
        
        super().reset()

def train_kg_lfm(config):
    """Training function for Ray Tune hyperparameter optimization with Accelerate."""
    try:
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)
        
        # Get the base configuration
        base_config_path = config["base_config"]
        config_obj = load_yaml_config(base_config_path)
        
        # Update the configuration with the sweep parameters
        config_obj.train_conf.learning_rate = config["learning_rate"]
        config_obj.train_conf.weight_decay = config["weight_decay"]
        config_obj.model.num_heads = config["num_heads"]
        config_obj.model.num_quantizers = config["num_quantizers"]
        config_obj.model.codebook_size = config["codebook_size"]
        
        # Ensure the run name is unique for each trial
        trial_id = str(uuid.uuid4())
        config_obj.train_conf.run_name = f"{config_obj.train_conf.run_name}_{trial_id}"
        
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
            deepspeed_plugin=DeepSpeedPlugin(deepspeed_config_path),
            kwargs_handlers=[kwargs],
            cpu=False,
        )
        
        # Initialize the trainer with the updated config AND the accelerator
        trainer = KG_LFM_Trainer(
            config=config_obj, 
            run_name=config_obj.train_conf.run_name,
            resume=False,
            enable_wandb=False,  # Disable wandb for Ray Tune
            save_checkpoints=False,  # Let Ray handle checkpoints
            metrics_tracker=RayTuneMetricsTracker(),
            accelerator=accelerator  # Pass the accelerator we created
        )
        
        # Run training and report metrics
        trainer.train()
        
        logger.info(f"Training completed for trial {trial_id}.")
        
        if train.get_context().get_world_rank() == 0:
            # Report final metrics to Ray Tune
            final_metrics = trainer.best_val_loss
            train.report({"validation_loss": final_metrics})
        
    except Exception as e:
        logger.error(f"Error during training: {e}")
        
        # Report failure to Ray Tune
        if train.get_context().get_world_rank() == 0:
            train.report({"validation_loss": 1e10})  # Report a high loss to indicate failure

def launch_torch_trainer(config):
    """Launches the TorchTrainer with the given configuration."""
    scaling = ScalingConfig(num_workers=4, use_gpu=True, resources_per_worker={"GPU": 1, "CPU": 7})
    
    torch_trainer = TorchTrainer(
        train_loop_per_worker=train_kg_lfm,
        train_loop_config=config,
        scaling_config=scaling,
        run_config=train.RunConfig(
            name="kg_lfm_hyperparameter_sweep",
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
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        filename="logs/sweep.log",
    )
    
    base_config_path = args.base_config
    time_budget = args.time_budget
    
    ray.init()
    
    # get the number of gpus and cpus
    num_gpus = ray.cluster_resources().get("GPU", 0)
    num_concurrent_trials = num_gpus//4
    logging.info(f"Number of GPUs available: {num_gpus}, Number of concurrent trials: {num_concurrent_trials}")
    
    storage_path = "~/ray_results"
    exp_name = "kg_lfm_hyperparameter_sweep"
    path = os.path.join(storage_path, exp_name)
    
    if tune.Tuner.can_restore(path):
        tuner = tune.Tuner.restore(path, trainable=launch_torch_trainer, resume_errored=True)
    else:
        algo = OptunaSearch(
            sampler=optuna.samplers.TPESampler(
                n_startup_trials=max(num_concurrent_trials, 6),  # Random trials before TPE
                n_ei_candidates=num_concurrent_trials,   # Number of candidates per iteration
                seed=42
            ),
            metric="validation_loss",
            mode="min"
        )
        
        # Limit concurrency for better performance
        algo = ConcurrencyLimiter(algo, max_concurrent=num_concurrent_trials)

        scheduler = AsyncHyperBandScheduler(
            grace_period=2
        )
        
        param_space = {
            "learning_rate": tune.loguniform(1e-4, 1e-3),
            "weight_decay": tune.loguniform(1e-5, 1e-2),
            "num_heads": tune.choice([4, 8, 16]),
            "num_quantizers": tune.choice([4, 8, 10, 16]),
            "codebook_size": tune.choice([128, 256, 512]),
            "base_config": base_config_path,
        }
        
        tuner = tune.Tuner(
            trainable=launch_torch_trainer,
            tune_config=tune.TuneConfig(
                metric="validation_loss",
                mode="min",
                num_samples=-1,
                scheduler=scheduler,
                time_budget_s=time_budget,
                search_alg=algo
            ),
            run_config=air.RunConfig(
                name="kg_lfm_hyperparameter_sweep",
                stop=TimeoutStopper(time_budget//4),  # Stop after time_budget
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