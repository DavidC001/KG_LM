import copy
import os
from dataclasses import dataclass as og_dataclass
from dataclasses import is_dataclass
from dataclasses import field
from typing import Optional, Union

import yaml

IGNORE_INDEX = -100
SPECIAL_KG_TOKEN = " <KG_EMBEDDING>"

def dataclass(*args, **kwargs):
    """
    Creates a dataclass that can handle nested dataclasses
    and automatically convert dictionaries to dataclasses.
    """

    def wrapper(cls):
        cls = og_dataclass(cls, **kwargs)
        original_init = cls.__init__

        def __init__(self, *args, **kwargs):
            for name, value in kwargs.items():
                field_type = cls.__annotations__.get(name, None)
                if is_dataclass(field_type) and isinstance(value, dict):
                    new_obj = field_type(**value)
                    kwargs[name] = new_obj
            original_init(self, *args, **kwargs)

        cls.__init__ = __init__
        return cls

    return wrapper(args[0]) if args else wrapper

@dataclass
class DatasetConfig:
    name: str = "trirex"
    """
    Type of the dataset. Defaults to 'trirex'.
    Options are 'trirex', 'trirex-bite', ...
    """
    
    lite: bool = False
    """Whether to use the lite version of the dataset. Defaults to False."""
    
    # base path from $FAST/datase
    base_path: str = os.path.join(os.getenv("FAST", ""), "dataset", "KG_LFM")
    """Base path for the dataset. Defaults to $FAST/dataset/KG_LFM."""
    
    graph_embs_base_path: str = os.path.join(os.getenv("FAST", ""), "dataset", "KG_LFM", "graph_embs")
    """Base path for the graph embeddings. Defaults to $FAST/dataset/KG_LFM/graph_embs."""
    
    graph_nodes_embedding_model: str = "Qwen/Qwen3-Embedding-0.6B"
    """Model used for graph nodes embedding. Is set to the same as the one used in the model config."""
    
    preload_nodes_embeddings: bool = True
    """Whether to preload graph nodes embeddings into memory for faster access during training."""

@dataclass
class DataLoaderConfig():
    """
    Configuration for the combined TriREx and TRExStar dataloader.
    """
    batch_size: int = 32
    
    shuffle: bool = True
    num_workers: int = 0
    
    max_sequence_length: int = None
    include_graphs: bool = True
    
    # Collation options
    padding: Union[bool, str] = True
    pad_to_multiple_of: Optional[int] = None
    
    return_tensors: str = "pt"
    
    pin_memory: bool = False
    """Whether to use pinned memory for faster GPU transfers."""

    persistent_workers: bool = False
    """Whether to keep data loading workers alive between epochs to avoid startup overhead."""
    
    
@dataclass
class ModelConfig:
    # LLM Configuration
    llm_model_name: str = "Qwen/Qwen3-8B"
    """Name of the base language model to use. Defaults to 'qwen/Qwen-7B-Chat'."""
    
    graph_pooling: bool = True
    """Whether to apply global mean pooling to the graph representations. Defaults to True."""
    
    dropout: float = 0.2
    """Dropout rate for the model. Defaults to 0.2."""
    
    num_heads: int = 1
    """Number of attention heads in GATv2Conv. Defaults to 1."""
    
    gat_layers: int = 1
    """Number of GATv2 layers to apply."""

    num_quantizers: int = 3
    """Number of quantizers for residual vector quantization. Defaults to 3."""
    
    codebook_size: int = 512
    """Size of the codebook for vector quantization. Defaults to 512."""
    
    shared_codebook: bool = False
    """Whether to use a shared codebook across quantizers. Defaults to False."""
    
    codebook_dim: int = 0
    """Dimension of the downsampled codebook. If 0, no downsampling is applied."""

    # Training Configuration
    tune_language_model: bool = False
    """Whether to tune the language model base parameters (does not affect lora params) during training. Defaults to False."""
    
    use_lora: bool = True
    """Whether to use LoRA for training. Defaults to True."""
    lora_r: int = 8
    """Rank for LoRA. Defaults to 8."""
    lora_alpha: int = 16
    """Alpha for LoRA scaling. Defaults to 16."""
    lora_target_modules: list[str] = field(default_factory=lambda: ["q_proj", "k_proj"])
    
    tune_kg_encoder: bool = True
    """Whether to tune the knowledge graph encoder during training. Defaults to True."""
    
    # Graph Node Embedding Configuration
    graph_nodes_embedding_model: str = "Qwen/Qwen3-Embedding-0.6B"
    """Model used for graph nodes embedding. Defaults to 'Qwen/Qwen3-Embedding-0.6B'."""
    
@dataclass
class TrainConfig:
    """
    Configuration for pretraining on Tri-REx.
    """
    run_name: str = "pretrain_trirex"
    
    steps_train: float = 1.0
    """Number of training steps per training before validation. Can be a float representing a percentage of the dataset. Defaults to 1.0 (100%)."""
    eval_perc: float = 1.0
    """Percentage of the evaluation dataset to use for validation after each training step. Defaults to 1 (100%)."""
    
    epochs: int = 20
    """Number of epochs for pretraining. Defaults to 20."""
    
    early_stopping_patience: int = 3
    """Patience for early stopping during pretraining. Defaults to 3."""
    
    scheduler_patience: int = 2
    """Patience for the ReduceLROnPlateau scheduler. Defaults to 2."""
    
    scheduler_metric: str = "validation_loss"
    """Metric to monitor for the learning rate scheduler. Defaults to 'validation_loss'."""
    
    KG_learning_rate: float = 1e-4
    """Learning rate for the KG optimizer. Defaults to 1e-4."""
    
    LLM_learning_rate: float = 1e-4
    """Learning rate for the LLM optimizer. Defaults to 1e-4."""

    weight_decay: float = 0.01
    """Weight decay for the pretraining optimizer. Defaults to 0.01."""
    
    gradient_accumulation_steps: int = 1
    """Number of gradient accumulation steps. Defaults to 1."""
    
    clip_grad_norm: float = 1.0
    """Maximum gradient norm for clipping. Defaults to 1.0."""
    
    rvq_loss_weight: float = 1.0
    """Weight for the RVQ loss component. Defaults to 1.0."""
    
    dataloader: DataLoaderConfig = field(default_factory=DataLoaderConfig)
    
    checkpoint_dir: str = "out/pretrain"
    """Directory to save checkpoints during pretraining. Defaults to 'out/pretrain'."""
    checkpoint_frequency: int = 1
    """Frequency of saving checkpoints during pretraining. Defaults to every epoch."""
    
    start_from_checkpoint: Optional[str] = None
    """Path to a checkpoint to load model weights from before training. Defaults to None."""
    
    resume : bool = False
    
    def __post_init__(self):
        # Ensure that learning rate and weight decay are numbers
        self.KG_learning_rate = float(self.KG_learning_rate)
        self.LLM_learning_rate = float(self.LLM_learning_rate)
        self.weight_decay = float(self.weight_decay)


@dataclass
class ProjectConfig:
    seed: int = 42
    """Random seed for reproducibility. Defaults to 42."""
    
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    """General configuration for the datasets."""
    
    train_conf: TrainConfig = field(default_factory=TrainConfig)
    """Configuration for pretraining on Tri-REx."""
    
    model: ModelConfig = field(default_factory=ModelConfig)
    """Configuration for the model architecture and training."""
    
    def __post_init__(self):
        # Ensure that the dataset base path is set correctly
        self.dataset.graph_nodes_embedding_model = self.model.graph_nodes_embedding_model
    
    def recursive_dict(self, in_dict):
        """
        Convert the ProjectConfig instance to a nested dictionary.
        This is useful for serialization or logging.
        """
        for key, value in in_dict.items():
            if is_dataclass(value):
                in_dict[key] = self.recursive_dict(value.__dict__)
            elif isinstance(value, list):
                in_dict[key] = [self.recursive_dict(item) if is_dataclass(item) else item for item in value]
            elif isinstance(value, dict):
                in_dict[key] = self.recursive_dict(value)
        return in_dict
    
    def to_dict(self):
        """
        Convert the ProjectConfig instance to a dictionary.
        This is useful for serialization or logging.
        """
        return self.recursive_dict(copy.deepcopy(self.__dict__))
                


def load_yaml_config(path) -> ProjectConfig:
    with open(path) as file:
        return ProjectConfig(**yaml.load(file, Loader=yaml.FullLoader))
