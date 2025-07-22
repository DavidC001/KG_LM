import copy
import os
from dataclasses import dataclass as og_dataclass
from dataclasses import is_dataclass
from dataclasses import field
from typing import Optional, Union

import yaml

IGNORE_INDEX = -100

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
class TRex_DatasetConfig:
    lite: bool = False
    """Whether to use the lite version of the dataset. Defaults to False."""
    
    # base path from $FAST/datase
    base_path: str = os.path.join(os.getenv("FAST", ""), "dataset", "Tri-Rex_V1")
    """Base path for the dataset. Defaults to $FAST/dataset/Tri-Rex_V1."""
    
    
    graph_embs_base_path: str = os.path.join(os.getenv("FAST", ""), "dataset", "Tri-Rex_V1", "graph_embs")
    """Base path for the graph embeddings. Defaults to $FAST/dataset/Tri-Rex_V1/graph_embs."""
    
    graph_nodes_embedding_model: str = "Qwen/Qwen3-Embedding-0.6B"
    """Model used for graph nodes embedding. Is set to the same as the one used in the model config."""
    
    preload_nodes_embeddings: bool = True
    """Whether to preload graph nodes embeddings into memory for faster access during training."""
    
    def __post_init__(self):
        # Ensure that the base path is set correctly
        assert os.path.exists(self.base_path), f"Base path {self.base_path} does not exist."
        assert os.path.exists(self.graph_embs_base_path), f"Graph embeddings path {self.graph_embs_base_path} does not exist."

@dataclass
class TriRex_DataLoaderConfig():
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
    
    pin_memory: bool = True
    """Whether to use pinned memory for faster GPU transfers."""
    
    persistent_workers: bool = True
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
    
    num_quantizers: int = 3
    """Number of quantizers for residual vector quantization. Defaults to 3."""
    
    codebook_size: int = 512
    """Size of the codebook for vector quantization. Defaults to 512."""
    
    shared_codebook: bool = False
    """Whether to use a shared codebook across quantizers. Defaults to False."""
    
    # Training Configuration
    tune_language_model: bool = False
    """Whether to tune the language model during training. Defaults to False."""
    
    tune_kg_encoder: bool = True
    """Whether to tune the knowledge graph encoder during training. Defaults to True."""
    
    # Graph Node Embedding Configuration
    graph_nodes_embedding_model: str = "Qwen/Qwen3-Embedding-0.6B"
    """Model used for graph nodes embedding. Defaults to 'Qwen/Qwen3-Embedding-0.6B'."""
    
@dataclass
class PretrainConfig:
    """
    Configuration for pretraining on Tri-REx.
    """
    run_name: str = "pretrain_trirex"
    
    steps_train: int = -1
    """Number of training steps per training before validation. Defaults to -1 (no limit)."""
    eval_perc: float = 1
    """Percentage of the evaluation dataset to use for validation after each training step. Defaults to 1 (100%)."""
    
    epochs: int = 20
    """Number of epochs for pretraining. Defaults to 20."""
    
    early_stopping_patience: int = 3
    """Patience for early stopping during pretraining. Defaults to 3."""
    
    learning_rate: float = 1e-4
    """Learning rate for the pretraining optimizer. Defaults to 1e-4."""
    
    scheduler_eta_min: float = 1e-5
    """Minimum learning rate for the scheduler. Defaults to 1e-5."""
    
    weight_decay: float = 0.01
    """Weight decay for the pretraining optimizer. Defaults to 0.01."""
    
    gradient_accumulation_steps: int = 1
    """Number of gradient accumulation steps. Defaults to 1."""
    
    clip_grad_norm: float = 1.0
    """Maximum gradient norm for clipping. Defaults to 1.0."""
    
    dataloader: TriRex_DataLoaderConfig = field(default_factory=TriRex_DataLoaderConfig)
    
    checkpoint_dir: str = "out/pretrain"
    """Directory to save checkpoints during pretraining. Defaults to 'out/pretrain'."""
    checkpoint_frequency: int = 1
    """Frequency of saving checkpoints during pretraining. Defaults to every epoch."""
    
    resume : bool = False
    
    def __post_init__(self):
        # Ensure that learning rate and weight decay are numbers
        self.learning_rate = float(self.learning_rate)
        self.weight_decay = float(self.weight_decay)
        self.scheduler_eta_min = float(self.scheduler_eta_min)


@dataclass
class ProjectConfig:
    seed: int = 42
    """Random seed for reproducibility. Defaults to 42."""
    
    dataset: TRex_DatasetConfig = field(default_factory=TRex_DatasetConfig)
    """General configuration for the datasets."""
    
    pretrain_conf: PretrainConfig = field(default_factory=PretrainConfig)
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
