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
    big_graph_training_epochs: int = 0
    """Number of epochs for training on big graphs. Originally 1000. Set to 0 to use untrained embeddings."""
    
    def __post_init__(self):
        # Ensure that the base path is set correctly
        assert os.path.exists(self.base_path), f"Base path {self.base_path} does not exist."
        assert os.path.exists(self.graph_embs_base_path), f"Graph embeddings path {self.graph_embs_base_path} does not exist."
        
        # Ensure that the graph nodes embedding model is set correctly
        assert self.big_graph_training_epochs >= 0, "big_graph_training_epochs must be non-negative."

@dataclass
class TriRex_DataLoaderConfig():
    """
    Configuration for the combined TriREx and TRExStar dataloader.
    """
    batch_size: int = 32
    
    shuffle: bool = True
    num_workers: int = 0
    
    max_sequence_length: int = 512
    include_graphs: bool = True
    
    # Collation options
    padding: Union[bool, str] = True
    pad_to_multiple_of: Optional[int] = None
    
    return_tensors: str = "pt"
    
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
    
    kg_encoder_lr: Optional[float] = 1e-4
    """Learning rate for the KG encoder optimizer. Defaults to 1e-4."""
    
    # Graph Node Embedding Configuration
    graph_nodes_embedding_model: str = "Qwen/Qwen3-Embedding-0.6B"
    """Model used for graph nodes embedding. Defaults to 'Qwen/Qwen3-Embedding-0.6B'."""
    


@dataclass
class ProjectConfig:
    
    dataset: TRex_DatasetConfig = field(default_factory=TRex_DatasetConfig)
    """General configuration for the datasets."""
    
    pretrain_data: TriRex_DataLoaderConfig = field(default_factory=TriRex_DataLoaderConfig)
    
    model: ModelConfig = field(default_factory=ModelConfig)
    
    """Configuration for pretraining on Tri-REx."""
    
    # def __post_init__(self):
    #     # Ensure that the dataset base path is set correctly
    #     self.pretrain.dataloader.graph_nodes_embedding_model = self.model.graph_nodes_embedding_model


def load_yaml_config(path) -> ProjectConfig:
    with open(path) as file:
        return ProjectConfig(**yaml.load(file, Loader=yaml.FullLoader))
