import copy
from typing import Optional, Tuple, Union


from transformers import (
    AutoConfig, AutoModel, 
    AutoTokenizer, AutoModelForCausalLM,
    PreTrainedModel, GenerationConfig
)
from transformers.modeling_outputs import CausalLMOutputWithPast

import torch
import torch.nn as nn
from model.KG_encoder import KGEncoder

from torch_geometric.data import Batch

class KG_LFMConfig(AutoConfig):
    """
    Configuration class for the Knowledge Graph Language Foundation Model (KG_LFM).
    This class extends AutoConfig to include specific parameters for the KG_LFM model.
    
    """
    
    model_type = "kg_lfm"
    llm_model_name: str = "qwen/Qwen-7B-Chat"
    node_embding_dim: int = 1024  # Dimension of node embeddings
    edge_embedding_dim: int = 1024  # Dimension of edge embeddings
    final_embedding_dim: int = 1024  # Final embedding dimension after processing
    dropout: float = 0.2  # Dropout rate for the model
    num_heads: int = 1  # Number of attention heads in GATv2Conv
    num_quantizers: int = 3  # Number of quantizers for residual vector quantization
    codebook_size: int = 512  # Size of the codebook for vector quantization
    shared_codebook: bool = False  # Whether to use a shared codebook across quantizers
    graph_pooling: bool = True  # Whether to apply global mean pooling to the graph representations

    llm_config = None  # Configuration for the underlying LLM model


class KG_LFM(PreTrainedModel):
    """
    Knowledge Graph Language Foundation Model (KG_LFM) that combines a pre-trained language model 
    with a knowledge graph encoder.
    This model is designed to process both text and graph data, integrating them into a unified representation.
    
    """
    config_class = KG_LFMConfig
    
    main_input_name = "input_ids"
    supports_gradient_checkpointing = True
    
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        config: KG_LFMConfig,    
    ):
        self.config = config
        
        self.tokenizer = tokenizer
        self.special_kg_token = self.tokenizer.encode("<KG_EMBEDDING>", add_special_tokens=False)[0]
        
        self.llm_model_name = config.llm_model_name
        self.llm = AutoModel.from_pretrained(self.llm_model_name)
        self.config.llm_config = self.llm.config
        
        self.llm_embedding_dim = self.llm.config.hidden_size
        
        # Initialize the KGEncoder with the specified dimensions
        self.kg_encoder = KGEncoder(
            node_embedding_dim=config.node_embding_dim,
            edge_embedding_dim=config.edge_embedding_dim,
            final_embedding_dim=config.final_embedding_dim,
            dropout=config.dropout,
            num_heads=config.num_heads,
            num_quantizers=config.num_quantizers,
            codebook_size=config.codebook_size,
            shared_codebook=config.shared_codebook,
            graph_pooling=config.graph_pooling
        )
        
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        """
        Load a pre-trained KG_LFM model from the specified path.
        
        Args:
            pretrained_model_name_or_path (str): Path to the pre-trained model.
            *model_args: Additional positional arguments for the model.
            **kwargs: Additional keyword arguments for the model.
        
        Returns:
            KG_LFM: An instance of the KG_LFM model.
        """
        
        raise NotImplementedError("The from_pretrained method is not implemented yet. Please implement the from_pretrained method for KG_LFM.")
        
        
    def save_pretrained(self, output_dir, state_dict=None):
        """
        Save the KG_LFM model to the specified directory.
        
        Args:
            output_dir (str): Directory where the model will be saved.
            state_dict (dict, optional): State dictionary to save. If None, uses the model's current state.
        """
        if state_dict is None:
            state_dict = self.state_dict()
            
        if getattr(self, 'tokenizer', None) is not None:
            self.tokenizer.save_pretrained(output_dir)
            
        if getattr(self, 'llm', None) is not None:
            print(f"Saving LLM model to {output_dir}")
            llm_state_dict = {k: v for k, v in state_dict.items() if k.startswith('llm.')}
            self.llm.save_pretrained(output_dir, state_dict=llm_state_dict)
            self.config.llm_config = self.llm.config
            
        if getattr(self, 'kg_encoder', None) is not None:
            print(f"Saving KGEncoder model to {output_dir}")
            kg_state_dict = {k: v for k, v in state_dict.items() if k.startswith('kg_encoder.')}
            torch.save(kg_state_dict, f"{output_dir}/kg_encoder.pth")
            
        # Save the configuration
        self.config._name_or_path = output_dir
        self.config.save_pretrained(output_dir)
    
    def prepare_inputs_labels_for_multimodal(
        self,
    ):
        """ 
        Prepares the inputs and labels for multimodal training.
        """
        raise NotImplementedError("The prepare_inputs_labels_for_multimodal method is not implemented yet. Please implement the prepare_inputs_labels_for_multimodal method for KG_LFM.")
    
    @torch.inference_mode()
    def generate(
        self,
        input_ids: torch.Tensor,
        graphs: Batch,
        attention_mask: Optional[torch.Tensor] = None,
        cfg: Optional[float] = 3.0,
        **generation_kwargs
    ) -> torch.Tensor:
        """
        Generate text based on the input IDs and graphs.
        
        Args:
            input_ids (torch.Tensor): Input token IDs for the language model.
            graphs (Batch): A batch of graphs from PyTorch Geometric.
            attention_mask (Optional[torch.Tensor]): Attention mask for the input tokens.
            cfg (Optional[float]): Guidance scale for controlled generation.
            **generation_kwargs: Additional keyword arguments for generation.
        
        Returns:
            torch.Tensor: Generated token IDs.
        """
        
        raise NotImplementedError("The generate method is not implemented yet. Please implement the generate method for KG_LFM.")
     
    @torch.inference_mode()
    def generate_graphs():
        """
        Etension of the previous method where the model can use the quantized graph embeddings to generate triplets.
        It will implement a <ASK> and <TELL> token added to the model to generate triplets from the graph embeddings.
        THIS IS NOT TO BE IMPLEMENTED YET, IT IS A FUTURE WORK.
        """
        raise NotImplementedError("The generate_graphs method is not implemented yet. Please implement the generate_graphs method for KG_LFM.")
    
    @property
    def default_generation_config(self):
        """
        Returns the default generation configuration for the KG_LFM model.
        
        This method retrieves the generation configuration from the underlying LLM model.
        """
        generation_config = copy.deepcopy(self.llm.generation_config)
        if self.tokenizer.eos_token is None:
            raise ValueError("The tokenizer does not have an end-of-sequence token defined. Please set the eos_token in the tokenizer.")
        if generation_config.max_length == GenerationConfig().max_length:
            generation_config.max_length = self.tokenizer.model_max_length
        if generation_config.pad_token_id is None:
            generation_config.pad_token_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        if generation_config.bos_token_id is None:
            generation_config.bos_token_id = self.tokenizer.bos_token_id or self.tokenizer.eos_token_id
        if generation_config.eos_token_id is None:
            generation_config.eos_token_id = self.tokenizer.convert_tokens_to_ids(infer_stop_tokens(self.tokenizer))
        return generation_config
    
    def forward(
        self, 
        input_ids : torch.Tensor, 
        graphs: Batch,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        """
        Forward pass of the KG_LFM model.
        """
        
        raise NotImplementedError("The forward method is not implemented yet. Please implement the forward method for KG_LFM.")
    
    
AutoConfig.register(KG_LFMConfig.model_type, KG_LFMConfig)
AutoModel.register(KG_LFMConfig, KG_LFM)