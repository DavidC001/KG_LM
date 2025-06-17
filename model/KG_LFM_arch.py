import copy
import logging
import os
import os.path as osp
import warnings

from abc import ABC
from collections import OrderedDict
from typing import Optional, Tuple, Union, List

from transformers import (
    AutoConfig, AutoModel, AutoModelForCausalLM,
    AutoTokenizer, PreTrainedTokenizer,
    PreTrainedModel, GenerationConfig
)
from transformers.modeling_outputs import CausalLMOutputWithPast

import torch
from model.KG_encoder import KGEncoder

from torch_geometric.data import Batch

# Constants
from configuration import IGNORE_INDEX, ModelConfig

def infer_stop_tokens(tokenizer):
    """Simple implementation of infer_stop_tokens for KG_LFM"""
    stop_tokens = []
    if tokenizer.eos_token:
        stop_tokens.append(tokenizer.eos_token)
    return stop_tokens

class KG_LFMConfig(AutoConfig):
    """
    Configuration class for the Knowledge Graph Language Foundation Model (KG_LFM).
    This class extends AutoConfig to include specific parameters for the KG_LFM model.
    
    """
    
    model_type = "kg_lfm"
    llm_model_name: str = "Qwen/Qwen3-8B"
    
    node_embedding_dim: int = 1024  # Dimension of node embeddings
    edge_embedding_dim: int = 1024  # Dimension of edge embeddings
    graph_pooling: bool = True  # Whether to apply global mean pooling to the graph representations
    
    dropout: float = 0.2  # Dropout rate for the model
    num_heads: int = 1  # Number of attention heads in GATv2Conv
    num_quantizers: int = 3  # Number of quantizers for residual vector quantization
    
    codebook_size: int = 512  # Size of the codebook for vector quantization
    shared_codebook: bool = False  # Whether to use a shared codebook across quantizers
    
    tune_language_model: bool = False  # Whether to tune the language model
    tune_kg_encoder: bool = False  # Whether to tune the knowledge graph encoder

def set_KGLM_model_args(config :KG_LFMConfig, model_args: ModelConfig):
    """
    Set the model arguments from a ModelConfig instance.
    
    Args:
        config (KG_LFMConfig): An instance of KG_LFMConfig to be updated with model parameters.
        model_args (ModelConfig): An instance of ModelConfig containing model parameters.
    """
    # get config for the LLM model to know the shape of the embeddings
    graph_nodes_embedding_model = model_args.graph_nodes_embedding_model
    graph_llm_config = AutoConfig.from_pretrained(
        graph_nodes_embedding_model,
    )
    # Check if the model is compatible with KG_LFM
    if not hasattr(graph_llm_config, 'hidden_size'):
        raise ValueError(f"The model {graph_nodes_embedding_model} does not have a 'hidden_size' attribute. "
                         "Please use a compatible model for KG_LFM.")
    
    config.model_type = "kg_lfm"    
    
    config.node_embedding_dim = graph_llm_config.hidden_size
    config.edge_embedding_dim = graph_llm_config.hidden_size
    
    config.llm_model_name = model_args.llm_model_name
    
    config.graph_pooling = model_args.graph_pooling
    config.dropout = model_args.dropout
    config.num_heads = model_args.num_heads
    config.num_quantizers = model_args.num_quantizers
    config.codebook_size = model_args.codebook_size
    config.shared_codebook = model_args.shared_codebook
    config.tune_language_model = model_args.tune_language_model
    config.tune_kg_encoder = model_args.tune_kg_encoder
    
    return config


class KG_LFMMetaModel(ABC):
    def init_KGLM(self, config: KG_LFMConfig = None, *args, **kwargs):
        if hasattr(self, "llm") or hasattr(self, "kg_encoder"):
            return 
        
        model_dtype = getattr(config, "model_dtype", "torch.float16")
        if not hasattr(config, "model_dtype"):
            warnings.warn("model_dtype not found in config, defaulting to torch.float16.")
            config.model_dtype = model_dtype

        self.llm = AutoModelForCausalLM.from_pretrained(config.llm_model_name, trust_remote_code=True)
        
        self.llm_embedding_dim = self.llm.config.hidden_size
        
        # Initialize the KGEncoder with the specified dimensions
        self.kg_encoder = KGEncoder(
            node_embedding_dim=config.node_embedding_dim,
            edge_embedding_dim=config.edge_embedding_dim,
            final_embedding_dim=self.llm_embedding_dim,
            dropout=config.dropout,
            num_heads=config.num_heads,
            num_quantizers=config.num_quantizers,
            codebook_size=config.codebook_size,
            shared_codebook=config.shared_codebook,
            graph_pooling=config.graph_pooling
        )

        assert (
            self.llm is not None and self.kg_encoder is not None
        ), "KG_LFM model initialization failed. Please check the configuration and ensure that the LLM and KGEncoder are properly initialized."

        self.is_loaded = True
        
    @classmethod
    def load_pretrained(cls, model_path_or_config, *args, **kwargs):
        raise NotImplementedError(
            "This method should be implemented."
        )

    def save_pretrained(self, output_dir, state_dict=None):
        if state_dict is None:
            state_dict = self.state_dict()
        
        if getattr(self, "tokenizer", None):
            self.tokenizer.save_pretrained(osp.join(output_dir, "llm"))

        if self.get_llm():
            print(f"saving llm to {osp.join(output_dir, 'llm')}")
            self.llm.config._name_or_path = osp.join(output_dir, "llm")
            llm_state_dict = OrderedDict({k.split("llm.")[-1]: v for k, v in state_dict.items() if "llm" in k})
            self.llm.save_pretrained(os.path.join(output_dir, "llm"), state_dict=llm_state_dict)

        if self.get_kg_encoder():
            print(f"saving kg_encoder to {osp.join(output_dir, 'kg_encoder')}")
            kg_encoder_state_dict = OrderedDict(
                {k.split("kg_encoder.")[-1]: v for k, v in state_dict.items() if "kg_encoder" in k}
            )
            torch.save(kg_encoder_state_dict, f"{output_dir}/kg_encoder.pth")

        self.config._name_or_path = output_dir
        self.config.architectures = [self.__class__.__name__]
        self.config.save_pretrained(output_dir)

    def get_llm(self):
        llm = getattr(self, "llm", None)
        if type(llm) is list:
            llm = llm[0]
        return llm

    def get_lm_head(self):
        lm_head = getattr(self.get_llm(), "lm_head", None)
        return lm_head

    def get_kg_encoder(self):
        kg_encoder = getattr(self, "kg_encoder", None)
        if type(kg_encoder) is list:
            kg_encoder = kg_encoder[0]
        return kg_encoder

    def correct_train(self):
        '''
        To ensure the expected behaviors for modules like dropout, batchnorm, etc., we need to call model.eval() for the freezed modules.
        '''
        self.train()
        
        if self.get_llm() and not getattr(self.config, "tune_language_model", False):
            self.get_llm().eval()
            logging.info("Freezed llm model to eval mode.")
        if self.get_kg_encoder() and not getattr(self.config, "tune_kg_encoder", False):
            self.get_kg_encoder().eval()
    
    def encode_graphs(self, graphs):
        kg_encoder = self.get_kg_encoder()
        graph_features, indexes, RVQ_loss = kg_encoder(graphs)
        return graph_features, indexes, RVQ_loss
    
    def _temporary_reorder_cache(self, past_key_values, sorted_idx):
        return self.get_llm()._temporary_reorder_cache(past_key_values, sorted_idx)

    def get_input_embeddings(self):
        return self.get_llm().get_input_embeddings()

    def get_output_embeddings(self):
        return self.get_llm().get_output_embeddings()


class KG_LFMMetaForCausalLM(ABC):
    def prepare_inputs_labels_for_multimodal(
        self,
        input_ids,
        position_ids,
        attention_mask,
        past_key_values,
        labels,
        graphs,
    ):
        """
        Prepares the inputs and labels for multimodal training by integrating graph embeddings
        into the text sequence at positions marked by special KG tokens.
        
        This method handles the complex process of:
        1. Encoding graphs into embeddings
        2. Replacing special KG tokens with actual graph embeddings
        3. Properly padding and aligning sequences for batch processing
        4. Managing attention masks and position IDs for the modified sequences
        """
        kg_encoder = self.get_kg_encoder()
        
        # Early return for cases where no multimodal processing is needed
        if kg_encoder is None or graphs is None or input_ids.shape[1] == 1:
            # Handle generation case where we have past key values and single token input
            if (
                past_key_values is not None
                and kg_encoder is not None
                and graphs is not None
                and input_ids.shape[1] == 1
            ):
                # Extend attention mask to account for cached key-value pairs
                target_shape = past_key_values[-1][-1].shape[-2] + 1
                attention_mask = torch.cat(
                    (
                        attention_mask,
                        torch.ones(
                            (
                                attention_mask.shape[0],
                                target_shape - attention_mask.shape[1],
                            ),
                            dtype=attention_mask.dtype,
                            device=attention_mask.device,
                        ),
                    ),
                    dim=1,
                )
                # Update position IDs to point to the last valid position
                position_ids = torch.sum(attention_mask, dim=1).unsqueeze(-1) - 1
            return (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                None,  # No input embeddings needed
                labels,
            )

        # Encode the knowledge graphs into embeddings that will replace KG tokens
        graph_features, _, RVQ_loss = kg_encoder(graphs)

        # Store original inputs for later reference
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask

        # If position IDs are not provided, create them based on input IDs
        if position_ids is None:
            position_ids = torch.arange(
                input_ids.shape[1], dtype=torch.long, device=input_ids.device
            ).unsqueeze(0).expand(input_ids.shape[0], -1)

        # Create attention mask if not provided (all tokens are attended to)
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()

        # Create labels if not provided (ignore all tokens in loss calculation)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        # Get embeddings for text tokens, replacing KG tokens with padding token (0) temporarily
        # This prevents embedding lookup errors for the special KG token
        input_ids_copy = input_ids.clone()
        input_ids_copy[input_ids_copy == self.special_kg_token] = 0
        input_embeds = self.llm.model.embed_tokens(input_ids_copy)

        # Extract valid tokens based on attention mask for each sample in the batch
        # This removes padding tokens from processing
        input_ids = [
            cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)
        ]
        input_embeds_1 = [
            cur_input_embeds[cur_attention_mask]
            for cur_input_embeds, cur_attention_mask in zip(input_embeds, attention_mask)
        ]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        # Lists to store the final processed embeddings and labels
        new_input_embeds = []
        new_labels = []
        
        # Process each sample in the batch individually
        for batch_idx, cur_input_ids in enumerate(input_ids):
            # Count how many KG tokens are in this sample
            num_kg_tokens = (cur_input_ids == self.special_kg_token).sum()
            
            if num_kg_tokens == 0:
                # No KG tokens found, just use the original text embeddings
                cur_input_embeds_1 = input_embeds_1[batch_idx]
                new_input_embeds.append(cur_input_embeds_1)
                new_labels.append(labels[batch_idx])
                continue

            # Get the current embeddings and labels for this sample
            cur_input_embeds = input_embeds_1[batch_idx]
            
            # Find positions of KG tokens and create segments between them
            # Add -1 at start and sequence length at end to handle boundaries
            kg_token_indices = (
                [-1] + torch.where(cur_input_ids == self.special_kg_token)[0].tolist() + [cur_input_ids.shape[0]]
            )
            cur_labels = labels[batch_idx]

            # Split the sequence into segments: text before first KG token, between KG tokens, after last KG token
            cur_input_ids_nokg = []
            cur_labels_nokg = []
            cur_input_embeds_no_kg = []

            # Extract segments between KG tokens (excluding the KG tokens themselves)
            for i in range(len(kg_token_indices) - 1):
                cur_input_ids_nokg.append(cur_input_ids[kg_token_indices[i] + 1 : kg_token_indices[i + 1]])
                cur_labels_nokg.append(cur_labels[kg_token_indices[i] + 1 : kg_token_indices[i + 1]])
                cur_input_embeds_no_kg.append(cur_input_embeds[kg_token_indices[i] + 1 : kg_token_indices[i + 1]])

            # Reconstruct the sequence by interleaving text segments with graph embeddings
            cur_new_input_embeds = []
            cur_new_labels = []

            for i in range(num_kg_tokens + 1):
                # Add the text segment
                cur_new_input_embeds.append(cur_input_embeds_no_kg[i])
                cur_new_labels.append(cur_labels_nokg[i])
                
                # Insert graph embedding after each text segment (except the last one)
                if i < num_kg_tokens:
                    # Get the graph features for this batch sample and add batch dimension
                    cur_graph_features = graph_features[batch_idx] #TODO: not sure if this is correct
                    cur_new_input_embeds.append(cur_graph_features)
                    
                    # Mark graph embedding positions to be ignored in loss calculation
                    # Graph embeddings should not contribute to language modeling loss
                    cur_new_labels.append(
                        torch.full(
                            (cur_graph_features.shape[0],),
                            IGNORE_INDEX,
                            device=cur_labels.device,
                            dtype=cur_labels.dtype,
                        )
                    )

            # Concatenate all segments and graph embeddings into final sequence
            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)

        # Handle sequence length limits by truncating if necessary
        tokenizer_model_max_length = getattr(self.llm.config, "tokenizer_model_max_length", None)
        if tokenizer_model_max_length is not None:
            if any(len(x) > tokenizer_model_max_length for x in new_input_embeds):
                warnings.warn("Inputs truncated!")
            new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]

        # Batch processing: pad all sequences to the same length
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        # Initialize padded tensors with appropriate default values
        new_input_embeds_padded = []
        new_labels_padded = torch.full(
            (batch_size, max_len),
            IGNORE_INDEX,  # Padded positions should be ignored in loss
            dtype=new_labels[0].dtype,
            device=new_labels[0].device,
        )
        # Create new attention mask for the modified sequences
        attention_mask = torch.zeros(
            (batch_size, max_len),
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        # Create new position IDs for the modified sequences
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

        # Pad each sequence and update corresponding masks and position IDs
        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            
            # Handle left padding (useful for some model architectures)
            if getattr(self.llm.config, "tokenizer_padding_side", "right") == "left":
                # Pad embeddings on the left side with zeros
                new_input_embeds_padded.append(
                    torch.cat(
                        (
                            torch.zeros(
                                (max_len - cur_len, cur_new_embed.shape[1]),
                                dtype=cur_new_embed.dtype,
                                device=cur_new_embed.device,
                            ),
                            cur_new_embed,
                        ),
                        dim=0,
                    )
                )
                # Update labels, attention mask, and position IDs for left padding
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(
                        0, cur_len, dtype=position_ids.dtype, device=position_ids.device
                    )
            else:
                # Handle right padding (default case)
                # Pad embeddings on the right side with zeros
                new_input_embeds_padded.append(
                    torch.cat(
                        (
                            cur_new_embed,
                            torch.zeros(
                                (max_len - cur_len, cur_new_embed.shape[1]),
                                dtype=cur_new_embed.dtype,
                                device=cur_new_embed.device,
                            ),
                        ),
                        dim=0,
                    )
                )
                # Update labels, attention mask, and position IDs for right padding
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(
                        0, cur_len, dtype=position_ids.dtype, device=position_ids.device
                    )

        # Stack all padded embeddings into a single batch tensor
        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        # Preserve None values for optional inputs that were originally None
        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            # Ensure attention mask has the same dtype as the original
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None

        # Return the processed inputs ready for the language model
        return (
            None,  # input_ids set to None since we're using input_embeds
            position_ids,
            attention_mask,
            past_key_values,
            new_input_embeds,  # The main output: text + graph embeddings
            new_labels,
            RVQ_loss,  # Return the RVQ loss for training purposes
        )
    
    @torch.inference_mode()
    def generate(
        self,
        input_ids: torch.Tensor,
        graphs: Batch,
        attention_mask: Optional[torch.Tensor] = None,
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
        if graphs is not None:
            (_, _, attention_mask, _, inputs_embeds, _) = self.prepare_inputs_labels_for_multimodal(
                input_ids, None, attention_mask, None, None, graphs
            )
        else:
            inputs_embeds = self.get_input_embeddings()(input_ids)
        
        inputs_embeds = inputs_embeds.to(self.dtype)

        outputs = self.llm.generate(inputs_embeds=inputs_embeds, attention_mask=attention_mask, **generation_kwargs)

        return outputs
     
    @torch.inference_mode()
    def generate_graphs(
        self,
        input_ids: torch.Tensor,
        graphs: Batch,
        attention_mask: Optional[torch.Tensor] = None,
        **generation_kwargs
    ):
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


class KG_LFM(KG_LFMMetaModel, KG_LFMMetaForCausalLM, PreTrainedModel):
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
        config: KG_LFMConfig, 
    ):
        super(KG_LFM, self).__init__(config)
        self.config = config
        
        # Initialize components
        self.init_KGLM(config)
        
        
        self.tokenizer : PreTrainedTokenizer = AutoTokenizer.from_pretrained(
            config.llm_model_name, 
            use_fast=True, 
            trust_remote_code=True
        )
        if "<KG_EMBEDDING>" not in self.tokenizer.get_vocab():
            self.tokenizer.add_special_tokens({"additional_special_tokens": ["<KG_EMBEDDING>"]})
        self.special_kg_token = self.tokenizer.encode("<KG_EMBEDDING>", add_special_tokens=False)[0]
        
        
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
        
        return cls.load_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
    
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
        
        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels,
                RVQ_loss
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                graphs
            )
            
        if inputs_embeds is not None:
            inputs_embeds = inputs_embeds.to(dtype=self.llm.dtype)

        out = self.llm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        # If RVQ_loss is not None, add it to the output
        if return_dict:
            if hasattr(out, "loss"):
                out.loss = out.loss + RVQ_loss if RVQ_loss is not None else out.loss
        else:
            out = (out[0] + RVQ_loss,) + out[1:] if RVQ_loss is not None else out
            
        return out
        
    
AutoConfig.register(KG_LFMConfig.model_type, KG_LFMConfig)
AutoModel.register(KG_LFMConfig, KG_LFM)