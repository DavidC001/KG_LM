#!/usr/bin/env python3
"""
Comprehensive evaluation script for KG-LFM model.

This script loads the best trained model and evaluates it on various metrics
relevant for KG-augmented generation including:
- Perplexity
- BLEU scores
- ROUGE scores  
- Top-k accuracy for KG predictions
- Graph embedding coherence metrics
- Knowledge utilization metrics
"""

import logging
import json
import math
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from collections import defaultdict
import gc

import torch
from tqdm.auto import tqdm

# Project imports
from KG_LFM.configuration import load_yaml_config, ProjectConfig, IGNORE_INDEX
from KG_LFM.model.KG_LFM_arch import KG_LFM
from KG_LFM.utils.Dataloader import create_dataloader

from transformers import AutoTokenizer, AutoModelForCausalLM
from accelerate import Accelerator
from accelerate.utils import set_seed, broadcast_object_list

from copy import deepcopy

def align_logits_with_labels(logits: torch.Tensor, labels: torch.Tensor, num_quantizers: int, special_kg_token: int) -> torch.Tensor:
    """Align logits with labels by removing KG tokens from the GNN."""
    
    # if size already matches, return logits
    if logits.size(1) == labels.size(1):
        return logits
    
    pos_kg_token = torch.where(labels == special_kg_token)
    new_logits = torch.ones((labels.size(0), labels.size(1), logits.size(2)), dtype=logits.dtype, device=logits.device)
    
    # Remove special tokens from GNN
    for i in range(pos_kg_token[0].size(0)):
        batch_pos = pos_kg_token[0][i]
        new_logits[batch_pos] = torch.concat(
            [
                logits[batch_pos, :pos_kg_token[1][i]+1, :], 
                logits[batch_pos, pos_kg_token[1][i] + num_quantizers:, :],
            ], 
            dim=0
        )
        
    return new_logits

def compute_hit_k(
        logits: torch.Tensor, input_ids: torch.Tensor, k_values: List[int],
        object_boundaries: List[Tuple[int, int]], num_quantizers: int,
        attention_mask: torch.Tensor, special_token: int, tokenizer
    ):
    # Align logits with labels by removing KG tokens
    logits = align_logits_with_labels(logits, input_ids, num_quantizers, special_token)
    # Shift for causal LM: predict next token
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = input_ids[..., 1:].contiguous()

    logger = logging.getLogger()

    hit_k_correct = {k: 0 for k in k_values}
    total_objects = 0
    average_num_tokens = 0

    # Process each sample in the batch
    for sample_idx in range(input_ids.size(0)):
        sample_logits = shift_logits[sample_idx]  # (seq_len, vocab_size)
        sample_labels = shift_labels[sample_idx]  # (seq_len,)
        
        sample_boundaries = object_boundaries[sample_idx]
        # Get the positions of the object tokens, -1 is because of the shift
        object_positions = [i for i in range(sample_boundaries[0] - 1, sample_boundaries[1] - 1)]
        
        if not object_positions:
            logger.warning(f"No object tokens found for a sample in batch. Skipping.")
            continue
        
        total_objects += 1
        
        # Get logits for the object tokens
        object_logits = sample_logits[object_positions]  # (num_object_tokens, vocab_size)
        object_labels = sample_labels[object_positions]  # (num_object_tokens,)
        
        # for debugging print the correspondent tokens decoded
        if logger.isEnabledFor(logging.DEBUG):
            decoded_tokens = tokenizer.decode(object_labels)
            logger.debug(f"Decoded tokens for sample: {decoded_tokens}")
            # decoding from logits
            decoded_logits = tokenizer.decode(object_logits.argmax(dim=-1))
            logger.debug(f"Decoded logits for sample: {decoded_logits}")
        # Calculate average number of tokens of entire input sequence by summing all position of input_ids which are not padding
        average_num_tokens += (
            (attention_mask[sample_idx] == 1).sum().item() +
            torch.sum(input_ids[sample_idx] == special_token).item() * num_quantizers
        )
        # Compute ranks for each token
        token_ranks = []
        for logits, true_label in zip(object_logits, object_labels):
            # Get the rank of the true label in the sorted logits (descending order)
            sorted_indices = torch.argsort(logits, descending=True)
            rank = (sorted_indices == true_label).nonzero(as_tuple=True)[0].item() + 1  # +1 for 1-based rank
            token_ranks.append(rank)
        
        # Sequence rank is the maximum of all token ranks
        sequence_rank = max(token_ranks)
        
        # Check Hit@k for each k value
        for k in k_values:
            if sequence_rank <= k:
                hit_k_correct[k] += 1

    return hit_k_correct, average_num_tokens, total_objects

class KGLFMEvaluator:
    """Comprehensive evaluator for KG-LFM model."""
    
    def __init__(
        self,
        config_path: str,
        batch_size: int = 8,
        max_samples: Optional[int] = None
    ):
        self.config_path = config_path
        self.batch_size = batch_size
        self.max_samples = max_samples
        
        # Initialize accelerator
        self.accelerator = Accelerator()
        
        self.logger = logging.getLogger(__name__)
        
        # Load configuration
        self.config : ProjectConfig = load_yaml_config(config_path)
        self.model_path = Path(self.config.train_conf.start_from_checkpoint)
        self.config.train_conf.dataloader.batch_size = batch_size
        
        set_seed(self.config.seed)
        
        # Initialize model
        self.model = None
        self.tokenizer = None
        
        # Initialize metrics storage
        self.results = defaultdict(dict)
        
    def remove_kg_stuff(self, batch) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Remove KG-related tokens from input_ids and attention_mask."""
        
        batch_sentences = []
        for sample_idx in range(len(batch["input_ids"])):
            conversation = batch["conversations"][sample_idx]
            
            # remove the one with the special token
            kg_string = self.tokenizer.decode(self.model.special_kg_token)
            conversation = [turn for turn in conversation if kg_string not in turn['content']]

            # If tokenizer has a apply_chat_template method, use it
            try:
                sentence = self.clean_tokenizer.apply_chat_template(
                    conversation=conversation,
                    tokenize=False,
                    add_generation_prompt=False,
                    enable_thinking=False,
                )
            except ValueError:
                sentence = ""
                for turn in conversation:
                    sentence += turn['role'] + ": " + turn['content'] + "\n"
                sentence = sentence.strip()

            obj_start, obj_end = batch["objects"][sample_idx]["boundaries"]
            obj_str = batch["sentences"][sample_idx][obj_start:obj_end]

            new_obj_start = sentence.rfind(obj_str)
            new_obj_end = new_obj_start + len(obj_str)
            batch["objects"][sample_idx]["boundaries"] = (new_obj_start, new_obj_end)
            batch_sentences.append(sentence)
        
        tokenized = self.clean_tokenizer(
            batch_sentences,
            padding=True,
            return_tensors='pt',
        )
        batch["labels"] = torch.full(tokenized['input_ids'].shape, IGNORE_INDEX, dtype=tokenized['input_ids'].dtype)

        for sample_idx in range(len(batch["sentences"])):
            obj_start, obj_end = batch["objects"][sample_idx]["boundaries"]
            tok_start = tokenized.char_to_token(sample_idx, obj_start)
            tok_end = tokenized.char_to_token(sample_idx, obj_end)
            batch["objects"][sample_idx]["token_boundaries"] = (tok_start, tok_end)

            batch["labels"][sample_idx][tok_start:tok_end] = tokenized['input_ids'][sample_idx][tok_start:tok_end]

        out = {
            "sentences": batch_sentences,
            "objects": batch["objects"],
            "input_ids": tokenized['input_ids'],
            "attention_mask": tokenized['attention_mask'],
            "graphs": batch["graphs"]
        }
        
        return out

    def kg_textualization(self, batch) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert KG graphs to textual format for model input."""
        # Convert graphs to textual representation
        
        # deep copy to avoid modifying original batch
        batch = deepcopy(batch)
        
        for sample_idx in range(len(batch["sentences"])):
            graph = batch["graphs"][sample_idx]
            
            graph_text = " Information from the knowledge graph: "
            
            central_node_label = graph["central_node_label"]
            neighbors = graph["neighbour_node_labels"]
            edges = graph["edge_labels"]
            
            for neighbor, edge in zip(neighbors, edges):
                # Create a textual representation of the graph
                graph_text += f"{central_node_label} {edge} {neighbor}. "

            graph_text += "\n"

            # in the turn with SPECIAL_KG_TOKEN substitute it with <SPECIAL_KG_TOKEN>
            special_tok = self.tokenizer.decode(self.model.special_kg_token)
            num_turns = len(batch["conversations"][sample_idx])
            batch["conversations"][sample_idx] = [{
                "role": batch["conversations"][sample_idx][i]["role"],
                "content": batch["conversations"][sample_idx][i]["content"].replace(special_tok, graph_text)
            } for i in range(num_turns)]

        return self.remove_kg_stuff(batch)
        
    def llm_no_lora(self, **kwargs):
        """Return the model output without LoRA layers."""
        with self.model.llm.disable_adapter():
            results = self.model(**kwargs)
            
        return results

    def load_model(self):
        """Load the best trained model."""
        if self.accelerator.is_main_process:
            self.logger.info(f"Loading model from {self.model_path}")
        
        try:
            self.model = KG_LFM.from_pretrained(self.model_path)
            self.model.eval()
            self.tokenizer = self.model.tokenizer
            
            # if the config requires to tune the model also load clean model
            if self.model.config.tune_language_model:
                self.clean_model = AutoModelForCausalLM.from_pretrained(
                    self.model.config.llm_model_name,
                )
                self.clean_tokenizer = AutoTokenizer.from_pretrained(
                    self.model.config.llm_model_name,
                )
                self.clean_model.eval()
            elif self.model.config.use_lora:
                self.clean_model = self.llm_no_lora
                self.clean_tokenizer = self.tokenizer
            else:
                self.clean_model = self.model
                self.clean_tokenizer = self.tokenizer
            
            if self.accelerator.is_main_process:
                self.logger.info("Model loaded successfully")
                self.logger.info("Clean model loaded successfully")
        except Exception as e:
            if self.accelerator.is_main_process:
                self.logger.error(f"Error loading model: {e}")
            raise
        
        self.tests = {
            "original_LLM": (self.remove_kg_stuff, self.clean_model),
            "textualization": (self.kg_textualization, self.clean_model),
            "KG_LFM": (lambda x: x, self.model),  # No preprocessing needed for KG_LFM
        }
    
    def setup_data(self, split: str = "test"):
        """Setup data loaders for evaluation."""
        if self.accelerator.is_main_process:
            self.logger.info(f"Setting up {split} data loader")
        
        self.dataloader = create_dataloader(
            self.config.dataset,
            self.config.train_conf.dataloader,
            tokenizer=self.tokenizer,
            split=split
        )
        
        if self.accelerator.is_main_process:
            self.logger.info(f"Data loader setup complete. {len(self.dataloader)} batches available.")
    
    def prepare_accelerator(self):
        """Prepare the accelerator for distributed training."""
        if self.accelerator.is_main_process:
            self.logger.info("Preparing accelerator...")
        
        # Prepare model and dataloader
        if self.model.config.tune_language_model:
            self.model, self.clean_model, self.dataloader = self.accelerator.prepare(
                self.model, self.clean_model, self.dataloader
            )
        else:
            self.model, self.dataloader = self.accelerator.prepare(
                self.model, self.dataloader
            )
            self.clean_model = self.model  # Use the same model if not tuning

    def compute_hit_k_metrics(self, k_values: List[int] = [1, 3, 5, 10]) -> Dict[str, float]:
        """Compute Hit@k metrics for object label prediction.
        
        To quantify recall, we adopt the widely used Hit@k metric. For an object label 
        split into T tokens, we record the rank (r_t) of each token t in the model's 
        output logits. The sequence rank is taken as r = max{r_1,...,r_T}, and it counts 
        as a "hit" if r ≤ k (i.e., all tokens appear in the top-k predictions at their 
        respective timesteps). This approach is robust to multi-token entities, a common 
        challenge in IR tasks involving named entities ("New York Times" vs. "NYT").
        """
        if self.accelerator.is_main_process:
            self.logger.info(f"Computing Hit@k metrics for k={k_values}...")
        
        results = {}
        
        for name, (preprocess_func, model) in self.tests.items():
            if self.accelerator.is_main_process:
                self.logger.info(f"Evaluating {name} model for Hit@k metrics...")
            
            hit_k_correct = {k: 0 for k in k_values}
            total_objects = 0
            average_num_tokens = 0

            with torch.no_grad():
                for batch_idx, batch in enumerate(tqdm(self.dataloader, desc=f"Computing Hit@k metrics for {name}", disable=not self.accelerator.is_main_process)):
                    if self.max_samples and (batch_idx * self.batch_size * self.accelerator.num_processes) >= self.max_samples:
                        break

                    batch = preprocess_func(batch)
                    # No need to move to device manually - accelerator handles this

                    object_boundaries = [obj["token_boundaries"] for obj in batch['objects']]

                    batch['input_ids'] = batch['input_ids'].to(self.accelerator.device)
                    batch['attention_mask'] = batch['attention_mask'].to(self.accelerator.device)
                    batch['labels'] = batch['input_ids'].to(self.accelerator.device)
                    
                    model_input = {
                        'input_ids': batch['input_ids'],
                        'attention_mask': batch['attention_mask'],
                    }
                    if batch['graphs']: model_input['graphs'] = batch['graphs']

                    outputs = model(**model_input)
                    # Get logits and labels
                    logits = outputs.logits
                    input_ids = batch['input_ids']

                    hit_k_correct, batch_avg_num_tokens, new_objects = compute_hit_k(
                        logits, input_ids, k_values,
                        object_boundaries, self.model.config.num_quantizers,
                        batch['attention_mask'],
                        special_token=self.model.special_kg_token, tokenizer=self.tokenizer
                    )

                    average_num_tokens += batch_avg_num_tokens
                    total_objects += new_objects

                    for k in k_values:
                        hit_k_correct[k] += hit_k_correct[k]

            # Gather hit counts and total objects from all processes
            hit_k_tensors = {}
            for k in k_values:
                hit_k_tensor = torch.tensor(hit_k_correct[k], device=self.accelerator.device)
                hit_k_tensors[k] = self.accelerator.gather(hit_k_tensor).sum().item()
            
            total_objects_tensor = torch.tensor(total_objects, device=self.accelerator.device)
            total_objects_gathered = self.accelerator.gather(total_objects_tensor).sum().item()

            average_num_tokens_gathered = self.accelerator.gather(torch.tensor(average_num_tokens, device=self.accelerator.device)).sum().item()

            # Synchronize across processes
            self.accelerator.wait_for_everyone()
            
            # Compute Hit@k metrics on main process
            metrics = {}
            if self.accelerator.is_main_process:
                if total_objects_gathered > 0:
                    average_num_tokens_gathered /= total_objects_gathered
                    metrics['average_num_tokens'] = average_num_tokens_gathered
                    
                    for k in k_values:
                        metrics[f'hit_at_{k}'] = hit_k_tensors[k] / total_objects_gathered
                    
                    self.logger.info(f"Hit@k computed on {total_objects_gathered} objects with average {average_num_tokens_gathered:.2f} tokens per object.")
                    for k in k_values:
                        self.logger.info(f"Hit@{k}: {metrics[f'hit_at_{k}']:.4f}")
                else:
                    self.logger.warning("No valid objects found for Hit@k computation")
                    for k in k_values:
                        metrics[f'hit_at_{k}'] = 0.0
            else:
                # Non-main processes set dummy values
                for k in k_values:
                    metrics[f'hit_at_{k}'] = 0.0
                    
            # Store results for this preprocessing method
            results[name] = metrics

        # Broadcast results to all processes
        if self.accelerator.is_main_process:
            results_to_broadcast = results
        else:
            results_to_broadcast = None
        
        results = broadcast_object_list([results_to_broadcast])[0]
        
        return results

    def evaluate(self, output_file: Optional[str] = None) -> Dict[str, Any]:
        """Run all evaluation metrics and return comprehensive results."""
        if self.accelerator.is_main_process:
            self.logger.info("Starting comprehensive evaluation...")
        
        # Load model and setup data
        self.load_model()
        self.setup_data()
        self.prepare_accelerator()
        
        # Run all evaluations
        evaluations = {
            'hit_at_k': self.compute_hit_k_metrics,
        }
        
        # Run evaluations
        for eval_name, eval_func in evaluations.items():
            if self.accelerator.is_main_process:
                self.logger.info(f"Running {eval_name}...")
            self.results[eval_name] = eval_func()
            # Clear GPU memory after each evaluation
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            
        # Add metadata (only on main process)
        if self.accelerator.is_main_process:
            self.results['metadata'] = {
                'model_path': str(self.model_path),
                'config_path': self.config_path,
                'batch_size': self.batch_size,
                'max_samples': self.max_samples,
                'num_processes': self.accelerator.num_processes,
                'process_index': self.accelerator.process_index,
            }
        
        # Save results if output file specified (only on main process)
        if output_file and self.accelerator.is_main_process:
            self.save_results(output_file)
        
        return dict(self.results)
    
    def save_results(self, output_file: str):
        """Save evaluation results to a JSON file."""
        if not self.accelerator.is_main_process:
            return
            
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        self.logger.info(f"Results saved to {output_path}")
    
    def print_summary(self):
        """Print a summary of evaluation results."""
        if not self.accelerator.is_main_process:
            return
            
        print("\n" + "="*80)
        print("KG-LFM EVALUATION SUMMARY")
        print("="*80)
        
        for category, metrics in self.results.items():
            if category == 'metadata':
                continue
            
            print(f"\n{category.upper().replace('_', ' ')}:")
            print("-" * 40)
            
            if isinstance(metrics, dict) and 'error' not in metrics:
                for metric_name, value in metrics.items():
                    if isinstance(value, float):
                        print(f"  {metric_name}: {value:.4f}")
                    else:
                        print(f"  {metric_name}: {value}")
            elif 'error' in metrics:
                print(f"  Error: {metrics['error']}")
        
        print("\n" + "="*80)