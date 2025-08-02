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
from KG_LFM.configuration import load_yaml_config, ProjectConfig
from KG_LFM.model.KG_LFM_arch import KG_LFM
from KG_LFM.utils.Dataloaders.pretrain_data import create_dataloader

from transformers import AutoTokenizer, AutoModelForCausalLM
from accelerate import Accelerator
from accelerate.utils import set_seed, broadcast_object_list

from copy import deepcopy

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
        self.model_path = Path(self.config.train_conf.checkpoint_dir) / self.config.train_conf.run_name / "best_model"
        self.config.train_conf.dataloader.batch_size = batch_size
        
        set_seed(self.config.seed)
        
        # Initialize model
        self.model = None
        self.tokenizer = None
        
        # Initialize metrics storage
        self.results = defaultdict(dict)
        
    def remove_kg_stuff(self, batch) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Remove KG-related tokens from input_ids and attention_mask."""
        
        sentences = []
        batch_sentences = []
        for sample_idx in range(len(batch["input_ids"])):
            sentence : str = batch["sentences"][sample_idx]
            
            # remove special KG token if present
            kg_token = self.tokenizer.decode(self.model.special_kg_token, add_special_tokens=False)
            while kg_token in sentence:
                sentence = sentence.replace(kg_token, "")
                # move the boundaries of the object
                batch["objects"][sample_idx]["boundaries"] = (
                    batch["objects"][sample_idx]["boundaries"][0] - len(kg_token),
                    batch["objects"][sample_idx]["boundaries"][1] - len(kg_token)
                )
            
            sentences.append(sentence)
            
            # if tokenizer has a apply_chat_template method, use it
            if hasattr(self.clean_tokenizer, 'apply_chat_template'):
                sentence = self.clean_tokenizer.apply_chat_template(
                conversation=[
                    {
                        'role': 'assistant',
                        'content': sentence
                    }
                ],
                tokenize=False,
                add_generation_prompt=False,
                enable_thinking=False,
            )
            
            batch_sentences.append(sentence)
        
        tokenized = self.clean_tokenizer(
            batch_sentences,
            padding=True,
            return_tensors='pt',
        )
        
        out = {
            "sentences": sentences,
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
            sentence: str = batch["sentences"][sample_idx]
            
            graph = batch["graphs"][sample_idx]
            
            graph_text = "Information from the knowledge graph: "
            
            central_node_label = graph["central_node_label"]
            neighbors = graph["neighbour_node_labels"]
            edges = graph["edge_labels"]
            
            for neighbor, edge in zip(neighbors, edges):
                # Create a textual representation of the graph
                graph_text += f"{central_node_label} {edge} {neighbor}. "

            graph_text += "\n"

            len_graph = len(graph_text)
            batch["sentences"][sample_idx] = graph_text+sentence
            batch["objects"][sample_idx]["boundaries"] = (
                batch["objects"][sample_idx]["boundaries"][0] + len_graph,
                batch["objects"][sample_idx]["boundaries"][1] + len_graph
            )
            
        return self.remove_kg_stuff(batch)
        
    def load_model(self):
        """Load the best trained model."""
        if self.accelerator.is_main_process:
            self.logger.info(f"Loading model from {self.model_path}")
        
        try:
            self.model = KG_LFM.from_pretrained(self.model_path)
            self.model.eval()
            self.tokenizer = self.model.tokenizer
            
            # if the config requires to tune the model also load clean model
            if self.config.model.tune_language_model:
                self.clean_model = AutoModelForCausalLM.from_pretrained(
                    self.config.model.llm_model_name,
                    cache_dir=self.config.train_conf.cache_dir
                )
                self.clean_tokenizer = AutoTokenizer.from_pretrained(
                    self.config.model.llm_model_name,
                    cache_dir=self.config.train_conf.cache_dir
                )
                self.clean_model.eval()
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
        if self.config.model.tune_language_model:
            self.model, self.clean_model, self.dataloader = self.accelerator.prepare(
                self.model, self.clean_model, self.dataloader
            )
        else:
            self.model, self.dataloader = self.accelerator.prepare(
                self.model, self.dataloader
            )
            self.clean_model = self.model  # Use the same model if not tuning
            
    def compute_perplexity(self, ) -> float:
        """Compute perplexity on the test set."""
        if self.accelerator.is_main_process:
            self.logger.info("Computing perplexity...")
        
        results = {}
        
        for name, (preprocess_func, model) in self.tests.items():
            if self.accelerator.is_main_process:
                self.logger.info(f"Evaluating {name} model...")
            
            total_loss = 0.0
            total_tokens = 0
            
            with torch.no_grad():
                for batch_idx, batch in enumerate(tqdm(self.dataloader, desc="Computing perplexity", disable=not self.accelerator.is_main_process)):
                    if self.max_samples and (batch_idx * self.batch_size * self.accelerator.num_processes) >= self.max_samples:
                        break
                    
                    batch = preprocess_func(batch)
                    
                    batch['input_ids'] = batch['input_ids'].to(self.accelerator.device)
                    batch['attention_mask'] = batch['attention_mask'].to(self.accelerator.device)
                    batch['labels'] = batch['input_ids'].to(self.accelerator.device)
                    
                    # No need to move to device manually - accelerator handles this
                    model_input = {
                        'input_ids': batch['input_ids'],
                        'attention_mask': batch['attention_mask'],
                        "labels": batch['input_ids'],
                    }
                    if batch['graphs']: model_input['graphs'] = batch['graphs']

                    outputs = model(**model_input)

                    # Extract loss and count valid tokens
                    loss = outputs.loss
                    if loss is not None:
                        # Count valid tokens (non-padded)
                        valid_tokens = (batch['attention_mask'] == 1).sum().item() 
                        
                        # remove special token ignored during loss computation
                        valid_tokens -= torch.sum(batch['input_ids'] == self.model.special_kg_token).item()
                        
                        
                        # Gather losses and token counts from all processes
                        loss_gathered = self.accelerator.gather(loss.unsqueeze(0))
                        valid_tokens_tensor = torch.tensor(valid_tokens, device=self.accelerator.device)
                        tokens_gathered = self.accelerator.gather(valid_tokens_tensor.unsqueeze(0))
                        
                        if self.accelerator.is_main_process:
                            total_loss += (loss_gathered.sum() * tokens_gathered.sum()).item()
                            total_tokens += tokens_gathered.sum().item()
            
            # Synchronize across processes
            self.accelerator.wait_for_everyone()
            
            if self.accelerator.is_main_process:
                if total_tokens > 0:
                    avg_loss = total_loss / total_tokens
                    perplexity = math.exp(avg_loss)
                    self.logger.info(f"Perplexity: {perplexity:.4f}")
                    results[name] = perplexity
                else:
                    self.logger.warning("No valid tokens found for perplexity computation")
                    results[name] = float('inf')
            else:
                # Non-main processes set dummy values
                results[name] = 0.0

        # Broadcast results to all processes
        if self.accelerator.is_main_process:
            results_to_broadcast = results
        else:
            results_to_broadcast = None
        
        results = broadcast_object_list([results_to_broadcast])[0]
        
        # Return the results for all preprocessing methods
        return results

    def _align_logits_with_labels(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Align logits with labels by removing KG tokens from the GNN."""
        
        # if size already matches, return logits
        if logits.size(1) == labels.size(1):
            return logits
        
        pos_kg_token = torch.where(labels == self.model.special_kg_token)
        new_logits = torch.ones((labels.size(0), labels.size(1), logits.size(2)), dtype=logits.dtype, device=logits.device)
        
        # Remove special tokens from GNN
        for i in range(pos_kg_token[0].size(0)):
            batch_pos = pos_kg_token[0][i]
            new_logits[batch_pos] = torch.concat(
                [
                    logits[batch_pos, :pos_kg_token[1][i]+1, :], 
                    logits[batch_pos, pos_kg_token[1][i] + self.config.model.num_quantizers:, :],
                ], 
                dim=0
            )
            
        return new_logits

    def _obj_token_positions(self, labels:torch.Tensor, sentence: str, object_boundaries: List[int]) -> List[int]:
        """Get token positions of the object in the sentence."""
        obj_start, obj_end = object_boundaries
        # Tokenize the object text to get its tokens
        object_text = sentence[obj_start-1:obj_end] # keep the space before the object
        object_tokens_num = len(self.tokenizer.encode(object_text, add_special_tokens=False))
        
        # Find the position of object tokens in the input sequence
        tokens = self.tokenizer.encode(sentence[:obj_end], add_special_tokens=False, return_tensors='pt')[0].to(self.accelerator.device)
        sentence_tokens = len(tokens)
        
        # Finding start index of the sentence in the tokenized sequence (reversed to avoid matching the textualization of the KG)
        end_index = next((i for i in range(len(labels), sentence_tokens-1, -1) if (labels[i - sentence_tokens:i] == tokens).all()), -1) 
        
        if end_index == -1:
            self.logger.warning(f"Sentence tokens not found in sequence: {sentence}. Skipping.")
            return []
        
        obj_end = end_index
        obj_start = obj_end - object_tokens_num
        
        # Object token positions in the sequence
        object_positions = [i for i in range(obj_start, obj_end)]
        
        return object_positions
        

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
                for batch_idx, batch in enumerate(tqdm(self.dataloader, desc="Computing Hit@k metrics", disable=not self.accelerator.is_main_process)):
                    if self.max_samples and (batch_idx * self.batch_size * self.accelerator.num_processes) >= self.max_samples:
                        break

                    batch = preprocess_func(batch)
                    # No need to move to device manually - accelerator handles this

                    object_boundaries = [obj["boundaries"] for obj in batch['objects']]
                    
                    batch['input_ids'] = batch['input_ids'].to(self.accelerator.device)
                    batch['attention_mask'] = batch['attention_mask'].to(self.accelerator.device)
                    batch['labels'] = batch['input_ids'].to(self.accelerator.device)
                    
                    model_input = {
                        'input_ids': batch['input_ids'],
                        'attention_mask': batch['attention_mask'],
                        "labels": batch['input_ids'],
                    }
                    if batch['graphs']: model_input['graphs'] = batch['graphs']

                    outputs = model(**model_input)
                    # Get logits and labels
                    logits = outputs.logits
                    labels = batch['input_ids']
                    
                    # Align logits with labels by removing KG tokens
                    logits = self._align_logits_with_labels(logits, labels)
                    
                    # Shift for causal LM: predict next token
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = labels[..., 1:].contiguous()
                    
                    # Process each sample in the batch
                    for sample_idx in range(labels.size(0)):
                        sample_logits = shift_logits[sample_idx]  # (seq_len, vocab_size)
                        sample_labels = shift_labels[sample_idx]  # (seq_len,)
                        
                        sentence = batch["sentences"][sample_idx]  # Original sentence
                        
                        sample_boundaries = object_boundaries[sample_idx]
                        
                        object_positions = self._obj_token_positions(
                            sample_labels, 
                            sentence, 
                            sample_boundaries
                        )
                        
                        if not object_positions:
                            if self.accelerator.is_main_process:
                                self.logger.warning(f"No object tokens found for sample {sample_idx} in batch {batch_idx}. Skipping.")
                            continue
                        
                        total_objects += 1
                        
                        # Get logits for the object tokens
                        object_logits = sample_logits[object_positions]  # (num_object_tokens, vocab_size)
                        object_labels = sample_labels[object_positions]  # (num_object_tokens,)
                        
                        # Calculate average number of tokens of entire input sequence by summing all position of input_ids which are not padding
                        average_num_tokens += (
                            (batch['attention_mask'][sample_idx] == 1).sum().item() +
                            torch.sum(batch['input_ids'][sample_idx] == self.model.special_kg_token).item() * self.config.model.num_quantizers
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
            'perplexity': self.compute_perplexity,
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