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

import argparse
import logging
import os
import json
import math
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from collections import defaultdict
import gc

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import numpy as np

# For text evaluation metrics
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import nltk
nltk.download('punkt', quiet=True)


# For KG-specific metrics
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import spearmanr, pearsonr

# Project imports
from KG_LFM.configuration import load_yaml_config, ProjectConfig
from KG_LFM.model.KG_LFM_arch import KG_LFM
from KG_LFM.utils.Dataloaders.pretrain_data import create_dataloader


class KGLFMEvaluator:
    """Comprehensive evaluator for KG-LFM model."""
    
    def __init__(
        self,
        config_path: str,
        device: str = "cuda",
        batch_size: int = 8,
        max_samples: Optional[int] = None
    ):
        self.config_path = config_path
        self.device = device
        self.batch_size = batch_size
        self.max_samples = max_samples
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Load configuration
        self.config = load_yaml_config(config_path)
        self.model_path = Path(self.config.train_conf.checkpoint_dir) / self.config.train_conf.run_name / "best_model"
        
        # Initialize model
        self.model = None
        self.tokenizer = None
        
        # Initialize metrics storage
        self.results = defaultdict(dict)
        
        # Setup text evaluation metrics
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        self.smoothing = SmoothingFunction().method1
        
    def load_model(self):
        """Load the best trained model."""
        self.logger.info(f"Loading model from {self.model_path}")
        
        try:
            self.model = KG_LFM.from_pretrained(self.model_path)
            self.model.to(self.device)
            self.model.eval()
            self.tokenizer = self.model.tokenizer
            self.logger.info("Model loaded successfully")
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            raise
    
    def setup_data(self, split: str = "test"):
        """Setup data loaders for evaluation."""
        self.logger.info(f"Setting up {split} data loader")
        
        self.dataloader = create_dataloader(
            self.config.dataset,
            self.config.train_conf.dataloader,
            tokenizer=self.tokenizer,
            split=split
        )
        
        self.logger.info(f"Data loader setup complete. {len(self.dataloader)} batches available.")
    
    def compute_perplexity(self) -> float:
        """Compute perplexity on the test set."""
        self.logger.info("Computing perplexity...")
        
        total_loss = 0.0
        total_tokens = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(self.dataloader, desc="Computing perplexity")):
                if self.max_samples and batch_idx * self.batch_size >= self.max_samples:
                    break
                
                try:
                    # Move batch to device
                    batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                            for k, v in batch.items()}

                    outputs = self.model(
                        input_ids=batch['input_ids'],
                        graphs=batch['graphs'],
                        attention_mask=batch['attention_mask'],
                        labels=batch['input_ids'],
                        use_cache=False,
                    )
                    
                    # Extract loss and count valid tokens
                    loss = outputs.loss
                    
                    if loss is not None and not torch.isnan(loss):
                        # Count valid tokens (non-padded)
                        valid_tokens = (batch['attention_mask'] == 1).sum().item()
                        total_loss += loss.item() * valid_tokens
                        total_tokens += valid_tokens
                        num_batches += 1
                
                except Exception as e:
                    self.logger.warning(f"Error in batch {batch_idx}: {e}")
                    continue
        
        if total_tokens > 0:
            avg_loss = total_loss / total_tokens
            perplexity = math.exp(avg_loss)
            self.logger.info(f"Perplexity: {perplexity:.4f}")
            return perplexity
        else:
            self.logger.warning("No valid tokens found for perplexity computation")
            return float('inf')
    
    def compute_generation_metrics(self, num_samples: int = 100) -> Dict[str, float]:
        """Compute text generation metrics (BLEU, ROUGE)."""
        self.logger.info(f"Computing generation metrics on {num_samples} samples...")
        
        bleu_scores = []
        rouge_scores = defaultdict(list)
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(self.dataloader, desc="Computing generation metrics")):
                if batch_idx >= num_samples // self.batch_size:
                    break
                
                try:
                    # Move batch to device
                    batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                            for k, v in batch.items()}
                    
                    # Use first part of sequence as prompt, rest as target
                    input_ids = batch['input_ids']
                    attention_mask = batch['attention_mask']
                    
                    # Split sequence: first 70% as prompt, rest as target
                    seq_len = input_ids.size(1)
                    prompt_len = int(seq_len * 0.7)
                    
                    prompt_ids = input_ids[:, :prompt_len]
                    target_ids = input_ids[:, prompt_len:]
                    prompt_attention = attention_mask[:, :prompt_len]
                    
                    # Generate continuation
                    generated = self.model.generate(
                        input_ids=prompt_ids,
                        graphs=batch['graphs'],
                        attention_mask=prompt_attention,
                        max_new_tokens=target_ids.size(1),
                        do_sample=False,  # Greedy decoding for reproducibility
                        pad_token_id=self.tokenizer.eos_token_id,
                    )
                    
                    # Extract only the generated part
                    generated_new = generated[:, prompt_len:]
                    
                    # Decode texts
                    for i in range(input_ids.size(0)):
                        try:
                            target_text = self.tokenizer.decode(target_ids[i], skip_special_tokens=True).strip()
                            generated_text = self.tokenizer.decode(generated_new[i], skip_special_tokens=True).strip()
                            
                            if target_text and generated_text:
                                # BLEU score
                                reference = [target_text.split()]
                                candidate = generated_text.split()
                                bleu = sentence_bleu(reference, candidate, smoothing_function=self.smoothing)
                                bleu_scores.append(bleu)
                                
                                # ROUGE scores
                                rouge_scores_sample = self.rouge_scorer.score(target_text, generated_text)
                                for metric, score in rouge_scores_sample.items():
                                    rouge_scores[metric].append(score.fmeasure)
                        
                        except Exception as e:
                            self.logger.warning(f"Error processing sample {i} in batch {batch_idx}: {e}")
                            continue
                
                except Exception as e:
                    self.logger.warning(f"Error in generation batch {batch_idx}: {e}")
                    continue
        
        # Compute averages
        metrics = {}
        if bleu_scores:
            metrics['bleu'] = np.mean(bleu_scores)
            metrics['bleu_std'] = np.std(bleu_scores)
        
        for metric, scores in rouge_scores.items():
            if scores:
                metrics[f'{metric}_f'] = np.mean(scores)
                metrics[f'{metric}_f_std'] = np.std(scores)
        
        self.logger.info(f"Generation metrics computed on {len(bleu_scores)} samples")
        return metrics
    
    def compute_kg_embedding_metrics(self) -> Dict[str, float]:
        """Compute KG-specific metrics related to graph embeddings."""
        self.logger.info("Computing KG embedding metrics...")
        
        graph_embeddings = []
        rq_losses = []
        graph_similarities = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(self.dataloader, desc="Computing KG metrics")):
                if self.max_samples and batch_idx * self.batch_size >= self.max_samples:
                    break
                
                try:
                    # Move graphs to device
                    graphs = batch['graphs'].to(self.device)
                    
                    # Extract graph embeddings and quantization info
                    graph_features, indices, rvq_loss = self.model.encode_graphs(graphs)
                    
                    if rvq_loss is not None and not torch.isnan(rvq_loss):
                        rq_losses.append(rvq_loss.item())
                    
                    # Store embeddings for similarity analysis
                    graph_embeddings.append(graph_features.cpu().numpy())
                    
                    # Compute pairwise similarities within batch
                    if graph_features.size(0) > 1:
                        similarities = F.cosine_similarity(
                            graph_features.unsqueeze(1), 
                            graph_features.unsqueeze(0), 
                            dim=-1
                        )
                        # Get upper triangular part (excluding diagonal)
                        mask = torch.triu(torch.ones_like(similarities, dtype=torch.bool), diagonal=1)
                        batch_similarities = similarities[mask].cpu().numpy()
                        graph_similarities.extend(batch_similarities)
                
                except Exception as e:
                    self.logger.warning(f"Error in KG metrics batch {batch_idx}: {e}")
                    continue
        
        # Compute metrics
        metrics = {}
        
        if rq_losses:
            metrics['avg_rvq_loss'] = np.mean(rq_losses)
            metrics['std_rvq_loss'] = np.std(rq_losses)
        
        if graph_similarities:
            metrics['avg_graph_similarity'] = np.mean(graph_similarities)
            metrics['std_graph_similarity'] = np.std(graph_similarities)
            metrics['min_graph_similarity'] = np.min(graph_similarities)
            metrics['max_graph_similarity'] = np.max(graph_similarities)
        
        # Global embedding statistics
        if graph_embeddings:
            all_embeddings = np.concatenate(graph_embeddings, axis=0)
            metrics['embedding_dim'] = all_embeddings.shape[-1]
            metrics['avg_embedding_norm'] = np.mean(np.linalg.norm(all_embeddings, axis=-1))
            metrics['std_embedding_norm'] = np.std(np.linalg.norm(all_embeddings, axis=-1))
            
            # Embedding diversity (variance across dimensions)
            metrics['embedding_variance'] = np.mean(np.var(all_embeddings, axis=0))
        
        return metrics
    
    def compute_hit_k_metrics(self, k_values: List[int] = [1, 3, 5, 10]) -> Dict[str, float]:
        """Compute Hit@k metrics for object label prediction.
        
        To quantify recall, we adopt the widely used Hit@k metric. For an object label 
        split into T tokens, we record the rank (r_t) of each token t in the model's 
        output logits. The sequence rank is taken as r = max{r_1,...,r_T}, and it counts 
        as a "hit" if r ≤ k (i.e., all tokens appear in the top-k predictions at their 
        respective timesteps). This approach is robust to multi-token entities, a common 
        challenge in IR tasks involving named entities ("New York Times" vs. "NYT").
        """
        self.logger.info(f"Computing Hit@k metrics for k={k_values}...")
        
        hit_k_correct = {k: 0 for k in k_values}
        total_objects = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(self.dataloader, desc="Computing Hit@k metrics")):
                if self.max_samples and batch_idx * self.batch_size >= self.max_samples:
                    break
                
                try:
                    # Move batch to device
                    batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                            for k, v in batch.items()}
                    
                    # shift from adding " special_kg_token "
                    special_kg_token_len = len(self.tokenizer.decode(self.model.special_kg_token, add_special_tokens=False))+1
                    # Retrieve character boundaries for the objects in the batch
                    object_boundaries = [
                        (obj["boundaries"][0] + special_kg_token_len, obj["boundaries"][1] + special_kg_token_len)
                        for obj in batch['objects']
                    ]

                    outputs = self.model(
                        input_ids=batch['input_ids'],
                        graphs=batch['graphs'],
                        attention_mask=batch['attention_mask'],
                        use_cache=False,
                    )
                    
                    logits = outputs.logits
                    labels = batch['input_ids']
                    
                    # Find special KG tokens and remove them from logits
                    pos_kg = torch.where(labels == self.model.special_kg_token)
                    
                    attention_mask = batch['attention_mask']
                    new_logits = torch.ones((labels.size(0), labels.size(1), logits.size(2)), dtype=logits.dtype, device=logits.device)
                    # Remove special KG tokens from GNN
                    for i in range(pos_kg[0].size(0)):
                        batch_pos = pos_kg[0][i]
                        new_logits[batch_pos] = torch.concat([logits[batch_pos, :pos_kg[1][i]+1:, :], logits[batch_pos, pos_kg[1][i] + self.config.model.num_quantizers:, :]], dim=0)
                    logits = new_logits
                    
                    # Shift for causal LM: predict next token
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = labels[..., 1:].contiguous()
                    attention_mask = attention_mask[..., 1:]
                    
                    # Process each sample in the batch
                    for sample_idx in range(labels.size(0)):
                        sample_logits = shift_logits[sample_idx]  # (seq_len, vocab_size)
                        sample_labels = shift_labels[sample_idx]  # (seq_len,)
                        sample_attention = attention_mask[sample_idx]  # (seq_len,)
                        
                        sentence = batch["sentences"][sample_idx]  # Original sentence
                        
                        sample_boundaries = object_boundaries[sample_idx]
                        obj_start = sample_boundaries[0] # character start index
                        obj_end = sample_boundaries[1] # character end index
                        
                        # Extract the object substring from the sentence
                        object_text = sentence[obj_start:obj_end]
                        
                        # Tokenize the object text to get its tokens
                        object_tokens_num = len(self.tokenizer.encode(object_text, add_special_tokens=False))

                        # Find the position of object tokens in the input sequence
                        # We need to map character positions to token positions
                        prefix_text = sentence[:obj_start-1] # remove the space that gets tokenized otherwise
                        prefix_tokens = self.tokenizer.encode(prefix_text, add_special_tokens=False, return_tensors='pt')[0].to(sample_labels.device)
                        len_prefix_tokens = len(prefix_tokens)
                        
                        # Finding first occurrence using next()  
                        index = next((i for i in range(len(sample_labels) - len_prefix_tokens + 1) if (sample_labels[i:i + len_prefix_tokens] == prefix_tokens).all()), -1)
                        if index == -1:
                            self.logger.warning(f"Prefix tokens not found in sample {sample_idx} of batch {batch_idx}. Skipping.")
                            continue
                        end_object_index = index + len_prefix_tokens
                        
                        object_positions = [i for i in range(end_object_index, end_object_index + object_tokens_num)]
                        
                        total_objects += 1
                        
                        # Get logits for the object tokens
                        object_logits = sample_logits[object_positions]  # (num_object_tokens, vocab_size)
                        object_labels = sample_labels[object_positions]  # (num_object_tokens,)
                        
                        # Compute ranks for each token
                        token_ranks = []
                        for token_idx, (logits, true_label) in enumerate(zip(object_logits, object_labels)):
                            # Get the rank of the true label in the sorted logits (descending order)
                            sorted_indices = torch.argsort(logits, descending=True)
                            rank = (sorted_indices == true_label).nonzero(as_tuple=True)[0].item() + 1  # 1-indexed rank
                            token_ranks.append(rank)
                        
                        # Sequence rank is the maximum of all token ranks
                        sequence_rank = max(token_ranks)
                        
                        # Check Hit@k for each k value
                        for k in k_values:
                            if sequence_rank <= k:
                                hit_k_correct[k] += 1
                        
                except Exception as e:
                    self.logger.warning(f"Error in Hit@k batch {batch_idx}: {e}")
                    continue
        
        # Compute Hit@k metrics
        metrics = {}
        if total_objects > 0:
            for k in k_values:
                metrics[f'hit_at_{k}'] = hit_k_correct[k] / total_objects
            
            self.logger.info(f"Hit@k computed on {total_objects} objects")
            for k in k_values:
                self.logger.info(f"Hit@{k}: {metrics[f'hit_at_{k}']:.4f}")
        else:
            self.logger.warning("No valid objects found for Hit@k computation")
            for k in k_values:
                metrics[f'hit_at_{k}'] = 0.0
        
        return metrics
    
    def compute_knowledge_utilization_metrics(self) -> Dict[str, float]:
        """Compute metrics related to how well the model utilizes KG information."""
        self.logger.info("Computing knowledge utilization metrics...")
        
        # Compare performance with and without KG information
        with_kg_losses = []
        without_kg_losses = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(self.dataloader, desc="Computing knowledge utilization")):
                if self.max_samples and batch_idx * self.batch_size >= self.max_samples:
                    break
                
                try:
                    # Move batch to device
                    batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                            for k, v in batch.items()}
                    
                    # Performance with KG
                    outputs_with_kg = self.model(
                        input_ids=batch['input_ids'],
                        graphs=batch['graphs'],
                        attention_mask=batch['attention_mask'],
                        labels=batch['input_ids'],
                        use_cache=False,
                    )
                    
                    if outputs_with_kg.loss is not None:
                        with_kg_losses.append(outputs_with_kg.loss.item())
                    
                    # remove from input_ids the special token for the kg
                    pos_kg = torch.where(batch['input_ids'] == self.model.special_kg_token)
                    new_input_ids = torch.ones((batch['input_ids'].size(0), batch['input_ids'].size(1) - 1), dtype=batch['input_ids'].dtype, device=batch['input_ids'].device)
                    new_attention_mask = torch.ones((batch['attention_mask'].size(0), batch['attention_mask'].size(1) - 1), dtype=batch['attention_mask'].dtype, device=batch['attention_mask'].device)
                    for i in range(pos_kg[0].size(0)):
                        batch_pos = pos_kg[0][i]
                        new_input_ids[batch_pos] = torch.concat([batch['input_ids'][batch_pos, :pos_kg[1][i]], batch['input_ids'][batch_pos, pos_kg[1][i] + 1:]], dim=0)
                        new_attention_mask[batch_pos] = torch.concat([batch['attention_mask'][batch_pos, :pos_kg[1][i]], batch['attention_mask'][batch_pos, pos_kg[1][i] + 1:]], dim=0)
                    batch['attention_mask'] = new_attention_mask
                    batch['input_ids'] = new_input_ids
                    
                    # Performance without KG (pass None for graphs)
                    outputs_without_kg = self.model(
                        input_ids=batch['input_ids'],
                        graphs=None,  # No KG information
                        attention_mask=batch['attention_mask'],
                        labels=batch['input_ids'],
                        use_cache=False,
                    )
                    
                    if outputs_without_kg.loss is not None:
                        without_kg_losses.append(outputs_without_kg.loss.item())
                
                except Exception as e:
                    self.logger.warning(f"Error in knowledge utilization batch {batch_idx}: {e}")
                    continue
        
        # Compute metrics
        metrics = {}
        if with_kg_losses and without_kg_losses:
            metrics['avg_loss_with_kg'] = np.mean(with_kg_losses)
            metrics['avg_loss_without_kg'] = np.mean(without_kg_losses)
            metrics['kg_improvement'] = np.mean(without_kg_losses) - np.mean(with_kg_losses)
            metrics['relative_kg_improvement'] = metrics['kg_improvement'] / np.mean(without_kg_losses)
            
            # Perplexity comparison
            metrics['perplexity_with_kg'] = math.exp(np.mean(with_kg_losses))
            metrics['perplexity_without_kg'] = math.exp(np.mean(without_kg_losses))
            
            self.logger.info(f"KG improvement: {metrics['kg_improvement']:.4f} ({metrics['relative_kg_improvement']:.2%})")
        
        return metrics
    
    def run_comprehensive_evaluation(self, output_file: Optional[str] = None) -> Dict[str, Any]:
        """Run all evaluation metrics and return comprehensive results."""
        self.logger.info("Starting comprehensive evaluation...")
        
        # Load model and setup data
        self.load_model()
        self.setup_data()
        
        # Run all evaluations
        evaluations = {
            'perplexity': lambda: {'perplexity': self.compute_perplexity()},
            'hit_at_k_metrics': self.compute_hit_k_metrics,
            'kg_embedding_metrics': self.compute_kg_embedding_metrics,
            'knowledge_utilization': self.compute_knowledge_utilization_metrics,
            'generation_metrics': self.compute_generation_metrics,
        }
        
        # Run evaluations
        for eval_name, eval_func in evaluations.items():
            try:
                self.logger.info(f"Running {eval_name}...")
                self.results[eval_name] = eval_func()
                # Clear GPU memory after each evaluation
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
            except Exception as e:
                self.logger.error(f"Error in {eval_name}: {e}")
                self.results[eval_name] = {'error': str(e)}
        
        # Add metadata
        self.results['metadata'] = {
            'model_path': str(self.model_path),
            'config_path': self.config_path,
            'device': self.device,
            'batch_size': self.batch_size,
            'max_samples': self.max_samples,
        }
        
        # Save results if output file specified
        if output_file:
            self.save_results(output_file)
        
        return dict(self.results)
    
    def save_results(self, output_file: str):
        """Save evsaluation results to a JSON file."""
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        self.logger.info(f"Results saved to {output_path}")
    
    def print_summary(self):
        """Print a summary of evaluation results."""
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


def main():
    parser = argparse.ArgumentParser(description="Evaluate KG-LFM model")
    parser.add_argument("--config", type=str, default="configs/pretrain_config.yaml",
                       help="Path to the configuration file")
    parser.add_argument("--output_file", type=str, default="eval/out.json",
                       help="Path to save evaluation results JSON")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use for evaluation")
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Batch size for evaluation")
    parser.add_argument("--max_samples", type=int, default=None,
                       help="Maximum number of samples to evaluate (for quick testing)")
    
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'logs/eval.log'),
            logging.StreamHandler()
        ]
    )
    
    # Create evaluator
    evaluator = KGLFMEvaluator(
        config_path=args.config,
        device=args.device,
        batch_size=args.batch_size,
        max_samples=args.max_samples
    )
    
    # Run evaluation
    results = evaluator.run_comprehensive_evaluation(args.output_file)
    
    # Print summary
    evaluator.print_summary()


if __name__ == "__main__":
    main()
