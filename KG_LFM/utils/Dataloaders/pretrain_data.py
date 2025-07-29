import random
from typing import Dict, List, Optional, Union, Tuple, Any
import warnings

import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler
import networkx as nx
from datasets import Dataset as HFDataset
from transformers import PreTrainedTokenizer, DataCollatorWithPadding

from KG_LFM.utils.Datasets.factory import trirex_factory, trex_star_graphs_factory
from KG_LFM.utils.BigGraphNodeEmb import BigGraphAligner

from KG_LFM.configuration import TriRex_DataLoaderConfig, TRex_DatasetConfig

from torch_geometric.data import Data, Batch

class TriRexStarDataset(Dataset):
    """
    Combined dataset for TriREx sentences and TRExStar graphs.
    
    This dataset provides paired samples of:
    - TriREx sentences with subject-predicate-object triples
    - Corresponding TRExStar graph data for entities
    """
    
    def __init__(
        self,
        trirex_dataset: HFDataset,
        star_graphs: Dict[str, nx.DiGraph],
        tokenizer: PreTrainedTokenizer,
        big_graph_aligner: BigGraphAligner,
    ):
        self.trirex_dataset = trirex_dataset
        self.star_graphs = star_graphs
        self.tokenizer = tokenizer
        self.big_graph_aligner = big_graph_aligner
        
        # add to tokenizer a special token <KG_EMBEDDING> for graph embeddings
        if "<KG_EMBEDDING>" not in self.tokenizer.get_vocab():
            warnings.warn(
                "The <KG_EMBEDDING> token is not in the tokenizer vocabulary. "
                "Adding it to the tokenizer. This may lead to unexpected behavior."
            )
            self.tokenizer.add_special_tokens({'additional_special_tokens': ['<KG_EMBEDDING>']})
    
    def __len__(self) -> int:
        return len(self.trirex_dataset)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a sample containing both TriREx sentence data and corresponding graph data.
        
        Returns:
            Dict containing:
            - sentence: The original sentence
            - subject: Subject entity information
            - predicate: Predicate information  
            - object: Object entity information
            - subject_graph: NetworkX graph for subject entity (if available)
            - object_graph: NetworkX graph for object entity (if available)
            - tokenized_input: Tokenized sentence (if tokenizer provided)
        """
        # print(f"Fetching sample {idx} from TriREx dataset")
        sample = self.trirex_dataset[idx]
        
        # add to the sentence a special token for graph embedding after the subject
        subject_boundaries = sample['subject']['boundaries']
        start_subject = subject_boundaries[0]
        end_subject = subject_boundaries[1]
        
        # Insert <KG_EMBEDDING> token after the subject
        sample['sentence'] = (
            sample['sentence'][:end_subject] +
            ' <KG_EMBEDDING> ' +
            sample['sentence'][end_subject:]
        )
        
        new_chars = len(' <KG_EMBEDDING>')
        # add to the object boundaries the " <KG_EMBEDDING>" token
        object_boundaries = sample["object"]['boundaries']
        start_object = object_boundaries[0] + new_chars
        end_object = object_boundaries[1] + new_chars
        
        # Insert <KG_EMBEDDING> token after the object
        sample["object"]['boundaries'] = [start_object, end_object]

        result = {
            'sentence': sample['sentence'],
            'subject': sample['subject'],
            'predicate': sample['predicate'],
            'object': sample['object']
        }
        
        # Add graph data if available and requested
        subject_id = sample['subject']['id']
        
        # Get subject graph
        subject_graph = self.star_graphs.get(subject_id, None)
        assert subject_graph is not None, f"Subject graph for {subject_id} not found."
        
        result['graph'] = self._process_graph(subject_graph)
        
        # Tokenize the sentence
        if self.tokenizer is None:
            raise ValueError("Tokenizer must be provided for text processing.")
            
        tokenized = self.tokenizer(
            sample['sentence'],
            return_tensors='pt'
        )
        result['input_ids'] = tokenized['input_ids'].squeeze(0)
        result['attention_mask'] = tokenized['attention_mask'].squeeze(0)
            
        return result
    
    def _process_graph(self, graph: str) -> Data:
        """
        Process and potentially sample the graph based on configuration.
        
        Args:
            graph: NetworkX DiGraph
            
        Returns:
            Processed graph data as torch geometric format.
        """
        # Convert NetworkX graph to a format suitable for PyTorch Geometric
        neighbour_ids, edge_ids = [], []
        contral_node_id = None
        
        neighbour_node_labels, edge_labels = [], []
        central_node_label = None
        
        nodes = graph.nodes(data=True)
        for central_node_id, neighbour_node_id, edge in graph.edges(data=True):
            edge_ids.append(edge['id'])
            neighbour_ids.append(neighbour_node_id)
            if contral_node_id is None:
                contral_node_id = central_node_id
                central_node_label = nodes[central_node_id]['label']
                
            edge_labels.append(edge['label'])
            neighbour_node_labels.append(nodes[neighbour_node_id]['label'])
        

        central_node_emb = self.big_graph_aligner.node_embedding(contral_node_id)
        neighbour_node_embs = self.big_graph_aligner.node_embedding_batch(neighbour_ids)
        edge_embs = self.big_graph_aligner.edge_embedding_batch(edge_ids)
        
        # Create edge index tensor for the star graph structure
        num_neighbors = len(neighbour_ids)
        
        # Each neighbor connects to the central node
        # Create edges: neighbors->central and central->neighbors
        edge_index = torch.tensor([
            list(range(1, num_neighbors + 1)) + [0] * num_neighbors,  # Neighbor nodes to central node
            [0] * num_neighbors  + list(range(1, num_neighbors + 1))  # Central node to neighbor nodes
        ], dtype=torch.long)
        
        # Node features: central node + neighbor nodes
        node_features = torch.cat([
            central_node_emb.unsqueeze(0),  # Central node embedding
            neighbour_node_embs  # Neighbor node embeddings
        ], dim=0)
        
        # Edge features: duplicate for bidirectional edges (neighbors->central and central->neighbors)
        edge_features = torch.cat([edge_embs, edge_embs], dim=0)
        
        # Create a PyTorch Geometric graph data object
        graph_data = Data(
            x=node_features,
            edge_index=edge_index,
            edge_attr=edge_features,
            num_nodes=node_features.shape[0],
            central_node_label=central_node_label,
            neighbour_node_labels=neighbour_node_labels,
            edge_labels=edge_labels,
        )
        
        return graph_data


class TriRexStarCollator:
    """
    Advanced collator with support for dynamic padding, graph batching, and various optimizations.
    """
    
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        padding: Union[bool, str] = True,
        max_length: Optional[int] = None,
        pad_to_multiple_of: Optional[int] = None,
        return_tensors: str = "pt",
    ):
        self.tokenizer = tokenizer
        self.padding = padding
        self.max_length = max_length if max_length is not None else tokenizer.model_max_length
        self.pad_to_multiple_of = pad_to_multiple_of
        self.return_tensors = return_tensors
        
        # Initialize HF collator if tokenizer provided
        self.hf_collator = DataCollatorWithPadding(
            tokenizer=tokenizer,
            padding=padding,
            max_length=max_length,
            pad_to_multiple_of=pad_to_multiple_of,
            return_tensors=return_tensors
        )
    
    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Advanced collation with dynamic padding and graph optimization.
        
        Args:
            batch: List of samples from TriRexStarDataset
            
        Returns:
            Optimally batched data
        """
        batch_size = len(batch)
        
        # Initialize result dictionary
        result = {
            'sentences': [sample['sentence'] for sample in batch],
            'subjects': [sample['subject'] for sample in batch],
            'predicates': [sample['predicate'] for sample in batch],
            'objects': [sample['object'] for sample in batch],
            'graphs': [sample.get('graph') for sample in batch],
        }
        
        # Use HF collator for optimal padding
        text_features = []
        for sample in batch:
            text_features.append({
                'input_ids': sample['input_ids'],
                'attention_mask': sample['attention_mask']
            })
        
        # Apply HF collator
        collated_text = self.hf_collator(text_features)
        result['input_ids'] = collated_text['input_ids']
        result['attention_mask'] = collated_text['attention_mask']
        
        result['graphs'] = self._process_graph_batch(result['graphs'])
        
        # Add batch metadata
        result['batch_size'] = batch_size
        
        return result
    
    def _process_graph_batch(self, graphs: List[Optional[Data]]) -> Batch:
        """Process and potentially optimize graph batch."""
        batched_graph = Batch.from_data_list(graphs)
        
        return batched_graph

class TriRexStarDataLoader:
    """
    High-level dataloader factory for TriREx and TRExStar datasets.
    """
    
    def __init__(
        self,
        dataset_config: TRex_DatasetConfig,
        dataloader_config: Optional[TriRex_DataLoaderConfig] = None,
        tokenizer: Optional[PreTrainedTokenizer] = None
    ):
        self.dataset_config = dataset_config
        self.dataloader_config = dataloader_config or TriRex_DataLoaderConfig()
        self.tokenizer = tokenizer
        
        # Load datasets
        print("Loading TriREx and TRExStar datasets...")
        self.train_dataset, self.val_dataset, self.test_dataset = trirex_factory(dataset_config)
        self.star_graphs = trex_star_graphs_factory(dataset_config)
        
        self.collator = self._get_collator()
        
        print("Loading BigGraphAligner...")
        # BigGraphAligner for graph processing
        self.big_graph_aligner = BigGraphAligner(
            graphs=self.star_graphs,
            config=dataset_config,
        )
        
        print(f"Loaded {len(self.train_dataset)} training samples")
        print(f"Loaded {len(self.val_dataset)} validation samples")
        print(f"Loaded {len(self.test_dataset)} test samples")
        print(f"Loaded {len(self.star_graphs)} star graphs")
    
    def get_train_dataloader(self) -> DataLoader:
        """Get training dataloader with distributed sampling support."""
        dataset = TriRexStarDataset(
            self.train_dataset,
            self.star_graphs,
            self.tokenizer,
            self.big_graph_aligner,
        )
        
        return DataLoader(
            dataset,
            batch_size=self.dataloader_config.batch_size,
            shuffle=True,
            num_workers=self.dataloader_config.num_workers,
            collate_fn=self.collator,
            pin_memory=self.dataloader_config.pin_memory,
            persistent_workers=self.dataloader_config.persistent_workers and self.dataloader_config.num_workers > 0
        )
    
    def get_val_dataloader(self) -> DataLoader:
        """Get validation dataloader with distributed sampling support."""
        dataset = TriRexStarDataset(
            self.val_dataset,
            self.star_graphs,
            self.tokenizer,
            self.big_graph_aligner,
        )
        
        return DataLoader(
            dataset,
            batch_size=self.dataloader_config.batch_size,
            shuffle=True, # to allow to only use a subset of the validation dataset during training we want to do sample it with iid
            num_workers=self.dataloader_config.num_workers,
            collate_fn=self.collator,
            pin_memory=self.dataloader_config.pin_memory,
            persistent_workers=self.dataloader_config.persistent_workers and self.dataloader_config.num_workers > 0
        )
    
    def get_test_dataloader(self) -> DataLoader:
        """Get test dataloader with distributed sampling support."""
        dataset = TriRexStarDataset(
            self.test_dataset,
            self.star_graphs,
            self.tokenizer,
            self.big_graph_aligner,
        )
        
        return DataLoader(
            dataset,
            batch_size=self.dataloader_config.batch_size,
            shuffle=False,
            num_workers=self.dataloader_config.num_workers,
            collate_fn=self.collator,
            pin_memory=self.dataloader_config.pin_memory,
            persistent_workers=self.dataloader_config.persistent_workers and self.dataloader_config.num_workers > 0
        )
    
    def _get_collator(self):
        """Get the appropriate collator based on configuration."""
        return TriRexStarCollator(
            tokenizer=self.tokenizer,
            padding=self.dataloader_config.padding,
            max_length=self.dataloader_config.max_sequence_length,
            pad_to_multiple_of=self.dataloader_config.pad_to_multiple_of,
            return_tensors=self.dataloader_config.return_tensors,
        )
    
    def get_all_dataloaders(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Get all dataloaders (train, val, test)."""
        return (
            self.get_train_dataloader(),
            self.get_val_dataloader(),
            self.get_test_dataloader()
        )

def create_dataloader(
    dataset_config: TRex_DatasetConfig,
    dataloader_config: Optional[TriRex_DataLoaderConfig] = None,
    tokenizer: Optional[PreTrainedTokenizer] = None,
    split: str = "train",
) -> DataLoader:
    """
    Convenience function to create a single dataloader with distributed support.
    
    Args:
        dataset_config: Configuration for the dataset
        dataloader_config: Configuration for the dataloader
        tokenizer: Optional tokenizer for text processing
        split: Which split to load ("train", "val", "test", "all")
        distributed: Whether to use distributed sampling
        
    Returns:
        DataLoader for the specified split or a tuple of dataloaders for all splits.
    """
    print(f"Creating dataloader for split")
    dataloader_factory = TriRexStarDataLoader(dataset_config, dataloader_config, tokenizer)
    
    if split == "train":
        return dataloader_factory.get_train_dataloader()
    elif split == "val" or split == "validation":
        return dataloader_factory.get_val_dataloader()
    elif split == "test":
        return dataloader_factory.get_test_dataloader()
    elif split == "all":
        print("Returning all dataloaders (train, val, test)...")
        train_loader, val_loader, test_loader = dataloader_factory.get_all_dataloaders()
        return train_loader, val_loader, test_loader
    else:
        raise ValueError(f"Unknown split: {split}. Use 'train', 'val', 'test', or 'all'.")

def create_datasets(
    dataset_config: TRex_DatasetConfig,
    tokenizer: Optional[PreTrainedTokenizer] = None
) -> Tuple[HFDataset, HFDataset, HFDataset]:
    """
    Convenience function to create datasets for all splits.
    
    Args:
        dataset_config: Configuration for the dataset
        tokenizer: Optional tokenizer for text processing
        
    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    train_dataset, val_dataset, test_dataset = trirex_factory(dataset_config)
    graphs = trex_star_graphs_factory(dataset_config)
    
    graph_aligner = BigGraphAligner(
        graphs=graphs,
        config=dataset_config,
    )
    
    # Create TriRexStarDataset instances for each split
    train_dataset = TriRexStarDataset(
        train_dataset,
        graphs,
        tokenizer,
        graph_aligner
    )
    
    val_dataset = TriRexStarDataset(
        val_dataset,
        graphs,
        tokenizer,
        graph_aligner
    )
    
    test_dataset = TriRexStarDataset(
        test_dataset,
        graphs,
        tokenizer,
        graph_aligner
    )
    
    return train_dataset, val_dataset, test_dataset
    

if __name__ == "__main__":
    from transformers import AutoTokenizer
        
    # Create dataset config
    dataset_config = TRex_DatasetConfig(lite=True)  # Use lite for faster loading
    dataloader_config = TriRex_DataLoaderConfig(
        batch_size=8,
        max_sequence_length=256,
        include_graphs=True
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
    # Create dataloader
    train_loader = create_dataloader(
        dataset_config, 
        dataloader_config, 
        tokenizer, 
        split="train"
    )
    
    # Iterate through batches
    for batch_idx, batch in enumerate(train_loader):
        print(f"Batch {batch_idx}:")
        print(f"  Input IDs shape: {batch['tokenized_input']['input_ids'].shape}")
        print(f"  Attention mask shape: {batch['tokenized_input']['attention_mask'].shape}")
        
        breakpoint()

        if batch_idx >= 1:
            break
