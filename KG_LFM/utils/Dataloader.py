from typing import Dict, List, Optional, Union, Tuple, Any

from torch.utils.data import DataLoader, Dataset
import networkx as nx
from transformers import PreTrainedTokenizer, DataCollatorWithPadding

from KG_LFM.utils.Datasets.factories.factory import (
    trirex_factory, trex_star_graphs_factory, trex_bite_factory, 
    web_qsp_factory, grailqa_factory, grailqa_factory,
    simplequestions_factory
)
from KG_LFM.utils.Datasets.TriRex_data import TriRexStarDataset
from KG_LFM.utils.Datasets.QA_data import QADataset

from KG_LFM.utils.BigGraphNodeEmb import BigGraphAligner

from KG_LFM.configuration import DataLoaderConfig, DatasetConfig

from torch_geometric.data import Data, Batch


class KGLFM_Collator:
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
        self.max_length = max_length
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
            batch: List of samples from KGLFM_Dataset
        Returns:
            Optimally batched data
        """
        batch_size = len(batch)
        
        # Initialize result dictionary
        result = {
            'sentences': [sample['sentence'] for sample in batch],
            "conversations": [sample['conversation'] for sample in batch],
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
                'attention_mask': sample['attention_mask'],
            })
        # Apply HF collator
        collated_text = self.hf_collator(text_features)
        result['input_ids'] = collated_text['input_ids']
        result['attention_mask'] = collated_text['attention_mask']

        # Use HF collator for labels
        label_features = []
        for sample in batch:
            label_features.append({
                'input_ids': sample['labels'],
                'attention_mask': sample['attention_mask'],
            })
        # Apply HF collator
        collated_labels = self.hf_collator(label_features)
        result['labels'] = collated_labels['input_ids']

        result['graphs'] = self._process_graph_batch(result['graphs'])
        
        # Add batch metadata
        result['batch_size'] = batch_size
        
        return result
    
    def _process_graph_batch(self, graphs: List[Optional[Data]]) -> Batch:
        """Process and potentially optimize graph batch."""
        batched_graph = Batch.from_data_list(graphs)
        
        return batched_graph

class KGLFM_DataLoader:
    """
    High-level dataloader factory for TriREx and TRExStar datasets.
    """
    
    def __init__(
        self,
        dataset_config: DatasetConfig,
        dataloader_config: Optional[DataLoaderConfig] = None,
        tokenizer: Optional[PreTrainedTokenizer] = None
    ):
        self.dataset_config = dataset_config
        self.dataloader_config = dataloader_config or DataLoaderConfig()
        self.tokenizer = tokenizer
        
        self.dataset_name = dataset_config.name
        
        # Load datasets
        print(f"Loading {dataset_config.name} datasets...")
        
        dataset_factory = {
            "trirex": lambda conf: (
                trirex_factory(conf), trex_star_graphs_factory(conf)
            ),
            "trirex-bite": lambda conf: (
                trex_bite_factory(conf), trex_star_graphs_factory(conf)
            ),
            "web-qsp": self._web_qsp_factory,  # Special case for WebQSP
            "grailqa": grailqa_factory,  # GrailQA dataset
            "simple-questions": simplequestions_factory,
        }
        
        self.dataset_class = {
            "trirex": TriRexStarDataset,
            "trirex-bite": TriRexStarDataset,
            "web-qsp": QADataset,
            "grailqa": QADataset, 
            "simple-questions": QADataset
        }
        

        (self.train_dataset, self.val_dataset, self.test_dataset), self.star_graphs = dataset_factory[self.dataset_name](dataset_config)
        self.collator = self._get_collator()
        
        print("Loading BigGraphAligner...")
        # BigGraphAligner for graph processing
        self.big_graph_aligner = BigGraphAligner(
            graphs=self.star_graphs,
            config=dataset_config,
        )
        
        print(f"Loaded {len(self.train_dataset)} training samples") if self.train_dataset else "No training dataset"
        print(f"Loaded {len(self.val_dataset)} validation samples") if self.val_dataset else "No validation dataset"
        print(f"Loaded {len(self.test_dataset)} test samples") if self.test_dataset else "No test dataset"
        print(f"Loaded {len(self.star_graphs)} star graphs")
    
    def _web_qsp_factory(self, config) -> Tuple[Tuple[Dataset, Dataset, Dataset], Dict[str, nx.DiGraph]]:
        """Load WebQSP dataset and corresponding star graphs."""
        web_qsp_dataset, web_qsp_star_graphs = web_qsp_factory(config)
        return (
            (
                None,  None,  # No train & val split for WebQSP
                web_qsp_dataset
            ),
            web_qsp_star_graphs
        )

    def get_train_dataloader(self) -> DataLoader:
        """Get training dataloader with distributed sampling support."""
        if self.train_dataset is None:
            raise ValueError("No training dataset available.")
        dataset = self.dataset_class[self.dataset_name](
            self.train_dataset,
            self.star_graphs,
            self.tokenizer,
            self.big_graph_aligner,
            self.dataset_config.corrupt,
            self.dataset_config.drop_obj_entities_prob,
        )
        
        return DataLoader(
            dataset,
            batch_size=self.dataloader_config.batch_size,
            shuffle=True,
            num_workers=self.dataloader_config.num_workers,
            collate_fn=self.collator,
            pin_memory=self.dataloader_config.pin_memory,
            persistent_workers=self.dataloader_config.persistent_workers and self.dataloader_config.num_workers > 0,
            prefetch_factor=1 if self.dataloader_config.num_workers > 0 else None
        )
    
    def get_val_dataloader(self) -> DataLoader:
        """Get validation dataloader with distributed sampling support."""
        if self.val_dataset is None:
            raise ValueError("No validation dataset available.")
        dataset = self.dataset_class[self.dataset_name](
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
            persistent_workers=self.dataloader_config.persistent_workers and self.dataloader_config.num_workers > 0,
            prefetch_factor=1 if self.dataloader_config.num_workers > 0 else None
        )
    
    def get_test_dataloader(self) -> DataLoader:
        """Get test dataloader with distributed sampling support."""
        if self.test_dataset is None:
            raise ValueError("No test dataset available.")
        dataset = self.dataset_class[self.dataset_name](
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
            persistent_workers=self.dataloader_config.persistent_workers and self.dataloader_config.num_workers > 0,
            prefetch_factor=1 if self.dataloader_config.num_workers > 0 else None
        )
    
    def _get_collator(self):
        """Get the appropriate collator based on configuration."""
        return KGLFM_Collator(
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
    dataset_config: DatasetConfig,
    dataloader_config: Optional[DataLoaderConfig] = None,
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
    dataloader_factory = KGLFM_DataLoader(dataset_config, dataloader_config, tokenizer)
    
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
    

if __name__ == "__main__":
    from transformers import AutoTokenizer
        
    # Create dataset config
    dataset_config = DatasetConfig(lite=True)  # Use lite for faster loading
    dataloader_config = DataLoaderConfig(
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
        print(f"  Labels shape: {batch['labels'].shape}")

        breakpoint()

        if batch_idx >= 1:
            break
