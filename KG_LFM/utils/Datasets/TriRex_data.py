from typing import Dict, Any
import warnings

import torch
from torch.utils.data import Dataset
import networkx as nx
from datasets import Dataset as HFDataset
from transformers import PreTrainedTokenizer

from KG_LFM.utils.BigGraphNodeEmb import BigGraphAligner

from KG_LFM.configuration import SPECIAL_KG_TOKEN, DatasetConfig

from torch_geometric.data import Data

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
        
        # add to tokenizer a special token SPECIAL_KG_TOKEN for graph embeddings
        if SPECIAL_KG_TOKEN not in self.tokenizer.get_vocab():
            warnings.warn(
                f"The {SPECIAL_KG_TOKEN} token is not in the tokenizer vocabulary. "
                "Adding it to the tokenizer. This may lead to unexpected behavior."
            )
            self.tokenizer.add_special_tokens({'additional_special_tokens': [SPECIAL_KG_TOKEN]})

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

        # Insert SPECIAL_KG_TOKEN after the subject
        sample['sentence'] = (
            sample['sentence'][:end_subject] +
            SPECIAL_KG_TOKEN +
            sample['sentence'][end_subject:]
        )

        new_chars = len(SPECIAL_KG_TOKEN)
        # add to the object boundaries the SPECIAL_KG_TOKEN token
        object_boundaries = sample["object"]['boundaries']
        start_object = object_boundaries[0] + new_chars
        end_object = object_boundaries[1] + new_chars
        
        # Insert SPECIAL_KG_TOKEN token after the object
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
        
        sentence = sample['sentence']
        # if tokenizer has chat template, use it
        if hasattr(self.tokenizer, 'apply_chat_template'):
            sentence = self.tokenizer.apply_chat_template(
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
        
        tokenized = self.tokenizer(
            sentence,
            return_tensors='pt'
        )
        result['input_ids'] = tokenized['input_ids'].squeeze(0)
        result['attention_mask'] = tokenized['attention_mask'].squeeze(0)
            
        return result
    
    def _process_graph(self, graph: nx.DiGraph) -> Data:
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


if __name__ == "__main__":
    from transformers import AutoTokenizer
        
    # Create dataset config
    dataset_config = DatasetConfig(lite=True)  # Use lite for faster loading
    dataset_config.name = "trirex"  # Use TriREx dataset