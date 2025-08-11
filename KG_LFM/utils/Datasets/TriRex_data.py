from typing import Dict, Any
import warnings

import torch
from torch.utils.data import Dataset
import networkx as nx
from datasets import Dataset as HFDataset
from transformers import PreTrainedTokenizer

from KG_LFM.utils.BigGraphNodeEmb import BigGraphAligner

from KG_LFM.configuration import SPECIAL_KG_TOKEN, DatasetConfig, IGNORE_INDEX

from torch_geometric.data import Data


user_questions = [
    "what do you know about {subject}?",
    "tell me all you know about {subject}",
    "what can you find about {subject}?",
    "can you provide more details on {subject}?",
    "give me background on {subject}.",
    "what's important to know about {subject}?",
    "describe {subject} in general terms.",
    "summarize the key facts about {subject}.",
    "what knowledge is available about {subject}?",
    "give a high-level overview of {subject}.",
    "what are the basics of {subject}?",
    "explain {subject} briefly.",
    "what is {subject} known for?",
    "outline the main points about {subject}.",
    "provide general information on {subject}.",
    "what should I know first about {subject}?",
    "share notable facts about {subject}.",
    "what connections are relevant to {subject}?",
    "what context is useful for understanding {subject}?",
    "how would you characterize {subject} overall?",
]


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
        subject_text = sample['subject']['label']

        result = {
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

        result["conversation"] = [
            {
                'role': 'user',
                'content': user_questions[idx % len(user_questions)].format(subject=subject_text)
            },
            {
                'role': 'tool' if "tool" in self.tokenizer.chat_template else 'assistant',
                'content': subject_text + SPECIAL_KG_TOKEN
            },
            {
                'role': 'assistant',
                'content': sample['sentence']
            }
        ]
        # if tokenizer has chat template, use it
        try:
            sentence = self.tokenizer.apply_chat_template(
                conversation=result["conversation"],
                tokenize=False,
                add_generation_prompt=False,
                enable_thinking=False,
            )
        except ValueError:
            # Fallback for tokenizers without chat template support
            sentence = f"user: {user_questions[idx % len(user_questions)].format(subject=subject_text)}\ntool: {SPECIAL_KG_TOKEN}\nassistant: {sample['sentence']}"

        tokenized = self.tokenizer(
            sentence,
            return_tensors='pt'
        )
        
        result["sentence"] = sentence
        result['input_ids'] = tokenized['input_ids'].squeeze(0)
        result['attention_mask'] = tokenized['attention_mask'].squeeze(0)

        # compute labels by masking everything except for the object
        shift = sentence.find(sample["sentence"])
        object_char_start, object_char_end = sample['object']['boundaries']
        object_char_start += shift
        object_char_end += shift
        result['object']['boundaries'] = [object_char_start, object_char_end]
        
        # tokenized to char mappings
        obj_tok_start = tokenized.char_to_token(object_char_start)
        obj_tok_end = tokenized.char_to_token(object_char_end)
        result["object"]["token_boundaries"] = [obj_tok_start, obj_tok_end]

        # compute labels
        labels = torch.full(result['input_ids'].shape, IGNORE_INDEX, dtype=result['input_ids'].dtype)
        labels[obj_tok_start:obj_tok_end] = result['input_ids'][obj_tok_start:obj_tok_end]
        result['labels'] = labels

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
    from KG_LFM.utils.Datasets.factories.factory import trirex_factory, trex_star_graphs_factory
        
    # Create dataset config
    dataset_config = DatasetConfig(lite=True)  # Use lite for faster loading
    dataset_config.name = "trirex"  # Use WebQSP dataset
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
    
    # Load TRIRex dataset and graphs
    trirex_dataset, _, _ = trirex_factory(dataset_config)
    graphs = trex_star_graphs_factory(dataset_config)
    
    
    graph_aligner = BigGraphAligner(
        graphs=graphs,
        config=dataset_config,
    )
    
    # Create the combined dataset
    combined_dataset = TriRexStarDataset(
        trirex_dataset=trirex_dataset,
        star_graphs=graphs,
        tokenizer=tokenizer,
        big_graph_aligner=graph_aligner
    )
    
    # Example usage
    sample = combined_dataset[0]
    print(sample)
    print("Sample question:", sample['sentence'])
    print("Subject:", sample['subject'])
    print("Predicate:", sample['predicate'])
    print("Object:", sample['object'])
    print("Graph data:", sample['graph'])
    print("Tokenized input IDs:", sample['input_ids'])
    print("Attention mask:", sample['attention_mask'])