from typing import Dict, List, Optional, Union, Tuple, Any
import warnings

import torch
from torch.utils.data import Dataset
import networkx as nx
from datasets import Dataset as HFDataset
from transformers import PreTrainedTokenizer

from KG_LM.utils.BigGraphNodeEmb import BigGraphAligner

from KG_LM.configuration import DatasetConfig, SPECIAL_KG_TOKEN, IGNORE_INDEX

from torch_geometric.data import Data


negative_answers = [
    "The Knowledge Graph does not contain this information.",
    "The Knowledge Graph does not have this information.",
    "This information is not available in the Knowledge Graph.",
    "The KG returned no results for this query.",
    "The KG does not contain this information.",
    "With the current knowledge I have, I cannot provide an answer to that question.",
    "I don't have that information in my knowledge base.",
    "I'm sorry, but I don't have that information.",
    "I don't have that information.",
    "I don't know.",
]


class QADataset(Dataset):
    """
    Combined dataset for TriREx sentences and TRExStar graphs.
    
    This dataset provides paired samples of:
    - TriREx sentences with subject-predicate-object triples
    - Corresponding TRExStar graph data for entities
    """
    
    def __init__(
        self,
        webqsp_dataset: HFDataset,
        star_graphs: Dict[str, nx.DiGraph],
        tokenizer: PreTrainedTokenizer,
        big_graph_aligner: BigGraphAligner,
        corrupted: bool = False,
        drop_answer_prob: float = 0.0,
    ):
        self.webqsp_dataset = webqsp_dataset
        self.star_graphs = star_graphs
        self.tokenizer = tokenizer
        self.big_graph_aligner = big_graph_aligner
        
        self.corrupted = corrupted
        self.drop_answer_prob = drop_answer_prob
        
        # add to tokenizer a special token SPECIAL_KG_TOKEN for graph embeddings
        if SPECIAL_KG_TOKEN not in self.tokenizer.get_vocab():
            warnings.warn(
                f"The {SPECIAL_KG_TOKEN} token is not in the tokenizer vocabulary. "
                "Adding it to the tokenizer. This may lead to unexpected behavior."
            )
            self.tokenizer.add_special_tokens({'additional_special_tokens': [SPECIAL_KG_TOKEN]})

    def __len__(self) -> int:
        return len(self.webqsp_dataset)
    
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
        # print(f"Fetching sample {idx} from WebQSP dataset")
        sample = self.webqsp_dataset[idx]

        # add to the question a special token for graph embedding after the subject
        subject_text = sample['subject']['label']

        # Get question ID for WebQSP (which has multiple answers per question)
        # For other datasets, use incremental ID so each sample is treated independently
        question_id = sample.get('question_id', f"sample_{idx}")
        
        result = {
            'question_id': question_id,  # Add question ID for grouping (for WebQSP) or unique ID (for others)
            'subject': sample['subject'],
            'predicate': sample['predicate'],
            'object': sample['object']
        }
        
        object_id = None
        
        # Add graph data if available and requested
        subject_id = sample['subject']['id']
        
        # Get subject graph
        subject_graph = self.star_graphs.get(subject_id, None)
        assert subject_graph is not None, f"Subject graph for {subject_id} not found."
        
        answer = sample['answer']
        
        random_value = torch.rand(1).item()
        if self.corrupted:
            object_id = sample['object']['id']
        elif (self.drop_answer_prob and random_value < self.drop_answer_prob):
            object_id = sample['object']['id']
            answer = negative_answers[idx % len(negative_answers)]
            
        result['graph'] = self._process_graph(subject_graph, object_node_id=object_id)
        # Tokenize the sentence
        if self.tokenizer is None:
            raise ValueError("Tokenizer must be provided for text processing.")
        
        
        result["conversation"]=[
            {
                'role': 'user',
                'content': sample['question']
            },
            {
                'role': 'tool' if "tool" in self.tokenizer.chat_template else 'assistant',
                'content': subject_text + SPECIAL_KG_TOKEN
            },
            {
                'role': 'assistant',
                'content': ("the answer is: " if answer not in negative_answers else "") + answer
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
            print("Tokenizer does not support chat templates. Using fallback method.")
            # Fallback for tokenizers without chat template support
            sentence = f"user: {sample['question']} \ntool: {subject_text}{SPECIAL_KG_TOKEN}  \nassistant: {sample['answer']}"
        result['sentence'] = sentence
        
        tokenized = self.tokenizer(
            sentence,
            return_tensors='pt'
        )
        result['input_ids'] = tokenized['input_ids'].squeeze(0)
        result['attention_mask'] = tokenized['attention_mask'].squeeze(0)
        
        # find the index of the answer in the sentence
        answer_start = sentence.find(answer)
        if answer_start == -1:
            raise ValueError("Answer not found in the tokenized sentence.")
        # Adjust the start and end indices of the object boundaries
        result['object']['boundaries'] = [
            answer_start,
            answer_start + len(answer)
        ]
        
        # compute labels by masking everything except for the object
        tok_obj_start = tokenized.char_to_token(result['object']['boundaries'][0])
        tok_obj_end = tokenized.char_to_token(result['object']['boundaries'][1])
        result["object"]["token_boundaries"] = (tok_obj_start, tok_obj_end)
        
        labels = torch.full(result['input_ids'].shape, IGNORE_INDEX, dtype=result['input_ids'].dtype)
        labels[tok_obj_start:tok_obj_end] = result['input_ids'][tok_obj_start:tok_obj_end]
        result['labels'] = labels

        return result

    def _process_graph(self, graph: nx.DiGraph, object_node_id: int=None) -> Data:
        """
        Process and potentially sample the graph based on configuration.
        
        Args:
            graph: NetworkX DiGraph
            object_node_id: ID of the object node. If provided, it will not be included in the neighbors.
            
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
            if object_node_id == neighbour_node_id:
                continue  # Skip the object node if specified
            
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
    from KG_LM.utils.Datasets.factories.factory import web_qsp_factory
        
    # Create dataset config
    dataset_config = DatasetConfig(lite=True)  # Use lite for faster loading
    dataset_config.name = "web-qsp"  # Use WebQSP dataset
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
    
    # Load WebQSP dataset
    webqsp_dataset, graphs = web_qsp_factory(dataset_config)
    
    graph_aligner = BigGraphAligner(
        graphs=graphs,
        config=dataset_config,
    )
    
    # Create the combined dataset
    combined_dataset = QADataset(
        webqsp_dataset=webqsp_dataset,
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