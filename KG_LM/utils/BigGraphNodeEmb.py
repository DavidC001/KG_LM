import csv
import logging
import os
from pathlib import Path
from typing import List, Dict

import h5py
import networkx as nx
import numpy as np
import torch
from tqdm import tqdm
from torchbiggraph.graph_storages import FORMAT_VERSION_ATTR, FORMAT_VERSION

from sentence_transformers import SentenceTransformer
from transformers import AutoConfig
from KG_LM.configuration import DatasetConfig


class BigGraphAligner:

    def __init__(self, graphs: Dict[str, nx.DiGraph], config: DatasetConfig = DatasetConfig(), batch_size: int = 64):
        
        self.config: DatasetConfig = config

        if "trirex" in self.config.name:
            self.dataset_name: str = "lite" if self.config.lite else "full"
        else:
            self.dataset_name: str = self.config.name
        
        self.folder: Path =  Path(self.config.graph_embs_base_path) / "big_graph_aligner" / self.config.graph_nodes_embedding_model / self.dataset_name
        os.makedirs(f'{self.folder}', exist_ok=True)
        
        self.batch_size: int = batch_size
        
        self.graphs: Dict[str, nx.DiGraph] = graphs

        self.entity_index: Dict[str, np.array] = dict()   # Maps entity IDs to their embedding value
        self.relation_index: Dict[str, np.array] = dict() # Maps relation IDs to their embedding value

        # embedder configuration
        self.embedder: AutoConfig = AutoConfig.from_pretrained(self.config.graph_nodes_embedding_model)
        # embedder dimension
        self.embedder_dim: int = self.embedder.hidden_size
        
        self.preload = self.config.preload_nodes_embeddings
        
        self.prepare()
        self.build_index()

    def prepare(self):
        if os.path.isfile(f'{self.folder}/init/embeddings_entity_0.v0.h5') and os.path.isfile(f'{self.folder}/init/embeddings_relation_0.v0.h5'):
            print("Already prepared")
            return

        print("Preparing BigGraphAligner...")

        embedder: SentenceTransformer = SentenceTransformer(
            self.config.graph_nodes_embedding_model,
            model_kwargs={
                "attn_implementation": "flash_attention_2",
                "device_map": "cuda:0",
                "torch_dtype": torch.bfloat16
            } if torch.cuda.is_available() else {},
            tokenizer_kwargs={"padding_side": "left"},
        )
        
        os.makedirs(f'{self.folder}/init', exist_ok=True)

        entities = dict()
        relations = dict()
        triples = set()

        # Open a file to write the edges
        for G in tqdm(self.graphs.values(), desc='Processing graphs', total=len(self.graphs)):
            for central_node_id, neighbour_node_id, edge in G.edges(data=True):
                assert central_node_id == G.graph['central_node'], "Graph has wrong format, expect all edges to be from central node"
                relation_id = edge['id']
                relation_label = edge['label']

                neighbour_node_label = G.nodes[neighbour_node_id]['label']
                central_node_label = G.nodes[central_node_id]['label']

                # Write the edge in the format required by PBG
                entities[neighbour_node_id] = neighbour_node_label
                entities[central_node_id] = central_node_label

                relations[relation_id] = relation_label

                triples.add((central_node_id, relation_id, neighbour_node_id))

        entity_list = [(key, value) for key, value in entities.items()]
        sorted_entity_list = sorted(entity_list, key=lambda x: int(x[0][1:]))

        relation_list = [(key, value) for key, value in relations.items()]
        sorted_relation_list = sorted(relation_list, key=lambda x: int(x[0][1:]))

        triples_list = list(triples)
        sorted_triples_list = sorted(triples_list, key=lambda x: (int(x[0][1:]), int(x[1][1:]), int(x[2][1:])))

        with open(f'{self.folder}/graph_edges.csv', 'w') as file:
            writer = csv.writer(file)
            writer.writerow(['Subject', 'Predicate', 'Object'])
            writer.writerows(sorted_triples_list)

        # Write entities to a file
        with open(f'{self.folder}/entities.csv', 'w') as file:
            writer = csv.writer(file)
            writer.writerow(['ID', 'Label'])
            for entity in tqdm(sorted_entity_list, desc='Writing entities'):
                writer.writerow(entity)

        # Write relations to a file
        with open(f'{self.folder}/relations.csv', 'w') as file:
            writer = csv.writer(file)
            writer.writerow(['ID', 'Label'])
            for relation in tqdm(sorted_relation_list, desc='Writing relations'):
                writer.writerow(relation)

        # Batch encode entities for better GPU utilization
        entity_labels = [entity[1] for entity in sorted_entity_list]
        all_embeddings = []
        for i in tqdm(range(0, len(entity_labels), self.batch_size), desc='Encoding entity labels'):
            batch_labels = entity_labels[i:i+self.batch_size]
            batch_embeddings = embedder.encode(batch_labels, convert_to_tensor=True)
            # if any is nan breakpoint
            if torch.isnan(batch_embeddings).any():
                # send warning
                logging.warning(f"NaN found in batch {i // self.batch_size}, substituting with zero vector.")
                batch_embeddings = torch.nan_to_num(batch_embeddings, nan=0.0)
            all_embeddings.append(batch_embeddings.cpu())
        
        # Save Entity embeddings
        dataset = torch.cat(all_embeddings, dim=0)
        with h5py.File(f'{self.folder}/init/embeddings_entity_0.v0.h5', 'w') as hf:
            hf.create_dataset("embeddings", data=dataset.cpu().float().numpy(), dtype=np.float32)
            hf.attrs[FORMAT_VERSION_ATTR] = FORMAT_VERSION

        # Batch encode relations
        all_embeddings = []
        relation_labels = [relation[1] for relation in sorted_relation_list]
        for i in tqdm(range(0, len(relation_labels), self.batch_size), desc='Encoding relation labels'):
            batch_labels = relation_labels[i:i+self.batch_size]
            batch_embeddings = embedder.encode(batch_labels, convert_to_tensor=True)
            all_embeddings.append(batch_embeddings.cpu())
        
        # Save Relation embeddings
        dataset = torch.cat(all_embeddings, dim=0)
        with h5py.File(f'{self.folder}/init/embeddings_relation_0.v0.h5', 'w') as hf:
            hf.create_dataset("embeddings", data=dataset.cpu().float().numpy(), dtype=np.float32)
            hf.attrs[FORMAT_VERSION_ATTR] = FORMAT_VERSION

        print("BigGraphAligner prepared successfully.")

    def build_index(self):
        self.entity_index = {}
        self.relation_index = {}
        
        # Store file paths for lazy loading (solves pickling issue)
        self.entity_file_path = f'{self.folder}/init/embeddings_entity_0.v0.h5'
        self.relation_file_path = f'{self.folder}/init/embeddings_relation_0.v0.h5'
        
        # For non-preload mode, we'll open files lazily to avoid pickling issues
        self.entity_mmap_file = None
        self.relation_mmap_file = None
        
        if self.preload:
            # Load everything into memory (original behavior)
            with h5py.File(self.entity_file_path, 'r') as hf:
                trained_embeddings = hf['embeddings'][:]
                    
            with h5py.File(self.relation_file_path, 'r') as hf:
                trained_embeddings_rel = hf['embeddings'][:]

        total_entities = 0
        with open(f'{self.folder}/entities.csv', 'r') as file:
            reader = csv.reader(file)
            next(reader)  # Skip the header row
            for i, row in tqdm(enumerate(reader), desc='Building entity index'):
                entity_id, _ = row
                if self.preload:
                    self.entity_index[entity_id] = torch.from_numpy(trained_embeddings[i, :])
                else:
                    # Store index for memory-mapped access
                    self.entity_index[entity_id] = i
                total_entities += 1
        with open(f'{self.folder}/relations.csv', 'r') as file:
            reader = csv.reader(file)
            next(reader)  # Skip the header row
            for i, row in tqdm(enumerate(reader), desc='Building relation index'):
                relation_id, _ = row
                if self.preload:
                    self.relation_index[relation_id] = torch.from_numpy(trained_embeddings_rel[i, :])
                else:
                    # Store index for memory-mapped access
                    self.relation_index[relation_id] = i

        assert len(self.entity_index) > 0, "Entity index is empty"
        assert len(self.relation_index) > 0, "Relation index is empty"

    def _get_entity_mmap_file(self):
        """Lazily open entity embeddings file. This way each subprocess has its own and does not need to be pickled."""
        if self.entity_mmap_file is None:
            self.entity_mmap_file = h5py.File(self.entity_file_path, 'r')
        return self.entity_mmap_file

    def _get_relation_mmap_file(self):
        """Lazily open relation embeddings file. This way each subprocess has its own and does not need to be pickled."""
        if self.relation_mmap_file is None:
            self.relation_mmap_file = h5py.File(self.relation_file_path, 'r')
        return self.relation_mmap_file

    def node_embedding(self, entity_id: str) -> torch.Tensor:
        if entity_id not in self.entity_index:
            raise ValueError(f"Entity ID {entity_id} not found in the entity index.")
        
        if self.preload:
            # Original behavior - embeddings are in memory
            emb = self.entity_index[entity_id]
        else:
            # Memory-mapped access with lazy loading
            idx = self.entity_index[entity_id]
            entity_file = self._get_entity_mmap_file()
            emb = torch.from_numpy(entity_file['embeddings'][idx, :])
        
        # Check for NaN values
        if torch.isnan(emb).any():
            logging.warning(f"NaN found in entity ID {entity_id}, Substituting with zero vector.")
            return torch.zeros(self.embedder_dim)
            
        return emb

    def edge_embedding(self, predicate_id: str) -> torch.Tensor:
        if predicate_id not in self.relation_index:
            raise ValueError(f"Predicate ID {predicate_id} not found in the relation index.")
        
        if self.preload:
            # Original behavior - embeddings are in memory
            return self.relation_index[predicate_id]
        else:
            # Memory-mapped access with lazy loading
            idx = self.relation_index[predicate_id]
            relation_file = self._get_relation_mmap_file()
            embedding = torch.from_numpy(relation_file['embeddings'][idx, :])
            return embedding

    def node_embedding_batch(self, entity_ids: List[str]) -> torch.Tensor:
        # Original behavior - fast memory access
        batch_embedding = torch.zeros((len(entity_ids), self.embedder_dim))
        for i, entity_id in enumerate(entity_ids):
            batch_embedding[i, :] = self.node_embedding(entity_id)
        
        self.close_files()  # Close files after use if not preloading
        return batch_embedding

    def edge_embedding_batch(self, predicate_ids: List[str]) -> torch.Tensor:
        # Original behavior - fast memory access
        batch_embedding = torch.zeros((len(predicate_ids), self.embedder_dim))
        for i, predicate_id in enumerate(predicate_ids):
            batch_embedding[i, :] = self.edge_embedding(predicate_id)

        self.close_files()  # Close files after use if not preloading
        return batch_embedding 
    
    def close_files(self):
        """Close memory-mapped files if they are open."""
        if self.entity_mmap_file is not None:
            self.entity_mmap_file.close()
            self.entity_mmap_file = None
        if self.relation_mmap_file is not None:
            self.relation_mmap_file.close()
            self.relation_mmap_file = None
    
    def __del__(self):
        """Clean up memory-mapped files when object is destroyed."""
        if hasattr(self, 'entity_mmap_file') and self.entity_mmap_file is not None:
            self.entity_mmap_file.close()
        if hasattr(self, 'relation_mmap_file') and self.relation_mmap_file is not None:
            self.relation_mmap_file.close()