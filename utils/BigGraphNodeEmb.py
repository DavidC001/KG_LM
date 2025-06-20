import csv
import json
import os
from pathlib import Path
from typing import List, Dict

import h5py
import networkx as nx
import numpy as np
import torch
import wandb
from tqdm import tqdm
from torchbiggraph.converters.importers import TSVEdgelistReader
from torchbiggraph.graph_storages import FORMAT_VERSION_ATTR, FORMAT_VERSION
from torchbiggraph.config import ConfigSchema, EntitySchema, RelationSchema
from torchbiggraph.converters.importers import convert_input_data
from torchbiggraph.train import train
from torchbiggraph.util import SubprocessInitializer

from sentence_transformers import SentenceTransformer
from transformers import AutoConfig
from configuration import TRex_DatasetConfig
import socket


class BigGraphAligner:

    def __init__(self, graphs: Dict[str, nx.DiGraph], config: TRex_DatasetConfig = TRex_DatasetConfig(), batch_size: int = 64):
        
        self.config: TRex_DatasetConfig = config
        
        self.dataset_name: str = "lite" if self.config.lite else "full"
        
        self.folder: Path =  Path(self.config.graph_embs_base_path) / "big_graph_aligner" / self.config.graph_nodes_embedding_model / self.dataset_name
        os.makedirs(f'{self.folder}', exist_ok=True)
        
        self.batch_size: int = batch_size
        
        self.graphs: Dict[str, nx.DiGraph] = graphs
        
        self.epochs: int = self.config.big_graph_training_epochs
        self._use_untrained: bool = self.epochs == 0

        self.entity_index: Dict[str, np.array] = dict()   # Maps entity IDs to their embedding value
        self.relation_index: Dict[str, np.array] = dict() # Maps relation IDs to their embedding value

        # embedder configuration
        self.embedder: AutoConfig = AutoConfig.from_pretrained(self.config.graph_nodes_embedding_model)
        # embedder dimension
        self.embedder_dim: int = self.embedder.hidden_size
        
        self.prepare()
        self.train()
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
                "torch_dtype": torch.float16
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
        
        if not os.path.isfile(f'{self.folder}/init/embeddings_entity_0.v0.h5'):
            for i in tqdm(range(0, len(entity_labels), self.batch_size), desc='Encoding entity labels'):
                batch_labels = entity_labels[i:i+self.batch_size]
                batch_embeddings = embedder.encode(batch_labels, convert_to_tensor=True)
                # is any is nan breakpoint
                if torch.isnan(batch_embeddings).any():
                    breakpoint()
                all_embeddings.append(batch_embeddings.cpu())
            
            # Save Entity embeddings
            dataset = torch.cat(all_embeddings, dim=0)
            with h5py.File(f'{self.folder}/init/embeddings_entity_0.v0.h5', 'w') as hf:
                hf.create_dataset("embeddings", data=dataset.cpu().numpy(), dtype=np.float32)
                hf.attrs[FORMAT_VERSION_ATTR] = FORMAT_VERSION
        
        if not os.path.isfile(f'{self.folder}/init/embeddings_relation_0.v0.h5'):
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
                hf.create_dataset("embeddings", data=dataset.cpu().numpy(), dtype=np.float32)
                hf.attrs[FORMAT_VERSION_ATTR] = FORMAT_VERSION

        print("BigGraphAligner prepared successfully.")

    def train(self):
        if self._use_untrained:
            print("Using untrained BigGraphAligner")
            return

        if os.path.isfile(f'{self.folder}/model_checkpoint/embeddings_entity_0.v{self.epochs}.h5'):
            print("Already trained")
            return

        config = ConfigSchema(
            entity_path=str(self.folder),

            entities={
                "entity": EntitySchema(
                    num_partitions=1,
                ),
            },

            relations=[
                RelationSchema(
                    name="edge",
                    lhs="entity",
                    rhs="entity",
                    operator="diagonal",
                )
            ],

            init_path=f"{self.folder}/init",
            dynamic_relations=True,

            dimension=self.embedder_dim,  # Dimension of the embeddings
            global_emb=False,
            loss_fn="logistic",
            comparator="dot",
            workers=torch.get_num_threads() // 2,
            num_epochs=self.epochs,  # Number of training epochs
            batch_size=500,  # Batch size for training
            bias=True,
            num_uniform_negs=50,
            lr=0.01,
            edge_paths=[
                f"{self.folder}/edges"
            ],

            checkpoint_path=f"{self.folder}/model_checkpoint",  # Where to store model checkpoints
        )

        convert_input_data(
            entity_configs=config.entities,
            relation_configs=config.relations,
            entity_path=config.entity_path,
            edge_paths_out=config.edge_paths,
            edge_paths_in=[self.folder / "graph_edges.csv"],
            edgelist_reader=TSVEdgelistReader(
                lhs_col=0,
                rel_col=1,
                rhs_col=2,
                delimiter=','
            ),
            dynamic_relations=config.dynamic_relations
        )

        subprocess_init = SubprocessInitializer()
        train(config, subprocess_init=subprocess_init)

        wandb.init(
            project="5_align_graph",
            name=f"{self.config.graph_nodes_embedding_model}-{config.loss_fn}-{config.comparator}",

            # track hyperparameters and run metadata
            config={
                "llm": self.config.graph_nodes_embedding_model,
                "dataset": self.dataset_name,
                "dynamic_relations": config.dynamic_relations,
                "dimension": config.dimension,
                "global_emb": config.global_emb,
                "num_epochs": config.num_epochs,
                "batch_size": config.batch_size,
                "lr": config.lr,
                "num_uniform_negs": config.num_uniform_negs,
                "loss_fn": config.loss_fn,
                "comparator": config.comparator,
                "bias": config.bias,
            }
        )

        with open(f"{self.folder}/model_checkpoint/training_stats.json", "rt") as file:
            for i, line in tqdm(enumerate(file), desc='Reporting training statistics'):
                if i % 2 != 1:
                    continue
                data = json.loads(line.rstrip('\n'))
                print(data)

                epoch = data["epoch_idx"]

                loss = data["eval_stats_chunk_avg"]["metrics"]["loss"]
                wandb.log({"train/loss": loss, "epoch": epoch})

                pos_rank = data["eval_stats_chunk_avg"]["metrics"]["pos_rank"]
                wandb.log({"train/pos_rank": pos_rank, "epoch": epoch})

                mrr = data["eval_stats_chunk_avg"]["metrics"]["mrr"]
                wandb.log({"train/mrr": mrr, "epoch": epoch})

                r1 = data["eval_stats_chunk_avg"]["metrics"]["r1"]
                wandb.log({"train/r1": r1, "epoch": epoch})

                r10 = data["eval_stats_chunk_avg"]["metrics"]["r10"]
                wandb.log({"train/r10": r10, "epoch": epoch})

                r50 = data["eval_stats_chunk_avg"]["metrics"]["r50"]
                wandb.log({"train/r50": r50, "epoch": epoch})

                auc = data["eval_stats_chunk_avg"]["metrics"]["auc"]
                wandb.log({"train/auc": auc, "epoch": epoch})

        wandb.finish()

    def build_index(self):
        self.entity_index = {}
        self.relation_index = {}

        if self._use_untrained:
            with h5py.File(f'{self.folder}/init/embeddings_entity_0.v0.h5', 'r') as hf:
                trained_embeddings = hf['embeddings'][:]
        else:
            with h5py.File(f'{self.folder}/model_checkpoint/embeddings_entity_0.v{self.epochs}.h5', 'r') as hf:
                trained_embeddings = hf['embeddings'][:]
                
        with h5py.File(f'{self.folder}/init/embeddings_relation_0.v{self.epochs}.h5', 'r') as hf:
            trained_embeddings_rel = hf['embeddings'][:]

        total_entities = 0
        with open(f'{self.folder}/entities.csv', 'r') as file:
            reader = csv.reader(file)
            next(reader)  # Skip the header row
            for i, row in tqdm(enumerate(reader), desc='Building entity index'):
                entity_id, _ = row
                self.entity_index[entity_id] = torch.from_numpy(trained_embeddings[i, :])
                total_entities += 1

        assert trained_embeddings.shape == (
        total_entities, self.embedder_dim), "Trained embedding shape mismatch"

        cpu = torch.device("cpu")
        with open(f'{self.folder}/relations.csv', 'r') as file:
            reader = csv.reader(file)
            next(reader)  # Skip the header row
            for i, row in tqdm(enumerate(reader), desc='Building relation index'):
                relation_id, _ = row
                self.relation_index[relation_id] = torch.from_numpy(trained_embeddings_rel[i, :]).to(cpu)

        assert len(self.entity_index) > 0, "Entity index is empty"
        assert len(self.relation_index) > 0, "Relation index is empty"

    def node_embedding(self, entity_id: str) -> torch.Tensor:
        # check if NaN and break if so
        if entity_id not in self.entity_index:
            raise ValueError(f"Entity ID {entity_id} not found in the entity index.")
        
        # Return the embedding for the given entity ID
        if torch.isnan(self.entity_index[entity_id]).any():
            print(f"NaN found in entity ID {entity_id}, Substituting with zero vector.")
            return torch.zeros(self.embedder_dim)
            
        return self.entity_index.get(entity_id)

    def edge_embedding(self, predicate_id: str) -> torch.Tensor:
        return self.relation_index.get(predicate_id)

    def node_embedding_batch(self, entity_ids: List[str]) -> torch.Tensor:
        batch_embedding = torch.zeros((len(entity_ids), self.embedder_dim))
        for i, entity_id in enumerate(entity_ids):
            batch_embedding[i, :] = self.node_embedding(entity_id)
        return batch_embedding

    def edge_embedding_batch(self, predicate_ids: List[str]) -> torch.Tensor:
        batch_embedding = torch.zeros((len(predicate_ids), self.embedder_dim))
        for i, predicate_id in enumerate(predicate_ids):
            batch_embedding[i, :] = self.edge_embedding(predicate_id)
        return batch_embedding