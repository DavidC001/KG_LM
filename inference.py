#!/usr/bin/env python3
"""
KG-LFM Inference Script with Chat UI and Entity Recognition

This script provides a chat interface for the KG-LFM model with integrated entity recognition
to automatically retrieve relevant knowledge graphs for user queries.
"""

import argparse
import json
import logging
import os
import re
import sys
from typing import Dict, List, Optional, Set, Tuple, Any
import warnings
from datetime import datetime

import torch
import torch.nn.functional as F
import networkx as nx
import gradio as gr
from transformers import AutoTokenizer, StoppingCriteria, StoppingCriteriaList
import spacy
from spacy.matcher import Matcher
from fuzzywuzzy import fuzz, process

# Import your model components
from KG_LFM.model.KG_LFM_arch import KG_LFM, KG_LFMConfig
from KG_LFM.utils.Datasets.factory import trex_star_graphs_factory
from KG_LFM.utils.BigGraphNodeEmb import BigGraphAligner
from KG_LFM.configuration import DatasetConfig, SPECIAL_KG_TOKEN
from torch_geometric.data import Data, Batch

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EntityRecognizer:
    """
    Entity recognition and linking component that maps user mentions to knowledge graph entities.
    """
    
    def __init__(self, entity_graphs: Dict[str, nx.DiGraph], fuzzy_threshold: int = 80):
        """
        Initialize entity recognizer.
        
        Args:
            entity_graphs: Dictionary of entity IDs to their corresponding graphs
            fuzzy_threshold: Minimum similarity score for fuzzy matching (0-100)
        """
        self.entity_graphs = entity_graphs
        self.fuzzy_threshold = fuzzy_threshold
        
        # Load spaCy model for NER
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("spaCy English model not found. Installing...")
            os.system("python -m spacy download en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm")
        
        # Create entity name mappings from graph data
        self._build_entity_mappings()
        
        # Set up custom matcher for entity patterns
        self.matcher = Matcher(self.nlp.vocab)
        self._setup_custom_patterns()
    
    def _build_entity_mappings(self):
        """Build mapping from entity names/labels to entity IDs."""
        self.entity_to_id = {}
        self.entity_labels = set()
        
        logger.info("Building entity mappings from knowledge graphs...")
        
        for entity_id, graph in self.entity_graphs.items():
            # Extract entity labels from graph nodes
            for node_id, node_data in graph.nodes(data=True):
                if 'label' in node_data:
                    label = node_data['label'].lower().strip()
                    if label:
                        self.entity_to_id[label] = entity_id
                        self.entity_labels.add(label)
            
            # Also use entity ID itself as a potential match
            if isinstance(entity_id, str):
                entity_name = entity_id.lower().replace('_', ' ').replace('-', ' ')
                self.entity_to_id[entity_name] = entity_id
                self.entity_labels.add(entity_name)
        
        logger.info(f"Built mappings for {len(self.entity_labels)} entity labels")
    
    def _setup_custom_patterns(self):
        """Set up custom patterns for entity matching."""
        # Add patterns for common entity types
        patterns = [
            [{"LOWER": {"IN": ["president", "ceo", "director", "minister"]}}],
            [{"ENT_TYPE": "PERSON"}],
            [{"ENT_TYPE": "ORG"}],
            [{"ENT_TYPE": "GPE"}],  # Geopolitical entities
            [{"ENT_TYPE": "EVENT"}],
            [{"ENT_TYPE": "PRODUCT"}],
            [{"ENT_TYPE": "WORK_OF_ART"}],
        ]
        
        for i, pattern in enumerate(patterns):
            self.matcher.add(f"ENTITY_PATTERN_{i}", [pattern])
    
    def extract_entities(self, text: str) -> List[Tuple[str, str, float]]:
        """
        Extract and link entities from text.
        
        Args:
            text: Input text to analyze
            
        Returns:
            List of tuples (entity_mention, entity_id, confidence_score)
        """
        doc = self.nlp(text)
        entities = []
        
        # Extract named entities using spaCy
        spacy_entities = [(ent.text.lower().strip(), ent.label_) for ent in doc.ents]
        
        # Extract custom pattern matches
        matches = self.matcher(doc)
        custom_entities = [(doc[start:end].text.lower().strip(), "CUSTOM") 
                          for match_id, start, end in matches]
        
        # Combine all candidate entities
        all_candidates = spacy_entities + custom_entities
        
        for entity_text, entity_type in all_candidates:
            if not entity_text or len(entity_text) < 2:
                continue
                
            # Direct match
            if entity_text in self.entity_to_id:
                entities.append((entity_text, self.entity_to_id[entity_text], 1.0))
                continue
            
            # Fuzzy matching
            matches = process.extract(
                entity_text, 
                self.entity_labels, 
                limit=3, 
                scorer=fuzz.ratio
            )
            
            for match_text, score in matches:
                if score >= self.fuzzy_threshold:
                    confidence = score / 100.0
                    entity_id = self.entity_to_id[match_text]
                    entities.append((entity_text, entity_id, confidence))
                    break
        
        # Remove duplicates and sort by confidence
        unique_entities = {}
        for mention, entity_id, confidence in entities:
            key = (mention, entity_id)
            if key not in unique_entities or unique_entities[key][2] < confidence:
                unique_entities[key] = (mention, entity_id, confidence)
        
        result = list(unique_entities.values())
        result.sort(key=lambda x: x[2], reverse=True)
        
        return result

class KGChatBot:
    """
    Main chatbot class that integrates KG-LFM model with entity recognition.
    """
    
    def __init__(
        self,
        model_path: str,
        dataset_config: DatasetConfig,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        device: str = "auto"
    ):
        """
        Initialize the KG-LFM chatbot.
        
        Args:
            model_path: Path to the trained KG-LFM model
            dataset_config: Configuration for loading knowledge graphs
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            device: Device to run the model on
        """
        self.device = self._setup_device(device)
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        
        logger.info("Loading KG-LFM model...")
        self.model = self._load_model(model_path)
        
        logger.info("Loading knowledge graphs...")
        self.entity_graphs = trex_star_graphs_factory(dataset_config)
        
        logger.info("Setting up graph aligner...")
        self.graph_aligner = BigGraphAligner(
            graphs=self.entity_graphs,
            config=dataset_config
        )
        
        logger.info("Initializing entity recognizer...")
        self.entity_recognizer = EntityRecognizer(self.entity_graphs)
        
        # Set up stopping criteria
        self.stopping_criteria = self._setup_stopping_criteria()
        
        logger.info("KG-LFM chatbot initialized successfully!")
    
    def _setup_device(self, device: str) -> torch.device:
        """Setup computation device."""
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        
        logger.info(f"Using device: {device}")
        return torch.device(device)
    
    def _load_model(self, model_path: str) -> KG_LFM:
        """Load the trained KG-LFM model."""
        try:
            model = KG_LFM.from_pretrained(model_path)
            model.to(self.device)
            model.eval()
            return model
        except Exception as e:
            logger.error(f"Failed to load model from {model_path}: {e}")
            raise
    
    def _setup_stopping_criteria(self) -> StoppingCriteriaList:
        """Setup stopping criteria for generation."""
        class CustomStoppingCriteria(StoppingCriteria):
            def __init__(self, tokenizer):
                self.tokenizer = tokenizer
                
            def __call__(self, input_ids, scores, **kwargs) -> bool:
                # Stop if we hit the EOS token
                if input_ids[0, -1] == self.tokenizer.eos_token_id:
                    return True
                return False
        
        return StoppingCriteriaList([CustomStoppingCriteria(self.model.tokenizer)])
    
    def _prepare_graph_input(self, entity_ids: List[str]) -> Optional[Batch]:
        """
        Prepare graph input from entity IDs.
        
        Args:
            entity_ids: List of entity IDs to include
            
        Returns:
            Batched graph data or None if no valid graphs
        """
        if not entity_ids:
            return None
        
        graphs = []
        for entity_id in entity_ids[:3]:  # Limit to 3 entities to avoid memory issues
            if entity_id in self.entity_graphs:
                graph_nx = self.entity_graphs[entity_id]
                graph_data = self._networkx_to_torch_geometric(graph_nx, entity_id)
                if graph_data is not None:
                    graphs.append(graph_data)
        
        if not graphs:
            return None
        
        return Batch.from_data_list(graphs)
    
    def _networkx_to_torch_geometric(self, graph: nx.DiGraph, central_entity: str) -> Optional[Data]:
        """
        Convert NetworkX graph to PyTorch Geometric format.
        
        Args:
            graph: NetworkX DiGraph
            central_entity: Central entity ID
            
        Returns:
            PyTorch Geometric Data object
        """
        try:
            # Extract graph structure (similar to pretrain_data.py)
            neighbour_ids, edge_ids = [], []
            central_node_id = None
            neighbour_node_labels, edge_labels = [], []
            central_node_label = None
            
            nodes = graph.nodes(data=True)
            
            # Find edges and build node/edge lists
            for central_node_id, neighbour_node_id, edge in graph.edges(data=True):
                edge_ids.append(edge['id'])
                neighbour_ids.append(neighbour_node_id)
                
                if central_node_id is None:
                    central_node_id = central_node_id
                    central_node_label = nodes[central_node_id]['label']
                
                edge_labels.append(edge['label'])
                neighbour_node_labels.append(nodes[neighbour_node_id]['label'])
            
            if not neighbour_ids:
                return None
            
            # Get embeddings
            central_node_emb = self.graph_aligner.node_embedding(central_node_id)
            neighbour_node_embs = self.graph_aligner.node_embedding_batch(neighbour_ids)
            edge_embs = self.graph_aligner.edge_embedding_batch(edge_ids)
            
            # Create edge index for star graph
            num_neighbors = len(neighbour_ids)
            edge_index = torch.tensor([
                list(range(1, num_neighbors + 1)) + [0] * num_neighbors,
                [0] * num_neighbors + list(range(1, num_neighbors + 1))
            ], dtype=torch.long)
            
            # Node features: central node + neighbor nodes
            node_features = torch.cat([
                central_node_emb.unsqueeze(0),
                neighbour_node_embs
            ], dim=0)
            
            # Edge features: duplicate for bidirectional edges
            edge_features = torch.cat([edge_embs, edge_embs], dim=0)
            
            return Data(
                x=node_features,
                edge_index=edge_index,
                edge_attr=edge_features,
                num_nodes=node_features.shape[0],
                central_node_label=central_node_label,
                neighbour_node_labels=neighbour_node_labels,
                edge_labels=edge_labels,
            )
        
        except Exception as e:
            logger.warning(f"Failed to convert graph for entity {central_entity}: {e}")
            return None
    
    def _prepare_text_with_kg_tokens(self, text: str, num_entities: int) -> str:
        """
        Prepare text by inserting KG tokens where entities are mentioned.
        
        Args:
            text: Input text
            num_entities: Number of entities to insert tokens for
            
        Returns:
            Text with KG tokens inserted
        """
        # For now, just add KG tokens at the beginning
        # In a more sophisticated version, you could insert them at entity positions
        kg_tokens = SPECIAL_KG_TOKEN * num_entities
        return kg_tokens + " " + text
    
    def generate_response(
        self, 
        user_input: str, 
        conversation_history: List[Dict[str, str]] = None,
        max_entities: int = 2
    ) -> Tuple[str, List[Tuple[str, str, float]]]:
        """
        Generate a response to user input using the KG-LFM model.
        
        Args:
            user_input: User's input text
            conversation_history: Previous conversation messages
            max_entities: Maximum number of entities to extract
            
        Returns:
            Tuple of (generated_response, extracted_entities)
        """
        # Extract entities from user input
        entities = self.entity_recognizer.extract_entities(user_input)
        entities = entities[:max_entities]  # Limit number of entities
        
        logger.info(f"Extracted entities: {[(e[0], e[1], f'{e[2]:.2f}') for e in entities]}")
        
        # Prepare conversation context
        if conversation_history:
            # Format conversation history
            context_parts = []
            for msg in conversation_history[-5:]:  # Keep last 5 messages
                role = msg.get("role", "user")
                content = msg.get("content", "")
                context_parts.append(f"{role}: {content}")
            context = "\n".join(context_parts) + f"\nuser: {user_input}\nassistant:"
        else:
            context = f"user: {user_input}\nassistant:"
        
        # Prepare text with KG tokens if entities found
        if entities:
            entity_ids = [e[1] for e in entities]
            text_input = self._prepare_text_with_kg_tokens(context, len(entity_ids))
            graphs = self._prepare_graph_input(entity_ids)
        else:
            text_input = context
            graphs = None
        
        # Tokenize input
        with torch.no_grad():
            # Apply chat template if available
            if hasattr(self.model.tokenizer, 'apply_chat_template'):
                messages = [{"role": "user", "content": user_input}]
                text_input = self.model.tokenizer.apply_chat_template(
                    messages, 
                    tokenize=False, 
                    add_generation_prompt=True
                )
                if entities:
                    # Insert KG tokens at the beginning for now
                    text_input = SPECIAL_KG_TOKEN * len(entities) + " " + text_input
            
            inputs = self.model.tokenizer(
                text_input,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=1024
            ).to(self.device)
            
            if graphs is not None:
                graphs = graphs.to(self.device)
            
            # Generate response
            try:
                with torch.inference_mode():
                    output = self.model.generate(
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                        graphs=graphs,
                        max_new_tokens=self.max_new_tokens,
                        temperature=self.temperature,
                        top_p=self.top_p,
                        do_sample=True,
                        pad_token_id=self.model.tokenizer.pad_token_id,
                        eos_token_id=self.model.tokenizer.eos_token_id,
                        stopping_criteria=self.stopping_criteria,
                    )
                
                # Decode the response
                input_length = inputs["input_ids"].shape[1]
                generated_tokens = output[0][input_length:]
                response = self.model.tokenizer.decode(generated_tokens, skip_special_tokens=True)
                
                # Clean up the response
                response = response.strip()
                if response.startswith("assistant:"):
                    response = response[10:].strip()
                
                return response, entities
                
            except Exception as e:
                logger.error(f"Generation failed: {e}")
                return f"I apologize, but I encountered an error while generating a response: {str(e)}", entities

def create_gradio_interface(chatbot: KGChatBot) -> gr.Interface:
    """
    Create a Gradio chat interface for the KG-LFM model.
    
    Args:
        chatbot: Initialized KGChatBot instance
        
    Returns:
        Gradio interface
    """
    
    def chat_fn(message: str, history: List[List[str]]) -> Tuple[str, List[List[str]]]:
        """Process chat message and return response."""
        try:
            # Convert history to the format expected by the model
            conversation_history = []
            for user_msg, bot_msg in history:
                conversation_history.append({"role": "user", "content": user_msg})
                if bot_msg:
                    conversation_history.append({"role": "assistant", "content": bot_msg})
            
            # Generate response
            response, entities = chatbot.generate_response(message, conversation_history)
            
            # Add entity information to response if entities were found
            if entities:
                entity_info = "\n\n🔍 **Detected entities:** " + ", ".join([
                    f"{e[0]} ({e[2]:.0%})" for e in entities[:3]
                ])
                response += entity_info
            
            # Update history
            history.append([message, response])
            
            return "", history
            
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            history.append([message, error_msg])
            return "", history
    
    # Create the chat interface
    with gr.Blocks(title="KG-LFM Chat", theme=gr.themes.Soft()) as interface:
        gr.Markdown("""
        # 🧠 KG-LFM Knowledge Graph Chat
        
        Chat with a knowledge graph-enhanced language model! The model can understand entities in your questions 
        and use relevant knowledge graph information to provide more informed responses.
        
        **Features:**
        - 🔍 Automatic entity recognition and linking
        - 📊 Knowledge graph integration
        - 💬 Conversational interface
        """)
        
        chatbot_ui = gr.Chatbot(
            label="Chat History",
            height=500,
            show_copy_button=True,
            placeholder="Start chatting! Ask questions about entities, people, places, or any topic."
        )
        
        with gr.Row():
            msg_box = gr.Textbox(
                label="Your message",
                placeholder="Type your message here...",
                lines=2,
                scale=4
            )
            submit_btn = gr.Button("Send", variant="primary", scale=1)
        
        with gr.Row():
            clear_btn = gr.Button("Clear Chat", variant="secondary")
        
        # Event handlers
        submit_btn.click(
            chat_fn,
            inputs=[msg_box, chatbot_ui],
            outputs=[msg_box, chatbot_ui]
        )
        
        msg_box.submit(
            chat_fn,
            inputs=[msg_box, chatbot_ui],
            outputs=[msg_box, chatbot_ui]
        )
        
        clear_btn.click(lambda: ([], ""), outputs=[chatbot_ui, msg_box])
        
        # Add examples
        gr.Examples(
            examples=[
                "Tell me about Barack Obama",
                "What do you know about Einstein's theory of relativity?",
                "Who was the first person to walk on the moon?",
                "What is the capital of France?",
                "Tell me about the company Apple",
            ],
            inputs=msg_box,
            label="Example questions"
        )
        
        gr.Markdown("""
        ---
        **Note:** The model uses entity recognition to identify relevant entities in your questions 
        and retrieves corresponding knowledge graphs to enhance responses.
        """)
    
    return interface

def main():
    """Main function to run the inference script."""
    parser = argparse.ArgumentParser(description="KG-LFM Inference with Chat UI")
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to the trained KG-LFM model")
    parser.add_argument("--config", type=str, default=None,
                       help="Path to dataset configuration YAML file")
    parser.add_argument("--lite", action="store_true",
                       help="Use lite version of the dataset")
    parser.add_argument("--max_new_tokens", type=int, default=256,
                       help="Maximum number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7,
                       help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.9,
                       help="Top-p sampling parameter")
    parser.add_argument("--device", type=str, default="auto",
                       choices=["auto", "cpu", "cuda", "mps"],
                       help="Device to run the model on")
    parser.add_argument("--host", type=str, default="127.0.0.1",
                       help="Host to serve the Gradio interface on")
    parser.add_argument("--port", type=int, default=7860,
                       help="Port to serve the Gradio interface on")
    parser.add_argument("--share", action="store_true",
                       help="Create a public link for the Gradio interface")
    
    args = parser.parse_args()
    
    # Set up dataset configuration
    if args.config:
        from KG_LFM.configuration import load_yaml_config
        config = load_yaml_config(args.config)
        dataset_config = config.dataset
    else:
        dataset_config = DatasetConfig(lite=args.lite)
    
    try:
        # Initialize the chatbot
        logger.info("Initializing KG-LFM chatbot...")
        chatbot = KGChatBot(
            model_path=args.model_path,
            dataset_config=dataset_config,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            device=args.device
        )
        
        # Create and launch the interface
        logger.info("Creating Gradio interface...")
        interface = create_gradio_interface(chatbot)
        
        logger.info(f"Launching interface on {args.host}:{args.port}")
        interface.launch(
            server_name=args.host,
            server_port=args.port,
            share=args.share
        )
        
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    except Exception as e:
        logger.error(f"Failed to start inference server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
