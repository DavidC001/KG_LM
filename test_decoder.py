#!/usr/bin/env python3
"""
Test script for KG Decoder integration.
This script tests the KG decoder functionality with the KG-LFM model.
"""

import torch
import logging
from KG_LFM.configuration import load_yaml_config
from KG_LFM.model.KG_LFM_arch import KG_LFM, set_KGLM_model_args, KG_LFMConfig
from KG_LFM.model.KG_decoder import KGDecoder
from torch_geometric.data import Data, Batch

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_dummy_graph_batch(batch_size=2, num_nodes=5, embedding_dim=1024):
    """Create a dummy graph batch for testing."""
    graphs = []
    
    for i in range(batch_size):
        # Create random node features
        x = torch.randn(num_nodes, embedding_dim)
        
        # Create simple star graph: all nodes connect to central node (node 0)
        edge_index = torch.tensor([
            [0, 1, 0, 2, 0, 3, 0, 4, 1, 0, 2, 0, 3, 0, 4, 0],  # source nodes
            [1, 0, 2, 0, 3, 0, 4, 0, 0, 1, 0, 2, 0, 3, 0, 4]   # target nodes
        ], dtype=torch.long)
        
        # Create random edge features
        edge_attr = torch.randn(edge_index.shape[1], embedding_dim)
        
        graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        graphs.append(graph)
    
    return Batch.from_data_list(graphs)

def test_decoder_standalone():
    """Test the KG decoder standalone."""
    logger.info("Testing KG Decoder standalone...")
    
    # Parameters
    node_dim = 1024
    final_dim = 1024
    batch_size = 2
    num_quantizers = 3
    
    # Create decoder
    decoder = KGDecoder(
        node_embedding_dim=node_dim,
        edge_embedding_dim=node_dim,
        final_embedding_dim=final_dim,
        num_quantizers=num_quantizers,
        codebook_size=128,
        graph_pooling=True,
        max_nodes=20
    )
    
    # Create dummy inputs
    quantized_tokens = torch.randn(batch_size, num_quantizers, final_dim)
    quantized_indices = torch.randint(0, 128, (batch_size, num_quantizers))
    
    # Create dummy target features
    target_node_features = torch.randn(batch_size, 5, node_dim)
    target_edge_features = torch.randn(batch_size, 8, node_dim)
    
    # Forward pass
    with torch.no_grad():
        output = decoder(
            quantized_tokens=quantized_tokens,
            quantized_indices=quantized_indices,
            target_node_features=target_node_features,
            target_edge_features=target_edge_features
        )
    
    logger.info(f"Decoder output keys: {output.keys()}")
    logger.info(f"Reconstruction loss: {output['reconstruction_loss']}")
    logger.info("✓ Standalone decoder test passed!")

def test_model_integration():
    """Test the decoder integration with the full model."""
    logger.info("Testing KG Decoder integration with KG-LFM...")
    
    # Load configuration
    config_path = "configs/decoder_test_config.yaml"
    try:
        project_config = load_yaml_config(config_path)
    except FileNotFoundError:
        logger.warning(f"Config file {config_path} not found. Using minimal config for testing.")
        # Create minimal config for testing
        from KG_LFM.configuration import ModelConfig
        model_config = ModelConfig()
        model_config.use_kg_decoder = True
        model_config.tune_kg_decoder = True
        model_config.max_nodes = 20
        model_config.num_edge_types = 500
        model_config.graph_pooling = True
        model_config.tune_kg_encoder = True
        project_config = type('MockConfig', (), {'model': model_config})()
    
    # Create model config
    kg_lfm_config = KG_LFMConfig()
    kg_lfm_config = set_KGLM_model_args(kg_lfm_config, project_config.model)
    
    logger.info(f"Decoder enabled: {kg_lfm_config.use_kg_decoder}")
    logger.info(f"Max nodes: {kg_lfm_config.max_nodes}")
    
    # Create model (this might take a while to download)
    logger.info("Creating KG-LFM model...")
    try:
        model = KG_LFM(kg_lfm_config)
        logger.info("✓ Model created successfully!")
        
        # Check if decoder was initialized
        if model.get_kg_decoder() is not None:
            logger.info("✓ KG Decoder was initialized!")
        else:
            logger.warning("✗ KG Decoder was not initialized!")
            return
        
        # Create dummy inputs
        logger.info("Creating dummy inputs...")
        batch_size = 2
        seq_len = 10
        vocab_size = model.llm.config.vocab_size
        
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        # Add some special KG tokens
        input_ids[0, 3] = model.special_kg_token
        input_ids[1, 5] = model.special_kg_token
        
        attention_mask = torch.ones_like(input_ids)
        labels = input_ids.clone()
        
        # Create dummy graph batch
        graphs = create_dummy_graph_batch(batch_size)
        
        # Forward pass
        logger.info("Running forward pass...")
        model.eval()
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                graphs=graphs
            )
        
        logger.info(f"Output keys: {list(outputs.keys())}")
        logger.info(f"Main loss: {outputs.loss}")
        logger.info(f"RVQ loss: {outputs.get('RVQ_loss')}")
        logger.info(f"Decoder loss: {outputs.get('decoder_loss')}")
        
        logger.info("✓ Full model integration test passed!")
        
    except Exception as e:
        logger.error(f"Model integration test failed: {e}")
        import traceback
        traceback.print_exc()

def test_config_parameters():
    """Test that configuration parameters are properly set."""
    logger.info("Testing configuration parameters...")
    
    from KG_LFM.configuration import ModelConfig
    
    # Test ModelConfig with decoder parameters
    config = ModelConfig()
    config.use_kg_decoder = True
    config.tune_kg_decoder = True
    config.max_nodes = 50
    config.num_edge_types = 1000
    config.reconstruction_weight = 1.0
    config.structure_weight = 0.1
    
    logger.info(f"✓ use_kg_decoder: {config.use_kg_decoder}")
    logger.info(f"✓ tune_kg_decoder: {config.tune_kg_decoder}")
    logger.info(f"✓ max_nodes: {config.max_nodes}")
    logger.info(f"✓ num_edge_types: {config.num_edge_types}")
    logger.info(f"✓ reconstruction_weight: {config.reconstruction_weight}")
    logger.info(f"✓ structure_weight: {config.structure_weight}")
    
    # Test KG_LFMConfig with decoder parameters
    kg_config = KG_LFMConfig()
    kg_config = set_KGLM_model_args(kg_config, config)
    
    logger.info(f"✓ KG_LFMConfig.use_kg_decoder: {kg_config.use_kg_decoder}")
    logger.info(f"✓ KG_LFMConfig.max_nodes: {kg_config.max_nodes}")
    
    logger.info("✓ Configuration parameters test passed!")

if __name__ == "__main__":
    logger.info("Starting KG Decoder tests...")
    
    # Test 1: Configuration parameters
    test_config_parameters()
    
    # Test 2: Standalone decoder
    test_decoder_standalone()
    
    # Test 3: Full model integration (commented out as it requires model downloads)
    # Uncomment this if you want to test with actual models
    # test_model_integration()
    
    logger.info("All tests completed!")
