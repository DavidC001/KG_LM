#!/usr/bin/env python3
"""
Script to evaluate spillage between datasets.

This script checks what percentage of entity graphs in the test sets of various datasets
(GrailQA, SimpleQuestions, WebQSP) are contained in the train split of the TRiREx dataset.

This is important to understand potential data leakage and ensure fair evaluation.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Set, Dict, Tuple
from collections import defaultdict

import networkx as nx
from tqdm import tqdm
import pandas as pd

# Add the project root to the path
sys.path.append(str(Path(__file__).parent))

from KG_LFM.configuration import DatasetConfig
from KG_LFM.utils.Datasets.factories.factory import (
    trirex_factory,
    grailqa_factory, 
    simplequestions_factory,
    web_qsp_factory
)


def get_entity_ids_from_dataset(dataset) -> Set[str]:
    """Extract entity IDs from a dataset."""
    entity_ids = set()
    print(f"Extracting entity IDs from dataset with {len(dataset)} samples...")
    
    for sample in tqdm(dataset, desc="Processing samples"):
        if 'subject' in sample and 'id' in sample['subject']:
            entity_ids.add(sample['subject']['id'])
    
    return entity_ids


def get_graph_entity_ids(graphs: Dict[str, nx.DiGraph]) -> Set[str]:
    """Extract entity IDs from graph dictionary."""
    return set(graphs.keys())


def calculate_spillage(test_entities: Set[str], train_entities: Set[str]) -> Tuple[float, int, int]:
    """
    Calculate spillage percentage between test and train entity sets.
    
    Returns:
        - spillage_percentage: Percentage of test entities that appear in train
        - overlap_count: Number of overlapping entities
        - test_count: Total number of test entities
    """
    overlap = test_entities.intersection(train_entities)
    spillage_percentage = (len(overlap) / len(test_entities)) * 100 if test_entities else 0.0
    
    return spillage_percentage, len(overlap), len(test_entities)


def main():
    parser = argparse.ArgumentParser(description='Evaluate spillage between datasets')
    parser.add_argument('--base_path', type=str, 
                       default=os.path.join(os.getenv("FAST", ""), "dataset", "KG_LFM"),
                       help='Base path for datasets')
    parser.add_argument('--lite', action='store_true', 
                       help='Use lite versions of datasets for faster processing')
    parser.add_argument('--output', type=str, default='spillage_results.json',
                       help='Output file for results')
    
    args = parser.parse_args()
    
    print("🔍 Evaluating dataset spillage...")
    print(f"Base path: {args.base_path}")
    print(f"Using lite versions: {args.lite}")
    print("-" * 60)
    
    # Create dataset config
    config = DatasetConfig(
        base_path=args.base_path,
        lite=args.lite
    )
    
    results = {}
    
    # Step 1: Load TRiREx train split to get training entities
    print("📚 Loading TRiREx train split...")
    try:
        trirex_train, trirex_val, trirex_test = trirex_factory(config)
        trirex_train_entities = get_entity_ids_from_dataset(trirex_train)
        
        print(f"✅ TRiREx train entities: {len(trirex_train_entities)}")
        results['trirex_train_entities'] = len(trirex_train_entities)
        
        # Also check TRiREx test spillage (should be 0 if properly split)
        trirex_test_entities = get_entity_ids_from_dataset(trirex_test)
        trirex_spillage, trirex_overlap, trirex_test_count = calculate_spillage(
            trirex_test_entities, trirex_train_entities
        )
        
        results['trirex_test'] = {
            'spillage_percentage': trirex_spillage,
            'overlap_count': trirex_overlap,
            'test_count': trirex_test_count,
            'description': 'TRiREx test vs TRiREx train (should be ~0% for proper split)'
        }
        
        print(f"🔄 TRiREx test spillage: {trirex_spillage:.2f}% ({trirex_overlap}/{trirex_test_count})")
        
    except Exception as e:
        print(f"❌ Error loading TRiREx: {e}")
        return
    
    # Step 2: Evaluate other datasets
    datasets_to_evaluate = [
        ('grailqa', grailqa_factory, 'GrailQA'),
        ('simplequestions', simplequestions_factory, 'SimpleQuestions'), 
        ('webqsp', web_qsp_factory, 'WebQSP')
    ]
    
    for dataset_name, factory_func, display_name in datasets_to_evaluate:
        print(f"\n📊 Evaluating {display_name}...")
        
        try:
            if dataset_name == 'webqsp':
                # WebQSP returns (test_dataset, graphs)
                test_dataset, graphs = factory_func(config)
                test_entities = get_entity_ids_from_dataset(test_dataset)
                
                # Also check against graph entities
                graph_entities = get_graph_entity_ids(graphs) 
                graph_spillage, graph_overlap, graph_count = calculate_spillage(
                    graph_entities, trirex_train_entities
                )
                
                results[f'{dataset_name}_graphs'] = {
                    'spillage_percentage': graph_spillage,
                    'overlap_count': graph_overlap,
                    'test_count': graph_count,
                    'description': f'{display_name} graph entities vs TRiREx train'
                }
                
                print(f"  📈 Graph spillage: {graph_spillage:.2f}% ({graph_overlap}/{graph_count})")
                
            else:
                # GrailQA and SimpleQuestions return ((train, val, test), graphs)
                (train_dataset, val_dataset, test_dataset), graphs = factory_func(config)
                test_entities = get_entity_ids_from_dataset(test_dataset)
                
                # Also check against graph entities
                graph_entities = get_graph_entity_ids(graphs)
                graph_spillage, graph_overlap, graph_count = calculate_spillage(
                    graph_entities, trirex_train_entities
                )
                
                results[f'{dataset_name}_graphs'] = {
                    'spillage_percentage': graph_spillage,
                    'overlap_count': graph_overlap,
                    'test_count': graph_count,
                    'description': f'{display_name} graph entities vs TRiREx train'
                }
                
                print(f"  📈 Graph spillage: {graph_spillage:.2f}% ({graph_overlap}/{graph_count})")
            
            # Calculate spillage for test entities
            spillage, overlap, test_count = calculate_spillage(test_entities, trirex_train_entities)
            
            results[f'{dataset_name}_test'] = {
                'spillage_percentage': spillage,
                'overlap_count': overlap,
                'test_count': test_count,
                'description': f'{display_name} test entities vs TRiREx train'
            }
            
            print(f"  🎯 Test spillage: {spillage:.2f}% ({overlap}/{test_count})")
            
        except Exception as e:
            print(f"  ❌ Error evaluating {display_name}: {e}")
            results[f'{dataset_name}_error'] = str(e)
    
    # Step 3: Save results
    print(f"\n💾 Saving results to {args.output}...")
    
    # Add metadata
    results['metadata'] = {
        'base_path': args.base_path,
        'lite_version': args.lite,
        'description': 'Spillage analysis between dataset test sets and TRiREx train set'
    }
    
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Step 4: Print summary
    print("\n" + "="*60)
    print("📋 SPILLAGE SUMMARY")
    print("="*60)
    
    for key, value in results.items():
        if key.endswith('_test') or key.endswith('_graphs'):
            if isinstance(value, dict) and 'spillage_percentage' in value:
                print(f"{value['description']}: {value['spillage_percentage']:.2f}%")
    
    print(f"\n✅ Complete results saved to: {args.output}")


if __name__ == "__main__":
    main()
