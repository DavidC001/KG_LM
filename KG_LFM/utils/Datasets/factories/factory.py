import json
from typing import Tuple, Dict, Set

import networkx as nx
from datasets import Dataset
from tqdm import tqdm

from KG_LFM.utils.Datasets.factories.WebQSPSentences import WebQSPSentences
from KG_LFM.utils.Datasets.factories.WebQSPStar import WebQSPStar
from KG_LFM.utils.Datasets.factories.TRExBite import TRExBite
from KG_LFM.utils.Datasets.factories.TRExBiteLite import TRExBiteLite
from KG_LFM.utils.Datasets.factories.TriREx import TriREx
from KG_LFM.utils.Datasets.factories.TriRExLite import TriRExLite
from KG_LFM.utils.Datasets.factories.TREx import TREx
from KG_LFM.utils.Datasets.factories.TRExLite import TRExLite
from KG_LFM.utils.Datasets.factories.TRExStar import TRExStar
from KG_LFM.utils.Datasets.factories.TRExStarLite import TRExStarLite
from KG_LFM.utils.Datasets.factories.GrailQA import GrailQA
from KG_LFM.utils.Datasets.factories.GrailQAStar import GrailQAStar
from KG_LFM.utils.Datasets.factories.SimpleQuestionsSentences import SimpleQuestionsSentences
from KG_LFM.utils.Datasets.factories.SimpleQuestionsStar import SimpleQuestionsStar

from KG_LFM.configuration import DatasetConfig

# Global cache for TRiREx training entities to avoid reloading
_TRIREX_TRAIN_ENTITIES_CACHE = None


def get_trirex_train_entities(conf: DatasetConfig) -> Set[str]:
    """
    Get the set of entity IDs from TRiREx training split.
    Uses caching to avoid reloading for multiple factory calls.
    """
    global _TRIREX_TRAIN_ENTITIES_CACHE
    
    if _TRIREX_TRAIN_ENTITIES_CACHE is not None:
        return _TRIREX_TRAIN_ENTITIES_CACHE
    
    print("Loading TRiREx train entities for filtering...")
    
    # Load TRiREx train split
    if conf.lite:
        trirex_builder = TriRExLite(conf.base_path)
    else:
        trirex_builder = TriREx(conf.base_path)
    
    if not trirex_builder.info.splits:
        trirex_builder.download_and_prepare()
    
    train_dataset = trirex_builder.as_dataset(split="train")
    
    # Extract entity IDs
    entity_ids = set()
    for sample in tqdm(train_dataset, desc="Extracting TRiREx train entities"):
        if 'subject' in sample and 'id' in sample['subject']:
            entity_ids.add(sample['subject']['id'])
    
    _TRIREX_TRAIN_ENTITIES_CACHE = entity_ids
    print(f"Cached {len(entity_ids)} TRiREx training entities for filtering")
    
    return entity_ids

def trex_factory(conf: DatasetConfig) -> Tuple[Dataset, Dataset, Dataset]:
    if conf.lite:
        trex_builder = TRExLite()
    else:
        trex_builder = TREx()

    if not trex_builder.info.splits:
        trex_builder.download_and_prepare()

    train_dataset = trex_builder.as_dataset(split="train")

    validation_dataset = trex_builder.as_dataset(split="validation")

    test_dataset = trex_builder.as_dataset(split="test")
    return train_dataset, validation_dataset, test_dataset


def trex_star_factory(conf: DatasetConfig) -> Dataset:
    if conf.lite:
        trex_star_builder = TRExStarLite(conf.base_path)
    else:
        trex_star_builder = TRExStar(conf.base_path)

    if not trex_star_builder.info.splits:
        trex_star_builder.download_and_prepare()

    return trex_star_builder.as_dataset(split="all")

def trex_star_graphs_factory(conf: DatasetConfig) -> Dict[str, nx.DiGraph]:
    dataset = trex_star_factory(conf)
    graphs = {}
    for datapoint in tqdm(dataset, desc="Loading nx graphs"):
        data = json.loads(datapoint['json'])
        graphs[datapoint['entity']] = nx.node_link_graph(data)
    return graphs


def trex_bite_factory(conf: DatasetConfig) -> Tuple[Dataset, Dataset, Dataset]:
    if conf.lite:
        trex_bite_builder = TRExBiteLite(conf.base_path)
    else:
        trex_bite_builder = TRExBite(conf.base_path)

    if not trex_bite_builder.info.splits:
        trex_bite_builder.download_and_prepare()

    train_dataset = trex_bite_builder.as_dataset(split="train")
    validation_dataset = trex_bite_builder.as_dataset(split="validation")
    test_dataset = trex_bite_builder.as_dataset(split="test")
    return train_dataset, validation_dataset, test_dataset


def trirex_factory(conf: DatasetConfig) -> Tuple[Dataset, Dataset, Dataset]:
    if conf.lite:
        trirex_builder = TriRExLite(conf.base_path)
    else:
        trirex_builder = TriREx(conf.base_path)

    if not trirex_builder.info.splits:
        trirex_builder.download_and_prepare()

    train_dataset = trirex_builder.as_dataset(split="train")
    validation_dataset = trirex_builder.as_dataset(split="validation")
    test_dataset = trirex_builder.as_dataset(split="test")

    return train_dataset, validation_dataset, test_dataset


def web_qsp_factory(conf: DatasetConfig) -> Tuple[Dataset, Dict[str, nx.DiGraph]]:
    web_qsp_sentence_builder = WebQSPSentences(conf.base_path)
    web_qsp_star_builder = WebQSPStar(conf.base_path)

    if not web_qsp_sentence_builder.info.splits:
        web_qsp_sentence_builder.download_and_prepare()

    if not web_qsp_star_builder.info.splits:
        web_qsp_star_builder.download_and_prepare()

    test_dataset = web_qsp_sentence_builder.as_dataset(split="test")

    graphs = {}
    for datapoint in tqdm(web_qsp_star_builder.as_dataset(split="all"), desc="Loading nx graphs"):
        data = json.loads(datapoint['json'])
        graphs[datapoint['entity']] = nx.node_link_graph(data)

    return test_dataset, graphs


def grailqa_factory(conf: DatasetConfig) -> Tuple[Dataset, Dataset, Dataset, Dict[str, nx.DiGraph]]:
    """Factory function for GrailQA dataset with sentences and graphs."""
    grailqa_builder = GrailQA(conf.base_path)
    grailqa_star_builder = GrailQAStar(conf.base_path)

    if not grailqa_builder.info.splits:
        grailqa_builder.download_and_prepare()

    if not grailqa_star_builder.info.splits:
        grailqa_star_builder.download_and_prepare()

    train_dataset = grailqa_builder.as_dataset(split="train")
    validation_dataset = grailqa_builder.as_dataset(split="validation")
    test_dataset = grailqa_builder.as_dataset(split="test")

    # Load graphs
    graphs = {}
    for datapoint in tqdm(grailqa_star_builder.as_dataset(split="all"), desc="Loading GrailQA nx graphs"):
        data = json.loads(datapoint['json'])
        graphs[datapoint['entity']] = nx.node_link_graph(data)

    # filter out all samples from dataset which do not have a graph
    train_dataset = train_dataset.filter(lambda x: x['subject']['id'] in graphs)
    validation_dataset = validation_dataset.filter(lambda x: x['subject']['id'] in graphs)
    test_dataset = test_dataset.filter(lambda x: x['subject']['id'] in graphs)

    return (train_dataset, validation_dataset, test_dataset), graphs


def simplequestions_factory(conf: DatasetConfig) -> Tuple[Tuple[Dataset, Dataset, Dataset], Dict[str, nx.DiGraph]]:
    """Factory function for SimpleQuestions dataset with sentences and graphs."""
    simplequestions_builder = SimpleQuestionsSentences(conf.base_path)
    simplequestions_star_builder = SimpleQuestionsStar(conf.base_path)

    if not simplequestions_builder.info.splits:
        simplequestions_builder.download_and_prepare()

    if not simplequestions_star_builder.info.splits:
        simplequestions_star_builder.download_and_prepare()

    train_dataset = simplequestions_builder.as_dataset(split="train")
    validation_dataset = simplequestions_builder.as_dataset(split="validation")
    test_dataset = simplequestions_builder.as_dataset(split="test")

    # Load graphs
    graphs = {}
    for datapoint in tqdm(simplequestions_star_builder.as_dataset(split="all"), desc="Loading SimpleQuestions nx graphs"):
        data = json.loads(datapoint['json'])
        graphs[datapoint['entity']] = nx.node_link_graph(data)

    # filter out all samples from dataset which do not have a graph
    train_dataset = train_dataset.filter(lambda x: x['subject']['id'] in graphs)
    validation_dataset = validation_dataset.filter(lambda x: x['subject']['id'] in graphs)
    test_dataset = test_dataset.filter(lambda x: x['subject']['id'] in graphs)

    return (train_dataset, validation_dataset, test_dataset), graphs


# ===== FILTERED FACTORY FUNCTIONS =====
# These factory functions filter out entities that appear in TRiREx training data


def web_qsp_filtered_factory(conf: DatasetConfig) -> Tuple[Dataset, Dict[str, nx.DiGraph]]:
    """WebQSP factory that filters out entities present in TRiREx train split."""
    # Get original data
    test_dataset, graphs = web_qsp_factory(conf)
    
    # Get TRiREx training entities
    trirex_entities = get_trirex_train_entities(conf)
    
    # Filter dataset
    original_size = len(test_dataset)
    test_dataset = test_dataset.filter(lambda x: x.get('subject', {}).get('id', '') not in trirex_entities)
    filtered_size = len(test_dataset)
    
    # Filter graphs
    graphs_original_size = len(graphs)
    graphs = {entity_id: graph for entity_id, graph in graphs.items() if entity_id not in trirex_entities}
    graphs_filtered_size = len(graphs)
    
    print(f"WebQSP filtering: {original_size} -> {filtered_size} test samples ({original_size - filtered_size} removed)")
    print(f"WebQSP graph filtering: {graphs_original_size} -> {graphs_filtered_size} graphs ({graphs_original_size - graphs_filtered_size} removed)")
    
    return test_dataset, graphs


def simplequestions_filtered_factory(conf: DatasetConfig) -> Tuple[Tuple[Dataset, Dataset, Dataset], Dict[str, nx.DiGraph]]:
    """SimpleQuestions factory that filters out entities present in TRiREx train split."""
    # Get original data
    (train_dataset, validation_dataset, test_dataset), graphs = simplequestions_factory(conf)
    
    # Get TRiREx training entities
    trirex_entities = get_trirex_train_entities(conf)
    
    # Filter datasets
    original_train_size = len(train_dataset)
    train_dataset = train_dataset.filter(lambda x: x.get('subject', {}).get('id', '') not in trirex_entities)
    filtered_train_size = len(train_dataset)
    
    original_val_size = len(validation_dataset)
    validation_dataset = validation_dataset.filter(lambda x: x.get('subject', {}).get('id', '') not in trirex_entities)
    filtered_val_size = len(validation_dataset)
    
    original_test_size = len(test_dataset)
    test_dataset = test_dataset.filter(lambda x: x.get('subject', {}).get('id', '') not in trirex_entities)
    filtered_test_size = len(test_dataset)
    
    # Filter graphs
    graphs_original_size = len(graphs)
    graphs = {entity_id: graph for entity_id, graph in graphs.items() if entity_id not in trirex_entities}
    graphs_filtered_size = len(graphs)
    
    print(f"SimpleQuestions filtering:")
    print(f"  Train: {original_train_size} -> {filtered_train_size} samples ({original_train_size - filtered_train_size} removed)")
    print(f"  Val: {original_val_size} -> {filtered_val_size} samples ({original_val_size - filtered_val_size} removed)")
    print(f"  Test: {original_test_size} -> {filtered_test_size} samples ({original_test_size - filtered_test_size} removed)")
    print(f"  Graphs: {graphs_original_size} -> {graphs_filtered_size} graphs ({graphs_original_size - graphs_filtered_size} removed)")
    
    return (train_dataset, validation_dataset, test_dataset), graphs


def grailqa_filtered_factory(conf: DatasetConfig) -> Tuple[Tuple[Dataset, Dataset, Dataset], Dict[str, nx.DiGraph]]:
    """GrailQA factory that filters out entities present in TRiREx train split."""
    # Get original data
    (train_dataset, validation_dataset, test_dataset), graphs = grailqa_factory(conf)
    
    # Get TRiREx training entities
    trirex_entities = get_trirex_train_entities(conf)
    
    # Filter datasets
    original_train_size = len(train_dataset)
    train_dataset = train_dataset.filter(lambda x: x.get('subject', {}).get('id', '') not in trirex_entities)
    filtered_train_size = len(train_dataset)
    
    original_val_size = len(validation_dataset)
    validation_dataset = validation_dataset.filter(lambda x: x.get('subject', {}).get('id', '') not in trirex_entities)
    filtered_val_size = len(validation_dataset)
    
    original_test_size = len(test_dataset)
    test_dataset = test_dataset.filter(lambda x: x.get('subject', {}).get('id', '') not in trirex_entities)
    filtered_test_size = len(test_dataset)
    
    # Filter graphs
    graphs_original_size = len(graphs)
    graphs = {entity_id: graph for entity_id, graph in graphs.items() if entity_id not in trirex_entities}
    graphs_filtered_size = len(graphs)
    
    print(f"GrailQA filtering:")
    print(f"  Train: {original_train_size} -> {filtered_train_size} samples ({original_train_size - filtered_train_size} removed)")
    print(f"  Val: {original_val_size} -> {filtered_val_size} samples ({original_val_size - filtered_val_size} removed)")
    print(f"  Test: {original_test_size} -> {filtered_test_size} samples ({original_test_size - filtered_test_size} removed)")
    print(f"  Graphs: {graphs_original_size} -> {graphs_filtered_size} graphs ({graphs_original_size - graphs_filtered_size} removed)")
    
    return (train_dataset, validation_dataset, test_dataset), graphs
