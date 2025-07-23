import json
from typing import Tuple, Dict

import networkx as nx
from datasets import Dataset
from tqdm import tqdm

from KG_LFM.utils.Datasets.WebQSPFinetuneSentences import WebQSPFinetuneSentences
from KG_LFM.utils.Datasets.WebQSPFinetuneStar import WebQSPFinetuneStar
from KG_LFM.utils.Datasets.WebQSPSentences import WebQSPSentences
from KG_LFM.utils.Datasets.WebQSPStar import WebQSPStar
from KG_LFM.utils.Datasets.TRExBite import TRExBite
from KG_LFM.utils.Datasets.TRExBiteLite import TRExBiteLite
from KG_LFM.utils.Datasets.TriREx import TriREx
from KG_LFM.utils.Datasets.TriRExLite import TriRExLite
from KG_LFM.utils.Datasets.TREx import TREx
from KG_LFM.utils.Datasets.TRExLite import TRExLite
from KG_LFM.utils.Datasets.TRExStar import TRExStar
from KG_LFM.utils.Datasets.TRExStarLite import TRExStarLite

from KG_LFM.configuration import TRex_DatasetConfig

def trex_factory(conf: TRex_DatasetConfig) -> Tuple[Dataset, Dataset, Dataset]:
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


def trex_star_factory(conf: TRex_DatasetConfig) -> Dataset:
    if conf.lite:
        trex_star_builder = TRExStarLite(conf.base_path)
    else:
        trex_star_builder = TRExStar(conf.base_path)

    if not trex_star_builder.info.splits:
        trex_star_builder.download_and_prepare()

    return trex_star_builder.as_dataset(split="all")

def trex_star_graphs_factory(conf: TRex_DatasetConfig) -> Dict[str, nx.DiGraph]:
    dataset = trex_star_factory(conf)
    graphs = {}
    for datapoint in tqdm(dataset, desc="Loading nx graphs"):
        data = json.loads(datapoint['json'])
        graphs[datapoint['entity']] = nx.node_link_graph(data)
    return graphs


def trex_bite_factory(conf: TRex_DatasetConfig) -> Tuple[Dataset, Dataset, Dataset]:
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


def trirex_factory(conf: TRex_DatasetConfig) -> Tuple[Dataset, Dataset, Dataset]:
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



# TODO: Update the following functions to use the new configuration system
def web_qsp_factory() -> Tuple[Dataset, Dict[str, nx.DiGraph]]:
    web_qsp_sentence_builder = WebQSPSentences()
    web_qsp_star_builder = WebQSPStar()

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

def web_qsp_finetune_factory() -> Tuple[Dataset, Dataset, Dict[str, nx.DiGraph]]:
    web_qsp_sentence_builder = WebQSPFinetuneSentences()
    web_qsp_star_builder = WebQSPFinetuneStar()

    if not web_qsp_sentence_builder.info.splits:
        web_qsp_sentence_builder.download_and_prepare()

    if not web_qsp_star_builder.info.splits:
        web_qsp_star_builder.download_and_prepare()

    train_dataset = web_qsp_sentence_builder.as_dataset(split="train")
    test_dataset = web_qsp_sentence_builder.as_dataset(split="test")

    graphs = {}
    for datapoint in tqdm(web_qsp_star_builder.as_dataset(split="all"), desc="Loading nx graphs"):
        data = json.loads(datapoint['json'])
        graphs[datapoint['entity']] = nx.node_link_graph(data)

    return train_dataset, test_dataset, graphs