from KG_LFM.utils.Datasets.factories.factory import (
    trex_factory,
    trex_star_factory,
    trex_bite_factory,
    trirex_factory,
    trex_star_graphs_factory,
    web_qsp_factory,
    grailqa_factory,
    simplequestions_factory
)
from KG_LFM.configuration import ProjectConfig, load_yaml_config
from argparse import ArgumentParser
from pathlib import Path
from generate_webQSP import generate_webqsp
from generate_grailQA import generate_grailqa
from generate_simplequestions import generate_simplequestions
import logging

def main(config: ProjectConfig):
    """
    Main function to create Hugging Face datasets for Tri-REx and TRExStar.
    
    Args:
        config (ProjectConfig): Configuration object containing dataset settings.
    """
    print("Creating dataset trex...")
    train, val, test = trex_factory(config.dataset)
    print(f"Train dataset size: {len(train)}")
    print(f"Validation dataset size: {len(val)}")
    print(f"Test dataset size: {len(test)}")
    
    print("Creating dataset trex_bite...")
    train, val, test = trex_bite_factory(config.dataset)
    print(f"Train bite dataset size: {len(train)}")
    print(f"Validation bite dataset size: {len(val)}")
    print(f"Test bite dataset size: {len(test)}")
    
    print("Creating dataset trex_star...")
    data = trex_star_factory(config.dataset)
    print(f"Total records in trex_star dataset: {len(data)}")
    
    print("Creating dataset trex_star_graphs...")
    graphs = trex_star_graphs_factory(config.dataset)
    print(f"Total graphs in trex_star_graphs dataset: {len(graphs)}")
    
    print("Creating dataset trirex...")
    train, val, test = trirex_factory(config.dataset)
    print(f"Train trirex dataset size: {len(train)}")
    print(f"Validation trirex dataset size: {len(val)}")
    print(f"Test trirex dataset size: {len(test)}")
    
    
    base = Path(config.dataset.base_path)
    
    
    print("Creating dataset webQSP...")
    sent_tar = base / 'WebQSP_sentences_v1' / 'publish' / 'WebQSP_sentences_v1.tar'
    star_tar = base / 'WebQSP_star_v1' / 'publish' / 'WebQSP_star_v1.tar'
    # If missing, try generating from raw file; otherwise skip gracefully
    if not (sent_tar.exists() and star_tar.exists()):
        print("WebQSP artifacts not found; attempting generation from raw file...")
        ok = generate_webqsp(base_path=base, version=1)
        if not ok:
            print(
                f"WebQSP raw file missing at {base/'webqsp.examples.test.wikidata.json'}. Skipping WebQSP dataset creation."
            )
        else:
            print("WebQSP artifacts generated.")
    # Only try to load if artifacts exist
    if sent_tar.exists() and star_tar.exists():
        test, graphs = web_qsp_factory(config.dataset)
        print(f"Total records in webQSP dataset: {len(test)}")
        print(f"Total graphs in webQSP dataset: {len(graphs)}")
    else:
        print("WebQSP dataset not created due to missing artifacts.")
    
        
    print("Creating dataset simplequestions...")
    sent_tar = base / 'SimpleQuestions_sentences_v1' / 'publish' / 'SimpleQuestions_sentences_v1.tar'
    star_tar = base / 'SimpleQuestions_star_v1' / 'publish' / 'SimpleQuestions_star_v1.tar'
    if not (sent_tar.exists() and star_tar.exists()):
        print("SimpleQuestions artifacts not found; attempting generation from raw files...")
        ok = generate_simplequestions(base_path=base, version=1)
        if not ok:
            print(
                f"SimpleQuestions raw files missing. Expected files: annotated_wd_data_{{train,valid,test}}.txt. "
                f"Download from https://github.com/askplatypus/wikidata-simplequestions and place in {base}. "
                "Skipping SimpleQuestions dataset creation."
            )
        else:
            print("SimpleQuestions artifacts generated.")
    # Only try to load if artifacts exist
    if sent_tar.exists() and star_tar.exists():
        (train, val, test), graphs = simplequestions_factory(config.dataset)
        print(f"Train simplequestions dataset size: {len(train)}")
        print(f"Validation simplequestions dataset size: {len(val)}")
        print(f"Test simplequestions dataset size: {len(test)}")
        print(f"Total graphs in simplequestions dataset: {len(graphs)}")
    else:
        print("SimpleQuestions dataset not created due to missing artifacts.")
    
    
    print("Creating dataset grailqa...")
    sent_tar = base / 'GrailQA_sentences_v1' / 'publish' / 'GrailQA_sentences_v1.tar'
    star_tar = base / 'GrailQA_star_v1' / 'publish' / 'GrailQA_star_v1.tar'
    if not (sent_tar.exists() and star_tar.exists()):
        print("GrailQA artifacts not found; attempting generation from raw file...")
        ok = generate_grailqa(base_path=base, version=1)
        if not ok:
            print(
                f"GrailQA raw file missing at {base/'grailqa.examples.test.wikidata.json'}. Skipping GrailQA dataset creation."
            )
        else:
            print("GrailQA artifacts generated.")
    # Only try to load if artifacts exist
    if sent_tar.exists() and star_tar.exists():
        print("OK")
        (train, val, test), graphs = grailqa_factory(config.dataset)
        print(f"Train grailqa dataset size: {len(train)}")
        print(f"Validation grailqa dataset size: {len(val)}")
        print(f"Test grailqa dataset size: {len(test)}")
    else:
        print("GrailQA dataset not created due to missing artifacts.")

    print("All datasets created successfully.")
    
    
if __name__ == "__main__":
    parser = ArgumentParser(description="Create Hugging Face datasets for Tri-REx and WebQSP.")
    parser.add_argument("--config", type=str, default="configs/base_config.yaml", help="Path to YAML config file")
    parser.add_argument("--lite", action="store_true", help="Create lite dataset")
    parser.add_argument("--full", action="store_true", help="Create full dataset")

    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
        ]
    )

    # Load configuration
    if args.config:
        config = load_yaml_config(args.config)
    else:
        config = ProjectConfig()
        
    if not (args.lite or args.full):
        print("Please specify either --lite or --full to create datasets.")
        exit(1)

    if args.lite:
        print(f"Creating lite datasets with config: {config.dataset}")
        config.dataset.lite = True
        main(config)

    if args.full:
        print(f"Creating full datasets with config: {config.dataset}")
        config.dataset.lite = False
        main(config)

    print("Dataset creation completed.")