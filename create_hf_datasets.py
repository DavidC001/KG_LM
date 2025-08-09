from KG_LFM.utils.Datasets.factories.factory import (
    trex_factory,
    trex_star_factory,
    trex_bite_factory,
    trirex_factory,
    trex_star_graphs_factory,
    web_qsp_factory,
)
from KG_LFM.configuration import ProjectConfig, load_yaml_config
from argparse import ArgumentParser
from pathlib import Path
from generate_webQSP import generate_webqsp

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
    
    print("Creating dataset webQSP...")
    base = Path(config.dataset.base_path)
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
    
    print("All datasets created successfully.")
    
    
if __name__ == "__main__":
    parser = ArgumentParser(description="Create Hugging Face datasets for Tri-REx and WebQSP.")
    parser.add_argument("--config", type=str, default="configs/pretrain_config.yaml", help="Path to YAML config file")
    parser.add_argument("--lite", action="store_true", help="Create lite dataset")
    parser.add_argument("--full", action="store_true", help="Create full dataset")

    args = parser.parse_args()

    # Load configuration
    if args.config:
        config = load_yaml_config(args.config)
    else:
        config = ProjectConfig()

    if args.lite:
        print(f"Creating lite datasets with config: {config.dataset}")
        config.dataset.lite = True
        main(config)

    if args.full:
        print(f"Creating full datasets with config: {config.dataset}")
        config.dataset.lite = False
        main(config)

    print("Dataset creation completed.")