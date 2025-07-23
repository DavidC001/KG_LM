from KG_LFM.utils.Datasets.factory import trex_factory, trex_star_factory, trex_bite_factory, trirex_factory, trex_star_graphs_factory
from KG_LFM.configuration import load_yaml_config
from argparse import ArgumentParser

def main(config):
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
    
    
if __name__ == "__main__":
    parser = ArgumentParser(description="Create Hugging Face datasets for Tri-REx and TRExStar.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to the configuration file.")
    args = parser.parse_args()
    
    # Load configuration from the specified path
    config = load_yaml_config(args.config)
    
    print(f"Using dataset configuration: {config.dataset}")
    
    main(config)