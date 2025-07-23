from KG_LFM.utils.BigGraphNodeEmb import BigGraphAligner
from KG_LFM.utils.Datasets.factory import trex_star_graphs_factory
from KG_LFM.configuration import load_yaml_config, ProjectConfig
from argparse import ArgumentParser

def main(config: ProjectConfig, batch_size: int = 128):
    print("Creating dataset trex_star_graphs...")
    graphs = trex_star_graphs_factory(config.dataset)
    print(f"Total graphs in trex_star_graphs dataset: {len(graphs)}")
    
    aligner = BigGraphAligner(
        graphs = graphs,
        config = config.dataset,
        batch_size = batch_size,
    )
    
    print("=" * 50)
    print("DONE!")
    # breakpoint()
    
if __name__ == "__main__":
    parser = ArgumentParser(description="Create Hugging Face datasets for Tri-REx and TRExStar.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to the configuration file.")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for processing.")
    args = parser.parse_args()
    
    # Load configuration from the specified path
    config = load_yaml_config(args.config)
    
    main(config, batch_size=args.batch_size)