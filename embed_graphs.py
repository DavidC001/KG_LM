from KG_LFM.utils.BigGraphNodeEmb import BigGraphAligner
from KG_LFM.utils.Datasets.factories.factory import trex_star_graphs_factory, web_qsp_factory
from KG_LFM.configuration import load_yaml_config, ProjectConfig
from argparse import ArgumentParser

graph_build_functions= {
    "trirex": trex_star_graphs_factory,
    "trirex-bite": trex_star_graphs_factory,
    "web-qsp": lambda config: web_qsp_factory(config)[1]
}

def main(config: ProjectConfig, batch_size: int = 128):
    dataset_name = config.dataset.name
    
    print(f"Creating graphs for dataset {dataset_name}...")
    graphs = graph_build_functions[dataset_name](config.dataset)
    print(f"Total graphs in {dataset_name} dataset: {len(graphs)}")

    BigGraphAligner(
        graphs = graphs,
        config = config.dataset,
        batch_size = batch_size,
    )
    
    print("=" * 50)
    print("DONE!")
    
if __name__ == "__main__":
    parser = ArgumentParser(description="Embed TRExStar and WebQA graphs using BigGraphNodeEmb.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to the configuration file.")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for processing.")
    args = parser.parse_args()
    
    # Load configuration from the specified path
    config = load_yaml_config(args.config)
    
    main(config, batch_size=args.batch_size)