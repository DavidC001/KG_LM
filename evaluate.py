from KG_LFM.evaluator import KGLFMEvaluator
import argparse
import logging

def none_or_int(value):
    if value == 'None':
        return None
    return int(value)

def main():
    parser = argparse.ArgumentParser(description="Evaluate KG-LFM model")
    parser.add_argument("--config", type=str, default="configs/pretrain_config.yaml",
                       help="Path to the configuration file")
    parser.add_argument("--output_file", type=str, default="eval/out.json",
                       help="Path to save evaluation results JSON")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use for evaluation")
    parser.add_argument("--batch_size", type=int, default=16,
                       help="Batch size for evaluation")
    parser.add_argument("--max_samples", type=none_or_int, default=None,
                       help="Maximum number of samples to evaluate (for quick testing)")
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug mode")

    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.INFO if not args.debug else logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler() if args.debug else logging.FileHandler(f'logs/eval.log')
        ]
    )
    
    # Create evaluator
    evaluator = KGLFMEvaluator(
        config_path=args.config,
        batch_size=args.batch_size,
        max_samples=args.max_samples
    )
    
    # Run evaluation
    results = evaluator.evaluate(args.output_file)
    
    # Print summary
    evaluator.print_summary()


if __name__ == "__main__":
    main()
