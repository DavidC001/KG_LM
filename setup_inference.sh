#!/bin/bash
# Install dependencies for KG-LFM inference
echo "Installing inference dependencies..."

# Download spaCy English model
python -m spacy download en_core_web_sm

echo "Dependencies installed successfully!"
echo ""
echo "Usage:"
echo "python inference.py --model_path /path/to/your/trained/model --lite"
echo ""
echo "Optional arguments:"
echo "  --config: Path to dataset configuration YAML"
echo "  --lite: Use lite version of dataset (faster loading)"
echo "  --max_new_tokens: Maximum tokens to generate (default: 256)" 
echo "  --temperature: Sampling temperature (default: 0.7)"
echo "  --top_p: Top-p sampling (default: 0.9)"
echo "  --device: Device to use (auto/cpu/cuda/mps, default: auto)"
echo "  --host: Host address (default: 127.0.0.1)"
echo "  --port: Port number (default: 7860)"
echo "  --share: Create public Gradio link"
