# KG-LFM: Knowledge Grounded Language Foundation Model

A novel architecture that integrates Knowledge Graphs (KGs) with Large Language Models (LLMs) to enhance text generation with structured knowledge. The model combines a pre-trained language model with a specialized knowledge graph encoder using residual vector quantization.

## 🏗️ Architecture

KG-LFM introduces a unique multimodal approach where:

- **Knowledge Graphs** are encoded using Graph Attention Networks (GAT) with residual vector quantization
- **Special KG tokens** (`<KG_EMBEDDING>`) in text sequences are replaced with quantized graph embeddings
- **Seamless integration** between textual and graph representations in a unified embedding space

### Key Components

1. **Base Language Model**: Qwen3-8B (configurable)
2. **Graph Encoder**: GATv2Conv with global pooling and residual vector quantization
3. **Multimodal Fusion**: Dynamic replacement of KG tokens with graph embeddings
4. **Training Strategy**: LoRA fine-tuning with optional full model tuning

## 📊 Dataset

The model is primarily trained and evaluated on the **Tri-Rex dataset**, which provides:
- Factual statements with associated knowledge graph contexts
- Graph neighborhoods around central entities

Other datasets used are the test split of the web-QSP data mapped to WikiData entities, and the GrailQA dataset.

### Data preparation

To prepare the datasets, run the `create_hf_datasets.py` script with the appropriate configuration file.
The script will use the base path of the search for the following dataset files:
- TriRex_v1.tar (+lite) [webpage](https://zenodo.org/records/15166163)
- TRExStar_v1.tar (+lite) [webpage](https://zenodo.org/records/15165974)
- TrexBite_v1.tar (+lite) [webpage](https://zenodo.org/records/15165883)
- grailqa_v1.0_train.json & grailqa_v1.0_dev.json (the test set lacks fields needed for our preprocessing) from this [archive](https://dl.orangedox.com/WyaCpL?dl=1) 
- webqsp.examples.test.wikidata.json (used only for evaluation): you can get it from the folder `input` in the following [zip](https://public.ukp.informatik.tu-darmstadt.de/coling2018-graph-neural-networks-question-answering/WebQSP_WD_v1.zip) (link from [Github](https://github.com/UKPLab/coling2018-graph-neural-networks-question-answering/blob/master/WEBQSP_WD_README.md))
- [train](https://github.com/askplatypus/wikidata-simplequestions/raw/master/annotated_wd_data_train_answerable.txt), [val](https://github.com/askplatypus/wikidata-simplequestions/raw/master/annotated_wd_data_valid_answerable.txt) and [test](https://github.com/askplatypus/wikidata-simplequestions/raw/master/annotated_wd_data_test_answerable.txt) splits of the simple questions dataset mapped to WD.
These files need to be downloaded and placed in the base path provided.

mapping file from freebase to WD is from this [repo](https://github.com/askplatypus/wikidata-simplequestions/tree/master)

NOTE: some of the steps require internet access, if your compute nodes do not have it you can try to run the scipt for the lite version of the data, then you should be able to run the full version without internet access.

## 🔧 Configuration

The model behavior is controlled through YAML configuration files in the `configs/` directory

## 📈 Training & Evaluation

### Training Pipeline

1. **Data Preparation**: Graphs are embedded using sentence transformers
2. **Graph Encoding**: PyTorch BigGraph creates entity/relation embeddings
3. **Model Training**: KG-LFM learns to integrate graph and text representations
4. **Evaluation**: Multiple metrics assess generation quality and knowledge utilization

### Evaluation Metrics

- **Perplexity**: Language modeling performance
- **Hit@K**: Knowledge retrieval accuracy (K=1,3,5,10)
- **Comparative Analysis**: Performance vs baseline LLM and textualization approaches

### Distributed Training

The framework supports distributed training with:
- **Accelerate** for multi-GPU training
- **DeepSpeed** integration for memory optimization
- **Ray** for hyperparameter sweeping


## 🛠️ Key Scripts

| Script | Purpose |
|--------|---------|
| `train.py` | Main training loop with experiment tracking |
| `evaluate.py` | Comprehensive model evaluation |
| `embed_graphs.py` | Graph preprocessing and embedding generation |
| `create_hf_datasets.py` | Dataset preparation for HuggingFace format |
| `sweep.py` | Hyperparameter optimization with Ray/Optuna |

## 📁 Project Structure

```
KG_LM/
├── KG_LFM/                    # Core model implementation
│   ├── model/
│   │   ├── KG_LFM_arch.py     # Main model architecture
│   │   └── KG_encoder.py      # Graph encoder with RVQ
│   ├── utils/                 # Data loading and preprocessing
│   ├── trainer.py             # Training orchestration
│   ├── evaluator.py           # Evaluation framework
│   └── configuration.py       # Configuration management
├── configs/                   # YAML configuration files
├── launchers/                 # Shell scripts for common tasks
├── train.py                  # Main training script
├── evaluate.py               # Model evaluation script
├── embed_graphs.py           # Graph embedding generation
├── create_hf_datasets.py     # Dataset preparation for HuggingFace format
├── sweep.py                  # Hyperparameter optimization with Ray/Optuna
└── README.md                 # This file
```

## 🔬 Research Features

1. **Residual Vector Quantization for Graphs**: Efficient compression of graph embeddings
2. **Dynamic KG Token Replacement**: Seamless multimodal sequence processing
4. **Scalable Architecture**: Support for large graphs and models


## 📊 Performance

The model demonstrates improved performance over baseline approaches:

- **Enhanced Knowledge Grounding**: Better utilization of structured information
- **Improved Generation Quality**: Higher coherence in knowledge-intensive tasks compared to standard LLMs
- **Efficient Training**: LoRA enables resource-efficient fine-tuning
- **Scalable Inference**: Quantized embeddings reduce memory requirements compared to regular graph textualization

## 🙏 Acknowledgments

- Built on top of PyTorch and HuggingFace Transformers
- Uses PyTorch Geometric for graph neural networks
- Integrates PyTorch BigGraph for large-scale graph embeddings
- Tri-Rex dataset for knowledge-grounded evaluation