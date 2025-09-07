**This repository is the implementation of RuleNet: rule-priority-aware multi-agent trajectory prediction in ambiguous traffic scenarios published in Transportation Research Part C: Emerging Technologies.**

RuleNet is a specialized framework for safety-aware multi-agent trajectory prediction. It combines graph attention networks with rule-based robustness attention mechanisms to simultaneously predict future trajectories of multiple traffic participants while evaluating trajectory safety.

## 📋 Requirements

- Python >= 3.8
- PyTorch >= 2.1.0
- PyTorch Geometric >= 2.3.1
- PyTorch Lightning >= 2.0.3
- lanelet2 >= 1.1.0
- numpy, pandas, tqdm

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/Ruolin99/RuleNet.git
cd RuleNet
```

> **Note**: This implementation is tested on Ubuntu 20.04 with CUDA 12.4

### 2. Install Dependencies

```bash
# Create conda environment
conda create -n RuleNet python=3.8
conda activate RuleNet

# Install PyTorch with CUDA support
conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=12.1 -c pytorch -c nvidia

# Install PyTorch Geometric
pip install torch_geometric==2.3.1

# Install PyTorch Lightning
conda install pytorch-lightning==2.0.3

# Install additional dependencies
pip install pandas numpy tqdm scipy shapely networkx

# Install lanelet2 (system-level installation required)
# Ubuntu/Debian:
sudo apt-get install liblanelet2-dev python3-lanelet2

# Or using conda:
conda install -c conda-forge lanelet2
```

### 3. Data Preparation

Download the INTERACTION dataset from the [official website](https://interaction-dataset.com/).

Data directory structure:

```
/path/to/INTERACTION_root/
├── maps/
├── test_conditional-multi-agent/
├── test_multi-agent/
├── train/
│   ├── DR_CHN_Merging_ZS0_train
│   ├── ...
└── val/
    ├── DR_CHN_Merging_ZS0_val
    ├── ...
```

## 🏃‍♂️ Training

### Single GPU Training

```bash
python train.py \
    --root /path/to/interaction/data \
    --devices 1 \
    --max_epochs 100
```

### Multi-GPU Training

```bash
python train.py \
    --root /path/to/interaction/data \
    --devices 4 \
    --accelerator gpu \
    --strategy ddp \
    --max_epochs 100
```

### Main Training Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--hidden_dim` | Hidden dimension | 128 |
| `--num_modes` | Number of trajectory modes | 6 |
| `--num_attn_layers` | Number of attention layers | 8 |
| `--lr` | Learning rate | 5e-4 |
| `--weight_decay` | Weight decay | 1e-4 |

## 🔍 Evaluation

### Validation

```bash
python val.py \
    --root /path/to/interaction/data \
    --ckpt_path /path/to/checkpoint.ckpt \
    --val_batch_size 2
    --devices 1
```

### Testing

```bash
python test.py \
    --root /path/to/interaction/data \
    --ckpt_path /path/to/checkpoint.ckpt \
    --test_batch_size 2 \
    --devices 1
```

## 📊 Performance Metrics

RuleNet's main metrics on INTERACTION dataset. Pre-trained models are available [here](https://github.com/Ruolin99/RuleNet/tree/main/logs/loss.csv/).


## 🔧 Core Components

### RuleNet Architecture

RuleNet integrates multiple key components:

- **Map Encoder**: Processes high-definition map data for spatial context
- **Graph Attention**: Models interactions between traffic agents
- **Robustness Attention**: Implements rule-based safety constraints
- **Trajectory Decoder**: Generates multi-modal trajectory predictions

### Safety Features

- **Time-to-Collision (TTC) Assessment**: Evaluates collision risks
- **Distance Violation Detection**: Monitors spatial constraints
- **Multi-Agent Safety**: Handles interactions between vehicles and VRUs

### Key Features

- **Safety-Oriented Design**: Integrates Time-to-Collision (TTC) and distance violation assessment mechanisms
- **Multi-Modal Prediction**: Supports multi-trajectory prediction modes for improved robustness
- **Graph Attention Networks**: Utilizes graph structures to model interactions between agents
- **Robustness Attention**: Rule-based attention mechanisms to enhance prediction accuracy
  

## 🤝 Acknowledgements

We sincerely appreciate the following projects for their awesome codebases:
- [INTERACTION Dataset](https://interaction-dataset.com/)
- [HPNet](https://github.com/XiaolongTang23/HPNet)



## 📜 Citation

If RuleNet has been helpful in your research, please consider citing our work:

```bibtex
@article{shi2025rulenet,
title = {RuleNet: rule-priority-aware multi-agent trajectory prediction in ambiguous traffic scenarios},
journal = {Transportation Research Part C: Emerging Technologies},
volume = {180},
pages = {105339},
year = {2025},
issn = {0968-090X},
doi = {https://doi.org/10.1016/j.trc.2025.105339},
url = {https://www.sciencedirect.com/science/article/pii/S0968090X25003432},
author = {Ruolin Shi and Xuesong Wang and Yang Zhou and Meixin Zhu}
```


## 📞 Contact
If you have any questions, feel free to contact us:
- **Email**: 2111224@tongji.edu.cn
- **GitHub**: [@Ruolin99](https://github.com/Ruolin99)

---

