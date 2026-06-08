# Federated Deepfake Detection using Flower and PyTorch

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.3-red)
![Flower](https://img.shields.io/badge/Flower-Federated%20Learning-orange)
![License](https://img.shields.io/badge/License-Apache%202.0-green)

## Overview

This project implements a **Federated Learning (FL) framework for Deepfake Detection** using **PyTorch** and the **Flower (FLWR)** federated learning framework.

The objective is to train a deepfake detection model collaboratively across multiple decentralized clients while ensuring that raw image data never leaves the client devices.

Unlike traditional centralized machine learning pipelines, federated learning enables privacy-preserving model training by exchanging only model parameters instead of datasets.

The system is designed to support:

- Multiple federated clients
- Deepfake image classification
- Vision Transformer (ViT) based learning
- CNN-based model comparisons
- Federated Averaging (FedAvg)
- Adversarial client simulation
- Model poisoning attack experiments
- GPU-accelerated distributed training

---

## Research Motivation

Deepfake generation technologies have rapidly advanced, creating significant challenges in:

- Digital media authenticity
- Social media misinformation
- Identity theft
- Cybersecurity
- Digital forensics

Most deepfake detection systems require centralized datasets, which introduces:

- Privacy concerns
- Data ownership issues
- Legal restrictions
- Large communication overhead

Federated Learning addresses these challenges by enabling collaborative model training without exposing local datasets.

This project explores the intersection of:

- Federated Learning
- Computer Vision
- Deepfake Detection
- Privacy-Preserving Artificial Intelligence
- Distributed Machine Learning
- Adversarial Robustness

---

## Dataset

### 140K Real and Fake Faces Dataset

Dataset Source:

https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces

Dataset Characteristics:

| Property     | Value                 |
| ------------ | --------------------- |
| Total Images | ~140,000              |
| Classes      | Real, Fake            |
| Task         | Binary Classification |
| Image Type   | Human Faces           |
| Format       | JPG/PNG               |
| Labels       | Real / Fake           |

The dataset is partitioned across multiple federated clients to simulate decentralized data ownership.

---

## Project Architecture

```text
                    Global Server
                           |
        ---------------------------------------
        |                 |                  |
     Client 1          Client 2          Client 3
        |                 |                  |
   Local Dataset     Local Dataset     Local Dataset
        |                 |                  |
   Local Training    Local Training    Local Training
        |                 |                  |
        -------- Model Aggregation ----------
                           |
                   Updated Global Model
```

---

## Features

### Federated Learning

- Flower-based federated architecture
- Federated Averaging (FedAvg)
- Multi-client simulation
- Distributed model training
- Configurable communication rounds

### Deepfake Detection

- Binary classification:
  - Real Faces
  - Fake Faces

### Multiple Model Support

The framework currently supports:

| Model                         | Type        |
| ----------------------------- | ----------- |
| Vision Transformer (ViT-B/16) | Transformer |
| ResNet18                      | CNN         |
| EfficientNet-B0               | CNN         |

Model selection can be configured through:

```toml
model-name = "vit"
```

Available options:

```toml
model-name = "vit"
model-name = "resnet18"
model-name = "efficientnet"
```

---

## Adversarial Federated Learning

The framework supports malicious client simulation.

Example:

```python
MALICIOUS_CLIENTS = {3}
```

A malicious client performs parameter poisoning before sending updates to the server.

Current attack implementation:

- Gaussian Noise Injection
- Parameter Perturbation

Purpose:

- Study robustness of federated learning
- Evaluate impact of poisoned updates
- Investigate secure aggregation techniques

---

## Repository Structure

```text
rishikrishnah-fl/
│
├── README.md
├── ingest.py
├── split.py
├── project_prompt.txt
│
└── quickstart-pytorch/
    │
    ├── pyproject.toml
    ├── LICENSE
    ├── README.md
    │
    ├── data/
    │
    └── pytorchexample/
        ├── __init__.py
        ├── client_app.py
        ├── server_app.py
        ├── model.py
        ├── task.py
        └── utils.py
```

---

## Component Description

### client_app.py

Responsible for:

- Loading local client datasets
- Training local models
- Performing local evaluation
- Returning model updates
- Simulating malicious client behavior

---

### server_app.py

Responsible for:

- Initializing global models
- Aggregating client updates
- Federated Averaging (FedAvg)
- Saving global checkpoints
- Managing communication rounds

---

### model.py

Contains:

- Vision Transformer implementation
- ResNet18 implementation
- EfficientNet implementation
- Transfer learning setup

---

### task.py

Responsible for:

- Dataset loading
- Image preprocessing
- Training loops
- Evaluation logic
- Metric computation
- Checkpoint generation

---

### utils.py

Contains:

- Parameter poisoning functions
- Experimental attack utilities
- Helper functions

---

### split.py

Dataset partitioning utility.

Responsible for:

- Splitting dataset among clients
- Creating train/validation/test partitions
- Simulating federated client ownership

---

## Installation

### Clone Repository

```bash
git clone https://github.com/<your-username>/rishikrishnah-fl.git

cd rishikrishnah-fl
```

---

### Create Virtual Environment

Windows:

```bash
python -m venv venv

venv\Scripts\activate
```

Linux/macOS:

```bash
python3 -m venv venv

source venv/bin/activate
```

---

### Install Dependencies

```bash
pip install -r requirements.txt

cd quickstart-pytorch

pip install -e .
```

---

## Dependencies

Core dependencies:

```text
Python 3.11.9
Flower 1.13.1
Ray 2.10.0
PyTorch 2.3.1
TorchVision 0.18.1
TorchAudio 2.3.1
CUDA 12.1
NumPy 1.26.4
Scikit-Learn
TQDM
```

---

## Dataset Preparation

Download the dataset from Kaggle.

Organize the dataset and run:

```bash
python split.py
```

This creates client partitions:

```text
dataset/
├── client1/
├── client2/
└── client3/
```

Each client contains:

```text
train/
val/
test/
```

with:

```text
real/
fake/
```

subdirectories.

---

## Running Federated Learning

Default execution:

```bash
cd quickstart-pytorch

flwr run . --stream
```

---

### Custom Configuration

Example:

```bash
flwr run . \
--run-config \
"num-server-rounds=10 local-epochs=5 model-name=resnet18" \
--stream
```

---

## Training Configuration

Current default configuration:

```toml
num-server-rounds = 3

local-epochs = 3

model-name = "vit"
```

Simulation settings:

```toml
options.num-supernodes = 3
```

---

## Output

After each communication round:

```text
saved_models/
├── global_model_round_1.pth
├── global_model_round_2.pth
├── global_model_round_3.pth
```

These checkpoints contain the aggregated global model.

---

## Evaluation Metrics

The framework supports:

- Accuracy
- Precision
- Recall
- F1 Score
- Loss

These metrics can be extended for future experiments.

---

## Experimental Objectives

The project investigates:

### Model Comparison

- Vision Transformer vs ResNet18
- Vision Transformer vs EfficientNet
- CNN vs Transformer architectures

### Federated Learning Performance

- Accuracy across communication rounds
- Convergence behavior
- Client heterogeneity impact

### Security Evaluation

- Model poisoning attacks
- Malicious client behavior
- Robust aggregation strategies

---

## Future Work

Planned extensions include:

### Federated Learning

- FedProx
- FedNova
- SCAFFOLD
- Secure Aggregation
- Differential Privacy

### Deepfake Detection

- Video Deepfake Detection
- Multi-class Manipulation Detection
- Temporal Analysis

### Robustness

- Byzantine-resilient aggregation
- Backdoor attack detection
- Adversarial defense mechanisms

### Deployment

- Edge-device federated learning
- Docker deployment
- Kubernetes orchestration
- Cross-device federated systems

---

## Sample Training Output

```text
INFO : Starting Flower ServerApp

INFO : Federated Round 1

Client 1 starting training...

Client 2 starting training...

Client 3 is malicious!

INFO : Aggregating updates

Saved: saved_models/global_model_round_1.pth
```

---

## References

### Flower Framework

https://flower.ai/

### PyTorch

https://pytorch.org/

### Vision Transformer Paper

Dosovitskiy et al.

"An Image is Worth 16x16 Words:
Transformers for Image Recognition at Scale"

### Federated Learning Paper

McMahan et al.

"Communication-Efficient Learning of Deep Networks from Decentralized Data"

### Dataset

140K Real and Fake Faces Dataset

https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces

---

## License

This project is licensed under the Apache License 2.0.

See the LICENSE file for additional details.

---

## Author

**Rishi Krishna**

B.Tech Computer Science and Engineering (AI & Robotics)

Vellore Institute of Technology (VIT), Chennai

---

## Acknowledgements

- Flower Framework Team
- PyTorch Community
- Kaggle Dataset Contributors
- VIT Chennai
- Federated Learning Research Community
- Open Source AI Community
