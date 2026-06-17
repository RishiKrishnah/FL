# Federated Artifact-Guided Frequency-Aware Transformer (FAFT) for Deepfake Detection

### Privacy-Preserving Deepfake Detection using Federated Learning, Artifact-Aware Transformers, and Prototype Memory Aggregation

![Python](https://img.shields.io/badge/Python-3.11-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.3-red)
![Flower](https://img.shields.io/badge/Flower-1.13-orange)
![Federated Learning](https://img.shields.io/badge/Federated-Learning-green)
![Deepfake Detection](https://img.shields.io/badge/Deepfake-Detection-purple)
![License](https://img.shields.io/badge/License-Apache%202.0-success)

---

# Overview

This repository presents a complete Federated Learning framework for Deepfake Detection using PyTorch and Flower.

The project investigates how deepfake detection models can be trained collaboratively across multiple decentralized clients without sharing raw image data.

Unlike traditional centralized approaches, only model parameters and prototype representations are exchanged, preserving data privacy while enabling collaborative learning.

The framework supports:

* Federated Learning with Flower
* Deepfake Image Classification
* Vision Transformers
* CNN Architectures
* Hybrid CNN-Transformer Models
* Artifact-Aware Learning
* Frequency Domain Feature Extraction
* Prototype Memory Aggregation
* Adversarial Client Simulation
* GPU Accelerated Training

---

# Research Motivation

Deepfake generation technologies have become increasingly sophisticated, posing significant threats to digital media authenticity, cybersecurity, and public trust.

Traditional deepfake detection approaches require centralized access to large datasets, creating challenges related to:

* Data privacy
* Data ownership
* Regulatory compliance
* Communication overhead

Federated Learning offers a privacy-preserving alternative by allowing multiple clients to collaboratively train a global model without sharing raw data.

This project explores the intersection of:

* Federated Learning
* Deepfake Detection
* Computer Vision
* Transformer Architectures
* Privacy-Preserving AI
* Adversarial Robustness

---

# Proposed FAFT Architecture

The proposed **Federated Artifact-Guided Frequency-Aware Transformer (FAFT)** introduces three complementary components:

## 1. Spatial Artifact Extraction

A ResNet18 backbone extracts forensic spatial features from facial images.

## 2. Frequency Artifact Analysis

A dedicated frequency branch processes FFT-transformed images to capture manipulation traces that may be invisible in the spatial domain.

## 3. Artifact-Guided Transformer Learning

Artifact representations are converted into transformer attention biases, guiding the model toward forensic regions relevant for deepfake detection.

---

# Federated Prototype Memory Aggregation

Each client generates an artifact prototype representation during local training.

The server aggregates these prototypes into a global forensic memory representation and redistributes it to all participating clients.

Benefits include:

* Improved global feature consistency
* Better knowledge sharing across clients
* Reduced client drift
* Faster convergence
* Enhanced generalization

---

# System Architecture

```text
                          Flower Server
                                  │
                    Prototype Memory Aggregation
                                  │
        ┌──────────────┬──────────────┬──────────────┬──────────────┬──────────────┐
        │              │              │              │              │
     Client 1       Client 2       Client 3       Client 4       Client 5
        │              │              │              │              │
   Local Data     Local Data     Local Data     Local Data     Local Data
        │              │              │              │              │
     FAFT Model     FAFT Model     FAFT Model     FAFT Model     FAFT Model
        │              │              │              │              │
        └──────────────── Federated Averaging ────────────────┘
                                  │
                          Updated Global Model
```

---

# Key Features

## Federated Learning

* Flower-based federated architecture
* Federated Averaging (FedAvg)
* Multi-client simulation
* Configurable communication rounds
* GPU support
* Client heterogeneity support

## Deepfake Detection

Binary classification:

* Real Faces
* Fake Faces

## Artifact-Aware Learning

* Frequency-domain analysis
* Spatial artifact extraction
* Attention-guided transformer learning
* Prototype memory aggregation

## Security Research

* Malicious client simulation
* Model poisoning attacks
* Federated robustness experiments

---

# Supported Models

| Model                         | Category              |
| ----------------------------- | --------------------- |
| FAFT                          | Proposed Architecture |
| Artifact-Guided Deepfake Net  | Artifact-Aware Hybrid |
| Vision Transformer (ViT-B/16) | Transformer           |
| Swin Transformer              | Transformer           |
| ResNet18                      | CNN                   |
| EfficientNet-B0               | CNN                   |
| ResNet18 + ViT                | Hybrid                |
| ResNet18 + Swin               | Hybrid                |

---

# Repository Structure

```text
FL/
│
├── README.md
├── requirements.txt
├── ingest.py
├── project_prompt.txt
│
└── quickstart-pytorch/
    │
    ├── pyproject.toml
    ├── split.py
    ├── evaluate_saved_model.py
    │
    ├── dataset/
    │   ├── client1/
    │   ├── client2/
    │   ├── client3/
    │   ├── client4/
    │   └── client5/
    │
    ├── saved_models/
    │
    └── pytorchexample/
        ├── client_app.py
        ├── server_app.py
        ├── model.py
        ├── task.py
        └── utils.py
```

---

# Dataset

## 140K Real and Fake Faces Dataset

Dataset Source:

https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces

### Dataset Statistics

| Property     | Value                 |
| ------------ | --------------------- |
| Total Images | ~140,000              |
| Classes      | Real / Fake           |
| Task         | Binary Classification |
| Domain       | Deepfake Detection    |

---

# Dataset Partitioning

The dataset is partitioned into five federated clients.

```text
dataset/
├── client1/
├── client2/
├── client3/
├── client4/
└── client5/
```

Each client contains:

```text
train/
val/
test/
```

Each split contains:

```text
real/
fake/
```

subdirectories.

---

# Installation

## Clone Repository

```bash
git clone https://github.com/<YOUR_USERNAME>/<YOUR_REPOSITORY>.git

cd FL
```

## Create Virtual Environment

### Linux / macOS

```bash
python3 -m venv .venv

source .venv/bin/activate
```

### Windows

```bash
python -m venv .venv

.venv\Scripts\activate
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

Install the Flower application:

```bash
cd quickstart-pytorch

pip install -e .
```

---

# Configuration

Default configuration located in:

```text
quickstart-pytorch/pyproject.toml
```

Current defaults:

```toml
num-server-rounds = 5
local-epochs = 5
model-name = "faft"
options.num-supernodes = 5
```

---

# Running Federated Training

## Default FAFT Training

```bash
cd quickstart-pytorch

flwr run . --stream
```

---

## Train Vision Transformer

```bash
flwr run . \
--run-config "model-name=vit" \
--stream
```

---

## Train ResNet18

```bash
flwr run . \
--run-config "model-name=resnet18" \
--stream
```

---

## Train EfficientNet-B0

```bash
flwr run . \
--run-config "model-name=efficientnet" \
--stream
```

---

## Train Swin Transformer

```bash
flwr run . \
--run-config "model-name=swin" \
--stream
```

---

## Train Hybrid ResNet + ViT

```bash
flwr run . \
--run-config "model-name=hybrid" \
--stream
```

---

## Train Hybrid ResNet + Swin

```bash
flwr run . \
--run-config "model-name=hybrid_swin" \
--stream
```

---

# Model Evaluation

Evaluate a saved global model checkpoint:

```bash
python evaluate_saved_model.py
```

The evaluation script automatically detects the model architecture and reports:

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC
* Confusion Matrix
* Classification Report

---

# Saved Models

Global checkpoints are automatically stored after each federated round.

```text
saved_models/
├── global_model_round_1.pth
├── global_model_round_2.pth
├── global_model_round_3.pth
├── global_model_round_4.pth
└── global_model_round_5.pth
```

---

# Adversarial Client Simulation

The framework supports malicious client experiments.

Inside `client_app.py`:

```python
MALICIOUS_CLIENTS = {3}
```

Supported attack:

* Gaussian Noise Parameter Poisoning

Research applications:

* Byzantine Client Analysis
* Robust Federated Aggregation
* Federated Security Evaluation

---

# Experimental Research Directions

This framework can be extended for:

## Federated Learning

* FedProx
* FedNova
* SCAFFOLD
* FedOpt
* Personalized Federated Learning

## Privacy

* Differential Privacy
* Secure Aggregation
* Homomorphic Encryption

## Deepfake Detection

* Video Deepfake Detection
* Multi-Class Manipulation Detection
* Temporal Forensics

## Security

* Backdoor Attack Detection
* Byzantine-Resilient Aggregation
* Adversarial Defense Mechanisms

---

# Results

The framework enables:

* Federated multi-client training
* Prototype memory aggregation
* Artifact-aware transformer learning
* Deepfake detection benchmarking
* CNN vs Transformer comparisons
* Security and robustness experimentation

---

# Citation

```bibtex
@misc{krishna2026faft,
  title={Federated Artifact-Guided Frequency-Aware Transformer for Deepfake Detection},
  author={Rishi Krishna},
  year={2026}
}
```

---

# Author

**Rishi Krishna**

B.Tech Computer Science and Engineering (AI & Robotics)

Vellore Institute of Technology (VIT), Chennai

---

# Acknowledgements

* Flower Framework
* PyTorch
* TorchVision
* Kaggle Community
* Open Source AI Community
* VIT Chennai

---

# License

This project is licensed under the Apache License 2.0.

See the LICENSE file for details.
