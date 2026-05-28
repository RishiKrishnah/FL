# Federated Learning for Deepfake Detection using PyTorch and Flower

## Overview

This project implements a **Federated Learning (FL)** system using the **Flower Framework** and **PyTorch**. The current implementation uses the **MNIST dataset** to simulate federated training across multiple clients. The project serves as the foundational architecture for a larger research-oriented system focused on **Federated Deepfake Detection**.

The system demonstrates how multiple distributed clients can collaboratively train a machine learning model without directly sharing their local datasets, thereby preserving privacy and reducing centralized data dependency.

---

# Project Objectives

* Understand and implement Federated Learning concepts
* Build a distributed training pipeline using Flower
* Simulate multiple FL clients locally
* Train a neural network collaboratively using PyTorch
* Create a scalable base architecture for Deepfake Detection research
* Explore privacy-preserving AI systems

---

# Technologies Used

| Technology    | Purpose                      |
| ------------- | ---------------------------- |
| Python        | Programming Language         |
| PyTorch       | Deep Learning Framework      |
| Flower (FLWR) | Federated Learning Framework |
| NumPy         | Numerical Operations         |
| TorchVision   | Dataset Handling             |
| MNIST Dataset | Initial Experimental Dataset |

---

# Project Structure

```bash
rishikrishnah-fl/
│
├── README.md
│
└── quickstart-pytorch/
    │
    ├── README.md
    ├── LICENSE
    ├── pyproject.toml
    │
    ├── data/
    │   └── MNIST/
    │       └── raw/
    │
    └── pytorchexample/
        ├── __init__.py
        ├── client_app.py
        ├── model.py
        ├── server_app.py
        ├── task.py
        └── utils.py
```

---

# File Descriptions

## `client_app.py`

Defines the Flower client logic.

Responsibilities:

* Loads local dataset partition
* Performs local model training
* Evaluates model performance
* Sends updated model weights back to the FL server

---

## `server_app.py`

Defines the federated server strategy.

Responsibilities:

* Initializes global model
* Aggregates client updates
* Coordinates communication rounds
* Manages federated averaging (FedAvg)

---

## `model.py`

Contains the PyTorch neural network architecture.

Responsibilities:

* Defines model layers
* Implements forward propagation
* Serves as the global and local model structure

---

## `task.py`

Handles dataset loading and training operations.

Responsibilities:

* Dataset preprocessing
* Data partitioning
* Local training loop
* Evaluation functions

---

## `utils.py`

Contains helper utility functions used across the project.

---

## `pyproject.toml`

Project configuration file.

Contains:

* Dependency management
* Flower app configuration
* Federation setup
* Simulation settings

---

# Federated Learning Workflow

```text
          Global Server
                 │
        -------------------
        │                 │
     Client 1         Client 2
        │                 │
   Local Training    Local Training
        │                 │
        -------Aggregation-------
                 │
          Updated Global Model
```

---

# How Federated Learning Works in this Project

1. The server initializes a global model.
2. Multiple clients receive the model.
3. Each client trains the model locally on its own dataset partition.
4. Clients send updated parameters back to the server.
5. The server aggregates updates using Federated Averaging (FedAvg).
6. The updated global model is redistributed.
7. The process repeats for multiple communication rounds.

---

# Installation Guide

## Prerequisites

Ensure the following are installed:

* Python 3.10+
* pip
* virtualenv (recommended)

---

# Step 1: Clone the Repository

```bash
git clone https://github.com/your-username/rishikrishnah-fl.git
cd rishikrishnah-fl/quickstart-pytorch
```

---

# Step 2: Create Virtual Environment

## Windows

```bash
python -m venv venv
venv\Scripts\activate
```

## Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

---

# Step 3: Install Dependencies

```bash
pip install -e .
```

---

# Dependencies

The project uses the following dependencies:

```toml
dependencies = [
    "flwr[simulation]==1.13.1",
    "torch",
    "torchvision",
    "numpy==1.26.4"
]
```

---

# Running the Federated Learning Simulation

Navigate to:

```bash
cd quickstart-pytorch
```

Run the FL simulation:

```bash
flwr run . --stream
```

---

# Running with Custom Configurations

Example:

```bash
flwr run . --run-config "num-server-rounds=5 learning-rate=0.05" --stream
```

---

# Current Configuration

## Number of Clients

```toml
options.num-supernodes = 2
```

## Number of Federated Rounds

```toml
num-server-rounds = 3
```

---

# Dataset Information

## MNIST Dataset

The current implementation uses the MNIST handwritten digit dataset.

Features:

* 70,000 grayscale images
* 28×28 image size
* 10 output classes (digits 0–9)

Purpose:

* Initial FL experimentation
* System validation
* Architecture testing

---

# Future Scope: Deepfake Detection

This project is intended to evolve into a **Federated Deepfake Detection System**.

Planned improvements include:

* Replace MNIST with deepfake datasets
* Integrate CNN/ViT architectures
* Add secure aggregation
* Implement differential privacy
* Add client heterogeneity simulation
* Use real-world distributed clients
* Improve communication efficiency
* Add adversarial robustness

---

# Research Motivation

Traditional deepfake detection systems require centralized datasets, which may introduce:

* Privacy concerns
* Data-sharing restrictions
* High communication costs

Federated Learning addresses these issues by enabling decentralized model training while keeping raw data local.

This project explores the intersection of:

* Artificial Intelligence
* Computer Vision
* Federated Learning
* Privacy-Preserving Machine Learning

---

# Example Output

```text
INFO : Starting Flower ServerApp
INFO : Requesting initial parameters from one random client
INFO : Federated Learning Round 1
INFO : Aggregating client updates
INFO : Evaluation completed
```

---

# Learning Outcomes

Through this project, the following concepts can be understood:

* Federated Learning architecture
* Client-server FL communication
* Federated Averaging (FedAvg)
* Distributed training
* PyTorch model integration with Flower
* Privacy-preserving ML systems

---

# Challenges Faced

* Flower framework configuration
* Local client synchronization
* Dataset partitioning
* Large dependency management
* GitHub large file handling
* Simulation optimization on CPU systems

---

# References

* Flower Documentation
  https://flower.ai/docs/

* PyTorch Documentation
  https://pytorch.org/docs/

* Federated Learning Research Paper
  McMahan et al., "Communication-Efficient Learning of Deep Networks from Decentralized Data"

---

# License

This project is licensed under the Apache License 2.0.

See the `LICENSE` file for details.

---

# Author

## Rishi Krishna

B.Tech CSE (AI & Robotics)
VIT Chennai

---

# Acknowledgements

Special thanks to:

* Flower Framework developers
* PyTorch community
* VIT Chennai
* Research contributors in Federated Learning and Deepfake Detection

---
