# FAFT — Federated Artifact-Guided Frequency-Aware Transformer

![Python](https://img.shields.io/badge/Python-3.11-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.3-red)
![Flower](https://img.shields.io/badge/Flower-1.13-orange)
![Federated Learning](https://img.shields.io/badge/Federated-Learning-green)
![Deepfake Detection](https://img.shields.io/badge/Deepfake-Detection-purple)
![License](https://img.shields.io/badge/License-Apache%202.0-success)


**Privacy-preserving deepfake detection trained across decentralized clients with Flower, PyTorch, and prototype-memory federated aggregation.**

FAFT combines a ResNet18 spatial branch, an FFT-based frequency branch, and an artifact-guided transformer to detect real vs. fake faces. Instead of exchanging raw images, clients exchange model weights and compact class **prototype embeddings**, which the server aggregates into a global forensic memory and redistributes every round — on top of standard federated averaging.

---

## Features

**Federated Learning**
- Flower (`flwr[simulation]==1.13.1`) `ServerApp` / `ClientApp` simulation with 5 supernodes
- Custom `FedAvg` subclasses: `SaveModelStrategy` (checkpointing) and `FAFTStrategy` (weights + prototype-memory aggregation)
- Configurable rounds, local epochs, and per-round model checkpointing
- Malicious-client simulation with Gaussian parameter poisoning

**Model Zoo** (selectable via `model-name`)
- `faft` — proposed Artifact-Guided Frequency-Aware Transformer
- `artifact_vit` — Artifact-Guided Deepfake Net (ResNet18 + frequency fusion + frozen ViT-B/16)
- `vit`, `resnet18`, `efficientnet`, `swin` — standard backbones fine-tuned for binary classification
- `hybrid`, `hybrid_swin` — ResNet18 combined with ViT-B/16 or Swin-T features

**Dataset Support**
- 140K Real and Fake Faces (via `split.py`)
- FaceForensics++ C23, six manipulation methods (via `preprocess_ffpp.py`)
- OpenForensics (via `preprocess_openforensics.py`)
- Cross-dataset holdout evaluation on FF++ `DeepFakeDetection` / `FaceShifter`

**Training & Evaluation Tooling**
- Mixed-precision training (`torch.amp`), GPU memory/throughput logging, per-batch ETA
- `GPUManager` — file-lock-based multi-GPU load balancing for Ray client actors
- `evaluate_saved_model.py` / `evaluate_holdout.py` — checkpoint architecture auto-detection, full metric suite (accuracy, precision, recall, F1, ROC-AUC, confusion matrix)
- `run_experiment.py` — orchestrates training → evaluation → holdout with GPU-availability waiting and automatic retries, logging every stage to `experiment_logs/`

---

## Why This Project?

Deepfake detectors are typically trained on centrally pooled data, which conflicts with data-privacy, ownership, and regulatory constraints. Federated Learning removes the need to centralize raw images: each client trains locally on its own partition and only shares model updates.

FAFT extends plain FedAvg with two forensic-specific ideas:

1. **Artifact-guided attention** — spatial (ResNet18) and frequency (FFT-CNN) features are fused into an "artifact embedding" that biases the transformer's self-attention toward manipulation-relevant regions.
2. **Federated prototype memory** — each client also shares a running real/fake prototype vector. The server aggregates these into a global memory that is broadcast alongside model weights, giving every client access to a consistent, dataset-wide notion of "real" and "fake" without ever exchanging images.

---

## Architecture Overview

```text
   Client Image
        │
        ├────────────► ResNet18 (spatial branch) ───┐
        │                                            │
        └────────────► FFT → CNN (frequency branch) ─┤
                                                       ▼
                                              Fusion (Linear + ReLU)
                                                       │
                                          Artifact Attention (gating)
                                                       │
                                        Artifact Projection → 768-d embedding
                                            │                      │
                              Artifact Bias Generator      Update Prototypes
                              (low-rank attention bias)     (real / fake, EMA)
                                            │
                     Patch Embed (Conv 16×16) + CLS + Pos Embed
                                            │
                         4 × FAFTBlock (8-head attention + artifact bias)
                                            │
                                      LayerNorm → Classifier
                                            │
                                     Real / Fake logits
```

---

## Repository Structure

```text
FL-main/
├── README.md
├── requirements.txt              # Frozen environment (CUDA 12.1 / torch 2.5.1)
├── ingest.py                     # Dev utility: dumps repo source into project_prompt.txt
├── project_prompt.txt            # Generated source dump (not part of the ML pipeline)
│
└── quickstart-pytorch/           # Flower application
    ├── pyproject.toml            # Flower app config, dependencies, federation settings
    ├── README.md                 # Flower quickstart template notes
    ├── LICENSE                   # Apache 2.0
    │
    ├── split.py                  # Partitions the 140K dataset into 5 clients
    ├── preprocess_ffpp.py        # FaceForensics++ frame extraction / face crop / FL partitioning
    ├── preprocess_openforensics.py  # OpenForensics resize + FL partitioning
    ├── recreate_holdout_eval.py  # Builds FF++ holdout_eval/{real,fake} via symlinks
    ├── fix_holdout_real.py       # Balances holdout real/fake symlink counts
    │
    ├── evaluate_saved_model.py   # In-distribution evaluation across all 5 clients
    ├── evaluate_holdout.py       # Cross-dataset (FF++ holdout) evaluation
    ├── run_experiment.py         # GPU-aware pipeline runner (train → eval → holdout)
    │
    ├── preprocess.log            # FF++ preprocessing run log
    ├── logs_ffpp/
    │   └── processed_videos.csv  # Per-video frame extraction telemetry
    │
    ├── experiment_logs/
    │   └── <timestamp>/
    │       ├── training.log
    │       ├── evaluation.log
    │       ├── TRAINING_SUCCESS
    │       └── EVALUATION_SUCCESS
    │
    ├── saved_models/             # Created at runtime
    │   └── global_model_round_N.pth
    │
    └── pytorchexample/           # Flower ClientApp / ServerApp package
        ├── __init__.py
        ├── client_app.py         # FlowerClient: fit / evaluate / poisoning hook
        ├── server_app.py         # SaveModelStrategy, FAFTStrategy, server_fn
        ├── model.py              # FAFTNet + all model variants + load_model()
        ├── task.py                # Data loading, train(), test(), save_model()
        ├── utils.py               # poison_parameters()
        └── gpu_manager.py         # Multi-GPU load-balancing for Ray actors
```

> `dataset/`, `dataset_ffpp/`, and `dataset_openforensic/` are not shipped in the repository — they are produced locally by the preprocessing scripts above.

---

## Workflow

```text
Raw dataset (source path)
        │
        ▼
split.py / preprocess_ffpp.py / preprocess_openforensics.py
        │
        ▼
dataset*/client{1..5}/{train,val,test}/{real,fake}
        │
        ▼
flwr run .  (5 rounds × 5 clients, local-epochs configurable)
        │
   ┌────┴─────────────────────────────────┐
   │ per round: client fit → local train  │
   │ → weights + prototypes → server      │
   │ → FedAvg + prototype EMA aggregation │
   │ → checkpoint saved → broadcast       │
   └────┬─────────────────────────────────┘
        ▼
saved_models/global_model_round_5.pth
        │
        ├──► evaluate_saved_model.py   (in-distribution, all 5 clients)
        └──► evaluate_holdout.py       (FF++ holdout, cross-dataset)
```

---

## Quick Start

```bash
git clone https://github.com/<YOUR_USERNAME>/<YOUR_REPOSITORY>.git
cd FL-main

python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cd quickstart-pytorch
pip install -e .

# Prepare the dataset (edit SOURCE inside split.py to point at your local copy first)
python split.py

# Train FAFT with the default federation
flwr run . --stream

# Evaluate the final global checkpoint
python evaluate_saved_model.py
```

---

## Installation

### Clone the repository

```bash
git clone https://github.com/<YOUR_USERNAME>/<YOUR_REPOSITORY>.git
cd FL-main
```

### Create a virtual environment

```bash
# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate

# Windows
python -m venv .venv
.venv\Scripts\activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

Install the Flower application package (registers `pytorchexample` and reads `pyproject.toml`):

```bash
cd quickstart-pytorch
pip install -e .
```

> **GPU / PyTorch note:** `quickstart-pytorch/pyproject.toml` pins `torch==2.3.1` / `torchvision==0.18.1`, while the root `requirements.txt` freezes a working GPU environment on `torch==2.5.1+cu121` (CUDA 12.1 wheels). Install the CUDA build that matches your driver from [pytorch.org](https://pytorch.org/get-started/locally/) if `pip install -r requirements.txt` does not resolve GPU wheels on your platform. Training and evaluation both fall back to CPU automatically when CUDA is unavailable (`torch.device("cuda" if torch.cuda.is_available() else "cpu")`).

---

## Dataset Preparation

### 140K Real and Fake Faces

`split.py` reads `real-vs-fake/{train,valid,test}/{real,fake}` from a source path (edit `SOURCE` in the script) and copies images into 5 client partitions under `dataset/`:

```text
dataset/
├── client1/
│   ├── train/{real,fake}
│   ├── val/{real,fake}
│   └── test/{real,fake}
├── client2/ ...
├── client3/ ...
├── client4/ ...
└── client5/ ...
```

### FaceForensics++ (C23)

`preprocess_ffpp.py` extracts frames from `SOURCE_ROOT` (`original` + `Deepfakes`, `Face2Face`, `FaceSwap`, `NeuralTextures`, `FaceShifter`, `DeepFakeDetection`), applies face cropping, blur filtering (threshold 40.0), and perceptual-hash duplicate removal, then writes a 70/15/15 train/val/test split across 5 IID client partitions into `dataset_ffpp/`. Per-video statistics are logged to `logs_ffpp/processed_videos.csv`.

### OpenForensics

`preprocess_openforensics.py` resizes `Real`/`Fake` images to 224×224 and partitions them across 5 clients into `dataset_openforensic/`.

### Holdout / cross-dataset set

`recreate_holdout_eval.py` and `fix_holdout_real.py` build a class-balanced FF++ holdout evaluation set from `dataset_ffpp/holdout/{DeepFakeDetection,FaceShifter}` (using symlinks) for out-of-distribution testing:

```text
dataset_ffpp/holdout_eval/
├── DeepFakeDetection/{real,fake}
└── FaceShifter/{real,fake}
```

`task.py` selects the active root via `dataset-name`:

| `dataset-name` | Root directory          |
| --------------- | ------------------------ |
| `140k`          | `dataset`                |
| `ffpp`          | `dataset_ffpp`           |
| `openforensics` | `dataset_openforensic`   |

---

## Configuration

Defined in `quickstart-pytorch/pyproject.toml`:

| Parameter | Description | Default | Purpose |
| --- | --- | --- | --- |
| `num-server-rounds` | Number of federated communication rounds | `5` | Controls training duration |
| `local-epochs` | Local epochs each client trains per round | `1` | Local compute budget per round |
| `model-name` | Architecture to train (`faft`, `artifact_vit`, `vit`, `resnet18`, `efficientnet`, `swin`, `hybrid`, `hybrid_swin`) | `"faft"` | Selects `load_model()` branch |
| `dataset-name` | Dataset root to load (`140k`, `ffpp`, `openforensics`) | `"140k"` | Selects data partition source |
| `options.num-supernodes` | Number of simulated Flower clients | `5` | Must match number of client partitions |
| `client-resources.num-cpus` | CPUs allocated per simulated client | `2` | Ray simulation resource budget |
| `client-resources.num-gpus` | GPU fraction allocated per simulated client | `1.0` | Ray simulation resource budget |

---

## Training

```bash
cd quickstart-pytorch
```

**Default FAFT training:**

```bash
flwr run . --stream
```

**Other architectures**, overridden via `--run-config`:

```bash
flwr run . --run-config "model-name=vit" --stream
flwr run . --run-config "model-name=resnet18" --stream
flwr run . --run-config "model-name=efficientnet" --stream
flwr run . --run-config "model-name=swin" --stream
flwr run . --run-config "model-name=hybrid" --stream
flwr run . --run-config "model-name=hybrid_swin" --stream
flwr run . --run-config "model-name=artifact_vit" --stream
```

Each round, `SaveModelStrategy` / `FAFTStrategy.aggregate_fit` reconstructs the aggregated global model and writes `saved_models/global_model_round_<N>.pth`.

**Orchestrated pipeline (train → eval → optional holdout, with GPU wait/retry):**

```bash
python run_experiment.py
```

`run_experiment.py` waits for a GPU under `MAX_GPU_UTILIZATION` / `MAX_GPU_MEMORY_GB`, runs `flwr run . --stream`, then `evaluate_saved_model.py`, retrying each stage on failure and writing `training.log` / `evaluation.log` / success markers to `experiment_logs/<timestamp>/`.

---

## Evaluation

**In-distribution evaluation** across all 5 client test splits:

```bash
python evaluate_saved_model.py
```

`detect_model_type()` inspects the checkpoint's state-dict keys to automatically identify the architecture (`faft`, `artifact_vit`, `hybrid_swin`, `hybrid`, `vit`, `swin`, `resnet18`, `efficientnet`), loads `saved_models/global_model_round_5.pth`, and reports:

- Accuracy, Precision, Recall, F1 Score, ROC-AUC
- Confusion matrix
- Full `sklearn` classification report (`Fake` / `Real`)
- Throughput (img/s) and peak GPU memory

**Cross-dataset holdout evaluation** on FF++ `DeepFakeDetection` / `FaceShifter`:

```bash
python evaluate_holdout.py
```

Both scripts auto-select the best available GPU via `nvidia-smi` free-memory querying and fall back to CPU when CUDA is unavailable.

---

## Model Architecture

### FAFTNet (`faft`) — proposed architecture

| Component | Detail |
| --- | --- |
| Spatial branch | ResNet18 (ImageNet-pretrained), `fc` replaced with `Identity` → 512-d |
| Frequency branch | `torch.fft.fft2` → magnitude → `log1p` → 3-layer CNN (32→64→128) → 128-d |
| Fusion | `Linear(640, 512)` + ReLU + Dropout(0.3) |
| Artifact attention | 2-layer gated MLP (`ArtifactAttention`), Sigmoid gate over 512-d fused features |
| Artifact projection | `Linear(512, 768)` + LayerNorm + ReLU → transformer-space embedding |
| Bias generator | `ArtifactBiasGenerator`: low-rank (rank=32) query/key projection producing an 8-head, 197×197 attention bias from the artifact embedding |
| Patch embedding | `Conv2d(3, 768, kernel_size=16, stride=16)` → 14×14 = 196 patches + CLS token, learned position embedding (197 tokens) |
| Transformer | 4 × `FAFTBlock`: LayerNorm → 8-head self-attention biased by the artifact bias (+ optional global-memory bias) → residual → LayerNorm → MLP (ratio 4, GELU) → residual |
| Class memory | Real/fake prototypes (EMA α=0.9) updated every training batch from the artifact embedding; server-broadcast global memory is fused into the attention bias via cosine-similarity-weighted, `sigmoid(gamma)`-gated blending |
| Classifier | `Linear(768, 512)` → ReLU → Dropout(0.3) → `Linear(512, 2)` |

### Artifact-Guided Deepfake Net (`artifact_vit`)

Same ResNet18 + frequency fusion + artifact-attention pipeline as FAFT, but instead of a custom transformer it concatenates the 768-d artifact embedding with features from a **frozen ViT-B/16** (only the last encoder block is fine-tuned) and classifies the 1536-d concatenation.

### Hybrid models (`hybrid`, `hybrid_swin`)

ResNet18 (512-d) concatenated with either a frozen ViT-B/16 (768-d, last block fine-tuned) or a frozen Swin-T (768-d, last stage fine-tuned), followed by `Linear(1280, 512) → ReLU → Dropout(0.3) → Linear(512, 2)`.

### Single-backbone baselines (`vit`, `resnet18`, `efficientnet`, `swin`)

Standard `torchvision` ImageNet-pretrained backbones with their classification head replaced by a 2-class `Linear` layer (`vit_b_16`, `resnet18`, `efficientnet_b0`, `swin_t`).

---

## Federated Learning Pipeline

- **Framework:** Flower `ServerApp` / `ClientApp`, simulation backend, `num-supernodes=5`.
- **Clients:** `FlowerClient(NumPyClient)` loads its architecture and its `client{id}` data partition; `fit()` runs `train()` locally, `evaluate()` runs `test()` against the local test split.
- **Base strategy:** `FedAvg` (`fraction_fit=1.0`, `fraction_evaluate=1.0`, `min_fit_clients=5`, `min_evaluate_clients=5`, `min_available_clients=5`).
- **Checkpointing (`SaveModelStrategy`):** used for all non-FAFT models — after each round's `aggregate_fit`, reconstructs the global model from aggregated ndarrays and saves `saved_models/global_model_round_<N>.pth`.
- **Prototype aggregation (`FAFTStrategy`):** used when `model-name=faft`. Client updates carry `[model_weights..., real_prototype, fake_prototype]`. The server:
  1. Splits weights from the two 768-d prototype tensors.
  2. FedAvgs model weights with `fedavg_weights()` (example-count weighted).
  3. Aggregates prototypes with `np.average(..., weights=example_counts)`, then blends into `global_real_memory` / `global_fake_memory` via EMA (`0.9 * old + 0.1 * new`).
  4. Saves the aggregated checkpoint and broadcasts `weights + [global_real_memory, global_fake_memory]` back to all clients.
- **Communication:** parameters serialized via `ndarrays_to_parameters` / `parameters_to_ndarrays`; clients detect and load the trailing 2 memory tensors in `set_parameters()`.
- **Aggregation metrics:** `weighted_average()` computes example-weighted distributed accuracy for `evaluate_metrics_aggregation_fn`.
- **Adversarial simulation:** client IDs listed in `MALICIOUS_CLIENTS` (empty set by default in `client_app.py`) have their outgoing parameters perturbed by `poison_parameters()` — additive Gaussian noise (σ=0.05) applied layer-by-layer, with prototype tensors excluded from poisoning when present.
- **Multi-GPU scheduling:** `GPUManager` (used by data/training utilities) scores GPUs by `free_memory − 5 × assigned_clients` using a file-locked `/tmp/fl_gpu_lock.json`, with a `FORCE_GPU` environment-variable override for manual pinning under Ray.

---

## Outputs

| Artifact | Location |
| --- | --- |
| Global checkpoints | `saved_models/global_model_round_<1..N>.pth` |
| Training / evaluation logs (orchestrated runs) | `experiment_logs/<timestamp>/{training,evaluation,holdout}.log` |
| Stage success markers | `experiment_logs/<timestamp>/{TRAINING_SUCCESS,EVALUATION_SUCCESS,HOLDOUT_SUCCESS}` |
| FF++ preprocessing telemetry | `preprocess.log`, `logs_ffpp/processed_videos.csv` |
| Console metrics | Accuracy / precision / recall / F1 / ROC-AUC / confusion matrix / classification report, printed by both evaluation scripts |

---

## Results

Reference run: FAFT (`model-name=faft`), 140K Real-and-Fake-Faces dataset, 5 clients, 5 server rounds, 1 local epoch/round.

**Federated evaluation accuracy per round** (`flwr run . --stream`, `experiment_logs/20260721_113245/training.log`):

| Round | Distributed Accuracy | Distributed Loss |
| --- | --- | --- |
| 1 | 0.9072 | 78.72 |
| 2 | 0.9756 | 10.01 |
| 3 | 0.9873 | 5.48 |
| 4 | 0.9793 | 6.82 |
| 5 | 0.9852 | 5.17 |

**Final checkpoint evaluation** (`evaluate_saved_model.py`, `global_model_round_5.pth`, 20,000 test images across 5 clients):

| Metric | Value |
| --- | --- |
| Accuracy | 0.9886 |
| Precision | 0.9993 |
| Recall | 0.9779 |
| F1 Score | 0.9885 |
| ROC-AUC | 0.9999 |
| Throughput | 371.9 img/s |
| Peak GPU allocated | 0.86 GB |

![Training Curve](docs/images/training.png)
![Confusion Matrix](docs/images/confusion_matrix.png)

---

## Dependencies

**Federated Learning**
- `flwr[simulation]==1.13.1`
- `ray==2.10.0`

**Deep Learning**
- `torch==2.3.1` (`pyproject.toml`) / `torch==2.5.1+cu121` (`requirements.txt`)
- `torchvision==0.18.1` / `0.20.1+cu121`
- `torchaudio==2.3.1` / `2.5.1+cu121`
- `timm==1.0.27`

**Data & Metrics**
- `numpy==1.26.4` / `2.4.4`
- `scikit-learn==1.9.0`
- `Pillow==12.2.0`
- `PyYAML==6.0.3`

**Utilities**
- `tqdm`, `requests`, `rich`, `typer`

**Preprocessing-only (imported by `preprocess_ffpp.py` / `gpu_manager.py`, not pinned in `requirements.txt`)**
- `insightface` (face detection/analysis)
- `opencv-python` (`cv2`)
- `imagehash`
- `pynvml`

---

## Roadmap

**Implemented**
- [x] Flower FedAvg simulation with 5-client partitioning
- [x] FAFTNet: artifact-guided, frequency-aware transformer with bias-conditioned attention
- [x] Federated prototype-memory aggregation (`FAFTStrategy`)
- [x] Model zoo: FAFT, artifact-guided hybrid, ViT, Swin, ResNet18, EfficientNet-B0, ResNet+ViT / ResNet+Swin hybrids
- [x] 140K Faces, FaceForensics++, and OpenForensics preprocessing pipelines
- [x] Cross-dataset holdout evaluation (FF++ `DeepFakeDetection` / `FaceShifter`)
- [x] Malicious-client simulation via Gaussian parameter poisoning
- [x] Multi-GPU scheduling for Ray-based client actors
- [x] GPU-aware experiment orchestration with retries (`run_experiment.py`)

**Planned**
- [ ] FedProx, FedNova, SCAFFOLD, FedOpt aggregation strategies
- [ ] Personalized Federated Learning
- [ ] Differential Privacy
- [ ] Secure Aggregation / Homomorphic Encryption
- [ ] Video-level and temporal deepfake forensics
- [ ] Multi-class manipulation-type detection
- [ ] Backdoor attack detection and Byzantine-resilient aggregation
- [ ] Adversarial defense mechanisms

---

## Contributing

Contributions are welcome.

1. Fork the repository and create a feature branch.
2. Keep changes scoped — one architectural or pipeline change per pull request.
3. Match existing code style (see `pytorchexample/`) and avoid introducing untracked dependencies without updating `pyproject.toml` / `requirements.txt`.
4. Verify `flwr run . --stream` completes for at least one round before submitting.
5. Open a pull request describing the motivation, the change, and any new configuration flags.

---

## Citation

```bibtex
@misc{krishna2026faft,
  title={Federated Artifact-Guided Frequency-Aware Transformer for Deepfake Detection},
  author={Rishi Krishna},
  year={2026}
}
```

---

## License

This project is licensed under the **Apache License 2.0**. See [`quickstart-pytorch/LICENSE`](quickstart-pytorch/LICENSE) for details.

---

## Acknowledgements

- [Flower](https://flower.ai/) — federated learning framework
- [PyTorch](https://pytorch.org/) / [TorchVision](https://pytorch.org/vision/) — model and training backbone
- [InsightFace](https://github.com/deepinsight/insightface) — face analysis used in FaceForensics++ preprocessing
- [scikit-learn](https://scikit-learn.org/) — evaluation metrics
- [Kaggle: 140K Real and Fake Faces](https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces) — primary dataset source