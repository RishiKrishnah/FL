import time
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
)

from pytorchexample.model import load_model
import subprocess

# ==================================================
# SETTINGS
# ==================================================

MODEL_PATH = "saved_models/global_model_round_5.pth"

HOLDOUT_ROOT = "dataset_ffpp/holdout_eval"

BATCH_SIZE = 128


def get_best_gpu():

    if not torch.cuda.is_available():
        return torch.device("cpu")

    result = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )

    best_gpu = 0
    best_free = -1

    for line in result.strip().split("\n"):
        gpu_id, mem_used, mem_total = line.split(",")

        gpu_id = int(gpu_id.strip())
        mem_used = float(mem_used.strip())
        mem_total = float(mem_total.strip())

        free_mem = mem_total - mem_used

        print(
            f"GPU {gpu_id}: used={mem_used / 1024:.1f}GB free={free_mem / 1024:.1f}GB"
        )

        if free_mem > best_free:
            best_free = free_mem
            best_gpu = gpu_id

    print(f"\nUsing GPU {best_gpu}")

    return torch.device(f"cuda:{best_gpu}")


DEVICE = get_best_gpu()

print(f"Using device: {DEVICE}")
if DEVICE.type == "cuda":
    result = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )

    gpu_id = int(str(DEVICE).split(":")[1])

    for line in result.strip().split("\n"):
        idx, mem_used, mem_total = line.split(",")

        idx = int(idx.strip())

        if idx == gpu_id:
            mem_used = float(mem_used.strip())
            mem_total = float(mem_total.strip())

            free_gb = (mem_total - mem_used) / 1024

            print(f"GPU {gpu_id} free memory: {free_gb:.1f} GB")

            if free_gb < 8:
                raise RuntimeError(f"GPU {gpu_id} only has {free_gb:.1f} GB free.")

torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


# ==================================================
# TRANSFORMS
# ==================================================


def convert_rgb(img):
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


transform = transforms.Compose(
    [
        transforms.Lambda(convert_rgb),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)


# ==================================================
# MODEL DETECTION
# ==================================================


def detect_model_type(state_dict):

    keys = list(state_dict.keys())

    if any(k.startswith("patch_embed") for k in keys):
        return "faft"

    if any(k.startswith("frequency_branch") for k in keys):
        return "artifact_vit"

    if any(k.startswith("resnet.") for k in keys) and any(
        k.startswith("swin.") for k in keys
    ):
        return "hybrid_swin"

    if any(k.startswith("resnet.") for k in keys) and any(
        k.startswith("vit.") for k in keys
    ):
        return "hybrid"

    if "class_token" in keys:
        return "vit"

    if "head.weight" in keys and any(k.startswith("features.") for k in keys):
        return "swin"

    if "conv1.weight" in keys:
        return "resnet18"

    if "features.0.0.weight" in keys and "classifier.1.weight" in keys:
        return "efficientnet"

    raise ValueError("Unable to determine model type")


# ==================================================
# LOAD MODEL
# ==================================================

print("\nLoading checkpoint...")

checkpoint = torch.load(
    MODEL_PATH,
    map_location="cpu",
    weights_only=True,
)

model_name = detect_model_type(checkpoint)

print(f"Detected model: {model_name}")

model = load_model(model_name)

model.load_state_dict(
    checkpoint,
    strict=True,
)

model = model.to(DEVICE)
model.eval()
if DEVICE.type == "cuda":
    torch.cuda.synchronize()

    print(f"Model memory: {torch.cuda.memory_allocated(DEVICE) / 1024**3:.2f} GB")
if DEVICE.type == "cuda":
    print("Skipping torch.compile for FAFT")


# ==================================================
# EVALUATION FUNCTION
# ==================================================


def evaluate_dataset(dataset_name):

    print("\n" + "=" * 80)
    print(f"EVALUATING: {dataset_name}")
    print("=" * 80)

    dataset_path = Path(HOLDOUT_ROOT) / dataset_name

    dataset = datasets.ImageFolder(
        root=dataset_path,
        transform=transform,
    )
    print(f"Dataset size : {len(dataset):,} images")

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    print(f"Batches      : {len(loader):,}")
    print(f"Batch size   : {BATCH_SIZE}")

    criterion = nn.CrossEntropyLoss()

    all_labels = []
    all_predictions = []
    all_probabilities = []

    total_loss = 0.0
    total_samples = 0

    start_time = time.time()

    with torch.inference_mode():
        pbar = tqdm(
            loader,
            desc=dataset_name,
            unit="batch",
        )

        for batch_idx, (images, labels) in enumerate(pbar):
            images = images.to(
                DEVICE,
                non_blocking=True,
            )

            labels = labels.to(
                DEVICE,
                non_blocking=True,
            )
            batch_start = time.time()
            with torch.amp.autocast(
                "cuda",
                enabled=(DEVICE.type == "cuda"),
            ):
                outputs = model(images)

                loss = criterion(
                    outputs,
                    labels,
                )

            predictions = outputs.argmax(dim=1)

            probabilities = torch.softmax(outputs.float(), dim=1)[:, 1]

            all_labels.extend(labels.cpu().numpy())

            all_predictions.extend(predictions.cpu().numpy())

            all_probabilities.extend(probabilities.cpu().numpy())

            total_loss += loss.item() * labels.size(0)

            total_samples += labels.size(0)
            if DEVICE.type == "cuda":
                torch.cuda.synchronize(DEVICE)

            batch_time = time.time() - batch_start

            gpu_mem = (
                torch.cuda.memory_allocated(DEVICE) / 1024**3
                if DEVICE.type == "cuda"
                else 0
            )

            throughput = labels.size(0) / max(batch_time, 1e-6)

            percent = 100.0 * total_samples / len(dataset)

            pbar.set_postfix(
                done=f"{percent:.1f}%",
                processed=f"{total_samples:,}",
                gpu=f"{gpu_mem:.1f}GB",
                img_s=f"{throughput:.1f}",
            )
            if batch_idx % 25 == 0:
                elapsed_now = time.time() - start_time

                eta = (len(dataset) - total_samples) / max(
                    total_samples / elapsed_now, 1e-6
                )

                print(
                    f"\n[{dataset_name}] "
                    f"{total_samples:,}/{len(dataset):,} "
                    f"({100 * total_samples / len(dataset):.2f}%) "
                    f"| ETA: {eta / 60:.1f} min "
                    f"| GPU: {gpu_mem:.1f} GB"
                )

    elapsed = time.time() - start_time

    accuracy = accuracy_score(
        all_labels,
        all_predictions,
    )

    precision = precision_score(
        all_labels,
        all_predictions,
        zero_division=0,
    )

    recall = recall_score(
        all_labels,
        all_predictions,
        zero_division=0,
    )

    f1 = f1_score(
        all_labels,
        all_predictions,
        zero_division=0,
    )

    try:
        auc = roc_auc_score(
            all_labels,
            all_probabilities,
        )
    except Exception:
        auc = 0.0

    cm = confusion_matrix(
        all_labels,
        all_predictions,
    )

    print("\nDataset Statistics")
    print("------------------")
    print(f"Images     : {total_samples}")
    print(f"Loss       : {total_loss / total_samples:.6f}")
    print(f"Accuracy   : {accuracy:.6f}")
    print(f"Precision  : {precision:.6f}")
    print(f"Recall     : {recall:.6f}")
    print(f"F1 Score   : {f1:.6f}")
    print(f"ROC-AUC    : {auc:.6f}")

    print("\nPerformance")
    print("-----------")
    print(f"Time       : {elapsed:.2f} sec")
    print(f"Throughput : {total_samples / elapsed:.2f} img/s")

    print("\nGround Truth Distribution")
    print(Counter(all_labels))

    print("\nPrediction Distribution")
    print(Counter(all_predictions))

    print("\nConfusion Matrix")
    print(cm)

    print("\nClassification Report")
    print(
        classification_report(
            all_labels,
            all_predictions,
            zero_division=0,
        )
    )

    return {
        "dataset": dataset_name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc,
        "images": total_samples,
    }


# ==================================================
# MAIN
# ==================================================


def main():

    holdout_sets = [
        "DeepFakeDetection",
        "FaceShifter",
    ]

    results = []

    for dataset_name in holdout_sets:
        result = evaluate_dataset(dataset_name)

        results.append(result)

    print("\n")
    print("=" * 80)
    print("FINAL HOLDOUT SUMMARY")
    print("=" * 80)

    print(
        f"{'Dataset':20}"
        f"{'Images':>12}"
        f"{'Acc':>12}"
        f"{'Prec':>12}"
        f"{'Recall':>12}"
        f"{'F1':>12}"
        f"{'AUC':>12}"
    )

    for r in results:
        print(
            f"{r['dataset']:20}"
            f"{r['images']:>12}"
            f"{r['accuracy']:>12.4f}"
            f"{r['precision']:>12.4f}"
            f"{r['recall']:>12.4f}"
            f"{r['f1']:>12.4f}"
            f"{r['auc']:>12.4f}"
        )

    print("=" * 80)


if __name__ == "__main__":
    main()
