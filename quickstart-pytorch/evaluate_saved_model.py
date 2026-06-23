import time
from collections import Counter

import torch
import torch.nn as nn
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
from pytorchexample.task import load_data
from pytorchexample.gpu_manager import GPUManager

DEVICE = GPUManager().get_best_gpu()
print(f"Using device: {DEVICE}")

# ==================================================
# SETTINGS
# ==================================================
NUM_CLIENTS = 5
MODEL_PATH = "saved_models/global_model_round_5.pth"

# H100 can easily handle this
EVAL_BATCH_SIZE = 256

torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


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
# EVALUATION
# ==================================================
def evaluate():

    print("\nLoading checkpoint...")

    checkpoint = torch.load(
        MODEL_PATH,
        map_location="cpu",
        weights_only=True,
    )

    model_name = detect_model_type(checkpoint)

    print(f"Detected model: {model_name}")

    model = load_model(model_name)
    model.load_state_dict(checkpoint)

    model = model.to(DEVICE)
    model = model.to(memory_format=torch.channels_last)

    model.eval()

    # Compile model (PyTorch 2.x)
    if DEVICE.type == "cuda":
        print("Compiling model...")
        model = torch.compile(model)

    print("\nWarming up GPU...")

    dummy = torch.randn(
        32,
        3,
        224,
        224,
        dtype=torch.float32,
        device=DEVICE,
    )

    dummy = dummy.to(memory_format=torch.channels_last)

    with torch.inference_mode():
        with torch.amp.autocast(
            "cuda",
            enabled=(DEVICE.type == "cuda"),
        ):
            for _ in range(5):
                _ = model(dummy)

    if DEVICE.type == "cuda":
        torch.cuda.synchronize(DEVICE)

    print("Warmup complete.\n")
    if DEVICE.type == "cuda":
        torch.cuda.reset_peak_memory_stats(DEVICE)
    overall_start = time.time()
    criterion = nn.CrossEntropyLoss()

    all_labels = []
    all_predictions = []
    all_probabilities = []

    total_loss = 0.0
    total_samples = 0

    client_bar = tqdm(
        range(1, NUM_CLIENTS + 1),
        desc="Clients",
    )
    with torch.inference_mode():
        for client_id in client_bar:
            print(f"\nEvaluating Client {client_id}")

            _, _, old_loader = load_data(client_id)

            # -----------------------------------
            # REBUILD TESTLOADER
            # -----------------------------------
            testloader = DataLoader(
                old_loader.dataset,
                batch_size=EVAL_BATCH_SIZE,
                shuffle=False,
                num_workers=0,
                pin_memory=True,
            )

            print("Batch size:", testloader.batch_size)
            print("Batches:", len(testloader))

            processed = 0
            client_start = time.time()
            num_batches = len(testloader)
            batch_bar = tqdm(
                testloader,
                desc=f"Client {client_id}",
                leave=False,
            )

            for images, labels in batch_bar:
                images = images.to(
                    DEVICE,
                    dtype=torch.float32,
                    non_blocking=True,
                    memory_format=torch.channels_last,
                )

                labels = labels.to(
                    DEVICE,
                    non_blocking=True,
                )

                infer_start = time.perf_counter()

                with torch.amp.autocast(
                    "cuda",
                    enabled=(DEVICE.type == "cuda"),
                ):
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                if DEVICE.type == "cuda":
                    torch.cuda.synchronize(DEVICE)

                infer_time = time.perf_counter() - infer_start
                predictions = outputs.detach().argmax(dim=1).cpu().numpy()

                probabilities = (
                    torch.softmax(outputs.float(), dim=1)[:, 1].detach().cpu().numpy()
                )

                all_labels.extend(labels.detach().cpu().numpy())
                all_predictions.extend(predictions)
                all_probabilities.extend(probabilities)
                total_loss += loss.item() * labels.size(0)
                total_samples += labels.size(0)
                processed += labels.size(0)
                elapsed = time.time() - client_start
                img_s = processed / elapsed

                gpu_alloc = (
                    torch.cuda.memory_allocated(DEVICE) / 1024**3
                    if DEVICE.type == "cuda"
                    else 0
                )

                gpu_reserved = (
                    torch.cuda.memory_reserved(DEVICE) / 1024**3
                    if DEVICE.type == "cuda"
                    else 0
                )

                eta = (num_batches - batch_bar.n - 1) * infer_time
                batch_bar.set_postfix(
                    loss=f"{loss.item():.4f}",
                    img_s=f"{img_s:.1f}",
                    infer_ms=f"{infer_time * 1000:.1f}",
                    eta=f"{eta:.1f}s",
                    alloc=f"{gpu_alloc:.1f}GB",
                    reserv=f"{gpu_reserved:.1f}GB",
                )

            client_time = time.time() - client_start

            print(
                f"Client {client_id}: "
                f"{processed} images "
                f"in {client_time:.2f}s "
                f"({processed / client_time:.1f} img/s)"
            )
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    avg_loss = total_loss / total_samples

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

    auc = roc_auc_score(
        all_labels,
        all_probabilities,
    )

    cm = confusion_matrix(
        all_labels,
        all_predictions,
    )

    total_time = time.time() - overall_start

    if DEVICE.type == "cuda":
        peak_gpu_alloc = torch.cuda.max_memory_allocated(DEVICE) / 1024**3

        peak_gpu_reserved = torch.cuda.max_memory_reserved(DEVICE) / 1024**3

    print("\nGround Truth Distribution")
    print(Counter(all_labels))

    print("\nPrediction Distribution")
    print(Counter(all_predictions))

    print("\n" + "=" * 80)
    print("FINAL EVALUATION RESULTS")
    print("=" * 80)

    print(f"Model      : {model_name}")
    print(f"Loss       : {avg_loss:.6f}")
    print(f"Accuracy   : {accuracy:.6f}")
    print(f"Precision  : {precision:.6f}")
    print(f"Recall     : {recall:.6f}")
    print(f"F1 Score   : {f1:.6f}")
    print(f"ROC-AUC    : {auc:.6f}")

    print("\nPerformance")
    print(f"Images     : {total_samples}")
    print(f"Time       : {total_time:.2f} sec")
    print(f"Throughput : {total_samples / total_time:.1f} img/s")

    print(f"Peak GPU allocated : {peak_gpu_alloc:.2f} GB")
    print(f"Peak GPU reserved  : {peak_gpu_reserved:.2f} GB")

    print("\nConfusion Matrix")
    print(cm)

    print("\nClassification Report")

    print(
        classification_report(
            all_labels,
            all_predictions,
            labels=[0, 1],
            target_names=["Fake", "Real"],
            zero_division=0,
        )
    )

    print("=" * 80)


if __name__ == "__main__":
    evaluate()
