import torch
import torch.nn as nn

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
from pytorchexample.task import load_data, DEVICE


MODEL_PATH = (
    "saved_models/global_model_round_3.pth"
)


def detect_model_type(state_dict):

    keys = list(state_dict.keys())

    if "class_token" in keys:
        return "vit"

    if "conv1.weight" in keys:
        return "resnet18"

    if "features.0.0.weight" in keys:
        return "efficientnet"

    raise ValueError(
        "Unable to determine model type."
    )


def evaluate():

    print("\nLoading checkpoint...")

    checkpoint = torch.load(
        MODEL_PATH,
        map_location="cpu"
    )

    model_name = detect_model_type(
        checkpoint
    )

    print(
        f"Detected model: {model_name}"
    )

    model = load_model(
        model_name
    )

    model.load_state_dict(
        checkpoint
    )

    model = model.to(
        DEVICE
    )

    model.eval()

    criterion = nn.CrossEntropyLoss()

    all_labels = []
    all_predictions = []
    all_probabilities = []

    total_loss = 0.0
    total_samples = 0

    print("\nStarting evaluation...")

    for client_id in [1, 2, 3]:

        print(
            f"\nEvaluating Client {client_id}"
        )

        _, _, testloader = load_data(
            client_id
        )

        with torch.no_grad():

            for images, labels in testloader:

                images = images.to(
                    DEVICE
                )

                labels = labels.to(
                    DEVICE
                )

                outputs = model(
                    images
                )

                loss = criterion(
                    outputs,
                    labels
                )

                total_loss += (
                    loss.item()
                    * labels.size(0)
                )

                total_samples += (
                    labels.size(0)
                )

                probabilities = (
                    torch.softmax(
                        outputs,
                        dim=1
                    )[:, 1]
                )

                predictions = (
                    outputs.argmax(
                        dim=1
                    )
                )

                all_labels.extend(
                    labels.cpu().numpy()
                )

                all_predictions.extend(
                    predictions.cpu().numpy()
                )

                all_probabilities.extend(
                    probabilities
                    .cpu()
                    .numpy()
                )

    from collections import Counter

    print("\nGround Truth Distribution:")
    print(Counter(all_labels))

    print("\nPrediction Distribution:")
    print(Counter(all_predictions))

    avg_loss = (
        total_loss
        / total_samples
    )

    accuracy = accuracy_score(
        all_labels,
        all_predictions
    )

    precision = precision_score(
        all_labels,
        all_predictions,
        average="binary",
        zero_division=0
    )

    recall = recall_score(
        all_labels,
        all_predictions,
        average="binary",
        zero_division=0
    )

    f1 = f1_score(
        all_labels,
        all_predictions,
        average="binary",
        zero_division=0
    )

    if len(set(all_labels)) > 1:
        auc = roc_auc_score(
            all_labels,
            all_probabilities
        )
    else:
        auc = float("nan")

    cm = confusion_matrix(
        all_labels,
        all_predictions
    )

    print("\n")
    print("=" * 70)
    print("FINAL EVALUATION RESULTS")
    print("=" * 70)

    print(
        f"Model      : {model_name}"
    )

    print(
        f"Loss       : {avg_loss:.6f}"
    )

    print(
        f"Accuracy   : {accuracy:.6f}"
    )

    print(
        f"Precision  : {precision:.6f}"
    )

    print(
        f"Recall     : {recall:.6f}"
    )

    print(
        f"F1 Score   : {f1:.6f}"
    )

    print(
        f"ROC-AUC    : {auc:.6f}"
    )

    print("\nConfusion Matrix")

    print(cm)

    print("\nClassification Report")

    print(
        classification_report(
            all_labels,
            all_predictions,
            labels=[0, 1],
            target_names=[
                "Fake",
                "Real"
            ],
            zero_division=0
        )
    )

    print("=" * 70)


if __name__ == "__main__":
    evaluate()