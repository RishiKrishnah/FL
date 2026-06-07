import torch.nn as nn

from torchvision.models import (
    vit_b_16,
    resnet18,
    efficientnet_b0
)

def load_model(model_name):

    if model_name == "vit":

        model = vit_b_16(
            weights="IMAGENET1K_V1"
        )

        for param in model.parameters():
            param.requires_grad = False

        model.heads.head = nn.Linear(
            model.heads.head.in_features,
            2
        )

    elif model_name == "resnet18":

        model = resnet18(
            weights="IMAGENET1K_V1"
        )

        model.fc = nn.Linear(
            model.fc.in_features,
            2
        )

    elif model_name == "efficientnet":

        model = efficientnet_b0(
            weights="IMAGENET1K_V1"
        )

        model.classifier[1] = nn.Linear(
            model.classifier[1].in_features,
            2
        )

    else:

        raise ValueError(
            f"Unknown model: {model_name}"
        )

    return model