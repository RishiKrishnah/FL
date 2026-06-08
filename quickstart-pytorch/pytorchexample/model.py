import torch
import torch.nn as nn

from torchvision.models import (
    vit_b_16,
    resnet18,
    efficientnet_b0
)


class HybridResNetViT(nn.Module):

    def __init__(self):

        super().__init__()

        # ResNet branch
        self.resnet = resnet18(
            weights="IMAGENET1K_V1"
        )

        self.resnet.fc = nn.Identity()

        # ViT branch
        self.vit = vit_b_16(
            weights="IMAGENET1K_V1"
        )

        self.vit.heads = nn.Identity()

        # Optional: freeze ViT to reduce training cost
        for param in self.vit.parameters():
            param.requires_grad = False

        # Unfreeze last transformer block
        for param in self.vit.encoder.layers[-1].parameters():
            param.requires_grad = True

        # Fusion classifier
        self.classifier = nn.Sequential(
            nn.Linear(512 + 768, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 2)
        )

    def forward(self, x):

        resnet_features = self.resnet(x)

        vit_features = self.vit(x)

        combined = torch.cat(
            [resnet_features, vit_features],
            dim=1
        )

        return self.classifier(combined)


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

    elif model_name == "hybrid":

        model = HybridResNetViT()

    else:

        raise ValueError(
            f"Unknown model: {model_name}"
        )

    return model