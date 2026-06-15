import torch
import torch.nn as nn

from torchvision.models import (
    vit_b_16,
    resnet18,
    efficientnet_b0,
    swin_t,
)


class FrequencyBranch(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

    def forward(self, x):
        fft = torch.fft.fft2(x)
        fft = torch.abs(fft)
        fft = torch.log1p(fft)
        out = self.features(fft)
        return out.flatten(1)


class ArtifactGuidedDeepfakeNet(nn.Module):
    def __init__(self):

        super().__init__()

        # =====================================================
        # Spatial Branch (ResNet18)
        # =====================================================
        self.resnet = resnet18(weights="IMAGENET1K_V1")
        self.resnet.fc = nn.Identity()

        # =====================================================
        # Frequency Branch
        # =====================================================
        self.frequency_branch = FrequencyBranch()

        # =====================================================
        # ViT Branch
        # =====================================================
        self.vit = vit_b_16(weights="IMAGENET1K_V1")
        self.vit.heads = nn.Identity()

        # Freeze ViT
        for param in self.vit.parameters():
            param.requires_grad = False

        # Fine-tune last transformer block
        for param in self.vit.encoder.layers[-1].parameters():
            param.requires_grad = True

        # =====================================================
        # Artifact Fusion
        # =====================================================
        self.fusion = nn.Sequential(
            nn.Linear(512 + 128, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

        # =====================================================
        # Artifact Attention Module
        # =====================================================
        self.attention = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.Sigmoid(),
        )

        # =====================================================
        # Artifact Projection
        # Maps forensic representation to transformer space
        # =====================================================
        self.artifact_projection = nn.Sequential(
            nn.Linear(512, 768),
            nn.LayerNorm(768),
            nn.ReLU(),
        )

        # =====================================================
        # Final Classifier
        # Artifact Embedding (768)
        # +
        # ViT Embedding (768)
        # =====================================================
        self.classifier = nn.Sequential(
            nn.Linear(768 + 768, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 2),
        )

    def forward(self, x):

        # -----------------------------------------------------
        # Spatial Features
        # -----------------------------------------------------
        Fs = self.resnet(x)

        # -----------------------------------------------------
        # Frequency Features
        # -----------------------------------------------------
        Ff = self.frequency_branch(x)

        # -----------------------------------------------------
        # Artifact Fusion
        # -----------------------------------------------------
        artifact_features = self.fusion(torch.cat([Fs, Ff], dim=1))

        # -----------------------------------------------------
        # Artifact Attention
        # -----------------------------------------------------
        attention_map = self.attention(artifact_features)

        guided_artifacts = artifact_features * attention_map

        # -----------------------------------------------------
        # Project Artifact Features
        # into Transformer Feature Space
        # -----------------------------------------------------
        artifact_embedding = self.artifact_projection(guided_artifacts)

        # -----------------------------------------------------
        # Global Semantic Features
        # -----------------------------------------------------
        vit_features = self.vit(x)

        # -----------------------------------------------------
        # Final Fusion
        # -----------------------------------------------------
        final_features = torch.cat(
            [
                artifact_embedding,
                vit_features,
            ],
            dim=1,
        )

        logits = self.classifier(final_features)

        return logits


# ==========================================================
# ResNet18 + ViT Hybrid
# ==========================================================
class HybridResNetViT(nn.Module):
    def __init__(self):

        super().__init__()

        # ResNet branch
        self.resnet = resnet18(weights="IMAGENET1K_V1")
        self.resnet.fc = nn.Identity()

        # ViT branch
        self.vit = vit_b_16(weights="IMAGENET1K_V1")
        self.vit.heads = nn.Identity()

        # Freeze ViT
        for param in self.vit.parameters():
            param.requires_grad = False

        # Fine-tune last transformer block
        for param in self.vit.encoder.layers[-1].parameters():
            param.requires_grad = True

        self.classifier = nn.Sequential(
            nn.Linear(512 + 768, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 2),
        )

    def forward(self, x):

        resnet_features = self.resnet(x)

        vit_features = self.vit(x)

        combined = torch.cat(
            [resnet_features, vit_features],
            dim=1,
        )

        return self.classifier(combined)


# ==========================================================
# ResNet18 + Swin Transformer Hybrid
# ==========================================================
class HybridResNetSwin(nn.Module):
    def __init__(self):

        super().__init__()

        # ResNet branch
        self.resnet = resnet18(weights="IMAGENET1K_V1")
        self.resnet.fc = nn.Identity()

        # Swin branch
        self.swin = swin_t(weights="IMAGENET1K_V1")
        self.swin.head = nn.Identity()

        # Freeze Swin
        for param in self.swin.parameters():
            param.requires_grad = False

        # Fine-tune final Swin stage
        for param in self.swin.features[-1].parameters():
            param.requires_grad = True

        self.classifier = nn.Sequential(
            nn.Linear(512 + 768, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 2),
        )

    def forward(self, x):

        resnet_features = self.resnet(x)

        swin_features = self.swin(x)

        combined = torch.cat(
            [resnet_features, swin_features],
            dim=1,
        )

        return self.classifier(combined)


# ==========================================================
# Model Loader
# ==========================================================
def load_model(model_name):

    # ------------------------------------------------------
    # Vision Transformer
    # ------------------------------------------------------
    if model_name == "vit":
        model = vit_b_16(weights="IMAGENET1K_V1")

        for param in model.parameters():
            param.requires_grad = False

        model.heads.head = nn.Linear(
            model.heads.head.in_features,
            2,
        )

    # ------------------------------------------------------
    # ResNet18
    # ------------------------------------------------------
    elif model_name == "resnet18":
        model = resnet18(weights="IMAGENET1K_V1")

        model.fc = nn.Linear(
            model.fc.in_features,
            2,
        )

    # ------------------------------------------------------
    # EfficientNet-B0
    # ------------------------------------------------------
    elif model_name == "efficientnet":
        model = efficientnet_b0(weights="IMAGENET1K_V1")

        model.classifier[1] = nn.Linear(
            model.classifier[1].in_features,
            2,
        )

    # ------------------------------------------------------
    # Swin Transformer
    # ------------------------------------------------------
    elif model_name == "swin":
        model = swin_t(weights="IMAGENET1K_V1")

        model.head = nn.Linear(
            model.head.in_features,
            2,
        )

    # ------------------------------------------------------
    # ResNet + ViT Hybrid
    # ------------------------------------------------------
    elif model_name == "hybrid":
        model = HybridResNetViT()

    # ------------------------------------------------------
    # ResNet + Swin Hybrid
    # ------------------------------------------------------
    elif model_name == "hybrid_swin":
        model = HybridResNetSwin()

    elif model_name == "artifact_vit":
        model = ArtifactGuidedDeepfakeNet()

    else:
        raise ValueError(f"Unknown model: {model_name}")

    return model
