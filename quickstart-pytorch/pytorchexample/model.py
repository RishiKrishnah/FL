import torch
import torch.nn as nn

from torchvision.models import (
    vit_b_16,
    resnet18,
    efficientnet_b0,
    swin_t,
)


class ArtifactAttention(nn.Module):
    def __init__(self, feature_dim=512):
        super().__init__()

        self.attention = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim),
            nn.Sigmoid(),
        )

    def forward(self, x):

        attention_map = self.attention(x)
        guided_features = x * attention_map

        return guided_features, attention_map


class FAFTAttention(nn.Module):
    def __init__(
        self,
        embed_dim=768,
        num_heads=8,
        dropout=0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, artifact_bias=None):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(
                B,
                N,
                3,
                self.num_heads,
                self.head_dim,
            )
            .permute(2, 0, 3, 1, 4)
        )

        q, k, v = qkv[0], qkv[1], qkv[2]
        attention = (q @ k.transpose(-2, -1)) * self.scale
        if artifact_bias is not None:
            attention = attention + artifact_bias
        attention = attention.softmax(dim=-1)
        attention = self.dropout(attention)
        out = attention @ v
        out = out.transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        return out


class FAFTBlock(nn.Module):
    def __init__(
        self,
        embed_dim=768,
        num_heads=8,
        mlp_ratio=4,
        dropout=0.1,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)

        self.attn = FAFTAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(
                embed_dim,
                embed_dim * mlp_ratio,
            ),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(
                embed_dim * mlp_ratio,
                embed_dim,
            ),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x,
        artifact_bias=None,
    ):
        x = x + self.attn(
            self.norm1(x),
            artifact_bias,
        )
        x = x + self.mlp(self.norm2(x))
        return x


class ArtifactBiasGenerator(nn.Module):
    def __init__(
        self,
        artifact_dim=768,
        num_heads=8,
        num_tokens=197,
    ):
        super().__init__()

        self.num_heads = num_heads
        self.num_tokens = num_tokens

        self.bias = nn.Sequential(
            nn.Linear(
                artifact_dim,
                512,
            ),
            nn.ReLU(),
            nn.Linear(
                512,
                num_heads,
            ),
        )

    def forward(self, artifact_embedding):

        B = artifact_embedding.size(0)
        bias = self.bias(artifact_embedding)

        bias = bias.view(
            B,
            self.num_heads,
            1,
            1,
        )

        return bias


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


class FAFTNet(nn.Module):
    def __init__(self):
        super().__init__()

        # ----------------------------------
        # Spatial Branch
        # ----------------------------------
        self.resnet = resnet18(weights="IMAGENET1K_V1")
        self.resnet.fc = nn.Identity()

        # ----------------------------------
        # Frequency Branch
        # ----------------------------------
        self.frequency_branch = FrequencyBranch()

        # ----------------------------------
        # Artifact Fusion
        # ----------------------------------
        self.fusion = nn.Sequential(
            nn.Linear(512 + 128, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

        # ----------------------------------
        # Artifact Attention
        # ----------------------------------
        self.artifact_attention = ArtifactAttention(feature_dim=512)

        # ----------------------------------
        # Projection
        # ----------------------------------
        self.artifact_projection = nn.Sequential(
            nn.Linear(512, 768),
            nn.LayerNorm(768),
            nn.ReLU(),
        )

        # ----------------------------------
        # Artifact Bias Generator
        # ----------------------------------
        self.bias_generator = ArtifactBiasGenerator(
            artifact_dim=768,
            num_heads=8,
            num_tokens=197,
        )

        # ----------------------------------
        # CLS Token
        # ----------------------------------
        self.cls_token = nn.Parameter(torch.randn(1, 1, 768))

        # ----------------------------------
        # Patch Projection
        # ----------------------------------
        self.patch_embed = nn.Conv2d(
            3,
            768,
            kernel_size=16,
            stride=16,
        )

        # ----------------------------------
        # Position Embedding
        # ----------------------------------
        self.pos_embed = nn.Parameter(torch.randn(1, 197, 768))

        # ----------------------------------
        # FAFT Transformer
        # ----------------------------------
        self.blocks = nn.ModuleList(
            [
                FAFTBlock(
                    embed_dim=768,
                    num_heads=8,
                )
                for _ in range(4)
            ]
        )

        self.norm = nn.LayerNorm(768)

        # ----------------------------------
        # Classifier
        # ----------------------------------
        self.classifier = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 2),
        )

        self.current_prototype = None
        self.global_memory = None

    def forward(self, x):

        # ----------------------------------
        # Spatial Features
        # ----------------------------------
        Fs = self.resnet(x)

        # ----------------------------------
        # Frequency Features
        # ----------------------------------
        Ff = self.frequency_branch(x)

        # ----------------------------------
        # Fusion
        # ----------------------------------
        artifact_features = self.fusion(torch.cat([Fs, Ff], dim=1))

        # ----------------------------------
        # Artifact Attention
        # ----------------------------------
        guided_artifacts, _ = self.artifact_attention(artifact_features)
        artifact_embedding = self.artifact_projection(guided_artifacts)
        self.current_prototype = artifact_embedding.detach().mean(dim=0)

        # ----------------------------------
        # Generate Bias
        # ----------------------------------
        artifact_bias = self.bias_generator(artifact_embedding)

        # ----------------------------------
        # Patch Embedding
        # ----------------------------------
        patches = self.patch_embed(x)

        patches = patches.flatten(2)

        patches = patches.transpose(1, 2)

        B = patches.size(0)

        cls_tokens = self.cls_token.expand(
            B,
            -1,
            -1,
        )

        tokens = torch.cat(
            [
                cls_tokens,
                patches,
            ],
            dim=1,
        )

        tokens = tokens + self.pos_embed

        # ----------------------------------
        # Global Memory Bias
        # ----------------------------------
        if self.global_memory is not None:
            memory_bias = self.bias_generator(
                self.global_memory.unsqueeze(0).expand(B, -1)
            )

            artifact_bias = artifact_bias + memory_bias

        # ----------------------------------
        # Transformer
        # ----------------------------------
        for block in self.blocks:
            tokens = block(
                tokens,
                artifact_bias,
            )

        tokens = self.norm(tokens)

        cls_feature = tokens[:, 0]

        logits = self.classifier(cls_feature)

        return logits

    def get_prototype(self):
        return self.current_prototype

    def set_global_memory(self, memory):
        self.global_memory = memory


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

    elif model_name == "faft":
        model = FAFTNet()

    else:
        raise ValueError(f"Unknown model: {model_name}")

    return model
