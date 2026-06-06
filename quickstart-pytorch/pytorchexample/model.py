import torch.nn as nn

from torchvision.models import vit_b_16


def load_model():

    model = vit_b_16(
        weights="IMAGENET1K_V1"
    )

    in_features = (
        model.heads.head.in_features
    )

    model.heads.head = nn.Linear(
        in_features,
        2
    )

    return model