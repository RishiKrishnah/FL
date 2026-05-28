import torch.nn as nn

from torchvision.models import (
    mobilenet_v2
)


def load_model():

    model = mobilenet_v2(
        weights=None
    )

    model.classifier[1] = nn.Linear(

        model.last_channel,

        2
    )

    return model