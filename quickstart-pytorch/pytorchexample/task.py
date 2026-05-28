import os

import torch

from torch.utils.data import DataLoader

from torchvision import (
    datasets,
    transforms
)

import torch.nn as nn
import torch.optim as optim


DEVICE = torch.device(

    "cuda"

    if torch.cuda.is_available()

    else "cpu"
)

print(f"Using device: {DEVICE}")


transform = transforms.Compose([

    transforms.Lambda(
        lambda img: img.convert("RGB")
    ),

    transforms.Resize((64, 64)),

    transforms.ToTensor(),
])


def load_data(client_id):

    train_path = (
        f"dataset/client{client_id}/train"
    )

    test_path = (
        f"dataset/client{client_id}/test"
    )

    trainset = datasets.ImageFolder(

        train_path,

        transform=transform
    )

    testset = datasets.ImageFolder(

        test_path,

        transform=transform
    )

    trainloader = DataLoader(

        trainset,

        batch_size=8,

        shuffle=True,

        num_workers=0
    )

    testloader = DataLoader(

        testset,

        batch_size=8,

        shuffle=False,

        num_workers=0
    )

    return trainloader, testloader


def train(model, trainloader, epochs=1):

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(

        model.parameters(),

        lr=0.001
    )

    model.train()

    for epoch in range(epochs):

        running_loss = 0.0

        for batch_idx, (
            images,
            labels
        ) in enumerate(trainloader):

            print(

                f"Batch "
                f"{batch_idx+1}/"
                f"{len(trainloader)}"
            )

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

        print(

            f"Epoch {epoch+1} "
            f"Loss: "
            f"{running_loss:.4f}"
        )


def test(model, testloader):

    criterion = nn.CrossEntropyLoss()

    correct = 0
    total = 0
    loss = 0.0

    model.eval()

    with torch.no_grad():

        for images, labels in testloader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)

            loss += criterion(
                outputs,
                labels
            ).item()

            _, predicted = torch.max(
                outputs,
                1
            )

            total += labels.size(0)

            correct += (

                predicted == labels

            ).sum().item()

    accuracy = correct / total

    print(
        f"Accuracy: "
        f"{accuracy:.4f}"
    )

    return loss, accuracy


def save_model(model, round_num):

    os.makedirs(
        "saved_models",
        exist_ok=True
    )

    path = (

        f"saved_models/"
        f"global_model_round_"
        f"{round_num}.pth"
    )

    torch.save(

        model.state_dict(),

        path
    )

    print(f"Saved: {path}")