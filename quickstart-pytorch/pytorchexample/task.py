import os

import torch
torch.backends.cudnn.benchmark = True

from torch.utils.data import (
    DataLoader,
    Subset
)

from torchvision import (
    datasets,
    transforms
)

import torch.nn as nn
import torch.optim as optim

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score
)

DEVICE = torch.device(

    "cuda"

    if torch.cuda.is_available()

    else "cpu"
)

print(f"Using device: {DEVICE}")

try:
    if DEVICE.type == "cuda":
        print("GPU Enabled")
        print(
            f"GPU Count: {torch.cuda.device_count()}"
        )

        if torch.cuda.device_count() > 0:
            print(
                torch.cuda.get_device_name(
                    torch.cuda.current_device()
                )
            )

except Exception as e:
    print(
        f"GPU info unavailable: {e}"
    )
from PIL import Image

DEBUG = True

def convert_rgb(img):
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img

transform = transforms.Compose([
    transforms.Lambda(convert_rgb),

    transforms.Resize((224, 224)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def load_data(client_id):

    train_path = f"dataset/client{client_id}/train"
    val_path   = f"dataset/client{client_id}/val"
    test_path  = f"dataset/client{client_id}/test"

    trainset = datasets.ImageFolder(
    train_path,
    transform=transform
)

    valset = datasets.ImageFolder(
        val_path,
        transform=transform
    )

    testset = datasets.ImageFolder(
        test_path,
        transform=transform
    )

    # DEBUG MODE
    if DEBUG:

        import random

        train_limit = min(2000, len(trainset))
        val_limit = min(500, len(valset))
        test_limit = min(500, len(testset))

        train_indices = random.sample(
            range(len(trainset)),
            train_limit
        )

        val_indices = random.sample(
            range(len(valset)),
            val_limit
        )

        test_indices = random.sample(
            range(len(testset)),
            test_limit
        )

        trainset = Subset(
            trainset,
            train_indices
        )

        valset = Subset(
            valset,
            val_indices
        )

        testset = Subset(
            testset,
            test_indices
        )

        print(
            f"[DEBUG MODE] "
            f"Train={len(trainset)}, "
            f"Val={len(valset)}, "
            f"Test={len(testset)}"
        )

    trainloader = DataLoader(
        trainset,
        batch_size=64,
        shuffle=True
    )

    valloader = DataLoader(
        valset,
        batch_size=64,
        shuffle=False
    )

    testloader = DataLoader(
        testset,
        batch_size=64,
        shuffle=False
    )

    print(
        f"Client {client_id}"
    )

    print(
        f"Train Images: "
        f"{len(trainset)}"
    )

    print(
        f"Test Images : "
        f"{len(testset)}"
    )
    print(
        f"Validate Images : "
        f"{len(valset)}"
    )
    return trainloader, valloader, testloader

from tqdm import tqdm
import time
import torch
import torch.nn as nn
import torch.optim as optim

def train(model, trainloader, epochs=1):

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        filter(
            lambda p: p.requires_grad,
            model.parameters()
        ),
        lr=0.001
    )

    scaler = torch.amp.GradScaler("cuda",enabled=(DEVICE.type == "cuda"))

    model.train()

    for epoch in range(epochs):

        running_loss = 0.0

        correct = 0
        total = 0

        start_time = time.time()

        progress_bar = tqdm(
            trainloader,
            desc=f"Epoch {epoch+1}/{epochs}",
            leave=True
        )

        for batch_idx, (images, labels) in enumerate(progress_bar):

            images = images.to(
                DEVICE,
                non_blocking=True
            )

            labels = labels.to(
                DEVICE,
                non_blocking=True
            )

            optimizer.zero_grad()

            with torch.amp.autocast("cuda", enabled=(DEVICE.type == "cuda")):

                outputs = model(images)

                loss = criterion(
                    outputs,
                    labels
                )

            scaler.scale(loss).backward()

            scaler.step(optimizer)

            scaler.update()

            running_loss += loss.item()

            _, predicted = torch.max(
                outputs,
                1
            )

            correct += (
                predicted == labels
            ).sum().item()

            total += labels.size(0)

            accuracy = (
                100.0 *
                correct /
                total
            )

            elapsed = (
                time.time()
                - start_time
            )

            images_processed = (
                (batch_idx + 1)
                * images.size(0)
            )

            img_per_sec = (
                images_processed
                / elapsed
            )

            eta_seconds = (
                elapsed /
                (batch_idx + 1)
            ) * (
                len(trainloader)
                - batch_idx
                - 1
            )

            progress_bar.set_postfix({
                "loss":
                    f"{loss.item():.4f}",
                "acc":
                    f"{accuracy:.2f}%",
                "img/s":
                    f"{img_per_sec:.1f}",
                "eta":
                    f"{eta_seconds/60:.1f}m"
            })

            if batch_idx % 100 == 0:

                if torch.cuda.is_available():

                    allocated = (
                        torch.cuda.memory_allocated()
                        / 1024**3
                    )

                    reserved = (
                        torch.cuda.memory_reserved()
                        / 1024**3
                    )

                    print(
                        f"\nBatch "
                        f"{batch_idx}/"
                        f"{len(trainloader)}"
                    )

                    print(
                        f"GPU Allocated: "
                        f"{allocated:.2f} GB"
                    )

                    print(
                        f"GPU Reserved : "
                        f"{reserved:.2f} GB"
                    )

        epoch_time = (
            time.time()
            - start_time
        )

        avg_loss = (
            running_loss
            / len(trainloader)
        )

        final_acc = (
            100.0 *
            correct /
            total
        )

        print("\n" + "=" * 50)

        print(
            f"Epoch {epoch+1} Complete"
        )

        print(
            f"Average Loss : "
            f"{avg_loss:.4f}"
        )

        print(
            f"Accuracy     : "
            f"{final_acc:.2f}%"
        )

        print(
            f"Time         : "
            f"{epoch_time:.2f}s"
        )

        print(
            f"Images       : "
            f"{total}"
        )

        print(
            f"Images/sec   : "
            f"{total/epoch_time:.2f}"
        )

        if torch.cuda.is_available():

            allocated = (
                torch.cuda.memory_allocated()
                / 1024**3
            )

            reserved = (
                torch.cuda.memory_reserved()
                / 1024**3
            )

            print(
                f"GPU Allocated: "
                f"{allocated:.2f} GB"
            )

            print(
                f"GPU Reserved : "
                f"{reserved:.2f} GB"
            )

        print("=" * 50)

def test(model, testloader):

    criterion = nn.CrossEntropyLoss()

    correct = 0
    total = 0
    loss = 0.0

    model.eval()

    with torch.no_grad():

        for images, labels in tqdm(testloader, desc="Testing",leave=True):

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