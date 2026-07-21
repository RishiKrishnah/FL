import shutil
import random
from pathlib import Path

random.seed(42)

SOURCE = Path("/home/ex5/research/rishi/dataset/real_vs_fake/real-vs-fake")

DEST = Path("dataset")

CLIENTS = 5


def split_images(images, clients):

    random.shuffle(images)
    size = len(images) // clients
    splits = []

    for i in range(clients):
        start = i * size

        if i == clients - 1:
            end = len(images)
        else:
            end = (i + 1) * size

        splits.append(images[start:end])

    return splits


for split in ["train", "valid", "test"]:
    for label in ["real", "fake"]:
        images = list((SOURCE / split / label).glob("*"))
        client_splits = split_images(images, CLIENTS)

        for client_id, subset in enumerate(client_splits, start=1):
            target_dir = (
                DEST
                / f"client{client_id}"
                / ("val" if split == "valid" else split)
                / label
            )

            target_dir.mkdir(parents=True, exist_ok=True)

            for img in subset:
                shutil.copy2(img, target_dir / img.name)

print("Done.")
