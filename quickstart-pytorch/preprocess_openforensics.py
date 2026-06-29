import random
from pathlib import Path
from PIL import Image
from tqdm import tqdm

random.seed(42)

# ======================================================
# CONFIGURATION
# ======================================================

SOURCE = Path("/root/Rishi/datasets/openforensics/Dataset")
DEST = Path("dataset_openforensic")

CLIENTS = 5
IMAGE_SIZE = (224, 224)

split_map = {
    "Train": "train",
    "Validation": "val",
    "Test": "test",
}

label_map = {
    "Real": "real",
    "Fake": "fake",
}


def split_images(images, clients):
    random.shuffle(images)

    size = len(images) // clients
    splits = []

    for i in range(clients):
        start = i * size
        end = len(images) if i == clients - 1 else (i + 1) * size
        splits.append(images[start:end])

    return splits


for src_split, dst_split in split_map.items():

    print(f"\n{'='*60}")
    print(f"{src_split}")
    print(f"{'='*60}")

    for src_label, dst_label in label_map.items():

        images = list((SOURCE / src_split / src_label).glob("*.jpg"))
        client_splits = split_images(images, CLIENTS)

        print(f"{src_label}: {len(images):,} images")

        for client_id, subset in enumerate(client_splits, start=1):

            target_dir = (
                DEST
                / f"client{client_id}"
                / dst_split
                / dst_label
            )

            target_dir.mkdir(parents=True, exist_ok=True)

            progress = tqdm(
                subset,
                desc=f"Client {client_id} | {dst_split} | {dst_label}",
                unit="img",
                leave=True,
            )

            for img_path in progress:

                try:
                    img = Image.open(img_path).convert("RGB")
                    img = img.resize(
                        IMAGE_SIZE,
                        Image.Resampling.LANCZOS,
                    )

                    img.save(
                        target_dir / img_path.name,
                        quality=95,
                    )

                except Exception as e:
                    tqdm.write(f"Failed: {img_path}")
                    tqdm.write(str(e))

print("\nDataset preparation completed successfully!")