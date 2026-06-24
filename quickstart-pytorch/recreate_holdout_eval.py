from pathlib import Path
import os

ROOT = Path("dataset_ffpp")

HOLDOUT_DIR = ROOT / "holdout"
HOLDOUT_EVAL_DIR = ROOT / "holdout_eval"

datasets = [
    "DeepFakeDetection",
    "FaceShifter",
]

for dataset_name in datasets:
    source_dir = HOLDOUT_DIR / dataset_name

    fake_dir = HOLDOUT_EVAL_DIR / dataset_name / "fake"
    real_dir = HOLDOUT_EVAL_DIR / dataset_name / "real"

    fake_dir.mkdir(parents=True, exist_ok=True)
    real_dir.mkdir(parents=True, exist_ok=True)

    for img_path in source_dir.glob("*.jpg"):
        filename = img_path.name.lower()

        if filename.startswith("original"):
            target = real_dir / img_path.name
        else:
            target = fake_dir / img_path.name

        os.symlink(
            img_path.resolve(),  # ABSOLUTE PATH
            target,
        )

    print(f"Finished: {dataset_name}")

print("\nDone.")
