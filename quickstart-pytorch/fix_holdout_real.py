from pathlib import Path
import os
import itertools

ROOT = Path("dataset_ffpp")

real_sources = []

for p in ROOT.glob("client*/test/real/*.jpg"):
    real_sources.append(p.resolve())

print("Real images found:", len(real_sources))

datasets = [
    "DeepFakeDetection",
    "FaceShifter",
]

for dataset_name in datasets:
    real_dir = ROOT / "holdout_eval" / dataset_name / "real"

    real_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    fake_count = len(
        list((ROOT / "holdout_eval" / dataset_name / "fake").glob("*.jpg"))
    )

    print(f"\n{dataset_name}")
    print("Fake images:", fake_count)

    created = 0

    for idx, src in enumerate(itertools.cycle(real_sources)):
        if created >= fake_count:
            break

        target = real_dir / f"{src.stem}_{created:06d}.jpg"

        if not target.exists():
            os.symlink(
                src,
                target,
            )
            created += 1

    print("Real images linked:", created)

print("\nDone.")
