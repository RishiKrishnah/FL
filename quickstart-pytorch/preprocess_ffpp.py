# ============================================================
# preprocess_ffpp.py
#
# FaceForensics++ preprocessing for FAFT
#
# Features:
# - Identity-level splitting
# - Cross-manipulation evaluation
# - Multiprocessing
# - Resume support
# - Blur filtering
# - Duplicate frame removal
# - Progress bars + ETA
# - Face extraction
# - Balanced FL partitioning
# - CSV logging
# ============================================================

from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from collections import defaultdict, Counter, deque
from tqdm import tqdm
import cv2
import imagehash
from PIL import Image
import numpy as np
import os
import csv
import random
import time
import traceback
from datetime import timedelta

# ============================================================
# RANDOM SEED
# ============================================================

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ============================================================
# DATASET PATHS
# ============================================================

SOURCE_ROOT = Path("/root/Rishi/datasets/ff-c23/FaceForensics++_C23")

CACHE_ROOT = Path("ffpp_cache")

DEST_ROOT = Path("dataset_ffpp")

LOG_DIR = Path("logs_ffpp")

LOG_DIR.mkdir(exist_ok=True)

# ============================================================
# VIDEO FOLDERS
# ============================================================

REAL_DIR = SOURCE_ROOT / "original"

TRAIN_MANIPULATIONS = [
    "Deepfakes",
    "Face2Face",
    "FaceSwap",
    "NeuralTextures",
]

HOLDOUT_MANIPULATIONS = [
    "FaceShifter",
    "DeepFakeDetection",
]

# ============================================================
# SPLITS
# ============================================================

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# ============================================================
# FEDERATED SETTINGS
# ============================================================

NUM_CLIENTS = 5

IID = True

# future support
DIRICHLET_ALPHA = 0.5

# ============================================================
# FRAME EXTRACTION
# ============================================================

FRAME_INTERVAL = 10

TARGET_SIZE = 224

JPEG_QUALITY = 95

MIN_FRAMES_PER_VIDEO = 20

# ============================================================
# FILTERS
# ============================================================

ENABLE_BLUR_FILTER = True

BLUR_THRESHOLD = 100.0

ENABLE_DUPLICATE_FILTER = False

HASH_DISTANCE_THRESHOLD = 5

# ============================================================
# FACE EXTRACTION
# ============================================================

ENABLE_FACE_CROP = True

FACE_MARGIN = 0.2

# RetinaFace will be used later

# ============================================================
# MULTIPROCESSING
# ============================================================

NUM_WORKERS = os.cpu_count()

# ============================================================
# RESUME SUPPORT
# ============================================================

SKIP_EXISTING_VIDEOS = True

# ============================================================
# IMAGE EXTENSION
# ============================================================

IMAGE_EXT = ".jpg"

# ============================================================
# STATS
# ============================================================

stats = defaultdict(int)

stats["videos_processed"] = 0
stats["videos_skipped"] = 0
stats["videos_corrupted"] = 0

stats["frames_read"] = 0
stats["frames_sampled"] = 0
stats["frames_saved"] = 0

stats["blur_rejected"] = 0
stats["duplicate_rejected"] = 0

# ============================================================
# TIMING
# ============================================================

START_TIME = time.time()

# ============================================================
# CSV LOG FILES
# ============================================================

VIDEO_LOG = LOG_DIR / "processed_videos.csv"

SUMMARY_LOG = LOG_DIR / "dataset_summary.csv"

ERROR_LOG = LOG_DIR / "errors.txt"

# ============================================================
# CLIENT DIRECTORIES
# ============================================================

for client_id in range(1, NUM_CLIENTS + 1):
    for split in ["train", "val", "test"]:
        for label in ["real", "fake"]:
            (DEST_ROOT / f"client{client_id}" / split / label).mkdir(
                parents=True,
                exist_ok=True,
            )

# ============================================================
# PRINT CONFIG
# ============================================================

print("\n" + "=" * 80)
print("FAFT FaceForensics++ Preprocessing")
print("=" * 80)

print(f"Source Root          : {SOURCE_ROOT}")
print(f"Cache Root           : {CACHE_ROOT}")
print(f"Destination Root     : {DEST_ROOT}")

print()

print("Training Manipulations")
for m in TRAIN_MANIPULATIONS:
    print("  ", m)

print()

print("Holdout Manipulations")
for m in HOLDOUT_MANIPULATIONS:
    print("  ", m)

print()

print(f"Clients              : {NUM_CLIENTS}")
print(f"Workers              : {NUM_WORKERS}")

print(f"Frame Interval       : {FRAME_INTERVAL}")
print(f"Target Size          : {TARGET_SIZE}")

print(f"Blur Filter          : {ENABLE_BLUR_FILTER}")
print(f"Duplicate Filter     : {ENABLE_DUPLICATE_FILTER}")
print(f"Face Crop            : {ENABLE_FACE_CROP}")

print("=" * 80)

# ============================================================
# UTILITIES
# ============================================================


def format_time(seconds):

    return str(timedelta(seconds=int(seconds)))


def elapsed_time():

    return time.time() - START_TIME


def print_stats():

    elapsed = elapsed_time()

    print("\n" + "=" * 80)
    print("Current Statistics")
    print("=" * 80)

    print(f"Videos Processed     : {stats['videos_processed']}")
    print(f"Videos Skipped       : {stats['videos_skipped']}")
    print(f"Corrupted Videos     : {stats['videos_corrupted']}")

    print()

    print(f"Frames Read          : {stats['frames_read']}")
    print(f"Frames Sampled       : {stats['frames_sampled']}")
    print(f"Frames Saved         : {stats['frames_saved']}")

    print()

    print(f"Blur Rejected        : {stats['blur_rejected']}")
    print(f"Duplicate Rejected   : {stats['duplicate_rejected']}")

    print()

    print(f"Elapsed Time         : {format_time(elapsed)}")

    print("=" * 80)


# ============================================================
# ERROR LOGGING
# ============================================================


def log_error(message):

    with open(ERROR_LOG, "a") as f:
        f.write(message + "\n")


# ============================================================
# VIDEO CSV LOG
# ============================================================

if not VIDEO_LOG.exists():
    with open(VIDEO_LOG, "w", newline="") as f:
        writer = csv.writer(f)

        writer.writerow(
            [
                "video",
                "split",
                "client",
                "class",
                "manipulation",
                "frames_read",
                "frames_saved",
                "blur_removed",
                "duplicates_removed",
                "time_sec",
            ]
        )


def log_video(
    video_name,
    split,
    client,
    label,
    manipulation,
    frames_read,
    frames_saved,
    blur_removed,
    duplicate_removed,
    elapsed_sec,
):

    with open(VIDEO_LOG, "a", newline="") as f:
        writer = csv.writer(f)

        writer.writerow(
            [
                video_name,
                split,
                client,
                label,
                manipulation,
                frames_read,
                frames_saved,
                blur_removed,
                duplicate_removed,
                round(elapsed_sec, 2),
            ]
        )


# ============================================================
# SUMMARY CSV
# ============================================================


def save_summary():

    with open(SUMMARY_LOG, "w", newline="") as f:
        writer = csv.writer(f)

        writer.writerow(
            [
                "metric",
                "value",
            ]
        )

        for k, v in stats.items():
            writer.writerow([k, v])


# ============================================================
# BLUR FILTER
# ============================================================


def is_blurry(frame):

    if not ENABLE_BLUR_FILTER:
        return False

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY,
    )

    variance = cv2.Laplacian(
        gray,
        cv2.CV_64F,
    ).var()

    return variance < BLUR_THRESHOLD


# ============================================================
# DUPLICATE DETECTION
# ============================================================


def frame_hash(frame):

    image = Image.fromarray(
        cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB,
        )
    )

    return imagehash.average_hash(image)


def is_duplicate(
    frame,
    previous_hashes,
):

    if not ENABLE_DUPLICATE_FILTER:
        return False

    current_hash = frame_hash(frame)

    for old_hash in previous_hashes:
        if abs(current_hash - old_hash) <= HASH_DISTANCE_THRESHOLD:
            return True

    previous_hashes.append(current_hash)

    return False


# ============================================================
# SAFE IMAGE SAVE
# ============================================================


def save_image(
    image,
    filename,
):

    cv2.imwrite(
        str(filename),
        image,
        [
            cv2.IMWRITE_JPEG_QUALITY,
            JPEG_QUALITY,
        ],
    )


# ============================================================
# RESUME SUPPORT
# ============================================================


def video_already_processed(
    manipulation,
    video_name,
    split,
    client,
    label,
):

    out_dir = DEST_ROOT / f"client{client}" / split / label

    matches = list(out_dir.glob(f"{manipulation}_{video_name}_*{IMAGE_EXT}"))

    return len(matches) > 0


# ============================================================
# ETA
# ============================================================


def compute_eta(
    processed,
    total,
    start_time,
):

    elapsed = time.time() - start_time

    if processed == 0:
        return "Unknown"

    avg = elapsed / processed

    remaining = total - processed

    eta = avg * remaining

    return format_time(eta)


# ============================================================
# CORRUPTION CHECK
# ============================================================


def video_valid(video_path):

    cap = cv2.VideoCapture(str(video_path))

    valid = cap.isOpened()

    cap.release()

    return valid


# ============================================================
# UNIQUE IMAGE NAME
# ============================================================


def image_filename(
    manipulation,
    video_name,
    frame_number,
):
    return f"{manipulation}_{video_name}_{frame_number:05d}{IMAGE_EXT}"


# ============================================================
# FACE DETECTOR
# ============================================================

try:
    from retinaface import RetinaFace

    RETINAFACE_AVAILABLE = True

except Exception:
    RETINAFACE_AVAILABLE = False

    print("\nWARNING")
    print("RetinaFace not installed.")
    print("Using center crop instead.\n")


# ============================================================
# FACE CROP
# ============================================================


def extract_face(frame):

    h, w = frame.shape[:2]

    if not ENABLE_FACE_CROP:
        return cv2.resize(
            frame,
            (
                TARGET_SIZE,
                TARGET_SIZE,
            ),
        )

    if RETINAFACE_AVAILABLE:
        try:
            detections = RetinaFace.detect_faces(frame)

            if isinstance(detections, dict):
                largest_area = 0
                best_box = None

                for _, det in detections.items():
                    x1, y1, x2, y2 = det["facial_area"]

                    area = (x2 - x1) * (y2 - y1)

                    if area > largest_area:
                        largest_area = area
                        best_box = [x1, y1, x2, y2]

                if best_box is not None:
                    x1, y1, x2, y2 = best_box

                    margin_x = int((x2 - x1) * FACE_MARGIN)
                    margin_y = int((y2 - y1) * FACE_MARGIN)

                    x1 = max(0, x1 - margin_x)
                    y1 = max(0, y1 - margin_y)

                    x2 = min(w, x2 + margin_x)
                    y2 = min(h, y2 + margin_y)

                    face = frame[y1:y2, x1:x2]

                    return cv2.resize(
                        face,
                        (
                            TARGET_SIZE,
                            TARGET_SIZE,
                        ),
                    )

        except Exception:
            pass

    # fallback center crop

    size = min(h, w)

    start_x = (w - size) // 2
    start_y = (h - size) // 2

    crop = frame[
        start_y : start_y + size,
        start_x : start_x + size,
    ]

    return cv2.resize(
        crop,
        (
            TARGET_SIZE,
            TARGET_SIZE,
        ),
    )


# ============================================================
# ADAPTIVE SAMPLING
# ============================================================


def compute_interval(frame_count):

    if frame_count < 300:
        return 3

    if frame_count < 700:
        return 5

    return 10


# ============================================================
# VIDEO PROCESSOR
# ============================================================


def process_video(job):

    (
        video_path,
        split,
        client_id,
        label,
        manipulation,
    ) = job

    start = time.time()

    video_name = video_path.stem

    try:
        if SKIP_EXISTING_VIDEOS:
            if video_already_processed(
                manipulation,
                video_name,
                split,
                client_id,
                label,
            ):
                return {"videos_skipped": 1}

        if not video_valid(video_path):
            log_error(f"Invalid video: {video_path}")

            return {"videos_corrupted": 1}

        cap = cv2.VideoCapture(str(video_path))

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        frame_interval = compute_interval(total_frames)

        frame_idx = 0
        saved = 0
        blur_removed = 0
        duplicate_removed = 0
        frames_read = 0
        frames_sampled = 0

        previous_hashes = deque(maxlen=20)
        cached_face = None

        while True:
            ret, frame = cap.read()

            if not ret:
                break
            frames_read += 1

            if frame_idx % frame_interval != 0:
                frame_idx += 1
                continue

            frames_sampled += 1

            if is_blurry(frame):
                blur_removed += 1

                frame_idx += 1
                continue

            if is_duplicate(
                frame,
                previous_hashes,
            ):
                duplicate_removed += 1

                frame_idx += 1
                continue

            if saved % 20 == 0 or cached_face is None:
                try:
                    cached_face = extract_face(frame)

                except Exception:
                    pass

            if cached_face is None:
                frame_idx += 1
                continue

            face = cached_face

            filename = image_filename(
                manipulation,
                video_name,
                saved,
            )

            save_dir = DEST_ROOT / f"client{client_id}" / split / label

            save_image(
                face,
                save_dir / filename,
            )

            saved += 1

            frame_idx += 1

        cap.release()

        if saved < MIN_FRAMES_PER_VIDEO:
            print(f"Rejected {video_name} only {saved} frames")

            return

        elapsed_sec = time.time() - start

        log_video(
            video_name,
            split,
            client_id,
            label,
            manipulation,
            total_frames,
            saved,
            blur_removed,
            duplicate_removed,
            elapsed_sec,
        )

        return {
            "videos_processed": 1,
            "frames_read": frames_read,
            "frames_sampled": frames_sampled,
            "frames_saved": saved,
            "blur_rejected": blur_removed,
            "duplicate_rejected": duplicate_removed,
        }

    except Exception:
        log_error(f"\n{video_path}\n" + traceback.format_exc())


# ============================================================
# IDENTITY SPLIT
# ============================================================

print("\nBuilding identity splits...")

real_videos = sorted(list(REAL_DIR.glob("*.mp4")))

identities = [int(v.stem) for v in real_videos]

random.shuffle(identities)

n_total = len(identities)

n_train = int(TRAIN_RATIO * n_total)
n_val = int(VAL_RATIO * n_total)

train_ids = set(identities[:n_train])

val_ids = set(identities[n_train : n_train + n_val])

test_ids = set(identities[n_train + n_val :])

print(f"Train identities : {len(train_ids)}")
print(f"Val identities   : {len(val_ids)}")
print(f"Test identities  : {len(test_ids)}")


# ============================================================
# REAL VIDEOS
# ============================================================

real_split = {}

for video in real_videos:
    identity = int(video.stem)

    if identity in train_ids:
        split = "train"

    elif identity in val_ids:
        split = "val"

    else:
        split = "test"

    real_split[video] = split


# ============================================================
# FAKE VIDEO SPLIT
# ============================================================


def parse_fake_identities(video_name):

    stem = Path(video_name).stem

    parts = stem.split("_")

    try:
        src = int(parts[0])
        tgt = int(parts[1])

        return src, tgt

    except Exception:
        return None, None


fake_split = {}

for manipulation in TRAIN_MANIPULATIONS:
    folder = SOURCE_ROOT / manipulation

    videos = sorted(list(folder.glob("*.mp4")))

    print(f"{manipulation}: {len(videos)} videos")

    for video in videos:
        src, tgt = parse_fake_identities(video.name)

        if src is None:
            continue

        # prevent identity leakage

        if src in train_ids and tgt in train_ids:
            split = "train"

        elif src in val_ids and tgt in val_ids:
            split = "val"

        elif src in test_ids and tgt in test_ids:
            split = "test"

        else:
            # mixed identities
            continue

        fake_split[video] = split


# ============================================================
# HOLDOUT MANIPULATIONS
# ============================================================

holdout_videos = []

for manipulation in HOLDOUT_MANIPULATIONS:
    folder = SOURCE_ROOT / manipulation

    videos = sorted(list(folder.glob("*.mp4")))

    holdout_videos.extend(videos)

print()
print("Holdout videos:", len(holdout_videos))


# ============================================================
# SUMMARY
# ============================================================

real_count = Counter(real_split.values())

fake_count = Counter(fake_split.values())

print("\nREAL")

print(real_count)

print("\nFAKE")

print(fake_count)


# ============================================================
# BUILD LISTS
# ============================================================

train_real = [v for v, s in real_split.items() if s == "train"]

val_real = [v for v, s in real_split.items() if s == "val"]

test_real = [v for v, s in real_split.items() if s == "test"]

train_fake = [v for v, s in fake_split.items() if s == "train"]

val_fake = [v for v, s in fake_split.items() if s == "val"]

test_fake = [v for v, s in fake_split.items() if s == "test"]

print()

print("Train real :", len(train_real))
print("Val real   :", len(val_real))
print("Test real  :", len(test_real))

print()

print("Train fake :", len(train_fake))
print("Val fake   :", len(val_fake))
print("Test fake  :", len(test_fake))

# ============================================================
# FEDERATED PARTITIONING
# ============================================================

print("\nBuilding federated partitions...")

client_jobs = defaultdict(list)


# ============================================================
# SPLIT INTO CLIENTS
# ============================================================


def split_clients(video_list):

    shuffled = video_list.copy()

    random.shuffle(shuffled)

    size = len(shuffled) // NUM_CLIENTS

    clients = []

    for i in range(NUM_CLIENTS):
        start = i * size

        if i == NUM_CLIENTS - 1:
            end = len(shuffled)
        else:
            end = (i + 1) * size

        clients.append(shuffled[start:end])

    return clients


# ============================================================
# MANIPULATION LABEL
# ============================================================


def manipulation_name(video_path):

    return video_path.parent.name


# ============================================================
# BUILD JOBS
# ============================================================


def build_jobs(
    videos,
    split_name,
    label,
):

    partitions = split_clients(videos)

    for client_id, subset in enumerate(
        partitions,
        start=1,
    ):
        for video in subset:
            client_jobs[client_id].append(
                (
                    video,
                    split_name,
                    client_id,
                    label,
                    manipulation_name(video),
                )
            )


# ============================================================
# REAL
# ============================================================

build_jobs(
    train_real,
    "train",
    "real",
)

build_jobs(
    val_real,
    "val",
    "real",
)

build_jobs(
    test_real,
    "test",
    "real",
)


# ============================================================
# FAKE
# ============================================================

build_jobs(
    train_fake,
    "train",
    "fake",
)

build_jobs(
    val_fake,
    "val",
    "fake",
)

build_jobs(
    test_fake,
    "test",
    "fake",
)


# ============================================================
# CLIENT STATISTICS
# ============================================================

print("\nClient Statistics")

for client_id in range(
    1,
    NUM_CLIENTS + 1,
):
    jobs = client_jobs[client_id]

    train_jobs = [x for x in jobs if x[1] == "train"]

    val_jobs = [x for x in jobs if x[1] == "val"]

    test_jobs = [x for x in jobs if x[1] == "test"]

    train_real_count = len([x for x in train_jobs if x[3] == "real"])

    train_fake_count = len([x for x in train_jobs if x[3] == "fake"])

    val_real_count = len([x for x in val_jobs if x[3] == "real"])

    val_fake_count = len([x for x in val_jobs if x[3] == "fake"])

    test_real_count = len([x for x in test_jobs if x[3] == "real"])

    test_fake_count = len([x for x in test_jobs if x[3] == "fake"])

    print()

    print(f"Client {client_id}")

    print(f"Train : R={train_real_count} F={train_fake_count}")

    print(f"Val   : R={val_real_count} F={val_fake_count}")

    print(f"Test  : R={test_real_count} F={test_fake_count}")


# ============================================================
# FLATTEN JOBS
# ============================================================

all_jobs = []

for client_id in range(
    1,
    NUM_CLIENTS + 1,
):
    all_jobs.extend(client_jobs[client_id])

print()

print("Total jobs:", len(all_jobs))


# ============================================================
# HOLDOUT JOBS
# ============================================================

holdout_jobs = []

for video in holdout_videos:
    holdout_jobs.append(
        (
            video,
            "holdout",
            0,
            "fake",
            video.parent.name,
        )
    )

print("Holdout jobs:", len(holdout_jobs))


# ============================================================
# PROGRESS BAR INFO
# ============================================================

TOTAL_VIDEOS = len(all_jobs)

print()

print("=" * 80)

print(f"Videos to process : {TOTAL_VIDEOS}")

print(f"Workers           : {NUM_WORKERS}")

print(f"Expected clients  : {NUM_CLIENTS}")

print("=" * 80)

# ============================================================
# MAIN PROCESSING
# ============================================================

print("\nStarting preprocessing...\n")

overall_start = time.time()

processed = 0

pbar = tqdm(
    total=TOTAL_VIDEOS,
    desc="Videos",
    dynamic_ncols=True,
)

# ============================================================
# MULTIPROCESSING
# ============================================================

with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
    for result in executor.map(
        process_video,
        all_jobs,
        chunksize=8,
    ):
        if result is not None:
            for k, v in result.items():
                stats[k] += v

        processed += 1

        elapsed = time.time() - overall_start

        video_per_sec = processed / elapsed if elapsed > 0 else 0

        eta = compute_eta(
            processed,
            TOTAL_VIDEOS,
            overall_start,
        )

        pbar.set_postfix(
            {
                "done": processed,
                "v/s": f"{video_per_sec:.2f}",
                "ETA": eta,
            }
        )

        pbar.update(1)

        if processed % 100 == 0:
            print_stats()

pbar.close()


# ============================================================
# HOLDOUT DATASET
# ============================================================

print("\nProcessing holdout manipulations...\n")

holdout_root = DEST_ROOT / "holdout"

for manipulation in HOLDOUT_MANIPULATIONS:
    (holdout_root / manipulation).mkdir(
        parents=True,
        exist_ok=True,
    )

holdout_bar = tqdm(
    holdout_jobs,
    desc="Holdout",
)

for (
    video_path,
    split,
    client,
    label,
    manipulation,
) in holdout_bar:
    try:
        cap = cv2.VideoCapture(str(video_path))

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        interval = compute_interval(frame_count)

        idx = 0
        saved = 0
        frames_read = 0
        frames_sampled = 0

        while True:
            ret, frame = cap.read()

            if not ret:
                break

            frames_read += 1

            if idx % interval != 0:
                idx += 1
                continue

            face = extract_face(frame)

            filename = f"{manipulation}_{video_path.stem}_{saved:05d}.jpg"

            save_image(
                face,
                holdout_root / manipulation / filename,
            )

            saved += 1

            idx += 1

        cap.release()

    except Exception:
        log_error(traceback.format_exc())


# ============================================================
# SAVE SUMMARY
# ============================================================

save_summary()

# ============================================================
# FINAL REPORT
# ============================================================

elapsed_total = time.time() - START_TIME

print("\n")
print("=" * 80)
print("PREPROCESSING COMPLETE")
print("=" * 80)

print()

print(f"Videos processed     : {stats['videos_processed']}")

print(f"Videos skipped       : {stats['videos_skipped']}")

print(f"Corrupted videos     : {stats['videos_corrupted']}")

print()

print(f"Frames read          : {stats['frames_read']}")

print(f"Frames sampled       : {stats['frames_sampled']}")

print(f"Frames saved         : {stats['frames_saved']}")

print()

print(f"Blur rejected        : {stats['blur_rejected']}")

print(f"Duplicate rejected   : {stats['duplicate_rejected']}")

print()

print(f"Elapsed              : {format_time(elapsed_total)}")

print()

print("Logs")

print(f"Video log    : {VIDEO_LOG}")

print(f"Summary log  : {SUMMARY_LOG}")

print(f"Error log    : {ERROR_LOG}")

print()

print(f"Dataset root : {DEST_ROOT}")

print(f"Holdout root : {holdout_root}")

print()

print("=" * 80)
print("DONE")
print("=" * 80)
