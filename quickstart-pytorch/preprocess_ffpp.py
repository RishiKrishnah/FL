# ============================================================
# preprocess_ffpp.py
#
# FaceForensics++ preprocessing for FAFT
#
# Features:
# - Identity-level splitting
# - Six-manipulation FF++ training
# - Multiprocessing
# - Resume support
# - Blur filtering
# - Duplicate frame removal
# - Progress bars + ETA
# - Face extraction
# - Balanced FL partitioning
# - CSV logging
# ============================================================

import os
import site

# ------------------------------------------------------------
# CUDA libraries for ONNX Runtime / InsightFace
# ------------------------------------------------------------
site_pkg = site.getsitepackages()[0]
cuda_root = os.path.join(site_pkg, "nvidia")

libs = [
    "cublas/lib",
    "cuda_runtime/lib",
    "cuda_nvrtc/lib",
    "cudnn/lib",
    "cufft/lib",
    "curand/lib",
    "cusolver/lib",
    "cusparse/lib",
]

paths = [os.path.join(cuda_root, p) for p in libs]

os.environ["LD_LIBRARY_PATH"] = (
    ":".join(paths) + ":" + os.environ.get("LD_LIBRARY_PATH", "")
)

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

from pathlib import Path
from concurrent.futures import (
    ProcessPoolExecutor,
    wait,
    FIRST_COMPLETED,
)
from collections import defaultdict, Counter, deque
from tqdm import tqdm
import cv2
import imagehash
from PIL import Image
import numpy as np
import csv
import random
import time
import traceback
from datetime import timedelta
import multiprocessing

# Import InsightFace AFTER setting LD_LIBRARY_PATH
from insightface.app import FaceAnalysis
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

TARGET_SIZE = 224

JPEG_QUALITY = 95

REAL_FRAMES_PER_VIDEO = 70

FAKE_FRAMES_PER_VIDEO = 12

MIN_FRAMES_PER_VIDEO = 8

# ============================================================
# FILTERS
# ============================================================

ENABLE_BLUR_FILTER = True

BLUR_THRESHOLD = 40.0

ENABLE_DUPLICATE_FILTER = True

HASH_DISTANCE_THRESHOLD = 5

# ============================================================
# FACE EXTRACTION
# ============================================================

ENABLE_FACE_CROP = True

FACE_MARGIN = 0.2
MIN_FACE_SIZE = 80

# RetinaFace will be used later

# ============================================================
# MULTIPROCESSING
# ============================================================

NUM_WORKERS = 4
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
stats["face_detected"] = 0
stats["face_failed"] = 0


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


print(f"Clients              : {NUM_CLIENTS}")
print(f"Workers              : {NUM_WORKERS}")

print(f"Real Frames / Video  : {REAL_FRAMES_PER_VIDEO}")
print(f"Fake Frames / Video  : {FAKE_FRAMES_PER_VIDEO}")
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

    print(f"Faces Detected       : {stats['face_detected']}")
    print(f"Detection Failed     : {stats['face_failed']}")

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
# UNIFORM FRAME SAMPLING
# ============================================================


def get_sample_indices(total_frames, label):
    """
    Uniformly sample frames across the entire video.

    Original videos : 70 frames
    Fake videos     : 12 frames
    """

    target_frames = REAL_FRAMES_PER_VIDEO if label == "real" else FAKE_FRAMES_PER_VIDEO

    # Handle very short videos
    target_frames = min(target_frames, total_frames)

    if target_frames <= 0:
        return set()

    indices = np.linspace(
        0,
        total_frames - 1,
        target_frames,
        dtype=int,
    )

    return set(indices.tolist())


# ============================================================
# FACE DETECTOR
# ============================================================

FACE_DETECTOR = None


def init_worker():
    global FACE_DETECTOR

    FACE_DETECTOR = FaceAnalysis(
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
    )

    FACE_DETECTOR.prepare(
        ctx_id=0,
        det_size=(256, 256),
    )


# ============================================================
# FACE CROP
# ============================================================


def extract_face(frame):

    faces = FACE_DETECTOR.get(frame)

    if len(faces) == 0:
        return None

    face = max(
        faces,
        key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
    )

    x1, y1, x2, y2 = map(int, face.bbox)

    h, w = frame.shape[:2]

    bw = x2 - x1
    bh = y2 - y1

    expand = 0.50

    x1 -= int(bw * expand)
    y1 -= int(bh * expand)
    x2 += int(bw * expand)
    y2 += int(bh * expand)

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)

    size = max(x2 - x1, y2 - y1)

    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2

    x1 = max(0, cx - size // 2)
    y1 = max(0, cy - size // 2)

    x2 = min(w, x1 + size)
    y2 = min(h, y1 + size)

    crop = frame[y1:y2, x1:x2]

    if crop.size == 0:
        return None

    return cv2.resize(crop, (224, 224))


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

        sample_indices = get_sample_indices(
            total_frames,
            label,
        )

        frame_idx = 0
        saved = 0
        blur_removed = 0
        duplicate_removed = 0
        frames_read = 0
        frames_sampled = 0
        face_detected = 0
        face_failed = 0

        previous_hashes = deque(maxlen=10)
        while True:
            ret, frame = cap.read()

            if not ret:
                break
            frames_read += 1

            if frame_idx not in sample_indices:
                frame_idx += 1
                continue

            frames_sampled += 1

            # --------------------------------------------------
            # Face Detection
            # --------------------------------------------------
            try:
                face = extract_face(frame)

            except Exception:
                face = None

            if face is None:
                face_failed += 1
                frame_idx += 1
                continue

            face_detected += 1

            # --------------------------------------------------
            # Blur check on cropped face
            # --------------------------------------------------
            if is_blurry(face):
                blur_removed += 1
                frame_idx += 1
                continue

            # --------------------------------------------------
            # Duplicate check on cropped face
            # --------------------------------------------------
            if is_duplicate(face, previous_hashes):
                duplicate_removed += 1
                frame_idx += 1
                continue

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
            return None

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
            "face_detected": face_detected,
            "face_failed": face_failed,
        }

    except Exception:
        log_error(f"\n{video_path}\n" + traceback.format_exc())


# ============================================================
# IDENTITY SPLIT
# ============================================================
def main():

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

    def split_clients_balanced(video_list):
        """
        Evenly distribute a list of videos across clients.
        """

        shuffled = video_list.copy()
        random.shuffle(shuffled)

        client_lists = [[] for _ in range(NUM_CLIENTS)]

        for i, video in enumerate(shuffled):
            client_lists[i % NUM_CLIENTS].append(video)

        return client_lists

    # ============================================================
    # MANIPULATION LABEL
    # ============================================================

    def manipulation_name(video_path):

        return video_path.parent.name

    # ============================================================
    # BUILD JOBS
    # ============================================================

    def build_jobs(videos, split_name, label):
        """
        Build federated jobs while preserving manipulation balance.
        """

        if label == "real":
            partitions = split_clients_balanced(videos)

            for client_id, subset in enumerate(partitions, start=1):
                for video in subset:
                    client_jobs[client_id].append(
                        (
                            video,
                            split_name,
                            client_id,
                            label,
                            "original",
                        )
                    )

            return

        # --------------------------------------------------
        # Fake videos
        # --------------------------------------------------

        by_manipulation = defaultdict(list)

        for video in videos:
            by_manipulation[video.parent.name].append(video)

        for manipulation, manipulation_videos in by_manipulation.items():
            partitions = split_clients_balanced(manipulation_videos)

            for client_id, subset in enumerate(partitions, start=1):
                for video in subset:
                    client_jobs[client_id].append(
                        (
                            video,
                            split_name,
                            client_id,
                            label,
                            manipulation,
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

    for client_id in range(1, NUM_CLIENTS + 1):
        print("\n" + "=" * 60)
        print(f"Client {client_id}")
        print("=" * 60)

        jobs = client_jobs[client_id]

        for split_name in ["train", "val", "test"]:
            print(f"\n{split_name.upper()}")

            split_jobs = [j for j in jobs if j[1] == split_name]

            counts = Counter()

            for _, _, _, label, manipulation in split_jobs:
                if label == "real":
                    counts["Original"] += 1
                else:
                    counts[manipulation] += 1

            for key in [
                "Original",
                "Deepfakes",
                "Face2Face",
                "FaceSwap",
                "NeuralTextures",
                "FaceShifter",
                "DeepFakeDetection",
            ]:
                print(f"{key:<20}: {counts[key]}")

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

    from multiprocessing import get_context

    from concurrent.futures import wait, FIRST_COMPLETED

    with ProcessPoolExecutor(
        max_workers=NUM_WORKERS,
        mp_context=get_context("fork"),
        initializer=init_worker,
    ) as executor:
        job_iter = iter(all_jobs)
        futures = set()

        # Start a few initial jobs
        for _ in range(NUM_WORKERS * 2):
            try:
                futures.add(executor.submit(process_video, next(job_iter)))
            except StopIteration:
                break

        while futures:
            done, futures = wait(
                futures,
                return_when=FIRST_COMPLETED,
            )

            for future in done:
                result = future.result()

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

                try:
                    futures.add(
                        executor.submit(
                            process_video,
                            next(job_iter),
                        )
                    )
                except StopIteration:
                    pass

    pbar.close()

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

    print(f"Faces detected       : {stats['face_detected']}")
    print(f"Face detection failed: {stats['face_failed']}")
    total_attempts = stats["face_detected"] + stats["face_failed"]

    if total_attempts > 0:
        detection_rate = 100.0 * stats["face_detected"] / total_attempts

        print(f"Detection rate       : {detection_rate:.2f}%")

    print()

    print(f"Elapsed              : {format_time(elapsed_total)}")

    print()

    print("Logs")

    print(f"Video log    : {VIDEO_LOG}")

    print(f"Summary log  : {SUMMARY_LOG}")

    print(f"Error log    : {ERROR_LOG}")

    print()

    print(f"Dataset root : {DEST_ROOT}")

    print()

    print("=" * 80)
    print("DONE")
    print("=" * 80)


if __name__ == "__main__":
    multiprocessing.set_start_method("fork", force=True)
    main()
