from pathlib import Path
from datetime import datetime
import subprocess
import sys
import time

# ==========================================================
# USER SETTINGS (EDIT THESE ONLY)
# ==========================================================

# Wait for GPU before training/retries
WAIT_FOR_GPU = True

# Pipeline stages
RUN_TRAINING = True
RUN_EVALUATION = True
RUN_HOLDOUT = False

# GPU availability criteria
MAX_GPU_UTILIZATION = 95  # %
MAX_GPU_MEMORY_GB = 70  # GB used threshold

# GPU stability checks
CHECK_INTERVAL = 3  # seconds
REQUIRED_STABLE_CHECKS = 3

# Safety timeout
MAX_WAIT_HOURS = 24

# Retry settings
MAX_TRAIN_RETRIES = 999999  # effectively infinite
MAX_EVAL_RETRIES = 50
MAX_HOLDOUT_RETRIES = 50

RETRY_WAIT_SECONDS = 60

# ==========================================================
# GPU WAIT FUNCTION
# ==========================================================


def wait_for_gpu():

    if not WAIT_FOR_GPU:
        print("\n[INFO] GPU waiting disabled. Starting immediately.\n")
        return

    print("\n[INFO] Waiting for GPU availability...")

    start_time = time.time()

    stable_gpu = None
    stable_count = 0

    while True:
        elapsed_hours = (time.time() - start_time) / 3600

        if elapsed_hours > MAX_WAIT_HOURS:
            raise RuntimeError(
                f"Timed out after {MAX_WAIT_HOURS} hours waiting for GPU."
            )

        try:
            result = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=index,utilization.gpu,memory.used,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
            )

            lines = result.strip().split("\n")

            best_gpu = None
            best_mem = float("inf")

            print("\n" + "-" * 80)

            for line in lines:
                gpu_id, util, mem_used, mem_total = line.split(",")

                gpu_id = int(gpu_id.strip())
                util = float(util.strip())

                mem_used = float(mem_used.strip()) / 1024.0
                mem_total = float(mem_total.strip()) / 1024.0

                print(
                    f"GPU {gpu_id}: "
                    f"util={util:5.1f}% | "
                    f"mem={mem_used:6.1f}/{mem_total:.1f} GB"
                )

                if (
                    util <= MAX_GPU_UTILIZATION
                    and mem_used <= MAX_GPU_MEMORY_GB
                    and mem_used < best_mem
                ):
                    best_mem = mem_used
                    best_gpu = gpu_id

            if best_gpu is not None:
                if stable_gpu == best_gpu:
                    stable_count += 1
                else:
                    stable_gpu = best_gpu
                    stable_count = 1

                print(
                    f"[INFO] GPU {stable_gpu} stable "
                    f"{stable_count}/{REQUIRED_STABLE_CHECKS}"
                )

                if stable_count >= REQUIRED_STABLE_CHECKS:
                    print("\n" + "=" * 80)
                    print(f"[READY] GPU {stable_gpu} selected")
                    print("[READY] Starting stage")
                    print("=" * 80 + "\n")

                    return

            else:
                stable_gpu = None
                stable_count = 0

                print(f"[WAIT] No GPU below {MAX_GPU_MEMORY_GB} GB memory")

            time.sleep(CHECK_INTERVAL)

        except Exception as e:
            print(f"[ERROR] GPU check failed: {e}")
            time.sleep(CHECK_INTERVAL)


# ==========================================================
# RUN COMMAND + LOG
# ==========================================================


def run_and_log(command, logfile):

    print("\n" + "=" * 80)
    print("RUNNING :", " ".join(command))
    print("LOG FILE:", logfile)
    print("=" * 80 + "\n")

    with open(logfile, "a", encoding="utf-8") as f:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        for line in process.stdout:
            print(line, end="")
            f.write(line)
            f.flush()

        process.wait()

        if process.returncode != 0:
            print("\n" + "=" * 80)
            print("FAILED")
            print("Command:", " ".join(command))
            print("Return Code:", process.returncode)
            print("Log File:", logfile)
            print("=" * 80)

            raise subprocess.CalledProcessError(
                process.returncode,
                command,
            )

    print("\n" + "=" * 80)
    print("COMPLETED:", " ".join(command))
    print("=" * 80)


# ==========================================================
# STAGE EXECUTION WITH RETRIES
# ==========================================================


def run_stage_with_retries(
    stage_name,
    command,
    logfile,
    success_marker,
    max_retries,
    wait_for_resources=False,
):

    if success_marker.exists():
        print(f"\n[SKIP] {stage_name} already completed ({success_marker.name})")
        return

    attempt = 1

    while attempt <= max_retries:
        print("\n" + "#" * 80)
        print(f"{stage_name} - ATTEMPT {attempt}/{max_retries}")
        print("#" * 80)

        try:
            if wait_for_resources:
                wait_for_gpu()
            with open(logfile, "a", encoding="utf-8") as f:
                f.write("\n")
                f.write("=" * 80 + "\n")
                f.write(f"{stage_name} ATTEMPT {attempt}/{max_retries}\n")
                f.write("=" * 80 + "\n")
            run_and_log(command, logfile)

            success_marker.touch()

            print(f"\n[SUCCESS] {stage_name} completed.")

            return

        except Exception as e:
            print(f"\n[FAILED] {stage_name} attempt {attempt}")

            print(f"Reason: {e}")

            attempt += 1

            if attempt > max_retries:
                raise RuntimeError(f"{stage_name} failed after {max_retries} attempts")

            print(f"\n[RETRY] Waiting {RETRY_WAIT_SECONDS} seconds...")

            time.sleep(RETRY_WAIT_SECONDS)


# ==========================================================
# MAIN
# ==========================================================


def main():

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    exp_dir = Path("experiment_logs") / timestamp

    exp_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("\n" + "=" * 80)
    print("FAFT EXPERIMENT MANAGER")
    print("=" * 80)

    print(f"Experiment Folder      : {exp_dir}")
    print(f"WAIT_FOR_GPU          : {WAIT_FOR_GPU}")
    print(f"RUN_TRAINING          : {RUN_TRAINING}")
    print(f"RUN_EVALUATION        : {RUN_EVALUATION}")
    print(f"RUN_HOLDOUT           : {RUN_HOLDOUT}")
    print(f"MAX_GPU_MEMORY_GB     : {MAX_GPU_MEMORY_GB}")
    print(f"MAX_GPU_UTILIZATION   : {MAX_GPU_UTILIZATION}")
    print(f"CHECK_INTERVAL        : {CHECK_INTERVAL}")
    print(f"REQUIRED_STABLE_CHECKS: {REQUIRED_STABLE_CHECKS}")
    print(f"MAX_TRAIN_RETRIES     : {MAX_TRAIN_RETRIES}")
    print(f"MAX_EVAL_RETRIES      : {MAX_EVAL_RETRIES}")
    print(f"MAX_HOLDOUT_RETRIES   : {MAX_HOLDOUT_RETRIES}")

    print("=" * 80)

    train_log = exp_dir / "training.log"
    eval_log = exp_dir / "evaluation.log"
    holdout_log = exp_dir / "holdout.log"

    training_success = exp_dir / "TRAINING_SUCCESS"

    evaluation_success = exp_dir / "EVALUATION_SUCCESS"

    holdout_success = exp_dir / "HOLDOUT_SUCCESS"

    # ======================================================
    # TRAINING
    # ======================================================

    if RUN_TRAINING:
        run_stage_with_retries(
            stage_name="TRAINING",
            command=[
                "flwr",
                "run",
                ".",
                "--stream",
            ],
            logfile=train_log,
            success_marker=training_success,
            max_retries=MAX_TRAIN_RETRIES,
            wait_for_resources=True,
        )

    # ======================================================
    # EVALUATION
    # ======================================================

    if RUN_EVALUATION:
        run_stage_with_retries(
            stage_name="EVALUATION",
            command=[
                sys.executable,
                "evaluate_saved_model.py",
            ],
            logfile=eval_log,
            success_marker=evaluation_success,
            max_retries=MAX_EVAL_RETRIES,
            wait_for_resources=True,
        )

    # ======================================================
    # HOLDOUT
    # ======================================================

    if RUN_HOLDOUT:
        run_stage_with_retries(
            stage_name="HOLDOUT",
            command=[
                sys.executable,
                "evaluate_holdout.py",
            ],
            logfile=holdout_log,
            success_marker=holdout_success,
            max_retries=MAX_HOLDOUT_RETRIES,
            wait_for_resources=True,
        )

    # ======================================================
    # COMPLETE
    # ======================================================

    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETE")
    print("=" * 80)

    print(f"Experiment Folder : {exp_dir}")

    if RUN_TRAINING:
        print(f"Training Log      : {train_log}")

    if RUN_EVALUATION:
        print(f"Evaluation Log    : {eval_log}")

    if RUN_HOLDOUT:
        print(f"Holdout Log       : {holdout_log}")

    print("=" * 80)


if __name__ == "__main__":
    try:
        main()

    except KeyboardInterrupt:
        print("\n\n[INFO] Experiment cancelled by user.")

        sys.exit(0)

    except Exception as e:
        print(f"\n\n[FATAL] {e}")

        sys.exit(1)
