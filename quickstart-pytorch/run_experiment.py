from pathlib import Path
from datetime import datetime
import subprocess
import sys


# ==========================================================
# RUN COMMAND + LOG TO TERMINAL AND FILE
# ==========================================================
def run_and_log(command, logfile):

    print("\n" + "=" * 80)
    print("RUNNING:", " ".join(command))
    print("LOG:", logfile)
    print("=" * 80 + "\n")

    with open(logfile, "w", encoding="utf-8") as f:
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
# MAIN
# ==========================================================
def main():

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    exp_dir = Path("experiment_logs") / timestamp
    exp_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print("FAFT EXPERIMENT")
    print("=" * 80)
    print("Experiment Folder:", exp_dir)
    print("=" * 80)

    # ------------------------------------------------------
    # TRAINING
    # ------------------------------------------------------

    train_log = exp_dir / "training.log"

    run_and_log(
        ["flwr", "run", ".", "--stream"],
        train_log,
    )

    # ------------------------------------------------------
    # EVALUATE SAVED MODEL
    # ------------------------------------------------------

    eval_log = exp_dir / "evaluation.log"

    run_and_log(
        [sys.executable, "evaluate_saved_model.py"],
        eval_log,
    )

    # ------------------------------------------------------
    # HOLDOUT EVALUATION
    # ------------------------------------------------------

    holdout_log = exp_dir / "holdout.log"

    run_and_log(
        [sys.executable, "evaluate_holdout.py"],
        holdout_log,
    )

    # ------------------------------------------------------
    # FINISHED
    # ------------------------------------------------------

    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETE")
    print("=" * 80)

    print("\nLogs:")
    print(f"Training   : {train_log}")
    print(f"Evaluation : {eval_log}")
    print(f"Holdout    : {holdout_log}")

    print("\nFolder:")
    print(exp_dir)

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
