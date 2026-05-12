#!/usr/bin/env python3
import argparse
import csv
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def should_include(row, args):
    if args.method_name and row["method_name"] not in args.method_name:
        return False
    if args.noise_name and row["noise_name"] not in args.noise_name:
        return False
    if args.seed and int(row["seed"]) not in args.seed:
        return False
    return True


def run_one(row, args):
    run_name = row["run_name"]
    run_dir = Path(row["run_dir"])

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    stdout_path = log_dir / f"{run_name}.out"
    stderr_path = log_dir / f"{run_name}.err"

    cmd = [
        "python",
        args.inference_script,
        "--model_dir",
        str(run_dir),
        "--fusion_method",
        args.fusion_method,
    ]

    if args.extra_args:
        cmd.extend(args.extra_args)

    if args.skip_existing:
        eval_yaml = run_dir / "eval.yaml"
        eval_global_yaml = run_dir / "eval_global_sort.yaml"
        if eval_yaml.exists() or eval_global_yaml.exists():
            return {
                "run_name": run_name,
                "status": "skipped_existing_eval",
                "returncode": 0,
                "cmd": " ".join(cmd),
            }

    with open(stdout_path, "w") as fout, open(stderr_path, "w") as ferr:
        process = subprocess.run(
            cmd,
            stdout=fout,
            stderr=ferr,
            text=True,
        )

    return {
        "run_name": run_name,
        "status": "ok" if process.returncode == 0 else "failed",
        "returncode": process.returncode,
        "cmd": " ".join(cmd),
        "stdout": str(stdout_path),
        "stderr": str(stderr_path),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        required=True,
        help="Path to generated manifest CSV.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=5,
        help="Number of parallel inference processes to run on the allocated GPU.",
    )
    parser.add_argument(
        "--inference_script",
        default="opencood/tools/inference.py",
    )
    parser.add_argument(
        "--fusion_method",
        default="intermediate",
    )
    parser.add_argument(
        "--log_dir",
        default="parallel_inference_logs",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip runs if eval.yaml or eval_global_sort.yaml already exists in the run directory.",
    )

    # Optional filters
    parser.add_argument(
        "--method_name",
        nargs="+",
        default=None,
        help="Only run selected method_name values from manifest.",
    )
    parser.add_argument(
        "--noise_name",
        nargs="+",
        default=None,
        help="Only run selected noise_name values from manifest.",
    )
    parser.add_argument(
        "--seed",
        nargs="+",
        type=int,
        default=None,
        help="Only run selected seeds from manifest.",
    )

    # For extra inference.py args, e.g. --global_sort_detections
    parser.add_argument(
        "extra_args",
        nargs=argparse.REMAINDER,
        help="Extra args passed to inference.py. Use after --.",
    )

    args = parser.parse_args()

    manifest_path = Path(args.manifest)

    with open(manifest_path, "r") as f:
        rows = list(csv.DictReader(f))

    selected_rows = [row for row in rows if should_include(row, args)]

    print(f"Manifest: {manifest_path}")
    print(f"Total runs in manifest: {len(rows)}")
    print(f"Selected runs: {len(selected_rows)}")
    print(f"Parallel workers: {args.workers}")

    if len(selected_rows) == 0:
        print("No runs selected.")
        return

    results = []

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(run_one, row, args) for row in selected_rows]

        for future in as_completed(futures):
            result = future.result()
            results.append(result)

            print(
                f"[{result['status']}] {result['run_name']} "
                f"(returncode={result['returncode']})"
            )

            if result["status"] == "failed":
                print(f"  stderr: {result.get('stderr')}")

    failed = [r for r in results if r["returncode"] != 0]

    print("\nDone.")
    print(f"Completed: {len(results) - len(failed)}")
    print(f"Failed: {len(failed)}")

    summary_path = Path(args.log_dir) / "summary.csv"
    with open(summary_path, "w", newline="") as f:
        fieldnames = ["run_name", "status", "returncode", "cmd", "stdout", "stderr"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({k: r.get(k, "") for k in fieldnames})

    print(f"Summary written to: {summary_path}")

    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()