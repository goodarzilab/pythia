#!/usr/bin/env python
"""Reproduce one deployed Pythia BEACON structural-task result.

Trains a single task (--task ssp/ssi/cmp/dmp) using the hyperparameters
recorded in ``pythia_beacon_config.yaml``, then writes that task's
``metrics_best.json`` into the configured figure directory for downstream
plotting. Run once per task (see run_pythia_beacon_structural_tasks.sh,
which runs all 4 in sequence).
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = SCRIPT_DIR.parent.parent / "src" / "pythia"
DEFAULT_CONFIG = SCRIPT_DIR / "pythia_beacon_config.yaml"

ALL_TASKS = ["ssp", "ssi", "cmp", "dmp"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Path to pythia_beacon_config.yaml.",
    )
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=ALL_TASKS,
        help="Which BEACON structural task to (re)train.",
    )
    parser.add_argument(
        "--output-base", type=Path, default=None, help="Override common.output_base."
    )
    parser.add_argument(
        "--figure-dir", type=Path, default=None, help="Override common.figure_dir."
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Passed through as --device to trainers.",
    )
    parser.add_argument(
        "--skip-plot", action="store_true", help="Skip the plot.py step."
    )
    parser.add_argument(
        "--skip-infer", action="store_true", help="Skip the SSI inference step."
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print commands without running them."
    )
    return parser.parse_args()


def _flags(d: dict) -> list:
    """Convert a {key: value} config section to CLI flags (key -> --key)."""
    args = []
    for key, value in d.items():
        flag = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            if value:
                args.append(flag)
        elif isinstance(value, list):
            args.append(flag)
            args.extend(str(v) for v in value)
        else:
            args.append(flag)
            args.append(str(value))
    return args


def resolve_path(value: str) -> Path:
    return Path(value)


def build_ssp_cmd(cfg: dict, output_dir: Path, common: dict, device: str) -> list:
    task_cfg = cfg["tasks"]["ssp"]
    cmd = [
        sys.executable,
        str(PACKAGE_DIR / task_cfg["script"]),
        "--task",
        task_cfg["task_arg"],
        "--csv",
        str(resolve_path(task_cfg["data"]["csv"])),
        "--output-dir",
        str(output_dir),
        "--seed",
        str(common["seed"]),
        "--num-workers",
        str(common["num_workers"]),
    ]
    cmd += _flags(task_cfg["training"])
    cmd += _flags(task_cfg["model"])
    if device:
        cmd += ["--device", device]
    return cmd


def build_cmp_cmd(cfg: dict, output_dir: Path, common: dict, device: str) -> list:
    task_cfg = cfg["tasks"]["cmp"]
    data = task_cfg["data"]
    cmd = [
        sys.executable,
        str(PACKAGE_DIR / task_cfg["script"]),
        "--task",
        task_cfg["task_arg"],
        "--train-csv",
        str(resolve_path(data["train_csv"])),
        "--val-csv",
        str(resolve_path(data["val_csv"])),
        "--contact-map-dir",
        str(resolve_path(data["contact_map_dir"])),
        "--test-splits",
        *[str(s) for s in data["test_splits"]],
        "--output-dir",
        str(output_dir),
        "--seed",
        str(common["seed"]),
        "--num-workers",
        str(common["num_workers"]),
    ]
    cmd += _flags(task_cfg["training"])
    cmd += _flags(task_cfg["model"])
    if device:
        cmd += ["--device", device]
    return cmd


def build_dmp_cmd(cfg: dict, output_dir: Path, common: dict, device: str) -> list:
    task_cfg = cfg["tasks"]["dmp"]
    data = task_cfg["data"]
    cmd = [
        sys.executable,
        str(PACKAGE_DIR / task_cfg["script"]),
        "--task",
        task_cfg["task_arg"],
        "--train-csv",
        str(resolve_path(data["train_csv"])),
        "--val-csv",
        str(resolve_path(data["val_csv"])),
        "--distance-dir",
        str(resolve_path(data["distance_dir"])),
        "--test-splits",
        *[str(s) for s in data["test_splits"]],
        "--output-dir",
        str(output_dir),
        "--seed",
        str(common["seed"]),
        "--num-workers",
        str(common["num_workers"]),
    ]
    cmd += _flags(task_cfg["training"])
    cmd += _flags(task_cfg["model"])
    if device:
        cmd += ["--device", device]
    return cmd


def build_ssi_train_cmd(cfg: dict, output_dir: Path, common: dict, device: str) -> list:
    task_cfg = cfg["tasks"]["ssi"]
    data = task_cfg["data"]
    cmd = [
        sys.executable,
        str(PACKAGE_DIR / task_cfg["script"]),
        "--task",
        task_cfg["task_arg"],
        "--train-csv",
        str(resolve_path(data["train_csv"])),
        "--val-csv",
        str(resolve_path(data["val_csv"])),
        "--test-csv",
        str(resolve_path(data["test_csv"])),
        "--output-dir",
        str(output_dir),
        "--seed",
        str(common["seed"]),
        "--num-workers",
        str(common["num_workers"]),
    ]
    cmd += _flags(task_cfg["model"])
    cmd += _flags(task_cfg["training"])
    if device:
        cmd += ["--device", device]
    return cmd


def build_ssi_infer_cmd(cfg: dict, output_dir: Path, common: dict, device: str) -> list:
    task_cfg = cfg["tasks"]["ssi"]
    data = task_cfg["data"]
    model = task_cfg["model"]
    cmd = [
        sys.executable,
        str(PACKAGE_DIR / "infer.py"),
        "--task",
        "ssi",
        "--checkpoint",
        str(output_dir / "ssi_best.pt"),
        "--input-csv",
        str(resolve_path(data["test_csv"])),
        "--output-csv",
        str(output_dir / "ssi_predictions.csv"),
        "--max-len",
        str(task_cfg["training"]["max_len"]),
        "--batch-size",
        str(task_cfg["training"]["batch_eval"]),
        "--num-workers",
        str(common["num_workers"]),
    ]
    cmd += _flags(model)
    if device:
        cmd += ["--device", device]
    return cmd


def build_plot_cmd(task_for_plot: str, metrics_json: Path, output_dir: Path) -> list:
    return [
        sys.executable,
        str(PACKAGE_DIR / "plot.py"),
        "--task",
        task_for_plot,
        "--metrics-json",
        str(metrics_json),
        "--output-dir",
        str(output_dir / "figures"),
    ]


def run(cmd: list, dry_run: bool) -> None:
    print("+ " + " ".join(cmd))
    if not dry_run:
        subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    common = cfg["common"]

    output_base = args.output_base or resolve_path(common["output_base"])
    figure_dir = args.figure_dir or resolve_path(common["figure_dir"])
    figure_dir.mkdir(parents=True, exist_ok=True)

    task = args.task
    task_cfg = cfg["tasks"][task]
    output_dir = output_base / task_cfg["output_subdir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== [{task}] {task_cfg['description']}: training ===")
    if task == "ssp":
        cmd = build_ssp_cmd(cfg, output_dir, common, args.device)
    elif task == "cmp":
        cmd = build_cmp_cmd(cfg, output_dir, common, args.device)
    elif task == "dmp":
        cmd = build_dmp_cmd(cfg, output_dir, common, args.device)
    elif task == "ssi":
        cmd = build_ssi_train_cmd(cfg, output_dir, common, args.device)
    else:
        raise ValueError(f"Unknown task: {task}")
    run(cmd, args.dry_run)

    metrics_path = output_dir / "metrics_best.json"
    dest_path = figure_dir / f"{task}_pythia_performance.json"
    print(f"=== [{task}] copying {metrics_path} -> {dest_path} ===")
    if not args.dry_run:
        shutil.copy(metrics_path, dest_path)
    else:
        print(f"+ cp {metrics_path} {dest_path}")

    if task == "ssi" and not args.skip_infer:
        print(f"=== [{task}] inference on test set ===")
        run(build_ssi_infer_cmd(cfg, output_dir, common, args.device), args.dry_run)

    if not args.skip_plot:
        print(f"=== [{task}] plotting ===")
        run(build_plot_cmd(task, metrics_path, output_dir), args.dry_run)

    print(f"\n=== [{task}] BEACON structural task reproduction complete ===")
    print(f"Performance JSON written to: {dest_path}")


if __name__ == "__main__":
    main()
