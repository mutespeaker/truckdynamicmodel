from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd
import torch

try:
    from .base_model import TruckTrailerNominalDynamics
    from .constants import (
        BASE_MODEL_PARAMS,
        LEARNING_RATE,
        MIN_LEARNING_RATE,
        RUNS_ROOT,
        TRAIN_BATCH_SIZE,
        TRAIN_EPOCHS,
        TRAIN_NUM_WORKERS,
        VXYR_SMOOTHNESS_BASE_FRACTION,
        VXYR_SMOOTHNESS_DELTA_R_DEGPS,
        VXYR_SMOOTHNESS_DELTA_VX_MPS,
        VXYR_SMOOTHNESS_DELTA_VY_MPS,
        VXYR_SMOOTHNESS_FINAL_MULTIPLIER,
        VXYR_SMOOTHNESS_WEIGHT,
        VXYR_SMOOTHNESS_ZERO_FRACTION,
    )
    from .data_utils import (
        SegmentData,
        build_train_val_by_segments,
        collect_control_and_trajectory_csvs,
        load_truck_trailer_data_as_segment,
    )
    from .training import (
        export_dataset_split_tables,
        plot_key_state_timeseries,
        plot_training_history,
        plot_trajectory,
        print_rollout_rmse,
        rollout_models_teacher_forcing,
        train_error_model_multirun,
    )
except ImportError:
    from base_model import TruckTrailerNominalDynamics
    from constants import (
        BASE_MODEL_PARAMS,
        LEARNING_RATE,
        MIN_LEARNING_RATE,
        RUNS_ROOT,
        TRAIN_BATCH_SIZE,
        TRAIN_EPOCHS,
        TRAIN_NUM_WORKERS,
        VXYR_SMOOTHNESS_BASE_FRACTION,
        VXYR_SMOOTHNESS_DELTA_R_DEGPS,
        VXYR_SMOOTHNESS_DELTA_VX_MPS,
        VXYR_SMOOTHNESS_DELTA_VY_MPS,
        VXYR_SMOOTHNESS_FINAL_MULTIPLIER,
        VXYR_SMOOTHNESS_WEIGHT,
        VXYR_SMOOTHNESS_ZERO_FRACTION,
    )
    from data_utils import (
        SegmentData,
        build_train_val_by_segments,
        collect_control_and_trajectory_csvs,
        load_truck_trailer_data_as_segment,
    )
    from training import (
        export_dataset_split_tables,
        plot_key_state_timeseries,
        plot_training_history,
        plot_trajectory,
        print_rollout_rmse,
        rollout_models_teacher_forcing,
        train_error_model_multirun,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the truck-trailer residual model. Supports either all runs under runs_root or a single run/csv.",
    )
    parser.add_argument(
        "--input-path",
        "--run-dir",
        dest="input_path",
        type=Path,
        default=None,
        help="Optional carsim_runs root, run directory, outputs directory, or control_and_trajectory.csv path.",
    )
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation ratio. Single-segment mode uses a tail split.")
    parser.add_argument("--seed", type=int, default=100, help="Random seed.")
    parser.add_argument("--epochs", type=int, default=TRAIN_EPOCHS, help="Training epochs.")
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE, help="Optimizer learning rate.")
    parser.add_argument(
        "--min-learning-rate",
        type=float,
        default=MIN_LEARNING_RATE,
        help="Cosine annealing minimum learning rate.",
    )
    parser.add_argument("--batch-size", type=int, default=TRAIN_BATCH_SIZE, help="Training batch size.")
    parser.add_argument("--num-workers", type=int, default=TRAIN_NUM_WORKERS, help="DataLoader workers.")
    parser.add_argument(
        "--vx-vy-r-smoothness-weight",
        type=float,
        default=VXYR_SMOOTHNESS_WEIGHT,
        help=(
            "Base local smoothness regularization weight. Training uses a staged schedule: "
            "0 for the first 25 percent of optimizer steps, this value for the next 25 percent, "
            "and 10x this value for the final 50 percent. Set 0 to disable."
        ),
    )
    parser.add_argument(
        "--summary-dir",
        type=Path,
        default=None,
        help="Optional parent directory for training runs. A dedicated run subdirectory will be created inside it.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Optional run name. Defaults to a timestamped name derived from the input scope.",
    )
    return parser.parse_args()


def sanitize_name(value: str) -> str:
    sanitized = re.sub(r"[^0-9A-Za-z._-]+", "_", value.strip())
    sanitized = sanitized.strip("._-")
    return sanitized or "run"


def compact_name(value: str, max_length: int = 48) -> str:
    sanitized = sanitize_name(value)
    if len(sanitized) <= max_length:
        return sanitized
    if max_length <= 0:
        return ""
    if max_length <= 8:
        return hashlib.sha1(sanitized.encode("utf-8")).hexdigest()[:max_length]
    digest = hashlib.sha1(sanitized.encode("utf-8")).hexdigest()[: min(6, max_length - 2)]
    head_length = max_length - len(digest) - 1
    return f"{sanitized[:head_length]}_{digest}"


def resolve_summary_root(args: argparse.Namespace, csv_list: list[Path]) -> Path:
    if args.summary_dir is not None:
        return args.summary_dir
    if args.input_path is not None and len(csv_list) == 1:
        return csv_list[0].parent / "truck_trailer_training_summary_modular"
    if args.input_path is not None and Path(args.input_path).is_dir():
        return Path(args.input_path) / "truck_trailer_multirun_training_summary_modular"
    return RUNS_ROOT / "truck_trailer_multirun_training_summary_modular"


def build_default_run_name(args: argparse.Namespace, csv_list: list[Path]) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.input_path is None:
        scope_token = "runs_root"
    elif len(csv_list) == 1:
        scope_token = csv_list[0].stem
    else:
        scope_token = Path(args.input_path).name
    scope_slug = compact_name(scope_token, max_length=20)
    return f"run_{timestamp}_{scope_slug}_e{int(args.epochs)}_s{int(args.seed)}"


def create_run_dir(summary_root: Path, run_name: str) -> Path:
    summary_root.mkdir(parents=True, exist_ok=True)
    candidate = summary_root / compact_name(run_name, max_length=48)
    if not candidate.exists():
        candidate.mkdir(parents=True, exist_ok=False)
        return candidate

    suffix = 2
    while True:
        numbered = summary_root / f"{compact_name(run_name, max_length=44)}_{suffix:02d}"
        if not numbered.exists():
            numbered.mkdir(parents=True, exist_ok=False)
            return numbered
        suffix += 1


def build_validation_dir_token(seg: SegmentData) -> str:
    stem = seg.csv_path.stem
    match = re.match(r"(?P<stamp>\d{8}_\d{6})\.(?P<frac>\d{5})_interpolated_train_segment_(?P<seg>\d+)$", stem)
    if match is not None:
        seg_index = int(match.group("seg"))
        stamp_token = match.group("stamp").replace("_", "")[2:]
        frac_token = match.group("frac")[:2]
        return f"{stamp_token}_s{seg_index:03d}_{frac_token}"
    return compact_name(stem, max_length=20)


def build_validation_plot_dir(run_dir: Path, seg: SegmentData, index: int) -> Path:
    scenario_token = compact_name(
        seg.csv_path.parent.parent.name if len(seg.csv_path.parents) >= 2 else seg.segment_name,
        max_length=12,
    )
    csv_token = build_validation_dir_token(seg)
    # Keep validation leaf directories short enough for Windows/PIL save paths.
    dir_name = compact_name(f"{index:03d}_{scenario_token}_{csv_token}", max_length=40)
    out_dir = run_dir / "val_rollouts" / dir_name
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def export_training_history_csv(history: dict[str, list[float]], output_dir: Path) -> Path:
    output_path = output_dir / "truck_trailer_training_history.csv"
    pd.DataFrame(history).to_csv(output_path, index=False, encoding="utf-8-sig")
    return output_path


def export_validation_rollout_summary(rows: list[dict[str, object]], output_dir: Path) -> Path:
    output_path = output_dir / "truck_trailer_validation_rollout_summary.csv"
    pd.DataFrame(rows).to_csv(output_path, index=False, encoding="utf-8-sig")
    return output_path


def save_run_metadata(metadata: dict[str, object], output_dir: Path) -> Path:
    output_path = output_dir / "run_metadata.json"
    output_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path


def main() -> None:
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device = {device}")

    csv_list = collect_control_and_trajectory_csvs(args.input_path, RUNS_ROOT)
    if not csv_list:
        raise FileNotFoundError(
            f"Could not find any control_and_trajectory.csv files under {RUNS_ROOT / 'python_run_*' / 'outputs'}"
        )

    if args.input_path is None:
        print(f"Found {len(csv_list)} candidate segments. Latest file: {csv_list[0]}")
    elif len(csv_list) == 1:
        print(f"Using specified input: {csv_list[0]}")
    else:
        print(f"Using specified root: {Path(args.input_path)}")
        print(f"Resolved {len(csv_list)} candidate segments. Latest file: {csv_list[0]}")

    segments: list[SegmentData] = []
    for csv_path in csv_list:
        try:
            seg = load_truck_trailer_data_as_segment(csv_path)
            segments.append(seg)
        except Exception as exc:
            print(f"[Skip] Failed to load {csv_path}: {exc}")

    if not segments:
        raise ValueError("No valid segments were loaded.")

    train_segments, val_segments = build_train_val_by_segments(segments, val_ratio=args.val_ratio, seed=args.seed)
    print("\n===== Segment Split =====")
    print(f"Train segments: {len(train_segments)}")
    print(f"Val segments  : {len(val_segments)}")
    if len(segments) == 1:
        print("Single-segment mode is active: train/val were split along time without changing model logic.")

    summary_root = resolve_summary_root(args, csv_list)
    run_name = args.run_name if args.run_name is not None else build_default_run_name(args, csv_list)
    run_dir = create_run_dir(summary_root, run_name)
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    print(f"Summary root      : {summary_root}")
    print(f"Active run dir    : {run_dir}")
    print(f"Run checkpoints   : {checkpoint_dir}")

    split_table_path, val_table_path = export_dataset_split_tables(train_segments, val_segments, run_dir)
    print(f"Saved split table : {split_table_path}")
    print(f"Saved val table   : {val_table_path}")

    base_model = TruckTrailerNominalDynamics(BASE_MODEL_PARAMS).to(device)
    error_model, feature_context, loss_context, history, checkpoint_paths = train_error_model_multirun(
        base_model=base_model,
        train_segments=train_segments,
        val_segments=val_segments,
        device=device,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        min_learning_rate=args.min_learning_rate,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        checkpoint_dir=checkpoint_dir,
        export_compatibility_checkpoints=True,
        vx_vy_r_smoothness_weight=args.vx_vy_r_smoothness_weight,
    )

    loss_plot = plot_training_history(history, run_dir)
    history_csv = export_training_history_csv(history, run_dir)
    print(f"Saved training history plot: {loss_plot}")
    print(f"Saved training history csv : {history_csv}")
    for label, path in checkpoint_paths.items():
        print(f"Saved checkpoint [{label}]: {path}")

    print("\n===== Validation Rollout =====")
    validation_rows: list[dict[str, object]] = []
    for index, seg in enumerate(val_segments):
        base_rollout, corrected_rollout = rollout_models_teacher_forcing(
            base_model=base_model,
            error_model=error_model,
            real_rollout=seg.real_rollout,
            control_sequence=seg.control_sequence,
            trailer_mass_kg=seg.trailer_mass_kg,
            dt_values=seg.dt_values,
            feature_context=feature_context,
            loss_context=loss_context,
            device=device,
        )
        validation_plot_dir = build_validation_plot_dir(run_dir, seg, index)
        traj_path = plot_trajectory(seg.real_rollout, base_rollout, corrected_rollout, validation_plot_dir)
        state_path = plot_key_state_timeseries(seg.time, seg.real_rollout, base_rollout, corrected_rollout, validation_plot_dir)
        rmse = print_rollout_rmse(seg.real_rollout, base_rollout, corrected_rollout)
        validation_rows.append(
            {
                "index_in_val": index,
                "segment_name": seg.segment_name,
                "csv_path": str(seg.csv_path),
                "plot_dir": str(validation_plot_dir),
                **rmse,
            }
        )
        print(f"[{seg.segment_name}] trajectory plot: {traj_path}")
        print(f"[{seg.segment_name}] state plot     : {state_path}")
        print(f"[{seg.segment_name}] rmse summary   : {rmse}")

    validation_summary_path = export_validation_rollout_summary(validation_rows, run_dir)
    metadata_path = save_run_metadata(
        {
            "run_name": run_dir.name,
            "run_dir": str(run_dir),
            "summary_root": str(summary_root),
            "input_path": None if args.input_path is None else str(args.input_path),
            "resolved_csv_count": len(csv_list),
            "loaded_segment_count": len(segments),
            "train_segment_count": len(train_segments),
            "val_segment_count": len(val_segments),
            "seed": int(args.seed),
            "epochs": int(args.epochs),
            "learning_rate": float(args.learning_rate),
            "min_learning_rate": float(args.min_learning_rate),
            "batch_size": int(args.batch_size),
            "num_workers": int(args.num_workers),
            "vx_vy_r_smoothness_base_weight": float(args.vx_vy_r_smoothness_weight),
            "vx_vy_r_smoothness_zero_fraction": float(VXYR_SMOOTHNESS_ZERO_FRACTION),
            "vx_vy_r_smoothness_base_fraction": float(VXYR_SMOOTHNESS_BASE_FRACTION),
            "vx_vy_r_smoothness_final_multiplier": float(VXYR_SMOOTHNESS_FINAL_MULTIPLIER),
            "vx_vy_r_smoothness_regularized_states": ["vx_t", "vy_t", "r_t", "vx_s", "vy_s", "r_s"],
            "vx_vy_r_smoothness_delta_vx_mps": float(VXYR_SMOOTHNESS_DELTA_VX_MPS),
            "vx_vy_r_smoothness_delta_vy_mps": float(VXYR_SMOOTHNESS_DELTA_VY_MPS),
            "vx_vy_r_smoothness_delta_r_degps": float(VXYR_SMOOTHNESS_DELTA_R_DEGPS),
            "device": str(device),
            "checkpoint_paths": {key: str(path) for key, path in checkpoint_paths.items()},
            "artifacts": {
                "split_table": str(split_table_path),
                "validation_segments_table": str(val_table_path),
                "training_history_plot": str(loss_plot),
                "training_history_csv": str(history_csv),
                "validation_rollout_summary": str(validation_summary_path),
            },
        },
        run_dir,
    )
    print(f"Saved validation summary : {validation_summary_path}")
    print(f"Saved run metadata       : {metadata_path}")


if __name__ == "__main__":
    main()
