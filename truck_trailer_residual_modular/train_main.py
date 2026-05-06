from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

try:
    from .base_model import TruckTrailerNominalDynamics
    from .constants import BASE_MODEL_PARAMS, LEARNING_RATE, MIN_LEARNING_RATE, RUNS_ROOT, TRAIN_BATCH_SIZE, TRAIN_EPOCHS, TRAIN_NUM_WORKERS
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
    from constants import BASE_MODEL_PARAMS, LEARNING_RATE, MIN_LEARNING_RATE, RUNS_ROOT, TRAIN_BATCH_SIZE, TRAIN_EPOCHS, TRAIN_NUM_WORKERS
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
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation ratio. Single-segment mode uses a tail split.")
    parser.add_argument("--seed", type=int, default=10, help="Random seed.")
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
        "--summary-dir",
        type=Path,
        default=None,
        help="Optional directory for summary outputs. Defaults to a global folder or the target run's outputs folder.",
    )
    return parser.parse_args()


def resolve_summary_dir(args: argparse.Namespace, csv_list: list[Path]) -> Path:
    if args.summary_dir is not None:
        return args.summary_dir
    if args.input_path is not None and len(csv_list) == 1:
        return csv_list[0].parent / "truck_trailer_training_summary_modular"
    if args.input_path is not None and Path(args.input_path).is_dir():
        return Path(args.input_path) / "truck_trailer_multirun_training_summary_modular"
    return RUNS_ROOT / "truck_trailer_multirun_training_summary_modular"


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

    global_plot_dir = resolve_summary_dir(args, csv_list)
    split_table_path, val_table_path = export_dataset_split_tables(train_segments, val_segments, global_plot_dir)
    print(f"Saved split table : {split_table_path}")
    print(f"Saved val table   : {val_table_path}")

    base_model = TruckTrailerNominalDynamics(BASE_MODEL_PARAMS).to(device)
    error_model, feature_context, loss_context, history = train_error_model_multirun(
        base_model=base_model,
        train_segments=train_segments,
        val_segments=val_segments,
        device=device,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        min_learning_rate=args.min_learning_rate,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    loss_plot = plot_training_history(history, global_plot_dir)
    print(f"Saved training history plot: {loss_plot}")

    print("\n===== Validation Rollout =====")
    for seg in val_segments:
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
        traj_path = plot_trajectory(seg.real_rollout, base_rollout, corrected_rollout, seg.plot_dir)
        state_path = plot_key_state_timeseries(seg.time, seg.real_rollout, base_rollout, corrected_rollout, seg.plot_dir)
        rmse = print_rollout_rmse(seg.real_rollout, base_rollout, corrected_rollout)
        print(f"[{seg.segment_name}] trajectory plot: {traj_path}")
        print(f"[{seg.segment_name}] state plot     : {state_path}")
        print(f"[{seg.segment_name}] rmse summary   : {rmse}")


if __name__ == "__main__":
    main()
