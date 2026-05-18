from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

try:
    from .constants import RUNS_ROOT
    from .data_utils import build_feature_context_tensors, collect_control_and_trajectory_csvs, load_truck_trailer_data_as_segment
    from .inference_main import (
        BASE_LABEL,
        NN_LABEL,
        REAL_LABEL,
        InferenceSegment,
        build_base_model,
        compute_rmse_summary,
        export_results_csv,
        extract_feature_context,
        load_error_model,
        plot_state_error_all,
        rollout_recursive,
        save_inference_figure,
    )
except ImportError:
    from constants import RUNS_ROOT
    from data_utils import build_feature_context_tensors, collect_control_and_trajectory_csvs, load_truck_trailer_data_as_segment
    from inference_main import (
        BASE_LABEL,
        NN_LABEL,
        REAL_LABEL,
        InferenceSegment,
        build_base_model,
        compute_rmse_summary,
        export_results_csv,
        extract_feature_context,
        load_error_model,
        plot_state_error_all,
        rollout_recursive,
        save_inference_figure,
    )


WINDOW_SECONDS_DEFAULT = 10.0
STRIDE_SECONDS_DEFAULT = 10.0
WINDOW_MODE_KEY = "recursive_rollout"


@dataclass
class WindowRolloutRecord:
    window_segment: InferenceSegment
    window_index: int
    start_step: int
    start_time_s: float
    start_time_from_segment_s: float
    duration_s: float
    base_rollout: np.ndarray
    nn_rollout: np.ndarray
    results_csv: Path
    error_png: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run recursive rollout on sliding 10-second windows, seeded every 10 seconds from the measured trajectory.",
    )
    parser.add_argument(
        "--input-path",
        "--run-dir",
        dest="input_path",
        type=Path,
        default=None,
        help="Optional carsim_runs root, run directory, outputs directory, or control_and_trajectory.csv path.",
    )
    parser.add_argument(
        "--summary-path",
        type=Path,
        default=None,
        help="Optional summary csv path. Defaults to a per-input file for single-input mode or a global file otherwise.",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=None,
        help="Optional checkpoint .pth path, run directory, or checkpoint directory.",
    )
    parser.add_argument(
        "--window-seconds",
        type=float,
        default=WINDOW_SECONDS_DEFAULT,
        help="Recursive rollout horizon for each window in seconds. Default: 10.",
    )
    parser.add_argument(
        "--stride-seconds",
        type=float,
        default=STRIDE_SECONDS_DEFAULT,
        help="Stride between window initial states in seconds. Default: 10.",
    )
    return parser.parse_args()


def resolve_output_dir(csv_path: Path) -> Path:
    out_dir = csv_path.parent / f"{csv_path.stem}_recursive_10s_eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def load_parent_segment(csv_path: Path) -> InferenceSegment:
    seg = load_truck_trailer_data_as_segment(csv_path)
    return InferenceSegment(
        csv_path=seg.csv_path,
        scenario_name=seg.segment_name,
        segment_name=csv_path.stem,
        out_dir=resolve_output_dir(csv_path),
        time=seg.time.astype(np.float32),
        dt_values=seg.dt_values.astype(np.float32),
        real_rollout=seg.real_rollout.astype(np.float32),
        initial_state=seg.initial_state.astype(np.float32),
        control_sequence=seg.control_sequence.astype(np.float32),
        trailer_mass_kg=seg.trailer_mass_kg.astype(np.float32),
    )


def compute_window_and_stride_steps(seg: InferenceSegment, window_seconds: float, stride_seconds: float) -> tuple[int, int, float]:
    if len(seg.dt_values) == 0:
        raise ValueError("Segment contains no dt values.")
    dt_nominal = float(np.mean(seg.dt_values))
    if dt_nominal <= 0.0:
        raise ValueError(f"Invalid nominal dt: {dt_nominal}")
    window_steps = max(1, int(round(window_seconds / dt_nominal)))
    stride_steps = max(1, int(round(stride_seconds / dt_nominal)))
    return window_steps, stride_steps, dt_nominal


def collect_window_start_indices(seg: InferenceSegment, window_steps: int, stride_steps: int) -> list[int]:
    max_start_step = len(seg.control_sequence) - window_steps
    if max_start_step < 0:
        return []
    return list(range(0, max_start_step + 1, stride_steps))


def build_window_segment(parent_seg: InferenceSegment, start_step: int, window_steps: int, window_index: int) -> InferenceSegment:
    end_step = start_step + window_steps
    start_time_s = float(parent_seg.time[start_step])
    start_time_from_segment_s = float(start_time_s - float(parent_seg.time[0]))
    window_out_dir = parent_seg.out_dir / f"w{window_index:03d}_t{start_time_from_segment_s:07.2f}s"
    window_out_dir.mkdir(parents=True, exist_ok=True)

    window_time = parent_seg.time[start_step : end_step + 1].copy().astype(np.float32)
    window_time = window_time - window_time[0]

    return InferenceSegment(
        csv_path=parent_seg.csv_path,
        scenario_name=parent_seg.scenario_name,
        segment_name=f"{parent_seg.segment_name}_w{window_index:03d}",
        out_dir=window_out_dir,
        time=window_time,
        dt_values=parent_seg.dt_values[start_step:end_step].copy().astype(np.float32),
        real_rollout=parent_seg.real_rollout[start_step : end_step + 1].copy().astype(np.float32),
        initial_state=parent_seg.real_rollout[start_step].copy().astype(np.float32),
        control_sequence=parent_seg.control_sequence[start_step:end_step].copy().astype(np.float32),
        trailer_mass_kg=parent_seg.trailer_mass_kg[start_step:end_step].copy().astype(np.float32),
    )


def plot_combined_window_trajectories(parent_seg: InferenceSegment, records: list[WindowRolloutRecord]) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for axis, x_index, y_index, title in (
        (axes[0], 0, 1, "Tractor Trajectory"),
        (axes[1], 6, 7, "Trailer Trajectory"),
    ):
        axis.plot(
            parent_seg.real_rollout[:, x_index],
            parent_seg.real_rollout[:, y_index],
            color="black",
            linewidth=2.0,
            alpha=0.9,
            label=REAL_LABEL,
        )
        for record in records:
            real_window = record.window_segment.real_rollout
            axis.plot(
                real_window[:, x_index],
                real_window[:, y_index],
                color="black",
                linewidth=1.0,
                alpha=0.15,
            )
            axis.plot(
                record.base_rollout[:, x_index],
                record.base_rollout[:, y_index],
                color="#d97706",
                linewidth=1.2,
                linestyle="--",
                alpha=0.65,
                label=BASE_LABEL if record.window_index == 0 else None,
            )
            axis.plot(
                record.nn_rollout[:, x_index],
                record.nn_rollout[:, y_index],
                color="#2563eb",
                linewidth=1.35,
                alpha=0.72,
                label=NN_LABEL if record.window_index == 0 else None,
            )
            axis.scatter(
                real_window[0, x_index],
                real_window[0, y_index],
                color="black",
                s=12,
                alpha=0.45,
            )
        axis.set_title(f"Recursive 10s Windows {title}")
        axis.set_xlabel("X (m)")
        axis.set_ylabel("Y (m)")
        axis.grid(True, linestyle="--", alpha=0.35)
        axis.set_aspect("equal", adjustable="box")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.02))

    out_path = parent_seg.out_dir / "recursive_10s_windows_trajectory.png"
    save_inference_figure(fig, out_path, top_margin=0.92)
    return out_path


def compute_window_summary(
    parent_seg: InferenceSegment,
    record: WindowRolloutRecord,
) -> dict[str, float | str]:
    summary: dict[str, float | str] = {
        "scenario_name": parent_seg.scenario_name,
        "segment_name": parent_seg.segment_name,
        "csv_path": str(parent_seg.csv_path),
        "output_dir": str(record.window_segment.out_dir),
        "window_index": int(record.window_index),
        "start_step": int(record.start_step),
        "start_time_s": float(record.start_time_s),
        "start_time_from_segment_s": float(record.start_time_from_segment_s),
        "duration_s": float(record.duration_s),
        "sample_count": int(len(record.window_segment.real_rollout)),
        "mean_trailer_mass_kg": float(np.mean(record.window_segment.trailer_mass_kg)),
    }
    summary.update(compute_rmse_summary(record.window_segment, record.base_rollout, record.nn_rollout, "recursive"))
    return summary


def export_summary_csv(rows: list[dict[str, float | str]], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False, encoding="utf-8-sig")
    return output_path


def main() -> None:
    args = parse_args()

    torch.manual_seed(42)
    np.random.seed(42)

    if args.window_seconds <= 0.0:
        raise ValueError("--window-seconds must be > 0.")
    if args.stride_seconds <= 0.0:
        raise ValueError("--stride-seconds must be > 0.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device = {device}")

    csvs = collect_control_and_trajectory_csvs(args.input_path, RUNS_ROOT)
    if not csvs:
        raise FileNotFoundError(f"Could not find data under {RUNS_ROOT}")
    if args.input_path is None:
        print(f"Found {len(csvs)} candidate segments. Latest file: {csvs[0]}")
    elif len(csvs) == 1:
        print(f"Using specified input: {csvs[0]}")
    else:
        print(f"Using specified root: {Path(args.input_path)}")
        print(f"Resolved {len(csvs)} candidate segments. Latest file: {csvs[0]}")

    error_model, checkpoint_metadata, checkpoint_path = load_error_model(device, args.checkpoint_path)
    print(f"Loaded checkpoint: {checkpoint_path}")

    base_model = build_base_model(checkpoint_metadata, device)
    feature_context = extract_feature_context(checkpoint_metadata)
    feature_context_tensors = None
    if feature_context is not None:
        feature_context_tensors = build_feature_context_tensors(feature_context, device)
    else:
        print("Checkpoint does not contain feature normalization statistics; raw features will be used.")

    all_summary_rows: list[dict[str, float | str]] = []
    processed_segment_count = 0
    processed_window_count = 0

    for csv_path in csvs:
        try:
            parent_seg = load_parent_segment(csv_path)
        except Exception as exc:
            print(f"[Skip] {csv_path}: {exc}")
            continue

        try:
            window_steps, stride_steps, dt_nominal = compute_window_and_stride_steps(
                parent_seg,
                args.window_seconds,
                args.stride_seconds,
            )
        except Exception as exc:
            print(f"[Skip] {csv_path}: {exc}")
            continue

        start_indices = collect_window_start_indices(parent_seg, window_steps, stride_steps)
        if not start_indices:
            print(
                f"[Skip] {csv_path}: segment too short for a full {args.window_seconds:.2f}s window "
                f"(need {window_steps} steps, have {len(parent_seg.control_sequence)})."
            )
            continue

        window_records: list[WindowRolloutRecord] = []
        segment_summary_rows: list[dict[str, float | str]] = []

        for window_index, start_step in enumerate(start_indices):
            window_seg = build_window_segment(parent_seg, start_step, window_steps, window_index)
            recursive_base, recursive_nn = rollout_recursive(
                base_model=base_model,
                error_model=error_model,
                initial_state=window_seg.initial_state,
                control_sequence=window_seg.control_sequence,
                trailer_mass_kg=window_seg.trailer_mass_kg,
                dt_values=window_seg.dt_values,
                device=device,
                feature_context_tensors=feature_context_tensors,
            )
            results_csv = export_results_csv(window_seg, WINDOW_MODE_KEY, recursive_base, recursive_nn)
            error_png = plot_state_error_all(window_seg, WINDOW_MODE_KEY, recursive_base, recursive_nn)
            duration_s = float(np.sum(window_seg.dt_values))
            record = WindowRolloutRecord(
                window_segment=window_seg,
                window_index=window_index,
                start_step=start_step,
                start_time_s=float(parent_seg.time[start_step]),
                start_time_from_segment_s=float(parent_seg.time[start_step] - parent_seg.time[0]),
                duration_s=duration_s,
                base_rollout=recursive_base,
                nn_rollout=recursive_nn,
                results_csv=results_csv,
                error_png=error_png,
            )
            summary_row = compute_window_summary(parent_seg, record)
            window_records.append(record)
            segment_summary_rows.append(summary_row)
            all_summary_rows.append(summary_row)
            processed_window_count += 1

        combined_trajectory_png = plot_combined_window_trajectories(parent_seg, window_records)
        segment_summary_csv = export_summary_csv(segment_summary_rows, parent_seg.out_dir / "recursive_10s_window_summary.csv")
        processed_segment_count += 1

        print(f"\n[OK] {parent_seg.segment_name}")
        print(f"  scene    : {parent_seg.scenario_name}")
        print(f"  out_dir  : {parent_seg.out_dir}")
        print(
            f"  windows  : {len(window_records)} "
            f"(window={args.window_seconds:.2f}s/{window_steps} steps, stride={args.stride_seconds:.2f}s/{stride_steps} steps, dt≈{dt_nominal:.4f}s)"
        )
        print(f"  traj_all : {combined_trajectory_png.name}")
        print(f"  summary  : {segment_summary_csv.name}")
        if window_records:
            first_record = window_records[0]
            first_summary = segment_summary_rows[0]
            print(
                f"  first_w  : {first_record.window_segment.out_dir.name}, "
                f"{first_record.results_csv.name}, {first_record.error_png.name}"
            )
            print(
                "  first recursive articulation rmse (deg): "
                f"base={first_summary['recursive_rmse_base_articulation_deg']:.4f}, "
                f"nn={first_summary['recursive_rmse_nn_articulation_deg']:.4f}"
            )

    if processed_window_count == 0:
        raise RuntimeError("No 10-second recursive windows were processed successfully.")

    if args.summary_path is not None:
        summary_output_path = args.summary_path
    elif args.input_path is not None and len(csvs) == 1:
        summary_output_path = resolve_output_dir(csvs[0]) / "truck_trailer_recursive_10s_summary.csv"
    elif args.input_path is not None and Path(args.input_path).is_dir():
        summary_output_path = Path(args.input_path) / "truck_trailer_recursive_10s_summary.csv"
    else:
        summary_output_path = RUNS_ROOT / "truck_trailer_recursive_10s_summary.csv"

    summary_path = export_summary_csv(all_summary_rows, summary_output_path)
    print(f"\nsummary csv: {summary_path}")
    print(f"Completed {processed_window_count} recursive windows from {processed_segment_count} segments.")


if __name__ == "__main__":
    main()
