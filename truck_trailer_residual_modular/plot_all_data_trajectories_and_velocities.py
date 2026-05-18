from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Batch-plot trajectory, vx/vy, yaw-rate, and angular-acceleration curves for all train-segment CSVs "
            "under a data root and save them into one folder."
        )
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=None,
        help=(
            "Root directory containing *_train_segment_*.csv files. "
            "Defaults to truck_trailer_residual_modular/data/data."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Directory where all figures and the summary CSV will be saved. "
            "Defaults to truck_trailer_residual_modular/data/all_data_plots."
        ),
    )
    return parser.parse_args()


def load_runtime_dependencies() -> tuple[Path, Any, Any]:
    try:
        from .data_utils import collect_control_and_trajectory_csvs, load_truck_trailer_data_as_segment
    except ImportError:
        from data_utils import collect_control_and_trajectory_csvs, load_truck_trailer_data_as_segment

    module_dir = Path(__file__).resolve().parent
    return module_dir, collect_control_and_trajectory_csvs, load_truck_trailer_data_as_segment


def default_input_root(module_dir: Path) -> Path:
    return module_dir / "data" / "data"


def default_output_dir(module_dir: Path) -> Path:
    return module_dir / "data" / "all_data_plots"


def has_distinct_trailer_trace(states: np.ndarray) -> bool:
    return not np.allclose(states[:, 6:12], states[:, 0:6], atol=1.0e-6, rtol=1.0e-6)


def sanitize_label(text: str) -> str:
    safe_chars = []
    for char in text:
        if char.isalnum() or char in {"-", "_"}:
            safe_chars.append(char)
        else:
            safe_chars.append("_")
    sanitized = "".join(safe_chars).strip("_")
    return sanitized or "segment"


def build_output_stem(csv_path: Path, input_root: Path) -> str:
    try:
        relative = csv_path.relative_to(input_root)
        pieces = relative.with_suffix("").parts
    except ValueError:
        pieces = csv_path.with_suffix("").parts[-3:]
    return "__".join(sanitize_label(piece) for piece in pieces)


def save_figure(fig: plt.Figure, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def compute_angular_acceleration_degps2(yaw_rate_radps: np.ndarray, time_s: np.ndarray) -> np.ndarray:
    yaw_rate_degps = np.rad2deg(np.asarray(yaw_rate_radps, dtype=np.float64))
    time_s = np.asarray(time_s, dtype=np.float64)
    if yaw_rate_degps.size < 2 or time_s.size < 2:
        return np.zeros_like(yaw_rate_degps, dtype=np.float64)
    return np.gradient(yaw_rate_degps, time_s, edge_order=1)


def plot_segment_figure(segment: Any, output_path: Path) -> None:
    time = segment.time.astype(np.float64)
    states = segment.real_rollout.astype(np.float64)
    show_trailer = has_distinct_trailer_trace(states)
    tractor_yaw_rate_degps = np.rad2deg(states[:, 5])
    trailer_yaw_rate_degps = np.rad2deg(states[:, 11]) if show_trailer else None
    tractor_angular_accel_degps2 = compute_angular_acceleration_degps2(states[:, 5], time)
    trailer_angular_accel_degps2 = compute_angular_acceleration_degps2(states[:, 11], time) if show_trailer else None

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    trajectory_axis = axes[0, 0]
    velocity_axis = axes[0, 1]
    yaw_rate_axis = axes[1, 0]
    angular_accel_axis = axes[1, 1]

    trajectory_axis.plot(states[:, 0], states[:, 1], linewidth=2.0, color="#1f77b4", label="Tractor")
    trajectory_axis.scatter(states[0, 0], states[0, 1], s=36, color="#1f77b4", marker="o", label="Tractor start")
    trajectory_axis.scatter(states[-1, 0], states[-1, 1], s=36, color="#1f77b4", marker="x", label="Tractor end")
    if show_trailer:
        trajectory_axis.plot(states[:, 6], states[:, 7], linewidth=1.8, color="#ff7f0e", label="Trailer")
        trajectory_axis.scatter(states[0, 6], states[0, 7], s=30, color="#ff7f0e", marker="o", label="Trailer start")
        trajectory_axis.scatter(states[-1, 6], states[-1, 7], s=30, color="#ff7f0e", marker="x", label="Trailer end")
    trajectory_axis.set_title("XY Trajectory")
    trajectory_axis.set_xlabel("X (m)")
    trajectory_axis.set_ylabel("Y (m)")
    trajectory_axis.axis("equal")
    trajectory_axis.grid(True, linestyle="--", alpha=0.35)
    trajectory_axis.legend(fontsize=8)

    velocity_axis.plot(time, states[:, 3], linewidth=1.8, color="#1f77b4", label="Tractor Vx")
    velocity_axis.plot(time, states[:, 4], linewidth=1.8, color="#2ca02c", label="Tractor Vy")
    if show_trailer:
        velocity_axis.plot(time, states[:, 9], linewidth=1.5, linestyle="--", color="#ff7f0e", label="Trailer Vx")
        velocity_axis.plot(time, states[:, 10], linewidth=1.5, linestyle="--", color="#d62728", label="Trailer Vy")
    velocity_axis.axhline(0.0, linewidth=1.0, linestyle="--", color="black", alpha=0.5)
    velocity_axis.set_title("Velocity vs Time")
    velocity_axis.set_xlabel("Time (s)")
    velocity_axis.set_ylabel("Velocity (m/s)")
    velocity_axis.grid(True, linestyle="--", alpha=0.35)
    velocity_axis.legend(fontsize=8)

    yaw_rate_axis.plot(time, tractor_yaw_rate_degps, linewidth=1.8, color="#9467bd", label="Tractor yaw rate")
    if show_trailer and trailer_yaw_rate_degps is not None:
        yaw_rate_axis.plot(
            time,
            trailer_yaw_rate_degps,
            linewidth=1.5,
            linestyle="--",
            color="#8c564b",
            label="Trailer yaw rate",
        )
    yaw_rate_axis.axhline(0.0, linewidth=1.0, linestyle="--", color="black", alpha=0.5)
    yaw_rate_axis.set_title("Yaw Rate vs Time")
    yaw_rate_axis.set_xlabel("Time (s)")
    yaw_rate_axis.set_ylabel("Yaw rate (deg/s)")
    yaw_rate_axis.grid(True, linestyle="--", alpha=0.35)
    yaw_rate_axis.legend(fontsize=8)

    angular_accel_axis.plot(
        time,
        tractor_angular_accel_degps2,
        linewidth=1.8,
        color="#17becf",
        label="Tractor angular accel",
    )
    if show_trailer and trailer_angular_accel_degps2 is not None:
        angular_accel_axis.plot(
            time,
            trailer_angular_accel_degps2,
            linewidth=1.5,
            linestyle="--",
            color="#bcbd22",
            label="Trailer angular accel",
        )
    angular_accel_axis.axhline(0.0, linewidth=1.0, linestyle="--", color="black", alpha=0.5)
    angular_accel_axis.set_title("Angular Acceleration vs Time")
    angular_accel_axis.set_xlabel("Time (s)")
    angular_accel_axis.set_ylabel("Angular accel (deg/s^2)")
    angular_accel_axis.grid(True, linestyle="--", alpha=0.35)
    angular_accel_axis.legend(fontsize=8)

    fig.suptitle(segment.segment_name, fontsize=13)
    save_figure(fig, output_path)


def plot_all_trajectories(segments: list[Any], output_path: Path) -> None:
    fig, axis = plt.subplots(figsize=(10, 8))
    for segment in segments:
        states = segment.real_rollout.astype(np.float64)
        label = segment.csv_path.parent.parent.name + "/" + segment.csv_path.parent.name
        axis.plot(states[:, 0], states[:, 1], linewidth=1.5, label=label)
    axis.set_title("All Tractor Trajectories")
    axis.set_xlabel("X (m)")
    axis.set_ylabel("Y (m)")
    axis.axis("equal")
    axis.grid(True, linestyle="--", alpha=0.35)
    axis.legend(fontsize=7, ncol=2)
    save_figure(fig, output_path)


def plot_all_velocities(segments: list[Any], output_path: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=False)
    for segment in segments:
        time = segment.time.astype(np.float64)
        states = segment.real_rollout.astype(np.float64)
        label = segment.csv_path.parent.parent.name + "/" + segment.csv_path.parent.name
        axes[0].plot(time, states[:, 3], linewidth=1.4, label=label)
        axes[1].plot(time, states[:, 4], linewidth=1.4, label=label)

    axes[0].set_title("All Tractor Vx Curves")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Vx (m/s)")
    axes[0].grid(True, linestyle="--", alpha=0.35)
    axes[0].legend(fontsize=7, ncol=2)

    axes[1].set_title("All Tractor Vy Curves")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Vy (m/s)")
    axes[1].grid(True, linestyle="--", alpha=0.35)
    axes[1].legend(fontsize=7, ncol=2)

    save_figure(fig, output_path)


def plot_all_yaw_rate_and_angular_acceleration(segments: list[Any], output_path: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=False)
    for segment in segments:
        time = segment.time.astype(np.float64)
        states = segment.real_rollout.astype(np.float64)
        label = segment.csv_path.parent.parent.name + "/" + segment.csv_path.parent.name
        axes[0].plot(time, np.rad2deg(states[:, 5]), linewidth=1.4, label=label)
        axes[1].plot(time, compute_angular_acceleration_degps2(states[:, 5], time), linewidth=1.4, label=label)

    axes[0].set_title("All Tractor Yaw-Rate Curves")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Yaw rate (deg/s)")
    axes[0].grid(True, linestyle="--", alpha=0.35)
    axes[0].legend(fontsize=7, ncol=2)

    axes[1].set_title("All Tractor Angular-Acceleration Curves")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Angular accel (deg/s^2)")
    axes[1].grid(True, linestyle="--", alpha=0.35)
    axes[1].legend(fontsize=7, ncol=2)

    save_figure(fig, output_path)


def write_summary_csv(rows: list[dict[str, str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["segment_name", "csv_path", "plot_path"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    module_dir, collect_control_and_trajectory_csvs, load_truck_trailer_data_as_segment = load_runtime_dependencies()
    input_root = args.input_root if args.input_root is not None else default_input_root(module_dir)
    output_dir = args.output_dir if args.output_dir is not None else default_output_dir(module_dir)

    csv_paths = collect_control_and_trajectory_csvs(input_root)
    csv_paths = [path for path in csv_paths if "_train_segment_" in path.name]
    if not csv_paths:
        raise FileNotFoundError(f"No *_train_segment_*.csv files found under {input_root}")

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, str]] = []
    segments: list[Any] = []

    print(f"Found {len(csv_paths)} train-segment CSV files under {input_root}")
    for csv_path in csv_paths:
        segment = load_truck_trailer_data_as_segment(csv_path)
        segments.append(segment)
        output_stem = build_output_stem(csv_path, input_root)
        plot_path = output_dir / f"{output_stem}.png"
        plot_segment_figure(segment, plot_path)
        summary_rows.append(
            {
                "segment_name": segment.segment_name,
                "csv_path": str(csv_path),
                "plot_path": str(plot_path),
            }
        )
        print(f"Saved plot: {plot_path}")

    plot_all_trajectories(segments, output_dir / "all_tractor_trajectories.png")
    plot_all_velocities(segments, output_dir / "all_tractor_vx_vy.png")
    plot_all_yaw_rate_and_angular_acceleration(segments, output_dir / "all_tractor_yaw_rate_angular_acceleration.png")
    write_summary_csv(summary_rows, output_dir / "plot_manifest.csv")
    print(f"All plots saved to: {output_dir}")


if __name__ == "__main__":
    main()
