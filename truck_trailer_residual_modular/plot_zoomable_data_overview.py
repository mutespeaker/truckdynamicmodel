from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from .data_utils import SegmentData


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot a zoomable time/trajectory overview and steering-wheel-angle timeline for one specified dataset.",
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        default=None,
        help="CSV path, outputs directory, run directory, or a root containing candidate control_and_trajectory.csv files.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Optional image output path. Defaults to the segment plot directory.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Save the figure without opening the interactive Matplotlib window.",
    )
    return parser.parse_args()


def configure_pyplot(no_show: bool):
    import matplotlib

    if no_show:
        matplotlib.use("Agg")
    else:
        current_backend = str(matplotlib.get_backend()).lower()
        if "agg" in current_backend:
            for candidate in ("TkAgg", "QtAgg", "Qt5Agg"):
                try:
                    matplotlib.use(candidate)
                    break
                except Exception:
                    continue
    import matplotlib.pyplot as plt

    return plt


def is_non_interactive_backend(backend_name: str) -> bool:
    normalized = backend_name.strip().lower()
    return normalized in {"agg", "pdf", "ps", "svg", "cairo", "template"}


def load_runtime_dependencies() -> tuple[Path, Any, Any, Any]:
    try:
        from .constants import RUNS_ROOT
        from .data_utils import collect_control_and_trajectory_csvs, load_truck_trailer_data_as_segment, resolve_control_and_trajectory_csv
    except ImportError:
        from constants import RUNS_ROOT
        from data_utils import collect_control_and_trajectory_csvs, load_truck_trailer_data_as_segment, resolve_control_and_trajectory_csv

    return RUNS_ROOT, collect_control_and_trajectory_csvs, load_truck_trailer_data_as_segment, resolve_control_and_trajectory_csv


def resolve_single_csv(
    input_path: Path | None,
    runs_root: Path,
    collect_control_and_trajectory_csvs: Any,
    resolve_control_and_trajectory_csv: Any,
) -> Path:
    if input_path is None:
        csv_list = collect_control_and_trajectory_csvs(None, runs_root)
        if not csv_list:
            raise FileNotFoundError(f"Could not find any control_and_trajectory.csv files under {runs_root}.")
        print(f"No input path specified. Using latest file: {csv_list[0]}")
        return csv_list[0]

    input_path = Path(input_path)
    try:
        return resolve_control_and_trajectory_csv(input_path)
    except FileNotFoundError:
        csv_list = collect_control_and_trajectory_csvs(input_path, runs_root)
        if not csv_list:
            raise
        print(f"Resolved {len(csv_list)} candidate files under {input_path}. Using latest file: {csv_list[0]}")
        return csv_list[0]


def build_steering_wheel_deg_series(segment: SegmentData) -> np.ndarray:
    steer_deg = np.rad2deg(segment.control_sequence[:, 0].astype(np.float64))
    target_length = int(segment.time.shape[0])
    if steer_deg.shape[0] == target_length:
        return steer_deg.astype(np.float32)
    if steer_deg.shape[0] == target_length - 1:
        return np.concatenate([steer_deg, steer_deg[-1:]], axis=0).astype(np.float32)
    if steer_deg.shape[0] < target_length:
        fill_value = float(steer_deg[-1]) if steer_deg.size > 0 else 0.0
        padding = np.full(target_length - steer_deg.shape[0], fill_value, dtype=np.float64)
        return np.concatenate([steer_deg, padding], axis=0).astype(np.float32)
    return steer_deg[:target_length].astype(np.float32)


def has_distinct_trailer_trace(segment: SegmentData) -> bool:
    return not np.allclose(segment.real_rollout[:, 6:12], segment.real_rollout[:, 0:6], atol=1.0e-6, rtol=1.0e-6)


def default_output_path(segment: SegmentData) -> Path:
    return segment.plot_dir / "truck_trailer_zoomable_data_overview.png"


def plot_zoomable_overview(segment: SegmentData, output_path: Path, no_show: bool, plt: Any) -> Path:
    time = segment.time.astype(np.float64)
    states = segment.real_rollout.astype(np.float64)
    steer_deg = build_steering_wheel_deg_series(segment).astype(np.float64)
    show_trailer = has_distinct_trailer_trace(segment)

    fig, axes = plt.subplots(1, 3, figsize=(19, 6))
    try:
        fig.canvas.manager.set_window_title(f"Truck-Trailer Zoomable Overview - {segment.segment_name}")
    except Exception:
        pass

    trajectory_axis = axes[0]
    trajectory_axis.plot(states[:, 0], states[:, 1], label="Tractor trajectory", linewidth=1.8, color="#1f77b4")
    trajectory_axis.scatter(states[0, 0], states[0, 1], label="Tractor start", s=36, color="#1f77b4", marker="o")
    if show_trailer:
        trajectory_axis.plot(states[:, 6], states[:, 7], label="Trailer trajectory", linewidth=1.7, color="#ff7f0e")
        trajectory_axis.scatter(states[0, 6], states[0, 7], label="Trailer start", s=36, color="#ff7f0e", marker="o")
    trajectory_axis.set_title("XY Trajectory")
    trajectory_axis.set_xlabel("X (m)")
    trajectory_axis.set_ylabel("Y (m)")
    trajectory_axis.axis("equal")
    trajectory_axis.grid(True, linestyle="--", alpha=0.35)
    trajectory_axis.legend()

    position_axis = axes[1]
    position_axis.plot(time, states[:, 0], label="Tractor X", linewidth=1.6, color="#1f77b4")
    position_axis.plot(time, states[:, 1], label="Tractor Y", linewidth=1.6, color="#2ca02c")
    if show_trailer:
        position_axis.plot(time, states[:, 6], label="Trailer X", linewidth=1.4, linestyle="--", color="#ff7f0e")
        position_axis.plot(time, states[:, 7], label="Trailer Y", linewidth=1.4, linestyle="--", color="#d62728")
    position_axis.set_title("Position vs Time")
    position_axis.set_xlabel("Time (s)")
    position_axis.set_ylabel("Position (m)")
    position_axis.grid(True, linestyle="--", alpha=0.35)
    position_axis.legend()

    steer_axis = axes[2]
    steer_axis.plot(time, steer_deg, label="Steering wheel angle", linewidth=1.8, color="#9467bd")
    steer_axis.axhline(0.0, linewidth=1.0, linestyle="--", color="black", alpha=0.5)
    steer_axis.set_title("Steering Wheel Angle vs Time")
    steer_axis.set_xlabel("Time (s)")
    steer_axis.set_ylabel("Angle (deg)")
    steer_axis.grid(True, linestyle="--", alpha=0.35)
    steer_axis.legend()

    fig.suptitle(f"Dataset Overview: {segment.segment_name}", fontsize=13)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    print(f"Saved zoomable overview image: {output_path}")

    if no_show:
        plt.close(fig)
    else:
        backend_name = str(plt.get_backend())
        if is_non_interactive_backend(backend_name):
            print(
                "Current Matplotlib backend is non-interactive "
                f"({backend_name}). The figure was saved, but no zoomable window can be shown."
            )
            plt.close(fig)
        else:
            print("Interactive window opened. Use the Matplotlib toolbar to zoom or pan.")
            plt.show()
    return output_path


def main() -> None:
    args = parse_args()
    plt = configure_pyplot(args.no_show)
    runs_root, collect_control_and_trajectory_csvs, load_truck_trailer_data_as_segment, resolve_control_and_trajectory_csv = (
        load_runtime_dependencies()
    )
    csv_path = resolve_single_csv(
        args.input_path,
        runs_root=runs_root,
        collect_control_and_trajectory_csvs=collect_control_and_trajectory_csvs,
        resolve_control_and_trajectory_csv=resolve_control_and_trajectory_csv,
    )
    print(f"Using dataset: {csv_path}")
    segment = load_truck_trailer_data_as_segment(csv_path)
    output_path = args.output_path if args.output_path is not None else default_output_path(segment)
    plot_zoomable_overview(segment, output_path=output_path, no_show=args.no_show, plt=plt)


if __name__ == "__main__":
    main()
