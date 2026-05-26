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

try:
    from .base_model import wrap_angle_error_np
    from .constants import RUNS_ROOT, STATE_NAMES
    from .data_utils import collect_control_and_trajectory_csvs, compute_articulation_series, load_truck_trailer_data_as_segment, save_figure
except ImportError:
    from base_model import wrap_angle_error_np
    from constants import RUNS_ROOT, STATE_NAMES
    from data_utils import collect_control_and_trajectory_csvs, compute_articulation_series, load_truck_trailer_data_as_segment, save_figure


REAL_LABEL = "Real"
INTEGRATED_LABEL = "Velocity Integrated"
POSE_NAMES = ("x_t", "y_t", "psi_t", "x_s", "y_s", "psi_s", "articulation")
VELOCITY_SOURCE_CHOICES = ("next", "current", "midpoint")
DEFAULT_VELOCITY_SOURCE = "midpoint"


@dataclass
class IntegrationSegment:
    csv_path: Path
    scenario_name: str
    segment_name: str
    out_dir: Path
    time: np.ndarray
    dt_values: np.ndarray
    real_rollout: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Integrate vehicle pose from measured velocity channels and compare the reconstructed trajectory against the original data.",
    )
    parser.add_argument(
        "--input-path",
        "--run-dir",
        dest="input_path",
        type=Path,
        default=None,
        help="Optional data root, segment directory, or csv path. Defaults to RUNS_ROOT lookup behavior used elsewhere in the repo.",
    )
    parser.add_argument(
        "--velocity-source",
        choices=VELOCITY_SOURCE_CHOICES,
        default=DEFAULT_VELOCITY_SOURCE,
        help="Which measured velocity sample to use for pose integration. Default is `midpoint`, which uses the average of the current and next measured velocity samples.",
    )
    parser.add_argument(
        "--summary-path",
        type=Path,
        default=None,
        help="Optional summary csv output path. Defaults to a per-input file similar to inference_main.py behavior.",
    )
    return parser.parse_args()


def resolve_output_dir(csv_path: Path) -> Path:
    out_dir = csv_path.parent / f"{csv_path.stem}_vel_integration"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def load_segment(csv_path: Path) -> IntegrationSegment:
    seg = load_truck_trailer_data_as_segment(csv_path)
    return IntegrationSegment(
        csv_path=seg.csv_path,
        scenario_name=seg.segment_name,
        segment_name=csv_path.stem,
        out_dir=resolve_output_dir(csv_path),
        time=seg.time,
        dt_values=seg.dt_values,
        real_rollout=seg.real_rollout,
    )


def _extract_velocity_triplet(
    real_rollout: np.ndarray,
    step: int,
    velocity_source: str,
    velocity_indices: tuple[int, int, int],
) -> tuple[float, float, float]:
    current_values = real_rollout[step, list(velocity_indices)].astype(np.float32)
    next_values = real_rollout[step + 1, list(velocity_indices)].astype(np.float32)

    if velocity_source == "midpoint":
        values = 0.5 * (current_values + next_values)
    elif velocity_source == "current":
        values = current_values
    elif velocity_source == "next":
        values = next_values
    else:
        raise ValueError(f"Unsupported velocity_source={velocity_source!r}")

    return float(values[0]), float(values[1]), float(values[2])


def _integrate_pose_step(
    x_prev: float,
    y_prev: float,
    psi_prev: float,
    vx_used: float,
    vy_used: float,
    r_used: float,
    dt: float,
    velocity_source: str,
) -> tuple[float, float, float]:
    if velocity_source == "midpoint":
        psi_next = wrap_angle_error_np(np.asarray([psi_prev + r_used * dt], dtype=np.float32))[0]
        yaw_for_translation = wrap_angle_error_np(np.asarray([psi_prev + 0.5 * r_used * dt], dtype=np.float32))[0]
    elif velocity_source == "current":
        yaw_for_translation = psi_prev
        psi_next = wrap_angle_error_np(np.asarray([psi_prev + r_used * dt], dtype=np.float32))[0]
    elif velocity_source == "next":
        psi_next = wrap_angle_error_np(np.asarray([psi_prev + r_used * dt], dtype=np.float32))[0]
        yaw_for_translation = psi_next
    else:
        raise ValueError(f"Unsupported velocity_source={velocity_source!r}")

    dx = (np.cos(yaw_for_translation) * vx_used - np.sin(yaw_for_translation) * vy_used) * dt
    dy = (np.sin(yaw_for_translation) * vx_used + np.cos(yaw_for_translation) * vy_used) * dt
    return float(x_prev + dx), float(y_prev + dy), float(psi_next)


def integrate_pose_from_measured_velocity(real_rollout: np.ndarray, dt_values: np.ndarray, velocity_source: str) -> np.ndarray:
    if velocity_source not in VELOCITY_SOURCE_CHOICES:
        raise ValueError(f"velocity_source must be one of {VELOCITY_SOURCE_CHOICES}, got {velocity_source!r}")

    integrated = real_rollout.copy().astype(np.float32)
    integrated[0] = real_rollout[0].astype(np.float32)

    for step, dt_value in enumerate(dt_values):
        dt = float(dt_value)
        prev_state = integrated[step].copy()
        next_measured_state = real_rollout[step + 1].astype(np.float32)

        vx_t, vy_t, r_t = _extract_velocity_triplet(real_rollout, step, velocity_source, (3, 4, 5))
        x_t_next, y_t_next, psi_t_next = _integrate_pose_step(
            x_prev=float(prev_state[0]),
            y_prev=float(prev_state[1]),
            psi_prev=float(prev_state[2]),
            vx_used=vx_t,
            vy_used=vy_t,
            r_used=r_t,
            dt=dt,
            velocity_source=velocity_source,
        )

        vx_s, vy_s, r_s = _extract_velocity_triplet(real_rollout, step, velocity_source, (9, 10, 11))
        x_s_next, y_s_next, psi_s_next = _integrate_pose_step(
            x_prev=float(prev_state[6]),
            y_prev=float(prev_state[7]),
            psi_prev=float(prev_state[8]),
            vx_used=vx_s,
            vy_used=vy_s,
            r_used=r_s,
            dt=dt,
            velocity_source=velocity_source,
        )

        # Recursive rollout uses the previously integrated pose and midpoint
        # measured body-frame velocities for the next step update.
        integrated[step + 1, 0] = x_t_next
        integrated[step + 1, 1] = y_t_next
        integrated[step + 1, 2] = psi_t_next
        integrated[step + 1, 3:6] = next_measured_state[3:6]
        integrated[step + 1, 6] = x_s_next
        integrated[step + 1, 7] = y_s_next
        integrated[step + 1, 8] = psi_s_next
        integrated[step + 1, 9:12] = next_measured_state[9:12]

    return integrated


def compute_pose_error_series(integrated_rollout: np.ndarray, real_rollout: np.ndarray, pose_name: str) -> np.ndarray:
    if pose_name == "articulation":
        return compute_articulation_series(integrated_rollout) - compute_articulation_series(real_rollout)
    state_index = STATE_NAMES.index(pose_name)
    if pose_name in {"psi_t", "psi_s"}:
        return np.rad2deg(wrap_angle_error_np(integrated_rollout[:, state_index] - real_rollout[:, state_index]))
    return integrated_rollout[:, state_index] - real_rollout[:, state_index]


def build_results_dataframe(
    seg: IntegrationSegment,
    integrated_rollout: np.ndarray,
    velocity_source: str,
) -> pd.DataFrame:
    time = seg.time.astype(np.float32)
    dt_series = np.concatenate([seg.dt_values.astype(np.float32), seg.dt_values[-1:].astype(np.float32)], axis=0)
    df = pd.DataFrame(
        {
            "time_s": time,
            "dt_s": dt_series,
            "velocity_source": np.full(len(time), velocity_source, dtype=object),
        }
    )

    for prefix, rollout in (("real", seg.real_rollout), ("integrated", integrated_rollout)):
        for index, name in enumerate(STATE_NAMES):
            values = rollout[:, index]
            if name in {"psi_t", "psi_s"}:
                df[f"{prefix}_{name}_deg"] = np.rad2deg(wrap_angle_error_np(values)).astype(np.float32)
            elif name in {"r_t", "r_s"}:
                df[f"{prefix}_{name}_degps"] = np.rad2deg(values).astype(np.float32)
            else:
                df[f"{prefix}_{name}"] = values.astype(np.float32)
        df[f"{prefix}_articulation_deg"] = compute_articulation_series(rollout).astype(np.float32)

    for pose_name in POSE_NAMES:
        df[f"err_{pose_name}"] = compute_pose_error_series(integrated_rollout, seg.real_rollout, pose_name).astype(np.float32)

    return df


def export_results_csv(seg: IntegrationSegment, integrated_rollout: np.ndarray, velocity_source: str) -> Path:
    out_path = seg.out_dir / "velocity_integration_results.csv"
    build_results_dataframe(seg, integrated_rollout, velocity_source).to_csv(out_path, index=False, encoding="utf-8-sig")
    return out_path


def plot_trajectory_compare(seg: IntegrationSegment, integrated_rollout: np.ndarray, velocity_source: str) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    title_suffix = f"({velocity_source} velocity)"

    axes[0].plot(seg.real_rollout[:, 0], seg.real_rollout[:, 1], label=REAL_LABEL, linewidth=1.9)
    axes[0].plot(integrated_rollout[:, 0], integrated_rollout[:, 1], label=INTEGRATED_LABEL, linewidth=1.7)
    axes[0].set_title(f"Tractor Trajectory {title_suffix}")
    axes[0].set_xlabel("X (m)")
    axes[0].set_ylabel("Y (m)")
    axes[0].grid(True, linestyle="--", alpha=0.35)
    axes[0].set_aspect("equal", adjustable="box")
    axes[0].legend()

    axes[1].plot(seg.real_rollout[:, 6], seg.real_rollout[:, 7], label=REAL_LABEL, linewidth=1.9)
    axes[1].plot(integrated_rollout[:, 6], integrated_rollout[:, 7], label=INTEGRATED_LABEL, linewidth=1.7)
    axes[1].set_title(f"Trailer Trajectory {title_suffix}")
    axes[1].set_xlabel("X (m)")
    axes[1].set_ylabel("Y (m)")
    axes[1].grid(True, linestyle="--", alpha=0.35)
    axes[1].set_aspect("equal", adjustable="box")
    axes[1].legend()

    out_path = seg.out_dir / "trajectory_compare.png"
    save_figure(fig, out_path)
    return out_path


def plot_pose_error(seg: IntegrationSegment, integrated_rollout: np.ndarray, velocity_source: str) -> Path:
    fig, axes = plt.subplots(4, 2, figsize=(16, 14), sharex=True)
    axes = axes.ravel()
    ylabel_map = {
        "x_t": "m",
        "y_t": "m",
        "psi_t": "deg",
        "x_s": "m",
        "y_s": "m",
        "psi_s": "deg",
        "articulation": "deg",
    }

    for axis, pose_name in zip(axes, POSE_NAMES, strict=False):
        error_values = compute_pose_error_series(integrated_rollout, seg.real_rollout, pose_name)
        axis.plot(seg.time, error_values, label=f"{INTEGRATED_LABEL} - {REAL_LABEL}", linewidth=1.5)
        axis.axhline(0.0, color="black", linewidth=0.8, linestyle=":")
        axis.set_title(f"{pose_name} error ({velocity_source} velocity)")
        axis.set_ylabel(ylabel_map[pose_name])
        axis.grid(True, linestyle="--", alpha=0.35)

    for axis in axes[len(POSE_NAMES):]:
        axis.axis("off")

    for axis in axes[-2:]:
        axis.set_xlabel("Time (s)")

    axes[0].legend(loc="best")
    out_path = seg.out_dir / "pose_error_compare.png"
    save_figure(fig, out_path)
    return out_path


def compute_summary(seg: IntegrationSegment, integrated_rollout: np.ndarray, velocity_source: str) -> dict[str, float | str]:
    summary: dict[str, float | str] = {
        "scenario_name": seg.scenario_name,
        "segment_name": seg.segment_name,
        "csv_path": str(seg.csv_path),
        "velocity_source": velocity_source,
        "sample_count": int(len(seg.real_rollout)),
    }
    for pose_name in POSE_NAMES:
        error_values = compute_pose_error_series(integrated_rollout, seg.real_rollout, pose_name).astype(np.float64)
        summary[f"rmse_{pose_name}"] = float(np.sqrt(np.mean(error_values * error_values)))
    return summary


def export_summary_csv(rows: list[dict[str, float | str]], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False, encoding="utf-8-sig")
    return output_path


def main() -> None:
    args = parse_args()
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

    summary_rows: list[dict[str, float | str]] = []
    processed_count = 0

    for csv_path in csvs:
        try:
            seg = load_segment(csv_path)
        except Exception as exc:
            print(f"[Skip] {csv_path}: {exc}")
            continue

        integrated_rollout = integrate_pose_from_measured_velocity(
            real_rollout=seg.real_rollout,
            dt_values=seg.dt_values,
            velocity_source=args.velocity_source,
        )
        result_csv = export_results_csv(seg, integrated_rollout, args.velocity_source)
        trajectory_png = plot_trajectory_compare(seg, integrated_rollout, args.velocity_source)
        error_png = plot_pose_error(seg, integrated_rollout, args.velocity_source)
        summary = compute_summary(seg, integrated_rollout, args.velocity_source)
        summary_rows.append(summary)
        processed_count += 1

        print(f"\n[OK] {seg.segment_name}")
        print(f"  out_dir : {seg.out_dir}")
        print(f"  results : {result_csv.name}")
        print(f"  traj    : {trajectory_png.name}")
        print(f"  errors  : {error_png.name}")
        print(
            "  pose rmse: "
            f"tractor_xy=({summary['rmse_x_t']:.4f}, {summary['rmse_y_t']:.4f}) "
            f"trailer_xy=({summary['rmse_x_s']:.4f}, {summary['rmse_y_s']:.4f}) "
            f"art={summary['rmse_articulation']:.4f} deg"
        )

    if processed_count == 0:
        raise RuntimeError("No segments were processed successfully.")

    if args.summary_path is not None:
        summary_output_path = args.summary_path
    elif args.input_path is not None and len(csvs) == 1:
        summary_output_path = resolve_output_dir(csvs[0]) / "velocity_integration_summary.csv"
    elif args.input_path is not None and Path(args.input_path).is_dir():
        summary_output_path = Path(args.input_path) / "velocity_integration_summary.csv"
    else:
        summary_output_path = RUNS_ROOT / "velocity_integration_summary.csv"

    summary_path = export_summary_csv(summary_rows, summary_output_path)
    print(f"\nsummary csv: {summary_path}")
    print(f"Completed {processed_count} velocity-integration comparisons.")


if __name__ == "__main__":
    main()
