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
    from .base_model import TruckTrailerNominalDynamics, wrap_angle_error_np
    from .constants import MLP_OUTPUT_NAMES, RUNS_ROOT, STATE_NAMES
    from .data_utils import (
        build_feature_context_tensors,
        build_mlp_input_feature_tensor,
        collect_control_and_trajectory_csvs,
        derive_full_error_from_mlp_output_np,
        load_truck_trailer_data_as_segment,
        normalize_feature_tensor,
        save_figure,
        to_tensor,
    )
    from .inference_main import (
        BASE_LABEL,
        NN_LABEL,
        REAL_LABEL,
        InferenceSegment,
        build_base_model,
        compute_state_error_series,
        extract_feature_context,
        extract_output_clip,
        load_error_model,
    )
except ImportError:
    from base_model import TruckTrailerNominalDynamics, wrap_angle_error_np
    from constants import MLP_OUTPUT_NAMES, RUNS_ROOT, STATE_NAMES
    from data_utils import (
        build_feature_context_tensors,
        build_mlp_input_feature_tensor,
        collect_control_and_trajectory_csvs,
        derive_full_error_from_mlp_output_np,
        load_truck_trailer_data_as_segment,
        normalize_feature_tensor,
        save_figure,
        to_tensor,
    )
    from inference_main import (
        BASE_LABEL,
        NN_LABEL,
        REAL_LABEL,
        InferenceSegment,
        build_base_model,
        compute_state_error_series,
        extract_feature_context,
        extract_output_clip,
        load_error_model,
    )


VELOCITY_STATE_NAMES = ("vx_t", "vy_t", "r_t", "vx_s", "vy_s", "r_s")
VELOCITY_STATE_INDICES = [STATE_NAMES.index(name) for name in VELOCITY_STATE_NAMES]
MOTION_OUTPUT_NAMES = tuple(MLP_OUTPUT_NAMES[:6])
MOTION_OUTPUT_PLOT_META: dict[str, tuple[str, str]] = {
    "vx_t": ("Tractor Vx", "m/s"),
    "vy_t": ("Tractor Vy", "m/s"),
    "r_t": ("Tractor Yaw Rate", "deg/s"),
    "vx_s": ("Trailer Vx", "m/s"),
    "vy_s": ("Trailer Vy", "m/s"),
    "r_s": ("Trailer Yaw Rate", "deg/s"),
}


@dataclass
class RecursiveVelocityTrace:
    base_rollout: np.ndarray
    nn_branch_base_rollout: np.ndarray
    nn_rollout: np.ndarray
    mlp_motion_output_trace: np.ndarray
    applied_velocity_delta_trace: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run open-loop recursive rollout and export/store base-model velocities together with MLP motion outputs.",
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
        help="Optional summary csv path. Defaults to a per-run file for single-input mode or a global file otherwise.",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=None,
        help="Optional checkpoint .pth path, run directory, or checkpoint directory.",
    )
    return parser.parse_args()


def resolve_output_dir(csv_path: Path) -> Path:
    out_dir = csv_path.parent / f"{csv_path.stem}_recursive_velocity_trace_eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def save_trace_figure(fig: plt.Figure, output_path: Path, top_margin: float = 1.0) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, top_margin))
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def pad_series(values: np.ndarray) -> np.ndarray:
    if len(values) == 0:
        return np.zeros((1,), dtype=np.float32)
    return np.concatenate([values.astype(np.float32), values[-1:].astype(np.float32)], axis=0)


def load_segment(csv_path: Path) -> InferenceSegment:
    seg = load_truck_trailer_data_as_segment(csv_path)
    return InferenceSegment(
        csv_path=seg.csv_path,
        scenario_name=seg.segment_name,
        segment_name=csv_path.stem,
        out_dir=resolve_output_dir(csv_path),
        time=seg.time,
        dt_values=seg.dt_values,
        real_rollout=seg.real_rollout,
        initial_state=seg.initial_state,
        control_sequence=seg.control_sequence,
        trailer_mass_kg=seg.trailer_mass_kg,
    )


@torch.no_grad()
def predict_base_next_from_state(
    base_model: TruckTrailerNominalDynamics,
    current_state: np.ndarray,
    control_step: np.ndarray,
    trailer_mass_kg_step: float,
    dt_value: float,
    device: torch.device,
) -> np.ndarray:
    dt = np.array([[dt_value]], dtype=np.float32)
    control = control_step.reshape(1, -1).astype(np.float32)
    state = current_state.reshape(1, -1).astype(np.float32)
    mass = np.array([[trailer_mass_kg_step]], dtype=np.float32)

    state_tensor = to_tensor(state, device)
    control_tensor = to_tensor(control, device)
    mass_tensor = to_tensor(mass, device)
    dt_tensor = to_tensor(dt, device)

    base_next = base_model(state_tensor, control_tensor, mass_tensor, dt_tensor).cpu().numpy().astype(np.float32)
    if not np.isfinite(base_next).all():
        raise FloatingPointError("Base inference produced a non-finite state.")
    return base_next[0]


@torch.no_grad()
def predict_nn_next_with_velocity_trace(
    base_model: TruckTrailerNominalDynamics,
    error_model: torch.nn.Module,
    current_state: np.ndarray,
    control_step: np.ndarray,
    trailer_mass_kg_step: float,
    dt_value: float,
    device: torch.device,
    feature_context_tensors: dict[str, torch.Tensor] | None,
    mlp_output_clip: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    dt = np.array([[dt_value]], dtype=np.float32)
    control = control_step.reshape(1, -1).astype(np.float32)
    state = current_state.reshape(1, -1).astype(np.float32)
    mass = np.array([[trailer_mass_kg_step]], dtype=np.float32)

    state_tensor = to_tensor(state, device)
    control_tensor = to_tensor(control, device)
    mass_tensor = to_tensor(mass, device)
    dt_tensor = to_tensor(dt, device)

    base_next = base_model(state_tensor, control_tensor, mass_tensor, dt_tensor).cpu().numpy().astype(np.float32)
    if not np.isfinite(base_next).all():
        raise FloatingPointError("Base inference produced a non-finite state.")

    features = build_mlp_input_feature_tensor(state_tensor, control_tensor, mass_tensor, dt_tensor)
    if feature_context_tensors is not None:
        features = normalize_feature_tensor(features, feature_context_tensors)
    predicted_mlp_output = error_model(features).cpu().numpy()[0].astype(np.float32)
    if mlp_output_clip is not None:
        predicted_mlp_output = np.clip(predicted_mlp_output, -mlp_output_clip, mlp_output_clip)
    corrected_error = derive_full_error_from_mlp_output_np(
        predicted_mlp_output.reshape(1, -1),
        base_next,
        np.array([dt_value], dtype=np.float32),
        np.array([trailer_mass_kg_step], dtype=np.float32),
    )[0]

    if not np.isfinite(corrected_error).all():
        corrected_error = np.zeros((len(STATE_NAMES),), dtype=np.float32)
    nn_next = base_next[0] + corrected_error
    nn_next[2] = wrap_angle_error_np(np.asarray([nn_next[2]], dtype=np.float32))[0]
    nn_next[8] = wrap_angle_error_np(np.asarray([nn_next[8]], dtype=np.float32))[0]
    if not np.isfinite(nn_next).all():
        nn_next = base_next[0].copy()
        corrected_error = np.zeros((len(STATE_NAMES),), dtype=np.float32)

    applied_velocity_delta = corrected_error[VELOCITY_STATE_INDICES].astype(np.float32)
    return base_next[0], nn_next.astype(np.float32), predicted_mlp_output[:6].astype(np.float32), applied_velocity_delta


def rollout_recursive_with_velocity_trace(
    base_model: TruckTrailerNominalDynamics,
    error_model: torch.nn.Module,
    initial_state: np.ndarray,
    control_sequence: np.ndarray,
    trailer_mass_kg: np.ndarray,
    dt_values: np.ndarray,
    device: torch.device,
    feature_context_tensors: dict[str, torch.Tensor] | None,
    mlp_output_clip: np.ndarray | None,
) -> RecursiveVelocityTrace:
    step_count = len(control_sequence) + 1
    base_rollout = np.zeros((step_count, len(STATE_NAMES)), dtype=np.float32)
    nn_branch_base_rollout = np.zeros((step_count, len(STATE_NAMES)), dtype=np.float32)
    nn_rollout = np.zeros((step_count, len(STATE_NAMES)), dtype=np.float32)
    mlp_motion_output_trace = np.full((step_count, len(MOTION_OUTPUT_NAMES)), np.nan, dtype=np.float32)
    applied_velocity_delta_trace = np.full((step_count, len(MOTION_OUTPUT_NAMES)), np.nan, dtype=np.float32)

    base_rollout[0] = initial_state.astype(np.float32)
    nn_branch_base_rollout[0] = initial_state.astype(np.float32)
    nn_rollout[0] = initial_state.astype(np.float32)

    for step in range(len(control_sequence)):
        base_rollout[step + 1] = predict_base_next_from_state(
            base_model=base_model,
            current_state=base_rollout[step],
            control_step=control_sequence[step],
            trailer_mass_kg_step=float(trailer_mass_kg[step]),
            dt_value=float(dt_values[step]),
            device=device,
        )

        nn_branch_base_next, nn_next, mlp_motion_output, applied_velocity_delta = predict_nn_next_with_velocity_trace(
            base_model=base_model,
            error_model=error_model,
            current_state=nn_rollout[step],
            control_step=control_sequence[step],
            trailer_mass_kg_step=float(trailer_mass_kg[step]),
            dt_value=float(dt_values[step]),
            device=device,
            feature_context_tensors=feature_context_tensors,
            mlp_output_clip=mlp_output_clip,
        )
        nn_branch_base_rollout[step + 1] = nn_branch_base_next
        nn_rollout[step + 1] = nn_next
        mlp_motion_output_trace[step + 1] = mlp_motion_output
        applied_velocity_delta_trace[step + 1] = applied_velocity_delta

    return RecursiveVelocityTrace(
        base_rollout=base_rollout,
        nn_branch_base_rollout=nn_branch_base_rollout,
        nn_rollout=nn_rollout,
        mlp_motion_output_trace=mlp_motion_output_trace,
        applied_velocity_delta_trace=applied_velocity_delta_trace,
    )


def convert_motion_series_for_export(values: np.ndarray, name: str) -> np.ndarray:
    if name in {"r_t", "r_s"}:
        return np.rad2deg(values.astype(np.float32))
    return values.astype(np.float32)


def build_results_dataframe(seg: InferenceSegment, trace: RecursiveVelocityTrace) -> pd.DataFrame:
    time = seg.time.astype(np.float32)
    dt_series = pad_series(seg.dt_values)
    steer_sw_deg = np.rad2deg(pad_series(seg.control_sequence[:, 0]))
    trailer_mass = pad_series(seg.trailer_mass_kg)

    df = pd.DataFrame(
        {
            "time_s": time,
            "dt_s": dt_series,
            "steer_sw_deg": steer_sw_deg,
            "trailer_mass_kg": trailer_mass,
        }
    )

    for name, state_index in zip(VELOCITY_STATE_NAMES, VELOCITY_STATE_INDICES, strict=False):
        real_values = seg.real_rollout[:, state_index]
        base_values = trace.base_rollout[:, state_index]
        nn_branch_base_values = trace.nn_branch_base_rollout[:, state_index]
        nn_values = trace.nn_rollout[:, state_index]
        if name in {"r_t", "r_s"}:
            df[f"real_{name}_degps"] = np.rad2deg(real_values).astype(np.float32)
            df[f"base_recursive_{name}_degps"] = np.rad2deg(base_values).astype(np.float32)
            df[f"nn_branch_base_{name}_degps"] = np.rad2deg(nn_branch_base_values).astype(np.float32)
            df[f"nn_recursive_{name}_degps"] = np.rad2deg(nn_values).astype(np.float32)
        else:
            df[f"real_{name}"] = real_values.astype(np.float32)
            df[f"base_recursive_{name}"] = base_values.astype(np.float32)
            df[f"nn_branch_base_{name}"] = nn_branch_base_values.astype(np.float32)
            df[f"nn_recursive_{name}"] = nn_values.astype(np.float32)

    for output_index, name in enumerate(MOTION_OUTPUT_NAMES):
        mlp_values = trace.mlp_motion_output_trace[:, output_index]
        applied_values = trace.applied_velocity_delta_trace[:, output_index]
        if name in {"r_t", "r_s"}:
            df[f"mlp_raw_{name}_degps"] = np.rad2deg(mlp_values).astype(np.float32)
            df[f"mlp_applied_delta_{name}_degps"] = np.rad2deg(applied_values).astype(np.float32)
        else:
            df[f"mlp_raw_{name}"] = mlp_values.astype(np.float32)
            df[f"mlp_applied_delta_{name}"] = applied_values.astype(np.float32)

    for name in VELOCITY_STATE_NAMES:
        df[f"err_base_recursive_{name}"] = compute_state_error_series(trace.base_rollout, seg.real_rollout, name).astype(np.float32)
        df[f"err_nn_branch_base_{name}"] = compute_state_error_series(trace.nn_branch_base_rollout, seg.real_rollout, name).astype(np.float32)
        df[f"err_nn_recursive_{name}"] = compute_state_error_series(trace.nn_rollout, seg.real_rollout, name).astype(np.float32)

    return df


def export_results_csv(seg: InferenceSegment, trace: RecursiveVelocityTrace) -> Path:
    output_path = seg.out_dir / "recursive_velocity_trace_results.csv"
    build_results_dataframe(seg, trace).to_csv(output_path, index=False, encoding="utf-8-sig")
    return output_path


def plot_recursive_velocity_comparison(seg: InferenceSegment, trace: RecursiveVelocityTrace) -> Path:
    fig, axes = plt.subplots(3, 2, figsize=(16, 12), sharex=True)
    axes = axes.ravel()

    for axis, name, state_index in zip(axes, VELOCITY_STATE_NAMES, VELOCITY_STATE_INDICES, strict=False):
        title, ylabel = MOTION_OUTPUT_PLOT_META[name]
        real_values = seg.real_rollout[:, state_index]
        base_values = trace.base_rollout[:, state_index]
        nn_values = trace.nn_rollout[:, state_index]
        if name in {"r_t", "r_s"}:
            real_values = np.rad2deg(real_values)
            base_values = np.rad2deg(base_values)
            nn_values = np.rad2deg(nn_values)

        axis.plot(seg.time, real_values, label=REAL_LABEL, linewidth=1.8)
        axis.plot(seg.time, base_values, label="Recursive Base", linewidth=1.5)
        axis.plot(seg.time, nn_values, label=NN_LABEL, linewidth=1.6)
        axis.set_title(f"Recursive {title}")
        axis.set_ylabel(ylabel)
        axis.grid(True, linestyle="--", alpha=0.35)

    for axis in axes[-2:]:
        axis.set_xlabel("Time (s)")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.02))

    output_path = seg.out_dir / "recursive_velocity_comparison.png"
    save_trace_figure(fig, output_path, top_margin=0.93)
    return output_path


def plot_nn_branch_velocity_components(seg: InferenceSegment, trace: RecursiveVelocityTrace) -> Path:
    fig, axes = plt.subplots(3, 2, figsize=(16, 12), sharex=True)
    axes = axes.ravel()

    for axis, name, state_index in zip(axes, VELOCITY_STATE_NAMES, VELOCITY_STATE_INDICES, strict=False):
        title, ylabel = MOTION_OUTPUT_PLOT_META[name]
        real_values = seg.real_rollout[:, state_index]
        base_values = trace.nn_branch_base_rollout[:, state_index]
        nn_values = trace.nn_rollout[:, state_index]
        if name in {"r_t", "r_s"}:
            real_values = np.rad2deg(real_values)
            base_values = np.rad2deg(base_values)
            nn_values = np.rad2deg(nn_values)

        axis.plot(seg.time, real_values, label=REAL_LABEL, linewidth=1.8)
        axis.plot(seg.time, base_values, label="NN-Branch Base", linewidth=1.5, linestyle="--")
        axis.plot(seg.time, nn_values, label=NN_LABEL, linewidth=1.6)
        axis.set_title(f"NN Branch {title}")
        axis.set_ylabel(ylabel)
        axis.grid(True, linestyle="--", alpha=0.35)

    for axis in axes[-2:]:
        axis.set_xlabel("Time (s)")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.02))

    output_path = seg.out_dir / "recursive_nn_branch_velocity_components.png"
    save_trace_figure(fig, output_path, top_margin=0.93)
    return output_path


def plot_mlp_motion_outputs(seg: InferenceSegment, trace: RecursiveVelocityTrace) -> Path:
    fig, axes = plt.subplots(3, 2, figsize=(16, 12), sharex=True)
    axes = axes.ravel()

    for axis, output_index, name in zip(axes, range(len(MOTION_OUTPUT_NAMES)), MOTION_OUTPUT_NAMES, strict=False):
        title, ylabel = MOTION_OUTPUT_PLOT_META[name]
        raw_values = trace.mlp_motion_output_trace[:, output_index]
        applied_values = trace.applied_velocity_delta_trace[:, output_index]
        if name in {"r_t", "r_s"}:
            raw_values = np.rad2deg(raw_values)
            applied_values = np.rad2deg(applied_values)

        axis.plot(seg.time, raw_values, label="Raw MLP Output", linewidth=1.5)
        axis.plot(seg.time, applied_values, label="Applied Delta", linewidth=1.5, linestyle="--")
        axis.axhline(0.0, color="black", linewidth=0.8, linestyle=":")
        axis.set_title(f"MLP Motion Output {title}")
        axis.set_ylabel(ylabel)
        axis.grid(True, linestyle="--", alpha=0.35)

    for axis in axes[-2:]:
        axis.set_xlabel("Time (s)")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.02))

    output_path = seg.out_dir / "recursive_mlp_motion_outputs.png"
    save_trace_figure(fig, output_path, top_margin=0.93)
    return output_path


def compute_summary(seg: InferenceSegment, trace: RecursiveVelocityTrace) -> dict[str, float | str]:
    summary: dict[str, float | str] = {
        "scenario_name": seg.scenario_name,
        "segment_name": seg.segment_name,
        "csv_path": str(seg.csv_path),
        "output_dir": str(seg.out_dir),
        "sample_count": int(len(seg.real_rollout)),
        "mean_trailer_mass_kg": float(np.mean(seg.trailer_mass_kg)),
    }

    for name, output_index in zip(VELOCITY_STATE_NAMES, range(len(MOTION_OUTPUT_NAMES)), strict=False):
        err_base = compute_state_error_series(trace.base_rollout, seg.real_rollout, name).astype(np.float64)
        err_nn_branch_base = compute_state_error_series(trace.nn_branch_base_rollout, seg.real_rollout, name).astype(np.float64)
        err_nn = compute_state_error_series(trace.nn_rollout, seg.real_rollout, name).astype(np.float64)
        summary[f"recursive_rmse_base_{name}"] = float(np.sqrt(np.mean(err_base * err_base)))
        summary[f"recursive_rmse_nn_branch_base_{name}"] = float(np.sqrt(np.mean(err_nn_branch_base * err_nn_branch_base)))
        summary[f"recursive_rmse_nn_{name}"] = float(np.sqrt(np.mean(err_nn * err_nn)))

        raw_values = convert_motion_series_for_export(trace.mlp_motion_output_trace[:, output_index], name).astype(np.float64)
        applied_values = convert_motion_series_for_export(trace.applied_velocity_delta_trace[:, output_index], name).astype(np.float64)
        summary[f"mean_abs_mlp_raw_{name}"] = float(np.nanmean(np.abs(raw_values)))
        summary[f"mean_abs_mlp_applied_delta_{name}"] = float(np.nanmean(np.abs(applied_values)))

    return summary


def export_summary_csv(rows: list[dict[str, float | str]], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False, encoding="utf-8-sig")
    return output_path


def main() -> None:
    args = parse_args()

    torch.manual_seed(42)
    np.random.seed(42)

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
    mlp_output_clip = extract_output_clip(checkpoint_metadata)
    feature_context_tensors = None
    if feature_context is not None:
        feature_context_tensors = build_feature_context_tensors(feature_context, device)
    else:
        print("Checkpoint does not contain feature normalization statistics; raw features will be used.")
    if mlp_output_clip is None:
        print("Checkpoint does not contain output scaling; residual clipping is disabled.")

    summary_rows: list[dict[str, float | str]] = []
    processed_count = 0

    for csv_path in csvs:
        try:
            seg = load_segment(csv_path)
        except Exception as exc:
            print(f"[Skip] {csv_path}: {exc}")
            continue

        trace = rollout_recursive_with_velocity_trace(
            base_model=base_model,
            error_model=error_model,
            initial_state=seg.initial_state,
            control_sequence=seg.control_sequence,
            trailer_mass_kg=seg.trailer_mass_kg,
            dt_values=seg.dt_values,
            device=device,
            feature_context_tensors=feature_context_tensors,
            mlp_output_clip=mlp_output_clip,
        )
        results_csv = export_results_csv(seg, trace)
        recursive_velocity_png = plot_recursive_velocity_comparison(seg, trace)
        nn_branch_velocity_png = plot_nn_branch_velocity_components(seg, trace)
        mlp_output_png = plot_mlp_motion_outputs(seg, trace)
        summary = compute_summary(seg, trace)
        segment_summary_csv = export_summary_csv([summary], seg.out_dir / "recursive_velocity_trace_summary.csv")
        summary_rows.append(summary)
        processed_count += 1

        print(f"\n[OK] {seg.segment_name}")
        print(f"  out_dir : {seg.out_dir}")
        print(f"  results : {results_csv.name}")
        print(f"  abs_vel : {recursive_velocity_png.name}")
        print(f"  nn_base : {nn_branch_velocity_png.name}")
        print(f"  mlp_out : {mlp_output_png.name}")
        print(f"  summary : {segment_summary_csv.name}")
        print(
            "  recursive tractor vx rmse: "
            f"base={summary['recursive_rmse_base_vx_t']:.4f}, "
            f"nn_branch_base={summary['recursive_rmse_nn_branch_base_vx_t']:.4f}, "
            f"nn={summary['recursive_rmse_nn_vx_t']:.4f}"
        )

    if processed_count == 0:
        raise RuntimeError("No segments were processed successfully.")

    if args.summary_path is not None:
        summary_output_path = args.summary_path
    elif args.input_path is not None and len(csvs) == 1:
        summary_output_path = resolve_output_dir(csvs[0]) / "truck_trailer_recursive_velocity_trace_summary.csv"
    elif args.input_path is not None and Path(args.input_path).is_dir():
        summary_output_path = Path(args.input_path) / "truck_trailer_recursive_velocity_trace_summary.csv"
    else:
        summary_output_path = RUNS_ROOT / "truck_trailer_recursive_velocity_trace_summary.csv"

    summary_path = export_summary_csv(summary_rows, summary_output_path)
    print(f"\nsummary csv: {summary_path}")
    print(f"Completed {processed_count} recursive velocity-trace runs.")


if __name__ == "__main__":
    main()
