from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

try:
    from .base_model import TruckTrailerNominalDynamics, wrap_angle_error_np, wrap_angle_error_torch
    from .constants import (
        BASE_MODEL_PARAMS,
        CONTROL_NAMES,
        DEFAULT_TRAILER_MASS_KG,
        FIXED_DT_S,
        FORCE_NO_TRAILER_MODE,
        MLP_OUTPUT_NAMES,
        MLP_NUMPY_DTYPE,
        MLP_TORCH_DTYPE,
        MOTION_STATE_NAMES,
        NO_TRAILER_MASS_THRESHOLD_KG,
        POSE_STATE_NAMES,
        ROAD_WHEEL_DEG_CANDIDATES,
        ROAD_WHEEL_RAD_CANDIDATES,
        RUNS_ROOT,
        STATE_LOSS_WEIGHTS,
        STATE_NAMES,
        TURNING_FOCUS_FULL_QUANTILE,
        TURNING_FOCUS_START_QUANTILE,
        TURNING_FOCUS_STEER_THRESHOLD_DEG,
        TURNING_GATE_BASE_WEIGHT,
        TURNING_SAMPLE_WEIGHT_MAX,
        TURNING_SCORE_ARTICULATION_REF_DEG,
        TURNING_SCORE_ARTICULATION_WEIGHT,
        TURNING_SCORE_COMPONENT_CLIP,
        TURNING_SCORE_LATERAL_SPEED_REF_MPS,
        TURNING_SCORE_LATERAL_SPEED_WEIGHT,
        TURNING_SCORE_YAW_RATE_REF_DEGPS,
        TURNING_SCORE_YAW_RATE_WEIGHT,
        STEERING_RATIO_CANDIDATES,
        STEER_SW_DEG_CANDIDATES,
        STEER_SW_RAD_CANDIDATES,
        TORQUE_FL_CANDIDATES,
        TORQUE_FR_CANDIDATES,
        TORQUE_RL_CANDIDATES,
        TORQUE_RR_CANDIDATES,
        TRACTOR_R_DEGPS_CANDIDATES,
        TRACTOR_VX_CANDIDATES,
        TRACTOR_VY_CANDIDATES,
        TRACTOR_X_CANDIDATES,
        TRACTOR_YAW_DEG_CANDIDATES,
        TRACTOR_Y_CANDIDATES,
        TRAILER_MASS_COLUMN_CANDIDATES,
        TRAILER_MASS_PARAMETER_CANDIDATES,
        TRAILER_R_DEGPS_CANDIDATES,
        TRAILER_VX_CANDIDATES,
        TRAILER_VY_CANDIDATES,
        TRAILER_X_CANDIDATES,
        TRAILER_YAW_DEG_CANDIDATES,
        TRAILER_Y_CANDIDATES,
    )
except ImportError:
    from base_model import TruckTrailerNominalDynamics, wrap_angle_error_np, wrap_angle_error_torch
    from constants import (
        BASE_MODEL_PARAMS,
        CONTROL_NAMES,
        DEFAULT_TRAILER_MASS_KG,
        FIXED_DT_S,
        FORCE_NO_TRAILER_MODE,
        MLP_OUTPUT_NAMES,
        MLP_NUMPY_DTYPE,
        MLP_TORCH_DTYPE,
        MOTION_STATE_NAMES,
        NO_TRAILER_MASS_THRESHOLD_KG,
        POSE_STATE_NAMES,
        ROAD_WHEEL_DEG_CANDIDATES,
        ROAD_WHEEL_RAD_CANDIDATES,
        RUNS_ROOT,
        STATE_LOSS_WEIGHTS,
        STATE_NAMES,
        TURNING_FOCUS_FULL_QUANTILE,
        TURNING_FOCUS_START_QUANTILE,
        TURNING_FOCUS_STEER_THRESHOLD_DEG,
        TURNING_GATE_BASE_WEIGHT,
        TURNING_SAMPLE_WEIGHT_MAX,
        TURNING_SCORE_ARTICULATION_REF_DEG,
        TURNING_SCORE_ARTICULATION_WEIGHT,
        TURNING_SCORE_COMPONENT_CLIP,
        TURNING_SCORE_LATERAL_SPEED_REF_MPS,
        TURNING_SCORE_LATERAL_SPEED_WEIGHT,
        TURNING_SCORE_YAW_RATE_REF_DEGPS,
        TURNING_SCORE_YAW_RATE_WEIGHT,
        STEERING_RATIO_CANDIDATES,
        STEER_SW_DEG_CANDIDATES,
        STEER_SW_RAD_CANDIDATES,
        TORQUE_FL_CANDIDATES,
        TORQUE_FR_CANDIDATES,
        TORQUE_RL_CANDIDATES,
        TORQUE_RR_CANDIDATES,
        TRACTOR_R_DEGPS_CANDIDATES,
        TRACTOR_VX_CANDIDATES,
        TRACTOR_VY_CANDIDATES,
        TRACTOR_X_CANDIDATES,
        TRACTOR_YAW_DEG_CANDIDATES,
        TRACTOR_Y_CANDIDATES,
        TRAILER_MASS_COLUMN_CANDIDATES,
        TRAILER_MASS_PARAMETER_CANDIDATES,
        TRAILER_R_DEGPS_CANDIDATES,
        TRAILER_VX_CANDIDATES,
        TRAILER_VY_CANDIDATES,
        TRAILER_X_CANDIDATES,
        TRAILER_YAW_DEG_CANDIDATES,
        TRAILER_Y_CANDIDATES,
    )


@dataclass
class SegmentData:
    csv_path: Path
    segment_name: str
    plot_dir: Path
    time: np.ndarray
    states: np.ndarray
    next_states: np.ndarray
    controls: np.ndarray
    trailer_mass_kg: np.ndarray
    dt_values: np.ndarray
    initial_state: np.ndarray
    real_rollout: np.ndarray
    control_sequence: np.ndarray


def to_tensor(array: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(array, dtype=torch.float32, device=device)


def to_mlp_tensor(array: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(array, dtype=MLP_TORCH_DTYPE, device=device)


def safe_log10(values: np.ndarray, eps: float = 1.0e-12) -> np.ndarray:
    return np.log10(np.clip(values, eps, None))


def save_figure(fig: plt.Figure, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def compute_articulation_series(states: np.ndarray) -> np.ndarray:
    return np.rad2deg(wrap_angle_error_np(states[:, 8] - states[:, 2]))


def smoothstep01_np(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values.astype(np.float32), 0.0, 1.0)
    return (clipped * clipped * (3.0 - 2.0 * clipped)).astype(np.float32)


def compute_turning_focus_score(states: np.ndarray, controls: np.ndarray) -> np.ndarray:
    steer_sw_deg = np.abs(np.rad2deg(controls[:, 0])).astype(np.float32)
    gate_mask = steer_sw_deg > float(TURNING_FOCUS_STEER_THRESHOLD_DEG)

    yaw_rate_degps = np.maximum(
        np.abs(np.rad2deg(states[:, 5])).astype(np.float32),
        np.abs(np.rad2deg(states[:, 11])).astype(np.float32),
    )
    lateral_speed_mps = np.maximum(np.abs(states[:, 4]).astype(np.float32), np.abs(states[:, 10]).astype(np.float32))
    articulation_deg = np.abs(compute_articulation_series(states)).astype(np.float32)

    yaw_rate_component = TURNING_SCORE_YAW_RATE_WEIGHT * np.clip(
        yaw_rate_degps / TURNING_SCORE_YAW_RATE_REF_DEGPS,
        0.0,
        TURNING_SCORE_COMPONENT_CLIP,
    )
    lateral_speed_component = TURNING_SCORE_LATERAL_SPEED_WEIGHT * np.clip(
        lateral_speed_mps / TURNING_SCORE_LATERAL_SPEED_REF_MPS,
        0.0,
        TURNING_SCORE_COMPONENT_CLIP,
    )
    articulation_component = TURNING_SCORE_ARTICULATION_WEIGHT * np.clip(
        articulation_deg / TURNING_SCORE_ARTICULATION_REF_DEG,
        0.0,
        TURNING_SCORE_COMPONENT_CLIP,
    )
    severity_score = np.maximum.reduce(
        [
            yaw_rate_component.astype(np.float32),
            lateral_speed_component.astype(np.float32),
            articulation_component.astype(np.float32),
        ]
    ).astype(np.float32)

    turn_scores = np.full_like(steer_sw_deg, -1.0, dtype=np.float32)
    turn_scores[gate_mask] = severity_score[gate_mask]
    return turn_scores


def fit_turning_focus_context(turn_scores: np.ndarray) -> dict[str, float]:
    scores = np.asarray(turn_scores, dtype=np.float32).reshape(-1)
    gated_scores = scores[scores >= 0.0]
    if gated_scores.size == 0:
        return {
            "threshold_deg": float(TURNING_FOCUS_STEER_THRESHOLD_DEG),
            "gate_base_weight": float(TURNING_GATE_BASE_WEIGHT),
            "start_quantile": float(TURNING_FOCUS_START_QUANTILE),
            "full_quantile": float(TURNING_FOCUS_FULL_QUANTILE),
            "score_start": 0.0,
            "score_full": 1.0,
            "sample_weight_max": float(TURNING_SAMPLE_WEIGHT_MAX),
        }
    if scores.size == 0:
        return {
            "threshold_deg": float(TURNING_FOCUS_STEER_THRESHOLD_DEG),
            "gate_base_weight": float(TURNING_GATE_BASE_WEIGHT),
            "start_quantile": float(TURNING_FOCUS_START_QUANTILE),
            "full_quantile": float(TURNING_FOCUS_FULL_QUANTILE),
            "score_start": 0.0,
            "score_full": 1.0,
            "sample_weight_max": float(TURNING_SAMPLE_WEIGHT_MAX),
        }

    score_start = float(np.quantile(gated_scores, TURNING_FOCUS_START_QUANTILE))
    score_full = float(np.quantile(gated_scores, TURNING_FOCUS_FULL_QUANTILE))
    if not np.isfinite(score_start):
        score_start = 0.0
    if not np.isfinite(score_full):
        score_full = score_start + 1.0
    if score_full <= score_start + 1.0e-6:
        score_full = score_start + 1.0e-3

    return {
        "threshold_deg": float(TURNING_FOCUS_STEER_THRESHOLD_DEG),
        "gate_base_weight": float(TURNING_GATE_BASE_WEIGHT),
        "start_quantile": float(TURNING_FOCUS_START_QUANTILE),
        "full_quantile": float(TURNING_FOCUS_FULL_QUANTILE),
        "score_start": score_start,
        "score_full": score_full,
        "sample_weight_max": float(TURNING_SAMPLE_WEIGHT_MAX),
    }


def compute_turning_focus_mask(turn_scores: np.ndarray, turning_focus_context: dict[str, float]) -> np.ndarray:
    _ = turning_focus_context
    scores = np.asarray(turn_scores, dtype=np.float32).reshape(-1)
    return (scores >= 0.0).astype(np.float32)


def compute_turning_sample_weights(turn_scores: np.ndarray, turning_focus_context: dict[str, float]) -> np.ndarray:
    scores = np.asarray(turn_scores, dtype=np.float32).reshape(-1)
    gate_mask = scores >= 0.0
    score_start = float(turning_focus_context["score_start"])
    score_full = float(turning_focus_context["score_full"])
    gate_base_weight = float(turning_focus_context["gate_base_weight"])
    sample_weight_max = float(turning_focus_context["sample_weight_max"])
    weights = np.ones_like(scores, dtype=np.float32)
    if np.any(gate_mask):
        normalized = (scores[gate_mask] - score_start) / max(score_full - score_start, 1.0e-6)
        focus_strength = smoothstep01_np(normalized)
        weights[gate_mask] = gate_base_weight + (sample_weight_max - gate_base_weight) * focus_strength
    return weights.astype(np.float32)


def describe_turning_focus_context(
    turning_focus_context: dict[str, float],
    train_turn_scores: np.ndarray,
    train_sample_weights: np.ndarray,
    val_turn_scores: np.ndarray | None = None,
    val_sample_weights: np.ndarray | None = None,
) -> None:
    print("Turning-focus weighting:")
    print(
        "  thresholds | "
        f"steer_gate={turning_focus_context['threshold_deg']:.1f} deg "
        f"start_q={turning_focus_context['start_quantile']:.2f} "
        f"full_q={turning_focus_context['full_quantile']:.2f} "
        f"base_w={turning_focus_context['gate_base_weight']:.2f} "
        f"max_w={turning_focus_context['sample_weight_max']:.2f}"
    )

    def _print_split(name: str, scores: np.ndarray, weights: np.ndarray) -> None:
        scores = np.asarray(scores, dtype=np.float32).reshape(-1)
        weights = np.asarray(weights, dtype=np.float32).reshape(-1)
        if scores.size == 0:
            print(f"  {name:<5} | no samples")
            return
        turn_mask = compute_turning_focus_mask(scores, turning_focus_context) > 0.5
        gated_scores = scores[turn_mask] if np.any(turn_mask) else np.array([], dtype=np.float32)
        boosted_ratio = float(np.mean(weights > 1.0 + 1.0e-3))
        strong_ratio = float(np.mean(weights > 2.0))
        turn_ratio = float(np.mean(turn_mask))
        if gated_scores.size > 0:
            q50, q90, q95 = np.quantile(gated_scores, [0.50, 0.90, 0.95]).astype(np.float32)
            score_text = f"sev_q50={q50:.4f} sev_q90={q90:.4f} sev_q95={q95:.4f}"
        else:
            score_text = "sev_q50=NA sev_q90=NA sev_q95=NA"
        print(
            f"  {name:<5} | {score_text} "
            f"| mean_w={float(np.mean(weights)):.3f} max_w={float(np.max(weights)):.3f} "
            f"| steer>5deg={turn_ratio:.1%} boosted={boosted_ratio:.1%} strong={strong_ratio:.1%}"
        )

    _print_split("train", train_turn_scores, train_sample_weights)
    if val_turn_scores is not None and val_sample_weights is not None:
        _print_split("val", val_turn_scores, val_sample_weights)


def compute_relative_pose_np(states: np.ndarray) -> np.ndarray:
    dx = states[:, 6:7] - states[:, 0:1]
    dy = states[:, 7:8] - states[:, 1:2]
    psi_t = states[:, 2:3]
    cos_t = np.cos(psi_t).astype(np.float32)
    sin_t = np.sin(psi_t).astype(np.float32)
    rel_x = cos_t * dx + sin_t * dy
    rel_y = -sin_t * dx + cos_t * dy
    rel_yaw = wrap_angle_error_np((states[:, 8:9] - states[:, 2:3]).reshape(-1)).reshape(-1, 1)
    return np.concatenate([rel_x, rel_y, rel_yaw], axis=1).astype(np.float32)


def relative_pose_to_absolute_np(tractor_pose: np.ndarray, relative_pose: np.ndarray) -> np.ndarray:
    psi_t = tractor_pose[:, 2:3]
    cos_t = np.cos(psi_t).astype(np.float32)
    sin_t = np.sin(psi_t).astype(np.float32)
    rel_x = relative_pose[:, 0:1].astype(np.float32)
    rel_y = relative_pose[:, 1:2].astype(np.float32)
    x_s = tractor_pose[:, 0:1] + cos_t * rel_x - sin_t * rel_y
    y_s = tractor_pose[:, 1:2] + sin_t * rel_x + cos_t * rel_y
    psi_s = wrap_angle_error_np((tractor_pose[:, 2:3] + relative_pose[:, 2:3]).reshape(-1)).reshape(-1, 1)
    return np.concatenate([x_s, y_s, psi_s], axis=1).astype(np.float32)


def compute_relative_pose_torch(states: torch.Tensor) -> torch.Tensor:
    dx = states[:, 6:7] - states[:, 0:1]
    dy = states[:, 7:8] - states[:, 1:2]
    psi_t = states[:, 2:3]
    cos_t = torch.cos(psi_t)
    sin_t = torch.sin(psi_t)
    rel_x = cos_t * dx + sin_t * dy
    rel_y = -sin_t * dx + cos_t * dy
    rel_yaw = wrap_angle_error_torch(states[:, 8:9] - states[:, 2:3])
    return torch.cat([rel_x, rel_y, rel_yaw], dim=1)


def relative_pose_to_absolute_torch(tractor_pose: torch.Tensor, relative_pose: torch.Tensor) -> torch.Tensor:
    psi_t = tractor_pose[:, 2:3]
    cos_t = torch.cos(psi_t)
    sin_t = torch.sin(psi_t)
    rel_x = relative_pose[:, 0:1]
    rel_y = relative_pose[:, 1:2]
    x_s = tractor_pose[:, 0:1] + cos_t * rel_x - sin_t * rel_y
    y_s = tractor_pose[:, 1:2] + sin_t * rel_x + cos_t * rel_y
    psi_s = wrap_angle_error_torch(tractor_pose[:, 2:3] + relative_pose[:, 2:3])
    return torch.cat([x_s, y_s, psi_s], dim=1)


def find_first_existing_column(frame: pd.DataFrame, candidates: list[str]) -> str | None:
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
    lower_map = {column.lower(): column for column in frame.columns}
    for candidate in candidates:
        mapped = lower_map.get(candidate.lower())
        if mapped is not None:
            return mapped
    return None


def require_column(frame: pd.DataFrame, candidates: list[str], field_name: str, csv_path: Path) -> str:
    column = find_first_existing_column(frame, candidates)
    if column is None:
        raise ValueError(f"Missing field `{field_name}` in {csv_path}. Candidate columns: {candidates}")
    return column


def read_column_as_float(frame: pd.DataFrame, candidates: list[str], field_name: str, csv_path: Path) -> np.ndarray:
    column = require_column(frame, candidates, field_name, csv_path)
    return frame[column].to_numpy(dtype=np.float32)


def try_read_column_as_float(frame: pd.DataFrame, candidates: list[str]) -> np.ndarray | None:
    column = find_first_existing_column(frame, candidates)
    if column is None:
        return None
    return frame[column].to_numpy(dtype=np.float32)


def find_all_real_data_csvs(runs_root: Path = RUNS_ROOT) -> list[Path]:
    if not runs_root.exists():
        raise FileNotFoundError(f"runs_root does not exist: {runs_root}")
    return sorted(
        runs_root.glob("python_run_*/outputs/control_and_trajectory.csv"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )


def find_all_real_data_csvs_under(root_dir: Path) -> list[Path]:
    root_dir = Path(root_dir)
    if not root_dir.exists():
        raise FileNotFoundError(f"root_dir does not exist: {root_dir}")
    return sorted(
        root_dir.glob("python_run_*/outputs/control_and_trajectory.csv"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )


def find_all_train_segment_csvs_under(root_dir: Path) -> list[Path]:
    root_dir = Path(root_dir)
    if not root_dir.exists():
        raise FileNotFoundError(f"root_dir does not exist: {root_dir}")
    return sorted(
        root_dir.rglob("*_train_segment_*.csv"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )


def resolve_control_and_trajectory_csv(input_path: Path) -> Path:
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    if input_path.is_file():
        return input_path

    direct_csv = input_path / "control_and_trajectory.csv"
    outputs_csv = input_path / "outputs" / "control_and_trajectory.csv"
    if direct_csv.exists():
        return direct_csv
    if outputs_csv.exists():
        return outputs_csv

    raise FileNotFoundError(
        "Could not locate control_and_trajectory.csv under input path. "
        f"Tried: {direct_csv} and {outputs_csv}"
    )


def collect_control_and_trajectory_csvs(
    input_path: Path | None = None,
    runs_root: Path = RUNS_ROOT,
) -> list[Path]:
    if input_path is None:
        return find_all_real_data_csvs(runs_root)
    input_path = Path(input_path)
    if input_path.is_dir():
        train_segment_csvs = find_all_train_segment_csvs_under(input_path)
        if train_segment_csvs:
            return train_segment_csvs
        nested_csvs = find_all_real_data_csvs_under(input_path)
        if nested_csvs:
            return nested_csvs
    return [resolve_control_and_trajectory_csv(input_path)]


def resolve_plot_dir_for_segment(csv_path: Path) -> Path:
    plot_dir = csv_path.parent / "truck_trailer_residual_plots_modular"
    plot_dir.mkdir(parents=True, exist_ok=True)
    return plot_dir


def resolve_steering_wheel_angle_rad(frame: pd.DataFrame, csv_path: Path) -> np.ndarray:
    # Steering input must come from a steering-wheel-angle column. In the current
    # train_segment data this is `Steer_deg_cmd`; do not fall back to
    # `Target_Steer_L1_deg_cmd`, which is a front-wheel target angle.
    sw_rad_col = find_first_existing_column(frame, STEER_SW_RAD_CANDIDATES)
    if sw_rad_col is not None:
        sw_rad = frame[sw_rad_col].to_numpy(dtype=np.float32)
        if np.isfinite(sw_rad).any():
            return np.where(np.isfinite(sw_rad), sw_rad, 0.0).astype(np.float32)

    sw_deg_col = find_first_existing_column(frame, STEER_SW_DEG_CANDIDATES)
    if sw_deg_col is not None:
        sw_deg = frame[sw_deg_col].to_numpy(dtype=np.float32)
        if np.isfinite(sw_deg).any():
            sw_deg = np.where(np.isfinite(sw_deg), sw_deg, 0.0).astype(np.float32)
            return np.deg2rad(sw_deg).astype(np.float32)

    road_rad_col = find_first_existing_column(frame, ROAD_WHEEL_RAD_CANDIDATES)
    road_deg_col = find_first_existing_column(frame, ROAD_WHEEL_DEG_CANDIDATES)
    ratio_col = find_first_existing_column(frame, STEERING_RATIO_CANDIDATES)
    ratio_values = np.full(len(frame), float(BASE_MODEL_PARAMS["steering_ratio"]), dtype=np.float32)
    if ratio_col is not None:
        raw_ratio = frame[ratio_col].to_numpy(dtype=np.float32)
        valid_ratio = np.isfinite(raw_ratio) & (np.abs(raw_ratio) > 1.0e-6)
        ratio_values = np.where(valid_ratio, raw_ratio, ratio_values).astype(np.float32)

    if road_rad_col is not None:
        road_rad = frame[road_rad_col].to_numpy(dtype=np.float32)
        road_rad = np.where(np.isfinite(road_rad), road_rad, 0.0).astype(np.float32)
        return (road_rad * ratio_values).astype(np.float32)

    if road_deg_col is not None:
        road_deg = frame[road_deg_col].to_numpy(dtype=np.float32)
        road_deg = np.where(np.isfinite(road_deg), road_deg, 0.0).astype(np.float32)
        road_rad = np.deg2rad(road_deg).astype(np.float32)
        return (road_rad * ratio_values).astype(np.float32)

    raise ValueError(f"Could not resolve steering wheel angle in file: {csv_path}")


def load_vehicle_parameter_table(csv_path: Path) -> pd.DataFrame | None:
    candidates = [
        csv_path.parent / "vehicle_parameters.csv",
        csv_path.parents[1] / "outputs" / "vehicle_parameters.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return pd.read_csv(candidate)
    return None


def extract_trailer_mass_from_vehicle_parameters(csv_path: Path) -> float | None:
    table = load_vehicle_parameter_table(csv_path)
    if table is None or "Parameter" not in table.columns or "Value" not in table.columns:
        return None

    parameter_map = {
        str(parameter).strip().lower(): value
        for parameter, value in zip(table["Parameter"], table["Value"], strict=False)
    }
    for candidate in TRAILER_MASS_PARAMETER_CANDIDATES:
        value = parameter_map.get(candidate.lower())
        if value is None:
            continue
        try:
            return float(value)
        except Exception:
            continue
    return None


def resolve_trailer_mass_signal(frame: pd.DataFrame, csv_path: Path) -> np.ndarray:
    mass_col = find_first_existing_column(frame, TRAILER_MASS_COLUMN_CANDIDATES)
    if mass_col is not None:
        return frame[mass_col].to_numpy(dtype=np.float32)

    mass_value = extract_trailer_mass_from_vehicle_parameters(csv_path)
    if mass_value is not None:
        return np.full(len(frame), float(mass_value), dtype=np.float32)

    print(
        f"[Warning] Trailer mass is missing in {csv_path}. Falling back to default {DEFAULT_TRAILER_MASS_KG:.1f} kg."
    )
    return np.full(len(frame), DEFAULT_TRAILER_MASS_KG, dtype=np.float32)


def load_truck_trailer_data_as_segment(csv_path: Path) -> SegmentData:
    dataframe = pd.read_csv(csv_path)
    time_column = require_column(dataframe, ["Time_s", "Time"], "time", csv_path)
    time = dataframe[time_column].to_numpy(dtype=np.float64)
    if len(time) < 2:
        raise ValueError(f"CSV must contain at least two rows: {csv_path}")

    # The current data pipeline assumes a fixed 50 Hz rate. Use fixed dt in the
    # base model and residual post-processing, but keep dt out of the MLP input.
    dt_values = np.full(len(time) - 1, FIXED_DT_S, dtype=np.float32)

    steer_sw_rad = resolve_steering_wheel_angle_rad(dataframe, csv_path)
    torque_fl = read_column_as_float(dataframe, TORQUE_FL_CANDIDATES, "torque_fl", csv_path)
    torque_fr = read_column_as_float(dataframe, TORQUE_FR_CANDIDATES, "torque_fr", csv_path)
    torque_rl = read_column_as_float(dataframe, TORQUE_RL_CANDIDATES, "torque_rl", csv_path)
    torque_rr = read_column_as_float(dataframe, TORQUE_RR_CANDIDATES, "torque_rr", csv_path)

    tractor_x = read_column_as_float(dataframe, TRACTOR_X_CANDIDATES, "tractor_x", csv_path)
    tractor_y = read_column_as_float(dataframe, TRACTOR_Y_CANDIDATES, "tractor_y", csv_path)
    tractor_yaw_deg = read_column_as_float(dataframe, TRACTOR_YAW_DEG_CANDIDATES, "tractor_yaw_deg", csv_path)
    tractor_vx = read_column_as_float(dataframe, TRACTOR_VX_CANDIDATES, "tractor_vx", csv_path)
    tractor_vy = read_column_as_float(dataframe, TRACTOR_VY_CANDIDATES, "tractor_vy", csv_path)
    tractor_r_degps = read_column_as_float(dataframe, TRACTOR_R_DEGPS_CANDIDATES, "tractor_yaw_rate_degps", csv_path)

    trailer_x = try_read_column_as_float(dataframe, TRAILER_X_CANDIDATES)
    trailer_y = try_read_column_as_float(dataframe, TRAILER_Y_CANDIDATES)
    trailer_yaw_deg = try_read_column_as_float(dataframe, TRAILER_YAW_DEG_CANDIDATES)
    trailer_vx = try_read_column_as_float(dataframe, TRAILER_VX_CANDIDATES)
    trailer_vy = try_read_column_as_float(dataframe, TRAILER_VY_CANDIDATES)
    trailer_r_degps = try_read_column_as_float(dataframe, TRAILER_R_DEGPS_CANDIDATES)

    trailer_fields_complete = all(
        value is not None
        for value in (trailer_x, trailer_y, trailer_yaw_deg, trailer_vx, trailer_vy, trailer_r_degps)
    )
    if FORCE_NO_TRAILER_MODE:
        print(f"[Info] FORCE_NO_TRAILER_MODE is enabled. Using tractor states as trailer placeholders for {csv_path}.")
        trailer_x = tractor_x.copy()
        trailer_y = tractor_y.copy()
        trailer_yaw_deg = tractor_yaw_deg.copy()
        trailer_vx = tractor_vx.copy()
        trailer_vy = tractor_vy.copy()
        trailer_r_degps = tractor_r_degps.copy()
        trailer_mass_kg = np.zeros(len(dataframe), dtype=np.float32)
    elif trailer_fields_complete:
        trailer_mass_kg = resolve_trailer_mass_signal(dataframe, csv_path)
    else:
        print(f"[Info] Trailer states are incomplete in {csv_path}. Falling back to no-trailer mode.")
        trailer_x = tractor_x.copy()
        trailer_y = tractor_y.copy()
        trailer_yaw_deg = tractor_yaw_deg.copy()
        trailer_vx = tractor_vx.copy()
        trailer_vy = tractor_vy.copy()
        trailer_r_degps = tractor_r_degps.copy()
        trailer_mass_kg = np.zeros(len(dataframe), dtype=np.float32)

    full_states = np.column_stack(
        [
            tractor_x,
            tractor_y,
            wrap_angle_error_np(np.deg2rad(tractor_yaw_deg)),
            tractor_vx,
            tractor_vy,
            np.deg2rad(tractor_r_degps).astype(np.float32),
            trailer_x,
            trailer_y,
            wrap_angle_error_np(np.deg2rad(trailer_yaw_deg)),
            trailer_vx,
            trailer_vy,
            np.deg2rad(trailer_r_degps).astype(np.float32),
        ]
    ).astype(np.float32)
    full_controls = np.column_stack([steer_sw_rad, torque_fl, torque_fr, torque_rl, torque_rr]).astype(np.float32)

    segment_name = csv_path.parents[1].name
    plot_dir = resolve_plot_dir_for_segment(csv_path)
    return SegmentData(
        csv_path=csv_path,
        segment_name=segment_name,
        plot_dir=plot_dir,
        time=time.astype(np.float32),
        states=full_states[:-1].copy(),
        next_states=full_states[1:].copy(),
        controls=full_controls[:-1].copy(),
        trailer_mass_kg=trailer_mass_kg[:-1].copy(),
        dt_values=dt_values.copy(),
        initial_state=full_states[0].copy(),
        real_rollout=full_states.copy(),
        control_sequence=full_controls[:-1].copy(),
    )


def build_mlp_state_features_np(states: np.ndarray, trailer_mass_kg: np.ndarray) -> np.ndarray:
    # Use translation-invariant state. Absolute x/y never enters the MLP; when a
    # trailer is present its pose is represented relative to the tractor frame.
    trailer_mass_kg = trailer_mass_kg.reshape(-1).astype(np.float32)
    has_trailer = (trailer_mass_kg > NO_TRAILER_MASS_THRESHOLD_KG).astype(np.float32)
    relative_pose = compute_relative_pose_np(states)
    return np.column_stack(
        [
            trailer_mass_kg,
            has_trailer,
            states[:, 3],
            states[:, 4],
            states[:, 5],
            states[:, 9],
            states[:, 10],
            states[:, 11],
            relative_pose[:, 0],
            relative_pose[:, 1],
            np.sin(relative_pose[:, 2]).astype(np.float32),
            np.cos(relative_pose[:, 2]).astype(np.float32),
        ]
    ).astype(np.float32)


def build_mlp_control_features_np(controls: np.ndarray) -> np.ndarray:
    rear_drive_torque_sum = controls[:, 3] + controls[:, 4]
    return np.column_stack([controls[:, 0], rear_drive_torque_sum]).astype(np.float32)


def build_mlp_input_feature_tensor(
    state: torch.Tensor,
    control: torch.Tensor,
    trailer_mass_kg: torch.Tensor,
    dt: torch.Tensor,
) -> torch.Tensor:
    if trailer_mass_kg.ndim == 1:
        trailer_mass_kg = trailer_mass_kg.unsqueeze(1)
    # dt is fixed at FIXED_DT_S and is therefore used outside the MLP only.
    _ = dt

    has_trailer = (trailer_mass_kg > NO_TRAILER_MASS_THRESHOLD_KG).to(dtype=MLP_TORCH_DTYPE)
    relative_pose = compute_relative_pose_torch(state)
    rear_drive_torque_sum = control[:, 3:4] + control[:, 4:5]
    return torch.cat(
        [
            trailer_mass_kg.to(dtype=MLP_TORCH_DTYPE),
            has_trailer,
            state[:, 3:4].to(dtype=MLP_TORCH_DTYPE),
            state[:, 4:5].to(dtype=MLP_TORCH_DTYPE),
            state[:, 5:6].to(dtype=MLP_TORCH_DTYPE),
            state[:, 9:10].to(dtype=MLP_TORCH_DTYPE),
            state[:, 10:11].to(dtype=MLP_TORCH_DTYPE),
            state[:, 11:12].to(dtype=MLP_TORCH_DTYPE),
            relative_pose[:, 0:1].to(dtype=MLP_TORCH_DTYPE),
            relative_pose[:, 1:2].to(dtype=MLP_TORCH_DTYPE),
            torch.sin(relative_pose[:, 2:3]).to(dtype=MLP_TORCH_DTYPE),
            torch.cos(relative_pose[:, 2:3]).to(dtype=MLP_TORCH_DTYPE),
            control[:, 0:1].to(dtype=MLP_TORCH_DTYPE),
            rear_drive_torque_sum.to(dtype=MLP_TORCH_DTYPE),
        ],
        dim=1,
    )


def build_training_features(
    states: np.ndarray,
    controls: np.ndarray,
    trailer_mass_kg: np.ndarray,
    dt_values: np.ndarray,
) -> np.ndarray:
    state_features = build_mlp_state_features_np(states, trailer_mass_kg)
    control_features = build_mlp_control_features_np(controls)
    # dt_values stays in the function signature to make the training data path
    # explicit, but the fixed 0.02 s step is not part of the MLP feature vector.
    _ = dt_values
    return np.concatenate([state_features, control_features], axis=1).astype(np.float32)


def build_feature_context(features: np.ndarray) -> dict[str, np.ndarray]:
    feature_mean = np.mean(features, axis=0, dtype=MLP_NUMPY_DTYPE).astype(MLP_NUMPY_DTYPE)
    feature_std = np.std(features, axis=0, dtype=MLP_NUMPY_DTYPE).astype(MLP_NUMPY_DTYPE)
    feature_scale = np.where(feature_std > 1.0e-6, feature_std, 1.0).astype(MLP_NUMPY_DTYPE)
    return {"feature_mean": feature_mean, "feature_scale": feature_scale}


def normalize_features_np(features: np.ndarray, feature_context: dict[str, np.ndarray]) -> np.ndarray:
    feature_mean = feature_context["feature_mean"].reshape(1, -1)
    feature_scale = feature_context["feature_scale"].reshape(1, -1)
    return ((features - feature_mean) / feature_scale).astype(MLP_NUMPY_DTYPE)


def build_feature_context_tensors(feature_context: dict[str, np.ndarray], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "feature_mean": to_mlp_tensor(feature_context["feature_mean"].reshape(1, -1), device),
        "feature_scale": to_mlp_tensor(feature_context["feature_scale"].reshape(1, -1), device),
    }


def normalize_feature_tensor(features: torch.Tensor, feature_context_tensors: dict[str, torch.Tensor]) -> torch.Tensor:
    return (features - feature_context_tensors["feature_mean"]) / feature_context_tensors["feature_scale"]


def derive_full_error_from_mlp_output_np(
    mlp_output: np.ndarray,
    base_next: np.ndarray,
    dt_values: np.ndarray,
    trailer_mass_kg: np.ndarray,
) -> np.ndarray:
    if mlp_output.shape[1] != len(MLP_OUTPUT_NAMES):
        raise ValueError(
            f"mlp_output has {mlp_output.shape[1]} columns, expected {len(MLP_OUTPUT_NAMES)}: {MLP_OUTPUT_NAMES}"
        )
    safe_dt = np.clip(dt_values.reshape(-1, 1).astype(np.float32), 1.0e-6, None)
    has_trailer = (trailer_mass_kg.reshape(-1, 1).astype(np.float32) > NO_TRAILER_MASS_THRESHOLD_KG).astype(np.float32)
    yaw_t = base_next[:, 2:3].astype(np.float32)

    evx_t = mlp_output[:, 0:1].astype(np.float32)
    evy_t = mlp_output[:, 1:2].astype(np.float32)
    er_t = mlp_output[:, 2:3].astype(np.float32)
    evx_s = mlp_output[:, 3:4].astype(np.float32)
    evy_s = mlp_output[:, 4:5].astype(np.float32)
    er_s = mlp_output[:, 5:6].astype(np.float32)

    dx_t = (np.cos(yaw_t) * evx_t - np.sin(yaw_t) * evy_t) * safe_dt
    dy_t = (np.sin(yaw_t) * evx_t + np.cos(yaw_t) * evy_t) * safe_dt
    dpsi_t = wrap_angle_error_np((er_t * safe_dt).reshape(-1)).reshape(-1, 1)

    tractor_pose_corrected = base_next[:, 0:3].astype(np.float32) + np.concatenate([dx_t, dy_t, dpsi_t], axis=1)
    tractor_pose_corrected[:, 2] = wrap_angle_error_np(tractor_pose_corrected[:, 2])

    base_relative_pose = compute_relative_pose_np(base_next)
    rel_delta = mlp_output[:, 6:9].astype(np.float32)
    corrected_relative_pose = base_relative_pose + rel_delta
    corrected_relative_pose[:, 2] = wrap_angle_error_np(corrected_relative_pose[:, 2])
    trailer_pose_corrected = relative_pose_to_absolute_np(tractor_pose_corrected, corrected_relative_pose)
    trailer_pose_error = trailer_pose_corrected - base_next[:, 6:9].astype(np.float32)
    trailer_pose_error[:, 2] = wrap_angle_error_np(trailer_pose_error[:, 2])

    no_trailer_pose_error = np.concatenate([dx_t, dy_t, dpsi_t], axis=1)
    trailer_pose_error = has_trailer * trailer_pose_error + (1.0 - has_trailer) * no_trailer_pose_error
    trailer_motion_error = has_trailer * np.concatenate([evx_s, evy_s, er_s], axis=1) + (1.0 - has_trailer) * np.concatenate(
        [evx_t, evy_t, er_t],
        axis=1,
    )

    return np.concatenate(
        [dx_t, dy_t, dpsi_t, evx_t, evy_t, er_t, trailer_pose_error, trailer_motion_error],
        axis=1,
    ).astype(np.float32)


def derive_full_error_from_mlp_output_torch(
    mlp_output: torch.Tensor,
    base_next: torch.Tensor,
    dt_values: torch.Tensor,
    trailer_mass_kg: torch.Tensor,
) -> torch.Tensor:
    if mlp_output.shape[1] != len(MLP_OUTPUT_NAMES):
        raise ValueError(
            f"mlp_output has {mlp_output.shape[1]} columns, expected {len(MLP_OUTPUT_NAMES)}: {MLP_OUTPUT_NAMES}"
        )
    if dt_values.ndim == 1:
        dt_values = dt_values.unsqueeze(1)
    if trailer_mass_kg.ndim == 1:
        trailer_mass_kg = trailer_mass_kg.unsqueeze(1)
    safe_dt = torch.clamp(dt_values, min=1.0e-6)
    has_trailer = (trailer_mass_kg > NO_TRAILER_MASS_THRESHOLD_KG).to(dtype=mlp_output.dtype, device=mlp_output.device)
    yaw_t = base_next[:, 2:3]

    evx_t = mlp_output[:, 0:1]
    evy_t = mlp_output[:, 1:2]
    er_t = mlp_output[:, 2:3]
    evx_s = mlp_output[:, 3:4]
    evy_s = mlp_output[:, 4:5]
    er_s = mlp_output[:, 5:6]

    dx_t = (torch.cos(yaw_t) * evx_t - torch.sin(yaw_t) * evy_t) * safe_dt
    dy_t = (torch.sin(yaw_t) * evx_t + torch.cos(yaw_t) * evy_t) * safe_dt
    dpsi_t = wrap_angle_error_torch(er_t * safe_dt)

    tractor_pose_corrected = base_next[:, 0:3] + torch.cat([dx_t, dy_t, dpsi_t], dim=1)
    tractor_pose_corrected = tractor_pose_corrected.clone()
    tractor_pose_corrected[:, 2:3] = wrap_angle_error_torch(tractor_pose_corrected[:, 2:3])

    base_relative_pose = compute_relative_pose_torch(base_next)
    rel_delta = mlp_output[:, 6:9]
    corrected_relative_pose = base_relative_pose + rel_delta
    corrected_relative_pose = corrected_relative_pose.clone()
    corrected_relative_pose[:, 2:3] = wrap_angle_error_torch(corrected_relative_pose[:, 2:3])
    trailer_pose_corrected = relative_pose_to_absolute_torch(tractor_pose_corrected, corrected_relative_pose)
    trailer_pose_error = trailer_pose_corrected - base_next[:, 6:9]
    trailer_pose_error = trailer_pose_error.clone()
    trailer_pose_error[:, 2:3] = wrap_angle_error_torch(trailer_pose_error[:, 2:3])

    no_trailer_pose_error = torch.cat([dx_t, dy_t, dpsi_t], dim=1)
    trailer_pose_error = has_trailer * trailer_pose_error + (1.0 - has_trailer) * no_trailer_pose_error
    trailer_motion_error = has_trailer * torch.cat([evx_s, evy_s, er_s], dim=1) + (1.0 - has_trailer) * torch.cat(
        [evx_t, evy_t, er_t],
        dim=1,
    )

    return torch.cat([dx_t, dy_t, dpsi_t, evx_t, evy_t, er_t, trailer_pose_error, trailer_motion_error], dim=1)


def build_loss_context(true_error: np.ndarray, true_mlp_output: np.ndarray, device: torch.device) -> dict[str, torch.Tensor]:
    error_std = np.std(true_error, axis=0).astype(np.float32)
    min_scale = np.array(
        [
            0.001,
            0.001,
            np.deg2rad(0.01),
            0.001,
            0.001,
            np.deg2rad(0.01),
            0.001,
            0.001,
            np.deg2rad(0.01),
            0.001,
            0.001,
            np.deg2rad(0.01),
        ],
        dtype=np.float32,
    )
    error_scale = np.maximum(error_std, min_scale).astype(np.float32)
    pose_indices = [STATE_NAMES.index(name) for name in POSE_STATE_NAMES]
    motion_indices = [STATE_NAMES.index(name) for name in MOTION_STATE_NAMES]
    pose_error_scale = error_scale[pose_indices].astype(np.float32)
    motion_error_scale = error_scale[motion_indices].astype(np.float32)
    output_min_scale = np.array(
        [
            0.001,
            0.001,
            np.deg2rad(0.01),
            0.001,
            0.001,
            np.deg2rad(0.01),
            0.001,
            0.001,
            np.deg2rad(0.01),
        ],
        dtype=np.float32,
    )
    output_scale = np.maximum(np.std(true_mlp_output, axis=0).astype(np.float32), output_min_scale).astype(np.float32)
    channel_weight = np.array([STATE_LOSS_WEIGHTS[name] for name in STATE_NAMES], dtype=np.float32)
    output_weight = np.array([1.0, 1.0, 5.0, 1.0, 1.0, 5.0, 1.0, 1.0, 2.0], dtype=np.float32)
    return {
        "error_scale": to_mlp_tensor(error_scale.reshape(1, -1), device),
        "pose_error_scale": to_mlp_tensor(pose_error_scale.reshape(1, -1), device),
        "motion_error_scale": to_mlp_tensor(motion_error_scale.reshape(1, -1), device),
        "output_scale": to_mlp_tensor(output_scale.reshape(1, -1), device),
        "channel_weight": to_mlp_tensor(channel_weight.reshape(1, -1), device),
        "output_weight": to_mlp_tensor(output_weight.reshape(1, -1), device),
    }


def describe_loss_context(loss_context: dict[str, torch.Tensor]) -> None:
    error_scale = loss_context["error_scale"].detach().cpu().numpy().ravel()
    output_scale = loss_context["output_scale"].detach().cpu().numpy().ravel()
    print("Per-state error scale:")
    for index, name in enumerate(STATE_NAMES):
        print(f"  {name:<6} weight={STATE_LOSS_WEIGHTS[name]:.3f} error_scale={error_scale[index]:.6f}")
    print("Per-MLP-output scale:")
    for index, name in enumerate(MLP_OUTPUT_NAMES):
        print(f"  {name:<12} error_scale={output_scale[index]:.6f}")


@torch.no_grad()
def compute_base_next_states(
    base_model: TruckTrailerNominalDynamics,
    states: np.ndarray,
    controls: np.ndarray,
    trailer_mass_kg: np.ndarray,
    dt_values: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    state_tensor = to_tensor(states, device)
    control_tensor = to_tensor(controls, device)
    mass_tensor = to_tensor(trailer_mass_kg.reshape(-1, 1), device)
    dt_tensor = to_tensor(dt_values.reshape(-1, 1), device)
    return base_model(state_tensor, control_tensor, mass_tensor, dt_tensor).cpu().numpy().astype(np.float32)


def build_train_val_by_segments(
    segments: list[SegmentData],
    val_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[list[SegmentData], list[SegmentData]]:
    if len(segments) == 1:
        return split_single_segment_for_train_val(segments[0], val_ratio=val_ratio)
    if len(segments) < 2:
        raise ValueError("At least two valid segments are required for a train/validation split.")
    indices = np.arange(len(segments))
    val_count = max(1, int(round(len(segments) * val_ratio)))
    val_count = min(val_count, len(segments) - 1)
    train_idx, val_idx = train_test_split(indices, test_size=val_count, random_state=seed, shuffle=True)
    return [segments[i] for i in train_idx], [segments[i] for i in val_idx]


def slice_segment(seg: SegmentData, start_step: int, end_step: int, suffix: str) -> SegmentData:
    total_steps = int(seg.states.shape[0])
    if not (0 <= start_step < end_step <= total_steps):
        raise ValueError(f"Invalid segment slice [{start_step}, {end_step}) for total_steps={total_steps}")

    sliced_plot_dir = seg.plot_dir / suffix
    sliced_plot_dir.mkdir(parents=True, exist_ok=True)
    return SegmentData(
        csv_path=seg.csv_path,
        segment_name=f"{seg.segment_name}_{suffix}",
        plot_dir=sliced_plot_dir,
        time=seg.time[start_step : end_step + 1].copy(),
        states=seg.states[start_step:end_step].copy(),
        next_states=seg.next_states[start_step:end_step].copy(),
        controls=seg.controls[start_step:end_step].copy(),
        trailer_mass_kg=seg.trailer_mass_kg[start_step:end_step].copy(),
        dt_values=seg.dt_values[start_step:end_step].copy(),
        initial_state=seg.real_rollout[start_step].copy(),
        real_rollout=seg.real_rollout[start_step : end_step + 1].copy(),
        control_sequence=seg.control_sequence[start_step:end_step].copy(),
    )


def split_single_segment_for_train_val(
    segment: SegmentData,
    val_ratio: float = 0.2,
    min_steps_per_split: int = 50,
) -> tuple[list[SegmentData], list[SegmentData]]:
    total_steps = int(segment.states.shape[0])
    if total_steps < max(2 * min_steps_per_split, 10):
        raise ValueError(
            f"Single segment is too short for train/validation split: total_steps={total_steps}, "
            f"min_steps_per_split={min_steps_per_split}"
        )

    val_steps = max(min_steps_per_split, int(round(total_steps * val_ratio)))
    val_steps = min(val_steps, total_steps - min_steps_per_split)
    split_step = total_steps - val_steps

    train_segment = slice_segment(segment, 0, split_step, "train_slice")
    val_segment = slice_segment(segment, split_step, total_steps, "val_slice")
    return [train_segment], [val_segment]


def concat_segments_for_training(
    base_model: TruckTrailerNominalDynamics,
    segments: list[SegmentData],
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    all_features: list[np.ndarray] = []
    all_true_mlp_output: list[np.ndarray] = []
    all_true_error: list[np.ndarray] = []
    all_base_next: list[np.ndarray] = []
    all_dt_values: list[np.ndarray] = []
    all_trailer_mass_kg: list[np.ndarray] = []
    all_turn_scores: list[np.ndarray] = []

    for seg in segments:
        features = build_training_features(seg.states, seg.controls, seg.trailer_mass_kg, seg.dt_values)
        base_next = compute_base_next_states(
            base_model=base_model,
            states=seg.states,
            controls=seg.controls,
            trailer_mass_kg=seg.trailer_mass_kg,
            dt_values=seg.dt_values,
            device=device,
        )
        has_trailer = (seg.trailer_mass_kg.reshape(-1, 1) > NO_TRAILER_MASS_THRESHOLD_KG).astype(np.float32)
        true_motion_error = np.concatenate(
            [
                seg.next_states[:, 3:6] - base_next[:, 3:6],
                seg.next_states[:, 9:12] - base_next[:, 9:12],
            ],
            axis=1,
        ).astype(np.float32)
        true_motion_error[:, 3:6] = has_trailer * true_motion_error[:, 3:6]

        true_relative_pose = compute_relative_pose_np(seg.next_states)
        base_relative_pose = compute_relative_pose_np(base_next)
        true_relative_error = true_relative_pose - base_relative_pose
        true_relative_error[:, 2] = wrap_angle_error_np(true_relative_error[:, 2])
        true_relative_error = (has_trailer * true_relative_error).astype(np.float32)
        true_mlp_output = np.concatenate([true_motion_error, true_relative_error], axis=1).astype(np.float32)

        true_error = (seg.next_states - base_next).astype(np.float32)
        true_error[:, 2] = wrap_angle_error_np(true_error[:, 2])
        true_error[:, 8] = wrap_angle_error_np(true_error[:, 8])

        all_features.append(features)
        all_true_mlp_output.append(true_mlp_output)
        all_true_error.append(true_error)
        all_base_next.append(base_next)
        all_dt_values.append(seg.dt_values.reshape(-1, 1).astype(np.float32))
        all_trailer_mass_kg.append(seg.trailer_mass_kg.reshape(-1, 1).astype(np.float32))
        all_turn_scores.append(compute_turning_focus_score(seg.states, seg.controls).astype(np.float32))

    return (
        np.concatenate(all_features, axis=0),
        np.concatenate(all_true_mlp_output, axis=0),
        np.concatenate(all_true_error, axis=0),
        np.concatenate(all_base_next, axis=0),
        np.concatenate(all_dt_values, axis=0),
        np.concatenate(all_trailer_mass_kg, axis=0),
        np.concatenate(all_turn_scores, axis=0),
    )
