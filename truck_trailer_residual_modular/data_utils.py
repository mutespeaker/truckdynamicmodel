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
        MLP_NUMPY_DTYPE,
        MLP_STATE_FEATURE_NAMES,
        MLP_TORCH_DTYPE,
        NO_TRAILER_MASS_THRESHOLD_KG,
        ROAD_WHEEL_DEG_CANDIDATES,
        ROAD_WHEEL_RAD_CANDIDATES,
        RUNS_ROOT,
        STATE_LOSS_WEIGHTS,
        STATE_NAMES,
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
        MLP_NUMPY_DTYPE,
        MLP_STATE_FEATURE_NAMES,
        MLP_TORCH_DTYPE,
        NO_TRAILER_MASS_THRESHOLD_KG,
        ROAD_WHEEL_DEG_CANDIDATES,
        ROAD_WHEEL_RAD_CANDIDATES,
        RUNS_ROOT,
        STATE_LOSS_WEIGHTS,
        STATE_NAMES,
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
        nested_csvs = find_all_real_data_csvs_under(input_path)
        if nested_csvs:
            return nested_csvs
    return [resolve_control_and_trajectory_csv(input_path)]


def resolve_plot_dir_for_segment(csv_path: Path) -> Path:
    plot_dir = csv_path.parent / "truck_trailer_residual_plots_modular"
    plot_dir.mkdir(parents=True, exist_ok=True)
    return plot_dir


def resolve_steering_wheel_angle_rad(frame: pd.DataFrame, csv_path: Path) -> np.ndarray:
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

    dt_values = np.diff(time)
    positive_dt = dt_values[dt_values > 0.0]
    nominal_dt = float(np.median(positive_dt)) if positive_dt.size else 0.02
    dt_values = np.where(dt_values > 0.0, dt_values, nominal_dt).astype(np.float32)

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
    if trailer_fields_complete:
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
    articulation = wrap_angle_error_np(states[:, 8] - states[:, 2])
    speed_t = np.sqrt(states[:, 3] ** 2 + states[:, 4] ** 2).astype(np.float32)
    speed_s = np.sqrt(states[:, 9] ** 2 + states[:, 10] ** 2).astype(np.float32)
    return np.column_stack(
        [
            trailer_mass_kg.astype(np.float32),
            states[:, 3],
            states[:, 4],
            states[:, 5],
            speed_t,
            states[:, 9],
            states[:, 10],
            states[:, 11],
            speed_s,
            articulation,
            np.sin(articulation).astype(np.float32),
            np.cos(articulation).astype(np.float32),
        ]
    ).astype(np.float32)


def build_mlp_input_feature_tensor(
    state: torch.Tensor,
    control: torch.Tensor,
    trailer_mass_kg: torch.Tensor,
    dt: torch.Tensor,
) -> torch.Tensor:
    if trailer_mass_kg.ndim == 1:
        trailer_mass_kg = trailer_mass_kg.unsqueeze(1)
    if dt.ndim == 1:
        dt = dt.unsqueeze(1)

    articulation = wrap_angle_error_torch(state[:, 8:9] - state[:, 2:3])
    speed_t = torch.sqrt(state[:, 3:4] * state[:, 3:4] + state[:, 4:5] * state[:, 4:5] + 1.0e-8)
    speed_s = torch.sqrt(state[:, 9:10] * state[:, 9:10] + state[:, 10:11] * state[:, 10:11] + 1.0e-8)
    return torch.cat(
        [
            trailer_mass_kg.to(dtype=MLP_TORCH_DTYPE),
            state[:, 3:4].to(dtype=MLP_TORCH_DTYPE),
            state[:, 4:5].to(dtype=MLP_TORCH_DTYPE),
            state[:, 5:6].to(dtype=MLP_TORCH_DTYPE),
            speed_t.to(dtype=MLP_TORCH_DTYPE),
            state[:, 9:10].to(dtype=MLP_TORCH_DTYPE),
            state[:, 10:11].to(dtype=MLP_TORCH_DTYPE),
            state[:, 11:12].to(dtype=MLP_TORCH_DTYPE),
            speed_s.to(dtype=MLP_TORCH_DTYPE),
            articulation.to(dtype=MLP_TORCH_DTYPE),
            torch.sin(articulation).to(dtype=MLP_TORCH_DTYPE),
            torch.cos(articulation).to(dtype=MLP_TORCH_DTYPE),
            control.to(dtype=MLP_TORCH_DTYPE),
            dt.to(dtype=MLP_TORCH_DTYPE),
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
    return np.concatenate(
        [state_features, controls.astype(np.float32), dt_values.reshape(-1, 1).astype(np.float32)],
        axis=1,
    ).astype(np.float32)


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


def derive_full_error_from_motion_error_np(
    motion_error: np.ndarray,
    base_next: np.ndarray,
    dt_values: np.ndarray,
) -> np.ndarray:
    safe_dt = np.clip(dt_values.reshape(-1, 1).astype(np.float32), 1.0e-6, None)
    yaw_t = base_next[:, 2:3].astype(np.float32)
    yaw_s = base_next[:, 8:9].astype(np.float32)

    evx_t = motion_error[:, 0:1].astype(np.float32)
    evy_t = motion_error[:, 1:2].astype(np.float32)
    er_t = motion_error[:, 2:3].astype(np.float32)
    evx_s = motion_error[:, 3:4].astype(np.float32)
    evy_s = motion_error[:, 4:5].astype(np.float32)
    er_s = motion_error[:, 5:6].astype(np.float32)

    dx_t = (np.cos(yaw_t) * evx_t - np.sin(yaw_t) * evy_t) * safe_dt
    dy_t = (np.sin(yaw_t) * evx_t + np.cos(yaw_t) * evy_t) * safe_dt
    dpsi_t = wrap_angle_error_np((er_t * safe_dt).reshape(-1)).reshape(-1, 1)

    dx_s = (np.cos(yaw_s) * evx_s - np.sin(yaw_s) * evy_s) * safe_dt
    dy_s = (np.sin(yaw_s) * evx_s + np.cos(yaw_s) * evy_s) * safe_dt
    dpsi_s = wrap_angle_error_np((er_s * safe_dt).reshape(-1)).reshape(-1, 1)

    return np.concatenate(
        [dx_t, dy_t, dpsi_t, evx_t, evy_t, er_t, dx_s, dy_s, dpsi_s, evx_s, evy_s, er_s],
        axis=1,
    ).astype(np.float32)


def derive_full_error_from_motion_error_torch(
    motion_error: torch.Tensor,
    base_next: torch.Tensor,
    dt_values: torch.Tensor,
) -> torch.Tensor:
    if dt_values.ndim == 1:
        dt_values = dt_values.unsqueeze(1)
    safe_dt = torch.clamp(dt_values, min=1.0e-6)
    yaw_t = base_next[:, 2:3]
    yaw_s = base_next[:, 8:9]

    evx_t = motion_error[:, 0:1]
    evy_t = motion_error[:, 1:2]
    er_t = motion_error[:, 2:3]
    evx_s = motion_error[:, 3:4]
    evy_s = motion_error[:, 4:5]
    er_s = motion_error[:, 5:6]

    dx_t = (torch.cos(yaw_t) * evx_t - torch.sin(yaw_t) * evy_t) * safe_dt
    dy_t = (torch.sin(yaw_t) * evx_t + torch.cos(yaw_t) * evy_t) * safe_dt
    dpsi_t = wrap_angle_error_torch(er_t * safe_dt)

    dx_s = (torch.cos(yaw_s) * evx_s - torch.sin(yaw_s) * evy_s) * safe_dt
    dy_s = (torch.sin(yaw_s) * evx_s + torch.cos(yaw_s) * evy_s) * safe_dt
    dpsi_s = wrap_angle_error_torch(er_s * safe_dt)

    return torch.cat([dx_t, dy_t, dpsi_t, evx_t, evy_t, er_t, dx_s, dy_s, dpsi_s, evx_s, evy_s, er_s], dim=1)


def build_loss_context(true_error: np.ndarray, device: torch.device) -> dict[str, torch.Tensor]:
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
    pose_error_scale = np.concatenate([error_scale[:3], error_scale[6:9]], axis=0).astype(np.float32)
    motion_error_scale = np.concatenate([error_scale[3:6], error_scale[9:12]], axis=0).astype(np.float32)
    channel_weight = np.array([STATE_LOSS_WEIGHTS[name] for name in STATE_NAMES], dtype=np.float32)
    return {
        "error_scale": to_mlp_tensor(error_scale.reshape(1, -1), device),
        "pose_error_scale": to_mlp_tensor(pose_error_scale.reshape(1, -1), device),
        "motion_error_scale": to_mlp_tensor(motion_error_scale.reshape(1, -1), device),
        "channel_weight": to_mlp_tensor(channel_weight.reshape(1, -1), device),
    }


def describe_loss_context(loss_context: dict[str, torch.Tensor]) -> None:
    error_scale = loss_context["error_scale"].detach().cpu().numpy().ravel()
    print("Per-state error scale:")
    for index, name in enumerate(STATE_NAMES):
        print(f"  {name:<6} weight={STATE_LOSS_WEIGHTS[name]:.3f} error_scale={error_scale[index]:.6f}")


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
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    all_features: list[np.ndarray] = []
    all_true_error: list[np.ndarray] = []
    all_base_next: list[np.ndarray] = []

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
        true_motion_error = np.concatenate(
            [
                seg.next_states[:, 3:6] - base_next[:, 3:6],
                seg.next_states[:, 9:12] - base_next[:, 9:12],
            ],
            axis=1,
        ).astype(np.float32)
        true_error = derive_full_error_from_motion_error_np(true_motion_error, base_next, seg.dt_values)

        all_features.append(features)
        all_true_error.append(true_error)
        all_base_next.append(base_next)

    return (
        np.concatenate(all_features, axis=0),
        np.concatenate(all_true_error, axis=0),
        np.concatenate(all_base_next, axis=0),
    )
