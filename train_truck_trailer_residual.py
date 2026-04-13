from __future__ import annotations

"""
卡车-挂车残差训练脚本。

整体思路沿用 `train.py`，但把名义模型换成 `dynamic_truck_m.py`
风格的牵引车 + 半挂车平面动力学，并把挂车质量显式纳入输入。
"""

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset


RUNS_ROOT = Path(r"D:\test_torch project\controltest\carsim_runs")
MODEL_CHECKPOINT = Path(__file__).with_name("best_truck_trailer_error_model.pth")
TRAIN_LOSS_MODEL_CHECKPOINT = Path(__file__).with_name("best_truck_trailer_error_model_train_loss.pth")

STATE_NAMES = [
    "x_t",
    "y_t",
    "psi_t",
    "vx_t",
    "vy_t",
    "r_t",
    "x_s",
    "y_s",
    "psi_s",
    "vx_s",
    "vy_s",
    "r_s",
]
POSE_STATE_NAMES = ["x_t", "y_t", "psi_t", "x_s", "y_s", "psi_s"]
MOTION_ERROR_NAMES = ["vx_t", "vy_t", "r_t", "vx_s", "vy_s", "r_s"]
CONTROL_NAMES = ["delta_f_rad", "torque_fl", "torque_fr", "torque_rl", "torque_rr"]
VELOCITY_STATE_INDICES = [3, 4, 5, 9, 10, 11]

MLP_STATE_FEATURE_NAMES = [
    "trailer_mass_kg",
    "vx_t",
    "vy_t",
    "r_t",
    "speed_t",
    "vx_s",
    "vy_s",
    "r_s",
    "speed_s",
    "articulation_rad",
    "sin_articulation",
    "cos_articulation",
]
MLP_INPUT_FEATURE_NAMES = MLP_STATE_FEATURE_NAMES + CONTROL_NAMES + ["dt"]

BASE_MODEL_PARAMS = {
    "m_t": 9300.0,
    "Iz_t": 48639.0,
    "L_t": 4.475,
    "a_t": 3.8,
    "m_s_base": 15004.0,
    "Iz_s_base": 96659.0,
    "L_s": 8.0,
    "c_s": 4.0,
    "Cf": 80000.0,
    "Cr": 80000.0,
    "Cs": 80000.0,
    "wheel_radius": 0.5,
    "track_width": 1.8,
    "steering_ratio": 16.39,
    "rho": 1.225,
    "CdA_t": 5.82,
    "CdA_s": 6.50,
    "rolling_coeff": 0.006,
    "hitch_x": -0.331,
    "hitch_y": 0.002,
    "min_speed_mps": 0.5,
}

STATE_LOSS_WEIGHTS = {
    "x_t": 1.0,
    "y_t": 1.0,
    "psi_t": 2.0,
    "vx_t": 1.0,
    "vy_t": 1.0,
    "r_t": 5.0,
    "x_s": 1.0,
    "y_s": 1.0,
    "psi_s": 2.0,
    "vx_s": 1.0,
    "vy_s": 1.0,
    "r_s": 5.0,
}

TRAIN_BATCH_SIZE = 2048
TRAIN_NUM_WORKERS = 0
TRAIN_EPOCHS = 4000
LEARNING_RATE = 3.0e-3
MOTION_ONLY_WARMUP_EPOCHS = 600
POSE_RAMP_EPOCHS = 300
GRADIENT_CLIP_NORM = 200.0
MLP_TORCH_DTYPE = torch.float32
MLP_NUMPY_DTYPE = np.float32
MLP_USE_LAYER_NORM = True
DEFAULT_TRAILER_MASS_KG = float(BASE_MODEL_PARAMS["m_s_base"])
NO_TRAILER_MASS_THRESHOLD_KG = 1.0

TRACTOR_X_CANDIDATES = ["X_t_m", "Tractor_X_m", "X_tractor_m", "TractorX_m", "X1_m", "X_m"]
TRACTOR_Y_CANDIDATES = ["Y_t_m", "Tractor_Y_m", "Y_tractor_m", "TractorY_m", "Y1_m", "Y_m"]
TRACTOR_YAW_DEG_CANDIDATES = ["Yaw_t_deg", "Tractor_Yaw_deg", "Yaw_tractor_deg", "Yaw1_deg", "Yaw_deg"]
TRACTOR_VX_CANDIDATES = ["Vx_t_mps", "Tractor_Vx_mps", "Vx_tractor_mps", "VxTractor_mps", "Vx_mps"]
TRACTOR_VY_CANDIDATES = ["Vy_t_mps", "Tractor_Vy_mps", "Vy_tractor_mps", "VyTractor_mps", "Vy_mps"]
TRACTOR_R_DEGPS_CANDIDATES = ["YawRate_t_degps", "Tractor_YawRate_degps", "YawRate_tractor_degps", "YawRate1_degps", "YawRate_degps"]

TRAILER_X_CANDIDATES = ["X_s_m", "Trailer_X_m", "X_trailer_m", "TrailerX_m", "X2_m"]
TRAILER_Y_CANDIDATES = ["Y_s_m", "Trailer_Y_m", "Y_trailer_m", "TrailerY_m", "Y2_m"]
TRAILER_YAW_DEG_CANDIDATES = ["Yaw_s_deg", "Trailer_Yaw_deg", "Yaw_trailer_deg", "Yaw2_deg"]
TRAILER_VX_CANDIDATES = ["Vx_s_mps", "Trailer_Vx_mps", "Vx_trailer_mps", "TrailerSpeedX_mps"]
TRAILER_VY_CANDIDATES = ["Vy_s_mps", "Trailer_Vy_mps", "Vy_trailer_mps", "TrailerSpeedY_mps"]
TRAILER_R_DEGPS_CANDIDATES = ["YawRate_s_degps", "Trailer_YawRate_degps", "YawRate_trailer_degps", "YawRate2_degps"]

STEER_SW_RAD_CANDIDATES = [
    "Steer_SW_rad",
    "SteeringWheel_rad",
    "Steering_Wheel_rad",
    "Steering_Wheel_Angle_rad",
    "steering_wheel_rad",
]
STEER_SW_DEG_CANDIDATES = [
    "Steer_SW_deg",
    "Steer_SW",
    "SteeringWheel_deg",
    "Steering_Wheel_deg",
    "Steering_Wheel_Angle_deg",
]
ROAD_WHEEL_RAD_CANDIDATES = ["Steer_L1_rad", "SteerRoadWheel_rad", "delta_f_rad"]
ROAD_WHEEL_DEG_CANDIDATES = ["Steer_deg_cmd", "Steer_L1", "Target_Steer_L1_deg_cmd", "FrontWheelAngle_deg"]
STEERING_RATIO_CANDIDATES = ["Steering_Ratio_SW_to_L1", "SteeringRatio", "Steer_Ratio_SW_to_L1"]

TORQUE_FL_CANDIDATES = ["Torque_FL_Nm_cmd", "Torque_L1_Nm_cmd", "torque_fl_nm", "wheel_torque_fl_nm"]
TORQUE_FR_CANDIDATES = ["Torque_FR_Nm_cmd", "Torque_R1_Nm_cmd", "torque_fr_nm", "wheel_torque_fr_nm"]
TORQUE_RL_CANDIDATES = ["Torque_RL_Nm_cmd", "Torque_L2_Nm_cmd", "torque_rl_nm", "wheel_torque_rl_nm"]
TORQUE_RR_CANDIDATES = ["Torque_RR_Nm_cmd", "Torque_R2_Nm_cmd", "torque_rr_nm", "wheel_torque_rr_nm"]

TRAILER_MASS_COLUMN_CANDIDATES = ["TrailerMass_kg", "Trailer_Mass_kg", "m_s_kg", "m_trailer_kg"]
TRAILER_MASS_PARAMETER_CANDIDATES = [
    "M_S",
    "M_ST",
    "M_TRAILER",
    "M_SEMI_TRAILER",
    "M_SEMITRAILER",
    "TRAILER_MASS_KG",
]


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


class TruckTrailerNominalDynamics(nn.Module):
    """名义牵引车-挂车动力学模型。"""

    def __init__(self, params: dict[str, float]) -> None:
        super().__init__()
        self.register_buffer("m_t", torch.tensor(float(params["m_t"])))
        self.register_buffer("Iz_t", torch.tensor(float(params["Iz_t"])))
        self.register_buffer("L_t", torch.tensor(float(params["L_t"])))
        self.register_buffer("a_t", torch.tensor(float(params["a_t"])))
        self.register_buffer("m_s_base", torch.tensor(float(params["m_s_base"])))
        self.register_buffer("Iz_s_base", torch.tensor(float(params["Iz_s_base"])))
        self.register_buffer("L_s", torch.tensor(float(params["L_s"])))
        self.register_buffer("c_s", torch.tensor(float(params["c_s"])))
        self.register_buffer("Cf", torch.tensor(float(params["Cf"])))
        self.register_buffer("Cr", torch.tensor(float(params["Cr"])))
        self.register_buffer("Cs", torch.tensor(float(params["Cs"])))
        self.register_buffer("wheel_radius", torch.tensor(float(params["wheel_radius"])))
        self.register_buffer("track_width", torch.tensor(float(params["track_width"])))
        self.register_buffer("steering_ratio", torch.tensor(float(params["steering_ratio"])))
        self.register_buffer("rho", torch.tensor(float(params["rho"])))
        self.register_buffer("CdA_t", torch.tensor(float(params["CdA_t"])))
        self.register_buffer("CdA_s", torch.tensor(float(params["CdA_s"])))
        self.register_buffer("rolling_coeff", torch.tensor(float(params["rolling_coeff"])))
        self.register_buffer("hitch_x", torch.tensor(float(params["hitch_x"])))
        self.register_buffer("hitch_y", torch.tensor(float(params["hitch_y"])))
        self.register_buffer("min_speed_mps", torch.tensor(float(params["min_speed_mps"])))
        self.register_buffer("g", torch.tensor(9.81))
        self.register_buffer("_eps", torch.tensor(1.0e-8))
        self.no_trailer_mass_threshold_kg = NO_TRAILER_MASS_THRESHOLD_KG

    def _signed_safe_velocity(self, velocity: torch.Tensor) -> torch.Tensor:
        sign = torch.where(velocity >= 0.0, 1.0, -1.0).to(dtype=velocity.dtype, device=velocity.device)
        return sign * torch.clamp(torch.abs(velocity), min=float(self.min_speed_mps.item()))

    def derivatives(self, state: torch.Tensor, control: torch.Tensor, trailer_mass_kg: torch.Tensor) -> torch.Tensor:
        if trailer_mass_kg.ndim == 2 and trailer_mass_kg.shape[1] == 1:
            trailer_mass_kg = trailer_mass_kg[:, 0]

        has_trailer = trailer_mass_kg > self.no_trailer_mass_threshold_kg
        trailer_mask = has_trailer.to(dtype=state.dtype, device=state.device)
        safe_trailer_mass_kg = torch.where(
            has_trailer,
            torch.clamp(trailer_mass_kg, min=1000.0),
            torch.ones_like(trailer_mass_kg),
        )
        trailer_inertia = self.Iz_s_base * (safe_trailer_mass_kg / torch.clamp(self.m_s_base, min=1.0))

        x_t = state[:, 0]
        y_t = state[:, 1]
        psi_t = state[:, 2]
        vx_t = state[:, 3]
        vy_t = state[:, 4]
        r_t = state[:, 5]
        x_s = state[:, 6]
        y_s = state[:, 7]
        psi_s = state[:, 8]
        vx_s = state[:, 9]
        vy_s = state[:, 10]
        r_s = state[:, 11]

        delta_f = control[:, 0]
        torque_fl = control[:, 1]
        torque_fr = control[:, 2]
        torque_rl = control[:, 3]
        torque_rr = control[:, 4]

        b_t = self.L_t - self.a_t

        vx_t_safe = self._signed_safe_velocity(vx_t)
        vx_s_safe = self._signed_safe_velocity(vx_s)

        alpha_f = delta_f - torch.atan2(vy_t + self.a_t * r_t, vx_t_safe + self._eps)
        alpha_r = -torch.atan2(vy_t - b_t * r_t, vx_t_safe + self._eps)
        alpha_s = -torch.atan2(vy_s - self.L_s * r_s, vx_s_safe + self._eps)

        fyf = self.Cf * alpha_f
        fyr = self.Cr * alpha_r
        fys = self.Cs * alpha_s * trailer_mask

        cos_psi_t = torch.cos(psi_t)
        sin_psi_t = torch.sin(psi_t)
        cos_psi_s = torch.cos(psi_s)
        sin_psi_s = torch.sin(psi_s)

        hitch_global_x = x_t + self.hitch_x * cos_psi_t - self.hitch_y * sin_psi_t
        hitch_global_y = y_t + self.hitch_x * sin_psi_t + self.hitch_y * cos_psi_t

        hitch_vel_t_x_body = vx_t - r_t * self.hitch_y
        hitch_vel_t_y_body = vy_t + r_t * self.hitch_x
        hitch_vel_t_x_global = hitch_vel_t_x_body * cos_psi_t - hitch_vel_t_y_body * sin_psi_t
        hitch_vel_t_y_global = hitch_vel_t_x_body * sin_psi_t + hitch_vel_t_y_body * cos_psi_t

        hitch_global_s_x = x_s + self.c_s * cos_psi_s
        hitch_global_s_y = y_s + self.c_s * sin_psi_s
        hitch_vel_s_x_body = vx_s
        hitch_vel_s_y_body = vy_s - r_s * self.c_s
        hitch_vel_s_x_global = hitch_vel_s_x_body * cos_psi_s - hitch_vel_s_y_body * sin_psi_s
        hitch_vel_s_y_global = hitch_vel_s_x_body * sin_psi_s + hitch_vel_s_y_body * cos_psi_s

        pos_error_x = hitch_global_x - hitch_global_s_x
        pos_error_y = hitch_global_y - hitch_global_s_y
        vel_error_x = hitch_vel_t_x_global - hitch_vel_s_x_global
        vel_error_y = hitch_vel_t_y_global - hitch_vel_s_y_global

        k_pos = 1.0e6
        k_vel = 1.0e4
        hitch_force_x_global = (-k_pos * pos_error_x - k_vel * vel_error_x) * trailer_mask
        hitch_force_y_global = (-k_pos * pos_error_y - k_vel * vel_error_y) * trailer_mask

        hitch_force_t_x_body = hitch_force_x_global * cos_psi_t + hitch_force_y_global * sin_psi_t
        hitch_force_t_y_body = -hitch_force_x_global * sin_psi_t + hitch_force_y_global * cos_psi_t
        hitch_force_s_x_body = -(hitch_force_x_global * cos_psi_s + hitch_force_y_global * sin_psi_s)
        hitch_force_s_y_body = -(-hitch_force_x_global * sin_psi_s + hitch_force_y_global * cos_psi_s)

        fx_fl = torque_fl / self.wheel_radius
        fx_fr = torque_fr / self.wheel_radius
        fx_rl = torque_rl / self.wheel_radius
        fx_rr = torque_rr / self.wheel_radius

        cos_delta = torch.cos(delta_f)
        sin_delta = torch.sin(delta_f)
        front_longitudinal = fx_fl + fx_fr
        rear_longitudinal = fx_rl + fx_rr
        fx_front_body = front_longitudinal * cos_delta
        fy_front_from_drive = front_longitudinal * sin_delta

        tractor_speed = torch.sqrt(vx_t * vx_t + vy_t * vy_t + self._eps)
        trailer_speed = torch.sqrt(vx_s * vx_s + vy_s * vy_s + self._eps)
        drag_t = -0.5 * self.rho * self.CdA_t * tractor_speed * vx_t
        drag_s = -0.5 * self.rho * self.CdA_s * trailer_speed * vx_s * trailer_mask
        roll_t = self.rolling_coeff * self.m_t * self.g * torch.tanh(10.0 * vx_t)
        roll_s = self.rolling_coeff * safe_trailer_mass_kg * self.g * torch.tanh(10.0 * vx_s) * trailer_mask

        fx_total_t = fx_front_body + rear_longitudinal + fyf * sin_delta + hitch_force_t_x_body + drag_t - roll_t
        fy_total_t = fyf * cos_delta + fyr + hitch_force_t_y_body + fy_front_from_drive

        dvx_t = fx_total_t / self.m_t + r_t * vy_t
        dvy_t = fy_total_t / self.m_t - r_t * vx_t
        dpsi_t = r_t
        dr_t = (
            self.a_t * (fyf * cos_delta + fy_front_from_drive)
            - b_t * fyr
            + (self.hitch_x * hitch_force_t_y_body - self.hitch_y * hitch_force_t_x_body)
            + (fx_fr - fx_fl) * (self.track_width * 0.5)
            + (fx_rr - fx_rl) * (self.track_width * 0.5)
        ) / self.Iz_t

        dvx_s_trailer = (hitch_force_s_x_body + drag_s - roll_s) / safe_trailer_mass_kg + r_s * vy_s
        dvy_s_trailer = (fys + hitch_force_s_y_body) / safe_trailer_mass_kg - r_s * vx_s
        dpsi_s_trailer = r_s
        dr_s_trailer = (-self.L_s * fys + self.c_s * hitch_force_s_y_body) / trailer_inertia

        dx_t = vx_t * cos_psi_t - vy_t * sin_psi_t
        dy_t = vx_t * sin_psi_t + vy_t * cos_psi_t
        dx_s_trailer = vx_s * cos_psi_s - vy_s * sin_psi_s
        dy_s_trailer = vx_s * sin_psi_s + vy_s * cos_psi_s

        # No-trailer mode: disable hitch/trailer dynamics and mirror tractor channels into
        # the trailer placeholder states so training/inference can run on single-vehicle data.
        dx_s = trailer_mask * dx_s_trailer + (1.0 - trailer_mask) * dx_t
        dy_s = trailer_mask * dy_s_trailer + (1.0 - trailer_mask) * dy_t
        dpsi_s = trailer_mask * dpsi_s_trailer + (1.0 - trailer_mask) * dpsi_t
        dvx_s = trailer_mask * dvx_s_trailer + (1.0 - trailer_mask) * dvx_t
        dvy_s = trailer_mask * dvy_s_trailer + (1.0 - trailer_mask) * dvy_t
        dr_s = trailer_mask * dr_s_trailer + (1.0 - trailer_mask) * dr_t

        return torch.stack(
            [dx_t, dy_t, dpsi_t, dvx_t, dvy_t, dr_t, dx_s, dy_s, dpsi_s, dvx_s, dvy_s, dr_s],
            dim=1,
        )

    def forward(self, state: torch.Tensor, control: torch.Tensor, trailer_mass_kg: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
        if dt.ndim == 1:
            dt = dt.unsqueeze(1)
        if trailer_mass_kg.ndim == 1:
            trailer_mass_kg = trailer_mass_kg.unsqueeze(1)

        k1 = self.derivatives(state, control, trailer_mass_kg)
        k2 = self.derivatives(state + 0.5 * dt * k1, control, trailer_mass_kg)
        k3 = self.derivatives(state + 0.5 * dt * k2, control, trailer_mass_kg)
        k4 = self.derivatives(state + dt * k3, control, trailer_mass_kg)
        next_state = state + dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
        next_state = next_state.clone()
        no_trailer_mask = (trailer_mass_kg[:, 0] <= self.no_trailer_mass_threshold_kg)
        if torch.any(no_trailer_mask):
            next_state[no_trailer_mask, 6] = next_state[no_trailer_mask, 0]
            next_state[no_trailer_mask, 7] = next_state[no_trailer_mask, 1]
            next_state[no_trailer_mask, 8] = next_state[no_trailer_mask, 2]
            next_state[no_trailer_mask, 9] = next_state[no_trailer_mask, 3]
            next_state[no_trailer_mask, 10] = next_state[no_trailer_mask, 4]
            next_state[no_trailer_mask, 11] = next_state[no_trailer_mask, 5]
        next_state[:, 2] = wrap_angle_error_torch(next_state[:, 2])
        next_state[:, 8] = wrap_angle_error_torch(next_state[:, 8])
        return next_state


class MLPErrorModel(nn.Module):
    """学习卡车-挂车单步运动残差。"""

    def __init__(self, input_dim: int, output_dim: int, dropout_p: float = 0.08, use_layer_norm: bool = MLP_USE_LAYER_NORM) -> None:
        super().__init__()
        safe_dropout = float(np.clip(dropout_p, 0.0, 0.5))
        self.use_layer_norm = bool(use_layer_norm)
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            self._build_norm(128),
            nn.Tanh(),
            nn.Dropout(safe_dropout),
            nn.Linear(128, 128),
            self._build_norm(128),
            nn.Tanh(),
            nn.Dropout(safe_dropout),
            nn.Linear(128, 128),
            self._build_norm(128),
            nn.Tanh(),
            nn.Dropout(safe_dropout),
            nn.Linear(128, 128),
            self._build_norm(128),
            nn.Tanh(),
            nn.Dropout(safe_dropout),
            nn.Linear(128, output_dim),
        )
        output_layer = self.network[-1]
        nn.init.zeros_(output_layer.weight)
        nn.init.zeros_(output_layer.bias)

    def _build_norm(self, hidden_dim: int) -> nn.Module:
        if self.use_layer_norm:
            return nn.LayerNorm(hidden_dim)
        return nn.Identity()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


def wrap_angle_error_np(angle: np.ndarray) -> np.ndarray:
    return ((angle + np.pi) % (2.0 * np.pi) - np.pi).astype(MLP_NUMPY_DTYPE)


def wrap_angle_error_torch(angle: torch.Tensor) -> torch.Tensor:
    return torch.remainder(angle + np.pi, 2.0 * np.pi) - np.pi


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
        raise ValueError(f"缺少字段 `{field_name}` | 候选列: {candidates} | 文件: {csv_path}")
    return column


def read_column_as_float(frame: pd.DataFrame, candidates: list[str], field_name: str, csv_path: Path) -> np.ndarray:
    column = require_column(frame, candidates, field_name, csv_path)
    return frame[column].to_numpy(dtype=np.float32)


def find_all_real_data_csvs(runs_root: Path) -> list[Path]:
    if not runs_root.exists():
        raise FileNotFoundError(f"runs_root 不存在: {runs_root}")
    return sorted(
        runs_root.glob("python_run_*/outputs/control_and_trajectory.csv"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )


def resolve_plot_dir_for_segment(csv_path: Path) -> Path:
    plot_dir = csv_path.parent / "truck_trailer_residual_plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    return plot_dir


def resolve_front_wheel_angle_rad(frame: pd.DataFrame, csv_path: Path) -> np.ndarray:
    road_rad_col = find_first_existing_column(frame, ROAD_WHEEL_RAD_CANDIDATES)
    road_deg_col = find_first_existing_column(frame, ROAD_WHEEL_DEG_CANDIDATES)
    if road_rad_col is not None:
        road_rad = frame[road_rad_col].to_numpy(dtype=np.float32)
        if np.isfinite(road_rad).any():
            road_rad = np.where(np.isfinite(road_rad), road_rad, 0.0).astype(np.float32)
            return road_rad.astype(np.float32)
    if road_deg_col is not None:
        road_deg = frame[road_deg_col].to_numpy(dtype=np.float32)
        if np.isfinite(road_deg).any():
            road_deg = np.where(np.isfinite(road_deg), road_deg, 0.0).astype(np.float32)
            return np.deg2rad(road_deg).astype(np.float32)

    ratio_col = find_first_existing_column(frame, STEERING_RATIO_CANDIDATES)
    ratio_values = np.full(len(frame), float(BASE_MODEL_PARAMS["steering_ratio"]), dtype=np.float32)
    if ratio_col is not None:
        raw_ratio = frame[ratio_col].to_numpy(dtype=np.float32)
        valid_ratio = np.isfinite(raw_ratio) & (np.abs(raw_ratio) > 1.0e-6)
        ratio_values = np.where(valid_ratio, raw_ratio, ratio_values).astype(np.float32)

    sw_rad_col = find_first_existing_column(frame, STEER_SW_RAD_CANDIDATES)
    if sw_rad_col is not None:
        sw_rad = frame[sw_rad_col].to_numpy(dtype=np.float32)
        if np.isfinite(sw_rad).any():
            sw_rad = np.where(np.isfinite(sw_rad), sw_rad, 0.0).astype(np.float32)
            return (sw_rad / ratio_values).astype(np.float32)

    sw_deg_col = find_first_existing_column(frame, STEER_SW_DEG_CANDIDATES)
    if sw_deg_col is not None:
        sw_deg = frame[sw_deg_col].to_numpy(dtype=np.float32)
        if np.isfinite(sw_deg).any():
            sw_deg = np.where(np.isfinite(sw_deg), sw_deg, 0.0).astype(np.float32)
            sw_rad = np.deg2rad(sw_deg).astype(np.float32)
            return (sw_rad / ratio_values).astype(np.float32)

    raise ValueError(
        "缺少前轮转角字段；未找到前轮转角列，也未找到可换算前轮转角的方向盘角列。"
        f" 文件: {csv_path}"
    )


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
        f"[Warning] 未找到挂车质量字段，回退到默认挂车质量 {DEFAULT_TRAILER_MASS_KG:.1f} kg | 文件: {csv_path}"
    )
    return np.full(len(frame), DEFAULT_TRAILER_MASS_KG, dtype=np.float32)


def try_read_column_as_float(frame: pd.DataFrame, candidates: list[str]) -> np.ndarray | None:
    column = find_first_existing_column(frame, candidates)
    if column is None:
        return None
    return frame[column].to_numpy(dtype=np.float32)


def load_truck_trailer_data_as_segment(csv_path: Path) -> SegmentData:
    dataframe = pd.read_csv(csv_path)
    time_column = require_column(dataframe, ["Time_s", "Time"], "time", csv_path)
    time = dataframe[time_column].to_numpy(dtype=np.float64)
    if len(time) < 2:
        raise ValueError(f"CSV 行数过少，至少需要 2 行: {csv_path}")

    dt_values = np.diff(time)
    positive_dt = dt_values[dt_values > 0.0]
    nominal_dt = float(np.median(positive_dt)) if positive_dt.size else 0.02
    dt_values = np.where(dt_values > 0.0, dt_values, nominal_dt).astype(np.float32)

    delta_f_rad = resolve_front_wheel_angle_rad(dataframe, csv_path)
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
        print(
            "[Info] 当前数据缺少挂车状态列，自动切换到无挂车模式："
            f" trailer_mass_kg=0 | 文件: {csv_path}"
        )
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
    full_controls = np.column_stack([delta_f_rad, torque_fl, torque_fr, torque_rl, torque_rr]).astype(np.float32)

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
    print("各状态误差尺度：")
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
    if len(segments) < 2:
        raise ValueError("至少需要 2 段有效数据才能做整段 train/val 划分。")
    indices = np.arange(len(segments))
    val_count = max(1, int(round(len(segments) * val_ratio)))
    val_count = min(val_count, len(segments) - 1)
    train_idx, val_idx = train_test_split(indices, test_size=val_count, random_state=seed, shuffle=True)
    return [segments[i] for i in train_idx], [segments[i] for i in val_idx]


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


def compute_loss_components(
    predicted_motion_error: torch.Tensor,
    true_error: torch.Tensor,
    base_next: torch.Tensor,
    dt_values: torch.Tensor,
    loss_context: dict[str, torch.Tensor],
    pose_loss_weight: float,
) -> dict[str, torch.Tensor]:
    error_scale = loss_context["error_scale"].to(dtype=MLP_TORCH_DTYPE)
    pose_error_scale = loss_context["pose_error_scale"].to(dtype=MLP_TORCH_DTYPE)
    motion_error_scale = loss_context["motion_error_scale"].to(dtype=MLP_TORCH_DTYPE)
    channel_weight = loss_context["channel_weight"].to(dtype=MLP_TORCH_DTYPE)

    predicted_error = derive_full_error_from_motion_error_torch(predicted_motion_error, base_next, dt_values)
    true_pose_error = torch.cat([true_error[:, :3], true_error[:, 6:9]], dim=1)
    true_motion_error = torch.cat([true_error[:, 3:6], true_error[:, 9:12]], dim=1)
    predicted_pose_error = torch.cat([predicted_error[:, :3], predicted_error[:, 6:9]], dim=1)

    pose_weight = torch.cat([channel_weight[:, :3], channel_weight[:, 6:9]], dim=1)
    motion_weight = torch.cat([channel_weight[:, 3:6], channel_weight[:, 9:12]], dim=1)

    pose_residual = (predicted_pose_error - true_pose_error) / pose_error_scale
    motion_residual = (predicted_motion_error - true_motion_error) / motion_error_scale
    pose_loss = torch.mean((pose_residual * pose_weight).square())
    motion_loss = torch.mean((motion_residual * motion_weight).square())

    full_residual = (predicted_error - true_error) / error_scale
    total_loss = motion_loss + pose_loss_weight * pose_loss + 0.05 * torch.mean((full_residual * channel_weight).square())
    return {
        "total_loss": total_loss,
        "pose_loss": pose_loss,
        "motion_loss": motion_loss,
    }


def build_checkpoint_payload(
    state_dict: dict[str, torch.Tensor],
    model_input_dim: int,
    feature_context: dict[str, np.ndarray],
    loss_context: dict[str, torch.Tensor],
) -> dict[str, object]:
    return {
        "state_dict": state_dict,
        "model_input_dim": int(model_input_dim),
        "model_output_dim": int(len(MOTION_ERROR_NAMES)),
        "mlp_use_layer_norm": bool(MLP_USE_LAYER_NORM),
        "input_feature_names": list(MLP_INPUT_FEATURE_NAMES),
        "motion_error_names": list(MOTION_ERROR_NAMES),
        "state_names": list(STATE_NAMES),
        "control_names": list(CONTROL_NAMES),
        "feature_mean": feature_context["feature_mean"],
        "feature_scale": feature_context["feature_scale"],
        "loss_error_scale": loss_context["error_scale"].detach().cpu().numpy(),
        "loss_pose_error_scale": loss_context["pose_error_scale"].detach().cpu().numpy(),
        "loss_motion_error_scale": loss_context["motion_error_scale"].detach().cpu().numpy(),
        "base_model_params": dict(BASE_MODEL_PARAMS),
    }


def clone_unwrapped_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def load_unwrapped_state_dict(model: nn.Module, state_dict: dict[str, torch.Tensor]) -> None:
    model.load_state_dict(state_dict)


def build_train_loader(
    x_tensor: torch.Tensor,
    y_tensor: torch.Tensor,
    base_next_tensor: torch.Tensor,
    dt_tensor: torch.Tensor,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader:
    dataset = TensorDataset(x_tensor, y_tensor, base_next_tensor, dt_tensor)
    return DataLoader(
        dataset,
        batch_size=max(1, min(int(batch_size), len(dataset))),
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )


def compute_pose_loss_weight(epoch: int) -> float:
    if epoch <= MOTION_ONLY_WARMUP_EPOCHS:
        return 0.0
    if POSE_RAMP_EPOCHS <= 0:
        return 1.0
    progress = min(max(epoch - MOTION_ONLY_WARMUP_EPOCHS, 0), POSE_RAMP_EPOCHS)
    return float(progress) / float(POSE_RAMP_EPOCHS)


def train_error_model_multirun(
    base_model: TruckTrailerNominalDynamics,
    train_segments: list[SegmentData],
    val_segments: list[SegmentData],
    device: torch.device,
    epochs: int = TRAIN_EPOCHS,
    learning_rate: float = LEARNING_RATE,
    batch_size: int = TRAIN_BATCH_SIZE,
    num_workers: int = TRAIN_NUM_WORKERS,
) -> tuple[MLPErrorModel, dict[str, np.ndarray], dict[str, torch.Tensor], dict[str, list[float]]]:
    x_train_raw, y_train, base_next_train = concat_segments_for_training(base_model, train_segments, device)
    x_val_raw, y_val, base_next_val = concat_segments_for_training(base_model, val_segments, device)

    if len(x_train_raw) < 5:
        raise ValueError("训练样本太少，无法训练。")

    train_dt_values = x_train_raw[:, -1:].copy()
    val_dt_values = x_val_raw[:, -1:].copy()

    feature_context = build_feature_context(x_train_raw)
    x_train = normalize_features_np(x_train_raw, feature_context)
    x_val = normalize_features_np(x_val_raw, feature_context)
    loss_context = build_loss_context(y_train, device)
    describe_loss_context(loss_context)

    model = MLPErrorModel(
        input_dim=x_train.shape[1],
        output_dim=len(MOTION_ERROR_NAMES),
        dropout_p=0.08,
        use_layer_norm=MLP_USE_LAYER_NORM,
    ).to(device=device, dtype=MLP_TORCH_DTYPE)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1.0e-5)

    x_train_tensor = torch.as_tensor(x_train, dtype=MLP_TORCH_DTYPE)
    y_train_tensor = torch.as_tensor(y_train, dtype=MLP_TORCH_DTYPE)
    base_next_train_tensor = torch.as_tensor(base_next_train, dtype=MLP_TORCH_DTYPE)
    dt_train_tensor = torch.as_tensor(train_dt_values, dtype=MLP_TORCH_DTYPE)
    x_val_tensor = torch.as_tensor(x_val, dtype=MLP_TORCH_DTYPE)
    y_val_tensor = torch.as_tensor(y_val, dtype=MLP_TORCH_DTYPE)
    base_next_val_tensor = torch.as_tensor(base_next_val, dtype=MLP_TORCH_DTYPE)
    dt_val_tensor = torch.as_tensor(val_dt_values, dtype=MLP_TORCH_DTYPE)

    pin_memory = device.type == "cuda"
    train_loader = build_train_loader(
        x_train_tensor,
        y_train_tensor,
        base_next_train_tensor,
        dt_train_tensor,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        TensorDataset(x_val_tensor, y_val_tensor, base_next_val_tensor, dt_val_tensor),
        batch_size=max(1, min(batch_size, len(x_val_tensor))),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    history = {
        "train_total": [],
        "val_total": [],
        "train_pose": [],
        "val_pose": [],
        "train_motion": [],
        "val_motion": [],
        "pose_loss_weight": [],
    }

    best_val_loss = float("inf")
    best_train_loss = float("inf")
    best_state_dict: dict[str, torch.Tensor] | None = None
    best_train_state_dict: dict[str, torch.Tensor] | None = None

    for epoch in range(1, epochs + 1):
        pose_loss_weight = compute_pose_loss_weight(epoch)
        model.train()
        train_total = 0.0
        train_pose = 0.0
        train_motion = 0.0
        train_count = 0

        for x_batch_cpu, y_batch_cpu, base_next_batch_cpu, dt_batch_cpu in train_loader:
            x_batch = x_batch_cpu.to(device, non_blocking=pin_memory)
            y_batch = y_batch_cpu.to(device, non_blocking=pin_memory)
            base_next_batch = base_next_batch_cpu.to(device, non_blocking=pin_memory)
            dt_batch = dt_batch_cpu.to(device, non_blocking=pin_memory)

            optimizer.zero_grad(set_to_none=True)
            predicted_motion_error = model(x_batch)
            losses = compute_loss_components(
                predicted_motion_error=predicted_motion_error,
                true_error=y_batch,
                base_next=base_next_batch,
                dt_values=dt_batch,
                loss_context=loss_context,
                pose_loss_weight=pose_loss_weight,
            )
            losses["total_loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_NORM)
            optimizer.step()

            batch_size_value = x_batch.shape[0]
            train_total += float(losses["total_loss"].detach().cpu()) * batch_size_value
            train_pose += float(losses["pose_loss"].detach().cpu()) * batch_size_value
            train_motion += float(losses["motion_loss"].detach().cpu()) * batch_size_value
            train_count += batch_size_value

        model.eval()
        val_total = 0.0
        val_pose = 0.0
        val_motion = 0.0
        val_count = 0
        with torch.no_grad():
            for x_batch_cpu, y_batch_cpu, base_next_batch_cpu, dt_batch_cpu in val_loader:
                x_batch = x_batch_cpu.to(device, non_blocking=pin_memory)
                y_batch = y_batch_cpu.to(device, non_blocking=pin_memory)
                base_next_batch = base_next_batch_cpu.to(device, non_blocking=pin_memory)
                dt_batch = dt_batch_cpu.to(device, non_blocking=pin_memory)
                predicted_motion_error = model(x_batch)
                losses = compute_loss_components(
                    predicted_motion_error=predicted_motion_error,
                    true_error=y_batch,
                    base_next=base_next_batch,
                    dt_values=dt_batch,
                    loss_context=loss_context,
                    pose_loss_weight=pose_loss_weight,
                )
                batch_size_value = x_batch.shape[0]
                val_total += float(losses["total_loss"].detach().cpu()) * batch_size_value
                val_pose += float(losses["pose_loss"].detach().cpu()) * batch_size_value
                val_motion += float(losses["motion_loss"].detach().cpu()) * batch_size_value
                val_count += batch_size_value

        train_total /= max(train_count, 1)
        train_pose /= max(train_count, 1)
        train_motion /= max(train_count, 1)
        val_total /= max(val_count, 1)
        val_pose /= max(val_count, 1)
        val_motion /= max(val_count, 1)

        history["train_total"].append(train_total)
        history["val_total"].append(val_total)
        history["train_pose"].append(train_pose)
        history["val_pose"].append(val_pose)
        history["train_motion"].append(train_motion)
        history["val_motion"].append(val_motion)
        history["pose_loss_weight"].append(pose_loss_weight)

        if train_total < best_train_loss:
            best_train_loss = train_total
            best_train_state_dict = clone_unwrapped_state_dict(model)
        if val_total < best_val_loss:
            best_val_loss = val_total
            best_state_dict = clone_unwrapped_state_dict(model)

        if epoch <= 10 or epoch % 100 == 0 or epoch == epochs:
            print(
                f"Epoch {epoch:5d}/{epochs} | pose_w={pose_loss_weight:.3f} | "
                f"train_total={train_total:.6e} val_total={val_total:.6e} | "
                f"train_motion={train_motion:.6e} val_motion={val_motion:.6e}"
            )

    if best_state_dict is not None:
        load_unwrapped_state_dict(model, best_state_dict)
        torch.save(
            build_checkpoint_payload(best_state_dict, int(x_train.shape[1]), feature_context, loss_context),
            MODEL_CHECKPOINT,
        )

    if best_train_state_dict is not None:
        torch.save(
            build_checkpoint_payload(best_train_state_dict, int(x_train.shape[1]), feature_context, loss_context),
            TRAIN_LOSS_MODEL_CHECKPOINT,
        )

    return model, feature_context, loss_context, history


@torch.no_grad()
def rollout_models_teacher_forcing(
    base_model: TruckTrailerNominalDynamics,
    error_model: nn.Module,
    real_rollout: np.ndarray,
    control_sequence: np.ndarray,
    trailer_mass_kg: np.ndarray,
    dt_values: np.ndarray,
    feature_context: dict[str, np.ndarray],
    loss_context: dict[str, torch.Tensor],
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    step_count = len(real_rollout)
    base_rollout = np.zeros_like(real_rollout, dtype=np.float32)
    corrected_rollout = np.zeros_like(real_rollout, dtype=np.float32)
    base_rollout[0] = real_rollout[0].astype(np.float32)
    corrected_rollout[0] = real_rollout[0].astype(np.float32)

    motion_error_clip = 3.0 * loss_context["motion_error_scale"].detach().cpu().numpy().ravel().astype(np.float32)
    feature_context_tensors = build_feature_context_tensors(feature_context, device)
    error_model.eval()

    for step in range(len(control_sequence)):
        current_state_tensor = to_tensor(real_rollout[step : step + 1], device)
        control_tensor = to_tensor(control_sequence[step : step + 1], device)
        mass_tensor = to_tensor(np.array([[trailer_mass_kg[step]]], dtype=np.float32), device)
        dt_tensor = to_tensor(np.array([[dt_values[step]]], dtype=np.float32), device)

        base_next_tensor = base_model(current_state_tensor, control_tensor, mass_tensor, dt_tensor)
        features = build_mlp_input_feature_tensor(current_state_tensor, control_tensor, mass_tensor, dt_tensor)
        features = normalize_feature_tensor(features, feature_context_tensors)
        predicted_motion_error = error_model(features).cpu().numpy()[0].astype(np.float32)
        predicted_motion_error = np.clip(predicted_motion_error, -motion_error_clip, motion_error_clip)

        base_next = base_next_tensor.cpu().numpy().astype(np.float32)
        corrected_error = derive_full_error_from_motion_error_np(
            predicted_motion_error.reshape(1, -1),
            base_next,
            np.array([dt_values[step]], dtype=np.float32),
        )[0]
        corrected_next = base_next[0] + corrected_error
        corrected_next[2] = wrap_angle_error_np(np.asarray([corrected_next[2]], dtype=np.float32))[0]
        corrected_next[8] = wrap_angle_error_np(np.asarray([corrected_next[8]], dtype=np.float32))[0]

        base_rollout[step + 1] = base_next[0]
        corrected_rollout[step + 1] = corrected_next.astype(np.float32)

    return base_rollout, corrected_rollout


def plot_training_history(history: dict[str, list[float]], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    epochs = np.arange(1, len(history["train_total"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(epochs, safe_log10(np.asarray(history["train_total"])), label="Train total")
    axes[0].plot(epochs, safe_log10(np.asarray(history["val_total"])), label="Val total")
    axes[0].set_title("Total Loss (log10)")
    axes[0].set_xlabel("Epoch")
    axes[0].grid(True, linestyle="--", alpha=0.35)
    axes[0].legend()

    axes[1].plot(epochs, safe_log10(np.asarray(history["train_motion"])), label="Train motion")
    axes[1].plot(epochs, safe_log10(np.asarray(history["val_motion"])), label="Val motion")
    axes[1].plot(epochs, np.asarray(history["pose_loss_weight"]), label="Pose weight")
    axes[1].set_title("Motion Loss / Pose Weight")
    axes[1].set_xlabel("Epoch")
    axes[1].grid(True, linestyle="--", alpha=0.35)
    axes[1].legend()

    output_path = output_dir / "truck_trailer_training_loss_log.png"
    save_figure(fig, output_path)
    return output_path


def compute_articulation_series(states: np.ndarray) -> np.ndarray:
    return np.rad2deg(wrap_angle_error_np(states[:, 8] - states[:, 2]))


def plot_trajectory(real_rollout: np.ndarray, base_rollout: np.ndarray, corrected_rollout: np.ndarray, plot_dir: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].plot(real_rollout[:, 0], real_rollout[:, 1], label="CarSim/TruckSim", linewidth=1.8)
    axes[0].plot(base_rollout[:, 0], base_rollout[:, 1], label="Base", linewidth=1.5)
    axes[0].plot(corrected_rollout[:, 0], corrected_rollout[:, 1], label="Base + NN", linewidth=1.6)
    axes[0].set_title("Tractor Trajectory")
    axes[0].set_xlabel("X (m)")
    axes[0].set_ylabel("Y (m)")
    axes[0].axis("equal")
    axes[0].grid(True, linestyle="--", alpha=0.35)
    axes[0].legend()

    axes[1].plot(real_rollout[:, 6], real_rollout[:, 7], label="CarSim/TruckSim", linewidth=1.8)
    axes[1].plot(base_rollout[:, 6], base_rollout[:, 7], label="Base", linewidth=1.5)
    axes[1].plot(corrected_rollout[:, 6], corrected_rollout[:, 7], label="Base + NN", linewidth=1.6)
    axes[1].set_title("Trailer Trajectory")
    axes[1].set_xlabel("X (m)")
    axes[1].set_ylabel("Y (m)")
    axes[1].axis("equal")
    axes[1].grid(True, linestyle="--", alpha=0.35)
    axes[1].legend()

    output_path = plot_dir / "truck_trailer_trajectory_comparison.png"
    save_figure(fig, output_path)
    return output_path


def plot_key_state_timeseries(
    time: np.ndarray,
    real_rollout: np.ndarray,
    base_rollout: np.ndarray,
    corrected_rollout: np.ndarray,
    plot_dir: Path,
) -> Path:
    fig, axes = plt.subplots(4, 2, figsize=(14, 12))
    axes = axes.ravel()
    series = [
        ("Tractor Vx", real_rollout[:, 3], base_rollout[:, 3], corrected_rollout[:, 3], "m/s"),
        ("Tractor Vy", real_rollout[:, 4], base_rollout[:, 4], corrected_rollout[:, 4], "m/s"),
        ("Tractor Yaw Rate", np.rad2deg(real_rollout[:, 5]), np.rad2deg(base_rollout[:, 5]), np.rad2deg(corrected_rollout[:, 5]), "deg/s"),
        ("Trailer Vx", real_rollout[:, 9], base_rollout[:, 9], corrected_rollout[:, 9], "m/s"),
        ("Trailer Vy", real_rollout[:, 10], base_rollout[:, 10], corrected_rollout[:, 10], "m/s"),
        ("Trailer Yaw Rate", np.rad2deg(real_rollout[:, 11]), np.rad2deg(base_rollout[:, 11]), np.rad2deg(corrected_rollout[:, 11]), "deg/s"),
        ("Articulation", compute_articulation_series(real_rollout), compute_articulation_series(base_rollout), compute_articulation_series(corrected_rollout), "deg"),
        ("Tractor Yaw", np.rad2deg(wrap_angle_error_np(real_rollout[:, 2])), np.rad2deg(wrap_angle_error_np(base_rollout[:, 2])), np.rad2deg(wrap_angle_error_np(corrected_rollout[:, 2])), "deg"),
    ]

    for axis, (title, real_values, base_values, corrected_values, unit) in zip(axes, series, strict=False):
        axis.plot(time, real_values, label="CarSim/TruckSim", linewidth=1.6)
        axis.plot(time, base_values, label="Base", linewidth=1.4)
        axis.plot(time, corrected_values, label="Base + NN", linewidth=1.5)
        axis.set_title(title)
        axis.set_xlabel("Time (s)")
        axis.set_ylabel(unit)
        axis.grid(True, linestyle="--", alpha=0.35)
        axis.legend()

    output_path = plot_dir / "truck_trailer_state_timeseries.png"
    save_figure(fig, output_path)
    return output_path


def print_rollout_rmse(real_rollout: np.ndarray, base_rollout: np.ndarray, corrected_rollout: np.ndarray) -> dict[str, float]:
    metrics = {
        "tractor_x_rmse_m": float(np.sqrt(np.mean((base_rollout[:, 0] - real_rollout[:, 0]) ** 2))),
        "tractor_x_rmse_m_corrected": float(np.sqrt(np.mean((corrected_rollout[:, 0] - real_rollout[:, 0]) ** 2))),
        "trailer_x_rmse_m": float(np.sqrt(np.mean((base_rollout[:, 6] - real_rollout[:, 6]) ** 2))),
        "trailer_x_rmse_m_corrected": float(np.sqrt(np.mean((corrected_rollout[:, 6] - real_rollout[:, 6]) ** 2))),
        "articulation_rmse_deg": float(np.sqrt(np.mean((compute_articulation_series(base_rollout) - compute_articulation_series(real_rollout)) ** 2))),
        "articulation_rmse_deg_corrected": float(
            np.sqrt(np.mean((compute_articulation_series(corrected_rollout) - compute_articulation_series(real_rollout)) ** 2))
        ),
    }
    print("Rollout RMSE:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.6f}")
    return metrics


def export_dataset_split_tables(
    train_segments: list[SegmentData],
    val_segments: list[SegmentData],
    output_dir: Path,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    split_rows: list[dict[str, object]] = []
    for split_name, segments in (("train", train_segments), ("val", val_segments)):
        for index, seg in enumerate(segments):
            split_rows.append(
                {
                    "split": split_name,
                    "index_in_split": index,
                    "segment_name": seg.segment_name,
                    "csv_path": str(seg.csv_path),
                    "sample_count": int(seg.states.shape[0]),
                    "time_start_s": float(seg.time[0]),
                    "time_end_s": float(seg.time[-1]),
                    "trailer_mass_mean_kg": float(np.mean(seg.trailer_mass_kg)),
                }
            )

    split_table_path = output_dir / "truck_trailer_dataset_split_segments.csv"
    pd.DataFrame(split_rows).to_csv(split_table_path, index=False, encoding="utf-8-sig")

    val_rows = [
        {
            "index_in_val": index,
            "segment_name": seg.segment_name,
            "csv_path": str(seg.csv_path),
            "sample_count": int(seg.states.shape[0]),
            "time_start_s": float(seg.time[0]),
            "time_end_s": float(seg.time[-1]),
            "trailer_mass_mean_kg": float(np.mean(seg.trailer_mass_kg)),
        }
        for index, seg in enumerate(val_segments)
    ]
    val_table_path = output_dir / "truck_trailer_validation_segments.csv"
    pd.DataFrame(val_rows).to_csv(val_table_path, index=False, encoding="utf-8-sig")
    return split_table_path, val_table_path


def main() -> None:
    torch.manual_seed(10)
    np.random.seed(10)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device = {device}")

    csv_list = find_all_real_data_csvs(RUNS_ROOT)
    if not csv_list:
        raise FileNotFoundError(
            f"未找到任何 control_and_trajectory.csv，请检查目录: {RUNS_ROOT}\\python_run_*\\outputs\\control_and_trajectory.csv"
        )

    print(f"共找到 {len(csv_list)} 段候选数据。最新的一段: {csv_list[0]}")

    segments: list[SegmentData] = []
    for csv_path in csv_list:
        try:
            seg = load_truck_trailer_data_as_segment(csv_path)
            segments.append(seg)
        except Exception as exc:
            print(f"[跳过] 加载失败: {csv_path} | 原因: {exc}")

    if len(segments) < 2:
        raise ValueError(
            "有效挂车数据段不足。当前脚本需要卡车/挂车完整状态列。"
            " 如果你现在看到这里，说明现有导出文件大概率还是单车版字段。"
        )

    train_segments, val_segments = build_train_val_by_segments(segments, val_ratio=0.2, seed=10)
    print("\n===== 整段划分结果 =====")
    print(f"Train 段数: {len(train_segments)}")
    print(f"Val 段数: {len(val_segments)}")

    global_plot_dir = RUNS_ROOT / "truck_trailer_multirun_training_summary"
    split_table_path, val_table_path = export_dataset_split_tables(train_segments, val_segments, global_plot_dir)
    print(f"数据划分表已保存: {split_table_path}")
    print(f"验证段表已保存: {val_table_path}")

    base_model = TruckTrailerNominalDynamics(BASE_MODEL_PARAMS).to(device)
    error_model, feature_context, loss_context, history = train_error_model_multirun(
        base_model=base_model,
        train_segments=train_segments,
        val_segments=val_segments,
        device=device,
        epochs=TRAIN_EPOCHS,
        learning_rate=LEARNING_RATE,
        batch_size=TRAIN_BATCH_SIZE,
        num_workers=TRAIN_NUM_WORKERS,
    )

    loss_plot = plot_training_history(history, global_plot_dir)
    print(f"训练曲线已保存: {loss_plot}")

    print("\n===== 对验证段逐段输出结果 =====")
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
        print(f"[{seg.segment_name}] 轨迹图: {traj_path}")
        print(f"[{seg.segment_name}] 状态图: {state_path}")
        print(f"[{seg.segment_name}] RMSE 摘要: {rmse}")


if __name__ == "__main__":
    main()
