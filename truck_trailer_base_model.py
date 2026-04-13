from __future__ import annotations

"""
独立的卡车-挂车 base model 仿真脚本。

用途：
1. 单独保存 `train_truck_trailer_residual.py` 中使用的名义动力学模型
2. 允许手动给定当前状态、控制量、dt 和总时长
3. 输出整个仿真过程的状态变化 CSV
4. 绘制轨迹图和状态时序图

说明：
- 动力学公式与训练脚本里的 `TruckTrailerNominalDynamics` 保持一致。
- 当 `trailer_mass_kg <= 1.0` 时自动进入“无挂车模式”，挂车占位状态跟随牵引车。
"""

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
import torch.nn as nn


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
CONTROL_NAMES = ["delta_f_rad", "torque_fl", "torque_fr", "torque_rl", "torque_rr"]

BASE_MODEL_PARAMS = {
    "m_t": 9300.0,
    "Iz_t": 48639.0,
    "L_t": 5.15,
    "a_t": 2.609,
    "m_s_base": 0.0,
    "Iz_s_base": 0.0,
    "L_s": 8.0,
    "c_s": 4.0,
    "Cf": 3.97e5,
    "Cr": 3.87e5,
    "Cs": 80000.0,
    "wheel_radius": 0.5,
    "track_width": 1.878,
    "steering_ratio": 32,
    "rho": 1.225,
    "CdA_t": 0.6*9.7,
    "CdA_s": 0.6*0,
    "rolling_coeff": 0.013,
    "hitch_x": -0.3319,
    "hitch_y": 0.0022,
    "min_speed_mps": 0.5,
}

NO_TRAILER_MASS_THRESHOLD_KG = 1.0


@dataclass
class ManualSimulationConfig:
    dt: float = 0.02
    total_time: float = 10.0
    trailer_mass_kg: float = 0.0
    device: str = "cpu"
    output_dir: Path = Path(__file__).with_name("truck_trailer_base_model_outputs")

    # 手动输入初始状态:
    # [x_t, y_t, psi_t, vx_t, vy_t, r_t, x_s, y_s, psi_s, vx_s, vy_s, r_s]
    initial_state: tuple[float, ...] = (
        0.0,
        0.0,
        0.0,
        12.5,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        12.5,
        0.0,
        0.0,
    )

    # 手动输入恒定控制:
    # [delta_f_rad, torque_fl, torque_fr, torque_rl, torque_rr]
    constant_control: tuple[float, ...] = (
        float(np.deg2rad(2.5)),
        0.0,
        0.0,
        180.0,
        180.0,
    )


def wrap_angle_error_np(angle: np.ndarray) -> np.ndarray:
    return ((angle + np.pi) % (2.0 * np.pi) - np.pi).astype(np.float32)


def wrap_angle_error_torch(angle: torch.Tensor) -> torch.Tensor:
    return torch.remainder(angle + np.pi, 2.0 * np.pi) - np.pi


class TruckTrailerNominalDynamics(nn.Module):
    """与训练脚本一致的卡车-挂车名义动力学模型。"""

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
        trailer_inertia = torch.where(
            has_trailer,
            self.Iz_s_base * (safe_trailer_mass_kg / torch.clamp(self.m_s_base, min=1.0)),
            torch.ones_like(safe_trailer_mass_kg),
        )

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

        no_trailer_mask = trailer_mass_kg[:, 0] <= self.no_trailer_mass_threshold_kg
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


def build_constant_control_sequence(control: np.ndarray, step_count: int) -> np.ndarray:
    control = np.asarray(control, dtype=np.float32).reshape(1, -1)
    return np.repeat(control, step_count, axis=0).astype(np.float32)


def build_results_dataframe(
    time: np.ndarray,
    states: np.ndarray,
    controls: np.ndarray,
    trailer_mass_kg: float,
    dt: float,
) -> pd.DataFrame:
    padded_controls = np.vstack([controls, controls[-1:]]) if len(controls) else np.zeros((len(states), 5), dtype=np.float32)
    articulation_deg = np.rad2deg(wrap_angle_error_np(states[:, 8] - states[:, 2]))
    data: dict[str, np.ndarray] = {
        "time_s": time.astype(np.float32),
        "dt_s": np.full(len(states), float(dt), dtype=np.float32),
        "trailer_mass_kg": np.full(len(states), float(trailer_mass_kg), dtype=np.float32),
        "articulation_deg": articulation_deg.astype(np.float32),
        "control_delta_f_deg": np.rad2deg(padded_controls[:, 0]).astype(np.float32),
        "control_torque_fl_nm": padded_controls[:, 1].astype(np.float32),
        "control_torque_fr_nm": padded_controls[:, 2].astype(np.float32),
        "control_torque_rl_nm": padded_controls[:, 3].astype(np.float32),
        "control_torque_rr_nm": padded_controls[:, 4].astype(np.float32),
    }
    for index, name in enumerate(STATE_NAMES):
        values = states[:, index]
        if name in {"psi_t", "psi_s"}:
            data[f"{name}_deg"] = np.rad2deg(wrap_angle_error_np(values)).astype(np.float32)
        elif name in {"r_t", "r_s"}:
            data[f"{name}_degps"] = np.rad2deg(values).astype(np.float32)
        else:
            data[name] = values.astype(np.float32)
    return pd.DataFrame(data)


def save_state_csv(
    output_dir: Path,
    time: np.ndarray,
    states: np.ndarray,
    controls: np.ndarray,
    trailer_mass_kg: float,
    dt: float,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "simulation_states.csv"
    build_results_dataframe(time, states, controls, trailer_mass_kg, dt).to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
    )
    return output_path


def plot_trajectory(output_dir: Path, states: np.ndarray) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].plot(states[:, 0], states[:, 1], label="Tractor", linewidth=2.0)
    axes[0].set_title("Tractor Trajectory")
    axes[0].set_xlabel("X (m)")
    axes[0].set_ylabel("Y (m)")
    axes[0].set_aspect("equal", adjustable="box")
    axes[0].grid(True, linestyle="--", alpha=0.35)
    axes[0].legend()

    axes[1].plot(states[:, 6], states[:, 7], label="Trailer", linewidth=2.0)
    axes[1].set_title("Trailer Trajectory")
    axes[1].set_xlabel("X (m)")
    axes[1].set_ylabel("Y (m)")
    axes[1].set_aspect("equal", adjustable="box")
    axes[1].grid(True, linestyle="--", alpha=0.35)
    axes[1].legend()

    output_path = output_dir / "trajectory.png"
    save_figure(fig, output_path)
    return output_path


def plot_state_timeseries(output_dir: Path, time: np.ndarray, states: np.ndarray) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(4, 2, figsize=(15, 12), sharex=True)
    axes = axes.ravel()

    series = [
        ("Tractor Vx", states[:, 3], "m/s"),
        ("Tractor Vy", states[:, 4], "m/s"),
        ("Tractor Yaw Rate", np.rad2deg(states[:, 5]), "deg/s"),
        ("Trailer Vx", states[:, 9], "m/s"),
        ("Trailer Vy", states[:, 10], "m/s"),
        ("Trailer Yaw Rate", np.rad2deg(states[:, 11]), "deg/s"),
        ("Tractor Yaw", np.rad2deg(wrap_angle_error_np(states[:, 2])), "deg"),
        ("Articulation", np.rad2deg(wrap_angle_error_np(states[:, 8] - states[:, 2])), "deg"),
    ]

    for axis, (title, values, unit) in zip(axes, series, strict=False):
        axis.plot(time, values, linewidth=1.8)
        axis.set_title(title)
        axis.set_ylabel(unit)
        axis.grid(True, linestyle="--", alpha=0.35)

    axes[-2].set_xlabel("Time (s)")
    axes[-1].set_xlabel("Time (s)")

    output_path = output_dir / "state_timeseries.png"
    save_figure(fig, output_path)
    return output_path


def save_figure(fig: plt.Figure, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def simulate_trajectory(
    initial_state: np.ndarray,
    control_sequence: np.ndarray,
    trailer_mass_kg: float,
    dt: float,
    total_time: float,
    params: dict[str, float] | None = None,
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray]:
    params = dict(BASE_MODEL_PARAMS if params is None else params)
    if dt <= 0.0:
        raise ValueError(f"dt must be positive, got {dt}")
    if total_time <= 0.0:
        raise ValueError(f"total_time must be positive, got {total_time}")
    if trailer_mass_kg > NO_TRAILER_MASS_THRESHOLD_KG and (
        float(params.get("m_s_base", 0.0)) <= 0.0 or float(params.get("Iz_s_base", 0.0)) <= 0.0
    ):
        raise ValueError(
            "When trailer_mass_kg > 1.0, BASE_MODEL_PARAMS['m_s_base'] and "
            "BASE_MODEL_PARAMS['Iz_s_base'] must both be positive."
        )

    initial_state = np.asarray(initial_state, dtype=np.float32).reshape(-1)
    if initial_state.shape[0] != len(STATE_NAMES):
        raise ValueError(
            f"initial_state length must be {len(STATE_NAMES)}, got {initial_state.shape[0]}"
        )

    step_count = int(round(total_time / dt))
    step_count = max(step_count, 1)
    control_sequence = np.asarray(control_sequence, dtype=np.float32)
    if control_sequence.ndim == 1:
        control_sequence = build_constant_control_sequence(control_sequence, step_count)
    if control_sequence.shape != (step_count, len(CONTROL_NAMES)):
        raise ValueError(
            f"control_sequence shape must be ({step_count}, {len(CONTROL_NAMES)}), "
            f"got {tuple(control_sequence.shape)}"
        )

    model = TruckTrailerNominalDynamics(params).to(device)
    model.eval()

    states = np.zeros((step_count + 1, len(STATE_NAMES)), dtype=np.float32)
    states[0] = initial_state.astype(np.float32)
    times = np.arange(step_count + 1, dtype=np.float32) * float(dt)

    with torch.no_grad():
        for step in range(step_count):
            state_tensor = torch.as_tensor(states[step : step + 1], dtype=torch.float32, device=device)
            control_tensor = torch.as_tensor(control_sequence[step : step + 1], dtype=torch.float32, device=device)
            mass_tensor = torch.as_tensor([[float(trailer_mass_kg)]], dtype=torch.float32, device=device)
            dt_tensor = torch.as_tensor([[float(dt)]], dtype=torch.float32, device=device)

            next_state = model(state_tensor, control_tensor, mass_tensor, dt_tensor).cpu().numpy()[0].astype(np.float32)
            if not np.isfinite(next_state).all():
                raise FloatingPointError(f"Non-finite state encountered at step={step}")
            states[step + 1] = next_state

    return times, states


def run_manual_simulation(config: ManualSimulationConfig) -> dict[str, object]:
    initial_state = np.asarray(config.initial_state, dtype=np.float32)
    constant_control = np.asarray(config.constant_control, dtype=np.float32)
    step_count = max(int(round(config.total_time / config.dt)), 1)
    control_sequence = build_constant_control_sequence(constant_control, step_count)

    times, states = simulate_trajectory(
        initial_state=initial_state,
        control_sequence=control_sequence,
        trailer_mass_kg=float(config.trailer_mass_kg),
        dt=float(config.dt),
        total_time=float(config.total_time),
        params=BASE_MODEL_PARAMS,
        device=config.device,
    )

    csv_path = save_state_csv(
        output_dir=config.output_dir,
        time=times,
        states=states,
        controls=control_sequence,
        trailer_mass_kg=float(config.trailer_mass_kg),
        dt=float(config.dt),
    )
    trajectory_path = plot_trajectory(config.output_dir, states)
    state_plot_path = plot_state_timeseries(config.output_dir, times, states)

    return {
        "time": times,
        "states": states,
        "control_sequence": control_sequence,
        "csv_path": csv_path,
        "trajectory_path": trajectory_path,
        "state_plot_path": state_plot_path,
    }


def main() -> None:
    config = ManualSimulationConfig()
    results = run_manual_simulation(config)

    final_state = results["states"][-1]
    print("Base model simulation finished.")
    print(f"Output dir   : {config.output_dir}")
    print(f"State CSV    : {results['csv_path']}")
    print(f"Trajectory   : {results['trajectory_path']}")
    print(f"State plots  : {results['state_plot_path']}")
    print("Final state  :")
    for name, value in zip(STATE_NAMES, final_state, strict=False):
        if name in {"psi_t", "psi_s"}:
            print(f"  {name:<6}: {float(np.rad2deg(value)):.6f} deg")
        elif name in {"r_t", "r_s"}:
            print(f"  {name:<6}: {float(np.rad2deg(value)):.6f} deg/s")
        else:
            print(f"  {name:<6}: {float(value):.6f}")


if __name__ == "__main__":
    main()
