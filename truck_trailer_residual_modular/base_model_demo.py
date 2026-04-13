from __future__ import annotations

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
    from .constants import BASE_MODEL_OUTPUT_DIR, BASE_MODEL_PARAMS, CONTROL_NAMES, STATE_NAMES
    from .data_utils import save_figure
except ImportError:
    from base_model import TruckTrailerNominalDynamics, wrap_angle_error_np
    from constants import BASE_MODEL_OUTPUT_DIR, BASE_MODEL_PARAMS, CONTROL_NAMES, STATE_NAMES
    from data_utils import save_figure


@dataclass
class ManualSimulationConfig:
    dt: float = 0.02
    total_time: float = 10.0
    trailer_mass_kg: float = 0.0
    device: str = "cpu"
    output_dir: Path = BASE_MODEL_OUTPUT_DIR
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
    constant_control: tuple[float, ...] = (
        0.0,
        0.0,
        0.0,
        1800.0,
        1800.0,
    )


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
        "control_steer_sw_deg": np.rad2deg(padded_controls[:, 0]).astype(np.float32),
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

    initial_state = np.asarray(initial_state, dtype=np.float32).reshape(-1)
    if initial_state.shape[0] != len(STATE_NAMES):
        raise ValueError(f"initial_state length must be {len(STATE_NAMES)}, got {initial_state.shape[0]}")

    step_count = int(round(total_time / dt))
    step_count = max(step_count, 1)
    control_sequence = np.asarray(control_sequence, dtype=np.float32)
    if control_sequence.ndim == 1:
        control_sequence = build_constant_control_sequence(control_sequence, step_count)
    if control_sequence.shape != (step_count, len(CONTROL_NAMES)):
        raise ValueError(
            f"control_sequence shape must be ({step_count}, {len(CONTROL_NAMES)}), got {tuple(control_sequence.shape)}"
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
