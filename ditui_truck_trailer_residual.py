from __future__ import annotations

"""
卡车-挂车残差模型 open-loop 推理脚本。

用途：
1. 读取 `train_truck_trailer_residual.py` 训练得到的 checkpoint
2. 对每个包含卡车/挂车状态的 run 做 open-loop rollout
3. 导出逐步结果 CSV、控制图、轨迹图、误差图和整体 summary
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

try:
    from controltest import train_truck_trailer_residual as base_train
except ImportError:
    import train_truck_trailer_residual as base_train


RUNS_ROOT = base_train.RUNS_ROOT
MODEL_CHECKPOINT = Path(__file__).with_name("best_truck_trailer_error_model.pth")
ALT_MODEL_CHECKPOINT = Path(__file__).with_name("best_truck_trailer_error_model_train_loss.pth")

MLP_TORCH_DTYPE = base_train.MLP_TORCH_DTYPE
MLP_NUMPY_DTYPE = base_train.MLP_NUMPY_DTYPE
STATE_NAMES = list(base_train.STATE_NAMES)
MOTION_ERROR_NAMES = list(base_train.MOTION_ERROR_NAMES)
CONTROL_NAMES = list(base_train.CONTROL_NAMES)


@dataclass
class InferenceSegment:
    csv_path: Path
    segment_name: str
    out_dir: Path
    time: np.ndarray
    dt_values: np.ndarray
    real_rollout: np.ndarray
    initial_state: np.ndarray
    control_sequence: np.ndarray
    trailer_mass_kg: np.ndarray


def save_figure(fig: plt.Figure, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def pick_checkpoint_path() -> Path:
    if MODEL_CHECKPOINT.exists():
        return MODEL_CHECKPOINT
    if ALT_MODEL_CHECKPOINT.exists():
        return ALT_MODEL_CHECKPOINT
    raise FileNotFoundError(
        f"未找到 checkpoint: {MODEL_CHECKPOINT} 或 {ALT_MODEL_CHECKPOINT}"
    )


def infer_model_dims_from_state_dict(state_dict: dict[str, torch.Tensor]) -> tuple[int, int]:
    linear_weights = [value for key, value in state_dict.items() if key.endswith(".weight") and value.ndim == 2]
    if not linear_weights:
        raise ValueError("Checkpoint does not contain linear-layer weights.")
    return int(linear_weights[0].shape[1]), int(linear_weights[-1].shape[0])


def infer_layer_norm_from_state_dict(state_dict: dict[str, torch.Tensor]) -> bool:
    return "network.1.weight" in state_dict and "network.1.bias" in state_dict


def split_checkpoint_payload(payload: object) -> tuple[dict[str, torch.Tensor], dict[str, object]]:
    if isinstance(payload, dict) and "state_dict" in payload:
        metadata: dict[str, object] = {}
        for key in (
            "feature_mean",
            "feature_scale",
            "loss_error_scale",
            "loss_pose_error_scale",
            "loss_motion_error_scale",
            "model_input_dim",
            "model_output_dim",
            "mlp_use_layer_norm",
            "input_feature_names",
            "motion_error_names",
            "state_names",
            "control_names",
            "base_model_params",
        ):
            if key in payload:
                metadata[key] = payload[key]
        return payload["state_dict"], metadata
    if isinstance(payload, dict):
        return payload, {}
    raise TypeError(f"Unsupported checkpoint payload type: {type(payload)!r}")


def extract_feature_context(metadata: dict[str, object]) -> dict[str, np.ndarray] | None:
    if "feature_mean" not in metadata or "feature_scale" not in metadata:
        return None
    return {
        "feature_mean": np.asarray(metadata["feature_mean"], dtype=MLP_NUMPY_DTYPE).reshape(-1),
        "feature_scale": np.asarray(metadata["feature_scale"], dtype=MLP_NUMPY_DTYPE).reshape(-1),
    }


def extract_motion_clip(metadata: dict[str, object]) -> np.ndarray | None:
    motion_scale = metadata.get("loss_motion_error_scale")
    if motion_scale is None:
        return None
    return 3.0 * np.asarray(motion_scale, dtype=MLP_NUMPY_DTYPE).reshape(-1)


def extract_input_feature_names(metadata: dict[str, object], input_dim: int) -> list[str]:
    raw_names = metadata.get("input_feature_names")
    if raw_names is None:
        names = list(base_train.MLP_INPUT_FEATURE_NAMES)
    else:
        names = [str(name) for name in np.asarray(raw_names).reshape(-1).tolist()]
    if len(names) != input_dim:
        raise ValueError(f"Checkpoint input_feature_names length={len(names)} does not match input_dim={input_dim}.")
    expected_names = list(base_train.MLP_INPUT_FEATURE_NAMES)
    if names != expected_names:
        raise ValueError(
            "Checkpoint input_feature_names 与当前 truck_trailer 输入定义不一致。"
            f" checkpoint={names}, expected={expected_names}。请使用前轮转角输入重新训练 checkpoint。"
        )
    return names


def load_error_model(device: torch.device) -> tuple[base_train.MLPErrorModel, dict[str, object], Path]:
    checkpoint_path = pick_checkpoint_path()
    payload = torch.load(checkpoint_path, map_location=device)
    state_dict, metadata = split_checkpoint_payload(payload)
    input_dim, output_dim = infer_model_dims_from_state_dict(state_dict)
    input_dim = int(metadata.get("model_input_dim", input_dim))
    output_dim = int(metadata.get("model_output_dim", output_dim))
    output_names_raw = metadata.get("motion_error_names")
    if output_names_raw is not None:
        output_names = [str(name) for name in np.asarray(output_names_raw).reshape(-1).tolist()]
        if output_names != MOTION_ERROR_NAMES:
            raise ValueError(f"Checkpoint output names={output_names}, expected {MOTION_ERROR_NAMES}.")
    if output_dim != len(MOTION_ERROR_NAMES):
        raise ValueError(f"Checkpoint output_dim={output_dim}, expected {len(MOTION_ERROR_NAMES)}.")

    use_layer_norm = bool(metadata.get("mlp_use_layer_norm", infer_layer_norm_from_state_dict(state_dict)))
    metadata["input_feature_names"] = extract_input_feature_names(metadata, input_dim)
    model = base_train.MLPErrorModel(
        input_dim=input_dim,
        output_dim=output_dim,
        use_layer_norm=use_layer_norm,
    ).to(device=device, dtype=MLP_TORCH_DTYPE)
    model.load_state_dict(state_dict)
    model.eval()
    return model, metadata, checkpoint_path


def build_base_model(metadata: dict[str, object], device: torch.device) -> base_train.TruckTrailerNominalDynamics:
    params = dict(base_train.BASE_MODEL_PARAMS)
    base_model_params = metadata.get("base_model_params")
    if isinstance(base_model_params, dict):
        params.update({str(key): float(value) for key, value in base_model_params.items()})
    model = base_train.TruckTrailerNominalDynamics(params).to(device)
    model.eval()
    return model


def load_segment(csv_path: Path) -> InferenceSegment:
    seg = base_train.load_truck_trailer_data_as_segment(csv_path)
    out_dir = csv_path.parent / "truck_trailer_open_loop_eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    return InferenceSegment(
        csv_path=seg.csv_path,
        segment_name=seg.segment_name,
        out_dir=out_dir,
        time=seg.time,
        dt_values=seg.dt_values,
        real_rollout=seg.real_rollout,
        initial_state=seg.initial_state,
        control_sequence=seg.control_sequence,
        trailer_mass_kg=seg.trailer_mass_kg,
    )


def find_all_csvs(runs_root: Path) -> list[Path]:
    return base_train.find_all_real_data_csvs(runs_root)


def rollout_open_loop(
    base_model: base_train.TruckTrailerNominalDynamics,
    error_model: base_train.MLPErrorModel,
    initial_state: np.ndarray,
    control_sequence: np.ndarray,
    trailer_mass_kg: np.ndarray,
    dt_values: np.ndarray,
    device: torch.device,
    feature_context: dict[str, np.ndarray] | None,
    motion_error_clip: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    step_count = len(control_sequence) + 1
    base_rollout = np.zeros((step_count, len(STATE_NAMES)), dtype=np.float32)
    corr_rollout = np.zeros((step_count, len(STATE_NAMES)), dtype=np.float32)
    base_rollout[0] = initial_state.astype(np.float32)
    corr_rollout[0] = initial_state.astype(np.float32)

    feature_context_tensors = None
    if feature_context is not None:
        feature_context_tensors = base_train.build_feature_context_tensors(feature_context, device)

    for step in range(len(control_sequence)):
        dt = np.array([[dt_values[step]]], dtype=np.float32)
        u = control_sequence[step : step + 1].astype(np.float32)
        mass = np.array([[trailer_mass_kg[step]]], dtype=np.float32)

        state_base = base_train.to_tensor(base_rollout[step : step + 1], device)
        state_corr = base_train.to_tensor(corr_rollout[step : step + 1], device)
        control_tensor = base_train.to_tensor(u, device)
        mass_tensor = base_train.to_tensor(mass, device)
        dt_tensor = base_train.to_tensor(dt, device)

        base_next = base_model(state_base, control_tensor, mass_tensor, dt_tensor).cpu().numpy().astype(np.float32)[0]
        if not np.isfinite(base_next).all():
            raise FloatingPointError(f"Base rollout produced non-finite state at step={step} in base path.")
        base_rollout[step + 1] = base_next

        if not np.isfinite(corr_rollout[step]).all():
            corr_rollout[step] = base_rollout[step].copy()
            state_corr = base_train.to_tensor(corr_rollout[step : step + 1], device)

        corr_base_next = base_model(state_corr, control_tensor, mass_tensor, dt_tensor).cpu().numpy().astype(np.float32)
        features = base_train.build_mlp_input_feature_tensor(state_corr, control_tensor, mass_tensor, dt_tensor)
        if feature_context_tensors is not None:
            features = base_train.normalize_feature_tensor(features, feature_context_tensors)
        predicted_motion_error = error_model(features).detach().cpu().numpy()[0].astype(np.float32)
        if motion_error_clip is not None:
            predicted_motion_error = np.clip(predicted_motion_error, -motion_error_clip, motion_error_clip)

        corrected_error = base_train.derive_full_error_from_motion_error_np(
            predicted_motion_error.reshape(1, -1),
            corr_base_next,
            np.array([dt_values[step]], dtype=np.float32),
        )[0]
        corr_next = corr_base_next[0] + corrected_error
        corr_next[2] = base_train.wrap_angle_error_np(np.asarray([corr_next[2]], dtype=np.float32))[0]
        corr_next[8] = base_train.wrap_angle_error_np(np.asarray([corr_next[8]], dtype=np.float32))[0]
        if not np.isfinite(corr_next).all():
            corr_next = corr_base_next[0].copy()
        corr_rollout[step + 1] = corr_next.astype(np.float32)

    return base_rollout, corr_rollout


def pad_control_series(values: np.ndarray) -> np.ndarray:
    if len(values) == 0:
        return np.zeros((1,), dtype=np.float32)
    return np.concatenate([values.astype(np.float32), values[-1:].astype(np.float32)], axis=0)


def compute_state_error_series(predicted_states: np.ndarray, reference_states: np.ndarray, state_name: str) -> np.ndarray:
    index = STATE_NAMES.index(state_name)
    if state_name in {"psi_t", "psi_s"}:
        return np.rad2deg(base_train.wrap_angle_error_np(predicted_states[:, index] - reference_states[:, index]))
    if state_name in {"r_t", "r_s"}:
        return np.rad2deg(predicted_states[:, index] - reference_states[:, index])
    return predicted_states[:, index] - reference_states[:, index]


def compute_articulation_series(states: np.ndarray) -> np.ndarray:
    return base_train.compute_articulation_series(states)


def build_open_loop_results_dataframe(
    seg: InferenceSegment,
    base_rollout: np.ndarray,
    corr_rollout: np.ndarray,
) -> pd.DataFrame:
    time = seg.time.astype(np.float32)
    delta_f_deg = np.rad2deg(pad_control_series(seg.control_sequence[:, 0]))
    torque_fl = pad_control_series(seg.control_sequence[:, 1])
    torque_fr = pad_control_series(seg.control_sequence[:, 2])
    torque_rl = pad_control_series(seg.control_sequence[:, 3])
    torque_rr = pad_control_series(seg.control_sequence[:, 4])
    trailer_mass = pad_control_series(seg.trailer_mass_kg)
    dt_series = pad_control_series(seg.dt_values)

    df = pd.DataFrame(
        {
            "time_s": time,
            "dt_s": dt_series,
            "delta_f_deg": delta_f_deg,
            "torque_fl_nm": torque_fl,
            "torque_fr_nm": torque_fr,
            "torque_rl_nm": torque_rl,
            "torque_rr_nm": torque_rr,
            "trailer_mass_kg": trailer_mass,
        }
    )

    for prefix, rollout in (("real", seg.real_rollout), ("base", base_rollout), ("corr", corr_rollout)):
        for index, name in enumerate(STATE_NAMES):
            values = rollout[:, index]
            if name in {"psi_t", "psi_s"}:
                df[f"{prefix}_{name}_deg"] = np.rad2deg(base_train.wrap_angle_error_np(values))
            elif name in {"r_t", "r_s"}:
                df[f"{prefix}_{name}_degps"] = np.rad2deg(values)
            else:
                df[f"{prefix}_{name}"] = values
        df[f"{prefix}_articulation_deg"] = compute_articulation_series(rollout)

    plot_error_names = STATE_NAMES + ["articulation"]
    for name in plot_error_names:
        if name == "articulation":
            err_base = compute_articulation_series(base_rollout) - compute_articulation_series(seg.real_rollout)
            err_corr = compute_articulation_series(corr_rollout) - compute_articulation_series(seg.real_rollout)
        else:
            err_base = compute_state_error_series(base_rollout, seg.real_rollout, name)
            err_corr = compute_state_error_series(corr_rollout, seg.real_rollout, name)
        df[f"err_base_{name}"] = err_base.astype(np.float32)
        df[f"err_corr_{name}"] = err_corr.astype(np.float32)

    return df


def compute_rmse_summary(seg: InferenceSegment, base_rollout: np.ndarray, corr_rollout: np.ndarray) -> dict[str, float]:
    summary: dict[str, float] = {
        "segment_name": seg.segment_name,
        "csv_path": str(seg.csv_path),
        "sample_count": int(len(seg.real_rollout)),
        "mean_trailer_mass_kg": float(np.mean(seg.trailer_mass_kg)),
    }
    for name in STATE_NAMES:
        err_base = compute_state_error_series(base_rollout, seg.real_rollout, name).astype(np.float64)
        err_corr = compute_state_error_series(corr_rollout, seg.real_rollout, name).astype(np.float64)
        summary[f"rmse_base_{name}"] = float(np.sqrt(np.mean(err_base * err_base)))
        summary[f"rmse_corr_{name}"] = float(np.sqrt(np.mean(err_corr * err_corr)))
    err_base_art = (compute_articulation_series(base_rollout) - compute_articulation_series(seg.real_rollout)).astype(np.float64)
    err_corr_art = (compute_articulation_series(corr_rollout) - compute_articulation_series(seg.real_rollout)).astype(np.float64)
    summary["rmse_base_articulation_deg"] = float(np.sqrt(np.mean(err_base_art * err_base_art)))
    summary["rmse_corr_articulation_deg"] = float(np.sqrt(np.mean(err_corr_art * err_corr_art)))
    return summary


def export_results_csv(seg: InferenceSegment, base_rollout: np.ndarray, corr_rollout: np.ndarray) -> Path:
    out_path = seg.out_dir / "open_loop_results.csv"
    build_open_loop_results_dataframe(seg, base_rollout, corr_rollout).to_csv(out_path, index=False, encoding="utf-8-sig")
    return out_path


def plot_controls(seg: InferenceSegment) -> Path:
    time = seg.time[:-1]
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    axes[0].plot(time, np.rad2deg(seg.control_sequence[:, 0]), label="Front Wheel", linewidth=1.8)
    axes[0].set_ylabel("deg")
    axes[0].set_title("Front Wheel Angle")
    axes[0].grid(True, linestyle="--", alpha=0.35)
    axes[0].legend()

    axes[1].plot(time, seg.control_sequence[:, 1], label="FL", linewidth=1.5)
    axes[1].plot(time, seg.control_sequence[:, 2], label="FR", linewidth=1.5)
    axes[1].set_ylabel("Nm")
    axes[1].set_title("Front Axle Torques")
    axes[1].grid(True, linestyle="--", alpha=0.35)
    axes[1].legend()

    axes[2].plot(time, seg.control_sequence[:, 3], label="RL", linewidth=1.5)
    axes[2].plot(time, seg.control_sequence[:, 4], label="RR", linewidth=1.5)
    axes[2].set_ylabel("Nm")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_title("Rear Axle Torques")
    axes[2].grid(True, linestyle="--", alpha=0.35)
    axes[2].legend()

    out_path = seg.out_dir / "controls.png"
    save_figure(fig, out_path)
    return out_path


def plot_trajectory(seg: InferenceSegment, base_rollout: np.ndarray, corr_rollout: np.ndarray) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].plot(seg.real_rollout[:, 0], seg.real_rollout[:, 1], label="CarSim/TruckSim", linewidth=1.9)
    axes[0].plot(base_rollout[:, 0], base_rollout[:, 1], label="Base OpenLoop", linewidth=1.6)
    axes[0].plot(corr_rollout[:, 0], corr_rollout[:, 1], label="Base+NN OpenLoop", linewidth=1.7)
    axes[0].set_title("Tractor Trajectory")
    axes[0].set_xlabel("X (m)")
    axes[0].set_ylabel("Y (m)")
    axes[0].grid(True, linestyle="--", alpha=0.35)
    axes[0].legend()
    axes[0].set_aspect("equal", adjustable="box")

    axes[1].plot(seg.real_rollout[:, 6], seg.real_rollout[:, 7], label="CarSim/TruckSim", linewidth=1.9)
    axes[1].plot(base_rollout[:, 6], base_rollout[:, 7], label="Base OpenLoop", linewidth=1.6)
    axes[1].plot(corr_rollout[:, 6], corr_rollout[:, 7], label="Base+NN OpenLoop", linewidth=1.7)
    axes[1].set_title("Trailer Trajectory")
    axes[1].set_xlabel("X (m)")
    axes[1].set_ylabel("Y (m)")
    axes[1].grid(True, linestyle="--", alpha=0.35)
    axes[1].legend()
    axes[1].set_aspect("equal", adjustable="box")

    out_path = seg.out_dir / "open_loop_trajectory.png"
    save_figure(fig, out_path)
    return out_path


def plot_state_error_all(seg: InferenceSegment, base_rollout: np.ndarray, corr_rollout: np.ndarray) -> Path:
    plot_names = [
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
        "articulation",
    ]
    fig, axes = plt.subplots(5, 3, figsize=(16, 16), sharex=True)
    axes = axes.ravel()

    for axis, name in zip(axes, plot_names, strict=False):
        if name == "articulation":
            err_base = compute_articulation_series(base_rollout) - compute_articulation_series(seg.real_rollout)
            err_corr = compute_articulation_series(corr_rollout) - compute_articulation_series(seg.real_rollout)
            ylabel = "deg"
        else:
            err_base = compute_state_error_series(base_rollout, seg.real_rollout, name)
            err_corr = compute_state_error_series(corr_rollout, seg.real_rollout, name)
            ylabel = "deg" if name in {"psi_t", "psi_s", "r_t", "r_s"} else "m/mps"
        axis.plot(seg.time, err_base, label="Base OpenLoop - Real", linewidth=1.4)
        axis.plot(seg.time, err_corr, label="Base+NN OpenLoop - Real", linewidth=1.6)
        axis.axhline(0.0, color="black", linewidth=0.8, linestyle=":")
        axis.set_title(f"{name} error")
        axis.set_ylabel(ylabel)
        axis.grid(True, linestyle="--", alpha=0.35)

    for axis in axes[len(plot_names):]:
        axis.axis("off")

    axes[-1].set_xlabel("Time (s)")
    axes[0].legend(loc="best")

    out_path = seg.out_dir / "open_loop_state_errors.png"
    save_figure(fig, out_path)
    return out_path


def export_summary_csv(rows: list[dict[str, float]], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False, encoding="utf-8-sig")
    return output_path


def main() -> None:
    torch.manual_seed(42)
    np.random.seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device = {device}")

    csvs = find_all_csvs(RUNS_ROOT)
    if not csvs:
        raise FileNotFoundError(f"未找到任何数据: {RUNS_ROOT}\\python_run_*\\outputs\\control_and_trajectory.csv")
    print(f"发现 {len(csvs)} 段候选数据，最新: {csvs[0]}")

    error_model, checkpoint_metadata, checkpoint_path = load_error_model(device)
    print(f"已加载误差模型: {checkpoint_path}")

    base_model = build_base_model(checkpoint_metadata, device)
    feature_context = extract_feature_context(checkpoint_metadata)
    motion_error_clip = extract_motion_clip(checkpoint_metadata)
    if feature_context is None:
        print("Checkpoint 缺少特征归一化统计量，将使用原始特征。")
    if motion_error_clip is None:
        print("Checkpoint 缺少 motion scale，关闭单步残差 clip。")

    summary_rows: list[dict[str, float]] = []
    processed_count = 0

    for csv_path in csvs:
        try:
            seg = load_segment(csv_path)
        except Exception as exc:
            print(f"[跳过] {csv_path} | {exc}")
            continue

        base_rollout, corr_rollout = rollout_open_loop(
            base_model=base_model,
            error_model=error_model,
            initial_state=seg.initial_state,
            control_sequence=seg.control_sequence,
            trailer_mass_kg=seg.trailer_mass_kg,
            dt_values=seg.dt_values,
            device=device,
            feature_context=feature_context,
            motion_error_clip=motion_error_clip,
        )

        result_csv = export_results_csv(seg, base_rollout, corr_rollout)
        controls_png = plot_controls(seg)
        traj_png = plot_trajectory(seg, base_rollout, corr_rollout)
        errors_png = plot_state_error_all(seg, base_rollout, corr_rollout)
        summary = compute_rmse_summary(seg, base_rollout, corr_rollout)
        summary_rows.append(summary)
        processed_count += 1

        print(f"\n[OK] {seg.segment_name}")
        print(f"  out_dir : {seg.out_dir}")
        print(f"  results : {result_csv.name}")
        print(f"  controls: {controls_png.name}")
        print(f"  traj    : {traj_png.name}")
        print(f"  errors  : {errors_png.name}")
        print(
            "  articulation rmse (deg): "
            f"base={summary['rmse_base_articulation_deg']:.4f}, "
            f"corr={summary['rmse_corr_articulation_deg']:.4f}"
        )

    if processed_count == 0:
        raise RuntimeError(
            "没有任何段成功完成推理。大概率是当前导出文件还缺少挂车状态列。"
        )

    summary_path = export_summary_csv(
        summary_rows,
        RUNS_ROOT / "truck_trailer_open_loop_summary.csv",
    )
    print(f"\nsummary csv: {summary_path}")
    print(f"完成 {processed_count} 段 open-loop 推理。")


if __name__ == "__main__":
    main()
