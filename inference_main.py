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
    from .constants import (
        BASE_MODEL_PARAMS,
        MODEL_CHECKPOINT,
        MLP_DROPOUT_P,
        MLP_HIDDEN_DIM,
        MLP_HIDDEN_LAYERS,
        MLP_INPUT_FEATURE_NAMES,
        MLP_OUTPUT_NAMES,
        MLP_NUMPY_DTYPE,
        MLP_TORCH_DTYPE,
        RUNS_ROOT,
        STATE_NAMES,
        TRACTOR_STATE_REFERENCE,
        TRAIN_LOSS_MODEL_CHECKPOINT,
    )
    from .data_utils import (
        collect_control_and_trajectory_csvs,
        build_feature_context_tensors,
        build_mlp_input_feature_tensor,
        compute_articulation_series,
        derive_full_error_from_mlp_output_np,
        find_all_real_data_csvs,
        load_truck_trailer_data_as_segment,
        normalize_feature_tensor,
        save_figure,
        to_tensor,
    )
    from .model_structure import MLPErrorModel
except ImportError:
    from base_model import TruckTrailerNominalDynamics, wrap_angle_error_np
    from constants import (
        BASE_MODEL_PARAMS,
        MODEL_CHECKPOINT,
        MLP_DROPOUT_P,
        MLP_HIDDEN_DIM,
        MLP_HIDDEN_LAYERS,
        MLP_INPUT_FEATURE_NAMES,
        MLP_OUTPUT_NAMES,
        MLP_NUMPY_DTYPE,
        MLP_TORCH_DTYPE,
        RUNS_ROOT,
        STATE_NAMES,
        TRACTOR_STATE_REFERENCE,
        TRAIN_LOSS_MODEL_CHECKPOINT,
    )
    from data_utils import (
        collect_control_and_trajectory_csvs,
        build_feature_context_tensors,
        build_mlp_input_feature_tensor,
        compute_articulation_series,
        derive_full_error_from_mlp_output_np,
        find_all_real_data_csvs,
        load_truck_trailer_data_as_segment,
        normalize_feature_tensor,
        save_figure,
        to_tensor,
    )
    from model_structure import MLPErrorModel


@dataclass
class InferenceSegment:
    csv_path: Path
    scenario_name: str
    segment_name: str
    out_dir: Path
    time: np.ndarray
    dt_values: np.ndarray
    real_rollout: np.ndarray
    initial_state: np.ndarray
    control_sequence: np.ndarray
    trailer_mass_kg: np.ndarray


REAL_LABEL = "Real"
BASE_LABEL = "Base"
NN_LABEL = "Base + NN"
MODE_TITLES = {
    "single_step": "Single-Step",
    "recursive_rollout": "Recursive Rollout",
}
STATE_PLOT_META: dict[str, tuple[str, str]] = {
    "x_t": ("Tractor Rear-Axle X", "m"),
    "y_t": ("Tractor Rear-Axle Y", "m"),
    "psi_t": ("Tractor Rear-Axle Yaw", "deg"),
    "vx_t": ("Tractor Rear-Axle Vx", "m/s"),
    "vy_t": ("Tractor Rear-Axle Vy", "m/s"),
    "r_t": ("Tractor Yaw Rate", "deg/s"),
    "x_s": ("Trailer X", "m"),
    "y_s": ("Trailer Y", "m"),
    "psi_s": ("Trailer Yaw", "deg"),
    "vx_s": ("Trailer Vx", "m/s"),
    "vy_s": ("Trailer Vy", "m/s"),
    "r_s": ("Trailer Yaw Rate", "deg/s"),
    "articulation": ("Articulation", "deg"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run open-loop inference for the truck-trailer residual model.",
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
        help="Optional checkpoint .pth path, run directory, or checkpoint directory. "
        "If omitted, inference_main.py falls back to the default checkpoint locations.",
    )
    return parser.parse_args()


def resolve_checkpoint_path_from_input(checkpoint_path: Path) -> Path:
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint path does not exist: {checkpoint_path}")

    if checkpoint_path.is_file():
        return checkpoint_path

    candidate_files = [
        checkpoint_path / MODEL_CHECKPOINT.name,
        checkpoint_path / TRAIN_LOSS_MODEL_CHECKPOINT.name,
        checkpoint_path / "checkpoints" / MODEL_CHECKPOINT.name,
        checkpoint_path / "checkpoints" / TRAIN_LOSS_MODEL_CHECKPOINT.name,
    ]
    for candidate in candidate_files:
        if candidate.exists() and candidate.is_file():
            return candidate

    pth_files = sorted(checkpoint_path.rglob("*.pth"), key=lambda path: path.stat().st_mtime, reverse=True)
    if pth_files:
        return pth_files[0]

    raise FileNotFoundError(
        "Could not find a .pth checkpoint under the specified checkpoint path. "
        f"Tried common candidates under: {checkpoint_path}"
    )


def pick_checkpoint_path(checkpoint_path: Path | None = None) -> Path:
    if checkpoint_path is not None:
        return resolve_checkpoint_path_from_input(checkpoint_path)
    if MODEL_CHECKPOINT.exists():
        return MODEL_CHECKPOINT
    if TRAIN_LOSS_MODEL_CHECKPOINT.exists():
        return TRAIN_LOSS_MODEL_CHECKPOINT
    legacy_model_checkpoint = MODEL_CHECKPOINT.parent.parent / MODEL_CHECKPOINT.name
    legacy_train_loss_checkpoint = TRAIN_LOSS_MODEL_CHECKPOINT.parent.parent / TRAIN_LOSS_MODEL_CHECKPOINT.name
    if legacy_model_checkpoint.exists():
        return legacy_model_checkpoint
    if legacy_train_loss_checkpoint.exists():
        return legacy_train_loss_checkpoint
    raise FileNotFoundError(
        "Checkpoint not found. Run train_main.py first, or place an existing truck-trailer residual checkpoint in "
        f"{MODEL_CHECKPOINT.parent}."
    )


def infer_model_dims_from_state_dict(state_dict: dict[str, torch.Tensor]) -> tuple[int, int]:
    linear_weights = [value for key, value in state_dict.items() if key.endswith(".weight") and value.ndim == 2]
    if not linear_weights:
        raise ValueError("Checkpoint does not contain linear layer weights.")
    return int(linear_weights[0].shape[1]), int(linear_weights[-1].shape[0])


def infer_hidden_config_from_state_dict(state_dict: dict[str, torch.Tensor]) -> tuple[int, int]:
    linear_weights = [value for key, value in state_dict.items() if key.endswith(".weight") and value.ndim == 2]
    if len(linear_weights) < 2:
        return MLP_HIDDEN_DIM, MLP_HIDDEN_LAYERS
    hidden_dim = int(linear_weights[0].shape[0])
    hidden_layers = int(len(linear_weights) - 1)
    return hidden_dim, hidden_layers


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
            "loss_output_scale",
            "model_input_dim",
            "model_output_dim",
            "mlp_use_layer_norm",
            "mlp_hidden_dim",
            "mlp_hidden_layers",
            "mlp_dropout_p",
            "input_feature_names",
            "mlp_control_feature_names",
            "mlp_output_names",
            "motion_error_names",
            "state_names",
            "control_names",
            "base_model_params",
            "tractor_state_reference",
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


def extract_output_clip(metadata: dict[str, object]) -> np.ndarray | None:
    output_scale = metadata.get("loss_output_scale")
    if output_scale is None:
        return None
    return 3.0 * np.asarray(output_scale, dtype=MLP_NUMPY_DTYPE).reshape(-1)


def extract_input_feature_names(metadata: dict[str, object], input_dim: int) -> list[str]:
    raw_names = metadata.get("input_feature_names")
    if raw_names is None:
        return []
    names = [str(name) for name in np.asarray(raw_names).reshape(-1).tolist()]
    if len(names) != input_dim:
        raise ValueError(f"Checkpoint input_feature_names length={len(names)} does not match input_dim={input_dim}.")
    return names


def load_error_model(device: torch.device, checkpoint_path: Path | None = None) -> tuple[MLPErrorModel, dict[str, object], Path]:
    checkpoint_path = pick_checkpoint_path(checkpoint_path)
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict, metadata = split_checkpoint_payload(payload)
    input_dim, output_dim = infer_model_dims_from_state_dict(state_dict)
    input_dim = int(metadata.get("model_input_dim", input_dim))
    output_dim = int(metadata.get("model_output_dim", output_dim))
    if input_dim != len(MLP_INPUT_FEATURE_NAMES):
        raise ValueError(
            f"Checkpoint input_dim={input_dim}, expected {len(MLP_INPUT_FEATURE_NAMES)} for the relative-pose "
            "truck-trailer residual MLP. Retrain train_main.py to create a compatible checkpoint."
        )
    output_names_raw = metadata.get("mlp_output_names", metadata.get("motion_error_names"))
    if output_names_raw is not None:
        output_names = [str(name) for name in np.asarray(output_names_raw).reshape(-1).tolist()]
        if output_names != MLP_OUTPUT_NAMES:
            raise ValueError(
                f"Checkpoint output names={output_names}, expected {MLP_OUTPUT_NAMES} for the relative-pose "
                "truck-trailer residual MLP. Retrain train_main.py to create a compatible checkpoint."
            )
    if output_dim != len(MLP_OUTPUT_NAMES):
        raise ValueError(
            f"Checkpoint output_dim={output_dim}, expected {len(MLP_OUTPUT_NAMES)} for the relative-pose "
            "truck-trailer residual MLP. Retrain train_main.py to create a compatible checkpoint."
        )
    tractor_state_reference = metadata.get("tractor_state_reference")
    if tractor_state_reference != TRACTOR_STATE_REFERENCE:
        raise ValueError(
            f"Checkpoint tractor_state_reference={tractor_state_reference!r}, expected {TRACTOR_STATE_REFERENCE!r}. "
            "Retrain train_main.py to create a checkpoint compatible with tractor rear-axle-center states."
        )

    use_layer_norm = bool(metadata.get("mlp_use_layer_norm", infer_layer_norm_from_state_dict(state_dict)))
    inferred_hidden_dim, inferred_hidden_layers = infer_hidden_config_from_state_dict(state_dict)
    hidden_dim = int(metadata.get("mlp_hidden_dim", inferred_hidden_dim))
    hidden_layers = int(metadata.get("mlp_hidden_layers", inferred_hidden_layers))
    dropout_p = float(metadata.get("mlp_dropout_p", MLP_DROPOUT_P))
    metadata["input_feature_names"] = extract_input_feature_names(metadata, input_dim)
    if metadata["input_feature_names"] and metadata["input_feature_names"] != MLP_INPUT_FEATURE_NAMES:
        raise ValueError(
            f"Checkpoint input features={metadata['input_feature_names']}, expected {MLP_INPUT_FEATURE_NAMES} "
            "for the relative-pose truck-trailer residual MLP. Retrain train_main.py to create a compatible checkpoint."
        )
    model = MLPErrorModel(
        input_dim=input_dim,
        output_dim=output_dim,
        dropout_p=dropout_p,
        use_layer_norm=use_layer_norm,
        hidden_dim=hidden_dim,
        hidden_layers=hidden_layers,
    ).to(device=device, dtype=MLP_TORCH_DTYPE)
    model.load_state_dict(state_dict)
    model.eval()
    return model, metadata, checkpoint_path


def build_base_model(metadata: dict[str, object], device: torch.device) -> TruckTrailerNominalDynamics:
    params = dict(BASE_MODEL_PARAMS)
    base_model_params = metadata.get("base_model_params")
    if isinstance(base_model_params, dict):
        params.update({str(key): float(value) for key, value in base_model_params.items()})
    model = TruckTrailerNominalDynamics(params).to(device)
    model.eval()
    return model


def resolve_output_dir(csv_path: Path) -> Path:
    out_dir = csv_path.parent / f"{csv_path.stem}_inference_eval_modular"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def save_inference_figure(fig: plt.Figure, output_path: Path, top_margin: float = 1.0) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, top_margin))
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def load_segment(csv_path: Path) -> InferenceSegment:
    seg = load_truck_trailer_data_as_segment(csv_path)
    out_dir = resolve_output_dir(csv_path)
    return InferenceSegment(
        csv_path=seg.csv_path,
        scenario_name=seg.segment_name,
        segment_name=csv_path.stem,
        out_dir=out_dir,
        time=seg.time,
        dt_values=seg.dt_values,
        real_rollout=seg.real_rollout,
        initial_state=seg.initial_state,
        control_sequence=seg.control_sequence,
        trailer_mass_kg=seg.trailer_mass_kg,
    )


@torch.no_grad()
def predict_base_and_nn_next_from_state(
    base_model: TruckTrailerNominalDynamics,
    error_model: MLPErrorModel,
    current_state: np.ndarray,
    control_step: np.ndarray,
    trailer_mass_kg_step: float,
    dt_value: float,
    device: torch.device,
    feature_context_tensors: dict[str, torch.Tensor] | None,
    mlp_output_clip: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
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
    nn_next = base_next[0] + corrected_error
    nn_next[2] = wrap_angle_error_np(np.asarray([nn_next[2]], dtype=np.float32))[0]
    nn_next[8] = wrap_angle_error_np(np.asarray([nn_next[8]], dtype=np.float32))[0]
    if not np.isfinite(nn_next).all():
        nn_next = base_next[0].copy()

    return base_next[0], nn_next.astype(np.float32)


def rollout_single_step(
    base_model: TruckTrailerNominalDynamics,
    error_model: MLPErrorModel,
    real_rollout: np.ndarray,
    control_sequence: np.ndarray,
    trailer_mass_kg: np.ndarray,
    dt_values: np.ndarray,
    device: torch.device,
    feature_context_tensors: dict[str, torch.Tensor] | None,
    mlp_output_clip: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    step_count = len(control_sequence) + 1
    base_rollout = np.zeros((step_count, len(STATE_NAMES)), dtype=np.float32)
    nn_rollout = np.zeros((step_count, len(STATE_NAMES)), dtype=np.float32)
    base_rollout[0] = real_rollout[0].astype(np.float32)
    nn_rollout[0] = real_rollout[0].astype(np.float32)

    for step in range(len(control_sequence)):
        base_next, nn_next = predict_base_and_nn_next_from_state(
            base_model=base_model,
            error_model=error_model,
            current_state=real_rollout[step],
            control_step=control_sequence[step],
            trailer_mass_kg_step=float(trailer_mass_kg[step]),
            dt_value=float(dt_values[step]),
            device=device,
            feature_context_tensors=feature_context_tensors,
            mlp_output_clip=mlp_output_clip,
        )
        base_rollout[step + 1] = base_next
        nn_rollout[step + 1] = nn_next

    return base_rollout, nn_rollout


def rollout_recursive(
    base_model: TruckTrailerNominalDynamics,
    error_model: MLPErrorModel,
    initial_state: np.ndarray,
    control_sequence: np.ndarray,
    trailer_mass_kg: np.ndarray,
    dt_values: np.ndarray,
    device: torch.device,
    feature_context_tensors: dict[str, torch.Tensor] | None,
    mlp_output_clip: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    step_count = len(control_sequence) + 1
    base_rollout = np.zeros((step_count, len(STATE_NAMES)), dtype=np.float32)
    nn_rollout = np.zeros((step_count, len(STATE_NAMES)), dtype=np.float32)
    base_rollout[0] = initial_state.astype(np.float32)
    nn_rollout[0] = initial_state.astype(np.float32)

    for step in range(len(control_sequence)):
        base_next, _ = predict_base_and_nn_next_from_state(
            base_model=base_model,
            error_model=error_model,
            current_state=base_rollout[step],
            control_step=control_sequence[step],
            trailer_mass_kg_step=float(trailer_mass_kg[step]),
            dt_value=float(dt_values[step]),
            device=device,
            feature_context_tensors=feature_context_tensors,
            mlp_output_clip=mlp_output_clip,
        )
        _, nn_next = predict_base_and_nn_next_from_state(
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
        base_rollout[step + 1] = base_next
        nn_rollout[step + 1] = nn_next

    return base_rollout, nn_rollout


def pad_control_series(values: np.ndarray) -> np.ndarray:
    if len(values) == 0:
        return np.zeros((1,), dtype=np.float32)
    return np.concatenate([values.astype(np.float32), values[-1:].astype(np.float32)], axis=0)


def compute_state_error_series(predicted_states: np.ndarray, reference_states: np.ndarray, state_name: str) -> np.ndarray:
    index = STATE_NAMES.index(state_name)
    if state_name in {"psi_t", "psi_s"}:
        return np.rad2deg(wrap_angle_error_np(predicted_states[:, index] - reference_states[:, index]))
    if state_name in {"r_t", "r_s"}:
        return np.rad2deg(predicted_states[:, index] - reference_states[:, index])
    return predicted_states[:, index] - reference_states[:, index]


def build_results_dataframe(
    seg: InferenceSegment,
    base_rollout: np.ndarray,
    nn_rollout: np.ndarray,
) -> pd.DataFrame:
    time = seg.time.astype(np.float32)
    steer_sw_deg = np.rad2deg(pad_control_series(seg.control_sequence[:, 0]))
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
            "steer_sw_deg": steer_sw_deg,
            "torque_fl_nm": torque_fl,
            "torque_fr_nm": torque_fr,
            "torque_rl_nm": torque_rl,
            "torque_rr_nm": torque_rr,
            "trailer_mass_kg": trailer_mass,
        }
    )

    for prefix, rollout in (("real", seg.real_rollout), ("base", base_rollout), ("nn", nn_rollout)):
        for index, name in enumerate(STATE_NAMES):
            values = rollout[:, index]
            if name in {"psi_t", "psi_s"}:
                df[f"{prefix}_{name}_deg"] = np.rad2deg(wrap_angle_error_np(values))
            elif name in {"r_t", "r_s"}:
                df[f"{prefix}_{name}_degps"] = np.rad2deg(values)
            else:
                df[f"{prefix}_{name}"] = values
        df[f"{prefix}_articulation_deg"] = compute_articulation_series(rollout)

    plot_error_names = STATE_NAMES + ["articulation"]
    for name in plot_error_names:
        if name == "articulation":
            err_base = compute_articulation_series(base_rollout) - compute_articulation_series(seg.real_rollout)
            err_nn = compute_articulation_series(nn_rollout) - compute_articulation_series(seg.real_rollout)
        else:
            err_base = compute_state_error_series(base_rollout, seg.real_rollout, name)
            err_nn = compute_state_error_series(nn_rollout, seg.real_rollout, name)
        df[f"err_base_{name}"] = err_base.astype(np.float32)
        df[f"err_nn_{name}"] = err_nn.astype(np.float32)

    return df


def compute_rmse_summary(
    seg: InferenceSegment,
    base_rollout: np.ndarray,
    nn_rollout: np.ndarray,
    prefix: str,
) -> dict[str, float]:
    summary: dict[str, float] = {}
    for name in STATE_NAMES:
        err_base = compute_state_error_series(base_rollout, seg.real_rollout, name).astype(np.float64)
        err_nn = compute_state_error_series(nn_rollout, seg.real_rollout, name).astype(np.float64)
        summary[f"{prefix}_rmse_base_{name}"] = float(np.sqrt(np.mean(err_base * err_base)))
        summary[f"{prefix}_rmse_nn_{name}"] = float(np.sqrt(np.mean(err_nn * err_nn)))
    err_base_art = (compute_articulation_series(base_rollout) - compute_articulation_series(seg.real_rollout)).astype(np.float64)
    err_nn_art = (compute_articulation_series(nn_rollout) - compute_articulation_series(seg.real_rollout)).astype(np.float64)
    summary[f"{prefix}_rmse_base_articulation_deg"] = float(np.sqrt(np.mean(err_base_art * err_base_art)))
    summary[f"{prefix}_rmse_nn_articulation_deg"] = float(np.sqrt(np.mean(err_nn_art * err_nn_art)))
    return summary


def export_results_csv(seg: InferenceSegment, mode_key: str, base_rollout: np.ndarray, nn_rollout: np.ndarray) -> Path:
    out_path = seg.out_dir / f"{mode_key}_results.csv"
    build_results_dataframe(seg, base_rollout, nn_rollout).to_csv(out_path, index=False, encoding="utf-8-sig")
    return out_path


def plot_controls(seg: InferenceSegment) -> Path:
    time = seg.time[:-1]
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    axes[0].plot(time, np.rad2deg(seg.control_sequence[:, 0]), label="Steering Wheel", linewidth=1.8)
    axes[0].set_ylabel("deg")
    axes[0].set_title("Steering Wheel Angle")
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


def plot_trajectory(seg: InferenceSegment, mode_key: str, base_rollout: np.ndarray, nn_rollout: np.ndarray) -> Path:
    mode_title = MODE_TITLES[mode_key]
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    axes[0].plot(seg.real_rollout[:, 0], seg.real_rollout[:, 1], label=REAL_LABEL, linewidth=1.9)
    axes[0].plot(base_rollout[:, 0], base_rollout[:, 1], label=BASE_LABEL, linewidth=1.6)
    axes[0].plot(nn_rollout[:, 0], nn_rollout[:, 1], label=NN_LABEL, linewidth=1.7)
    axes[0].set_title(f"{mode_title} Tractor Trajectory")
    axes[0].set_xlabel("X (m)")
    axes[0].set_ylabel("Y (m)")
    axes[0].grid(True, linestyle="--", alpha=0.35)
    axes[0].set_aspect("equal", adjustable="box")

    axes[1].plot(seg.real_rollout[:, 6], seg.real_rollout[:, 7], label=REAL_LABEL, linewidth=1.9)
    axes[1].plot(base_rollout[:, 6], base_rollout[:, 7], label=BASE_LABEL, linewidth=1.6)
    axes[1].plot(nn_rollout[:, 6], nn_rollout[:, 7], label=NN_LABEL, linewidth=1.7)
    axes[1].set_title(f"{mode_title} Trailer Trajectory")
    axes[1].set_xlabel("X (m)")
    axes[1].set_ylabel("Y (m)")
    axes[1].grid(True, linestyle="--", alpha=0.35)
    axes[1].set_aspect("equal", adjustable="box")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.02))

    out_path = seg.out_dir / f"{mode_key}_trajectory.png"
    save_inference_figure(fig, out_path, top_margin=0.92)
    return out_path


def plot_state_error_all(seg: InferenceSegment, mode_key: str, base_rollout: np.ndarray, nn_rollout: np.ndarray) -> Path:
    plot_names = STATE_NAMES + ["articulation"]
    mode_title = MODE_TITLES[mode_key]
    fig, axes = plt.subplots(5, 3, figsize=(18, 16), sharex=True)
    axes = axes.ravel()

    for axis, name in zip(axes, plot_names, strict=False):
        display_name, ylabel = STATE_PLOT_META[name]
        if name == "articulation":
            err_base = compute_articulation_series(base_rollout) - compute_articulation_series(seg.real_rollout)
            err_nn = compute_articulation_series(nn_rollout) - compute_articulation_series(seg.real_rollout)
        else:
            err_base = compute_state_error_series(base_rollout, seg.real_rollout, name)
            err_nn = compute_state_error_series(nn_rollout, seg.real_rollout, name)
        axis.plot(seg.time, err_base, label=f"{BASE_LABEL} - {REAL_LABEL}", linewidth=1.4)
        axis.plot(seg.time, err_nn, label=f"{NN_LABEL} - {REAL_LABEL}", linewidth=1.6)
        axis.axhline(0.0, color="black", linewidth=0.8, linestyle=":")
        axis.set_title(f"{mode_title} {display_name} Error")
        axis.set_ylabel(ylabel)
        axis.grid(True, linestyle="--", alpha=0.35)

    for axis in axes[len(plot_names):]:
        axis.axis("off")

    for axis in axes[-3:]:
        axis.set_xlabel("Time (s)")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.02))

    out_path = seg.out_dir / f"{mode_key}_state_errors.png"
    save_inference_figure(fig, out_path, top_margin=0.94)
    return out_path


def export_segment_summary_csv(summary: dict[str, float | str], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([summary]).to_csv(output_path, index=False, encoding="utf-8-sig")
    return output_path


def export_summary_csv(rows: list[dict[str, float]], output_path: Path) -> Path:
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
    if feature_context is None:
        print("Checkpoint does not contain feature normalization statistics; raw features will be used.")
    if mlp_output_clip is None:
        print("Checkpoint does not contain output scaling; residual clipping is disabled.")

    summary_rows: list[dict[str, float]] = []
    processed_count = 0

    for csv_path in csvs:
        try:
            seg = load_segment(csv_path)
        except Exception as exc:
            print(f"[Skip] {csv_path}: {exc}")
            continue

        single_step_base, single_step_nn = rollout_single_step(
            base_model=base_model,
            error_model=error_model,
            real_rollout=seg.real_rollout,
            control_sequence=seg.control_sequence,
            trailer_mass_kg=seg.trailer_mass_kg,
            dt_values=seg.dt_values,
            device=device,
            feature_context_tensors=feature_context_tensors,
            mlp_output_clip=mlp_output_clip,
        )
        recursive_base, recursive_nn = rollout_recursive(
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

        controls_png = plot_controls(seg)
        single_step_result_csv = export_results_csv(seg, "single_step", single_step_base, single_step_nn)
        single_step_traj_png = plot_trajectory(seg, "single_step", single_step_base, single_step_nn)
        single_step_errors_png = plot_state_error_all(seg, "single_step", single_step_base, single_step_nn)
        recursive_result_csv = export_results_csv(seg, "recursive_rollout", recursive_base, recursive_nn)
        recursive_traj_png = plot_trajectory(seg, "recursive_rollout", recursive_base, recursive_nn)
        recursive_errors_png = plot_state_error_all(seg, "recursive_rollout", recursive_base, recursive_nn)
        summary: dict[str, float | str] = {
            "scenario_name": seg.scenario_name,
            "segment_name": seg.segment_name,
            "csv_path": str(seg.csv_path),
            "output_dir": str(seg.out_dir),
            "sample_count": int(len(seg.real_rollout)),
            "mean_trailer_mass_kg": float(np.mean(seg.trailer_mass_kg)),
        }
        summary.update(compute_rmse_summary(seg, single_step_base, single_step_nn, "single_step"))
        summary.update(compute_rmse_summary(seg, recursive_base, recursive_nn, "recursive"))
        segment_summary_csv = export_segment_summary_csv(summary, seg.out_dir / "rmse_summary.csv")
        summary_rows.append(summary)
        processed_count += 1

        print(f"\n[OK] {seg.segment_name}")
        print(f"  scene   : {seg.scenario_name}")
        print(f"  out_dir : {seg.out_dir}")
        print(f"  controls: {controls_png.name}")
        print(f"  single  : {single_step_result_csv.name}, {single_step_traj_png.name}, {single_step_errors_png.name}")
        print(f"  recur   : {recursive_result_csv.name}, {recursive_traj_png.name}, {recursive_errors_png.name}")
        print(f"  summary : {segment_summary_csv.name}")
        print(
            "  single-step articulation rmse (deg): "
            f"base={summary['single_step_rmse_base_articulation_deg']:.4f}, "
            f"nn={summary['single_step_rmse_nn_articulation_deg']:.4f}"
        )
        print(
            "  recursive articulation rmse (deg): "
            f"base={summary['recursive_rmse_base_articulation_deg']:.4f}, "
            f"nn={summary['recursive_rmse_nn_articulation_deg']:.4f}"
        )

    if processed_count == 0:
        raise RuntimeError("No segments were processed successfully.")

    if args.summary_path is not None:
        summary_output_path = args.summary_path
    elif args.input_path is not None and len(csvs) == 1:
        summary_output_path = resolve_output_dir(csvs[0]) / "truck_trailer_inference_summary_modular.csv"
    elif args.input_path is not None and Path(args.input_path).is_dir():
        summary_output_path = Path(args.input_path) / "truck_trailer_inference_summary_modular.csv"
    else:
        summary_output_path = RUNS_ROOT / "truck_trailer_inference_summary_modular.csv"

    summary_path = export_summary_csv(summary_rows, summary_output_path)
    print(f"\nsummary csv: {summary_path}")
    print(f"Completed {processed_count} inference runs.")


if __name__ == "__main__":
    main()
