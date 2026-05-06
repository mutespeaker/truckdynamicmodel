from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

try:
    from .base_model import TruckTrailerNominalDynamics, wrap_angle_error_np
    from .constants import (
        CONTROL_NAMES,
        GRADIENT_CLIP_NORM,
        LEARNING_RATE,
        MIN_LEARNING_RATE,
        MLP_CONTROL_FEATURE_NAMES,
        MLP_DROPOUT_P,
        MLP_HIDDEN_DIM,
        MLP_HIDDEN_LAYERS,
        MLP_INPUT_FEATURE_NAMES,
        MLP_OUTPUT_NAMES,
        MLP_TORCH_DTYPE,
        MLP_USE_LAYER_NORM,
        MODEL_CHECKPOINT,
        MOTION_STATE_NAMES,
        NO_TRAILER_MASS_THRESHOLD_KG,
        POSE_LOSS_WARMUP_STEPS,
        POSE_STATE_NAMES,
        REAL_DATA_LABEL,
        STATE_NAMES,
        TRAIN_BATCH_SIZE,
        TRAIN_EPOCHS,
        TRAIN_LOSS_MODEL_CHECKPOINT,
        TRAIN_NUM_WORKERS,
        TURNING_SAMPLER_POWER,
        TURNING_SELECTION_BLEND,
        BASE_MODEL_PARAMS,
    )
    from .data_utils import (
        SegmentData,
        build_feature_context,
        build_feature_context_tensors,
        build_loss_context,
        build_mlp_input_feature_tensor,
        compute_turning_focus_mask,
        compute_turning_sample_weights,
        concat_segments_for_training,
        compute_articulation_series,
        describe_loss_context,
        describe_turning_focus_context,
        derive_full_error_from_mlp_output_np,
        derive_full_error_from_mlp_output_torch,
        fit_turning_focus_context,
        normalize_feature_tensor,
        normalize_features_np,
        safe_log10,
        save_figure,
        to_tensor,
    )
    from .model_structure import MLPErrorModel
except ImportError:
    from base_model import TruckTrailerNominalDynamics, wrap_angle_error_np
    from constants import (
        CONTROL_NAMES,
        GRADIENT_CLIP_NORM,
        LEARNING_RATE,
        MIN_LEARNING_RATE,
        MLP_CONTROL_FEATURE_NAMES,
        MLP_DROPOUT_P,
        MLP_HIDDEN_DIM,
        MLP_HIDDEN_LAYERS,
        MLP_INPUT_FEATURE_NAMES,
        MLP_OUTPUT_NAMES,
        MLP_TORCH_DTYPE,
        MLP_USE_LAYER_NORM,
        MODEL_CHECKPOINT,
        MOTION_STATE_NAMES,
        NO_TRAILER_MASS_THRESHOLD_KG,
        POSE_LOSS_WARMUP_STEPS,
        POSE_STATE_NAMES,
        REAL_DATA_LABEL,
        STATE_NAMES,
        TRAIN_BATCH_SIZE,
        TRAIN_EPOCHS,
        TRAIN_LOSS_MODEL_CHECKPOINT,
        TRAIN_NUM_WORKERS,
        TURNING_SAMPLER_POWER,
        TURNING_SELECTION_BLEND,
        BASE_MODEL_PARAMS,
    )
    from data_utils import (
        SegmentData,
        build_feature_context,
        build_feature_context_tensors,
        build_loss_context,
        build_mlp_input_feature_tensor,
        compute_turning_focus_mask,
        compute_turning_sample_weights,
        concat_segments_for_training,
        compute_articulation_series,
        describe_loss_context,
        describe_turning_focus_context,
        derive_full_error_from_mlp_output_np,
        derive_full_error_from_mlp_output_torch,
        fit_turning_focus_context,
        normalize_feature_tensor,
        normalize_features_np,
        safe_log10,
        save_figure,
        to_tensor,
    )
    from model_structure import MLPErrorModel


VALIDATION_DIRECTION_CHANNELS = (
    ("x_t", STATE_NAMES.index("x_t")),
    ("y_t", STATE_NAMES.index("y_t")),
    ("vx_t", STATE_NAMES.index("vx_t")),
    ("vy_t", STATE_NAMES.index("vy_t")),
    ("x_s", STATE_NAMES.index("x_s")),
    ("y_s", STATE_NAMES.index("y_s")),
    ("vx_s", STATE_NAMES.index("vx_s")),
    ("vy_s", STATE_NAMES.index("vy_s")),
)


def compute_loss_components(
    predicted_mlp_output: torch.Tensor,
    true_mlp_output: torch.Tensor,
    true_error: torch.Tensor,
    base_next: torch.Tensor,
    dt_values: torch.Tensor,
    trailer_mass_kg: torch.Tensor,
    loss_context: dict[str, torch.Tensor],
    pose_loss_weight: float,
    sample_weight: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    error_scale = loss_context["error_scale"].to(dtype=MLP_TORCH_DTYPE)
    pose_error_scale = loss_context["pose_error_scale"].to(dtype=MLP_TORCH_DTYPE)
    output_scale = loss_context["output_scale"].to(dtype=MLP_TORCH_DTYPE)
    channel_weight = loss_context["channel_weight"].to(dtype=MLP_TORCH_DTYPE)
    output_weight = loss_context["output_weight"].to(dtype=MLP_TORCH_DTYPE)
    pose_indices = [STATE_NAMES.index(name) for name in POSE_STATE_NAMES]
    motion_indices = [STATE_NAMES.index(name) for name in MOTION_STATE_NAMES]
    full_indices = pose_indices + motion_indices

    predicted_error = derive_full_error_from_mlp_output_torch(predicted_mlp_output, base_next, dt_values, trailer_mass_kg)
    true_pose_error = true_error[:, pose_indices]
    predicted_pose_error = predicted_error[:, pose_indices]

    pose_weight = channel_weight[:, pose_indices]

    pose_residual = (predicted_pose_error - true_pose_error) / pose_error_scale
    output_residual = (predicted_mlp_output - true_mlp_output) / output_scale
    if trailer_mass_kg.ndim == 1:
        trailer_mass_kg = trailer_mass_kg.unsqueeze(1)
    has_trailer = (trailer_mass_kg > NO_TRAILER_MASS_THRESHOLD_KG).to(dtype=MLP_TORCH_DTYPE)
    output_mask = torch.cat(
        [
            torch.ones_like(has_trailer).repeat(1, 3),
            has_trailer.repeat(1, predicted_mlp_output.shape[1] - 3),
        ],
        dim=1,
    )
    pose_loss_per_sample = torch.mean((pose_residual * pose_weight).square(), dim=1)
    output_loss_per_sample = torch.sum(((output_residual * output_weight) * output_mask).square(), dim=1) / torch.clamp(
        output_mask.sum(dim=1),
        min=1.0,
    )

    full_residual = (predicted_error[:, full_indices] - true_error[:, full_indices]) / error_scale[:, full_indices]
    full_weight = channel_weight[:, full_indices]
    full_loss_per_sample = torch.mean((full_residual * full_weight).square(), dim=1)
    total_loss_per_sample = output_loss_per_sample + pose_loss_weight * pose_loss_per_sample + 0.05 * full_loss_per_sample

    if sample_weight is not None:
        sample_weight = sample_weight.reshape(-1).to(device=predicted_mlp_output.device, dtype=MLP_TORCH_DTYPE)
        weight_sum = torch.clamp(sample_weight.sum(), min=1.0e-6)
        pose_loss = torch.sum(pose_loss_per_sample * sample_weight) / weight_sum
        output_loss = torch.sum(output_loss_per_sample * sample_weight) / weight_sum
        total_loss = torch.sum(total_loss_per_sample * sample_weight) / weight_sum
    else:
        pose_loss = torch.mean(pose_loss_per_sample)
        output_loss = torch.mean(output_loss_per_sample)
        total_loss = torch.mean(total_loss_per_sample)
    return {
        "total_loss": total_loss,
        "pose_loss": pose_loss,
        "motion_loss": output_loss,
        "predicted_error": predicted_error,
        "total_loss_per_sample": total_loss_per_sample,
    }


def build_checkpoint_payload(
    state_dict: dict[str, torch.Tensor],
    model_input_dim: int,
    feature_context: dict[str, np.ndarray],
    loss_context: dict[str, torch.Tensor],
    turning_focus_context: dict[str, float] | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "state_dict": state_dict,
        "model_input_dim": int(model_input_dim),
        "model_output_dim": int(len(MLP_OUTPUT_NAMES)),
        "mlp_use_layer_norm": bool(MLP_USE_LAYER_NORM),
        "mlp_hidden_dim": int(MLP_HIDDEN_DIM),
        "mlp_hidden_layers": int(MLP_HIDDEN_LAYERS),
        "mlp_dropout_p": float(MLP_DROPOUT_P),
        "input_feature_names": list(MLP_INPUT_FEATURE_NAMES),
        "mlp_control_feature_names": list(MLP_CONTROL_FEATURE_NAMES),
        "mlp_output_names": list(MLP_OUTPUT_NAMES),
        "state_names": list(STATE_NAMES),
        "control_names": list(CONTROL_NAMES),
        "feature_mean": feature_context["feature_mean"],
        "feature_scale": feature_context["feature_scale"],
        "loss_error_scale": loss_context["error_scale"].detach().cpu().numpy(),
        "loss_pose_error_scale": loss_context["pose_error_scale"].detach().cpu().numpy(),
        "loss_motion_error_scale": loss_context["motion_error_scale"].detach().cpu().numpy(),
        "loss_output_scale": loss_context["output_scale"].detach().cpu().numpy(),
        "base_model_params": dict(BASE_MODEL_PARAMS),
    }
    if turning_focus_context is not None:
        payload["turning_focus_context"] = {
            "threshold_deg": float(turning_focus_context["threshold_deg"]),
            "gate_base_weight": float(turning_focus_context["gate_base_weight"]),
            "start_quantile": float(turning_focus_context["start_quantile"]),
            "full_quantile": float(turning_focus_context["full_quantile"]),
            "score_start": float(turning_focus_context["score_start"]),
            "score_full": float(turning_focus_context["score_full"]),
            "sample_weight_max": float(turning_focus_context["sample_weight_max"]),
            "sampler_power": float(TURNING_SAMPLER_POWER),
            "selection_blend": float(TURNING_SELECTION_BLEND),
        }
    return payload


def clone_unwrapped_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def load_unwrapped_state_dict(model: nn.Module, state_dict: dict[str, torch.Tensor]) -> None:
    model.load_state_dict(state_dict)


def build_train_loader(
    x_tensor: torch.Tensor,
    y_output_tensor: torch.Tensor,
    y_tensor: torch.Tensor,
    base_next_tensor: torch.Tensor,
    dt_tensor: torch.Tensor,
    mass_tensor: torch.Tensor,
    turn_focus_mask_tensor: torch.Tensor,
    sample_weight_tensor: torch.Tensor,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader:
    dataset = TensorDataset(
        x_tensor,
        y_output_tensor,
        y_tensor,
        base_next_tensor,
        dt_tensor,
        mass_tensor,
        turn_focus_mask_tensor,
        sample_weight_tensor,
    )
    sampler = None
    shuffle = True
    if len(dataset) > 1:
        sampler_weights = torch.pow(
            sample_weight_tensor.reshape(-1).to(dtype=torch.double).cpu(),
            float(TURNING_SAMPLER_POWER),
        )
        sampler = WeightedRandomSampler(
            weights=sampler_weights,
            num_samples=len(dataset),
            replacement=True,
        )
        shuffle = False
    return DataLoader(
        dataset,
        batch_size=max(1, min(int(batch_size), len(dataset))),
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )


def compute_pose_loss_weight(global_step: int) -> float:
    if global_step < POSE_LOSS_WARMUP_STEPS:
        return 0.0
    return 1.0


def initialize_validation_mse_sums() -> dict[str, float]:
    return {f"{scope}_{name}": 0.0 for scope in ("base", "corr") for name, _ in VALIDATION_DIRECTION_CHANNELS}


def update_validation_mse_sums(
    mse_sums: dict[str, float],
    base_next: torch.Tensor,
    predicted_error: torch.Tensor,
    true_error: torch.Tensor,
) -> None:
    predicted_next = base_next + predicted_error
    true_next = base_next + true_error
    for name, index in VALIDATION_DIRECTION_CHANNELS:
        base_diff = base_next[:, index] - true_next[:, index]
        corr_diff = predicted_next[:, index] - true_next[:, index]
        mse_sums[f"base_{name}"] += float(torch.sum(base_diff.square()).detach().cpu())
        mse_sums[f"corr_{name}"] += float(torch.sum(corr_diff.square()).detach().cpu())


def finalize_validation_mse(mse_sums: dict[str, float], sample_count: int) -> dict[str, float]:
    denom = float(max(sample_count, 1))
    return {f"val_mse_{key}": value / denom for key, value in mse_sums.items()}


def format_validation_mse_lines(validation_mse: dict[str, float]) -> tuple[str, str]:
    displacement_line = " | ".join(
        f"{name}: base={validation_mse[f'val_mse_base_{name}']:.6e}, corr={validation_mse[f'val_mse_corr_{name}']:.6e}"
        for name in ("x_t", "y_t", "x_s", "y_s")
    )
    velocity_line = " | ".join(
        f"{name}: base={validation_mse[f'val_mse_base_{name}']:.6e}, corr={validation_mse[f'val_mse_corr_{name}']:.6e}"
        for name in ("vx_t", "vy_t", "vx_s", "vy_s")
    )
    return displacement_line, velocity_line


def get_current_learning_rate(optimizer: torch.optim.Optimizer) -> float:
    return float(optimizer.param_groups[0]["lr"])


def train_error_model_multirun(
    base_model: TruckTrailerNominalDynamics,
    train_segments: list[SegmentData],
    val_segments: list[SegmentData],
    device: torch.device,
    epochs: int = TRAIN_EPOCHS,
    learning_rate: float = LEARNING_RATE,
    min_learning_rate: float = MIN_LEARNING_RATE,
    batch_size: int = TRAIN_BATCH_SIZE,
    num_workers: int = TRAIN_NUM_WORKERS,
) -> tuple[MLPErrorModel, dict[str, np.ndarray], dict[str, torch.Tensor], dict[str, list[float]]]:
    if learning_rate <= 0.0:
        raise ValueError(f"learning_rate must be positive, got {learning_rate}.")
    if min_learning_rate < 0.0:
        raise ValueError(f"min_learning_rate must be non-negative, got {min_learning_rate}.")
    if min_learning_rate > learning_rate:
        raise ValueError(
            f"min_learning_rate ({min_learning_rate}) must not be greater than learning_rate ({learning_rate})."
        )

    x_train_raw, y_train_output, y_train, base_next_train, train_dt_values, train_mass, train_turn_scores = concat_segments_for_training(
        base_model,
        train_segments,
        device,
    )
    x_val_raw, y_val_output, y_val, base_next_val, val_dt_values, val_mass, val_turn_scores = concat_segments_for_training(
        base_model,
        val_segments,
        device,
    )

    if len(x_train_raw) < 5:
        raise ValueError("Not enough training samples to train the residual model.")

    feature_context = build_feature_context(x_train_raw)
    x_train = normalize_features_np(x_train_raw, feature_context)
    x_val = normalize_features_np(x_val_raw, feature_context)
    loss_context = build_loss_context(y_train, y_train_output, device)
    turning_focus_context = fit_turning_focus_context(train_turn_scores)
    train_turn_mask = compute_turning_focus_mask(train_turn_scores, turning_focus_context)
    val_turn_mask = compute_turning_focus_mask(val_turn_scores, turning_focus_context)
    train_sample_weights = compute_turning_sample_weights(train_turn_scores, turning_focus_context)
    val_sample_weights = compute_turning_sample_weights(val_turn_scores, turning_focus_context)
    describe_loss_context(loss_context)
    describe_turning_focus_context(
        turning_focus_context,
        train_turn_scores=train_turn_scores,
        train_sample_weights=train_sample_weights,
        val_turn_scores=val_turn_scores,
        val_sample_weights=val_sample_weights,
    )

    model = MLPErrorModel(
        input_dim=x_train.shape[1],
        output_dim=len(MLP_OUTPUT_NAMES),
        dropout_p=MLP_DROPOUT_P,
        use_layer_norm=MLP_USE_LAYER_NORM,
        hidden_dim=MLP_HIDDEN_DIM,
        hidden_layers=MLP_HIDDEN_LAYERS,
    ).to(device=device, dtype=MLP_TORCH_DTYPE)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1.0e-5)

    x_train_tensor = torch.as_tensor(x_train, dtype=MLP_TORCH_DTYPE)
    y_train_output_tensor = torch.as_tensor(y_train_output, dtype=MLP_TORCH_DTYPE)
    y_train_tensor = torch.as_tensor(y_train, dtype=MLP_TORCH_DTYPE)
    base_next_train_tensor = torch.as_tensor(base_next_train, dtype=MLP_TORCH_DTYPE)
    dt_train_tensor = torch.as_tensor(train_dt_values, dtype=MLP_TORCH_DTYPE)
    mass_train_tensor = torch.as_tensor(train_mass, dtype=MLP_TORCH_DTYPE)
    train_turn_mask_tensor = torch.as_tensor(train_turn_mask.reshape(-1, 1), dtype=MLP_TORCH_DTYPE)
    train_sample_weight_tensor = torch.as_tensor(train_sample_weights.reshape(-1, 1), dtype=MLP_TORCH_DTYPE)
    x_val_tensor = torch.as_tensor(x_val, dtype=MLP_TORCH_DTYPE)
    y_val_output_tensor = torch.as_tensor(y_val_output, dtype=MLP_TORCH_DTYPE)
    y_val_tensor = torch.as_tensor(y_val, dtype=MLP_TORCH_DTYPE)
    base_next_val_tensor = torch.as_tensor(base_next_val, dtype=MLP_TORCH_DTYPE)
    dt_val_tensor = torch.as_tensor(val_dt_values, dtype=MLP_TORCH_DTYPE)
    mass_val_tensor = torch.as_tensor(val_mass, dtype=MLP_TORCH_DTYPE)
    val_turn_mask_tensor = torch.as_tensor(val_turn_mask.reshape(-1, 1), dtype=MLP_TORCH_DTYPE)
    val_sample_weight_tensor = torch.as_tensor(val_sample_weights.reshape(-1, 1), dtype=MLP_TORCH_DTYPE)

    pin_memory = device.type == "cuda"
    train_loader = build_train_loader(
        x_train_tensor,
        y_train_output_tensor,
        y_train_tensor,
        base_next_train_tensor,
        dt_train_tensor,
        mass_train_tensor,
        train_turn_mask_tensor,
        train_sample_weight_tensor,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        TensorDataset(
            x_val_tensor,
            y_val_output_tensor,
            y_val_tensor,
            base_next_val_tensor,
            dt_val_tensor,
            mass_val_tensor,
            val_turn_mask_tensor,
            val_sample_weight_tensor,
        ),
        batch_size=max(1, min(batch_size, len(x_val_tensor))),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    scheduler_total_steps = max(1, int(epochs) * max(1, len(train_loader)))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=scheduler_total_steps,
        eta_min=float(min_learning_rate),
    )

    history = {
        "train_total": [],
        "val_total": [],
        "train_pose": [],
        "val_pose": [],
        "train_motion": [],
        "val_motion": [],
        "pose_loss_weight": [],
        "val_turn_focus_total": [],
        "val_selection_score": [],
        "learning_rate": [],
    }

    best_selection_score = float("inf")
    best_train_loss = float("inf")
    best_state_dict: dict[str, torch.Tensor] | None = None
    best_train_state_dict: dict[str, torch.Tensor] | None = None
    global_step = 0
    validation_mse_history = {f"val_mse_{scope}_{name}": [] for scope in ("base", "corr") for name, _ in VALIDATION_DIRECTION_CHANNELS}
    history.update(validation_mse_history)

    for epoch in range(1, epochs + 1):
        model.train()
        train_total = 0.0
        train_pose = 0.0
        train_motion = 0.0
        train_count = 0
        pose_loss_weight = compute_pose_loss_weight(global_step)
        epoch_learning_rate = get_current_learning_rate(optimizer)

        for x_batch_cpu, y_output_batch_cpu, y_batch_cpu, base_next_batch_cpu, dt_batch_cpu, mass_batch_cpu, _turn_mask_batch_cpu, sample_weight_batch_cpu in train_loader:
            x_batch = x_batch_cpu.to(device, non_blocking=pin_memory)
            y_output_batch = y_output_batch_cpu.to(device, non_blocking=pin_memory)
            y_batch = y_batch_cpu.to(device, non_blocking=pin_memory)
            base_next_batch = base_next_batch_cpu.to(device, non_blocking=pin_memory)
            dt_batch = dt_batch_cpu.to(device, non_blocking=pin_memory)
            mass_batch = mass_batch_cpu.to(device, non_blocking=pin_memory)
            sample_weight_batch = sample_weight_batch_cpu.to(device, non_blocking=pin_memory)
            pose_loss_weight = compute_pose_loss_weight(global_step)

            optimizer.zero_grad(set_to_none=True)
            predicted_mlp_output = model(x_batch)
            losses = compute_loss_components(
                predicted_mlp_output=predicted_mlp_output,
                true_mlp_output=y_output_batch,
                true_error=y_batch,
                base_next=base_next_batch,
                dt_values=dt_batch,
                trailer_mass_kg=mass_batch,
                loss_context=loss_context,
                pose_loss_weight=pose_loss_weight,
                sample_weight=sample_weight_batch,
            )
            losses["total_loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_NORM)
            optimizer.step()
            scheduler.step()
            global_step += 1

            batch_size_value = x_batch.shape[0]
            train_total += float(losses["total_loss"].detach().cpu()) * batch_size_value
            train_pose += float(losses["pose_loss"].detach().cpu()) * batch_size_value
            train_motion += float(losses["motion_loss"].detach().cpu()) * batch_size_value
            train_count += batch_size_value

        model.eval()
        val_pose_loss_weight = compute_pose_loss_weight(global_step)
        val_total = 0.0
        val_pose = 0.0
        val_motion = 0.0
        val_turn_focus_weighted_sum = 0.0
        val_turn_focus_weight_sum = 0.0
        val_count = 0
        validation_mse_sums = initialize_validation_mse_sums()
        with torch.no_grad():
            for x_batch_cpu, y_output_batch_cpu, y_batch_cpu, base_next_batch_cpu, dt_batch_cpu, mass_batch_cpu, turn_mask_batch_cpu, sample_weight_batch_cpu in val_loader:
                x_batch = x_batch_cpu.to(device, non_blocking=pin_memory)
                y_output_batch = y_output_batch_cpu.to(device, non_blocking=pin_memory)
                y_batch = y_batch_cpu.to(device, non_blocking=pin_memory)
                base_next_batch = base_next_batch_cpu.to(device, non_blocking=pin_memory)
                dt_batch = dt_batch_cpu.to(device, non_blocking=pin_memory)
                mass_batch = mass_batch_cpu.to(device, non_blocking=pin_memory)
                turn_mask_batch = turn_mask_batch_cpu.to(device, non_blocking=pin_memory)
                sample_weight_batch = sample_weight_batch_cpu.to(device, non_blocking=pin_memory)
                predicted_mlp_output = model(x_batch)
                losses = compute_loss_components(
                    predicted_mlp_output=predicted_mlp_output,
                    true_mlp_output=y_output_batch,
                    true_error=y_batch,
                    base_next=base_next_batch,
                    dt_values=dt_batch,
                    trailer_mass_kg=mass_batch,
                    loss_context=loss_context,
                    pose_loss_weight=val_pose_loss_weight,
                )
                turn_mask = turn_mask_batch.reshape(-1) > 0.5
                if torch.any(turn_mask):
                    turn_losses = losses["total_loss_per_sample"][turn_mask]
                    turn_weights = sample_weight_batch.reshape(-1)[turn_mask]
                    val_turn_focus_weighted_sum += float(torch.sum(turn_losses * turn_weights).detach().cpu())
                    val_turn_focus_weight_sum += float(torch.sum(turn_weights).detach().cpu())
                batch_size_value = x_batch.shape[0]
                val_total += float(losses["total_loss"].detach().cpu()) * batch_size_value
                val_pose += float(losses["pose_loss"].detach().cpu()) * batch_size_value
                val_motion += float(losses["motion_loss"].detach().cpu()) * batch_size_value
                val_count += batch_size_value
                update_validation_mse_sums(validation_mse_sums, base_next_batch, losses["predicted_error"], y_batch)

        train_total /= max(train_count, 1)
        train_pose /= max(train_count, 1)
        train_motion /= max(train_count, 1)
        val_total /= max(val_count, 1)
        val_pose /= max(val_count, 1)
        val_motion /= max(val_count, 1)
        if val_turn_focus_weight_sum > 1.0e-6:
            val_turn_focus_total = val_turn_focus_weighted_sum / val_turn_focus_weight_sum
        else:
            val_turn_focus_total = val_total
        val_selection_score = (1.0 - TURNING_SELECTION_BLEND) * val_total + TURNING_SELECTION_BLEND * val_turn_focus_total
        validation_mse = finalize_validation_mse(validation_mse_sums, val_count)

        history["train_total"].append(train_total)
        history["val_total"].append(val_total)
        history["train_pose"].append(train_pose)
        history["val_pose"].append(val_pose)
        history["train_motion"].append(train_motion)
        history["val_motion"].append(val_motion)
        history["pose_loss_weight"].append(val_pose_loss_weight)
        history["val_turn_focus_total"].append(val_turn_focus_total)
        history["val_selection_score"].append(val_selection_score)
        history["learning_rate"].append(epoch_learning_rate)
        for key, value in validation_mse.items():
            history[key].append(value)

        if train_total < best_train_loss:
            best_train_loss = train_total
            best_train_state_dict = clone_unwrapped_state_dict(model)
        if val_selection_score < best_selection_score:
            best_selection_score = val_selection_score
            best_state_dict = clone_unwrapped_state_dict(model)

        if epoch <= 10 or epoch % 1000 == 0 or epoch == epochs:
            displacement_line, velocity_line = format_validation_mse_lines(validation_mse)
            print(
                f"Epoch {epoch:5d}/{epochs} | step={global_step:6d} | lr={epoch_learning_rate:.6e} | "
                f"pose_w={val_pose_loss_weight:.3f} | "
                f"train_total={train_total:.6e} val_total={val_total:.6e} val_turn={val_turn_focus_total:.6e} "
                f"select={val_selection_score:.6e} | "
                f"train_output={train_motion:.6e} val_output={val_motion:.6e}"
            )
            print(f"  val_disp_mse | {displacement_line}")
            print(f"  val_vel_mse  | {velocity_line}")

    if best_state_dict is not None:
        load_unwrapped_state_dict(model, best_state_dict)
        torch.save(
            build_checkpoint_payload(best_state_dict, int(x_train.shape[1]), feature_context, loss_context, turning_focus_context),
            MODEL_CHECKPOINT,
        )

    if best_train_state_dict is not None:
        torch.save(
            build_checkpoint_payload(best_train_state_dict, int(x_train.shape[1]), feature_context, loss_context, turning_focus_context),
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
    base_rollout = np.zeros_like(real_rollout, dtype=np.float32)
    corrected_rollout = np.zeros_like(real_rollout, dtype=np.float32)
    base_rollout[0] = real_rollout[0].astype(np.float32)
    corrected_rollout[0] = real_rollout[0].astype(np.float32)

    mlp_output_clip = 3.0 * loss_context["output_scale"].detach().cpu().numpy().ravel().astype(np.float32)
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
        predicted_mlp_output = error_model(features).cpu().numpy()[0].astype(np.float32)
        predicted_mlp_output = np.clip(predicted_mlp_output, -mlp_output_clip, mlp_output_clip)

        base_next = base_next_tensor.cpu().numpy().astype(np.float32)
        corrected_error = derive_full_error_from_mlp_output_np(
            predicted_mlp_output.reshape(1, -1),
            base_next,
            np.array([dt_values[step]], dtype=np.float32),
            np.array([trailer_mass_kg[step]], dtype=np.float32),
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
    fig, axes = plt.subplots(2, 2, figsize=(18, 10))
    axes = axes.ravel()

    axes[0].plot(epochs, safe_log10(np.asarray(history["train_total"])), label="Train total")
    axes[0].plot(epochs, safe_log10(np.asarray(history["val_total"])), label="Val total")
    axes[0].set_title("Total Loss (log10)")
    axes[0].set_xlabel("Epoch")
    axes[0].grid(True, linestyle="--", alpha=0.35)
    axes[0].legend()

    axes[1].plot(epochs, safe_log10(np.asarray(history["train_motion"])), label="Train output")
    axes[1].plot(epochs, safe_log10(np.asarray(history["val_motion"])), label="Val output")
    axes[1].plot(epochs, np.asarray(history["pose_loss_weight"]), label="Pose weight")
    axes[1].set_title("Motion Loss / Pose Weight")
    axes[1].set_xlabel("Epoch")
    axes[1].grid(True, linestyle="--", alpha=0.35)
    axes[1].legend()

    axes[2].plot(epochs, safe_log10(np.asarray(history["val_turn_focus_total"])), label="Val turn-focused")
    axes[2].plot(epochs, safe_log10(np.asarray(history["val_selection_score"])), label="Val selection")
    axes[2].set_title("Turn-Focused Validation (log10)")
    axes[2].set_xlabel("Epoch")
    axes[2].grid(True, linestyle="--", alpha=0.35)
    axes[2].legend()

    axes[3].plot(epochs, np.asarray(history["learning_rate"]), label="Learning rate")
    axes[3].set_title("Cosine-Annealed Learning Rate")
    axes[3].set_xlabel("Epoch")
    axes[3].grid(True, linestyle="--", alpha=0.35)
    axes[3].legend()

    output_path = output_dir / "truck_trailer_training_loss_log.png"
    save_figure(fig, output_path)
    return output_path


def plot_trajectory(real_rollout: np.ndarray, base_rollout: np.ndarray, corrected_rollout: np.ndarray, plot_dir: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].plot(real_rollout[:, 0], real_rollout[:, 1], label=REAL_DATA_LABEL, linewidth=1.8)
    axes[0].plot(base_rollout[:, 0], base_rollout[:, 1], label="Base", linewidth=1.5)
    axes[0].plot(corrected_rollout[:, 0], corrected_rollout[:, 1], label="Base + NN", linewidth=1.6)
    axes[0].set_title("Tractor Trajectory")
    axes[0].set_xlabel("X (m)")
    axes[0].set_ylabel("Y (m)")
    axes[0].axis("equal")
    axes[0].grid(True, linestyle="--", alpha=0.35)
    axes[0].legend()

    axes[1].plot(real_rollout[:, 6], real_rollout[:, 7], label=REAL_DATA_LABEL, linewidth=1.8)
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
        axis.plot(time, real_values, label=REAL_DATA_LABEL, linewidth=1.6)
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
