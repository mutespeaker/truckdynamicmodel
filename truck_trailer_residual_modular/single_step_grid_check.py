from __future__ import annotations

import argparse
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
    from .constants import CONTROL_NAMES, MLP_OUTPUT_NAMES, MLP_NUMPY_DTYPE, STATE_NAMES
    from .data_utils import (
        build_feature_context_tensors,
        build_mlp_input_feature_tensor,
        derive_full_error_from_mlp_output_np,
        normalize_feature_tensor,
        to_tensor,
    )
    from .inference_main import (
        build_base_model,
        extract_feature_context,
        extract_output_clip,
        load_error_model,
        save_inference_figure,
    )
except ImportError:
    from constants import CONTROL_NAMES, MLP_OUTPUT_NAMES, MLP_NUMPY_DTYPE, STATE_NAMES
    from data_utils import (
        build_feature_context_tensors,
        build_mlp_input_feature_tensor,
        derive_full_error_from_mlp_output_np,
        normalize_feature_tensor,
        to_tensor,
    )
    from inference_main import (
        build_base_model,
        extract_feature_context,
        extract_output_clip,
        load_error_model,
        save_inference_figure,
    )


DEFAULT_CHECKPOINT = Path(
    r"D:\test_torch project\controltest\truck_trailer_residual_modular\data"
    r"\train_3000-05221455-v\run_20260524_012444_data_0515_vel_c42bbd__c00a8b"
    r"\checkpoints\best_truck_trailer_error_model.pth"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Check one-step Base and residual-MLP outputs on a Vx / rear-torque grid. "
            "Default states use zero pose, zero yaw, zero lateral velocity, zero yaw rate, "
            "and no trailer."
        ),
    )
    parser.add_argument("--checkpoint-path", type=Path, default=DEFAULT_CHECKPOINT, help="Residual model checkpoint.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to <checkpoint run>/single_step_grid_check.",
    )
    parser.add_argument("--dt", type=float, default=0.02, help="One-step integration time in seconds.")
    parser.add_argument("--vx-start", type=float, default=0.0, help="Initial tractor Vx grid start, m/s.")
    parser.add_argument("--vx-end", type=float, default=20.0, help="Initial tractor Vx grid end, m/s.")
    parser.add_argument("--vx-step", type=float, default=2.0, help="Initial tractor Vx grid step, m/s.")
    parser.add_argument("--torque-start", type=float, default=0.0, help="Rear torque total grid start, Nm.")
    parser.add_argument("--torque-end", type=float, default=5000.0, help="Rear torque total grid end, Nm.")
    parser.add_argument("--torque-step", type=float, default=500.0, help="Rear torque total grid step, Nm.")
    parser.add_argument("--steer-sw-rad", type=float, default=0.0, help="Steering-wheel angle, rad.")
    parser.add_argument("--trailer-mass-kg", type=float, default=0.0, help="Trailer mass. Use 0 for no-trailer mode.")
    parser.add_argument("--device", default=None, help="Torch device. Defaults to cuda when available, otherwise cpu.")
    return parser.parse_args()


def inclusive_grid(start: float, end: float, step: float) -> np.ndarray:
    if step <= 0.0:
        raise ValueError(f"Grid step must be positive, got {step}.")
    count = int(np.floor((end - start) / step + 0.5)) + 1
    values = start + step * np.arange(max(count, 1), dtype=np.float32)
    return values[values <= end + 1.0e-6].astype(np.float32)


def default_output_dir(checkpoint_path: Path) -> Path:
    if checkpoint_path.parent.name.lower() == "checkpoints":
        return checkpoint_path.parent.parent / "single_step_grid_check"
    return checkpoint_path.parent / "single_step_grid_check"


def build_state_control_grid(
    vx_values: np.ndarray,
    torque_values: np.ndarray,
    steer_sw_rad: float,
    trailer_mass_kg: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, float]]]:
    states: list[np.ndarray] = []
    controls: list[np.ndarray] = []
    masses: list[float] = []
    grid_rows: list[dict[str, float]] = []

    for vx in vx_values:
        for rear_torque_total in torque_values:
            state = np.zeros(len(STATE_NAMES), dtype=np.float32)
            state[STATE_NAMES.index("vx_t")] = float(vx)
            state[STATE_NAMES.index("vx_s")] = float(vx)

            control = np.zeros(len(CONTROL_NAMES), dtype=np.float32)
            control[CONTROL_NAMES.index("steer_sw_rad")] = float(steer_sw_rad)
            # The residual MLP sees rear_drive_torque_sum. Split the requested
            # total equally across rear-left and rear-right so the base model
            # gets the same net torque without an artificial left-right moment.
            control[CONTROL_NAMES.index("torque_rl")] = float(rear_torque_total) * 0.5
            control[CONTROL_NAMES.index("torque_rr")] = float(rear_torque_total) * 0.5

            states.append(state)
            controls.append(control)
            masses.append(float(trailer_mass_kg))
            grid_rows.append(
                {
                    "initial_vx_t_mps": float(vx),
                    "rear_drive_torque_sum_nm": float(rear_torque_total),
                    "torque_rl_nm": float(control[CONTROL_NAMES.index("torque_rl")]),
                    "torque_rr_nm": float(control[CONTROL_NAMES.index("torque_rr")]),
                    "steer_sw_rad": float(steer_sw_rad),
                    "trailer_mass_kg": float(trailer_mass_kg),
                },
            )

    return (
        np.stack(states, axis=0).astype(np.float32),
        np.stack(controls, axis=0).astype(np.float32),
        np.asarray(masses, dtype=np.float32).reshape(-1, 1),
        grid_rows,
    )


@torch.no_grad()
def run_grid_inference(
    checkpoint_path: Path,
    states: np.ndarray,
    controls: np.ndarray,
    masses: np.ndarray,
    dt_value: float,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    error_model, checkpoint_metadata, resolved_checkpoint_path = load_error_model(device, checkpoint_path)
    base_model = build_base_model(checkpoint_metadata, device)
    feature_context = extract_feature_context(checkpoint_metadata)
    feature_context_tensors = (
        build_feature_context_tensors(feature_context, device)
        if feature_context is not None
        else None
    )
    output_clip = extract_output_clip(checkpoint_metadata)

    dt_values = np.full((states.shape[0], 1), float(dt_value), dtype=np.float32)
    state_tensor = to_tensor(states, device)
    control_tensor = to_tensor(controls, device)
    mass_tensor = to_tensor(masses, device)
    dt_tensor = to_tensor(dt_values, device)

    base_next = base_model(state_tensor, control_tensor, mass_tensor, dt_tensor).cpu().numpy().astype(np.float32)

    features = build_mlp_input_feature_tensor(state_tensor, control_tensor, mass_tensor, dt_tensor)
    if feature_context_tensors is not None:
        features = normalize_feature_tensor(features, feature_context_tensors)
    mlp_raw = error_model(features).cpu().numpy().astype(np.float32)
    if output_clip is None:
        mlp_clipped = mlp_raw.copy()
    else:
        mlp_clipped = np.clip(mlp_raw, -output_clip.reshape(1, -1), output_clip.reshape(1, -1)).astype(np.float32)

    nn_full_error = derive_full_error_from_mlp_output_np(
        mlp_clipped,
        base_next,
        dt_values.reshape(-1),
        masses.reshape(-1),
    )
    nn_next = (base_next + nn_full_error).astype(np.float32)

    metadata = {
        "resolved_checkpoint_path": str(resolved_checkpoint_path),
        "output_clip": output_clip,
        "checkpoint_metadata": checkpoint_metadata,
    }
    return base_next, mlp_raw, mlp_clipped, nn_full_error, nn_next, metadata


def add_state_columns(frame: pd.DataFrame, prefix: str, values: np.ndarray) -> pd.DataFrame:
    columns = {
        f"{prefix}_{name}": values[:, index].astype(np.float32)
        for index, name in enumerate(STATE_NAMES)
    }
    return pd.concat([frame, pd.DataFrame(columns)], axis=1)


def add_mlp_columns(frame: pd.DataFrame, prefix: str, values: np.ndarray) -> pd.DataFrame:
    columns: dict[str, np.ndarray] = {}
    for index, name in enumerate(MLP_OUTPUT_NAMES):
        columns[f"{prefix}_{name}"] = values[:, index].astype(np.float32)
        if name in {"r_t", "r_s"}:
            columns[f"{prefix}_{name}_degps"] = np.rad2deg(values[:, index]).astype(np.float32)
        if name == "rel_yaw_s_t":
            columns[f"{prefix}_{name}_deg"] = np.rad2deg(values[:, index]).astype(np.float32)
    return pd.concat([frame, pd.DataFrame(columns)], axis=1)


def build_results_dataframe(
    grid_rows: list[dict[str, float]],
    states: np.ndarray,
    controls: np.ndarray,
    dt_value: float,
    base_next: np.ndarray,
    mlp_raw: np.ndarray,
    mlp_clipped: np.ndarray,
    nn_full_error: np.ndarray,
    nn_next: np.ndarray,
) -> pd.DataFrame:
    frame = pd.DataFrame(grid_rows)
    frame["dt_s"] = float(dt_value)
    frame = add_state_columns(frame, "initial", states)
    for index, name in enumerate(CONTROL_NAMES):
        frame[f"control_{name}"] = controls[:, index].astype(np.float32)

    frame = add_state_columns(frame, "base_next", base_next)
    frame = add_state_columns(frame, "base_delta", base_next - states)
    frame = add_mlp_columns(frame, "mlp_raw", mlp_raw)
    frame = add_mlp_columns(frame, "mlp_clipped", mlp_clipped)
    frame = add_state_columns(frame, "nn_residual_full_error", nn_full_error)
    frame = add_state_columns(frame, "nn_next", nn_next)
    frame = add_state_columns(frame, "nn_minus_base", nn_next - base_next)

    extra_columns: dict[str, np.ndarray] = {}
    for prefix in ("base_delta", "nn_residual_full_error", "nn_minus_base"):
        extra_columns[f"{prefix}_psi_t_deg"] = np.rad2deg(frame[f"{prefix}_psi_t"].to_numpy(dtype=np.float32))
        extra_columns[f"{prefix}_r_t_degps"] = np.rad2deg(frame[f"{prefix}_r_t"].to_numpy(dtype=np.float32))
    return pd.concat([frame, pd.DataFrame(extra_columns)], axis=1)


def plot_heatmap_grid(
    frame: pd.DataFrame,
    vx_values: np.ndarray,
    torque_values: np.ndarray,
    columns: list[tuple[str, str]],
    title: str,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharex=True, sharey=True)
    axes = axes.ravel()
    for axis, (column, label) in zip(axes, columns, strict=False):
        pivot = (
            frame.pivot(
                index="initial_vx_t_mps",
                columns="rear_drive_torque_sum_nm",
                values=column,
            )
            .reindex(index=vx_values, columns=torque_values)
            .to_numpy(dtype=np.float32)
        )
        image = axis.imshow(
            pivot,
            origin="lower",
            aspect="auto",
            extent=[
                float(torque_values[0]),
                float(torque_values[-1]),
                float(vx_values[0]),
                float(vx_values[-1]),
            ],
        )
        axis.set_title(label)
        axis.set_xlabel("Rear torque sum (Nm)")
        axis.set_ylabel("Initial Vx (m/s)")
        axis.grid(False)
        fig.colorbar(image, ax=axis, shrink=0.88)

    fig.suptitle(title, fontsize=15)
    save_inference_figure(fig, output_path, top_margin=0.94)


def plot_line_grid(
    frame: pd.DataFrame,
    columns: list[tuple[str, str]],
    title: str,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharex=True)
    axes = axes.ravel()
    for axis, (column, label) in zip(axes, columns, strict=False):
        for vx_value, group in frame.groupby("initial_vx_t_mps", sort=True):
            group = group.sort_values("rear_drive_torque_sum_nm")
            axis.plot(
                group["rear_drive_torque_sum_nm"],
                group[column],
                linewidth=1.2,
                label=f"Vx={vx_value:g}",
            )
        axis.set_title(label)
        axis.set_xlabel("Rear torque sum (Nm)")
        axis.grid(True, linestyle="--", alpha=0.35)
    axes[0].set_ylabel("Value")
    axes[3].set_ylabel("Value")
    axes[-1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8)
    fig.suptitle(title, fontsize=15)
    save_inference_figure(fig, output_path, top_margin=0.94)


def write_metadata(output_dir: Path, metadata: dict[str, object], args: argparse.Namespace) -> Path:
    output_clip = metadata.get("output_clip")
    lines = [
        "# Single-Step Grid Check",
        "",
        f"- checkpoint_path: `{metadata['resolved_checkpoint_path']}`",
        f"- dt_s: `{args.dt}`",
        f"- vx_grid_mps: `{args.vx_start}:{args.vx_step}:{args.vx_end}`",
        f"- rear_torque_sum_grid_nm: `{args.torque_start}:{args.torque_step}:{args.torque_end}`",
        f"- steer_sw_rad: `{args.steer_sw_rad}`",
        f"- trailer_mass_kg: `{args.trailer_mass_kg}`",
        "",
        "The requested rear torque is interpreted as total rear-drive torque and is split equally to rear-left and rear-right wheels.",
    ]
    if output_clip is not None:
        lines.extend(["", "## MLP Output Clip"])
        for name, value in zip(MLP_OUTPUT_NAMES, np.asarray(output_clip).reshape(-1), strict=False):
            lines.append(f"- `{name}`: +/- `{float(value):.9g}`")
    output_path = output_dir / "single_step_grid_check_summary.md"
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def main() -> None:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")
    output_dir = Path(args.output_dir) if args.output_dir is not None else default_output_dir(checkpoint_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if args.device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))
    vx_values = inclusive_grid(args.vx_start, args.vx_end, args.vx_step)
    torque_values = inclusive_grid(args.torque_start, args.torque_end, args.torque_step)
    states, controls, masses, grid_rows = build_state_control_grid(
        vx_values=vx_values,
        torque_values=torque_values,
        steer_sw_rad=args.steer_sw_rad,
        trailer_mass_kg=args.trailer_mass_kg,
    )

    base_next, mlp_raw, mlp_clipped, nn_full_error, nn_next, metadata = run_grid_inference(
        checkpoint_path=checkpoint_path,
        states=states,
        controls=controls,
        masses=masses,
        dt_value=args.dt,
        device=device,
    )

    frame = build_results_dataframe(
        grid_rows=grid_rows,
        states=states,
        controls=controls,
        dt_value=args.dt,
        base_next=base_next,
        mlp_raw=mlp_raw,
        mlp_clipped=mlp_clipped,
        nn_full_error=nn_full_error,
        nn_next=nn_next,
    )
    csv_path = output_dir / "single_step_grid_results.csv"
    frame.to_csv(csv_path, index=False, encoding="utf-8-sig")

    base_columns = [
        ("base_delta_x_t", "Base delta x_t (m)"),
        ("base_delta_vx_t", "Base delta vx_t (m/s)"),
        ("base_delta_y_t", "Base delta y_t (m)"),
        ("base_delta_vy_t", "Base delta vy_t (m/s)"),
        ("base_delta_psi_t_deg", "Base delta yaw (deg)"),
        ("base_delta_r_t_degps", "Base delta yaw rate (deg/s)"),
    ]
    mlp_columns = [
        ("mlp_clipped_vx_t", "MLP vx_t residual (m/s)"),
        ("mlp_clipped_vy_t", "MLP vy_t residual (m/s)"),
        ("mlp_clipped_r_t_degps", "MLP r_t residual (deg/s)"),
        ("mlp_clipped_vx_s", "MLP vx_s residual (m/s)"),
        ("mlp_clipped_vy_s", "MLP vy_s residual (m/s)"),
        ("mlp_clipped_r_s_degps", "MLP r_s residual (deg/s)"),
    ]
    correction_columns = [
        ("nn_minus_base_x_t", "NN correction x_t (m)"),
        ("nn_minus_base_vx_t", "NN correction vx_t (m/s)"),
        ("nn_minus_base_y_t", "NN correction y_t (m)"),
        ("nn_minus_base_vy_t", "NN correction vy_t (m/s)"),
        ("nn_minus_base_psi_t_deg", "NN correction yaw (deg)"),
        ("nn_minus_base_r_t_degps", "NN correction yaw rate (deg/s)"),
    ]

    plot_heatmap_grid(frame, vx_values, torque_values, base_columns, "Base Model Independent One-Step Output", output_dir / "base_independent_outputs_heatmap.png")
    plot_heatmap_grid(frame, vx_values, torque_values, mlp_columns, "Residual MLP Independent One-Step Output", output_dir / "mlp_independent_outputs_heatmap.png")
    plot_heatmap_grid(frame, vx_values, torque_values, correction_columns, "Base + NN Correction Relative To Base", output_dir / "nn_corrections_heatmap.png")
    plot_line_grid(frame, base_columns, "Base Model Independent One-Step Output", output_dir / "base_independent_outputs_lines.png")
    plot_line_grid(frame, mlp_columns, "Residual MLP Independent One-Step Output", output_dir / "mlp_independent_outputs_lines.png")
    plot_line_grid(frame, correction_columns, "Base + NN Correction Relative To Base", output_dir / "nn_corrections_lines.png")
    summary_path = write_metadata(output_dir, metadata, args)

    print(f"device       : {device}")
    print(f"checkpoint   : {metadata['resolved_checkpoint_path']}")
    print(f"grid rows    : {len(frame)}")
    print(f"results csv  : {csv_path}")
    print(f"summary md   : {summary_path}")
    print(f"plot dir     : {output_dir}")


if __name__ == "__main__":
    main()
