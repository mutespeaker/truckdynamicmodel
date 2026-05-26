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
    from .constants import CONTROL_NAMES, MLP_OUTPUT_NAMES, STATE_NAMES
    from .inference_main import save_inference_figure
    from .single_step_grid_check import (
        DEFAULT_CHECKPOINT,
        add_mlp_columns,
        add_state_columns,
        default_output_dir,
        run_grid_inference,
    )
except ImportError:
    from constants import CONTROL_NAMES, MLP_OUTPUT_NAMES, STATE_NAMES
    from inference_main import save_inference_figure
    from single_step_grid_check import (
        DEFAULT_CHECKPOINT,
        add_mlp_columns,
        add_state_columns,
        default_output_dir,
        run_grid_inference,
    )


DEFAULT_INPUT_CSV = Path(
    r"d:\WeChat\xwechat_files\wxid_dwdtn3werp5j32_d199\msg\file\2026-05\first_step_mlp_raw_io.csv",
)

INPUT_REQUIRED_COLUMNS = [
    "input_vx_t_mps",
    "input_vy_t_mps",
    "input_r_t_radps",
    "input_rel_x_m",
    "input_steer_sw_rad",
    "input_rear_torque_Nm",
]

OUTPUT_COMPARE_COLUMNS = {
    "vx_t": "output_raw_vx_t_mps",
    "vy_t": "output_raw_vy_t_mps",
    "r_t": "output_raw_r_t_radps",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run one-step Base and residual-MLP inference from a compact first-step MLP input/output CSV. "
            "Missing no-trailer placeholder features are reconstructed as vx_s/vy_s/r_s = vx_t/vy_t/r_t, "
            "rel_y = 0, rel_yaw = 0, trailer_mass = 0."
        ),
    )
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT_CSV, help="CSV containing input_* and output_raw_* columns.")
    parser.add_argument("--checkpoint-path", type=Path, default=DEFAULT_CHECKPOINT, help="Residual model checkpoint.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to <checkpoint run>/single_step_mlp_io_check/<csv_stem>.",
    )
    parser.add_argument("--dt", type=float, default=0.02, help="One-step integration time in seconds.")
    parser.add_argument(
        "--keep-negative-vx-in-plots",
        action="store_true",
        help="Keep rows whose one-step Base or NN tractor Vx result is negative in plots.",
    )
    parser.add_argument("--device", default=None, help="Torch device. Defaults to cuda when available, otherwise cpu.")
    return parser.parse_args()


def require_columns(frame: pd.DataFrame, columns: list[str]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Input CSV is missing required columns: {missing}")


def read_numeric(frame: pd.DataFrame, column: str, default: float = 0.0) -> np.ndarray:
    if column not in frame.columns:
        return np.full(len(frame), float(default), dtype=np.float32)
    values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=np.float32)
    return np.where(np.isfinite(values), values, float(default)).astype(np.float32)


def build_state_control_from_io_csv(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    require_columns(frame, INPUT_REQUIRED_COLUMNS)
    row_count = len(frame)
    states = np.zeros((row_count, len(STATE_NAMES)), dtype=np.float32)
    controls = np.zeros((row_count, len(CONTROL_NAMES)), dtype=np.float32)

    vx_t = read_numeric(frame, "input_vx_t_mps")
    vy_t = read_numeric(frame, "input_vy_t_mps")
    r_t = read_numeric(frame, "input_r_t_radps")
    rel_x = read_numeric(frame, "input_rel_x_m")

    states[:, STATE_NAMES.index("vx_t")] = vx_t
    states[:, STATE_NAMES.index("vy_t")] = vy_t
    states[:, STATE_NAMES.index("r_t")] = r_t
    states[:, STATE_NAMES.index("vx_s")] = vx_t
    states[:, STATE_NAMES.index("vy_s")] = vy_t
    states[:, STATE_NAMES.index("r_s")] = r_t
    # Absolute tractor pose is not part of the MLP input. Use zero tractor pose,
    # rel_y=0, rel_yaw=0, so trailer placeholder x_s is exactly rel_x.
    states[:, STATE_NAMES.index("x_s")] = rel_x

    controls[:, CONTROL_NAMES.index("steer_sw_rad")] = read_numeric(frame, "input_steer_sw_rad")
    rear_torque_sum = read_numeric(frame, "input_rear_torque_Nm")
    controls[:, CONTROL_NAMES.index("torque_rl")] = 0.5 * rear_torque_sum
    controls[:, CONTROL_NAMES.index("torque_rr")] = 0.5 * rear_torque_sum

    masses = np.zeros((row_count, 1), dtype=np.float32)
    return states, controls, masses


def output_dir_for_args(input_csv: Path, checkpoint_path: Path, output_dir: Path | None) -> Path:
    if output_dir is not None:
        return Path(output_dir)
    return default_output_dir(checkpoint_path).parent / "single_step_mlp_io_check" / input_csv.stem


def append_results_columns(
    input_frame: pd.DataFrame,
    states: np.ndarray,
    controls: np.ndarray,
    dt_value: float,
    base_next: np.ndarray,
    mlp_raw: np.ndarray,
    mlp_clipped: np.ndarray,
    nn_full_error: np.ndarray,
    nn_next: np.ndarray,
) -> pd.DataFrame:
    frame = input_frame.reset_index(drop=True).copy()
    frame.insert(0, "source_row_index", np.arange(len(frame), dtype=np.int32))
    frame["dt_s"] = float(dt_value)
    frame["assumed_trailer_mass_kg"] = 0.0
    frame["assumed_rel_y_m"] = 0.0
    frame["assumed_rel_yaw_rad"] = 0.0
    frame = add_state_columns(frame, "rebuilt_initial", states)
    for index, name in enumerate(CONTROL_NAMES):
        frame[f"rebuilt_control_{name}"] = controls[:, index].astype(np.float32)
    frame["rebuilt_rear_drive_torque_sum_nm"] = (
        controls[:, CONTROL_NAMES.index("torque_rl")] + controls[:, CONTROL_NAMES.index("torque_rr")]
    ).astype(np.float32)

    frame = add_state_columns(frame, "computed_base_next", base_next)
    frame = add_state_columns(frame, "computed_base_delta", base_next - states)
    frame = add_mlp_columns(frame, "computed_mlp_raw", mlp_raw)
    frame = add_mlp_columns(frame, "computed_mlp_clipped", mlp_clipped)
    frame = add_state_columns(frame, "computed_nn_residual_full_error", nn_full_error)
    frame = add_state_columns(frame, "computed_nn_next", nn_next)
    frame = add_state_columns(frame, "computed_nn_minus_base", nn_next - base_next)

    extra_columns: dict[str, np.ndarray] = {}
    for prefix in ("computed_base_delta", "computed_nn_residual_full_error", "computed_nn_minus_base"):
        extra_columns[f"{prefix}_psi_t_deg"] = np.rad2deg(frame[f"{prefix}_psi_t"].to_numpy(dtype=np.float32))
        extra_columns[f"{prefix}_r_t_degps"] = np.rad2deg(frame[f"{prefix}_r_t"].to_numpy(dtype=np.float32))

    for name, output_column in OUTPUT_COMPARE_COLUMNS.items():
        if output_column in frame.columns:
            original = pd.to_numeric(frame[output_column], errors="coerce").to_numpy(dtype=np.float32)
            computed = mlp_raw[:, MLP_OUTPUT_NAMES.index(name)]
            clipped = mlp_clipped[:, MLP_OUTPUT_NAMES.index(name)]
            extra_columns[f"diff_original_raw_minus_computed_raw_{name}"] = original - computed
            extra_columns[f"diff_original_raw_minus_computed_clipped_{name}"] = original - clipped

    frame = pd.concat([frame, pd.DataFrame(extra_columns)], axis=1)
    frame["negative_vx_single_step_result"] = (
        (frame["computed_base_next_vx_t"] < 0.0) | (frame["computed_nn_next_vx_t"] < 0.0)
    )
    return frame


def build_rowwise_comparison(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "source_row_index",
        "scenario_key",
        "sample_time_s",
        "negative_vx_single_step_result",
        "input_vx_t_mps",
        "input_vy_t_mps",
        "input_r_t_radps",
        "input_rel_x_m",
        "input_steer_sw_rad",
        "input_rear_torque_Nm",
        "computed_base_next_vx_t",
        "computed_base_next_vy_t",
        "computed_base_next_r_t",
        "computed_nn_next_vx_t",
        "computed_nn_next_vy_t",
        "computed_nn_next_r_t",
        "computed_base_delta_vx_t",
        "computed_base_delta_vy_t",
        "computed_base_delta_r_t_degps",
        "computed_nn_minus_base_vx_t",
        "computed_nn_minus_base_vy_t",
        "computed_nn_minus_base_r_t_degps",
    ]
    for name, output_column in OUTPUT_COMPARE_COLUMNS.items():
        columns.extend(
            [
                output_column,
                f"computed_mlp_raw_{name}",
                f"computed_mlp_clipped_{name}",
                f"diff_original_raw_minus_computed_raw_{name}",
                f"diff_original_raw_minus_computed_clipped_{name}",
            ],
        )
    return frame.loc[:, [column for column in columns if column in frame.columns]].copy()


def build_comparison_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    diff_columns = [
        column
        for column in frame.columns
        if column.startswith("diff_original_raw_minus_computed")
    ]
    for column in diff_columns:
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=np.float64)
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            continue
        rows.append(
            {
                "diff_column": column,
                "count": int(finite.size),
                "mean": float(np.mean(finite)),
                "mean_abs": float(np.mean(np.abs(finite))),
                "max_abs": float(np.max(np.abs(finite))),
                "p95_abs": float(np.percentile(np.abs(finite), 95)),
            },
        )
    return pd.DataFrame(rows).sort_values("max_abs", ascending=False).reset_index(drop=True)


def plot_by_row(
    frame: pd.DataFrame,
    columns: list[tuple[str, str]],
    title: str,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharex=True)
    axes = axes.ravel()
    groups = list(frame.groupby("scenario_key", sort=False)) if "scenario_key" in frame.columns else [("all", frame)]
    for axis, (column, label) in zip(axes, columns, strict=False):
        for group_name, group in groups:
            group = group.sort_values("source_row_index")
            axis.plot(
                group["source_row_index"],
                group[column],
                marker="o",
                markersize=3.2,
                linewidth=1.0,
                label=str(group_name),
            )
        axis.set_title(label)
        axis.set_xlabel("CSV row index")
        axis.grid(True, linestyle="--", alpha=0.35)
    axes[0].set_ylabel("Value")
    axes[3].set_ylabel("Value")
    axes[-1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8)
    fig.suptitle(title, fontsize=15)
    save_inference_figure(fig, output_path, top_margin=0.94)


def plot_original_vs_computed(frame: pd.DataFrame, output_path: Path) -> None:
    channels = [
        ("vx_t", "output_raw_vx_t_mps", "m/s"),
        ("vy_t", "output_raw_vy_t_mps", "m/s"),
        ("r_t", "output_raw_r_t_radps", "rad/s"),
    ]
    fig, axes = plt.subplots(3, 1, figsize=(16, 11), sharex=True)
    x = frame["source_row_index"].to_numpy(dtype=np.float32)
    for axis, (name, original_column, unit) in zip(axes, channels, strict=False):
        axis.plot(x, frame[original_column], label="Original output_raw", marker="o", markersize=3, linewidth=1.1)
        axis.plot(x, frame[f"computed_mlp_raw_{name}"], label="Computed MLP raw", marker=".", linewidth=1.1)
        axis.plot(x, frame[f"computed_mlp_clipped_{name}"], label="Computed MLP clipped", linestyle="--", linewidth=1.0)
        axis.set_title(f"{name} original vs computed ({unit})")
        axis.grid(True, linestyle="--", alpha=0.35)
        axis.legend()
    axes[-1].set_xlabel("CSV row index")
    fig.suptitle("Original Out Columns Compared With Current Checkpoint Output", fontsize=15)
    save_inference_figure(fig, output_path, top_margin=0.94)


def plot_diff_by_row(frame: pd.DataFrame, output_path: Path) -> None:
    columns = [
        ("diff_original_raw_minus_computed_raw_vx_t", "orig raw - computed raw vx_t"),
        ("diff_original_raw_minus_computed_raw_vy_t", "orig raw - computed raw vy_t"),
        ("diff_original_raw_minus_computed_raw_r_t", "orig raw - computed raw r_t"),
        ("diff_original_raw_minus_computed_clipped_vx_t", "orig raw - computed clipped vx_t"),
        ("diff_original_raw_minus_computed_clipped_vy_t", "orig raw - computed clipped vy_t"),
        ("diff_original_raw_minus_computed_clipped_r_t", "orig raw - computed clipped r_t"),
    ]
    plot_by_row(frame, columns, "Original Out Difference By Row", output_path)


def write_summary(
    output_dir: Path,
    input_csv: Path,
    checkpoint_path: str,
    result_csv: Path,
    rowwise_comparison_csv: Path,
    comparison_summary_csv: Path,
    row_count: int,
    plotted_row_count: int,
    negative_vx_row_count: int,
    dt_value: float,
) -> Path:
    lines = [
        "# Single-Step MLP IO Check",
        "",
        f"- input_csv: `{input_csv}`",
        f"- checkpoint_path: `{checkpoint_path}`",
        f"- row_count: `{row_count}`",
        f"- plotted_row_count_after_negative_vx_filter: `{plotted_row_count}`",
        f"- negative_vx_single_step_result_rows: `{negative_vx_row_count}`",
        f"- dt_s: `{dt_value}`",
        f"- result_csv: `{result_csv}`",
        f"- rowwise_comparison_csv: `{rowwise_comparison_csv}`",
        f"- comparison_summary_csv: `{comparison_summary_csv}`",
        "",
        "Input columns are interpreted as compact first-step MLP raw inputs.",
        "`vx_s/vy_s/r_s` are mirrored from tractor states, `rel_y=0`, `rel_yaw=0`, and `trailer_mass=0`.",
        "Rear torque is treated as total rear-drive torque and split equally to rear-left and rear-right wheels.",
        "Rows whose computed one-step `base_next_vx_t` or `nn_next_vx_t` is negative are kept in CSV outputs but excluded from plots by default.",
    ]
    output_path = output_dir / "single_step_mlp_io_check_summary.md"
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def main() -> None:
    args = parse_args()
    input_csv = Path(args.input_csv)
    checkpoint_path = Path(args.checkpoint_path)
    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV does not exist: {input_csv}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")

    output_dir = output_dir_for_args(input_csv, checkpoint_path, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if args.device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))

    input_frame = pd.read_csv(input_csv)
    states, controls, masses = build_state_control_from_io_csv(input_frame)
    base_next, mlp_raw, mlp_clipped, nn_full_error, nn_next, metadata = run_grid_inference(
        checkpoint_path=checkpoint_path,
        states=states,
        controls=controls,
        masses=masses,
        dt_value=args.dt,
        device=device,
    )
    result_frame = append_results_columns(
        input_frame=input_frame,
        states=states,
        controls=controls,
        dt_value=args.dt,
        base_next=base_next,
        mlp_raw=mlp_raw,
        mlp_clipped=mlp_clipped,
        nn_full_error=nn_full_error,
        nn_next=nn_next,
    )

    result_csv = output_dir / f"{input_csv.stem}_single_step_mlp_io_results.csv"
    result_frame.to_csv(result_csv, index=False, encoding="utf-8-sig")
    rowwise_comparison = build_rowwise_comparison(result_frame)
    rowwise_comparison_csv = output_dir / f"{input_csv.stem}_rowwise_comparison.csv"
    rowwise_comparison.to_csv(rowwise_comparison_csv, index=False, encoding="utf-8-sig")
    comparison_summary = build_comparison_summary(result_frame)
    comparison_summary_csv = output_dir / f"{input_csv.stem}_comparison_summary.csv"
    comparison_summary.to_csv(comparison_summary_csv, index=False, encoding="utf-8-sig")

    negative_vx_mask = result_frame["negative_vx_single_step_result"].astype(bool)
    plot_frame = result_frame if args.keep_negative_vx_in_plots else result_frame.loc[~negative_vx_mask].copy()
    if len(plot_frame) == 0:
        raise ValueError("No rows remain for plotting after filtering negative one-step Vx results.")

    base_columns = [
        ("computed_base_delta_x_t", "Base delta x_t (m)"),
        ("computed_base_delta_vx_t", "Base delta vx_t (m/s)"),
        ("computed_base_delta_y_t", "Base delta y_t (m)"),
        ("computed_base_delta_vy_t", "Base delta vy_t (m/s)"),
        ("computed_base_delta_psi_t_deg", "Base delta yaw (deg)"),
        ("computed_base_delta_r_t_degps", "Base delta yaw rate (deg/s)"),
    ]
    mlp_columns = [
        ("computed_mlp_clipped_vx_t", "MLP vx_t residual (m/s)"),
        ("computed_mlp_clipped_vy_t", "MLP vy_t residual (m/s)"),
        ("computed_mlp_clipped_r_t", "MLP r_t residual (rad/s)"),
        ("computed_mlp_raw_vx_t", "MLP raw vx_t (m/s)"),
        ("computed_mlp_raw_vy_t", "MLP raw vy_t (m/s)"),
        ("computed_mlp_raw_r_t", "MLP raw r_t (rad/s)"),
    ]
    correction_columns = [
        ("computed_nn_minus_base_x_t", "NN correction x_t (m)"),
        ("computed_nn_minus_base_vx_t", "NN correction vx_t (m/s)"),
        ("computed_nn_minus_base_y_t", "NN correction y_t (m)"),
        ("computed_nn_minus_base_vy_t", "NN correction vy_t (m/s)"),
        ("computed_nn_minus_base_psi_t_deg", "NN correction yaw (deg)"),
        ("computed_nn_minus_base_r_t_degps", "NN correction yaw rate (deg/s)"),
    ]
    plot_suffix = "" if args.keep_negative_vx_in_plots else " (negative one-step Vx rows removed)"
    plot_by_row(plot_frame, base_columns, "Base Model Independent One-Step Output From MLP IO CSV" + plot_suffix, output_dir / "base_independent_outputs_by_row.png")
    plot_by_row(plot_frame, mlp_columns, "Residual MLP Independent Output From MLP IO CSV" + plot_suffix, output_dir / "mlp_independent_outputs_by_row.png")
    plot_by_row(plot_frame, correction_columns, "Base + NN Correction Relative To Base From MLP IO CSV" + plot_suffix, output_dir / "nn_corrections_by_row.png")
    plot_original_vs_computed(plot_frame, output_dir / "original_out_vs_computed_mlp_by_row.png")
    plot_diff_by_row(plot_frame, output_dir / "original_out_diff_by_row.png")

    summary_path = write_summary(
        output_dir=output_dir,
        input_csv=input_csv,
        checkpoint_path=str(metadata["resolved_checkpoint_path"]),
        result_csv=result_csv,
        rowwise_comparison_csv=rowwise_comparison_csv,
        comparison_summary_csv=comparison_summary_csv,
        row_count=len(result_frame),
        plotted_row_count=len(plot_frame),
        negative_vx_row_count=int(negative_vx_mask.sum()),
        dt_value=args.dt,
    )

    print(f"device      : {device}")
    print(f"input csv   : {input_csv}")
    print(f"checkpoint  : {metadata['resolved_checkpoint_path']}")
    print(f"rows        : {len(result_frame)}")
    print(f"plot rows   : {len(plot_frame)}")
    print(f"neg vx rows : {int(negative_vx_mask.sum())}")
    print(f"results csv : {result_csv}")
    print(f"row compare : {rowwise_comparison_csv}")
    print(f"diff summary: {comparison_summary_csv}")
    print(f"summary md  : {summary_path}")
    print(f"plot dir    : {output_dir}")


if __name__ == "__main__":
    main()
