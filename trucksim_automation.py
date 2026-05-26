from __future__ import annotations

import argparse
import re
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


@dataclass(frozen=True)
class PulseSegment:
    """单个分段控制：在 [start_s, end_s) 时间内输出固定值。"""

    start_s: float
    end_s: float
    value: float

    def validate(self) -> None:
        if self.end_s <= self.start_s:
            raise ValueError(f"无效控制段: end_s({self.end_s}) 必须大于 start_s({self.start_s})")


@dataclass
class ControlPlan:
    """方向盘和四轮驱动/制动力矩控制计划。"""

    steer_segments: list[PulseSegment] = field(default_factory=list)
    wheel_torque_segments: dict[str, list[PulseSegment]] = field(default_factory=dict)
    tstop_s: float = 10.0

    def validate(self) -> None:
        for segment in self.steer_segments:
            segment.validate()
        for wheel_name, segments in self.wheel_torque_segments.items():
            if wheel_name not in {"L1", "R1", "L2", "R2"}:
                raise ValueError(f"未知车轮名称: {wheel_name}，只支持 L1/R1/L2/R2")
            for segment in segments:
                segment.validate()


DEFAULT_BASE_PAR = Path(__file__).with_name("LastRun_ECHO.PAR")
DEFAULT_INSTALL_DIR = Path(r"D:\carsim")
DEFAULT_DATABASE_DIR = Path(r"D:\documents\CarSim2021.0_Data")
DEFAULT_WORK_ROOT = Path(__file__).with_name("carsim_runs")
WHEEL_ORDER = ("L1", "R1", "L2", "R2")


def build_default_control_plan() -> ControlPlan:
    """
    这里给一个可直接运行的示例：
    1) 0.5~6.0 s 给前轮施加驱动力矩，让车起步并前进；
    2) 1.5~3.5 s 给一个正方向盘转角；
    3) 3.5~5.0 s 给一个反向方向盘转角，形成一次 S 形转向。

    如果你要做自己的工况，直接改这里的数值即可。
    """

    return ControlPlan(
        tstop_s=10.0,
        steer_segments=[
            PulseSegment(1.5, 3.5, 30.0),
            PulseSegment(3.5, 5.0, -18.0),
        ],
        wheel_torque_segments={
            "L1": [PulseSegment(0.5, 6.0, 450.0)],
            "R1": [PulseSegment(0.5, 6.0, 450.0)],
            "L2": [PulseSegment(0.5, 6.0, 0.0)],
            "R2": [PulseSegment(0.5, 6.0, 0.0)],
        },
    )


def remove_last_end_block(par_text: str) -> str:
    """去掉原始 parsfile 最后一个 END，方便追加新的 VS Commands。"""

    match = re.search(r"\nEND\s*$", par_text, flags=re.IGNORECASE)
    if not match:
        raise ValueError("基础 PAR 文件末尾没有找到 END，无法自动追加控制命令。")
    return par_text[: match.start()].rstrip()


def build_pulse_expression(segments: list[PulseSegment]) -> str:
    """
    把 Python 控制段转成 CarSim VS Commands 表达式。
    使用两个 IF_GT_0_THEN 叠加得到一个矩形脉冲。
    """

    if not segments:
        return "0"

    terms: list[str] = []
    for segment in segments:
        segment.validate()
        value = f"{segment.value:.6f}"
        start = f"{segment.start_s:.6f}"
        end = f"{segment.end_s:.6f}"
        terms.append(
            f"(IF_GT_0_THEN(T - {start}, {value}, 0) - IF_GT_0_THEN(T - {end}, {value}, 0))"
        )
    return " + ".join(terms)


def build_injected_vs_commands(plan: ControlPlan) -> str:
    """构造注入到基础 PAR 末尾的控制语句。"""

    plan.validate()
    steer_expression = build_pulse_expression(plan.steer_segments)

    lines = [
        "",
        "! ---------------- Python injected control block ----------------",
        "! 关闭 CarSim 自带驾驶员和车速控制器，改用下方导入量进行开环控制",
        "OPT_DM 0",
        "OPT_DRIVER_ACTION 0",
        "OPT_SC 0",
        "OPT_ERROR_DIALOG 0",
        "OPT_PAUSE 0",
        "OPT_VS_FILETYPE 4",
        f"TSTOP {plan.tstop_s:.6f}",
        "",
        "! 激活方向盘角导入量（直接替换内部方向盘命令）",
        "IMPORT IMP_STEER_SW VS_REPLACE 0 ; deg",
        f"EQ_IN IMP_STEER_SW = {steer_expression};",
        "",
        "! 激活四个车轮的外加轮端力矩导入量（单位 N-m）",
    ]

    for wheel_name in WHEEL_ORDER:
        torque_segments = plan.wheel_torque_segments.get(wheel_name, [])
        torque_expression = build_pulse_expression(torque_segments)
        lines.append(f"IMPORT IMP_MYSM_{wheel_name} VS_REPLACE 0 ; N-m")
        lines.append(f"EQ_IN IMP_MYSM_{wheel_name} = {torque_expression};")

    lines.extend(
        [
            "",
            "! 为后处理增加两个辅助输出（m/s）",
            "DEFINE_OUTPUT Vx_mps = Vx / 3.6; m/s; Longitudinal speed in SI",
            "DEFINE_OUTPUT Vy_mps = Vy / 3.6; m/s; Lateral speed in SI",
            "",
            "END",
        ]
    )
    return "\n".join(lines) + "\n"


def create_run_folder(work_root: Path) -> tuple[str, Path]:
    """创建本次仿真的独立工作目录。"""

    run_name = f"python_run_{datetime.now():%Y%m%d_%H%M%S}"
    run_dir = work_root / run_name
    (run_dir / "Results").mkdir(parents=True, exist_ok=True)
    (run_dir / "Runs").mkdir(parents=True, exist_ok=True)
    (run_dir / "outputs").mkdir(parents=True, exist_ok=True)
    (run_dir / "plots").mkdir(parents=True, exist_ok=True)
    return run_name, run_dir


def write_custom_par_file(base_par: Path, run_dir: Path, plan: ControlPlan) -> Path:
    """写出本次仿真专用的扩展 PAR 文件。"""

    raw_text = base_par.read_text(encoding="utf-8", errors="ignore")
    par_body = remove_last_end_block(raw_text)
    injected_text = build_injected_vs_commands(plan)
    custom_par_path = run_dir / "custom_run.par"
    custom_par_path.write_text(par_body + "\n" + injected_text, encoding="utf-8")
    return custom_par_path


def write_sim_file(
    run_name: str,
    run_dir: Path,
    install_dir: Path,
    database_dir: Path,
    custom_par_path: Path,
    solver_bits: int = 32,
) -> tuple[Path, Path]:
    """写出 VS_SolverWrapper_CLI 使用的 simfile。"""

    wrapper_exe = (
        install_dir
        / "Programs"
        / f"VS_SolverWrapper_CLI_{solver_bits}"
        / f"VS_SolverWrapper_CLI_{solver_bits}.exe"
    )
    solver_dll = install_dir / "Programs" / "solvers" / f"carsim_{solver_bits}.dll"
    sim_path = run_dir / "custom_run.sim"

    sim_text = (
        "SIMFILE\n\n"
        f"SET_MACRO $(ROOT_FILE_NAME)$ {run_name}\n"
        "SET_MACRO $(OUTPUT_PATH)$ Results\n"
        "SET_MACRO $(WORK_DIR)$ .\\\n"
        "SET_MACRO $(OUTPUT_FILE_PREFIX)$ $(OUTPUT_PATH)$\\$(ROOT_FILE_NAME)$\\LastRun\n\n"
        "FILEBASE $(OUTPUT_FILE_PREFIX)$\n"
        f"INPUT {custom_par_path.name}\n"
        "INPUTARCHIVE $(OUTPUT_FILE_PREFIX)$_all.par\n"
        "ECHO $(OUTPUT_FILE_PREFIX)$_echo.par\n"
        "FINAL $(OUTPUT_FILE_PREFIX)$_end.par\n"
        "LOGFILE $(OUTPUT_FILE_PREFIX)$_log.txt\n"
        "ERDFILE $(OUTPUT_FILE_PREFIX)$.vs\n"
        f"PROGDIR {install_dir}\\\n"
        f"DATADIR {database_dir}\\\n"
        f"RESOURCEDIR {install_dir}\\Resources\\\n"
        "PRODUCT_ID CarSim\n"
        "PRODUCT_VER 2021.0\n"
        "ANIFILE Runs\\animator.par\n"
        "VEHICLE_CODE i_i\n"
        "EXT_MODEL_STEP 0.00050000\n"
        "PORTS_IMP 0\n"
        "PORTS_EXP 0\n"
        f"DLLFILE {solver_dll}\n"
        "END\n"
    )

    sim_path.write_text(sim_text, encoding="utf-8")
    return sim_path, wrapper_exe


def run_solver(run_dir: Path, sim_path: Path, wrapper_exe: Path) -> Path:
    """
    调用 CarSim 命令行求解器。
    注意：CarSim 2021.0 需要本机已打开 CarSim Browser 或 License Manager，
    否则求解器会因为许可证不可用而退出。
    """

    if not wrapper_exe.exists():
        raise FileNotFoundError(f"未找到求解器包装器: {wrapper_exe}")

    command = [str(wrapper_exe), "-sim", sim_path.name, "-exit"]
    result = subprocess.run(
        command,
        cwd=run_dir,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="ignore",
        check=False,
    )

    combined_output = "\n".join(part for part in [result.stdout, result.stderr] if part)
    if result.returncode != 0:
        if "CarSim Browser" in combined_output or "License Manager" in combined_output:
            raise RuntimeError(
                "CarSim 求解器许可证不可用。请先打开 CarSim Browser 或 CarSim License Manager，"
                "确认 2021.0 求解器许可证可用后再运行。"
            )
        raise RuntimeError(f"CarSim 求解器执行失败。\n{combined_output}")

    csv_path = run_dir / "Results" / run_dir.name / "LastRun.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"求解器已执行，但未生成 CSV：{csv_path}\n"
            "请确认 CarSim 是否成功运行，以及 OPT_VS_FILETYPE 是否被设置为 4。"
        )
    return csv_path


def find_column(frame: pd.DataFrame, candidates: list[str]) -> str:
    """按候选名称查找列名，兼容大小写。"""

    for candidate in candidates:
        if candidate in frame.columns:
            return candidate

    lower_map = {column.lower(): column for column in frame.columns}
    for candidate in candidates:
        mapped = lower_map.get(candidate.lower())
        if mapped is not None:
            return mapped

    raise KeyError(f"CSV 中未找到这些列: {candidates}")


def extract_signals(csv_path: Path) -> pd.DataFrame:
    """从 CarSim CSV 中提取用户关心的关键信号。"""

    frame = pd.read_csv(csv_path)

    time_col = find_column(frame, ["Time", "T"])
    vx_col = find_column(frame, ["Vx", "Vx_mps"])
    vy_col = find_column(frame, ["Vy", "Vy_mps"])
    x_col = find_column(frame, ["Xo", "XCG_TM", "XCG_SM"])
    y_col = find_column(frame, ["Yo", "YCG_TM", "YCG_SM"])
    yaw_col = find_column(frame, ["Yaw", "Psi", "Yaw_SM"])
    yaw_rate_col = find_column(frame, ["AVz", "AV_Y", "YawRate"])

    selected = pd.DataFrame(
        {
            "Time_s": frame[time_col],
            "Vx_mps": frame[vx_col] if vx_col == "Vx_mps" else frame[vx_col] / 3.6,
            "Vy_mps": frame[vy_col] if vy_col == "Vy_mps" else frame[vy_col] / 3.6,
            "X_m": frame[x_col],
            "Y_m": frame[y_col],
            "Yaw_deg": frame[yaw_col],
            "YawRate_degps": frame[yaw_rate_col],
        }
    )
    selected["Speed_mps"] = (selected["Vx_mps"] ** 2 + selected["Vy_mps"] ** 2) ** 0.5
    return selected


def save_signal_tables(signals: pd.DataFrame, run_dir: Path) -> tuple[Path, Path]:
    """保存完整结果表和简短预览表。"""

    full_table_path = run_dir / "outputs" / "selected_signals.csv"
    preview_table_path = run_dir / "outputs" / "selected_signals_preview.csv"

    signals.to_csv(full_table_path, index=False, encoding="utf-8-sig")
    signals.head(30).to_csv(preview_table_path, index=False, encoding="utf-8-sig")
    return full_table_path, preview_table_path


def plot_signals(signals: pd.DataFrame, run_dir: Path) -> Path:
    """生成速度、位移、姿态的汇总图。"""

    figure, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].plot(signals["X_m"], signals["Y_m"], linewidth=2)
    axes[0, 0].set_title("Vehicle trajectory")
    axes[0, 0].set_xlabel("X [m]")
    axes[0, 0].set_ylabel("Y [m]")
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(signals["Time_s"], signals["Vx_mps"], label="Vx")
    axes[0, 1].plot(signals["Time_s"], signals["Vy_mps"], label="Vy")
    axes[0, 1].plot(signals["Time_s"], signals["Speed_mps"], label="Speed")
    axes[0, 1].set_title("Vehicle speed")
    axes[0, 1].set_xlabel("Time [s]")
    axes[0, 1].set_ylabel("Speed [m/s]")
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()

    axes[1, 0].plot(signals["Time_s"], signals["X_m"], label="X")
    axes[1, 0].plot(signals["Time_s"], signals["Y_m"], label="Y")
    axes[1, 0].set_title("Vehicle displacement")
    axes[1, 0].set_xlabel("Time [s]")
    axes[1, 0].set_ylabel("Displacement [m]")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()

    axes[1, 1].plot(signals["Time_s"], signals["Yaw_deg"], label="Yaw")
    axes[1, 1].plot(signals["Time_s"], signals["YawRate_degps"], label="Yaw rate")
    axes[1, 1].set_title("Yaw motion")
    axes[1, 1].set_xlabel("Time [s]")
    axes[1, 1].set_ylabel("Angle / Rate [deg, deg/s]")
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()

    figure.tight_layout()
    plot_path = run_dir / "plots" / "simulation_summary.png"
    figure.savefig(plot_path, dpi=160)
    plt.close(figure)
    return plot_path


def sample_plan_value(time_s: float, segments: list[PulseSegment]) -> float:
    """给预览图采样时，计算某一时刻的控制值。"""

    return sum(segment.value for segment in segments if segment.start_s <= time_s < segment.end_s)


def export_control_preview(plan: ControlPlan, run_dir: Path, sample_step_s: float = 0.05) -> tuple[Path, Path]:
    """在真正调用 CarSim 前，先把控制输入画出来，便于核对。"""

    total_steps = int(plan.tstop_s / sample_step_s) + 1
    times = [round(index * sample_step_s, 6) for index in range(total_steps)]

    preview_data = {
        "Time_s": times,
        "Steer_deg": [sample_plan_value(time_s, plan.steer_segments) for time_s in times],
    }
    for wheel_name in WHEEL_ORDER:
        preview_data[f"Torque_{wheel_name}_Nm"] = [
            sample_plan_value(time_s, plan.wheel_torque_segments.get(wheel_name, [])) for time_s in times
        ]

    preview_frame = pd.DataFrame(preview_data)
    preview_csv = run_dir / "outputs" / "control_preview.csv"
    preview_png = run_dir / "plots" / "control_preview.png"
    preview_frame.to_csv(preview_csv, index=False, encoding="utf-8-sig")

    figure, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    axes[0].plot(preview_frame["Time_s"], preview_frame["Steer_deg"], color="tab:blue", linewidth=2)
    axes[0].set_ylabel("Steer [deg]")
    axes[0].set_title("Command preview")
    axes[0].grid(True, alpha=0.3)

    for wheel_name in WHEEL_ORDER:
        axes[1].plot(
            preview_frame["Time_s"],
            preview_frame[f"Torque_{wheel_name}_Nm"],
            linewidth=2,
            label=wheel_name,
        )
    axes[1].set_xlabel("Time [s]")
    axes[1].set_ylabel("Wheel torque [N-m]")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    figure.tight_layout()
    figure.savefig(preview_png, dpi=160)
    plt.close(figure)
    return preview_csv, preview_png


def extract_vehicle_parameters(base_par: Path, run_dir: Path) -> Path:
    """
    从现有基础 PAR 中提取你提到的“几何尺寸 / 质量 / 传动比”等常用参数，
    方便与仿真结果一起保存归档。
    """

    keywords = [
        "H_CG_SU",
        "LX_CG_SU",
        "M_SU",
        "LEN_SM",
        "WID_SM",
        "HT_SM",
        "LX_AXLE(2)",
        "L_TRACK(1)",
        "L_TRACK(2)",
        "R_GEAR_DIFF(1)",
        "R_GEAR_TR(1)",
        "R_GEAR_TR(2)",
        "R_GEAR_TR(3)",
        "R_GEAR_TR(4)",
        "R_GEAR_TR(5)",
        "R_GEAR_TR(6)",
    ]

    content = base_par.read_text(encoding="utf-8", errors="ignore")
    rows: list[dict[str, str]] = []
    for keyword in keywords:
        pattern = rf"^{re.escape(keyword)}\s+([-+0-9.Ee]+)\s*;\s*([^\s!]+)"
        match = re.search(pattern, content, flags=re.MULTILINE)
        if match:
            rows.append(
                {
                    "Parameter": keyword,
                    "Value": match.group(1),
                    "Unit": match.group(2),
                }
            )

    parameter_path = run_dir / "outputs" / "vehicle_parameters.csv"
    pd.DataFrame(rows).to_csv(parameter_path, index=False, encoding="utf-8-sig")
    return parameter_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="通过 Python 调用本地 CarSim 2021.0，注入方向盘/四轮力矩控制并导出图表。"
    )
    parser.add_argument("--base-par", type=Path, default=DEFAULT_BASE_PAR, help="基础扩展 PAR 文件路径")
    parser.add_argument("--install-dir", type=Path, default=DEFAULT_INSTALL_DIR, help="CarSim 安装目录")
    parser.add_argument("--database-dir", type=Path, default=DEFAULT_DATABASE_DIR, help="CarSim 数据库目录")
    parser.add_argument("--work-root", type=Path, default=DEFAULT_WORK_ROOT, help="仿真工作目录根路径")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只生成控制文件、预览图和车辆参数表，不真正调用 CarSim 求解器",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.base_par.exists():
        raise FileNotFoundError(f"基础 PAR 文件不存在: {args.base_par}")

    plan = build_default_control_plan()
    run_name, run_dir = create_run_folder(args.work_root)

    parameter_path = extract_vehicle_parameters(args.base_par, run_dir)
    preview_csv, preview_png = export_control_preview(plan, run_dir)
    custom_par_path = write_custom_par_file(args.base_par, run_dir, plan)
    sim_path, wrapper_exe = write_sim_file(
        run_name=run_name,
        run_dir=run_dir,
        install_dir=args.install_dir,
        database_dir=args.database_dir,
        custom_par_path=custom_par_path,
        solver_bits=32,
    )

    print(f"[1/6] 已提取车辆参数: {parameter_path}")
    print(f"[2/6] 已生成控制预览表: {preview_csv}")
    print(f"[3/6] 已生成控制预览图: {preview_png}")
    print(f"[4/6] 已生成仿真 PAR: {custom_par_path}")
    print(f"[5/6] 已生成 SIM 文件: {sim_path}")

    if args.dry_run:
        print("[6/6] dry-run 模式：未调用 CarSim 求解器。")
        print(f"工作目录: {run_dir}")
        return

    csv_path = run_solver(run_dir=run_dir, sim_path=sim_path, wrapper_exe=wrapper_exe)
    signals = extract_signals(csv_path)
    full_table_path, preview_table_path = save_signal_tables(signals, run_dir)
    summary_plot_path = plot_signals(signals, run_dir)

    print(f"[6/6] 仿真完成，原始 CSV: {csv_path}")
    print(f"提取后的结果表: {full_table_path}")
    print(f"结果预览表: {preview_table_path}")
    print(f"结果图: {summary_plot_path}")
    print(f"工作目录: {run_dir}")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"执行失败: {error}")
        raise SystemExit(1)
