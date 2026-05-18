# Truck-Trailer Residual Modular：实验复现说明

这份文档只关注“怎么把当前实验重新跑出来”，把训练命令、开环验证命令，以及结果文件路径单独列出来，方便直接照着执行。

## 1. 复现前提

默认工作目录：

```text
d:\test_torch project
```

以下命令都默认从仓库根目录执行。

当前这批 `train_segment` 数据的前提口径是：

- 当前样本按无挂车模式处理
- `Steer_deg_cmd` 作为方向盘转角输入
- 不使用 `Target_Steer_L1_deg_cmd`
- 后轮驱动输入使用 `Torque_L2_Nm_cmd + Torque_R2_Nm_cmd`
- 固定 `dt = 0.02 s`

数据根目录：

```text
controltest/truck_trailer_residual_modular/data/data
```

## 2. 复现 100 epoch 训练

### 2.1 训练命令

```bash
python controltest/truck_trailer_residual_modular/train_main.py ^
  --input-path "controltest/truck_trailer_residual_modular/data/data" ^
  --epochs 100 ^
  --summary-dir "controltest/truck_trailer_residual_modular/data/train_segment_training_summary_100ep"
```

### 2.2 训练后的主要输出

模型 checkpoint：

- `controltest/truck_trailer_residual_modular/best_truck_trailer_error_model.pth`
- `controltest/truck_trailer_residual_modular/best_truck_trailer_error_model_train_loss.pth`

训练汇总目录：

- `controltest/truck_trailer_residual_modular/data/train_segment_training_summary_100ep`

其中重点文件：

- `truck_trailer_training_loss_log.png`
- `truck_trailer_dataset_split_segments.csv`
- `truck_trailer_validation_segments.csv`

## 3. 复现四类场景的开环验证

### 3.1 前提

开环验证默认读取：

```text
controltest/truck_trailer_residual_modular/best_truck_trailer_error_model.pth
```

因此请先完成第 2 节训练，保证当前 checkpoint 和当前代码结构一致。

### 3.2 accelerate 片段

命令：

```bash
python controltest/truck_trailer_residual_modular/inference_main.py ^
  --input-path "controltest/truck_trailer_residual_modular/data/data/accelerate/0-30kph/20260417_155250.00000_interpolated_train_segment_001.csv"
```

输出目录：

- `controltest/truck_trailer_residual_modular/data/data/accelerate/0-30kph/truck_trailer_open_loop_eval_modular`

输出文件：

- `open_loop_results.csv`
- `controls.png`
- `open_loop_trajectory.png`
- `open_loop_state_errors.png`
- `controltest/truck_trailer_residual_modular/data/data/accelerate/0-30kph/truck_trailer_open_loop_summary_modular.csv`

### 3.3 decelerate 片段

命令：

```bash
python controltest/truck_trailer_residual_modular/inference_main.py ^
  --input-path "controltest/truck_trailer_residual_modular/data/data/decelerate/15kph-0/20260417_154100.00000_interpolated_train_segment_001.csv"
```

输出目录：

- `controltest/truck_trailer_residual_modular/data/data/decelerate/15kph-0/truck_trailer_open_loop_eval_modular`

输出文件：

- `open_loop_results.csv`
- `controls.png`
- `open_loop_trajectory.png`
- `open_loop_state_errors.png`
- `controltest/truck_trailer_residual_modular/data/data/decelerate/15kph-0/truck_trailer_open_loop_summary_modular.csv`

### 3.4 turn 片段

命令：

```bash
python controltest/truck_trailer_residual_modular/inference_main.py ^
  --input-path "controltest/truck_trailer_residual_modular/data/data/turn/20kph/20260417_162035.00000_interpolated_train_segment_001.csv"
```

输出目录：

- `controltest/truck_trailer_residual_modular/data/data/turn/20kph/truck_trailer_open_loop_eval_modular`

输出文件：

- `open_loop_results.csv`
- `controls.png`
- `open_loop_trajectory.png`
- `open_loop_state_errors.png`
- `controltest/truck_trailer_residual_modular/data/data/turn/20kph/truck_trailer_open_loop_summary_modular.csv`

### 3.5 lane_change 片段

命令：

```bash
python controltest/truck_trailer_residual_modular/inference_main.py ^
  --input-path "controltest/truck_trailer_residual_modular/data/data/lane_change/20kph/20260417_172921.00003_interpolated_train_segment_001.csv"
```

输出目录：

- `controltest/truck_trailer_residual_modular/data/data/lane_change/20kph/truck_trailer_open_loop_eval_modular`

输出文件：

- `open_loop_results.csv`
- `controls.png`
- `open_loop_trajectory.png`
- `open_loop_state_errors.png`
- `controltest/truck_trailer_residual_modular/data/data/lane_change/20kph/truck_trailer_open_loop_summary_modular.csv`

## 4. 当前这次 100 epoch 实验已经存在的结果文件

如果只是查看当前仓库里已经生成好的结果，可以直接看这些路径。

### 4.1 训练结果

- `controltest/truck_trailer_residual_modular/data/train_segment_training_summary_100ep/truck_trailer_training_loss_log.png`
- `controltest/truck_trailer_residual_modular/data/train_segment_training_summary_100ep/truck_trailer_dataset_split_segments.csv`
- `controltest/truck_trailer_residual_modular/data/train_segment_training_summary_100ep/truck_trailer_validation_segments.csv`

### 4.2 四类开环汇总表

- `controltest/truck_trailer_residual_modular/data/train_segment_training_summary_100ep/open_loop_validation_summary_100ep.csv`

说明：

- 这份 `open_loop_validation_summary_100ep.csv` 是当前实验整理后的四类场景汇总表
- `inference_main.py` 本身会为每个场景单独生成 `truck_trailer_open_loop_summary_modular.csv`

### 4.3 各场景单独输出

accelerate：

- `controltest/truck_trailer_residual_modular/data/data/accelerate/0-30kph/truck_trailer_open_loop_eval_modular`
- `controltest/truck_trailer_residual_modular/data/data/accelerate/0-30kph/truck_trailer_open_loop_summary_modular.csv`

decelerate：

- `controltest/truck_trailer_residual_modular/data/data/decelerate/15kph-0/truck_trailer_open_loop_eval_modular`
- `controltest/truck_trailer_residual_modular/data/data/decelerate/15kph-0/truck_trailer_open_loop_summary_modular.csv`

turn：

- `controltest/truck_trailer_residual_modular/data/data/turn/20kph/truck_trailer_open_loop_eval_modular`
- `controltest/truck_trailer_residual_modular/data/data/turn/20kph/truck_trailer_open_loop_summary_modular.csv`

lane_change：

- `controltest/truck_trailer_residual_modular/data/data/lane_change/20kph/truck_trailer_open_loop_eval_modular`
- `controltest/truck_trailer_residual_modular/data/data/lane_change/20kph/truck_trailer_open_loop_summary_modular.csv`

## 5. 推荐的复现顺序

建议按下面顺序执行：

1. 先运行第 2 节训练命令，得到与当前代码兼容的 checkpoint。
2. 再运行第 3 节中的四条开环验证命令。
3. 最后查看第 4 节列出的结果路径。

## 6. 相关文档

- 训练逻辑和参数关系说明见：`TRAINING_PROCESS.md`
