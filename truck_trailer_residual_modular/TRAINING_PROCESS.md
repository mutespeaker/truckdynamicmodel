# Truck-Trailer Residual Modular：当前训练流程说明

本文档基于 `controltest/truck_trailer_residual_modular/` 当前代码整理，描述现在这套模型训练时的数据流、参数配置，以及各模块之间的逻辑关系。

它对应的是“当前实现说明”，不是早期脚本的历史说明。

## 1. 当前训练目标

当前方案是一个“名义模型 + 残差 MLP”的结构：

- `base_model.py` 负责名义动力学预测
- `model_structure.py` 里的 MLP 学习名义模型没有覆盖好的那一部分残差
- MLP 不直接接收绝对 `x/y`
- 牵引车的位置和航向由速度残差结合 `dt` 反推
- 挂车的位置和航向由“修正后的牵引车位姿 + 挂车相对位姿”重建

这样做的核心目的是让 MLP 尽量学习局部动力学误差，而不是去记忆世界坐标中的绝对位置。

## 2. 训练入口和整体流程

训练入口是：

```bash
python controltest/truck_trailer_residual_modular/train_main.py
```

如果要直接指定 `train_segment` 数据目录，可以这样跑：

```bash
python controltest/truck_trailer_residual_modular/train_main.py --input-path "controltest/truck_trailer_residual_modular/data/data"
```

整体流程如下：

```text
train_main.py
  -> 收集 CSV
  -> 每个 CSV 解析成 SegmentData
  -> 按 segment 划分 train / val
  -> 调用 base_model 计算名义下一步
  -> 构造 MLP 输入特征和监督目标
  -> 对输入做标准化
  -> 构造 loss scale / weight
  -> 训练 MLP
  -> 保存 best checkpoint
  -> 对验证集做 teacher-forcing rollout
  -> 导出图和汇总表
```

## 3. 数据来源解析逻辑

`train_main.py` 会调用 `data_utils.py` 中的 `collect_control_and_trajectory_csvs()`。

当前解析优先级是：

1. 如果 `--input-path` 是目录，且目录下存在 `*_train_segment_*.csv`，优先直接使用这些分段文件。
2. 如果没有 `train_segment`，再去找 `python_run_*/outputs/control_and_trajectory.csv`。
3. 如果 `--input-path` 直接指向某个 CSV，就只用这一个文件。
4. 如果没有传 `--input-path`，默认扫描 `controltest/carsim_runs`。

所以现在可以直接从下面这个目录训练：

```text
controltest/truck_trailer_residual_modular/data/data
```

不需要再套一层旧的 `python_run_*` 目录结构。

## 4. SegmentData 里有什么

每个 CSV 会被解析成一个 `SegmentData`，里面主要包括：

- `time`
- `states`
- `next_states`
- `controls`
- `trailer_mass_kg`
- `dt_values`
- `initial_state`
- `real_rollout`
- `control_sequence`

状态顺序固定为：

```text
[x_t, y_t, psi_t, vx_t, vy_t, r_t, x_s, y_s, psi_s, vx_s, vy_s, r_s]
```

其中当前代码约定：

- 牵引车 `x_t, y_t, vx_t, vy_t` 都表示牵引车后轴中心状态
- `base_model.py` 内部会先把牵引车后轴中心状态转换到名义动力学使用的内部参考点，再完成积分，并把结果转换回后轴中心

控制量顺序固定为：

```text
[steer_sw_rad, torque_fl, torque_fr, torque_rl, torque_rr]
```

## 5. 当前输入信号的解析规则

### 5.1 固定采样时间

当前代码假设数据是固定频率：

```text
FIXED_DT_S = 0.02
```

因此：

- `dt_values` 会直接填成固定的 `0.02`
- `dt` 仍然会传给名义模型和残差后处理
- `dt` 不作为 MLP 输入特征

## 5.2 方向盘转角

训练和推理都通过 `resolve_steering_wheel_angle_rad()` 来解析方向盘角。

优先级是：

1. 方向盘角 rad 列
2. 方向盘角 deg 列
3. 实在找不到时，才会退回到前轮转角列再乘转向比

当前一个重要约束是：

- 不再把 `Target_Steer_L1_deg_cmd` 当成方向盘转角输入

对目前这批 `train_segment` 数据来说，CSV 里没有 `Steer_SW_deg`，因此 100 epoch 那次实验实际使用的是：

```text
Steer_deg_cmd -> 方向盘转角
```

### 5.3 后轮驱动力矩

当前残差模型使用的后轴驱动输入是：

```text
rear_drive_torque_sum = Torque_L2_Nm_cmd + Torque_R2_Nm_cmd
```

在统一控制量布局里就是：

```text
rear_drive_torque_sum = torque_rl + torque_rr
```

这里是“求和”，不是“平均”。

### 5.4 是否带挂车

当前代码同时兼容“有挂车”和“无挂车”两种情况：

- 如果挂车状态列齐全，并且能解析到挂车质量，就走有挂车模式
- 否则就退回无挂车模式：
  - 挂车状态通道用牵引车状态镜像占位
  - `trailer_mass_kg = 0`

当前开关是：

```text
FORCE_NO_TRAILER_MODE = False
```

也就是说，代码层面支持挂车；只是当前这批 `train_segment` 数据因为挂车列不完整，实际训练时还是落在无挂车模式。

## 6. MLP 的输入和输出

### 6.1 输入特征

当前 MLP 输入是 14 维：

```text
[
  trailer_mass_kg,
  has_trailer,
  vx_t, vy_t, r_t,
  vx_s, vy_s, r_s,
  rel_x_s_t, rel_y_s_t,
  sin_rel_yaw_s_t, cos_rel_yaw_s_t,
  steer_sw_rad,
  rear_drive_torque_sum
]
```

几个关键点：

- 绝对 `x_t / y_t / x_s / y_s` 不进 MLP
- 挂车位姿不是直接喂绝对坐标，而是喂“挂车相对牵引车”的位姿
- `has_trailer` 的定义是：`trailer_mass_kg > 1.0` 时为 1，否则为 0

### 6.2 输出残差

当前 MLP 输出是 9 维：

```text
[
  vx_t, vy_t, r_t,
  vx_s, vy_s, r_s,
  rel_x_s_t, rel_y_s_t, rel_yaw_s_t
]
```

可以理解为：

- 前 6 维是牵引车 / 挂车的速度和横摆角速度残差
- 后 3 维是挂车相对牵引车位姿的残差

## 7. 为什么不把绝对位置喂给 MLP

当前逻辑是：

1. 先用名义模型算出 `base_next`
2. 再让 MLP 预测 motion residual 和 relative-pose residual
3. 用牵引车速度残差结合 `dt` 反推出牵引车位姿修正
4. 用“修正后的牵引车位姿 + 修正后的相对位姿”恢复挂车绝对位姿

所以这里要区分两件事：

- 开环递推仍然需要初始绝对状态作为世界坐标系锚点
- 但 MLP 自身不需要把绝对坐标作为输入特征

这件事在当前代码里已经实现，而且从建模逻辑上也是合理的。

## 8. 监督目标是怎么构造出来的

监督目标在 `concat_segments_for_training()` 里构造。

### 8.1 名义下一步

对每个样本，先算：

```text
base_next = base_model(state, control, trailer_mass, dt)
```

### 8.2 MLP 的直接监督目标

当前 MLP 的监督目标是：

```text
true_mlp_output =
[
  next_states[:, 3:6]  - base_next[:, 3:6],     # 牵引车 motion residual
  next_states[:, 9:12] - base_next[:, 9:12],    # 挂车 motion residual
  relative(next_state) - relative(base_next)    # 挂车相对位姿 residual
]
```

在无挂车模式下，会做额外处理：

- 挂车 motion residual 置零
- 挂车 relative-pose residual 置零

这样就不会让无挂车数据把“有挂车那部分输出”学偏。

### 8.3 完整状态误差

同时还会计算：

```text
true_error = next_state - base_next
```

其中 `psi_t` 和 `psi_s` 会做角度 wrap。

这个 12 维完整状态误差不会直接作为 MLP 输出目标，但会参与 pose/full-state 的辅助损失。

另外需要注意：在牵引车状态语义切换为“后轴中心”之后，旧 checkpoint 即使输入输出维度不变，也不再兼容当前 base model 语义，需要重新训练。

## 9. 特征标准化和 loss scale

训练前会构造两套上下文。

### 9.1 feature context

`build_feature_context()` 会从训练集输入特征里计算：

- `feature_mean`
- `feature_scale = max(std, 1.0)`

输入标准化公式是：

```text
(features - feature_mean) / feature_scale
```

这组统计量会被保存进 checkpoint，推理时继续复用。

### 9.2 loss context

`build_loss_context()` 会构造：

- `error_scale`：完整 12 状态误差的 scale
- `pose_error_scale`：位姿误差的 scale
- `motion_error_scale`：速度 / 横摆角速度误差的 scale
- `output_scale`：MLP 9 维输出的 scale
- `channel_weight`：来自 `STATE_LOSS_WEIGHTS`
- `output_weight = [1, 1, 5, 1, 1, 5, 1, 1, 2]`

角度相关通道都有最小 scale 下限，避免某些通道标准差过小导致训练不稳定。

## 10. 当前 MLP 结构

当前网络定义在 `model_structure.py` 中，配置为：

```text
input_dim = 14
hidden_dim = 128
hidden_layers = 3
output_dim = 9
activation = Tanh
normalization = LayerNorm
dropout = 0.08
```

展开后就是：

```text
Linear -> LayerNorm -> Tanh -> Dropout
Linear -> LayerNorm -> Tanh -> Dropout
Linear -> LayerNorm -> Tanh -> Dropout
Linear(out)
```

当前参数量：

```text
36,873
```

最后输出层采用零初始化，因此训练开始时网络等价于“先完全不修正名义模型”。

## 11. 当前 loss 的组成

`compute_loss_components()` 现在由三部分构成。

### 11.1 output loss

第一部分直接监督 9 维 MLP 输出：

```text
output_loss = 对 (predicted_mlp_output - true_mlp_output) 做归一化后的加权 MSE
```

如果当前样本 `trailer_mass_kg <= 1.0`，挂车相关输出会被 mask 掉。

### 11.2 pose loss

先通过 `derive_full_error_from_mlp_output_torch()` 把 MLP 输出还原成完整状态修正，再对以下 6 个位姿通道计算 pose loss：

```text
[x_t, y_t, psi_t, x_s, y_s, psi_s]
```

### 11.3 full-state auxiliary loss

另外还有一个完整状态辅助损失，作用在：

```text
[x_t, y_t, psi_t, vx_t, vy_t, r_t, x_s, y_s, psi_s, vx_s, vy_s, r_s]
```

### 11.4 total loss

当前总损失是：

```text
total_loss = output_loss + pose_loss_weight * pose_loss + 0.05 * full_state_loss + vx_vy_r_smoothness_weight * vx_vy_r_smoothness_loss
```

### 11.5 Vx/Vy/R 局部平滑正则

为了解决“控制输入相同、而 `Vx / Vy / R` 只发生很小变化时，模型给出的下一状态差别却过大”的问题，训练里新增了一项局部平滑正则：

- 只扰动牵引车输入特征 `vx_t`、`vy_t`、`r_t`
- 控制输入保持不变
- 扰动幅值默认是：
  `vx_t = 0.05 m/s`
  `vy_t = 0.02 m/s`
  `r_t = 0.2 deg/s`
- 对扰动前后预测得到的“修正后完整下一状态误差”做加权一致性约束

这个正则本质上是在限制残差网络对局部 `Vx / Vy / R` 变化的敏感度，减少不必要的下一状态跳变。对应训练参数：

```text
--vx-vy-r-smoothness-weight
```

默认值来自 `constants.py`：

```text
VXYR_SMOOTHNESS_WEIGHT = 1.0e-2
```

### 11.6 pose 权重调度

`pose_loss_weight` 现在按训练 step 控制：

- 前 `5000` 个 optimizer step：`0.0`
- 从第 `5001` 步开始：`1.0`

也就是说，当前配置不再按 epoch warmup/ramp，而是按真实训练步数切换 pose loss。

## 12. 训练集 / 验证集划分逻辑

当前划分是按 segment 做的：

- 如果有多个 segment，就按 segment 随机划分 train / val
- 如果只有一个 segment，就按时间切开，前半段训练、后半段验证

对最近这次 `train_segment` 训练来说，统计如下：

- 总 segment 数：`23`
- 训练 segment 数：`18`
- 验证 segment 数：`5`
- 训练样本数：`33,632`
- 验证样本数：`13,516`

对应的表格输出在：

- `data/train_segment_training_summary_100ep/truck_trailer_dataset_split_segments.csv`
- `data/train_segment_training_summary_100ep/truck_trailer_validation_segments.csv`

## 13. 当前默认超参数

这些配置来自 `constants.py`：

```text
TRAIN_BATCH_SIZE = 128
TRAIN_NUM_WORKERS = 0
TRAIN_EPOCHS = 4000
LEARNING_RATE = 3.0e-3
POSE_LOSS_WARMUP_STEPS = 5000
GRADIENT_CLIP_NORM = 200.0
MLP_USE_LAYER_NORM = True
MLP_HIDDEN_DIM = 128
MLP_HIDDEN_LAYERS = 3
MLP_DROPOUT_P = 0.08
NO_TRAILER_MASS_THRESHOLD_KG = 1.0
FIXED_DT_S = 0.02
VXYR_SMOOTHNESS_WEIGHT = 1.0e-2
```

优化器是：

```text
Adam(weight_decay = 1.0e-5)
```

设备选择逻辑：

- 有 CUDA 就用 CUDA
- 否则用 CPU

## 14. checkpoint 里保存了什么

当前 checkpoint 会保存：

- 模型权重
- 输入 / 输出维度
- 隐藏层结构配置
- dropout 和 LayerNorm 配置
- 输入特征名
- 控制特征名
- 输出名
- 输入标准化统计量
- loss scale 统计量
- base model 参数

主要文件有两个：

- `best_truck_trailer_error_model.pth`
- `best_truck_trailer_error_model_train_loss.pth`

checkpoint 里还会额外保存 `vx_vy_r_smoothness` 配置，包含：

- 正则权重
- 被约束的输入特征名
- 被约束的状态通道名
- `vx / vy / r` 扰动幅值
- 对应的特征索引和状态索引

`inference_main.py` 在加载时会检查 checkpoint 的特征布局是否和当前代码一致。不兼容的老 checkpoint 会被明确拒绝。

## 15. 训练后的验证方式：teacher forcing

训练结束后，`train_main.py` 会对验证集做 teacher-forcing rollout：

- 当前步输入使用真实状态
- 名义模型先算 `base_next`
- MLP 在 `base_next` 基础上做残差修正
- 修正后的下一步状态和真实下一步状态做比较

会导出的结果包括：

- 轨迹图
- 关键状态时间序列图
- rollout RMSE 汇总

teacher forcing 更接近“一步预测质量”的检查，但难度比真正的开环递推低。

## 16. 开环验证逻辑

开环验证入口是：

```bash
python controltest/truck_trailer_residual_modular/inference_main.py --input-path "<segment 或根目录>"
```

和 teacher forcing 的差别在于：

- 开环会把“上一步的预测状态”继续喂给下一步
- 误差会累积
- 更能暴露位置漂移和航向漂移问题

推理阶段会复用 checkpoint 中保存的：

- 输入标准化统计量
- `3 * loss_output_scale` 的输出裁剪阈值
- 完整状态恢复逻辑 `derive_full_error_from_mlp_output_np()`

## 17. 最近一次 100 epoch 训练的实际配置

最近一次用 `train_segment` 数据跑的 100 epoch 训练命令是：

```bash
python controltest/truck_trailer_residual_modular/train_main.py ^
  --input-path "controltest/truck_trailer_residual_modular/data/data" ^
  --epochs 100 ^
  --summary-dir "controltest/truck_trailer_residual_modular/data/train_segment_training_summary_100ep"
```

这次实验的几个关键事实是：

- 当前数据按无挂车模式处理
- steering 实际使用的是 `Steer_deg_cmd`
- 后轮驱动输入使用 `Torque_L2_Nm_cmd + Torque_R2_Nm_cmd`

需要注意：这份 `100 epoch` 历史结果生成于本次训练配置调整之前。当前代码已经改为：

- `TRAIN_BATCH_SIZE = 128`
- `MLP_HIDDEN_DIM = 128`
- 前 `5000` 个 optimizer step 不启用 pose loss

训练产物主要在：

- `data/train_segment_training_summary_100ep/truck_trailer_training_loss_log.png`
- `data/train_segment_training_summary_100ep/truck_trailer_dataset_split_segments.csv`
- `data/train_segment_training_summary_100ep/truck_trailer_validation_segments.csv`
- `best_truck_trailer_error_model.pth`

## 18. 最近一次四类工况开环验证结果

最近做过四类场景的开环检查：

- accelerate
- decelerate
- turn
- lane_change

汇总文件在：

```text
controltest/truck_trailer_residual_modular/data/train_segment_training_summary_100ep/open_loop_validation_summary_100ep.csv
```

对应的牵引车 RMSE 摘要如下：

| 场景 | x_t base -> corr | y_t base -> corr | psi_t base -> corr | vx_t base -> corr | vy_t base -> corr | r_t base -> corr |
| --- | --- | --- | --- | --- | --- | --- |
| accelerate | 279.774 -> 11.165 | 17.899 -> 50.935 | 158.103 -> 17.536 | 2.573 -> 0.902 | 0.880 -> 0.109 | 10.330 -> 0.648 |
| decelerate | 31.338 -> 9.594 | 63.476 -> 23.854 | 116.995 -> 15.500 | 3.253 -> 2.505 | 0.491 -> 0.037 | 7.902 -> 0.294 |
| turn | 52.763 -> 24.528 | 40.027 -> 43.431 | 90.117 -> 39.574 | 3.963 -> 2.920 | 1.240 -> 0.340 | 15.399 -> 6.471 |
| lane_change | 71.179 -> 18.812 | 50.260 -> 54.134 | 76.207 -> 20.461 | 1.673 -> 1.191 | 0.568 -> 0.087 | 7.019 -> 0.620 |

当前对这些结果的判断可以概括成：

- `x_t / psi_t / vy_t / r_t` 改善比较明显
- `vx_t` 也有改善，但在部分工况下幅度有限
- `y_t` 在 accelerate / turn / lane_change 仍然有漂移
- 这和“100 epoch 仍在 pose warmup 前期”是吻合的

## 19. 现在这套训练逻辑的关系图

核心依赖关系可以概括成：

```text
CSV 列
  -> SegmentData
  -> 名义下一步 base_next
  -> MLP 输入特征
  -> MLP 监督目标
  -> 标准化训练
  -> MLP 输出
  -> 完整状态修正重建
  -> 修正后的下一步状态
  -> teacher forcing / open-loop 验证
```

更具体地说：

```text
当前状态 + 控制量 + 挂车质量
  -> base_model
  -> base_next

当前状态中的速度 / 相对位姿 + steer + 后轮扭矩和
  -> MLP
  -> motion residual + relative-pose residual

motion residual + dt
  -> 牵引车位姿修正

修正后的牵引车位姿 + relative-pose residual
  -> 挂车绝对位姿修正
```

这就是当前实现里最核心的逻辑关系。

## 20. 实际使用时的几个提醒

1. 如果我们关心的是开环位置效果，100 epoch 还比较早，因为 pose ramp 还没启动。
2. 如果后续数据里真的提供了 `Steer_SW_deg`，解析器会优先使用它。
3. 如果后续数据补齐了挂车状态和挂车质量，当前训练入口不需要改，就可以直接训练“有挂车模式”。
4. 如果输入 / 输出布局再次变化，老 checkpoint 会和新代码不兼容。

## 21. 相关文档

- 更偏“怎么复现”的命令整理见：`EXPERIMENT_REPRODUCTION.md`
