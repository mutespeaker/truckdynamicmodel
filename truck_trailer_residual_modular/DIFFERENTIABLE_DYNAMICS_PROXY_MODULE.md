# 可微控制的动力学代理模型模块说明文档

本文档用于说明 `controltest/truck_trailer_residual_modular` 当前这套模型的功能定位、算法原理、实现结构，以及模块工作流中的关键输出结果。

该模块本质上是一个“名义动力学模型 + 残差神经网络”的动力学代理模块，用于在牵引车/挂车场景下近似真实车辆系统的一步状态演化规律，并为训练验证、开环递推评估以及后续可微控制优化预留统一接口。

当前版本中，牵引车 `x_t, y_t, vx_t, vy_t` 明确解释为牵引车后轴中心状态。因此，所有在该语义切换之前训练得到的 checkpoint 都需要重新训练后才能继续使用。

## 1. 功能概述

当前模块的核心目标是：在已有名义动力学模型的基础上，引入可学习的残差补偿项，提高车辆状态预测精度，并保持模块接口清晰、可训练、可验证、可递推。

从功能上看，它主要承担以下任务：

- 从轨迹 CSV 中读取车辆状态、控制量和挂车质量信息
- 用名义动力学模型计算下一时刻状态的基础预测值
- 用 MLP 学习名义模型无法覆盖的残差部分
- 将残差恢复为完整状态修正，生成更接近真实数据的下一步状态
- 支持 teacher-forcing 验证和 open-loop 开环递推验证
- 产出 checkpoint、loss 曲线、分段划分表、RMSE 指标和轨迹图

### 1.1 模块定位与核心职责

该模块在整体系统中可视作“动力学代理模型层”，其定位不是替代控制器，而是提供一个可训练、可微、可递推的状态转移近似器。

它的核心职责包括：

1. 将当前状态和控制量映射到下一时刻状态。
2. 对名义物理模型进行数据驱动残差修正。
3. 统一有挂车与无挂车两种数据模式。
4. 为训练、验证、推理和控制优化提供一致的状态更新接口。

之所以称为“可微控制的动力学代理模型”，是因为：

- 名义模型 `TruckTrailerNominalDynamics` 基于 PyTorch 实现
- 残差网络 `MLPErrorModel` 基于 PyTorch 实现
- 训练路径中存在完整的 torch 计算图
- 从输入状态 / 控制量到一步预测输出，理论上可保留梯度链路

需要注意的是：当前 `inference_main.py` 中的开环验证是以 numpy 推理为主，偏向工程评估；真正的“可微控制”使用场景应优先复用 torch 路径中的一步状态映射逻辑。

## 2. 算法总体原理与框架

当前算法不是单纯黑盒网络，而是“物理先验 + 神经残差”的混合框架。

整体原理可以概括为：

```text
当前状态 + 控制量 + 挂车质量 + dt
  -> 名义动力学模型
  -> base_next

当前状态中对动力学有意义的局部特征
  -> 残差 MLP
  -> motion residual + relative-pose residual

base_next + residual reconstruction
  -> corrected_next
```

也就是说：

- 先用物理模型给出一个基础预测
- 再由神经网络学习真实系统与物理模型之间的偏差
- 最后把这部分偏差恢复为完整状态修正

这样做的优点是：

- 比纯黑盒网络更稳定
- 比纯物理模型更有数据拟合能力
- 容易解释误差来源
- 更适合进一步嵌入控制优化流程

### 2.1 输入&输出

当前模块可以从两个层面理解输入输出。

#### 2.1.1 模块级输入输出

从状态转移模块视角看：

- 输入：
  - 当前状态 `state`
  - 当前控制量 `control`
  - 挂车质量 `trailer_mass_kg`
  - 时间步长 `dt`
- 输出：
  - 下一时刻状态 `next_state`

状态维度固定为 12：

```text
[x_t, y_t, psi_t, vx_t, vy_t, r_t, x_s, y_s, psi_s, vx_s, vy_s, r_s]
```

其中当前实现明确约定：

- 牵引车 `x_t, y_t, vx_t, vy_t` 对应牵引车后轴中心
- `base_model.py` 内部会把牵引车后轴中心状态转换为名义动力学使用的内部参考点，积分完成后再转回后轴中心

控制量维度固定为 5：

```text
[steer_sw_rad, torque_fl, torque_fr, torque_rl, torque_rr]
```

#### 2.1.2 MLP 输入输出

当前残差 MLP 的输入为 14 维：

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

当前残差 MLP 的输出为 9 维：

```text
[
  vx_t, vy_t, r_t,
  vx_s, vy_s, r_s,
  rel_x_s_t, rel_y_s_t, rel_yaw_s_t
]
```

几个关键设计点：

- MLP 不接收绝对 `x/y`
- 牵引车位姿由速度残差和 `dt` 恢复
- 挂车位姿由相对位姿残差恢复
- 后轮驱动输入采用 `torque_rl + torque_rr`
- 当前固定 `dt = 0.02 s`，不进入 MLP

#### 2.1.3 数据级输入输出

从训练流程视角看：

- 输入：一组 `*_train_segment_*.csv` 或 `control_and_trajectory.csv`
- 输出：
  - 训练好的 checkpoint
  - loss 曲线图
  - train/val 分段表
  - rollout 图
  - RMSE 指标汇总

### 2.2 pipeline 设计

当前 pipeline 可以分为 8 个阶段：

#### 阶段 1：数据收集

`train_main.py` 会调用 `collect_control_and_trajectory_csvs()` 自动收集：

- `*_train_segment_*.csv`
- 或 `python_run_*/outputs/control_and_trajectory.csv`

#### 阶段 2：信号解析与模式识别

`data_utils.py` 负责完成：

- 方向盘角解析
- 轮端扭矩解析
- 牵引车状态解析
- 挂车状态解析
- 挂车质量解析
- 无挂车 fallback

当前 `Steer_deg_cmd` 被按角度制列读取，并转换成弧度制 `steer_sw_rad` 后再进入模型。

#### 阶段 3：名义一步预测

`base_model.py` 中的 `TruckTrailerNominalDynamics` 根据当前状态、控制量、挂车质量和 `dt` 计算：

```text
base_next
```

这是物理先验部分。

#### 阶段 4：残差特征构造

对当前状态构造 MLP 输入特征，重点保留：

- 速度状态
- 横摆角速度
- 挂车相对位姿
- 转向输入
- 后轴驱动输入

绝对坐标不进入 MLP。

#### 阶段 5：监督目标构造

监督目标分两路：

1. `true_mlp_output`：直接监督 MLP 输出
2. `true_error`：完整 12 状态误差，用于 pose/full-state 辅助损失

#### 阶段 6：标准化与训练

训练前先计算：

- `feature_mean / feature_scale`
- `error_scale / output_scale / pose_error_scale`

然后进入 `training.py` 中的训练主循环，优化 MLP 参数。

#### 阶段 7：teacher-forcing 验证

训练结束后，会对验证集做逐步的一步预测验证：

- 每一步都使用真实当前状态作为输入
- 不把上一步预测状态继续回灌

所以它是“整段 one-step rollout”，不是严格意义的自由开环。

#### 阶段 8：open-loop 开环递推验证

`inference_main.py` 则使用真实初始状态作为起点，之后每一步把预测状态继续喂给下一步，实现真正的开环递推评估。

### 2.3 组织架构设计

当前模块按职责拆分为以下文件：

- `constants.py`
  - 统一管理状态定义、输入输出维度、超参数、列名候选和 checkpoint 路径

- `base_model.py`
  - 名义动力学模型
  - 角度 wrap 辅助函数

- `model_structure.py`
  - 残差神经网络 `MLPErrorModel`

- `data_utils.py`
  - CSV 解析
  - 特征工程
  - 监督目标构造
  - train/val split
  - 残差恢复逻辑

- `training.py`
  - loss 计算
  - 训练循环
  - teacher-forcing rollout
  - 图表和 RMSE 输出

- `train_main.py`
  - 训练入口

- `inference_main.py`
  - 开环验证入口

这种结构的优点是：

- 训练逻辑和数据逻辑分离
- 推理逻辑和训练逻辑分离
- 更容易局部修改特征、loss、checkpoint 校验和 rollout 逻辑

## 3. 算法具体子功能模块介绍

### 3.1 名义动力学子模块

名义动力学子模块由 `TruckTrailerNominalDynamics` 实现，用于给出一步状态预测的基础值 `base_next`。

这是整个系统的物理主干，主要作用是：

- 利用车辆参数和控制输入，给出基础动力学更新
- 降低残差网络的学习负担
- 为模型提供可解释的物理先验

### 3.2 残差学习子模块

残差子模块由 `MLPErrorModel` 实现，当前配置为：

- 14 维输入
- 3 层隐藏层
- 每层 64 单元
- `LayerNorm + Tanh + Dropout`
- 9 维输出

该子模块主要负责学习：

- 牵引车速度残差
- 挂车速度残差
- 挂车相对位姿残差

### 3.3 特征工程子模块

特征工程由 `data_utils.py` 负责，核心特点是：

- 使用 translation-invariant 特征
- 不把绝对 `x/y` 喂给 MLP
- 通过 `relative_pose` 表达挂车相对牵引车关系
- 用 `has_trailer` 显式编码数据模式

这样设计有利于：

- 减少无关坐标偏移对训练的影响
- 让模型更关注真实动力学差异

### 3.4 监督目标构造子模块

监督目标分为两类：

1. 直接输出目标 `true_mlp_output`
2. 完整状态误差 `true_error`

这样做的意义在于：

- 既能直接约束 MLP 学对自己的输出空间
- 又能从完整状态角度约束它的物理效果

### 3.5 损失函数子模块

当前损失由三部分构成：

- `output_loss`
- `pose_loss`
- `full_state_loss`

总损失形式为：

```text
total_loss = output_loss + pose_loss_weight * pose_loss + 0.05 * full_state_loss
```

其中：

- `output_loss` 直接监督 9 维残差输出
- `pose_loss` 约束位姿误差
- `full_state_loss` 约束完整状态的一致性

另外，`pose_loss_weight` 当前按训练 step 控制：

- 前 5000 个 optimizer step 不启用 pose loss
- 从第 5001 步开始启用 pose loss

### 3.6 模式兼容子模块

当前代码支持：

- 有挂车模式
- 无挂车模式

无挂车时：

- 挂车状态镜像牵引车占位
- `trailer_mass_kg = 0`
- 挂车专属输出在 loss 中会被 mask

这样可以保证同一套训练框架兼容两类数据。

### 3.7 验证与评估子模块

当前提供两种验证方式：

1. `Validation Rollout`
   - teacher forcing
   - 逐步 one-step 评估

2. `Open-loop Rollout`
   - 自由递推
   - 更能暴露累计漂移问题

评估结果会输出：

- `tractor_x_rmse_m`
- `trailer_x_rmse_m`
- `articulation_rmse_deg`
- 各状态误差图
- 轨迹对比图

### 3.8 checkpoint 管理子模块

checkpoint 中保存了：

- 模型参数
- 输入/输出维度
- MLP 结构超参数
- 特征标准化统计量
- loss scale
- base model 参数

`inference_main.py` 会严格检查 checkpoint 的输入输出布局，防止旧模型误加载到新结构上。

## 4. 模块工作流的关键结果

当前模块工作流的关键结果可以从四个层面理解。

### 4.1 模型产物

训练完成后，核心模型产物是：

- `best_truck_trailer_error_model.pth`
- `best_truck_trailer_error_model_train_loss.pth`

它们分别代表：

- 最优验证损失模型
- 最优训练损失模型

### 4.2 训练过程结果

训练过程会输出：

- `truck_trailer_training_loss_log.png`
- `truck_trailer_dataset_split_segments.csv`
- `truck_trailer_validation_segments.csv`

这些文件反映：

- loss 收敛情况
- 数据如何划分
- 哪些 segment 用于验证

### 4.3 验证与开环结果

验证和递推阶段会输出：

- `truck_trailer_trajectory_comparison.png`
- `truck_trailer_state_timeseries.png`
- `open_loop_trajectory.png`
- `open_loop_state_errors.png`
- `open_loop_results.csv`
- `truck_trailer_open_loop_summary_modular.csv`

这些结果用于回答三个关键问题：

1. 模型是否比名义模型更接近真实轨迹？
2. 模型的一步预测是否稳定？
3. 模型在开环递推下是否会累积明显漂移？

### 4.4 当前阶段的效果结论

根据现有 100 epoch 的 `train_segment` 实验结果，当前模块已经表现出以下特征：

- `x_t / psi_t / vy_t / r_t` 的开环误差改善明显
- `vx_t` 有改善，但在部分场景下幅度有限
- `y_t` 在 accelerate / turn / lane_change 工况下仍有漂移
- 100 epoch 仍处于 pose warmup 早期，因此位置层面改进尚不充分

从模块工作流角度看，这意味着：

- 当前结构已经具备有效残差补偿能力
- 但对横向位置长期递推稳定性的学习仍需继续强化
- 后续可以优先通过更长训练周期、pose loss 生效阶段、或者横向误差加权策略来继续优化

---

如需查看更细的训练逻辑、复现实验命令和结果路径，可继续参考：

- `TRAINING_PROCESS.md`
- `EXPERIMENT_REPRODUCTION.md`
