# `train_truck_trailer_residual.py` 中 Base Model 说明

本文档整理 [train_truck_trailer_residual.py](d:/test_torch%20project/controltest/train_truck_trailer_residual.py) 里 `TruckTrailerNominalDynamics` 的建模思路、物理量含义、主要公式，以及当前模型中不够合理的地方和改进建议。

## 1. 这个 Base Model 是干什么的

这个 `base model` 不是最终精确的整车仿真器，而是给残差网络打底的名义模型。

整体思路是：

1. 先用一个可解释的车挂二维动力学模型，计算一步预测 `base_next`。
2. 再让神经网络去学习 `真实系统 - 名义模型` 的残差。
3. 这样网络不用从零学习整套动力学，只需要修正名义模型没有覆盖好的部分。

所以它的目标更偏向“稳定、可微、可训练、结构正确”，而不是“一开始就把所有物理现象都刻画到位”。

## 2. 状态、输入和坐标定义

### 2.1 状态量

脚本中状态定义为 12 维：

```text
[x_t, y_t, psi_t, vx_t, vy_t, r_t, x_s, y_s, psi_s, vx_s, vy_s, r_s]
```

含义如下：

| 符号 | 含义 | 单位 |
| --- | --- | --- |
| `x_t, y_t` | 牵引车质心在全局坐标系中的位置 | m |
| `psi_t` | 牵引车航向角 | rad |
| `vx_t, vy_t` | 牵引车本体坐标系下的纵向/横向速度 | m/s |
| `r_t` | 牵引车横摆角速度 | rad/s |
| `x_s, y_s` | 挂车质心在全局坐标系中的位置 | m |
| `psi_s` | 挂车航向角 | rad |
| `vx_s, vy_s` | 挂车本体坐标系下的纵向/横向速度 | m/s |
| `r_s` | 挂车横摆角速度 | rad/s |

这里：

- 下标 `t` 表示 tractor，牵引车。
- 下标 `s` 表示 semi-trailer，半挂车。

### 2.2 控制输入

控制量定义为 5 维：

```text
[delta_f_rad, torque_fl, torque_fr, torque_rl, torque_rr]
```

含义如下：

| 符号 | 含义 | 单位 |
| --- | --- | --- |
| `delta_f_rad` | 前轮转角 | rad |
| `torque_fl` | 左前轮扭矩 | Nm |
| `torque_fr` | 右前轮扭矩 | Nm |
| `torque_rl` | 左后轮扭矩 | Nm |
| `torque_rr` | 右后轮扭矩 | Nm |

额外还有一个外部输入：

| 符号 | 含义 | 单位 |
| --- | --- | --- |
| `trailer_mass_kg` | 挂车质量 | kg |

这个量既进入 `base model`，也进入残差网络特征。

### 2.3 坐标系和几何关系

模型使用两个车体坐标系和一个全局坐标系：

- 全局坐标系：描述轨迹位置 `x, y`
- 牵引车本体坐标系：描述 `vx_t, vy_t`
- 挂车本体坐标系：描述 `vx_s, vy_s`

铰接角不是显式状态，而是隐含为：

```math
\phi = \psi_s - \psi_t
```

训练特征里使用了：

```math
\phi,\ \sin\phi,\ \cos\phi
```

这样做是为了让神经网络更容易学到车挂相对姿态关系，同时避免角度跨 `\pm \pi` 时的不连续。

## 3. 物理参数及其意义

脚本默认参数在 [train_truck_trailer_residual.py](d:/test_torch%20project/controltest/train_truck_trailer_residual.py#L63)。

| 参数 | 含义 | 默认值 |
| --- | --- | --- |
| `m_t` | 牵引车质量 | `9300 kg` |
| `Iz_t` | 牵引车绕 z 轴转动惯量 | `48639 kg·m²` |
| `L_t` | 牵引车轴距 | `4.475 m` |
| `a_t` | 牵引车质心到前轴距离 | `3.8 m` |
| `b_t = L_t - a_t` | 牵引车质心到后轴距离 | `0.675 m` |
| `m_s_base` | 挂车基准质量 | `15004 kg` |
| `Iz_s_base` | 挂车基准横摆惯量 | `96659 kg·m²` |
| `L_s` | 挂车铰接点到挂车轴的距离 | `8.0 m` |
| `c_s` | 挂车质心到铰接点的距离 | `4.0 m` |
| `Cf` | 牵引车前轴侧偏刚度 | `80000 N/rad` |
| `Cr` | 牵引车后轴侧偏刚度 | `80000 N/rad` |
| `Cs` | 挂车轴侧偏刚度 | `80000 N/rad` |
| `wheel_radius` | 车轮半径 | `0.5 m` |
| `track_width` | 轮距 | `1.8 m` |
| `steering_ratio` | 方向盘角换算到前轮转角的传动比，仅用于原始数据预处理 | `16.39` |
| `rho` | 空气密度 | `1.225 kg/m³` |
| `CdA_t` | 牵引车阻力系数与迎风面积乘积 | `5.82` |
| `CdA_s` | 挂车阻力系数与迎风面积乘积 | `6.50` |
| `rolling_coeff` | 滚阻系数 | `0.006` |
| `hitch_x, hitch_y` | 铰接点在牵引车本体坐标系中的位置 | `(-0.331, 0.002) m` |
| `min_speed_mps` | 低速保护阈值 | `0.5 m/s` |

## 4. 模型是怎么建立的

## 4.1 第一步：把车挂系统抽象成两个平面刚体

这个模型本质上是一个二维双刚体模型：

- 刚体 1：牵引车
- 刚体 2：挂车

每个刚体都只有平面运动自由度：

- 平移：`x, y`
- 转动：`psi`

以及对应的速度：

- 本体纵向速度 `vx`
- 本体横向速度 `vy`
- 横摆角速度 `r`

这意味着它忽略了：

- 侧倾
- 俯仰
- 悬架变形
- 车轮转速
- 垂向载荷转移

这就是一个“平面车体 + 简化轮胎 + 简化铰接”的名义模型。

## 4.2 第二步：统一成前轮转角输入

如果原始数据给的是方向盘角，预处理里先做：

```math
\delta_f = \frac{\delta_{sw}}{i_s}
```

其中：

- `\delta_f` 是最终送进 base model 的前轮转角输入
- `\delta_{sw}` 是原始方向盘转角
- `i_s` 是转向传动比 `steering_ratio`

对应代码在 [train_truck_trailer_residual.py](d:/test_torch%20project/controltest/train_truck_trailer_residual.py#L236)。

## 4.3 第三步：计算轮胎侧偏角

### 牵引车前轴

```math
\alpha_f = \delta_f - \arctan\frac{v_{y,t} + a_t r_t}{v_{x,t}}
```

### 牵引车后轴

```math
\alpha_r = -\arctan\frac{v_{y,t} - b_t r_t}{v_{x,t}}
```

### 挂车轴

```math
\alpha_s = -\arctan\frac{v_{y,s} - L_s r_s}{v_{x,s}}
```

对应代码在 [train_truck_trailer_residual.py](d:/test_torch%20project/controltest/train_truck_trailer_residual.py#L242)。

为了避免极低速或倒车时分母趋近于 0，代码没有直接用 `vx`，而是先做了一个有符号下限保护：

```math
v_x^{safe} = \operatorname{sign}(v_x)\max(|v_x|, v_{min})
```

对应代码在 [train_truck_trailer_residual.py](d:/test_torch%20project/controltest/train_truck_trailer_residual.py#L206)。

## 4.4 第四步：用线性轮胎模型计算侧向力

模型使用最简单的一阶线性关系：

```math
F_{yf} = C_f \alpha_f
```

```math
F_{yr} = C_r \alpha_r
```

```math
F_{ys} = C_s \alpha_s
```

对应代码在 [train_truck_trailer_residual.py](d:/test_torch%20project/controltest/train_truck_trailer_residual.py#L246)。

这一步是假设轮胎侧向力和侧偏角线性成正比。

## 4.5 第五步：建立铰接点的几何与速度约束

牵引车铰接点在全局系中的位置：

```math
x_{h,t} = x_t + h_x\cos\psi_t - h_y\sin\psi_t
```

```math
y_{h,t} = y_t + h_x\sin\psi_t + h_y\cos\psi_t
```

挂车铰接点在全局系中的位置：

```math
x_{h,s} = x_s + c_s\cos\psi_s
```

```math
y_{h,s} = y_s + c_s\sin\psi_s
```

对应代码在 [train_truck_trailer_residual.py](d:/test_torch%20project/controltest/train_truck_trailer_residual.py#L255) 和 [train_truck_trailer_residual.py](d:/test_torch%20project/controltest/train_truck_trailer_residual.py#L263)。

牵引车铰接点在本体系中的速度：

```math
v_{h,t}^x = v_{x,t} - r_t h_y
```

```math
v_{h,t}^y = v_{y,t} + r_t h_x
```

挂车铰接点在本体系中的速度：

```math
v_{h,s}^x = v_{x,s}
```

```math
v_{h,s}^y = v_{y,s} - r_s c_s
```

再旋转到全局系比较位置误差和速度误差：

```math
e_p = p_{h,t} - p_{h,s}
```

```math
e_v = v_{h,t} - v_{h,s}
```

## 4.6 第六步：用“虚拟弹簧-阻尼器”近似铰接约束力

代码中不是用严格约束方程，而是用罚函数形式：

```math
F_{h,x} = -K_p e_{p,x} - K_v e_{v,x}
```

```math
F_{h,y} = -K_p e_{p,y} - K_v e_{v,y}
```

其中：

- `K_p = 1e6`
- `K_v = 1e4`

对应代码在 [train_truck_trailer_residual.py](d:/test_torch%20project/controltest/train_truck_trailer_residual.py#L275)。

这相当于说：

- 铰接点位置不一致时，给一个很强的恢复力
- 铰接点速度不一致时，给一个阻尼力

然后再把这个全局系铰接力分别投影回牵引车和挂车本体系。

## 4.7 第七步：把轮扭矩换成纵向轮力

对每个轮：

```math
F_x = \frac{T}{R}
```

即：

```math
F_{x,fl} = \frac{T_{fl}}{R},\quad
F_{x,fr} = \frac{T_{fr}}{R},\quad
F_{x,rl} = \frac{T_{rl}}{R},\quad
F_{x,rr} = \frac{T_{rr}}{R}
```

对应代码在 [train_truck_trailer_residual.py](d:/test_torch%20project/controltest/train_truck_trailer_residual.py#L285)。

前轮因为有转角 `\delta_f`，需要把前轴纵向力投影回车体系：

```math
F_{x,f}^{body} = (F_{x,fl}+F_{x,fr})\cos\delta_f
```

```math
F_{y,f}^{drive} = (F_{x,fl}+F_{x,fr})\sin\delta_f
```

对应代码在 [train_truck_trailer_residual.py](d:/test_torch%20project/controltest/train_truck_trailer_residual.py#L290)。

## 4.8 第八步：加入气阻和滚阻

### 气阻

牵引车：

```math
F_{drag,t} = -\frac{1}{2}\rho C_dA_t \|v_t\| v_{x,t}
```

挂车：

```math
F_{drag,s} = -\frac{1}{2}\rho C_dA_s \|v_s\| v_{x,s}
```

### 滚阻

牵引车：

```math
F_{roll,t} = C_{rr} m_t g \tanh(10 v_{x,t})
```

挂车：

```math
F_{roll,s} = C_{rr} m_s g \tanh(10 v_{x,s})
```

对应代码在 [train_truck_trailer_residual.py](d:/test_torch%20project/controltest/train_truck_trailer_residual.py#L297)。

这里 `tanh` 的作用是让阻力在低速和倒车附近仍然连续可导。

## 4.9 第九步：写牵引车动力学方程

### 牵引车纵向

```math
\dot v_{x,t} =
\frac{F_{x,total,t}}{m_t} + r_t v_{y,t}
```

其中：

```math
F_{x,total,t}
= F_{x,f}^{body}
+ (F_{x,rl}+F_{x,rr})
+ F_{yf}\sin\delta_f
+ F_{h,t}^x
+ F_{drag,t}
- F_{roll,t}
```

### 牵引车横向

```math
\dot v_{y,t} =
\frac{F_{y,total,t}}{m_t} - r_t v_{x,t}
```

其中：

```math
F_{y,total,t}
= F_{yf}\cos\delta_f
+ F_{yr}
+ F_{h,t}^y
+ F_{y,f}^{drive}
```

### 牵引车横摆

```math
\dot\psi_t = r_t
```

```math
\dot r_t =
\frac{
a_t(F_{yf}\cos\delta_f + F_{y,f}^{drive})
- b_t F_{yr}
+ (h_x F_{h,t}^y - h_y F_{h,t}^x)
+ (F_{x,fr}-F_{x,fl})\frac{w}{2}
+ (F_{x,rr}-F_{x,rl})\frac{w}{2}
}{I_{z,t}}
```

对应代码在 [train_truck_trailer_residual.py](d:/test_torch%20project/controltest/train_truck_trailer_residual.py#L304)。

这里最后两项表示左右轮纵向力差带来的附加横摆力矩。

## 4.10 第十步：写挂车动力学方程

### 挂车纵向

```math
\dot v_{x,s} =
\frac{F_{h,s}^x + F_{drag,s} - F_{roll,s}}{m_s} + r_s v_{y,s}
```

### 挂车横向

```math
\dot v_{y,s} =
\frac{F_{ys} + F_{h,s}^y}{m_s} - r_s v_{x,s}
```

### 挂车横摆

```math
\dot\psi_s = r_s
```

```math
\dot r_s = \frac{-L_s F_{ys} + c_s F_{h,s}^y}{I_{z,s}}
```

对应代码在 [train_truck_trailer_residual.py](d:/test_torch%20project/controltest/train_truck_trailer_residual.py#L318)。

## 4.11 第十一步：写位置运动学方程

牵引车：

```math
\dot x_t = v_{x,t}\cos\psi_t - v_{y,t}\sin\psi_t
```

```math
\dot y_t = v_{x,t}\sin\psi_t + v_{y,t}\cos\psi_t
```

挂车：

```math
\dot x_s = v_{x,s}\cos\psi_s - v_{y,s}\sin\psi_s
```

```math
\dot y_s = v_{x,s}\sin\psi_s + v_{y,s}\cos\psi_s
```

对应代码在 [train_truck_trailer_residual.py](d:/test_torch%20project/controltest/train_truck_trailer_residual.py#L323)。

## 4.12 第十二步：用 RK4 做单步积分

`derivatives()` 给出的是连续时间导数：

```math
\dot x = f(x,u)
```

`forward()` 用四阶 Runge-Kutta：

```math
k_1 = f(x_k, u_k)
```

```math
k_2 = f(x_k + \frac{1}{2}\Delta t k_1, u_k)
```

```math
k_3 = f(x_k + \frac{1}{2}\Delta t k_2, u_k)
```

```math
k_4 = f(x_k + \Delta t k_3, u_k)
```

```math
x_{k+1} = x_k + \frac{\Delta t}{6}(k_1 + 2k_2 + 2k_3 + k_4)
```

对应代码在 [train_truck_trailer_residual.py](d:/test_torch%20project/controltest/train_truck_trailer_residual.py#L333)。

最后还会把 `psi_t` 和 `psi_s` wrap 到 `[-\pi, \pi]`。

## 5. 这个模型为什么这样建立

可以把它理解成“白盒主干 + 黑盒修正”。

### 5.1 为什么要先写一个名义车挂模型

因为如果直接让神经网络从控制量学 12 维下一状态，会有几个问题：

- 需要更多数据
- 容易学到不满足基本几何关系的假动力学
- open-loop 时误差容易快速发散

而名义模型已经把这些结构先写进去了：

- 牵引车和挂车是两个刚体
- 铰接点必须连续
- 车轮扭矩会变成纵向力
- 航向角和速度之间要满足基本运动学关系

这样残差网络只需要学：

- 参数不准导致的偏差
- 轮胎非线性
- 真实系统更复杂的载荷变化
- 未建模的执行器/轮胎/空气动力学效应

### 5.2 为什么把挂车质量单独作为输入

因为挂车质量变化对系统影响很大：

- 纵向加速度会变化
- 横摆响应会变化
- 铰接动态会变化
- 车挂夹角的演化速度会变化

所以脚本里做了两件事：

- 在 `base model` 中直接把 `m_s` 作为动力学参数
- 在 MLP 中也把 `trailer_mass_kg` 当作特征

这样白盒和黑盒都能看到挂车质量的影响。

## 6. 目前模型里不太合理的地方

下面这些点不是说模型完全不能用，而是说明它目前更偏“训练可用的名义模型”，还不是特别严谨的工程级车挂动力学模型。

### 6.1 挂车转动惯量按质量线性缩放，过于粗糙

代码中：

```math
I_{z,s} = I_{z,s}^{base}\frac{m_s}{m_{s,base}}
```

这等价于假设：

- 挂车几何不变
- 质量分布形状不变
- 载荷增加时转动惯量严格与质量成比例

这个近似在“同一挂车、装载位置变化不大”时还能接受，但一旦：

- 货物装载位置前后变化明显
- 挂车结构换型
- 重心高度/重心纵向位置变化

这个关系就会偏差很大。

建议：

- 最好把 `Iz_s` 也作为显式输入或显式标定量。
- 如果没有直接测量，可以用 `Iz_s = k m_s` 之外再加一个可调系数或经验模型。
- 更进一步，可以让 `c_s`、`L_s`、`Cs` 也随载荷变化。

### 6.2 轮胎模型过于线性，没有饱和，也没有纵横向耦合

当前模型用的是：

```math
F_y = C_\alpha \alpha
```

但真实轮胎在大侧偏角、大驱动/制动扭矩下会出现：

- 饱和
- 纵横向耦合
- 附着椭圆限制

当前模型没有把这些耦合写进去，所以一旦：

- 大转向
- 大驱动扭矩
- 低附着路面

名义模型就会明显偏软或偏硬。

建议：

- 至少给 `Fy` 加一个 `tanh` 或饱和函数。
- 更好的是引入简化的 friction circle / friction ellipse。
- 如果数据足够，再把法向载荷变化带进 `Cf/Cr/Cs`。

### 6.3 铰接约束用“大刚度弹簧 + 大阻尼”近似，数值上偏硬

当前模型不是严格的约束动力学，而是：

```math
F_h = -K_p e_p - K_v e_v
```

这会带来两个问题：

- `K_p` 和 `K_v` 太大时系统很硬，数值积分容易敏感
- 它本质上允许铰接点“有一点不重合”，只是靠大力把它拉回去

这对于训练是方便的，因为：

- 公式简单
- 容易写成可微形式

但从物理上看不够严格。

建议：

- 如果追求物理一致性，可以改成显式约束模型或拉格朗日乘子法。
- 如果仍保留罚函数形式，建议把 `K_p`、`K_v` 也做成可调参数，并根据采样周期重新标定。
- 还可以把铰接点误差日志打出来，检查它是否长期偏大。

### 6.4 只有挂车质量在变，但其它随载荷变化的参数没有变

现在脚本里虽然让 `m_s` 变化了，但以下量仍固定：

- `Cs`
- `CdA_s`
- `c_s`
- `L_s`

特别是 `Cs` 固定这一点，在不同挂车载荷下通常不够合理。因为载荷会影响轮胎法向载荷，进而影响横向刚度和极限侧向力。

建议：

- 至少尝试让 `Cs = f(m_s)`。
- 如果有挂车轴载数据，优先用轴载驱动 `Cs` 和附着极限。

### 6.5 前轮扭矩参与了前轴驱动侧向耦合，但真实车辆未必如此

代码里默认：

- 前轮也可能有扭矩
- 前轴纵向力会因为转角投影到横向，形成 `fy_front_from_drive`

这在四驱或前轴可施加驱动力时是合理的，但如果你的真实车是：

- 纯后驱
- 前轴只转向不驱动

那这部分就可能高估前轴贡献。

建议：

- 明确你的数据里前轴扭矩是否真实有效。
- 如果实际前轴不驱动，可以把前轴扭矩分支关掉，或者直接把前轮扭矩固定为 0。

### 6.6 轮距差动扭矩直接转成横摆力矩，仍然偏简化

当前有：

```math
(F_{x,r} - F_{x,l})\frac{w}{2}
```

这等价于“左右轮纵向力差立即形成横摆力矩”。

这个方向是对的，但实际还会受到：

- 轮胎滑移率
- 附着限制
- 左右轮法向载荷
- 差速器/制动分配

影响。

建议：

- 如果后续关注扭矩矢量控制效果，建议单独把这一块建得更细。
- 否则可以保留，但要把它视为“近似项”。

### 6.7 低速保护会改变极低速和倒车附近的真实动力学

`_signed_safe_velocity()` 的目的是数值稳定，但代价是：

- `vx` 很小时，侧偏角公式不再严格等于真实公式
- 倒车附近的响应会被平滑化

这对训练是有帮助的，但会让模型在极低速、掉头、倒车工况下出现系统偏差。

建议：

- 如果以后重点是倒车工况，建议单独写倒车稳定处理，而不是只靠速度钳位。
- 也可以在倒车区间使用专门的轮胎/几何模型。

### 6.8 缺少法向载荷转移、悬架和侧倾

目前模型是严格的二维平面模型，没有：

- 左右载荷转移
- 前后载荷转移
- 悬架柔度
- 侧倾/俯仰

这会影响：

- 大加速/大制动
- 急转向
- 高重心挂车

下的响应。

建议：

- 如果只是做残差打底，当前层级可以接受。
- 如果目标是更强 open-loop 可解释性，下一步优先补“载荷转移 + 轮胎饱和”，收益通常最大。

## 7. 对这个模型的总体评价

这个 `base model` 的优点是：

- 结构清楚
- 物理意义明确
- 已经把车挂耦合、质量变化、铰接约束都写进去了
- 用 RK4 积分，数值上比简单 Euler 更稳一些
- 很适合作为残差学习的底模

它的主要问题不是“方向错了”，而是“目前还是偏粗糙”：

- 轮胎太线性
- 铰接约束太硬
- 载荷变化只改了质量，没有系统地改其它参数
- 某些力矩项和驱动分配仍然比较经验化

所以比较准确的定位应该是：

这是一个适合做 `Base + Residual Learning` 的车挂名义模型，
而不是一个已经高度校准完成的高保真商用车辆动力学模型。

## 8. 建议的改进优先级

如果后面准备继续改这个模型，我建议按下面顺序推进：

1. 先改轮胎模型：给 `Fy` 加饱和和纵横向耦合。
2. 再改挂车载荷相关参数：至少让 `Iz_s` 和 `Cs` 随挂车质量变化。
3. 再改铰接约束：把固定 `K_p/K_v` 改成可调参数，必要时换成更严格的约束形式。
4. 最后再考虑载荷转移、悬架和更高阶动力学。

如果目标是“尽快把训练/推理跑通”，那当前模型已经可以作为很好的第一版；
如果目标是“减少 open-loop 漂移并提升外推能力”，那优先补轮胎饱和和挂车载荷耦合会最有效。
