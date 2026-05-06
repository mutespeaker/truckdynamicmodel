# Truck Trailer Residual Modular Version

This directory is a modular refactor of:

- `controltest/train_truck_trailer_residual.py`
- `controltest/ditui_truck_trailer_residual.py`
- the standalone base-model simulation logic used by `controltest/truck_trailer_base_model.py`

The goal is to keep the same training and inference behavior while separating the main responsibilities into different files.

## File Structure

- `constants.py`
  Shared constants, hyper-parameters, checkpoint paths, and CSV column candidates.

- `base_model.py`
  The nominal truck-trailer dynamics model `TruckTrailerNominalDynamics` and angle wrapping helpers.

- `model_structure.py`
  The residual neural network `MLPErrorModel`.

- `data_utils.py`
  Data loading, CSV column resolution, feature construction, normalization, residual target building, and train/val split helpers.

- `training.py`
  Training loop, loss calculation, checkpoint export, training plots, and teacher-forcing rollout evaluation.

- `train_main.py`
  Training entry. This is the modular replacement for `train_truck_trailer_residual.py`.

- `inference_main.py`
  Open-loop inference entry. This is the modular replacement for `ditui_truck_trailer_residual.py`.

- `base_model_demo.py`
  Standalone nominal base-model simulation entry, useful for checking the physics model without the residual network.

- `TRAINING_PROCESS.md`
  当前训练流程说明，整理了数据流、特征/标签构造、loss 设计、checkpoint 内容，以及最近一次 `train_segment` 实验。

- `EXPERIMENT_REPRODUCTION.md`
  实验复现说明，单独列出训练命令、开环验证命令，以及结果文件路径。

- `DIFFERENTIABLE_DYNAMICS_PROXY_MODULE.md`
  可微控制的动力学代理模型模块说明文档，从功能定位、算法原理、组织架构到关键结果，对当前模型做整体说明。

## How To Run

From the project root:

```bash
python controltest/truck_trailer_residual_modular/train_main.py
python controltest/truck_trailer_residual_modular/inference_main.py
python controltest/truck_trailer_residual_modular/base_model_demo.py
```

To train or infer on one specific run directory, you can pass the run folder directly:

```bash
python controltest/truck_trailer_residual_modular/train_main.py --run-dir "D:\test_torch project\controltest\carsim_runs\python_run_batch1_0002_20260309_174921"
python controltest/truck_trailer_residual_modular/inference_main.py --run-dir "D:\test_torch project\controltest\carsim_runs\python_run_batch1_0002_20260309_174921"
```

To explicitly train on all runs under the original root directory, you can also pass the root itself:

```bash
python controltest/truck_trailer_residual_modular/train_main.py --run-dir "D:\test_torch project\controltest\carsim_runs"
```

Single-run training is supported even when there is only one segment. In that case, the script keeps the same model and loss logic, and only changes the data split strategy: it uses the earlier part of the time series for training and the later part for validation.

## Relative-Pose Residual MLP

The residual model supports both no-trailer and trailer cases. If trailer columns are missing, the loader falls back to no-trailer mode:

- `trailer_mass_kg = 0`
- trailer state channels as placeholders mirrored from the tractor state

When trailer columns and mass are available, the loader keeps the trailer states and uses `trailer_mass_kg` as the mode signal. In both cases:

- rear-wheel drive torque is represented by the summed rear-axle command `torque_rl + torque_rr`
- `dt = 0.02 s`, used by the base model and residual post-processing but not passed into the MLP

The nominal base model still keeps the original 12-state / 5-control interface so existing loaders, plots, and rollout code remain compatible. The MLP itself stays translation-invariant:

- MLP input features, 14 dimensions:
  `trailer_mass_kg`, `has_trailer`, `vx_t`, `vy_t`, `r_t`, `vx_s`, `vy_s`, `r_s`, `rel_x_s_t`, `rel_y_s_t`, `sin_rel_yaw_s_t`, `cos_rel_yaw_s_t`, `steer_sw_rad`, `rear_drive_torque_sum`

- MLP output residuals, 9 dimensions:
  `vx_t`, `vy_t`, `r_t`, `vx_s`, `vy_s`, `r_s`, `rel_x_s_t`, `rel_y_s_t`, `rel_yaw_s_t`

- Default hidden structure:
  3 hidden layers, 128 units each, `LayerNorm + Tanh + Dropout`.

Absolute position and `dt` are not sent into the MLP. Tractor `x/y/yaw` are rebuilt from velocity residuals, base yaw, and the fixed `0.02 s` step. For trailer cases, the model predicts trailer-to-tractor relative-pose residuals in the tractor body frame; `data_utils.py` then reconstructs trailer absolute `x/y/yaw` from corrected tractor pose plus corrected relative pose. For no-trailer cases, trailer placeholder channels are mirrored from the tractor correction.

Older checkpoints trained with the previous 18-input / 6-output truck-trailer MLP, the intermediate 6-input / 3-output no-trailer MLP, or the 5-input fixed-dt MLP are not compatible with this relative-pose layout. Run `train_main.py` again to produce new checkpoints before running `inference_main.py`.

For a fuller walkthrough of the current training path, see `TRAINING_PROCESS.md`.
For the exact reproduction commands and output paths, see `EXPERIMENT_REPRODUCTION.md`.

## Inputs And Outputs

- Data root:
  `controltest/carsim_runs`

- Residual-model checkpoints:
  saved inside this directory as:
  `best_truck_trailer_error_model.pth`
  `best_truck_trailer_error_model_train_loss.pth`

- Training summary outputs:
  `controltest/carsim_runs/truck_trailer_multirun_training_summary_modular`

- Open-loop inference summary:
  `controltest/carsim_runs/truck_trailer_open_loop_summary_modular.csv`

- Per-segment inference outputs:
  each run folder gets:
  `truck_trailer_open_loop_eval_modular`

## Mapping Back To The Original Scripts

- Original `TruckTrailerNominalDynamics`:
  moved into `base_model.py`

- Original `MLPErrorModel`:
  moved into `model_structure.py`

- Original data parsing / feature engineering:
  moved into `data_utils.py`

- Original training loop and rollout evaluation:
  moved into `training.py`

- Original `main()` from training:
  moved into `train_main.py`

- Original `main()` from inference:
  moved into `inference_main.py`

## Notes

- The modular version keeps the same model equations and residual-learning pipeline.
- The modular version uses its own checkpoint files in this folder, so it does not overwrite the original script checkpoints.
- If this folder does not contain checkpoints yet, `inference_main.py` also tries the legacy checkpoints under `controltest/`.
- If no compatible checkpoint exists yet, run `train_main.py` before `inference_main.py`.
- The per-run inference output folder name is suffixed with `_modular` to avoid mixing results with the original scripts.
- With `FORCE_NO_TRAILER_MODE = False`, trailer data is used when complete trailer columns are present; otherwise the loader mirrors tractor channels into the trailer placeholder channels, sets trailer mass to zero, and uses a fixed `0.02 s` step.
