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
- No-trailer data is supported. If trailer state columns are missing, the loader automatically switches to no-trailer mode and mirrors tractor channels into the trailer placeholder channels, matching the existing model assumptions.
