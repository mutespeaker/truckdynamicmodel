from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


MODULE_DIR = Path(__file__).resolve().parent
CONTROLTEST_DIR = MODULE_DIR.parent

RUNS_ROOT = CONTROLTEST_DIR / "carsim_runs"
MODEL_CHECKPOINT = MODULE_DIR / "best_truck_trailer_error_model.pth"
TRAIN_LOSS_MODEL_CHECKPOINT = MODULE_DIR / "best_truck_trailer_error_model_train_loss.pth"
BASE_MODEL_OUTPUT_DIR = MODULE_DIR / "truck_trailer_base_model_outputs"
REAL_DATA_LABEL = "真实数据"

STATE_NAMES = [
    "x_t",
    "y_t",
    "psi_t",
    "vx_t",
    "vy_t",
    "r_t",
    "x_s",
    "y_s",
    "psi_s",
    "vx_s",
    "vy_s",
    "r_s",
]
# The MLP never receives absolute x/y coordinates. Tractor pose is advanced by
# velocity residuals; trailer pose is rebuilt from trailer-to-tractor relative
# pose, which keeps the rollout translation-invariant.
POSE_STATE_NAMES = ["x_t", "y_t", "psi_t", "x_s", "y_s", "psi_s"]
MOTION_STATE_NAMES = ["vx_t", "vy_t", "r_t", "vx_s", "vy_s", "r_s"]
MLP_OUTPUT_NAMES = [
    "vx_t",
    "vy_t",
    "r_t",
    "vx_s",
    "vy_s",
    "r_s",
    "rel_x_s_t",
    "rel_y_s_t",
    "rel_yaw_s_t",
]
CONTROL_NAMES = ["steer_sw_rad", "torque_fl", "torque_fr", "torque_rl", "torque_rr"]
VELOCITY_STATE_INDICES = [3, 4, 5, 9, 10, 11]

MLP_STATE_FEATURE_NAMES = [
    "trailer_mass_kg",
    "has_trailer",
    "vx_t",
    "vy_t",
    "r_t",
    "vx_s",
    "vy_s",
    "r_s",
    "rel_x_s_t",
    "rel_y_s_t",
    "sin_rel_yaw_s_t",
    "cos_rel_yaw_s_t",
]
# The current data has a fixed sample time, so dt is used by the base model and
# residual post-processing but is not passed into the MLP.
FIXED_DT_S = 0.02
# Rear-drive data uses the sum of left/right rear wheel torques as the net
# longitudinal drive command seen by the residual model.
MLP_CONTROL_FEATURE_NAMES = ["steer_sw_rad", "rear_drive_torque_sum"]
MLP_INPUT_FEATURE_NAMES = MLP_STATE_FEATURE_NAMES + MLP_CONTROL_FEATURE_NAMES

BASE_MODEL_PARAMS = {
    "m_t": 9300.0,
    "Iz_t": 48639.0,
    "L_t": 4.475,
    "a_t": 2.267,
    "m_s_base": 0.0,
    "Iz_s_base": 96659.0,
    "L_s": 8.0,
    "c_s": 4.0,
    "Cf": 2.64e5,
    "Cr": 3.35e5,
    "Cs": 80000.0,
    "wheel_radius": 0.5,
    "track_width": 1.878,
    "steering_ratio": 24.0,
    "rho": 1.225,
    "CdA_t": 0.6*9.7,
    "CdA_s": 0,
    "rolling_coeff": 0.013,
    "hitch_x": -0.3319,
    "hitch_y": 0.0022,
    "min_speed_mps": 0.5,
}

STATE_LOSS_WEIGHTS = {
    "x_t": 1.0,
    "y_t": 1.0,
    "psi_t": 5.0,
    "vx_t": 1.0,
    "vy_t": 1.0,
    "r_t": 5.0,
    "x_s": 1.0,
    "y_s": 1.0,
    "psi_s": 5.0,
    "vx_s": 1.0,
    "vy_s": 1.0,
    "r_s": 5.0,
}

TRAIN_BATCH_SIZE = 4096
TRAIN_NUM_WORKERS = 0
TRAIN_EPOCHS = 4000
LEARNING_RATE = 1.0e-3
# Cosine annealing gradually decays the learning rate from LEARNING_RATE to
# MIN_LEARNING_RATE across the whole training run.
MIN_LEARNING_RATE = 1.0e-5
# Pose loss stays disabled for the first 5000 optimizer steps, then turns on.
POSE_LOSS_WARMUP_STEPS = 5000
GRADIENT_CLIP_NORM = 200.0
# Turn-focused training first gates samples by steering-wheel angle, then uses
# a multi-signal turning severity score to rank those gated samples.
TURNING_FOCUS_STEER_THRESHOLD_DEG = 5.0
TURNING_GATE_BASE_WEIGHT = 3.0
TURNING_SCORE_COMPONENT_CLIP = 6.0
TURNING_SCORE_YAW_RATE_REF_DEGPS = 12.0
TURNING_SCORE_LATERAL_SPEED_REF_MPS = 0.8
TURNING_SCORE_ARTICULATION_REF_DEG = 3.0
TURNING_SCORE_YAW_RATE_WEIGHT = 1.0
TURNING_SCORE_LATERAL_SPEED_WEIGHT = 0.75
TURNING_SCORE_ARTICULATION_WEIGHT = 0.5
TURNING_FOCUS_START_QUANTILE = 0.70
TURNING_FOCUS_FULL_QUANTILE = 0.95
TURNING_SAMPLE_WEIGHT_MAX = 10.0
TURNING_SAMPLER_POWER = 1.5
TURNING_SELECTION_BLEND = 0.70
MLP_TORCH_DTYPE = torch.float32
MLP_NUMPY_DTYPE = np.float32
MLP_USE_LAYER_NORM = True
MLP_HIDDEN_DIM = 128
MLP_HIDDEN_LAYERS = 3
MLP_DROPOUT_P = 0.08
DEFAULT_TRAILER_MASS_KG = float(BASE_MODEL_PARAMS["m_s_base"])
NO_TRAILER_MASS_THRESHOLD_KG = 1.0
FORCE_NO_TRAILER_MODE = False

TRACTOR_X_CANDIDATES = ["X_t_m", "Tractor_X_m", "X_tractor_m", "TractorX_m", "X1_m", "X_m"]
TRACTOR_Y_CANDIDATES = ["Y_t_m", "Tractor_Y_m", "Y_tractor_m", "TractorY_m", "Y1_m", "Y_m"]
TRACTOR_YAW_DEG_CANDIDATES = ["Yaw_t_deg", "Tractor_Yaw_deg", "Yaw_tractor_deg", "Yaw1_deg", "Yaw_deg"]
TRACTOR_VX_CANDIDATES = ["Vx_t_mps", "Tractor_Vx_mps", "Vx_tractor_mps", "VxTractor_mps", "Vx_mps"]
TRACTOR_VY_CANDIDATES = ["Vy_t_mps", "Tractor_Vy_mps", "Vy_tractor_mps", "VyTractor_mps", "Vy_mps"]
TRACTOR_R_DEGPS_CANDIDATES = [
    "YawRate_t_degps",
    "Tractor_YawRate_degps",
    "YawRate_tractor_degps",
    "YawRate1_degps",
    "YawRate_degps",
]

TRAILER_X_CANDIDATES = ["X_s_m", "Trailer_X_m", "X_trailer_m", "TrailerX_m", "X2_m"]
TRAILER_Y_CANDIDATES = ["Y_s_m", "Trailer_Y_m", "Y_trailer_m", "TrailerY_m", "Y2_m"]
TRAILER_YAW_DEG_CANDIDATES = ["Yaw_s_deg", "Trailer_Yaw_deg", "Yaw_trailer_deg", "Yaw2_deg"]
TRAILER_VX_CANDIDATES = ["Vx_s_mps", "Trailer_Vx_mps", "Vx_trailer_mps", "TrailerSpeedX_mps"]
TRAILER_VY_CANDIDATES = ["Vy_s_mps", "Trailer_Vy_mps", "Vy_trailer_mps", "TrailerSpeedY_mps"]
TRAILER_R_DEGPS_CANDIDATES = ["YawRate_s_degps", "Trailer_YawRate_degps", "YawRate_trailer_degps", "YawRate2_degps"]

STEER_SW_RAD_CANDIDATES = [
    "Steer_SW_rad",
    "SteeringWheel_rad",
    "Steering_Wheel_rad",
    "Steering_Wheel_Angle_rad",
    "steering_wheel_rad",
]
STEER_SW_DEG_CANDIDATES = [
    "Steer_SW_deg",
    "Steer_SW",
    "Steer_deg_cmd",
    "SteeringWheel_deg",
    "Steering_Wheel_deg",
    "Steering_Wheel_Angle_deg",
]
ROAD_WHEEL_RAD_CANDIDATES = ["Steer_L1_rad", "SteerRoadWheel_rad", "delta_f_rad"]
ROAD_WHEEL_DEG_CANDIDATES = ["Steer_L1", "FrontWheelAngle_deg"]
STEERING_RATIO_CANDIDATES = ["Steering_Ratio_SW_to_L1", "SteeringRatio", "Steer_Ratio_SW_to_L1"]

TORQUE_FL_CANDIDATES = ["Torque_FL_Nm_cmd", "Torque_L1_Nm_cmd", "torque_fl_nm", "wheel_torque_fl_nm"]
TORQUE_FR_CANDIDATES = ["Torque_FR_Nm_cmd", "Torque_R1_Nm_cmd", "torque_fr_nm", "wheel_torque_fr_nm"]
TORQUE_RL_CANDIDATES = ["Torque_RL_Nm_cmd", "Torque_L2_Nm_cmd", "torque_rl_nm", "wheel_torque_rl_nm"]
TORQUE_RR_CANDIDATES = ["Torque_RR_Nm_cmd", "Torque_R2_Nm_cmd", "torque_rr_nm", "wheel_torque_rr_nm"]

TRAILER_MASS_COLUMN_CANDIDATES = ["TrailerMass_kg", "Trailer_Mass_kg", "m_s_kg", "m_trailer_kg"]
TRAILER_MASS_PARAMETER_CANDIDATES = [
    "M_S",
    "M_ST",
    "M_TRAILER",
    "M_SEMI_TRAILER",
    "M_SEMITRAILER",
    "TRAILER_MASS_KG",
]
