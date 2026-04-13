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
POSE_STATE_NAMES = ["x_t", "y_t", "psi_t", "x_s", "y_s", "psi_s"]
MOTION_ERROR_NAMES = ["vx_t", "vy_t", "r_t", "vx_s", "vy_s", "r_s"]
CONTROL_NAMES = ["steer_sw_rad", "torque_fl", "torque_fr", "torque_rl", "torque_rr"]
VELOCITY_STATE_INDICES = [3, 4, 5, 9, 10, 11]

MLP_STATE_FEATURE_NAMES = [
    "trailer_mass_kg",
    "vx_t",
    "vy_t",
    "r_t",
    "speed_t",
    "vx_s",
    "vy_s",
    "r_s",
    "speed_s",
    "articulation_rad",
    "sin_articulation",
    "cos_articulation",
]
MLP_INPUT_FEATURE_NAMES = MLP_STATE_FEATURE_NAMES + CONTROL_NAMES + ["dt"]

BASE_MODEL_PARAMS = {
    "m_t": 9300.0,
    "Iz_t": 48639.0,
    "L_t": 4.475,
    "a_t": 3.8,
    "m_s_base": 15004.0,
    "Iz_s_base": 96659.0,
    "L_s": 8.0,
    "c_s": 4.0,
    "Cf": 80000.0,
    "Cr": 80000.0,
    "Cs": 80000.0,
    "wheel_radius": 0.5,
    "track_width": 1.8,
    "steering_ratio": 16.39,
    "rho": 1.225,
    "CdA_t": 5.82,
    "CdA_s": 6.50,
    "rolling_coeff": 0.006,
    "hitch_x": -0.331,
    "hitch_y": 0.002,
    "min_speed_mps": 0.5,
}

STATE_LOSS_WEIGHTS = {
    "x_t": 1.0,
    "y_t": 1.0,
    "psi_t": 2.0,
    "vx_t": 1.0,
    "vy_t": 1.0,
    "r_t": 5.0,
    "x_s": 1.0,
    "y_s": 1.0,
    "psi_s": 2.0,
    "vx_s": 1.0,
    "vy_s": 1.0,
    "r_s": 5.0,
}

TRAIN_BATCH_SIZE = 2048
TRAIN_NUM_WORKERS = 0
TRAIN_EPOCHS = 4000
LEARNING_RATE = 3.0e-3
MOTION_ONLY_WARMUP_EPOCHS = 600
POSE_RAMP_EPOCHS = 300
GRADIENT_CLIP_NORM = 200.0
MLP_TORCH_DTYPE = torch.float32
MLP_NUMPY_DTYPE = np.float32
MLP_USE_LAYER_NORM = True
DEFAULT_TRAILER_MASS_KG = float(BASE_MODEL_PARAMS["m_s_base"])
NO_TRAILER_MASS_THRESHOLD_KG = 1.0

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
    "SteeringWheel_deg",
    "Steering_Wheel_deg",
    "Steering_Wheel_Angle_deg",
]
ROAD_WHEEL_RAD_CANDIDATES = ["Steer_L1_rad", "SteerRoadWheel_rad", "delta_f_rad"]
ROAD_WHEEL_DEG_CANDIDATES = ["Steer_deg_cmd", "Steer_L1", "Target_Steer_L1_deg_cmd", "FrontWheelAngle_deg"]
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
