from __future__ import annotations

import torch
import torch.nn as nn

try:
    from .constants import NO_TRAILER_MASS_THRESHOLD_KG
except ImportError:
    from constants import NO_TRAILER_MASS_THRESHOLD_KG


def wrap_angle_error_np(angle):
    import numpy as np

    return ((angle + np.pi) % (2.0 * np.pi) - np.pi).astype(np.float32)


def wrap_angle_error_torch(angle: torch.Tensor) -> torch.Tensor:
    return torch.remainder(angle + torch.pi, 2.0 * torch.pi) - torch.pi


class TruckTrailerNominalDynamics(nn.Module):
    def __init__(self, params: dict[str, float]) -> None:
        super().__init__()
        self.register_buffer("m_t", torch.tensor(float(params["m_t"])))
        self.register_buffer("Iz_t", torch.tensor(float(params["Iz_t"])))
        self.register_buffer("L_t", torch.tensor(float(params["L_t"])))
        self.register_buffer("a_t", torch.tensor(float(params["a_t"])))
        self.register_buffer("m_s_base", torch.tensor(float(params["m_s_base"])))
        self.register_buffer("Iz_s_base", torch.tensor(float(params["Iz_s_base"])))
        self.register_buffer("L_s", torch.tensor(float(params["L_s"])))
        self.register_buffer("c_s", torch.tensor(float(params["c_s"])))
        self.register_buffer("Cf", torch.tensor(float(params["Cf"])))
        self.register_buffer("Cr", torch.tensor(float(params["Cr"])))
        self.register_buffer("Cs", torch.tensor(float(params["Cs"])))
        self.register_buffer("wheel_radius", torch.tensor(float(params["wheel_radius"])))
        self.register_buffer("track_width", torch.tensor(float(params["track_width"])))
        self.register_buffer("steering_ratio", torch.tensor(float(params["steering_ratio"])))
        self.register_buffer("rho", torch.tensor(float(params["rho"])))
        self.register_buffer("CdA_t", torch.tensor(float(params["CdA_t"])))
        self.register_buffer("CdA_s", torch.tensor(float(params["CdA_s"])))
        self.register_buffer("rolling_coeff", torch.tensor(float(params["rolling_coeff"])))
        self.register_buffer("hitch_x", torch.tensor(float(params["hitch_x"])))
        self.register_buffer("hitch_y", torch.tensor(float(params["hitch_y"])))
        self.register_buffer("min_speed_mps", torch.tensor(float(params["min_speed_mps"])))
        self.register_buffer("g", torch.tensor(9.81))
        self.register_buffer("_eps", torch.tensor(1.0e-8))
        self.no_trailer_mass_threshold_kg = NO_TRAILER_MASS_THRESHOLD_KG

    def _signed_safe_velocity(self, velocity: torch.Tensor) -> torch.Tensor:
        sign = torch.where(velocity >= 0.0, 1.0, -1.0).to(dtype=velocity.dtype, device=velocity.device)
        return sign * torch.clamp(torch.abs(velocity), min=float(self.min_speed_mps.item()))

    def derivatives(self, state: torch.Tensor, control: torch.Tensor, trailer_mass_kg: torch.Tensor) -> torch.Tensor:
        if trailer_mass_kg.ndim == 2 and trailer_mass_kg.shape[1] == 1:
            trailer_mass_kg = trailer_mass_kg[:, 0]

        has_trailer = trailer_mass_kg > self.no_trailer_mass_threshold_kg
        trailer_mask = has_trailer.to(dtype=state.dtype, device=state.device)
        safe_trailer_mass_kg = torch.where(
            has_trailer,
            torch.clamp(trailer_mass_kg, min=1000.0),
            torch.ones_like(trailer_mass_kg),
        )
        trailer_inertia = self.Iz_s_base * (safe_trailer_mass_kg / torch.clamp(self.m_s_base, min=1.0))

        x_t = state[:, 0]
        y_t = state[:, 1]
        psi_t = state[:, 2]
        vx_t = state[:, 3]
        vy_t = state[:, 4]
        r_t = state[:, 5]
        x_s = state[:, 6]
        y_s = state[:, 7]
        psi_s = state[:, 8]
        vx_s = state[:, 9]
        vy_s = state[:, 10]
        r_s = state[:, 11]

        steer_sw_rad = control[:, 0]
        torque_fl = control[:, 1]
        torque_fr = control[:, 2]
        torque_rl = control[:, 3]
        torque_rr = control[:, 4]

        delta_f = steer_sw_rad / self.steering_ratio
        b_t = self.L_t - self.a_t

        vx_t_safe = self._signed_safe_velocity(vx_t)
        vx_s_safe = self._signed_safe_velocity(vx_s)

        alpha_f = delta_f - torch.atan2(vy_t + self.a_t * r_t, vx_t_safe + self._eps)
        alpha_r = -torch.atan2(vy_t - b_t * r_t, vx_t_safe + self._eps)
        alpha_s = -torch.atan2(vy_s - self.L_s * r_s, vx_s_safe + self._eps)

        fyf = self.Cf * alpha_f
        fyr = self.Cr * alpha_r
        fys = self.Cs * alpha_s * trailer_mask

        cos_psi_t = torch.cos(psi_t)
        sin_psi_t = torch.sin(psi_t)
        cos_psi_s = torch.cos(psi_s)
        sin_psi_s = torch.sin(psi_s)

        hitch_global_x = x_t + self.hitch_x * cos_psi_t - self.hitch_y * sin_psi_t
        hitch_global_y = y_t + self.hitch_x * sin_psi_t + self.hitch_y * cos_psi_t

        hitch_vel_t_x_body = vx_t - r_t * self.hitch_y
        hitch_vel_t_y_body = vy_t + r_t * self.hitch_x
        hitch_vel_t_x_global = hitch_vel_t_x_body * cos_psi_t - hitch_vel_t_y_body * sin_psi_t
        hitch_vel_t_y_global = hitch_vel_t_x_body * sin_psi_t + hitch_vel_t_y_body * cos_psi_t

        hitch_global_s_x = x_s + self.c_s * cos_psi_s
        hitch_global_s_y = y_s + self.c_s * sin_psi_s
        hitch_vel_s_x_body = vx_s
        hitch_vel_s_y_body = vy_s - r_s * self.c_s
        hitch_vel_s_x_global = hitch_vel_s_x_body * cos_psi_s - hitch_vel_s_y_body * sin_psi_s
        hitch_vel_s_y_global = hitch_vel_s_x_body * sin_psi_s + hitch_vel_s_y_body * cos_psi_s

        pos_error_x = hitch_global_x - hitch_global_s_x
        pos_error_y = hitch_global_y - hitch_global_s_y
        vel_error_x = hitch_vel_t_x_global - hitch_vel_s_x_global
        vel_error_y = hitch_vel_t_y_global - hitch_vel_s_y_global

        k_pos = 1.0e6
        k_vel = 1.0e4
        hitch_force_x_global = (-k_pos * pos_error_x - k_vel * vel_error_x) * trailer_mask
        hitch_force_y_global = (-k_pos * pos_error_y - k_vel * vel_error_y) * trailer_mask

        hitch_force_t_x_body = hitch_force_x_global * cos_psi_t + hitch_force_y_global * sin_psi_t
        hitch_force_t_y_body = -hitch_force_x_global * sin_psi_t + hitch_force_y_global * cos_psi_t
        hitch_force_s_x_body = -(hitch_force_x_global * cos_psi_s + hitch_force_y_global * sin_psi_s)
        hitch_force_s_y_body = -(-hitch_force_x_global * sin_psi_s + hitch_force_y_global * cos_psi_s)

        fx_fl = torque_fl / self.wheel_radius
        fx_fr = torque_fr / self.wheel_radius
        fx_rl = torque_rl / self.wheel_radius
        fx_rr = torque_rr / self.wheel_radius

        cos_delta = torch.cos(delta_f)
        sin_delta = torch.sin(delta_f)
        front_longitudinal = fx_fl + fx_fr
        rear_longitudinal = fx_rl + fx_rr
        fx_front_body = front_longitudinal * cos_delta
        fy_front_from_drive = front_longitudinal * sin_delta

        tractor_speed = torch.sqrt(vx_t * vx_t + vy_t * vy_t + self._eps)
        trailer_speed = torch.sqrt(vx_s * vx_s + vy_s * vy_s + self._eps)
        drag_t = -0.5 * self.rho * self.CdA_t * tractor_speed * vx_t
        drag_s = -0.5 * self.rho * self.CdA_s * trailer_speed * vx_s * trailer_mask
        roll_t = self.rolling_coeff * self.m_t * self.g * torch.tanh(10.0 * vx_t)
        roll_s = self.rolling_coeff * safe_trailer_mass_kg * self.g * torch.tanh(10.0 * vx_s) * trailer_mask

        fx_total_t = fx_front_body + rear_longitudinal + fyf * sin_delta + hitch_force_t_x_body + drag_t - roll_t
        fy_total_t = fyf * cos_delta + fyr + hitch_force_t_y_body + fy_front_from_drive

        dvx_t = fx_total_t / self.m_t + r_t * vy_t
        dvy_t = fy_total_t / self.m_t - r_t * vx_t
        dpsi_t = r_t
        dr_t = (
            self.a_t * (fyf * cos_delta + fy_front_from_drive)
            - b_t * fyr
            + (self.hitch_x * hitch_force_t_y_body - self.hitch_y * hitch_force_t_x_body)
            + (fx_fr - fx_fl) * (self.track_width * 0.5)
            + (fx_rr - fx_rl) * (self.track_width * 0.5)
        ) / self.Iz_t

        dvx_s_trailer = (hitch_force_s_x_body + drag_s - roll_s) / safe_trailer_mass_kg + r_s * vy_s
        dvy_s_trailer = (fys + hitch_force_s_y_body) / safe_trailer_mass_kg - r_s * vx_s
        dpsi_s_trailer = r_s
        dr_s_trailer = (-self.L_s * fys + self.c_s * hitch_force_s_y_body) / trailer_inertia

        dx_t = vx_t * cos_psi_t - vy_t * sin_psi_t
        dy_t = vx_t * sin_psi_t + vy_t * cos_psi_t
        dx_s_trailer = vx_s * cos_psi_s - vy_s * sin_psi_s
        dy_s_trailer = vx_s * sin_psi_s + vy_s * cos_psi_s

        dx_s = trailer_mask * dx_s_trailer + (1.0 - trailer_mask) * dx_t
        dy_s = trailer_mask * dy_s_trailer + (1.0 - trailer_mask) * dy_t
        dpsi_s = trailer_mask * dpsi_s_trailer + (1.0 - trailer_mask) * dpsi_t
        dvx_s = trailer_mask * dvx_s_trailer + (1.0 - trailer_mask) * dvx_t
        dvy_s = trailer_mask * dvy_s_trailer + (1.0 - trailer_mask) * dvy_t
        dr_s = trailer_mask * dr_s_trailer + (1.0 - trailer_mask) * dr_t

        return torch.stack(
            [dx_t, dy_t, dpsi_t, dvx_t, dvy_t, dr_t, dx_s, dy_s, dpsi_s, dvx_s, dvy_s, dr_s],
            dim=1,
        )

    def forward(self, state: torch.Tensor, control: torch.Tensor, trailer_mass_kg: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
        if dt.ndim == 1:
            dt = dt.unsqueeze(1)
        if trailer_mass_kg.ndim == 1:
            trailer_mass_kg = trailer_mass_kg.unsqueeze(1)

        k1 = self.derivatives(state, control, trailer_mass_kg)
        k2 = self.derivatives(state + 0.5 * dt * k1, control, trailer_mass_kg)
        k3 = self.derivatives(state + 0.5 * dt * k2, control, trailer_mass_kg)
        k4 = self.derivatives(state + dt * k3, control, trailer_mass_kg)
        next_state = state + dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
        next_state = next_state.clone()
        no_trailer_mask = trailer_mass_kg[:, 0] <= self.no_trailer_mass_threshold_kg
        if torch.any(no_trailer_mask):
            next_state[no_trailer_mask, 6] = next_state[no_trailer_mask, 0]
            next_state[no_trailer_mask, 7] = next_state[no_trailer_mask, 1]
            next_state[no_trailer_mask, 8] = next_state[no_trailer_mask, 2]
            next_state[no_trailer_mask, 9] = next_state[no_trailer_mask, 3]
            next_state[no_trailer_mask, 10] = next_state[no_trailer_mask, 4]
            next_state[no_trailer_mask, 11] = next_state[no_trailer_mask, 5]
        next_state[:, 2] = wrap_angle_error_torch(next_state[:, 2])
        next_state[:, 8] = wrap_angle_error_torch(next_state[:, 8])
        return next_state
