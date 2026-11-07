"""Open-loop posture helpers: joint ramps (stand up / down) and the emergency gain release."""

from __future__ import annotations

import torch


def stand_down_pose(default_pos_sdk: torch.Tensor) -> torch.Tensor:
    """Crouched pose in the same leg order as ``default_pos_sdk`` (front legs first, hip signs follow the default)."""
    hip_abs = (0.35, 0.35, 0.5, 0.5)
    pose = []
    for i in range(4):
        sign = 1.0 if default_pos_sdk[3 * i] >= 0 else -1.0
        pose += [sign * hip_abs[i], 1.36, -2.65]
    return torch.tensor(pose, dtype=torch.float32)


class JointRamp:
    """Linear interpolation q_start -> q_goal over ``duration`` seconds."""

    def __init__(self, q_start: torch.Tensor, q_goal: torch.Tensor, t0: float, duration: float):
        self.q0, self.q1, self.t0, self.duration = q_start.clone(), q_goal.clone(), t0, max(duration, 1e-6)

    def at(self, now_s: float) -> tuple[torch.Tensor, bool]:
        r = min(max((now_s - self.t0) / self.duration, 0.0), 1.0)
        return (1.0 - r) * self.q0 + r * self.q1, r >= 1.0


class EmergencyRelease:
    """Holds the latched posture while the PD gains decay to zero (robot goes limp)."""

    def __init__(self, q_hold: torch.Tensor, t0: float, duration: float, kp: float, kd: float):
        self.q_hold, self.t0, self.duration, self.kp, self.kd = q_hold.clone(), t0, duration, kp, kd

    def command(self, q: torch.Tensor, dq: torch.Tensor, now_s: float):
        r = min((now_s - self.t0) / self.duration, 1.0)
        scale = (1.0 - r) ** 10  # = 1 - alpha
        kp, kd = self.kp * scale, self.kd * scale
        # NOTE: kept from the original node: the PD torque is ALSO sent as feed-forward tau (double stiffness).
        tau = kp * (self.q_hold - q) - kd * dq
        return self.q_hold, kp, kd, tau