"""Velocity command selection: fixed override (headless) > joystick > /cmd_vel_nav (snapped, with timeout)."""

from __future__ import annotations

import torch


def snap(value: float, levels: tuple[float, ...], deadband: float) -> float:
    """0 inside the deadband, else sign(value) * smallest level >= |value| (saturating at the largest level)."""
    a = abs(value)
    if a < deadband:
        return 0.0
    level = next((l for l in levels if a <= l), levels[-1])
    return level if value > 0 else -level


class CommandSelector:
    def __init__(self, cfg):
        self.cfg = cfg
        self._fixed: tuple[float, float, float] | None = None
        self._joy = (0.0, 0.0, 0.0)
        self._nav = (0.0, 0.0, 0.0)
        self._nav_time: float | None = None

    def set_fixed(self, vx, vy, wz) -> None:
        """Constant command (headless runs); used as is, no snapping."""
        self._fixed = tuple(0.0 if v is None else float(v) for v in (vx, vy, wz))

    def on_joy(self, msg) -> None:
        a = msg.axes
        if len(a) >= 3:
            self._joy = (float(a[1]), float(a[0]), float(a[2]))  # forward, lateral, yaw

    def on_nav(self, msg, now_s: float) -> None:
        self._nav = (float(msg.linear.x), float(msg.linear.y), float(msg.angular.z))
        self._nav_time = now_s

    def get(self, now_s: float) -> torch.Tensor:
        c = self.cfg
        if self._fixed is not None:
            vel = self._fixed
        elif max(abs(v) for v in self._joy) > c.joy_deadband:
            vel = self._joy
        elif self._nav_time is not None and (now_s - self._nav_time) <= c.cmd_timeout:
            vx, vy, wz = self._nav
            vel = (snap(vx, c.cmd_levels_lin, c.cmd_deadband),
                   snap(vy, c.cmd_levels_lin, c.cmd_deadband),
                   snap(wz, c.cmd_levels_yaw, c.cmd_deadband))
        else:
            vel = (0.0, 0.0, 0.0)
        vel = [max(-c.cmd_max, min(c.cmd_max, v)) for v in vel]
        return torch.tensor(vel, dtype=torch.float32)