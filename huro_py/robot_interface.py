"""Everything that touches unitree messages / SDK: reading LowState, building LowCmd, releasing sport mode."""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch

from unitree_go.msg import LowCmd, LowState

from huro_py.crc_go import Crc


@dataclass
class RobotState:
    """Snapshot of /lowstate, joints in SDK order."""

    q: torch.Tensor  # [12]
    dq: torch.Tensor  # [12]
    gyro: torch.Tensor  # [3] body frame
    quat: torch.Tensor  # [4] (w, x, y, z)


def read_low_state(msg: LowState) -> RobotState:
    motors = msg.motor_state[:12]
    return RobotState(
        q=torch.tensor([m.q for m in motors], dtype=torch.float32),
        dq=torch.tensor([m.dq for m in motors], dtype=torch.float32),
        gyro=torch.tensor(list(msg.imu_state.gyroscope[:3]), dtype=torch.float32),
        quat=torch.tensor(list(msg.imu_state.quaternion[:4]), dtype=torch.float32),
    )


def build_low_cmd(q: torch.Tensor, kp: float, kd: float, tau: torch.Tensor | None = None) -> LowCmd:
    """PD command for the 12 joints (SDK order). ``tau`` is an optional feed-forward torque."""
    cmd = LowCmd()
    cmd.head[0] = 0xFE
    cmd.head[1] = 0xEF
    cmd.level_flag = 0xFF
    cmd.gpio = 0
    q = q.tolist()
    tau = [0.0] * 12 if tau is None else tau.tolist()
    for i in range(12):
        m = cmd.motor_cmd[i]
        m.mode = 0x01
        m.q = q[i]
        m.dq = 0.0
        m.kp = float(kp)
        m.kd = float(kd)
        m.tau = tau[i]
    cmd.crc = Crc(cmd)
    return cmd


def release_sport_mode(iface: str, logger) -> None:
    """Real robot only: sit down and disable the built-in sport mode so that low-level control is possible."""
    import sys

    sys.path.insert(0, "/tmp/unitree_sdk2_python")  # git clone https://github.com/unitreerobotics/unitree_sdk2_python
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize
    from unitree_sdk2py.go2.robot_state.robot_state_client import RobotStateClient
    from unitree_sdk2py.go2.sport.sport_client import SportClient

    ChannelFactoryInitialize(0, iface)
    sport = SportClient()
    sport.SetTimeout(10.0)
    sport.Init()
    logger.info("Sitting down...")
    sport.StandDown()
    time.sleep(2)
    logger.info("Robot seated.")

    robot_state = RobotStateClient()
    robot_state.Init()
    code = robot_state.ServiceSwitch("sport_mode", False)
    while code != 0:
        logger.warning(f"Error switching sport_mode off: {code}")
        time.sleep(1)
        code = robot_state.ServiceSwitch("sport_mode", False)
    logger.info("Sport mode disabled - low-level control available")
    time.sleep(1)


class RateMonitor:
    """Logs the achieved /lowcmd publish rate every ``period`` seconds."""

    def __init__(self, logger, period: float = 5.0):
        self.logger, self.period = logger, period
        self.count = self._last_count = 0
        self._t0 = self._last = time.perf_counter()

    def tick(self) -> None:
        self.count += 1
        now = time.perf_counter()
        if now - self._last < self.period:
            return
        avg = self.count / (now - self._t0)
        cur = (self.count - self._last_count) / (now - self._last)
        self.logger.info(f"lowcmd publish rate: avg={avg:.1f} Hz, interval={cur:.1f} Hz, count={self.count}")
        self._last, self._last_count = now, self.count