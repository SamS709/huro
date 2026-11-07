"""Single place for every deployment parameter (gains, grid geometry, command mapping, joystick buttons)."""

from __future__ import annotations

from dataclasses import dataclass, field

# Default joint pose in SDK order. Real robot SDK order: FR, FL, RR, RL. Simulator (mujoco) order: FL, FR, RL, RR.
DEFAULT_POS_SDK_REAL = (-0.1, 0.8, -1.5, 0.1, 0.8, -1.5, -0.1, 1.0, -1.5, 0.1, 1.0, -1.5)
DEFAULT_POS_SDK_SIM = (0.1, 0.8, -1.5, -0.1, 0.8, -1.5, 0.1, 1.0, -1.5, -0.1, 1.0, -1.5)


@dataclass
class DeployConfig:
    sim: bool = False
    policy_name: str = "policy_cnn_rnn_robust_1.pt"
    iface: str = "enp0s31f6"  # network interface of the real robot

    # ---- timing
    step_dt: float = 1.0 / 50.0  # policy rate; must equal the training control dt

    # ---- low-level PD gains
    kp_stand: float = 65.0
    kd_stand: float = 5.0
    kp_policy: float = 25.0  # MUST match the training actuator model
    kd_policy: float = 0.5
    action_scale: float = 0.25
    clip_actions: float | None = None  # training cfg.clamp_actions -> desired_clip_actions
    filter_alpha: float | None = None  # training cfg.filter_actions -> alpha (0.8 in the training env)

    time_to_stand: float = 5.0  # s, stand up / down ramp
    emergency_release_time: float = 5.0  # s, gains decay to zero

    # ---- height grid: MUST match Go2LidarRoughEnvCfg (x_range, y_range, res, desired_base_height).
    # Coordinates are the policy ray convention, i.e. the NEGATED lidar-frame hit coordinates.
    x_range: tuple[float, float] = (-0.3, 0.5)
    y_range: tuple[float, float] = (-0.5, 0.5)
    res: float = 0.1
    desired_base_height: float = 0.28
    lidar_max_age_scans: int = 1  # a cell survives this many scans after its last hit (old: delete_count)
    lidar_timeout: float = 1.0  # s without a cloud -> grid is cleared (all cells invalid)

    # ---- commands
    cmd_timeout: float = 0.5  # s, stale /cmd_vel_nav -> zero command
    joy_deadband: float = 1e-3
    cmd_deadband: float = 1e-3
    cmd_max: float = 1.0
    # Nav commands are snapped to the speeds the policy was trained with: smallest level >= |cmd|, saturating.
    cmd_levels_lin: tuple[float, ...] = (0.2, 0.5, 0.7)
    cmd_levels_yaw: tuple[float, ...] = (0.2, 0.5)

    # ---- joystick (sensor_msgs/Joy button indices)
    buttons: dict = field(default_factory=lambda: {"emergency": 8, "policy": 9, "stand_down": 0, "stand_up": 4})

    @property
    def default_pos_sdk(self) -> tuple[float, ...]:
        return DEFAULT_POS_SDK_SIM if self.sim else DEFAULT_POS_SDK_REAL

    @property
    def mapping_file(self) -> str:
        return "physx_to_mujoco_go2.yaml" if self.sim else "sim_to_real.yaml"