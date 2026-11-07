#!/usr/bin/env python3
"""
RL locomotion controller for the Unitree Go2 (perceptive CNN-RNN policy, 50 Hz).

The policy, observation layout and height grid come from the Isaac-free ``go2_perception`` package, i.e. the exact
code used in simulation. This node only does ROS I/O, the joystick state machine and low-level command publishing.

TO RUN (simulation):
    ros2 launch huro go2_rviz.launch.py
    ros2 run huro spacemouse_publisher.py
    ros2 run huro sim_go2
    ros2 run huro go2_publisher_cnn.py --sim true

Joystick (buttons configurable in DeployConfig.buttons):
    4 stand up | 0 stand down | 9 start policy | 8 emergency (latched, kill the process to restart)
"""

from __future__ import annotations

import argparse
import os
from enum import Enum, auto

import rclpy
import torch
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import Twist
from rclpy.node import Node
from sensor_msgs.msg import Joy, PointCloud2
from unitree_go.msg import LowCmd, LowState

from huro_py.commands import CommandSelector
from huro_py.deploy_config import DeployConfig
from huro_py.joint_mapper import JointMapper
from huro_py.lidar_map import LidarHeightMap, pointcloud2_xyz
from huro_py.policy_runner import PolicyRunner
from huro_py.posture import EmergencyRelease, JointRamp, stand_down_pose
from huro_py.robot_interface import RateMonitor, build_low_cmd, read_low_state, release_sport_mode


"""
cd /tmp
git clone https://github.com/unitreerobotics/unitree_sdk2_python.git
cd unitree_sdk2_python
pip3 install -e .
"""


class Mode(Enum):
    IDLE = auto()  # nothing is published
    STAND_UP = auto()
    STAND_DOWN = auto()
    POLICY = auto()
    EMERGENCY = auto()  # latched


class Go2PolicyController(Node):
    def __init__(self, cfg: DeployConfig, vx=None, vy=None, wz=None):
        params = [rclpy.parameter.Parameter("use_sim_time", value=True)] if cfg.sim else []
        super().__init__("go2_policy_controller", parameter_overrides=params)
        self.cfg = cfg
        self.get_logger().info(f"use_sim_time: {self.get_parameter('use_sim_time').value}")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.get_logger().info(f"Using device: {self.device}")

        # ---- model, joint mapping, sensors, commands
        share = get_package_share_directory("huro")
        policy_path = os.path.join(share, "resources", "models", "go2", cfg.policy_name)
        mapping_path = os.path.join(share, "resources", "mappings", "go2", cfg.mapping_file)
        for path in (policy_path, mapping_path):
            if not os.path.exists(path):
                raise FileNotFoundError(path)

        self.mapper = JointMapper(mapping_path, cfg.default_pos_sdk)
        self.runner = PolicyRunner(policy_path, self.mapper, cfg, self.device)
        self.lidar_map = LidarHeightMap(cfg, self.device)
        self.commands = CommandSelector(cfg)
        self.stand_up_pos = self.mapper.default_pos_sdk
        self.stand_down_pos = stand_down_pose(self.mapper.default_pos_sdk)
        self.get_logger().info(f"Policy loaded: {policy_path}")

        # ---- state machine
        self.mode = Mode.IDLE
        self.low_state: LowState | None = None
        self.last_cmd_q: torch.Tensor | None = None
        self.ramp: JointRamp | None = None
        self.release: EmergencyRelease | None = None
        self._events: list[str] = []  # button rising edges, consumed by the control tick
        self._prev_pressed: set[int] = set()
        if any(v is not None for v in (vx, vy, wz)):  # headless: constant command, start the policy directly
            self.commands.set_fixed(vx, vy, wz)
            self.mode = Mode.POLICY

        # ---- ROS I/O
        self.low_cmd_pub = self.create_publisher(LowCmd, "/lowcmd", 10)
        self.rate = RateMonitor(self.get_logger())
        self.create_subscription(LowState, "/lowstate", self._on_low_state, 10)
        self.create_subscription(Joy, "/joy", self._on_joy, 10)
        self.create_subscription(PointCloud2, "/utlidar/cloud", self._on_lidar, 10)
        self.create_subscription(Twist, "/cmd_vel_nav", self._on_cmd_vel, 10)

        if not cfg.sim:
            release_sport_mode(cfg.iface, self.get_logger())

        self.create_timer(cfg.step_dt, self.run)
        self.get_logger().info(f"Controller ready, policy at {1.0 / cfg.step_dt:.0f} Hz")

    def _now_s(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    # ------------------------------------------------------------------ callbacks
    def _on_low_state(self, msg: LowState) -> None:
        self.low_state = msg

    def _on_lidar(self, msg: PointCloud2) -> None:
        self.lidar_map.update(pointcloud2_xyz(msg), self._now_s())

    def _on_cmd_vel(self, msg: Twist) -> None:
        self.commands.on_nav(msg, self._now_s())

    def _on_joy(self, msg: Joy) -> None:
        self.commands.on_joy(msg)
        pressed = {i for i, b in enumerate(msg.buttons) if b}
        for name, idx in self.cfg.buttons.items():
            if idx in pressed and idx not in self._prev_pressed:
                self._events.append(name)
        self._prev_pressed = pressed

    # ------------------------------------------------------------------ publishing
    def _publish(self, q: torch.Tensor, kp: float, kd: float, tau: torch.Tensor | None = None) -> None:
        self.last_cmd_q = q.clone()
        self.low_cmd_pub.publish(build_low_cmd(q, kp, kd, tau))
        self.rate.tick()

    # ------------------------------------------------------------------ state machine
    def _handle_button(self, name: str, state, now: float) -> None:
        log = self.get_logger()
        if self.mode == Mode.EMERGENCY:
            return
        if name == "emergency":
            log.error("Emergency mode activated. Kill the process to restart.")
            self.release = EmergencyRelease(
                state.q, now, self.cfg.emergency_release_time, self.cfg.kp_stand, self.cfg.kd_stand
            )
            self.mode = Mode.EMERGENCY
        elif name in ("stand_up", "stand_down"):
            up = name == "stand_up"
            log.info("Standing up..." if up else "Standing down...")
            q0 = self.last_cmd_q if self.last_cmd_q is not None else state.q
            self.ramp = JointRamp(q0, self.stand_up_pos if up else self.stand_down_pos, now, self.cfg.time_to_stand)
            self.mode = Mode.STAND_UP if up else Mode.STAND_DOWN
        elif name == "policy":
            if self.mode == Mode.STAND_DOWN:
                log.warning("Policy request ignored while standing down: stand up first.")
            elif not self.lidar_map.has_data:
                log.warning("Policy request ignored: no lidar cloud received yet.")
            else:
                log.info("Policy running...")
                self.runner.reset()
                self.mode = Mode.POLICY

    def run(self) -> None:
        """Control tick (cfg.step_dt)."""
        if self.low_state is None:
            self.get_logger().info("Waiting for /lowstate...", throttle_duration_sec=2.0)
            return
        state = read_low_state(self.low_state)
        now = self._now_s()
        events, self._events = self._events, []
        for name in events:
            self._handle_button(name, state, now)

        c = self.cfg
        if self.mode in (Mode.STAND_UP, Mode.STAND_DOWN):
            q, _ = self.ramp.at(now)
            self._publish(q, c.kp_stand, c.kd_stand)
        elif self.mode == Mode.POLICY:
            self._policy_tick(state, now)
        elif self.mode == Mode.EMERGENCY:
            q, kp, kd, tau = self.release.command(state.q, state.dq, now)
            self._publish(q, kp, kd, tau)

    def _policy_tick(self, state, now: float) -> None:
        log = self.get_logger()
        if not self.lidar_map.has_data:
            log.info("Waiting for /utlidar/cloud...", throttle_duration_sec=2.0)
            return
        if self.lidar_map.stale(now):
            log.warning("No lidar cloud: height grid cleared (all cells invalid).", throttle_duration_sec=2.0)
            self.lidar_map.clear()
        cmd = self.commands.get(now)
        q_target = self.runner.step(state, cmd, self.lidar_map.observation())
        self._publish(q_target, self.cfg.kp_policy, self.cfg.kd_policy)


def main():
    def str2bool(v):
        if isinstance(v, bool):
            return v
        val = str(v).strip().lower()
        if val in {"true", "t", "1", "yes", "y", "on"}:
            return True
        if val in {"false", "f", "0", "no", "n", "off"}:
            return False
        raise argparse.ArgumentTypeError(f"Invalid boolean value: {v}")

    parser = argparse.ArgumentParser(description="Go2 RL policy controller")
    parser.add_argument("--sim", type=str2bool, default=False, help="simulation (true) or real robot (false)")
    parser.add_argument("--vx", type=float, default=None, help="constant forward velocity (starts the policy directly)")
    parser.add_argument("--vy", type=float, default=None, help="constant lateral velocity")
    parser.add_argument("--wz", type=float, default=None, help="constant yaw rate")
    parser.add_argument("--policy", type=str, default=None, help="policy file name in resources/models/go2")
    parser.add_argument("--iface", type=str, default=None, help="network interface of the real robot")
    args, _ = parser.parse_known_args()  # leave ROS args alone

    cfg = DeployConfig(sim=args.sim)
    if args.policy:
        cfg.policy_name = args.policy
    if args.iface:
        cfg.iface = args.iface

    rclpy.init()
    node = Go2PolicyController(cfg, vx=args.vx, vy=args.vy, wz=args.wz)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()