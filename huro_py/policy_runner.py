"""One locomotion policy step on the real robot, using the same classes as the simulator."""

from __future__ import annotations

import torch

from .go2_perception import LocomotionController
from .go2_perception.locomotion import build_proprio
from .go2_perception.math_utils import quat_apply, quat_conjugate

from huro_py.joint_mapper import JointMapper
from huro_py.robot_interface import RobotState


class PolicyRunner:
    """RobotState + command + height grid -> joint position targets (SDK order)."""

    def __init__(self, policy_path: str, mapper: JointMapper, cfg, device):
        self.mapper, self.device = mapper, device
        self.loc = LocomotionController(
            policy_path, num_envs=1, num_joints=12, action_scale=cfg.action_scale, device=device,
            clip_actions=cfg.clip_actions, filter_alpha=cfg.filter_alpha,
        )
        self.default_policy = mapper.default_pos_policy.to(device)
        self._gravity_w = torch.tensor([0.0, 0.0, -1.0], device=device)
        self._env0 = torch.zeros(1, dtype=torch.long, device=device)

    def reset(self) -> None:
        """Zero the GRU hidden state and the last-action memory (call whenever the policy is (re)started)."""
        self.loc.reset(self._env0)

    @torch.no_grad()
    def step(self, state: RobotState, cmd: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
        d = self.device
        q = self.mapper.to_policy(state.q).to(d)
        dq = self.mapper.to_policy(state.dq).to(d)
        gravity_b = quat_apply(quat_conjugate(state.quat.to(d)), self._gravity_w)  # projected gravity

        proprio = build_proprio(state.gyro.to(d)[None], gravity_b[None], (q - self.default_policy)[None], dq[None])
        obs = self.loc.build_obs(proprio, cmd.to(d)[None])  # [1, 45] = proprio(30) + cmd(3) + last action(12)
        self.loc.act(obs, grid)  # grid [1, 2, X, Y]
        target_policy = self.loc.joint_targets(self.default_policy[None])[0]
        return self.mapper.to_sdk(target_policy.cpu())