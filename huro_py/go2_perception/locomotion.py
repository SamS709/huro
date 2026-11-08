"""Frozen low-level locomotion policy (TorchScript) and its observation / action bookkeeping."""

from __future__ import annotations

import os

import torch

# Observation noise of the locomotion policy (ang vel, projected gravity, joint pos, joint vel). Sim only.
PROPRIO_NOISE = (0.1, 0.05, 0.01, 0.1)


def build_proprio(ang_vel, gravity, joint_pos_rel, joint_vel, noise_scale: float = 0.0) -> torch.Tensor:
    """Policy proprio [N,30]. ``noise_scale`` = 0 on the robot, 1.0 (randomize) in simulation."""
    parts = []
    for x, a in zip((ang_vel, gravity, joint_pos_rel, joint_vel), PROPRIO_NOISE):
        if noise_scale > 0.0:
            x = x + (2.0 * torch.rand_like(x) - 1.0) * a * noise_scale
        parts.append(x)
    return torch.cat(parts, dim=-1)


def build_loc_obs(proprio: torch.Tensor, cmd_vel: torch.Tensor, last_actions: torch.Tensor) -> torch.Tensor:
    """One actor FRAME: [proprio(30), cmd(3), last action(12)] = 45 values for 12 joints."""
    return torch.cat([proprio, cmd_vel, last_actions], dim=-1)


def loc_frame_dim(num_joints: int) -> int:
    """Size of one frame: proprio (3 + 3 + 2 * num_joints) + cmd (3) + last action (num_joints)."""
    return 9 + 3 * num_joints


class ProprioHistory:
    """Rolling history of actor frames: [frame(t - H), ..., frame(t)] (oldest first), flattened to [N, (H+1) * D].

    ``hist_length`` = H = number of PAST frames, so the policy input has H + 1 frames. H = 0 gives the plain
    current frame. Cleared to zeros at reset (identical in simulation and on the robot). Push exactly once per
    control step, with the frame as the policy sees it (i.e. after noise and sensor delay).
    """

    def __init__(self, num_envs: int, frame_dim: int, hist_length: int, device):
        self.n_frames = int(hist_length) + 1
        self.frame_dim = int(frame_dim)
        self.buf = torch.zeros(num_envs, self.n_frames, self.frame_dim, device=device)

    @property
    def dim(self) -> int:
        return self.n_frames * self.frame_dim

    def reset(self, env_ids: torch.Tensor) -> None:
        self.buf[env_ids] = 0.0

    def push(self, frame: torch.Tensor) -> torch.Tensor:
        if self.n_frames > 1:  # in place: keeps ``buf`` a normal tensor even when called under inference_mode
            self.buf[:, :-1] = self.buf[:, 1:].clone()
        self.buf[:, -1] = frame
        return self.buf.reshape(self.buf.shape[0], -1).clone()


class LocomotionController:
    """Wraps the TorchScript locomotion policy.

    Per control step:
        frame = loc.build_obs(proprio, cmd_vel)     # proprio [N,30] = (ang_vel, gravity, joint_pos_rel, joint_vel)
        loc.act(frame, height_grid)                 # height_grid [N, 2, X, Y] = (height, valid)
        targets = loc.joint_targets(default_joint_pos)

    ``act`` pushes the frame into the proprio history and gives the policy [frame(t-H), ..., frame(t)] flattened
    to [N, (H+1)*45]. ``proprio_hist_length`` (H) MUST equal the value the policy was trained with.
    """

    def __init__(self, policy_path: str, num_envs: int, num_joints: int, action_scale: float, device,
                 hidden_dim: int = 128, clip_actions: float | None = None, filter_alpha: float | None = None,
                 proprio_hist_length: int = 0):
        self.device = device
        self.num_envs = num_envs
        self.action_scale = float(action_scale)
        # Optional, must match the training env (cfg.clamp_actions / desired_clip_actions, cfg.filter_actions).
        # The raw (clipped) action is what the policy gets back as 'last action'; only the joint target is filtered.
        self.clip_actions = None if clip_actions is None else float(clip_actions)
        self.filter_alpha = None if filter_alpha is None else float(filter_alpha)

        path = os.path.expanduser(policy_path)
        if not os.path.isabs(path):
            path = os.path.join(os.getcwd(), path)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Could not find locomotion policy at '{path}' (expected a TorchScript file).")
        self.policy = torch.jit.load(path, map_location=device)
        self.policy.eval()
        with torch.no_grad():
            self.policy.hidden_state = torch.zeros(
                1, num_envs, hidden_dim, dtype=self.policy.hidden_state.dtype, device=self.policy.hidden_state.device
            )

        self.actions = torch.zeros(num_envs, num_joints, device=device)
        self.prev_actions = torch.zeros_like(self.actions)
        self.history = ProprioHistory(num_envs, loc_frame_dim(num_joints), proprio_hist_length, device)

    def reset(self, env_ids: torch.Tensor) -> None:
        with torch.no_grad():
            self.policy.hidden_state[:, env_ids, :] = 0.0
        self.actions[env_ids] = 0.0
        self.prev_actions[env_ids] = 0.0
        self.history.reset(env_ids)

    def build_obs(self, proprio: torch.Tensor, cmd_vel: torch.Tensor) -> torch.Tensor:
        """Current frame [N,45]. Uses the PREVIOUS action (call before ``act``)."""
        return build_loc_obs(proprio, cmd_vel, self.actions)

    def act(self, frame: torch.Tensor, height_grid: torch.Tensor) -> torch.Tensor:
        self.prev_actions.copy_(self.actions)
        obs = self.history.push(frame)  # [N, (H+1)*45]
        with torch.inference_mode():
            out = self.policy(obs, [height_grid])
        if isinstance(out, (tuple, list)):
            out = out[0]
        out = out.to(self.device)
        if self.clip_actions is not None:
            out = out.clamp(-self.clip_actions, self.clip_actions)
        self.actions.copy_(out)
        return self.actions

    def joint_targets(self, default_joint_pos: torch.Tensor) -> torch.Tensor:
        a = self.actions
        if self.filter_alpha is not None:
            a = self.filter_alpha * a + (1.0 - self.filter_alpha) * self.prev_actions
        return default_joint_pos + self.action_scale * a