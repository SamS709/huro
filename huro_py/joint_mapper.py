"""Joint order conversion between the policy (Isaac Lab) and the SDK, by joint name."""

from __future__ import annotations

import torch
import yaml


class JointMapper:
    def __init__(self, mapping_yaml_path: str, default_pos_sdk):
        with open(mapping_yaml_path) as f:
            cfg = yaml.safe_load(f)
        self.policy_names: list[str] = cfg["source_joint_names"]  # Isaac Lab order
        self.sdk_names: list[str] = cfg["target_joint_names"]  # Unitree SDK order
        if sorted(self.policy_names) != sorted(self.sdk_names):
            raise ValueError(f"Joint name sets differ in {mapping_yaml_path}")
        # policy[i] = sdk[policy_from_sdk[i]]  and  sdk[j] = policy[sdk_from_policy[j]]
        self.policy_from_sdk = torch.tensor([self.sdk_names.index(n) for n in self.policy_names])
        self.sdk_from_policy = torch.tensor([self.policy_names.index(n) for n in self.sdk_names])

        self.default_pos_sdk = torch.as_tensor(default_pos_sdk, dtype=torch.float32)
        self.default_pos_policy = self.to_policy(self.default_pos_sdk)

    def to_policy(self, values_sdk: torch.Tensor) -> torch.Tensor:
        return values_sdk[self.policy_from_sdk]

    def to_sdk(self, values_policy: torch.Tensor) -> torch.Tensor:
        return values_policy[self.sdk_from_policy]