"""Real lidar -> the SAME height grid the policy was trained with (go2_perception.HeightGrid).

The utlidar cloud is expressed in the lidar frame (hit relative to the sensor). The training convention is
``rays = -(hit relative to the scanner)``, so the points are simply negated and given to ``HeightGrid.compute``.
Cells without a hit are 0 with valid=0 (as in simulation). Because the real lidar is sparser than the simulated ray
caster, cells are kept for a few scans (``max_age_scans``) before being cleared.
"""

from __future__ import annotations

import numpy as np
import torch

from .go2_perception import HeightGrid


def pointcloud2_xyz(msg) -> np.ndarray:
    """sensor_msgs/PointCloud2 -> finite [P,3] float32 array (x, y, z must be the first three float32 fields)."""
    n = int(msg.width) * int(msg.height)
    step = int(msg.point_step)
    empty = np.empty((0, 3), dtype=np.float32)
    if n == 0 or step < 12:
        return empty
    raw = np.frombuffer(msg.data, dtype=np.uint8)
    if raw.size < n * step:
        return empty
    xyz = np.ndarray(shape=(n, 3), dtype=np.float32, buffer=raw, offset=0, strides=(step, 4))
    return xyz[np.isfinite(xyz).all(axis=1)]  # boolean indexing copies -> writable


class LidarHeightMap:
    def __init__(self, cfg, device):
        self.device = device
        self.grid = HeightGrid(cfg.x_range, cfg.y_range, cfg.res, cfg.desired_base_height, device)
        self.max_age = int(cfg.lidar_max_age_scans)
        self.timeout = float(cfg.lidar_timeout)
        n = self.grid.num_cells
        self.height = torch.zeros(n, device=device)  # final policy convention
        self.valid = torch.zeros(n, dtype=torch.bool, device=device)
        self.age = torch.zeros(n, device=device)
        self.last_update: float | None = None

    @property
    def has_data(self) -> bool:
        return self.last_update is not None

    def stale(self, now_s: float) -> bool:
        return self.last_update is not None and (now_s - self.last_update) > self.timeout

    def clear(self) -> None:
        self.height.zero_()
        self.valid.zero_()
        self.age.zero_()

    @torch.no_grad()
    def update(self, xyz: np.ndarray, now_s: float) -> None:
        rays = -torch.from_numpy(xyz).to(self.device)[None]  # [1,P,3], policy ray convention
        h, hit = self.grid.compute(rays)
        h, hit = h[0], hit[0]
        old = self.age > self.max_age
        self.height[old] = 0.0
        self.valid[old] = False
        self.age += 1.0
        self.height[hit] = h[hit]
        self.valid[hit] = True
        self.age[hit] = 0.0
        self.last_update = now_s

    def observation(self) -> torch.Tensor:
        """[1, 2, X, Y] = (height, valid), same layout as the locomotion actor grid in simulation."""
        h = torch.nan_to_num(self.height[None]).clamp(-10.0, 10.0)
        return torch.stack([self.grid.to_image(h), self.grid.to_image(self.valid[None].float())], dim=1)