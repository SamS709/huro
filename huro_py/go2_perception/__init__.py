"""Isaac-free perception stack for the Go2 navigation task.

Everything here depends on torch only (plus ``rsl_rl``/``tensordict`` for the odometry GRU), so the same code runs in
the simulator and on the robot. Submodules are imported lazily: ``from go2_perception import HeightGrid`` does NOT
import the odometry module, so a locomotion-only deployment does not need ``rsl_rl`` / ``tensordict``.

    LocomotionController  - frozen low-level policy (TorchScript) + its observation/action bookkeeping
    HeightGrid            - ray hits (lidar frame) -> 2D height grid (policy convention)
    OdometryEstimator     - GRU velocity/attitude network + dead-reckoning + pose histories
    MapPerception         - rolling Kalman elevation map + scan-to-map matching + ego-crop denoiser
"""

import importlib

_EXPORTS = {
    "HeightGrid": ".height_grid", "rays_in_lidar_frame": ".height_grid",
    "LocomotionController": ".locomotion", "build_loc_obs": ".locomotion", "build_proprio": ".locomotion",
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    if name in _EXPORTS:
        value = getattr(importlib.import_module(_EXPORTS[name], __name__), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(globals()) + __all__)