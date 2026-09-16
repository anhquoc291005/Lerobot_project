from __future__ import annotations

import atexit
import contextlib
import os
from pathlib import Path
from typing import Any

import numpy as np

from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig

_LEADER: SO101Leader | None = None

DEFAULT_KEYS = [
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
]

GRIPPER_SIM_MIN_DEG = -13.0
GRIPPER_SIM_MAX_DEG = 100.0
GRIPPER_LEADER_RANGE = 100.0


def real_leader_action(
    observation: Any,
    info: dict[str, Any] | None,
    step_index: int,
    episode_index: int,
    env: Any,
) -> np.ndarray:
    _ = observation, info, step_index, episode_index
    leader = _ensure_leader()
    raw = leader.get_action()

    keys = _get_keys()
    values = np.array([float(raw[k]) for k in keys], dtype=np.float32)
    values = _apply_affine(values)

    expected_dim = int(np.prod(getattr(env.action_space, "shape", values.shape)))
    if values.shape[0] < expected_dim:
        raise ValueError(
            f"REAL_TELEOP_KEYS has {values.shape[0]} entries, but env action dim is {expected_dim}."
        )
    values = values[:expected_dim]

    if _as_bool(os.getenv("REAL_TELEOP_CLIP", "true")):
        low = np.asarray(env.action_space.low, dtype=np.float32).reshape(-1)[:expected_dim]
        high = np.asarray(env.action_space.high, dtype=np.float32).reshape(-1)[:expected_dim]
        values = np.clip(values, low, high)

    if not np.isfinite(values).all():
        raise ValueError("Teleop action contains NaN or Inf after mapping.")
    return values


def close_real_leader() -> None:
    global _LEADER
    if _LEADER is not None and _LEADER.is_connected:
        with contextlib.suppress(Exception):
            _LEADER.disconnect()
    _LEADER = None


def _ensure_leader() -> SO101Leader:
    global _LEADER
    if _LEADER is not None and _LEADER.is_connected:
        return _LEADER

    port = os.getenv("REAL_TELEOP_PORT")
    if not port:
        raise ValueError("Set REAL_TELEOP_PORT to your leader serial port (e.g. /dev/ttyACM1).")
    teleop_id = os.getenv("REAL_TELEOP_ID", "my_so101_leader")
    use_degrees = _as_bool(os.getenv("REAL_TELEOP_USE_DEGREES", "true"))
    calibration_dir = Path(
        os.getenv("REAL_TELEOP_CALIBRATION_DIR", Path(__file__).resolve().parents[2] / "calibration")
    )

    cfg = SO101LeaderConfig(
        port=port,
        id=teleop_id,
        use_degrees=use_degrees,
        calibration_dir=calibration_dir,
    )
    _LEADER = SO101Leader(cfg)
    try:
        _LEADER.connect()
    except Exception:
        _LEADER = None
        raise
    return _LEADER


def _get_keys() -> list[str]:
    keys = os.getenv("REAL_TELEOP_KEYS")
    if not keys:
        return DEFAULT_KEYS
    parsed = [k.strip() for k in keys.split(",") if k.strip()]
    if not parsed:
        raise ValueError("REAL_TELEOP_KEYS is set but empty.")
    return parsed


def _apply_affine(values: np.ndarray) -> np.ndarray:
    default_scale = _default_scale(values.shape[0])
    default_offset = _default_offset(values.shape[0])
    scale = _parse_scalar_or_vec("REAL_TELEOP_SCALE", values.shape[0], default=default_scale)
    offset = _parse_scalar_or_vec("REAL_TELEOP_OFFSET", values.shape[0], default=default_offset)
    return values * scale + offset


def _default_scale(dim: int) -> np.ndarray:
    base = np.pi / 180.0 if _as_bool(os.getenv("REAL_TELEOP_USE_DEGREES", "true")) else 1.0
    scale = np.full((dim,), float(base), dtype=np.float32)
    keys = _get_keys()[:dim]
    if "gripper.pos" in keys:
        gripper_idx = keys.index("gripper.pos")
        scale[gripper_idx] = np.deg2rad((GRIPPER_SIM_MAX_DEG - GRIPPER_SIM_MIN_DEG) / GRIPPER_LEADER_RANGE)
    return scale


def _default_offset(dim: int) -> np.ndarray:
    offset = np.zeros((dim,), dtype=np.float32)
    keys = _get_keys()[:dim]
    if "gripper.pos" in keys:
        offset[keys.index("gripper.pos")] = np.deg2rad(GRIPPER_SIM_MIN_DEG)
    return offset


def _parse_scalar_or_vec(env_name: str, dim: int, default: float | np.ndarray) -> np.ndarray:
    raw = os.getenv(env_name)
    if raw is None or not raw.strip():
        if np.isscalar(default):
            return np.full((dim,), float(default), dtype=np.float32)
        default_vec = np.asarray(default, dtype=np.float32)
        if default_vec.shape != (dim,):
            raise ValueError(f"Default {env_name} has shape {default_vec.shape}, expected ({dim},).")
        return default_vec

    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if len(parts) == 1:
        return np.full((dim,), float(parts[0]), dtype=np.float32)
    if len(parts) != dim:
        raise ValueError(
            f"{env_name} length mismatch: expected 1 or {dim} values, got {len(parts)}."
        )
    return np.asarray([float(p) for p in parts], dtype=np.float32)


def _as_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


atexit.register(close_real_leader)
