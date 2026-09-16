from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np

if __package__ in (None, ""):
    import sys

    sys.path.append(str(Path(__file__).resolve().parent))
    from mujoco_xml_env import MujocoXmlEnv, MujocoXmlEnvConfig
else:
    from .mujoco_xml_env import MujocoXmlEnv, MujocoXmlEnvConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe a Gymnasium env to discover config values.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--env-id", help="Gymnasium env id, e.g. YourMujocoEnv-v0")
    source.add_argument(
        "--config",
        type=Path,
        help="Collector JSON config containing an env.backend='mujoco_xml' section.",
    )
    parser.add_argument(
        "--env-kwargs",
        default="{}",
        help="JSON object with kwargs passed to gym.make, e.g. '{\"render_mode\":\"rgb_array\"}'",
    )
    parser.add_argument(
        "--import-module",
        action="append",
        default=[],
        help="Optional module import before gym.make (use for custom env registration).",
    )
    parser.add_argument("--seed", type=int, default=0, help="Seed passed to env.reset().")
    args = parser.parse_args()

    for module_name in args.import_module:
        importlib.import_module(module_name)

    if args.config is not None:
        with args.config.open("r", encoding="utf-8") as f:
            raw_config = json.load(f)
        env_config = raw_config["env"]
        if str(env_config.get("backend", "")).strip().lower() != "mujoco_xml":
            raise ValueError("--config currently supports env.backend='mujoco_xml' only.")
        env = MujocoXmlEnv(MujocoXmlEnvConfig.from_dict(env_config))
        env_label = f"mujoco_xml ({env_config['xml_path']})"
    else:
        env_kwargs = json.loads(args.env_kwargs)
        env = gym.make(args.env_id, **env_kwargs)
        env_label = str(args.env_id)

    try:
        obs, info = env.reset(seed=args.seed)
        action_sample = env.action_space.sample()

        print("=== ENV INFO ===")
        print(f"env.id: {env_label}")
        print(f"env.metadata.render_fps: {env.metadata.get('render_fps', '<missing>')}")
        spec_max_steps = getattr(getattr(env, "spec", None), "max_episode_steps", None)
        print(f"env.spec.max_episode_steps: {spec_max_steps}")
        print(f"action_space: {env.action_space}")
        if hasattr(env.action_space, "shape"):
            print(f"action_space.shape: {tuple(env.action_space.shape)}")
        if hasattr(env.action_space, "low") and hasattr(env.action_space, "high"):
            low = np.asarray(env.action_space.low).reshape(-1)
            high = np.asarray(env.action_space.high).reshape(-1)
            print(f"action_space.low(min,max): ({float(low.min())}, {float(low.max())})")
            print(f"action_space.high(min,max): ({float(high.min())}, {float(high.max())})")

        print("\n=== OBS KEYS (from reset) ===")
        flat = _flatten(obs)
        for key, value in flat.items():
            _print_leaf(key, value)

        state_candidates = []
        image_candidates = []
        for key, value in flat.items():
            arr = _as_array(value)
            if arr is None:
                continue
            if arr.ndim == 1:
                state_candidates.append((key, arr.shape[0]))
            elif arr.ndim in (2, 3):
                h, w = _infer_hw(arr)
                image_candidates.append((key, (h, w), tuple(arr.shape), str(arr.dtype)))

        print("\n=== CANDIDATES ===")
        if state_candidates:
            print("State key candidates:")
            for key, dim in state_candidates:
                print(f"  - {key} (dim={dim})")
        else:
            print("State key candidates: none detected")

        if image_candidates:
            print("Image key candidates:")
            for key, hw, shape, dtype in image_candidates:
                print(f"  - {key} (shape={shape}, HxW={hw}, dtype={dtype})")
        else:
            print("Image key candidates: none detected")

        print("\n=== ONE STEP SANITY ===")
        next_obs, reward, terminated, truncated, _ = env.step(action_sample)
        print(
            f"step(sampled_action): reward={float(reward):.4f}, "
            f"terminated={bool(terminated)}, truncated={bool(truncated)}"
        )
        print(f"next_obs_type: {type(next_obs).__name__}")
    finally:
        env.close()


def _flatten(d: Any, prefix: str = "") -> dict[str, Any]:
    if not isinstance(d, dict):
        return {prefix or "<root>": d}
    out: dict[str, Any] = {}
    for key, value in d.items():
        full = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            out.update(_flatten(value, full))
        else:
            out[full] = value
    return out


def _as_array(value: Any) -> np.ndarray | None:
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, (list, tuple)):
        arr = np.asarray(value)
        if arr.dtype != object:
            return arr
    return None


def _infer_hw(arr: np.ndarray) -> tuple[int, int]:
    if arr.ndim == 2:
        return int(arr.shape[0]), int(arr.shape[1])
    if arr.ndim == 3:
        if arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
            return int(arr.shape[1]), int(arr.shape[2])
        return int(arr.shape[0]), int(arr.shape[1])
    raise ValueError(f"Unsupported image ndim: {arr.ndim}")


def _print_leaf(key: str, value: Any) -> None:
    arr = _as_array(value)
    if arr is not None:
        print(f"{key}: ndarray shape={tuple(arr.shape)} dtype={arr.dtype}")
    else:
        print(f"{key}: {type(value).__name__}")


if __name__ == "__main__":
    main()

