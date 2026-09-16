from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from lerobot.datasets import LeRobotDataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a local LeRobot dataset before Hub upload.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("src/sim_mujoco/config_sim_collect_real_like.json"),
        help="Path to JSON config file used for collection.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=200,
        help="Max number of frames to sample for deep checks.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    cfg = _load_json(args.config)
    failures = validate_dataset(cfg, max_frames=max(1, args.max_frames))

    print("\n=== PRE-UPLOAD VALIDATION ===")
    if not failures:
        print("PASS")
        return

    print("FAIL")
    for issue in failures:
        print(f"- {issue}")
    raise SystemExit(1)


def validate_dataset(config: dict[str, Any], max_frames: int) -> list[str]:
    failures: list[str] = []
    dataset_cfg = config["dataset"]
    adapter_cfg = config.get("adapter", {})

    repo_id = dataset_cfg["repo_id"]
    root = dataset_cfg["root"]
    expected_fps = int(dataset_cfg.get("fps", 30))
    expected_robot_type = dataset_cfg.get("robot_type")
    expected_action_dim = None
    expected_state_dim = None
    expected_hw = adapter_cfg.get("expected_image_hw")
    top_camera_key = str(adapter_cfg.get("top_dataset_camera_key", "top"))
    secondary_camera_key = str(
        adapter_cfg.get("secondary_dataset_camera_key", "gripper")
    )
    if adapter_cfg.get("action_names") is not None:
        expected_action_dim = len(adapter_cfg["action_names"])
    if adapter_cfg.get("state_names") is not None:
        expected_state_dim = len(adapter_cfg["state_names"])

    ds = LeRobotDataset(repo_id=repo_id, root=root, download_videos=False)

    required_feature_keys = [
        "action",
        "observation.state",
        f"observation.images.{top_camera_key}",
        f"observation.images.{secondary_camera_key}",
    ]
    for key in required_feature_keys:
        if key not in ds.features:
            failures.append(f"Missing required feature key: '{key}'")

    if ds.meta.fps != expected_fps:
        failures.append(f"FPS mismatch: expected {expected_fps}, got {ds.meta.fps}")
    if ds.meta.robot_type != expected_robot_type:
        failures.append(
            "robot_type mismatch: "
            f"expected {expected_robot_type!r}, got {ds.meta.robot_type!r}"
        )

    if ds.num_episodes <= 0:
        failures.append("Dataset has zero episodes.")
        return failures
    if ds.num_frames <= 0:
        failures.append("Dataset has zero frames.")
        return failures

    action_feature = ds.features.get("action")
    state_feature = ds.features.get("observation.state")
    if action_feature is not None:
        action_dim = int(action_feature["shape"][0])
        if expected_action_dim is not None and action_dim != expected_action_dim:
            failures.append(
                f"Action dim mismatch: expected {expected_action_dim}, found {action_dim} in dataset."
            )
    if state_feature is not None:
        state_dim = int(state_feature["shape"][0])
        if expected_state_dim is not None and state_dim != expected_state_dim:
            failures.append(
                f"State dim mismatch: expected {expected_state_dim}, found {state_dim} in dataset."
            )

    episode_rows = ds.meta.episodes
    if episode_rows is None or len(episode_rows) == 0:
        failures.append("Episode metadata is empty.")
    else:
        for idx in range(len(episode_rows)):
            row = episode_rows[idx]
            length = int(row.get("length", 0))
            if length <= 0:
                failures.append(f"Episode {idx} has non-positive length: {length}")

    frame_count = min(max_frames, ds.num_frames)
    sampled_indices = np.linspace(0, ds.num_frames - 1, num=frame_count, dtype=int)

    for idx in sampled_indices:
        raw = ds.get_raw_item(int(idx))
        action = np.asarray(raw.get("action"), dtype=np.float32)
        state = np.asarray(raw.get("observation.state"), dtype=np.float32)

        if action.ndim != 1 or not np.isfinite(action).all():
            failures.append(f"Invalid action at frame {idx} (ndim={action.ndim}, finite={np.isfinite(action).all()})")
            break
        if state.ndim != 1 or not np.isfinite(state).all():
            failures.append(f"Invalid state at frame {idx} (ndim={state.ndim}, finite={np.isfinite(state).all()})")
            break

        item = ds[int(idx)]
        for camera_key in (
            f"observation.images.{top_camera_key}",
            f"observation.images.{secondary_camera_key}",
        ):
            if camera_key not in item:
                failures.append(f"Missing camera key '{camera_key}' at frame {idx}.")
                continue
            image = item[camera_key]
            if image.ndim != 3:
                failures.append(f"Camera '{camera_key}' at frame {idx} is not 3D tensor.")
                continue
            if not np.isfinite(image.numpy()).all():
                failures.append(f"Camera '{camera_key}' has NaN/Inf at frame {idx}.")
                continue
            if expected_hw is not None:
                h, w = int(expected_hw[0]), int(expected_hw[1])
                tensor_h, tensor_w = int(image.shape[1]), int(image.shape[2])
                if (tensor_h, tensor_w) != (h, w):
                    failures.append(
                        f"Camera '{camera_key}' shape mismatch at frame {idx}: "
                        f"expected {(h, w)} got {(tensor_h, tensor_w)}"
                    )

    return failures


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    main()
