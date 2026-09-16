from __future__ import annotations

import argparse
import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# CRITICAL: import cv2 and configure Qt plugin path before lerobot/torch to prevent Qt/OpenMP deadlock on Linux
import cv2

try:
    _cv2_qt = Path(cv2.__file__).resolve().parent / "qt" / "plugins"
    if _cv2_qt.exists() and "QT_PLUGIN_PATH" not in os.environ:
        os.environ["QT_PLUGIN_PATH"] = str(_cv2_qt)
except Exception:
    pass

import gymnasium as gym
import numpy as np

from lerobot.datasets import LeRobotDataset
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.feature_utils import build_dataset_frame

PREVIEW_CONTINUE = "continue"
PREVIEW_FINISH_EPISODE = "finish_episode"
PREVIEW_RERECORD_EPISODE = "rerecord_episode"
PREVIEW_STOP_SESSION = "stop_session"

if __package__ in (None, ""):
    import sys

    sys.path.append(str(Path(__file__).resolve().parent))
    from action_source import ActionSourceConfig, create_action_source
    from mujoco_xml_env import MujocoXmlEnv, MujocoXmlEnvConfig
    from sim_adapter import SimAdapterConfig, SimDatasetAdapter
    from top_camera_grid_overlay import (
        DEFAULT_GRID_COLS,
        DEFAULT_GRID_ROWS,
        DEFAULT_MARGIN_BOTTOM,
        DEFAULT_MARGIN_LEFT,
        DEFAULT_MARGIN_RIGHT,
        DEFAULT_MARGIN_TOP,
        draw_top_camera_grid,
    )
else:
    from .action_source import ActionSourceConfig, create_action_source
    from .mujoco_xml_env import MujocoXmlEnv, MujocoXmlEnvConfig
    from .sim_adapter import SimAdapterConfig, SimDatasetAdapter
    from .top_camera_grid_overlay import (
        DEFAULT_GRID_COLS,
        DEFAULT_GRID_ROWS,
        DEFAULT_MARGIN_BOTTOM,
        DEFAULT_MARGIN_LEFT,
        DEFAULT_MARGIN_RIGHT,
        DEFAULT_MARGIN_TOP,
        draw_top_camera_grid,
    )


@dataclass
class EnvCollectConfig:
    backend: str
    id: str | None
    xml: MujocoXmlEnvConfig | None
    kwargs: dict[str, Any]
    seed: int | None
    reset_each_episode_seed: bool
    reset_between_episodes: bool
    max_steps_per_episode: int | None

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "EnvCollectConfig":
        backend = str(raw.get("backend", "gym")).strip().lower()
        if backend not in {"gym", "mujoco_xml"}:
            raise ValueError(f"Unsupported env.backend='{backend}'. Use 'gym' or 'mujoco_xml'.")

        env_id = raw.get("id")
        xml_cfg: MujocoXmlEnvConfig | None = None
        if backend == "mujoco_xml":
            xml_cfg = MujocoXmlEnvConfig.from_dict(raw)
            env_id = None
        else:
            if not env_id:
                raise ValueError("env.id is required when env.backend='gym'.")

        return cls(
            backend=backend,
            id=str(env_id) if env_id else None,
            xml=xml_cfg,
            kwargs=raw.get("kwargs", {}),
            seed=raw.get("seed"),
            reset_each_episode_seed=bool(raw.get("reset_each_episode_seed", True)),
            reset_between_episodes=bool(raw.get("reset_between_episodes", True)),
            max_steps_per_episode=raw.get("max_steps_per_episode"),
        )


@dataclass
class DatasetCollectConfig:
    repo_id: str
    root: str
    single_task: str
    robot_type: str | None
    fps: int
    num_episodes: int
    episode_time_s: float
    use_videos: bool
    video_backend: str | None
    vcodec: str
    data_files_size_in_mb: int | None
    video_files_size_in_mb: int | None
    push_to_hub: bool
    private: bool
    tags: list[str] | None
    streaming_encoding: bool
    encoder_threads: int | None
    encoder_queue_maxsize: int
    image_writer_threads: int
    auto_resume_if_root_exists: bool
    cleanup_local_after_push: bool

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "DatasetCollectConfig":
        return cls(
            repo_id=str(raw["repo_id"]),
            root=str(raw["root"]),
            single_task=str(raw.get("single_task", "")),
            robot_type=(
                str(raw["robot_type"]) if raw.get("robot_type") is not None else None
            ),
            fps=int(raw.get("fps", 30)),
            num_episodes=int(raw.get("num_episodes", 50)),
            episode_time_s=float(raw.get("episode_time_s", 60.0)),
            use_videos=bool(raw.get("use_videos", True)),
            video_backend=(
                str(raw["video_backend"]) if raw.get("video_backend") is not None else None
            ),
            vcodec=str(raw.get("vcodec", "libsvtav1")),
            data_files_size_in_mb=(
                int(raw["data_files_size_in_mb"])
                if raw.get("data_files_size_in_mb") is not None
                else None
            ),
            video_files_size_in_mb=(
                int(raw["video_files_size_in_mb"])
                if raw.get("video_files_size_in_mb") is not None
                else None
            ),
            push_to_hub=bool(raw.get("push_to_hub", True)),
            private=bool(raw.get("private", False)),
            tags=raw.get("tags"),
            streaming_encoding=bool(raw.get("streaming_encoding", True)),
            encoder_threads=raw.get("encoder_threads"),
            encoder_queue_maxsize=int(raw.get("encoder_queue_maxsize", 30)),
            image_writer_threads=int(raw.get("image_writer_threads", 0)),
            auto_resume_if_root_exists=bool(raw.get("auto_resume_if_root_exists", True)),
            cleanup_local_after_push=bool(raw.get("cleanup_local_after_push", False)),
        )


@dataclass
class TopGridConfig:
    enabled: bool
    rows: int
    cols: int
    labels: bool
    margin_left: int
    margin_top: int
    margin_right: int
    margin_bottom: int

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | bool | None) -> "TopGridConfig":
        if isinstance(raw, bool):
            return cls(
                enabled=raw,
                rows=DEFAULT_GRID_ROWS,
                cols=DEFAULT_GRID_COLS,
                labels=True,
                margin_left=DEFAULT_MARGIN_LEFT,
                margin_top=DEFAULT_MARGIN_TOP,
                margin_right=DEFAULT_MARGIN_RIGHT,
                margin_bottom=DEFAULT_MARGIN_BOTTOM,
            )

        raw = raw or {}
        rows = int(raw.get("rows", DEFAULT_GRID_ROWS))
        cols = int(raw.get("cols", DEFAULT_GRID_COLS))
        if rows <= 0 or cols <= 0:
            raise ValueError("recording.top_grid rows and cols must be > 0.")

        return cls(
            enabled=bool(raw.get("enabled", False)),
            rows=rows,
            cols=cols,
            labels=bool(raw.get("labels", True)),
            margin_left=int(raw.get("margin_left", DEFAULT_MARGIN_LEFT)),
            margin_top=int(raw.get("margin_top", DEFAULT_MARGIN_TOP)),
            margin_right=int(raw.get("margin_right", DEFAULT_MARGIN_RIGHT)),
            margin_bottom=int(raw.get("margin_bottom", DEFAULT_MARGIN_BOTTOM)),
        )


@dataclass
class RecordingConfig:
    sleep_to_fps: bool
    control_fps: int | None
    prepare_time_s: float
    require_enter_before_episode: bool
    save_partial_episode_on_interrupt: bool
    display_camera_preview: bool
    preview_window_name: str
    preview_scale: float
    top_grid: TopGridConfig

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> "RecordingConfig":
        raw = raw or {}
        control_fps = raw.get("control_fps")
        return cls(
            sleep_to_fps=bool(raw.get("sleep_to_fps", True)),
            control_fps=int(control_fps) if control_fps is not None else None,
            prepare_time_s=max(float(raw.get("prepare_time_s", 0.0)), 0.0),
            require_enter_before_episode=bool(raw.get("require_enter_before_episode", False)),
            save_partial_episode_on_interrupt=bool(
                raw.get("save_partial_episode_on_interrupt", True)
            ),
            display_camera_preview=bool(raw.get("display_camera_preview", False)),
            preview_window_name=str(raw.get("preview_window_name", "sim_collect_preview")),
            preview_scale=float(raw.get("preview_scale", 1.0)),
            top_grid=TopGridConfig.from_dict(raw.get("top_grid")),
        )


@dataclass
class SimCollectConfig:
    env: EnvCollectConfig
    dataset: DatasetCollectConfig
    adapter: SimAdapterConfig
    action_source: ActionSourceConfig
    recording: RecordingConfig
    resume: bool

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "SimCollectConfig":
        return cls(
            env=EnvCollectConfig.from_dict(raw["env"]),
            dataset=DatasetCollectConfig.from_dict(raw["dataset"]),
            adapter=SimAdapterConfig.from_dict(raw.get("adapter")),
            action_source=ActionSourceConfig.from_dict(raw.get("action_source")),
            recording=RecordingConfig.from_dict(raw.get("recording")),
            resume=bool(raw.get("resume", False)),
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect LeRobot dataset from a simulation environment.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("src/sim_mujoco/config_sim_collect_real_like.json"),
        help="Path to JSON config file.",
    )
    parser.add_argument(
        "--resume",
        nargs="?",
        const=True,
        default=False,
        type=_parse_bool,
        help="Resume an existing dataset. Accepts `--resume` or `--resume=true`.",
    )
    parser.add_argument(
        "--no-push",
        action="store_true",
        help="Do not push to Hugging Face Hub after collection.",
    )
    parser.add_argument("--teleop.type", dest="teleop_type")
    parser.add_argument("--teleop.port", dest="teleop_port")
    parser.add_argument("--teleop.id", dest="teleop_id")
    parser.add_argument("--teleop.calibration_dir", dest="teleop_calibration_dir")
    parser.add_argument(
        "--teleop.use_degrees",
        dest="teleop_use_degrees",
        type=_parse_bool,
    )
    parser.add_argument("--dataset.repo_id", dest="dataset_repo_id")
    parser.add_argument("--dataset.root", dest="dataset_root")
    parser.add_argument("--dataset.single_task", dest="dataset_single_task")
    parser.add_argument("--dataset.fps", dest="dataset_fps", type=int)
    parser.add_argument("--dataset.num_episodes", dest="dataset_num_episodes", type=int)
    parser.add_argument(
        "--dataset.episode_time_s",
        dest="dataset_episode_time_s",
        type=float,
    )
    parser.add_argument(
        "--dataset.reset_time_s",
        dest="dataset_reset_time_s",
        type=float,
        help="Timed preparation period between episodes; preparation actions are not recorded.",
    )
    parser.add_argument(
        "--dataset.push_to_hub",
        dest="dataset_push_to_hub",
        type=_parse_bool,
    )
    parser.add_argument(
        "--dataset.private",
        dest="dataset_private",
        type=_parse_bool,
    )
    parser.add_argument(
        "--dataset.cleanup_local_after_push",
        dest="dataset_cleanup_local_after_push",
        type=_parse_bool,
        help="Automatically delete local dataset files after successfully pushing to Hugging Face Hub.",
    )
    parser.add_argument(
        "--recording.control_fps",
        dest="recording_control_fps",
        type=int,
    )
    parser.add_argument(
        "--recording.require_enter_before_episode",
        dest="require_enter_before_episode",
        type=_parse_bool,
    )
    parser.add_argument(
        "--env.reset_between_episodes",
        dest="reset_between_episodes",
        type=_parse_bool,
    )
    parser.add_argument("--display_data", dest="display_data", type=_parse_bool)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        force=True,
    )
    logging.getLogger().setLevel(logging.INFO)
    cfg = SimCollectConfig.from_dict(_load_json(args.config))
    _apply_cli_overrides(cfg, args)

    _collect(cfg)


def _apply_cli_overrides(cfg: SimCollectConfig, args: argparse.Namespace) -> None:
    if args.teleop_type is not None and args.teleop_type != "so101_leader":
        raise ValueError(
            f"Unsupported --teleop.type={args.teleop_type!r}; use 'so101_leader'."
        )
    if args.teleop_port is not None:
        os.environ["REAL_TELEOP_PORT"] = args.teleop_port
    if args.teleop_id is not None:
        os.environ["REAL_TELEOP_ID"] = args.teleop_id
    if args.teleop_calibration_dir is not None:
        os.environ["REAL_TELEOP_CALIBRATION_DIR"] = args.teleop_calibration_dir
    if args.teleop_use_degrees is not None:
        os.environ["REAL_TELEOP_USE_DEGREES"] = str(args.teleop_use_degrees).lower()

    if args.dataset_repo_id is not None:
        cfg.dataset.repo_id = args.dataset_repo_id
    if args.dataset_root is not None:
        cfg.dataset.root = args.dataset_root
    if args.dataset_single_task is not None:
        cfg.dataset.single_task = args.dataset_single_task
    if args.dataset_fps is not None:
        cfg.dataset.fps = args.dataset_fps
        if args.recording_control_fps is None:
            cfg.recording.control_fps = args.dataset_fps
    if args.dataset_num_episodes is not None:
        cfg.dataset.num_episodes = args.dataset_num_episodes
    if args.dataset_episode_time_s is not None:
        cfg.dataset.episode_time_s = args.dataset_episode_time_s
    if args.dataset_reset_time_s is not None:
        cfg.recording.prepare_time_s = max(args.dataset_reset_time_s, 0.0)
        cfg.recording.require_enter_before_episode = False
    if args.dataset_push_to_hub is not None:
        cfg.dataset.push_to_hub = args.dataset_push_to_hub
    if args.dataset_private is not None:
        cfg.dataset.private = args.dataset_private
    if args.dataset_cleanup_local_after_push is not None:
        cfg.dataset.cleanup_local_after_push = args.dataset_cleanup_local_after_push
    if args.recording_control_fps is not None:
        cfg.recording.control_fps = args.recording_control_fps
    if args.require_enter_before_episode is not None:
        cfg.recording.require_enter_before_episode = args.require_enter_before_episode
    if args.reset_between_episodes is not None:
        cfg.env.reset_between_episodes = args.reset_between_episodes
    if args.display_data is not None:
        cfg.recording.display_camera_preview = args.display_data

    if args.resume:
        cfg.resume = True
    if args.no_push:
        cfg.dataset.push_to_hub = False


def _collect(cfg: SimCollectConfig) -> None:
    if cfg.dataset.fps <= 0:
        raise ValueError("dataset.fps must be > 0.")
    control_fps = cfg.recording.control_fps or cfg.dataset.fps
    if control_fps <= 0:
        raise ValueError("recording.control_fps must be > 0.")
    if control_fps < cfg.dataset.fps:
        raise ValueError("recording.control_fps must be >= dataset.fps.")
    if control_fps % cfg.dataset.fps != 0:
        logging.warning(
            "control_fps=%d is not an integer multiple of dataset.fps=%d. "
            "Recorded frame cadence will be approximate.",
            control_fps,
            cfg.dataset.fps,
        )
def _collect(cfg: SimCollectConfig) -> None:
    if cfg.dataset.fps <= 0:
        raise ValueError("dataset.fps must be > 0.")
    control_fps = cfg.recording.control_fps or cfg.dataset.fps
    if control_fps <= 0:
        raise ValueError("recording.control_fps must be > 0.")
    if control_fps < cfg.dataset.fps:
        raise ValueError("recording.control_fps must be >= dataset.fps.")
    if control_fps % cfg.dataset.fps != 0:
        logging.warning(
            "control_fps=%d is not an integer multiple of dataset.fps=%d. "
            "Recorded frame cadence will be approximate.",
            control_fps,
            cfg.dataset.fps,
        )
    if cfg.dataset.num_episodes <= 0:
        raise ValueError("dataset.num_episodes must be > 0.")
    if cfg.resume and not cfg.dataset.root:
        raise ValueError("`dataset.root` is required when resume=true.")

    preview_enabled = False
    if cfg.recording.display_camera_preview:
        try:
            cv2.namedWindow(cfg.recording.preview_window_name, cv2.WINDOW_NORMAL)
            preview_enabled = True
        except Exception as e:
            logging.warning("Camera preview disabled: OpenCV is not available (%s)", e)

    env = _make_env(cfg.env)
    action_source = create_action_source(cfg.action_source, env)
    adapter = SimDatasetAdapter(cfg.adapter)
    dataset: LeRobotDataset | None = None
    collection_finished_cleanly = False
    try:
        seed = cfg.env.seed
        init_obs, init_info = env.reset(seed=seed, options={"episode_index": 0})
        action_source.reset(episode_index=0, observation=init_obs, info=init_info)
        sample_action = action_source.next_action(
            observation=init_obs,
            info=init_info,
            step_index=0,
            episode_index=0,
        )
        _validate_action_shape_against_env(sample_action, env)
        adapter.prepare(init_obs, sample_action)

        dataset_features = adapter.build_dataset_features(use_videos=cfg.dataset.use_videos)
        dataset = _create_or_resume_dataset(cfg, dataset_features)
        step_limit = _resolve_step_limit(cfg, env, control_fps=control_fps)
        control_interval = 1.0 / control_fps
        record_every_steps = max(1, round(control_fps / cfg.dataset.fps))
        obs, info = init_obs, init_info

        # In continuous (real-like) mode, reset exactly once at process startup.
        # For a resumed dataset, use the next episode index so deterministic
        # placement plans continue from the correct row.
        if not cfg.env.reset_between_episodes and dataset.num_episodes > 0:
            startup_seed = cfg.env.seed
            if cfg.env.reset_each_episode_seed and startup_seed is not None:
                startup_seed += dataset.num_episodes
            obs, info = env.reset(
                seed=startup_seed,
                options={"episode_index": dataset.num_episodes},
            )

        logging.info("Dataset root: %s", dataset.root)
        logging.info("Existing saved episodes: %d", dataset.num_episodes)
        logging.info("Dataset action/state angle unit: %s", cfg.adapter.dataset_angle_unit)
        logging.info(
            "Environment reset between episodes: %s",
            cfg.env.reset_between_episodes,
        )
        logging.info(
            "Matrix placement log: %s",
            Path(dataset.root) / "meta" / "matrix_random_placements.jsonl",
        )
        logging.info(
            "Starting collection with %d episodes "
            "(control_fps=%d, dataset_fps=%d, record_every=%d control steps)",
            cfg.dataset.num_episodes,
            control_fps,
            cfg.dataset.fps,
            record_every_steps,
        )
        if preview_enabled:
            logging.info(
                "Preview controls: Right/Enter/Space saves current episode early; "
                "Left/R/Backspace discards and re-records it; Q/Esc stops collection."
            )

        print("\n" + "=" * 70, flush=True)
        print("  SO-101 SIMULATION DATA COLLECTION (MUJOCO)", flush=True)
        print("=" * 70, flush=True)
        print(f"  * Repo ID trên Hugging Face : {cfg.dataset.repo_id}", flush=True)
        print(f"  * Tự động đẩy lên Hub       : {cfg.dataset.push_to_hub}", flush=True)
        print(f"  * Số tập cần thu (episodes) : {cfg.dataset.num_episodes}", flush=True)
        print(f"  * Số tập hiện có trên máy   : {dataset.num_episodes}", flush=True)
        print("-" * 70, flush=True)
        print("  HƯỚNG DẪN THAO TÁC:", flush=True)
        print(f"  1. Cửa sổ camera preview ('{cfg.recording.preview_window_name}') đã mở trên màn hình.", flush=True)
        print("  2. Tay Leader đang điều khiển robot trong sim (chế độ chuẩn bị).", flush=True)
        print("  3. Nhấp chuột vào cửa sổ preview rồi bấm [ENTER] hoặc [SPACE] để BẮT ĐẦU GHI.", flush=True)
        print("  4. Sau khi gắp xếp xong 3 khối hộp, bấm [ENTER] để LƯU tập.", flush=True)
        print("  5. Bấm [Q] hoặc [ESC] bất kỳ lúc nào để DỪNG và ĐẨY DỮ LIỆU LÊN HUGGING FACE!", flush=True)
        print("=" * 70 + "\n", flush=True)

        stop_requested = False
        recorded_episodes = 0
        while recorded_episodes < cfg.dataset.num_episodes:
            episode_index = dataset.num_episodes
            episode_started_with_reset = cfg.env.reset_between_episodes
            if cfg.env.reset_between_episodes:
                episode_seed = None
                if cfg.env.reset_each_episode_seed and cfg.env.seed is not None:
                    episode_seed = cfg.env.seed + episode_index
                obs, info = env.reset(
                    seed=episode_seed,
                    options={"episode_index": episode_index},
                )
            elif recorded_episodes == 0:
                # The initial reset above is required to initialize MuJoCo. No
                # further reset happens between episodes in this process.
                episode_started_with_reset = True

            episode_start_info = dict(info or {})
            action_source.reset(episode_index=episode_index, observation=obs, info=info)
            matrix_placements = episode_start_info.get("matrix_random_placements")
            if matrix_placements and episode_started_with_reset:
                logging.info(
                    "Episode %d starting matrix placement metadata: %s",
                    episode_index,
                    matrix_placements,
                )
            wait_event, obs, info = _wait_before_episode(
                cfg=cfg,
                adapter=adapter,
                action_source=action_source,
                env=env,
                observation=obs,
                info=info,
                episode_index=episode_index,
                cv2=cv2,
                preview_enabled=preview_enabled,
                control_fps=control_fps,
            )
            if wait_event == PREVIEW_STOP_SESSION:
                stop_requested = True
                logging.info("Stop requested during episode preparation. Exiting collection loop.")
                break

            done = False
            interrupted = False
            rerecord_episode = False
            step = 0
            recorded_frames = 0
            episode_start_t = time.perf_counter()
            print(
                f"\n>>> [TẬP {episode_index + 1}/{cfg.dataset.num_episodes}] "
                f"BẮT ĐẦU GHI DỮ LIỆU! (Gắp xếp xong bấm ENTER hoặc SPACE trên preview để LƯU) <<<",
                flush=True,
            )

            try:
                while step < step_limit and not done:
                    loop_start_t = time.perf_counter()

                    action = action_source.next_action(
                        observation=obs,
                        info=info,
                        step_index=step,
                        episode_index=episode_index,
                    )
                    obs_values = adapter.observation_to_values(obs)

                    if step % record_every_steps == 0:
                        action_values = adapter.action_to_values(action)
                        obs_frame = build_dataset_frame(dataset.features, obs_values, prefix=OBS_STR)
                        action_frame = build_dataset_frame(
                            dataset.features,
                            action_values,
                            prefix=ACTION,
                        )
                        frame = {**obs_frame, **action_frame, "task": cfg.dataset.single_task}
                        dataset.add_frame(frame)
                        recorded_frames += 1

                    if preview_enabled and cv2 is not None:
                        preview_event = _show_camera_preview(
                            cv2=cv2,
                            top=obs_values[cfg.adapter.top_dataset_camera_key],
                            gripper=obs_values[cfg.adapter.secondary_dataset_camera_key],
                            window_name=cfg.recording.preview_window_name,
                            scale=cfg.recording.preview_scale,
                            top_grid=cfg.recording.top_grid,
                            status_text=f"[REC {episode_index + 1}/{cfg.dataset.num_episodes}] Frame {recorded_frames} | ENTER: Save | Q: Stop",
                        )
                        if preview_event == PREVIEW_FINISH_EPISODE:
                            logging.info("Early finish requested for episode %d.", episode_index)
                            done = True
                        elif preview_event == PREVIEW_RERECORD_EPISODE:
                            logging.info("Re-record requested for episode %d.", episode_index)
                            rerecord_episode = True
                            done = True
                        elif preview_event == PREVIEW_STOP_SESSION:
                            stop_requested = True
                            done = True

                    obs, _, terminated, truncated, info = env.step(action)
                    done = done or bool(terminated or truncated)
                    step += 1

                    if cfg.recording.sleep_to_fps:
                        elapsed_s = time.perf_counter() - loop_start_t
                        sleep_s = control_interval - elapsed_s
                        if sleep_s > 0:
                            time.sleep(sleep_s)
            except KeyboardInterrupt:
                interrupted = True
                logging.info("Interrupted during episode %d.", episode_index)

            if rerecord_episode:
                dataset.clear_episode_buffer()
                if cfg.env.reset_between_episodes:
                    logging.info("Discarded episode %d. Recording it again.", episode_index)
                else:
                    logging.info(
                        "Discarded episode %d. Continuous mode preserves the current "
                        "MuJoCo state; prepare it manually before recording again.",
                        episode_index,
                    )
                continue

            if recorded_frames == 0:
                dataset.clear_episode_buffer()
                if interrupted or stop_requested:
                    logging.info("No frames recorded for interrupted episode %d.", episode_index)
                    break
                raise RuntimeError("Episode has zero recorded frames. Check fps settings and action source.")

            if interrupted and not cfg.recording.save_partial_episode_on_interrupt:
                dataset.clear_episode_buffer()
                logging.info("Discarded interrupted episode %d.", episode_index)
                break

            dataset.save_episode()
            if episode_started_with_reset:
                _append_matrix_placement_log(
                    dataset_root=Path(dataset.root),
                    episode_index=episode_index,
                    info=episode_start_info,
                )
            recorded_episodes += 1
            duration_s = time.perf_counter() - episode_start_t
            print(
                f">>> [TẬP {episode_index + 1}/{cfg.dataset.num_episodes}] "
                f"ĐÃ LƯU THÀNH CÔNG! ({recorded_frames} frames, {duration_s:.1f}s) "
                f"| Đã lưu: {dataset.num_episodes}/{cfg.dataset.num_episodes} tập <<<",
                flush=True,
            )
            logging.info(
                "Saved episode %d (%d/%d in this run), recorded_frames=%d, "
                "control_steps=%d, duration=%.2fs",
                episode_index,
                recorded_episodes,
                cfg.dataset.num_episodes,
                recorded_frames,
                step,
                duration_s,
            )
            if stop_requested:
                logging.info("Stop requested from preview window. Exiting collection loop.")
                break
            if interrupted:
                logging.info("Interrupted run saved partial episode and will finalize.")
                break
        collection_finished_cleanly = True
    finally:
        if preview_enabled and cv2 is not None:
            try:
                cv2.destroyWindow(cfg.recording.preview_window_name)
            except Exception:
                pass
        if dataset is not None:
            dataset.finalize()
            if cfg.dataset.push_to_hub and (collection_finished_cleanly or dataset.num_episodes > 0):
                print("\n" + "=" * 70, flush=True)
                print(f"  ĐANG ĐẨY DỮ LIỆU LÊN HUGGING FACE ({cfg.dataset.repo_id})...", flush=True)
                print("=" * 70, flush=True)
                logging.info("Pushing dataset to Hub: %s", cfg.dataset.repo_id)
                dataset.push_to_hub(tags=cfg.dataset.tags, private=cfg.dataset.private)
                print(f"\n>>> ĐÃ ĐẨY THÀNH CÔNG {dataset.num_episodes} TẬP LÊN HUGGING FACE! <<<\n", flush=True)
                logging.info("Push completed.")

                if cfg.dataset.cleanup_local_after_push:
                    root_path = Path(cfg.dataset.root).resolve()
                    if root_path.exists():
                        import shutil
                        shutil.rmtree(root_path, ignore_errors=True)
                        print(f">>> ĐÃ TỰ ĐỘNG XÓA DỮ LIỆU CỤC BỘ ({root_path}) - GIẢI PHÓNG HOÀN TOÀN Ổ CỨNG! <<<", flush=True)
                        print(f">>> DỮ LIỆU ĐÃ ĐƯỢC LƯU AN TOÀN TRÊN CLOUD: https://huggingface.co/datasets/{cfg.dataset.repo_id} <<<\n", flush=True)
                        logging.info("Cleaned up local dataset root: %s", root_path)
            elif cfg.dataset.push_to_hub:
                logging.warning(
                    "Hub push skipped because no episodes were recorded."
                )
            else:
                logging.info("Push skipped (`push_to_hub=false`).")
        env.close()




def _make_env(cfg: EnvCollectConfig) -> Any:
    if cfg.backend == "mujoco_xml":
        if cfg.xml is None:
            raise ValueError("env.xml is required for env.backend='mujoco_xml'.")
        return MujocoXmlEnv(cfg.xml)
    if not cfg.id:
        raise ValueError("env.id is required for env.backend='gym'.")
    return gym.make(cfg.id, **cfg.kwargs)


def _append_matrix_placement_log(
    dataset_root: Path,
    episode_index: int,
    info: dict[str, Any],
) -> None:
    matrix_placements = info.get("matrix_random_placements")
    if not matrix_placements:
        return

    log_path = dataset_root / "meta" / "matrix_random_placements.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "episode_index": episode_index,
        "matrix_random_placements": matrix_placements,
    }
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def _create_or_resume_dataset(
    cfg: SimCollectConfig,
    dataset_features: dict[str, dict],
) -> LeRobotDataset:
    should_resume = cfg.resume
    root_path = Path(cfg.dataset.root)
    info_json_path = root_path / "meta" / "info.json"
    if not should_resume and cfg.dataset.auto_resume_if_root_exists:
        if info_json_path.is_file():
            logging.info(
                "Dataset root already exists with valid meta/info.json and auto_resume_if_root_exists=true. Switching to resume mode."
            )
            should_resume = True
        elif root_path.exists():
            logging.info(
                "Dataset root '%s' exists but has no meta/info.json (empty or incomplete). Cleaning up to create a fresh dataset.",
                root_path,
            )
            import shutil

            shutil.rmtree(root_path, ignore_errors=True)

    import inspect

    extra_create_kwargs: dict[str, Any] = {}
    extra_resume_kwargs: dict[str, Any] = {}
    create_params = inspect.signature(LeRobotDataset.create).parameters
    resume_params = inspect.signature(LeRobotDataset.resume).parameters

    if "rgb_encoder" in create_params and cfg.dataset.vcodec:
        try:
            from lerobot.configs.video import RGBEncoderConfig

            extra_create_kwargs["rgb_encoder"] = RGBEncoderConfig(
                vcodec=cfg.dataset.vcodec,
                video_backend=cfg.dataset.video_backend or "pyav",
            )
        except Exception:
            pass
    elif "vcodec" in create_params and cfg.dataset.vcodec:
        extra_create_kwargs["vcodec"] = cfg.dataset.vcodec

    if "rgb_encoder" in resume_params and "rgb_encoder" in extra_create_kwargs:
        extra_resume_kwargs["rgb_encoder"] = extra_create_kwargs["rgb_encoder"]
    elif "vcodec" in resume_params and cfg.dataset.vcodec:
        extra_resume_kwargs["vcodec"] = cfg.dataset.vcodec

    if should_resume:
        logging.info("Resuming dataset: %s", cfg.dataset.repo_id)
        try:
            dataset = LeRobotDataset.resume(
                repo_id=cfg.dataset.repo_id,
                root=cfg.dataset.root,
                video_backend=cfg.dataset.video_backend,
                streaming_encoding=cfg.dataset.streaming_encoding,
                encoder_threads=cfg.dataset.encoder_threads,
                encoder_queue_maxsize=cfg.dataset.encoder_queue_maxsize,
                image_writer_threads=cfg.dataset.image_writer_threads,
                **extra_resume_kwargs,
            )
            _assert_feature_compatibility(dataset.features, dataset_features)
            if dataset.fps != cfg.dataset.fps:
                raise ValueError(
                    f"Dataset FPS mismatch on resume: existing={dataset.fps}, "
                    f"configured={cfg.dataset.fps}."
                )
            if dataset.meta.robot_type != cfg.dataset.robot_type:
                raise ValueError(
                    "Dataset robot_type mismatch on resume: "
                    f"existing={dataset.meta.robot_type!r}, "
                    f"configured={cfg.dataset.robot_type!r}."
                )
            return dataset
        except Exception as e:
            err_msg = str(e)
            if "RepositoryNotFoundError" in type(e).__name__ or "FileNotFoundError" in type(e).__name__ or "404" in err_msg or "Not Found" in err_msg:
                logging.warning(
                    "Dataset '%s' chưa tồn tại (cục bộ hoặc trên Hub) để resume. Tự động chuyển sang chế độ tạo mới (Create mode).",
                    cfg.dataset.repo_id,
                )
                if root_path.exists() and not info_json_path.is_file():
                    import shutil
                    shutil.rmtree(root_path, ignore_errors=True)
            else:
                raise e

    logging.info("Creating dataset: %s", cfg.dataset.repo_id)
    return LeRobotDataset.create(
        repo_id=cfg.dataset.repo_id,
        fps=cfg.dataset.fps,
        features=dataset_features,
        root=cfg.dataset.root,
        robot_type=cfg.dataset.robot_type,
        use_videos=cfg.dataset.use_videos,
        video_backend=cfg.dataset.video_backend,
        streaming_encoding=cfg.dataset.streaming_encoding,
        encoder_threads=cfg.dataset.encoder_threads,
        encoder_queue_maxsize=cfg.dataset.encoder_queue_maxsize,
        image_writer_threads=cfg.dataset.image_writer_threads,
        data_files_size_in_mb=cfg.dataset.data_files_size_in_mb,
        video_files_size_in_mb=cfg.dataset.video_files_size_in_mb,
        **extra_create_kwargs,
    )


def _assert_feature_compatibility(existing: dict[str, dict], expected: dict[str, dict]) -> None:
    missing = sorted(set(expected.keys()) - set(existing.keys()))
    if missing:
        raise ValueError(
            "Dataset feature keys mismatch on resume.\n"
            f"Missing in existing dataset: {missing}"
        )

    for key in expected:
        exp = expected[key]
        got = existing[key]
        if exp.get("dtype") != got.get("dtype") or tuple(exp.get("shape", ())) != tuple(
            got.get("shape", ())
        ):
            raise ValueError(
                f"Feature mismatch for '{key}' on resume. "
                f"Expected dtype/shape={exp.get('dtype')}/{tuple(exp.get('shape', ()))}, "
                f"got {got.get('dtype')}/{tuple(got.get('shape', ()))}."
            )


def _resolve_step_limit(cfg: SimCollectConfig, env: Any, control_fps: int) -> int:
    by_time = int(round(cfg.dataset.episode_time_s * control_fps))
    if by_time <= 0:
        raise ValueError("episode_time_s * fps must be > 0.")

    by_env = cfg.env.max_steps_per_episode
    if by_env is None:
        env_spec_max_steps = getattr(getattr(env, "spec", None), "max_episode_steps", None)
        if env_spec_max_steps is not None:
            by_env = int(env_spec_max_steps)

    if by_env is None:
        return by_time
    if by_env <= 0:
        raise ValueError("max_steps_per_episode must be > 0.")
    return min(by_time, by_env)


def _validate_action_shape_against_env(action: np.ndarray, env: Any) -> None:
    action_shape = tuple(action.shape)
    space_shape = tuple(getattr(getattr(env, "action_space", None), "shape", ()) or ())
    if not space_shape:
        return
    if action_shape != space_shape:
        raise ValueError(
            f"Action source shape {action_shape} does not match env.action_space.shape {space_shape}."
        )


def _wait_before_episode(
    cfg: SimCollectConfig,
    adapter: SimDatasetAdapter,
    action_source: Any,
    env: Any,
    observation: Any,
    info: dict[str, Any] | None,
    episode_index: int,
    cv2: Any,
    preview_enabled: bool,
    control_fps: int,
) -> tuple[str, Any, dict[str, Any] | None]:
    if cfg.recording.require_enter_before_episode:
        if preview_enabled and cv2 is not None:
            logging.info(
                "Preparation mode for episode %d: leader control is active but frames "
                "are not recorded. Press Enter/Space/Right in the preview to start.",
                episode_index,
            )
            print(
                f"\n>>> [CHUẨN BỊ TẬP {episode_index + 1}/{cfg.dataset.num_episodes}] "
                "Tay leader đang điều khiển robot trong sim. "
                "Nhấp chuột vào cửa sổ Preview và bấm [ENTER] hoặc [SPACE] để BẮT ĐẦU GHI! <<<",
                flush=True,
            )
            preparation_step = 0
            control_interval = 1.0 / control_fps
            while True:
                loop_start_t = time.perf_counter()
                action = action_source.next_action(
                    observation=observation,
                    info=info,
                    step_index=preparation_step,
                    episode_index=episode_index,
                )
                observation, _, terminated, truncated, info = env.step(action)
                if terminated or truncated:
                    logging.warning(
                        "Environment ended during preparation for episode %d.",
                        episode_index,
                    )
                obs_values = adapter.observation_to_values(observation)
                preview_event = _show_camera_preview(
                    cv2=cv2,
                    top=obs_values[cfg.adapter.top_dataset_camera_key],
                    gripper=obs_values[cfg.adapter.secondary_dataset_camera_key],
                    window_name=cfg.recording.preview_window_name,
                    scale=cfg.recording.preview_scale,
                    top_grid=cfg.recording.top_grid,
                    status_text=f"[CHUAN BI {episode_index + 1}/{cfg.dataset.num_episodes}] Bam ENTER de BAT DAU GHI",
                )
                if preview_event == PREVIEW_FINISH_EPISODE:
                    return PREVIEW_CONTINUE, observation, info
                if preview_event == PREVIEW_STOP_SESSION:
                    return PREVIEW_STOP_SESSION, observation, info
                preparation_step += 1
                if cfg.recording.sleep_to_fps:
                    elapsed_s = time.perf_counter() - loop_start_t
                    sleep_s = control_interval - elapsed_s
                    if sleep_s > 0:
                        time.sleep(sleep_s)
        input(f"Prepare episode {episode_index}, then press ENTER to start recording...")
        return PREVIEW_CONTINUE, observation, info

    wait_s = cfg.recording.prepare_time_s
    if wait_s <= 0:
        return PREVIEW_CONTINUE, observation, info

    logging.info("Prepare episode %d. Recording starts in %.1fs.", episode_index, wait_s)
    deadline = time.perf_counter() + wait_s
    last_logged_second: int | None = None
    preparation_step = 0
    control_interval = 1.0 / control_fps

    while True:
        loop_start_t = time.perf_counter()
        remaining_s = deadline - time.perf_counter()
        if remaining_s <= 0:
            return PREVIEW_CONTINUE, observation, info

        action = action_source.next_action(
            observation=observation,
            info=info,
            step_index=preparation_step,
            episode_index=episode_index,
        )
        observation, _, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            logging.warning(
                "Environment ended during timed preparation for episode %d.",
                episode_index,
            )
        preparation_step += 1

        remaining_second = int(remaining_s) + 1
        if remaining_second != last_logged_second:
            logging.info("Episode %d starts in %ds.", episode_index, remaining_second)
            last_logged_second = remaining_second

        if preview_enabled and cv2 is not None:
            obs_values = adapter.observation_to_values(observation)
            preview_event = _show_camera_preview(
                cv2=cv2,
                top=obs_values[cfg.adapter.top_dataset_camera_key],
                gripper=obs_values[cfg.adapter.secondary_dataset_camera_key],
                window_name=cfg.recording.preview_window_name,
                scale=cfg.recording.preview_scale,
                top_grid=cfg.recording.top_grid,
                status_text=f"[BAT DAU GHI TRONG {remaining_second}s]",
            )
            if preview_event == PREVIEW_FINISH_EPISODE:
                return PREVIEW_CONTINUE, observation, info
            if preview_event == PREVIEW_STOP_SESSION:
                return PREVIEW_STOP_SESSION, observation, info

        if cfg.recording.sleep_to_fps:
            elapsed_s = time.perf_counter() - loop_start_t
            sleep_s = min(max(control_interval - elapsed_s, 0.0), remaining_s)
            if sleep_s > 0:
                time.sleep(sleep_s)


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(
        f"Expected a boolean value, got {value!r}. Use true or false."
    )


def _show_camera_preview(
    cv2: Any,
    top: np.ndarray,
    gripper: np.ndarray,
    window_name: str,
    scale: float,
    top_grid: TopGridConfig,
    status_text: str = "",
) -> str:
    top_bgr = cv2.cvtColor(top, cv2.COLOR_RGB2BGR)
    if top_grid.enabled:
        draw_top_camera_grid(
            cv2=cv2,
            image=top_bgr,
            rows=top_grid.rows,
            cols=top_grid.cols,
            labels=top_grid.labels,
            margin_left=top_grid.margin_left,
            margin_top=top_grid.margin_top,
            margin_right=top_grid.margin_right,
            margin_bottom=top_grid.margin_bottom,
        )
    gripper_bgr = cv2.cvtColor(gripper, cv2.COLOR_RGB2BGR)
    frame = np.concatenate([top_bgr, gripper_bgr], axis=1)

    scale = max(float(scale), 0.1)
    if abs(scale - 1.0) > 1e-6:
        new_w = int(frame.shape[1] * scale)
        new_h = int(frame.shape[0] * scale)
        frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

    cv2.putText(frame, "top", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(
        frame,
        "gripper",
        (frame.shape[1] // 2 + 10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2,
    )
    if status_text:
        text_color = (0, 255, 0) if "REC" in status_text else (0, 215, 255)
        cv2.putText(
            frame,
            status_text,
            (100, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            status_text,
            (100, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            text_color,
            1,
            cv2.LINE_AA,
        )
    help_text = "Right/Enter/Space: save early | Left/R/Backspace: redo | Q/Esc: stop"
    cv2.putText(
        frame,
        help_text,
        (10, max(frame.shape[0] - 12, 24)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (0, 0, 0),
        3,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        help_text,
        (10, max(frame.shape[0] - 12, 24)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (0, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.imshow(window_name, frame)
    wait_key = getattr(cv2, "waitKeyEx", cv2.waitKey)
    key = wait_key(1)
    return _preview_key_to_event(key)


def _preview_key_to_event(key: int) -> str:
    if key < 0:
        return PREVIEW_CONTINUE

    if key in (27, ord("q"), ord("Q")):
        return PREVIEW_STOP_SESSION

    # OpenCV waitKeyEx key codes: Windows arrows use 0x25/0x27 << 16; X11 uses 65361/65363.
    if key in (13, 10, 32, 2555904, 65363):
        return PREVIEW_FINISH_EPISODE

    if key in (8, 127, ord("r"), ord("R"), 2424832, 65361):
        return PREVIEW_RERECORD_EPISODE

    return PREVIEW_CONTINUE


if __name__ == "__main__":
    main()
