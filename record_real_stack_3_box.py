#!/usr/bin/env python3
"""Record real SO101 demonstrations and append them to one guarded Hub dataset."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from filelock import FileLock
from huggingface_hub import HfApi, hf_hub_download, snapshot_download

PROJECT_DIR = Path(__file__).resolve().parent
REPO_ID = "anhquoc29/stack_3_box_white_green_blue_simulation"
TASK = (
    "Stack the three blocks in this bottom-to-top order: white, green, blue. "
    "Place the white block at the target, stack the green block on the white block, "
    "then stack the blue block on the green block."
)
CAMERAS = {"top": "/dev/video4", "left": "/dev/video2"}
JOINTS = [
    "shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos",
    "wrist_flex.pos", "wrist_roll.pos", "gripper.pos",
]
DEFAULT_ROOT = PROJECT_DIR / "datasets/stack_3_box_white_green_blue_real"


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def immutable_hashes(root):
    """Existing data, video and episode metadata shards must never be rewritten."""
    result = {}
    for pattern in ["data/**/*.parquet", "videos/**/*.mp4", "meta/episodes/**/*.parquet"]:
        for path in sorted(Path(root).glob(pattern)):
            with path.open("rb") as stream:
                result[path.relative_to(root).as_posix()] = hashlib.file_digest(stream, "sha256").hexdigest()
    return result


def validate_schema(info):
    features = info["features"]
    camera_keys = {key for key in features if key.startswith("observation.images.")}
    expected = {f"observation.images.{name}" for name in CAMERAS}
    if camera_keys != expected or info["fps"] != 30 or info.get("robot_type") != "so_follower":
        raise RuntimeError("Dataset schema differs from this real setup: need top/left, 30 fps, so_follower.")
    for key in camera_keys:
        if features[key]["dtype"] != "video" or list(features[key]["shape"]) != [480, 640, 3]:
            raise RuntimeError(f"Incompatible camera format: {key}")
    for key in ["observation.state", "action"]:
        if features[key].get("names") != JOINTS or list(features[key]["shape"]) != [6]:
            raise RuntimeError(f"Incompatible joint features: {key}")
    if info.get("codebase_version") != "v3.0":
        raise RuntimeError("This recorder requires LeRobot dataset format v3.0.")


def inspect_hub(api):
    repo = api.dataset_info(REPO_ID, revision="main")
    info_path = hf_hub_download(REPO_ID, "meta/info.json", repo_type="dataset", revision=repo.sha)
    info = read_json(info_path)
    if info["total_episodes"]:
        validate_schema(info)
    elif info["total_frames"] or any(
        item.rfilename.startswith(("data/", "videos/")) for item in repo.siblings
    ):
        raise RuntimeError("Hub has data files but declares zero episodes; refusing to initialize over them.")
    return repo.sha, info


def state_path(root):
    return root.parent / f".{root.name}.hub_state.json"


def prepare_local(api, root):
    """Materialize old episodes before resuming; never replace an unknown local folder."""
    revision, remote = inspect_hub(api)
    marker = state_path(root)
    if marker.exists():
        state = read_json(marker)
        if state["repo_id"] != REPO_ID:
            raise RuntimeError("Local sync state belongs to a different Hub repository.")
        if state.get("local_pruned"):
            if root.exists():
                raise RuntimeError(
                    "A previous cleanup or restore was interrupted. Remaining local files are preserved; "
                    "use another --root to download the verified Hub copy."
                )
            if remote["total_episodes"] < state["synced_episodes"] or (
                remote["total_frames"] < state.get("synced_frames", 0)
            ):
                raise RuntimeError(
                    "Hub now has fewer episodes/frames than the verified upload; restore stopped."
                )
            # The previous session was verified on Hub and pruned. Start again from
            # the current main revision, including episodes added by another machine.
            return download_existing_dataset(root, revision, remote)
        if state["base_sha"] != revision:
            raise RuntimeError(
                "Hub changed since the local copy was synchronized. Local data is preserved. "
                "Use a different --root to download the latest dataset; do not upload this stale copy."
            )
        if not (root / "meta/info.json").exists():
            if state["synced_episodes"] or root.exists():
                raise RuntimeError(
                    "Local metadata is missing. Existing files are preserved; use another --root."
                )
            return state, False
        info = read_json(root / "meta/info.json")
        validate_schema(info)
        if info["total_episodes"] < remote["total_episodes"]:
            raise RuntimeError("Local dataset is older than Hub. Refusing to overwrite newer episodes.")
        verify_old_shards(root, state)
        return state, True
    if root.exists():
        raise RuntimeError(
            f"Unmanaged local directory exists: {root}. Use a new --root; no files were changed."
        )
    return download_existing_dataset(root, revision, remote)


def download_existing_dataset(root, revision, remote):
    if remote["total_episodes"]:
        print(f"Downloading all {remote['total_episodes']} old episodes before appending...", flush=True)
        snapshot_download(REPO_ID, repo_type="dataset", revision=revision, local_dir=root)
        validate_saved_dataset(root)
    state = {
        "repo_id": REPO_ID,
        "base_sha": revision,
        "synced_episodes": remote["total_episodes"],
        "immutable_hashes": immutable_hashes(root),
    }
    write_json(state_path(root), state)
    return state, bool(remote["total_episodes"])


def verify_old_shards(root, state):
    actual = immutable_hashes(root)
    for name, expected in state["immutable_hashes"].items():
        if actual.get(name) != expected:
            raise RuntimeError(
                f"An old shard is missing or changed: {name}. Upload stopped; Hub is unchanged."
            )


def validate_saved_dataset(root):
    """Check full episode/frame coverage and referenced files before publishing."""
    import pyarrow.parquet as pq

    info = read_json(root / "meta/info.json")
    validate_schema(info)
    frames = sum(pq.ParquetFile(path).metadata.num_rows for path in root.glob("data/**/*.parquet"))
    rows = []
    for path in root.glob("meta/episodes/**/*.parquet"):
        rows.extend(pq.ParquetFile(path).read().to_pylist())
    rows.sort(key=lambda row: row["episode_index"])
    if [row["episode_index"] for row in rows] != list(range(info["total_episodes"])):
        raise RuntimeError("Episode metadata is incomplete or duplicated. Upload stopped.")
    if frames != info["total_frames"] or sum(row["length"] for row in rows) != frames:
        raise RuntimeError("Saved frame counts do not match metadata. Upload stopped.")
    offset = 0
    for row in rows:
        if row["dataset_from_index"] != offset or row["dataset_to_index"] != offset + row["length"]:
            raise RuntimeError("Episode frame ranges are not contiguous. Upload stopped.")
        offset += row["length"]
        data_path = root / info["data_path"].format(
            chunk_index=row["data/chunk_index"], file_index=row["data/file_index"]
        )
        if not data_path.is_file():
            raise RuntimeError(f"Missing data shard: {data_path}")
        for camera in CAMERAS:
            key = f"observation.images.{camera}"
            video = root / info["video_path"].format(
                video_key=key,
                chunk_index=row[f"videos/{key}/chunk_index"],
                file_index=row[f"videos/{key}/file_index"],
            )
            if not video.is_file():
                raise RuntimeError(f"Missing episode video: {video}")
    return info


def write_dataset_card(root, info):
    (root / "README.md").write_text(
        "---\nlicense: apache-2.0\ntags:\n- LeRobot\n- robotics\n- so101\n- real\n- stack-3-blocks\n---\n\n"
        "# Real SO101: stack white, green, blue\n\n"
        "Real robot demonstrations collected with LeRobot. The repository name is retained as requested; "
        "these recorded episodes use real cameras and robot hardware.\n\n"
        f"Task: {TASK}\n\n"
        "Cameras: `observation.images.top` (overhead), `observation.images.left` (side), "
        "640 x 480 RGB at 30 fps. The five arm joints use degrees; gripper opening uses 0-100.\n\n"
        "The green workspace border is a separate live-view annotation and is absent from dataset videos.\n\n"
        f"Episodes: {info['total_episodes']}. Frames: {info['total_frames']}. "
        "Dataset format: LeRobot v3.0.\n\n"
        "New recording sessions append episode indices to the existing dataset. "
        "Existing data, video and episode metadata shards are preserved.\n"
    )


def update_version_tag(api, revision):
    """Move LeRobot's format tag to the new commit, as its standard uploader does."""
    refs = api.list_repo_refs(REPO_ID, repo_type="dataset")
    tag = next((item for item in refs.tags if item.name == "v3.0"), None)
    if tag and tag.target_commit == revision:
        return
    if api.dataset_info(REPO_ID, revision="main").sha != revision:
        raise RuntimeError("Hub advanced after upload; leaving the format tag unchanged.")
    if tag:
        api.delete_tag(REPO_ID, tag="v3.0", repo_type="dataset")
    api.create_tag(REPO_ID, tag="v3.0", revision=revision, repo_type="dataset")


def upload_pending(api, root, state):
    info = validate_saved_dataset(root)
    verify_old_shards(root, state)
    revision, remote = inspect_hub(api)
    if revision != state["base_sha"] or remote["total_episodes"] != state["synced_episodes"]:
        raise RuntimeError(
            "Hub changed during recording. Local episodes are safe; upload stopped to avoid overwriting."
        )
    count = info["total_episodes"]
    if count < state["synced_episodes"]:
        raise RuntimeError("Local episode count decreased. Upload stopped.")
    if count == state["synced_episodes"]:
        if count:
            update_version_tag(api, revision)
        print("No pending episodes to upload.", flush=True)
        return state
    write_dataset_card(root, info)
    print(f"Uploading episodes {state['synced_episodes']}..{count - 1} to {REPO_ID}...", flush=True)
    commit = api.upload_folder(
        repo_id=REPO_ID,
        repo_type="dataset",
        revision="main",
        parent_commit=state["base_sha"],
        folder_path=root,
        allow_patterns=["meta/**", "data/**", "videos/**", "README.md"],
        delete_patterns=None,
        commit_message=f"Append real demonstrations: episodes {state['synced_episodes']} to {count - 1}",
    )
    state = {
        "repo_id": REPO_ID,
        "base_sha": commit.oid,
        "synced_episodes": count,
        "immutable_hashes": immutable_hashes(root),
    }
    write_json(state_path(root), state)
    update_version_tag(api, commit.oid)
    print(f"Uploaded successfully: {count} total episodes. {commit.commit_url}", flush=True)
    return state


def cleanup_files(root):
    """List files that must match Hub; ignore only generated Hub cache records."""
    if root.is_symlink() or root.resolve() in {Path("/"), Path.home(), PROJECT_DIR, PROJECT_DIR / "datasets"}:
        raise RuntimeError("Refusing to clean a symlink or a protected directory.")
    files = {}
    for path in root.rglob("*"):
        if path.is_symlink():
            raise RuntimeError(f"Refusing to clean a dataset containing symlinks: {path}")
        if path.is_file():
            name = path.relative_to(root).as_posix()
            if not name.startswith(".cache/huggingface/"):
                files[name] = path
    return files


def file_checksums(path):
    """Compute both Git blob SHA-1 and LFS SHA-256 from the same file stream."""
    size = path.stat().st_size
    git_hash = hashlib.sha1(f"blob {size}\0".encode(), usedforsecurity=False)
    lfs_hash = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            git_hash.update(block)
            lfs_hash.update(block)
    return git_hash.hexdigest(), lfs_hash.hexdigest()


def cleanup_local_after_upload(api, root, state, *, validate_fn=validate_saved_dataset):
    """Prune only after server metadata and checksums confirm every local data file."""
    info = validate_fn(root)
    if not info["total_episodes"]:
        print("No recorded episodes; keeping the local directory.", flush=True)
        return state
    if state["repo_id"] != REPO_ID or info["total_episodes"] != state["synced_episodes"]:
        raise RuntimeError("Local episodes are not fully synchronized; cleanup cancelled.")
    revision, remote = inspect_hub(api)
    if revision != state["base_sha"] or (
        remote["total_episodes"], remote["total_frames"]
    ) != (info["total_episodes"], info["total_frames"]):
        raise RuntimeError("Hub revision or episode/frame counts differ; keeping all local data.")
    files = cleanup_files(root)
    fingerprints = {name: (path.stat().st_size, path.stat().st_mtime_ns) for name, path in files.items()}
    repo = api.dataset_info(REPO_ID, revision=revision, files_metadata=True)
    if repo.sha != revision:
        raise RuntimeError("Could not verify the uploaded Hub revision; keeping local data.")
    siblings = {item.rfilename: item for item in repo.siblings}
    print(f"Verifying {len(files)} local files against Hub checksums before cleanup...", flush=True)
    for name, path in files.items():
        item = siblings.get(name)
        if item is None or item.size != fingerprints[name][0]:
            raise RuntimeError(f"Hub file is missing or has a different size: {name}. Local data kept.")
        git_hash, lfs_hash = file_checksums(path)
        if item.lfs:
            expected = item.lfs.sha256
            actual = lfs_hash
        else:
            expected = item.blob_id
            actual = git_hash
        if not expected or actual != expected:
            raise RuntimeError(f"Hub checksum mismatch: {name}. Local data kept.")
    current_files = cleanup_files(root)
    if set(current_files) != set(files) or any(
        (path.stat().st_size, path.stat().st_mtime_ns) != fingerprints[name]
        for name, path in current_files.items()
    ):
        raise RuntimeError("Local files changed during verification; cleanup cancelled.")
    if api.dataset_info(REPO_ID, revision="main").sha != revision:
        raise RuntimeError("Hub changed during verification; keeping local data.")
    # Keep a small receipt outside the dataset. Mark before deleting so an
    # interruption cannot cause the next run to mistake partial files for a dataset.
    state = {
        **state, "local_pruned": True, "immutable_hashes": {},
        "synced_frames": info["total_frames"],
    }
    write_json(state_path(root), state)
    shutil.rmtree(root)
    print(f"Hub copy verified. Removed local dataset: {root}", flush=True)
    print("The next recording will resume from the verified Hub dataset.", flush=True)
    return state


def finish_session(api, root, state, *, keep_local=False):
    state = upload_pending(api, root, state)
    if not keep_local:
        state = cleanup_local_after_upload(api, root, state)
    return state


def build_record_config(args, resume):
    from lerobot.cameras.configs import Cv2Backends
    from lerobot.cameras.opencv import OpenCVCameraConfig
    from lerobot.configs.dataset import DatasetRecordConfig
    from lerobot.configs.video import RGBEncoderConfig
    from lerobot.robots.so_follower import SO101FollowerConfig
    from lerobot.scripts.lerobot_record import RecordConfig
    from lerobot.teleoperators.so_leader import SO101LeaderConfig

    cameras = {
        name: OpenCVCameraConfig(
            index_or_path=Path(device), width=640, height=480,
            fps=30, fourcc="MJPG", backend=Cv2Backends.V4L2,
        )
        for name, device in CAMERAS.items()
    }
    return RecordConfig(
        robot=SO101FollowerConfig(
            port=args.follower_port, id="my_awesome_follower_arm",
            calibration_dir=PROJECT_DIR / "calibration", cameras=cameras, use_degrees=True,
        ),
        teleop=SO101LeaderConfig(
            port=args.leader_port, id=getattr(args, "leader_id", "my_awesome_leader_arm"),
            calibration_dir=PROJECT_DIR / "calibration", use_degrees=True,
        ),
        dataset=DatasetRecordConfig(
            repo_id=REPO_ID, root=args.root, single_task=TASK, fps=30,
            num_episodes=args.episodes, episode_time_s=args.episode_seconds,
            reset_time_s=args.reset_seconds, push_to_hub=False, no_stamp=True,
            rgb_encoder=RGBEncoderConfig(vcodec="h264", crf=18, preset="veryfast"),
            encoder_threads=2, video_encoding_batch_size=1,
        ),
        display_data=True, display_mode="rerun",
        display_workspace=str(PROJECT_DIR / "workspace_boundary.json"),
        play_sounds=True, resume=resume, manual_episode_control=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=10, help="NEW episodes in this session (default: 10)")
    parser.add_argument("--episode-seconds", type=float, default=60)
    parser.add_argument("--reset-seconds", type=float, default=30)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--hub-direct", action="store_true",
        help="Stage one episode in RAM, upload immediately, then free RAM; download metadata only",
    )
    parser.add_argument("--ram-root", type=Path, default=Path("/dev/shm/lerobot_stack_3_box_hub"))
    parser.add_argument(
        "--keep-local", action="store_true",
        help="Keep local data after upload (default: verify Hub files, then delete the local dataset)",
    )
    parser.add_argument(
        "--leader-port", default="/dev/serial/by-id/usb-1a86_USB_Single_Serial_5A7C122417-if00"
    )
    parser.add_argument(
        "--leader-id", default="my_awesome_leader_arm",
        help="Leader calibration filename without .json under calibration/",
    )
    parser.add_argument(
        "--follower-port", default="/dev/serial/by-id/usb-1a86_USB_Single_Serial_5AB0158112-if00"
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="Read-only check; do not connect robot or record")
    mode.add_argument("--upload-only", action="store_true", help="Retry uploading saved local episodes")
    args = parser.parse_args()
    args.root = args.root.expanduser().resolve()
    if args.episodes < 1 or args.episode_seconds <= 0 or args.reset_seconds < 0:
        parser.error("Use episodes >= 1, episode-seconds > 0 and reset-seconds >= 0.")
    if args.hub_direct:
        if args.keep_local:
            parser.error("--hub-direct uses temporary RAM storage; omit --keep-local.")
        from record_real_stack_hub import run_hub_direct

        run_hub_direct(args)
        return
    api = HfApi()
    user = api.whoami()["name"]
    revision, info = inspect_hub(api)
    print(f"Hugging Face: {user}\nDataset: {REPO_ID}\nHub episodes: {info['total_episodes']} ({revision})")
    print(f"Task (bottom -> top): WHITE -> GREEN -> BLUE\nLocal: {args.root}", flush=True)
    print("Storage mode: local temporary dataset; upload when the recording session ends.", flush=True)
    if args.check:
        from lerobot.utils.workspace_boundary import WorkspaceBoundaryOverlay

        WorkspaceBoundaryOverlay(PROJECT_DIR / "workspace_boundary.json")
        config = build_record_config(args, resume=False)
        devices = {**CAMERAS, "leader": args.leader_port, "follower": args.follower_port}
        for label, device in devices.items():
            status = "present" if Path(device).exists() else "MISSING"
            print(f"{label}: {device} -> {Path(device).resolve()} ({status})")
        print(
            f"Config OK: {config.dataset.num_episodes} new episodes, border on top only, raw dataset images."
        )
        print("Controls: Space=start/save episode, Q=cancel episode, Esc=save and exit.")
        print(f"Cleanup after verified upload: {'disabled' if args.keep_local else 'enabled'}")
        if not info["total_episodes"]:
            print("Hub is empty: first successful recording initializes the top/left real-camera schema.")
        print("Read-only check complete; no robot connection or upload was performed.")
        return
    args.root.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(state_path(args.root)) + ".lock", timeout=0):
        if args.upload_only and state_path(args.root).exists() and not args.root.exists():
            previous = read_json(state_path(args.root))
            if previous.get("repo_id") == REPO_ID and previous.get("local_pruned"):
                print("Previous upload was verified and the local dataset was already cleaned.", flush=True)
                return
        state, resume = prepare_local(api, args.root)
        if args.upload_only:
            if not (args.root / "meta/info.json").exists():
                raise RuntimeError("No local recorded episodes to upload.")
            finish_session(api, args.root, state, keep_local=args.keep_local)
            return
        if resume:
            state = upload_pending(api, args.root, state)
        from lerobot.scripts.lerobot_record import record

        config = build_record_config(args, resume)
        print(
            f"Up to {args.episodes} NEW episodes. Space: start/save; Q: cancel; Esc: save/exit/upload.",
            flush=True,
        )
        error = None
        try:
            record(config)
        except (Exception, KeyboardInterrupt) as caught:
            error = caught
            print(f"Recording stopped: {type(caught).__name__}: {caught}", flush=True)
        if (args.root / "meta/info.json").exists():
            finish_session(api, args.root, state, keep_local=args.keep_local)
        if error is not None:
            raise error


if __name__ == "__main__":
    main()
