"""One-episode RAM staging and metadata-only resume for the real SO101 recorder."""

import os
import shutil
import tempfile
from pathlib import Path

from filelock import FileLock
from huggingface_hub import HfApi, snapshot_download

import record_real_stack_3_box as batch


def prepare_ram_episode(api, root):
    revision, remote = batch.inspect_hub(api)
    marker = batch.state_path(root)
    if root.exists():
        if not marker.exists():
            raise RuntimeError(f"Unknown RAM directory exists: {root}; choose another --ram-root.")
        state = batch.read_json(marker)
        if state["repo_id"] != batch.REPO_ID or state.get("local_pruned"):
            raise RuntimeError("RAM staging was interrupted; existing files are kept. Choose another --ram-root.")
        if state["base_sha"] != revision:
            raise RuntimeError("Hub changed while a RAM episode was pending; temporary data is kept.")
        return state, (root / "meta/info.json").exists()
    if marker.exists():
        previous = batch.read_json(marker)
        if not previous.get("local_pruned") and previous.get("synced_episodes", 0):
            raise RuntimeError("RAM staging disappeared before verification; check the previous session.")
        if remote["total_episodes"] < previous.get("synced_episodes", 0):
            raise RuntimeError("Hub episode count decreased since the verified upload.")
    repo = api.dataset_info(batch.REPO_ID, revision=revision)
    remote_files = [item.rfilename for item in repo.siblings]
    if remote["total_episodes"]:
        snapshot_download(
            batch.REPO_ID, repo_type="dataset", revision=revision, local_dir=root,
            allow_patterns=["meta/**"],
        )
        batch.validate_schema(batch.read_json(root / "meta/info.json"))
    state = {
        "repo_id": batch.REPO_ID, "base_sha": revision,
        "synced_episodes": remote["total_episodes"], "record_base_episodes": remote["total_episodes"],
        "record_base_frames": remote["total_frames"], "remote_files": remote_files,
        "immutable_hashes": batch.immutable_hashes(root),
    }
    batch.write_json(marker, state)
    return state, bool(remote["total_episodes"])


def validate_ram_episode(root, state):
    import pyarrow.parquet as pq

    info = batch.read_json(root / "meta/info.json")
    batch.validate_schema(info)
    start = state["record_base_episodes"]
    if info["total_episodes"] != start + 1:
        raise RuntimeError("RAM staging must contain exactly one new complete episode.")
    base_files = set(state["remote_files"])
    local_files = set(batch.cleanup_files(root))
    new_data = sorted(root.glob("data/**/*.parquet"))
    frames = sum(pq.ParquetFile(path).metadata.num_rows for path in new_data)
    if not frames or frames != info["total_frames"] - state["record_base_frames"]:
        raise RuntimeError("New RAM frame counts do not match the saved episode.")
    for path in new_data:
        indices = pq.ParquetFile(path).read(columns=["episode_index"])["episode_index"].to_pylist()
        if set(indices) != {start}:
            raise RuntimeError("RAM data contains unexpected episode indices.")
    if any(name in base_files for name in local_files if name.startswith(("data/", "videos/"))):
        raise RuntimeError("A new RAM shard would overwrite existing Hub data; upload stopped.")
    rows = []
    for path in root.glob("meta/episodes/**/*.parquet"):
        rows.extend(pq.ParquetFile(path).read().to_pylist())
    rows.sort(key=lambda row: row["episode_index"])
    if [row["episode_index"] for row in rows] != list(range(info["total_episodes"])):
        raise RuntimeError("RAM episode metadata has gaps or duplicates.")
    offset = 0
    for row in rows:
        if row["dataset_from_index"] != offset or row["dataset_to_index"] != offset + row["length"]:
            raise RuntimeError("RAM episode frame ranges are inconsistent.")
        offset += row["length"]
        names = [info["data_path"].format(
            chunk_index=row["data/chunk_index"], file_index=row["data/file_index"],
        )]
        for camera in batch.CAMERAS:
            key = f"observation.images.{camera}"
            names.append(info["video_path"].format(
                video_key=key, chunk_index=row[f"videos/{key}/chunk_index"],
                file_index=row[f"videos/{key}/file_index"],
            ))
        available = local_files if row["episode_index"] == start else base_files
        if any(name not in available for name in names):
            raise RuntimeError("An episode references a missing data/video shard.")
    if offset != info["total_frames"]:
        raise RuntimeError("RAM metadata frame total is inconsistent.")
    return info


def publish_ram_episode(api, root, state):
    info = validate_ram_episode(root, state)
    batch.verify_old_shards(root, state)
    revision, remote = batch.inspect_hub(api)
    if revision != state["base_sha"] or remote["total_episodes"] != state["synced_episodes"]:
        raise RuntimeError("Hub changed before upload. Pending episode remains in RAM.")
    if info["total_episodes"] > state["synced_episodes"]:
        batch.write_dataset_card(root, info)
        commit = api.upload_folder(
            repo_id=batch.REPO_ID, repo_type="dataset", revision="main", parent_commit=revision,
            folder_path=root, allow_patterns=["meta/**", "data/**", "videos/**", "README.md"],
            delete_patterns=None, commit_message=f"Append real episode {info['total_episodes'] - 1}",
        )
        state = {
            **state, "base_sha": commit.oid, "synced_episodes": info["total_episodes"],
            "immutable_hashes": batch.immutable_hashes(root),
        }
        batch.write_json(batch.state_path(root), state)
    batch.update_version_tag(api, state["base_sha"])
    return batch.cleanup_local_after_upload(
        api, root, state, validate_fn=lambda path: validate_ram_episode(path, state),
    )


def resume_ram_writer(config):
    from lerobot.datasets import LeRobotDataset

    return LeRobotDataset.resume(
        batch.REPO_ID, root=config.dataset.root, rgb_encoder=config.dataset.rgb_encoder,
        encoder_threads=2, streaming_encoding=True, encoder_queue_maxsize=30,
    )


def run_hub_direct(args):
    root = args.ram_root.expanduser().resolve()
    if root == Path("/dev/shm") or not root.is_relative_to("/dev/shm"):
        raise RuntimeError("--hub-direct requires --ram-root under /dev/shm (RAM filesystem).")
    # Xet transfers and encoder helpers also use temporary/cache paths. Keep
    # those in RAM, alongside the episode, rather than the default SSD caches.
    os.environ["HF_XET_CACHE"] = str(root / ".cache/huggingface/xet")
    os.environ["HF_XET_CHUNK_CACHE_SIZE_BYTES"] = "0"
    os.environ["TMPDIR"] = "/dev/shm"
    tempfile.tempdir = "/dev/shm"

    from lerobot.scripts.lerobot_record import record
    from lerobot.utils.workspace_boundary import WorkspaceBoundaryOverlay

    args.root = root
    config = batch.build_record_config(args, resume=False)
    config.dataset.streaming_encoding = True
    config.dataset.num_image_writer_threads_per_camera = 0
    WorkspaceBoundaryOverlay(batch.PROJECT_DIR / "workspace_boundary.json")
    api = HfApi()
    user = api.whoami()["name"]
    revision, remote = batch.inspect_hub(api)
    print(f"Hugging Face: {user}; dataset: {batch.REPO_ID}; episodes: {remote['total_episodes']}")
    print(f"RAM staging: {root}. Upload and release after each episode; download metadata only.", flush=True)
    if args.check:
        free = shutil.disk_usage("/dev/shm").free / 1024**3
        print(f"RAM filesystem available: {free:.2f} GiB. Read-only check complete; no recording/upload.")
        return
    if shutil.disk_usage("/dev/shm").free < 256 * 1024**2:
        raise RuntimeError("Insufficient free RAM staging space (need at least 256 MiB).")
    with FileLock(str(batch.state_path(root)) + ".lock", timeout=0):
        if args.upload_only:
            if not root.exists() or not batch.state_path(root).exists():
                raise RuntimeError("No pending RAM episode to upload.")
            publish_ram_episode(api, root, batch.read_json(batch.state_path(root)))
            return
        state, resume = prepare_ram_episode(api, root)
        if resume:
            local = batch.read_json(root / "meta/info.json")
            if local["total_episodes"] > state["record_base_episodes"]:
                publish_ram_episode(api, root, state)
                state, resume = prepare_ram_episode(api, root)
        config.resume = resume

        def episode_ready(dataset, has_next):
            nonlocal state
            dataset.finalize()
            state = publish_ram_episode(api, root, state)
            if not has_next:
                return dataset
            state, _ = prepare_ram_episode(api, root)
            return resume_ram_writer(config)

        try:
            record(config, episode_saved_callback=episode_ready)
        except BaseException:
            if root.exists():
                print(f"Temporary data retained in RAM: {root}", flush=True)
                print("Retry: bash record_real_stack_3_box.sh --hub-direct --upload-only", flush=True)
                print("RAM contents do not survive reboot/power loss.", flush=True)
            raise
