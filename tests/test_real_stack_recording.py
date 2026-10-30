"""Offline recording/resume and upload guards; never connect hardware or write to Hub."""

import hashlib
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import cv2
import numpy as np

import record_real_stack_3_box as recorder
import record_real_stack_hub as direct
from lerobot.configs.video import RGBEncoderConfig
from lerobot.datasets import LeRobotDataset
from lerobot.utils.workspace_boundary import WorkspaceBoundaryOverlay


class RealRecordingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = tempfile.TemporaryDirectory()
        cls.fixture_root = Path(cls.fixture.name) / "dataset"
        features = {
            key: {"dtype": "float32", "shape": (6,), "names": recorder.JOINTS}
            for key in ["observation.state", "action"]
        }
        features.update({
            f"observation.images.{camera}": {
                "dtype": "video", "shape": (480, 640, 3), "names": ["height", "width", "channels"],
            }
            for camera in recorder.CAMERAS
        })
        encoder = RGBEncoderConfig(vcodec="h264", preset="veryfast", crf=18)
        dataset = LeRobotDataset.create(
            recorder.REPO_ID, fps=30, root=cls.fixture_root, robot_type="so_follower",
            features=features, use_videos=True, rgb_encoder=encoder, encoder_threads=2,
        )
        cls._save_episode(dataset)
        dataset.finalize()
        cls.old_info = recorder.read_json(cls.fixture_root / "meta/info.json")
        cls.old_hashes = recorder.immutable_hashes(cls.fixture_root)
        resumed = LeRobotDataset.resume(
            recorder.REPO_ID, root=cls.fixture_root, rgb_encoder=encoder, encoder_threads=2,
        )
        cls._save_episode(resumed)
        resumed.finalize()

    @classmethod
    def tearDownClass(cls):
        cls.fixture.cleanup()

    @staticmethod
    def _save_episode(dataset):
        for _ in range(4):
            dataset.add_frame({
                "observation.state": np.zeros(6, dtype=np.float32),
                "action": np.ones(6, dtype=np.float32),
                "observation.images.top": np.full((480, 640, 3), [80, 60, 40], dtype=np.uint8),
                "observation.images.left": np.full((480, 640, 3), [40, 60, 80], dtype=np.uint8),
                "task": recorder.TASK,
            })
        dataset.save_episode()

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name) / "dataset"
        shutil.copytree(self.fixture_root, self.root)
        self.state = {
            "repo_id": recorder.REPO_ID, "base_sha": "old-commit", "synced_episodes": 1,
            "immutable_hashes": self.old_hashes,
        }

    def test_two_sessions_append_and_preserve_all_old_shards(self):
        info = recorder.validate_saved_dataset(self.root)
        self.assertEqual(info["total_episodes"], 2)
        self.assertEqual(info["total_frames"], 8)
        recorder.verify_old_shards(self.root, self.state)
        self.assertGreater(len(recorder.immutable_hashes(self.root)), len(self.old_hashes))

    def test_upload_is_guarded_and_never_deletes_remote_files(self):
        api = Mock()
        api.upload_folder.return_value = SimpleNamespace(
            oid="new-commit", commit_url="https://example.test/commit"
        )
        with patch.object(recorder, "inspect_hub", return_value=("old-commit", self.old_info)), \
                patch.object(recorder, "update_version_tag"):
            state = recorder.upload_pending(api, self.root, self.state)
        kwargs = api.upload_folder.call_args.kwargs
        self.assertEqual(kwargs["parent_commit"], "old-commit")
        self.assertEqual(kwargs["repo_id"], recorder.REPO_ID)
        self.assertIsNone(kwargs["delete_patterns"])
        self.assertEqual(state["synced_episodes"], 2)
        self.assertEqual(recorder.read_json(recorder.state_path(self.root))["base_sha"], "new-commit")
        recorder.verify_old_shards(self.root, self.state)
        new_info = recorder.read_json(self.root / "meta/info.json")
        with patch.object(recorder, "inspect_hub", return_value=("new-commit", new_info)):
            next_state, resume = recorder.prepare_local(api, self.root)
        self.assertTrue(resume)
        self.assertEqual(next_state["synced_episodes"], 2)

    def test_newer_hub_commit_prevents_upload(self):
        api = Mock()
        with patch.object(recorder, "inspect_hub", return_value=("someone-elses-commit", self.old_info)):
            with self.assertRaisesRegex(RuntimeError, "Hub changed"):
                recorder.upload_pending(api, self.root, self.state)
        api.upload_folder.assert_not_called()

    def test_changed_old_shard_prevents_upload(self):
        name = next(iter(self.old_hashes))
        with (self.root / name).open("ab") as stream:
            stream.write(b"modified")
        with self.assertRaisesRegex(RuntimeError, "old shard"):
            recorder.verify_old_shards(self.root, self.state)

    def test_unknown_local_directory_is_preserved(self):
        before = recorder.immutable_hashes(self.root)
        with patch.object(recorder, "inspect_hub", return_value=("old-commit", self.old_info)):
            with self.assertRaisesRegex(RuntimeError, "Unmanaged local directory"):
                recorder.prepare_local(Mock(), self.root)
        self.assertEqual(before, recorder.immutable_hashes(self.root))

    def test_saved_video_has_no_green_border(self):
        video = next(self.root.glob("videos/observation.images.top/**/*.mp4"))
        capture = cv2.VideoCapture(str(video))
        try:
            ok, frame = capture.read()
            self.assertTrue(ok)
            overlay = WorkspaceBoundaryOverlay(recorder.PROJECT_DIR / "workspace_boundary.json")
            x, y = np.rint(overlay.points_for_image("top", 640, 480)[0]).astype(int)
            rgb = frame[y, x, ::-1].astype(float)
            np.testing.assert_allclose(rgb, [80, 60, 40], atol=8)
        finally:
            capture.release()

    def _verified_hub_copy(self):
        info = recorder.read_json(self.root / "meta/info.json")
        state = {
            "repo_id": recorder.REPO_ID, "base_sha": "verified-commit", "synced_episodes": 2,
            "immutable_hashes": recorder.immutable_hashes(self.root),
        }
        siblings = []
        for name, path in recorder.cleanup_files(self.root).items():
            content = path.read_bytes()
            blob_id = hashlib.sha1(
                b"blob " + str(len(content)).encode() + b"\0" + content, usedforsecurity=False,
            ).hexdigest()
            lfs = (
                SimpleNamespace(sha256=hashlib.sha256(content).hexdigest()) if name.endswith(".mp4") else None
            )
            siblings.append(SimpleNamespace(rfilename=name, size=len(content), blob_id=blob_id, lfs=lfs))
        api = Mock()
        api.dataset_info.return_value = SimpleNamespace(sha="verified-commit", siblings=siblings)
        return api, info, state

    def test_verified_upload_cleans_and_next_session_restores_same_root(self):
        api, info, state = self._verified_hub_copy()
        cache = self.root / ".cache/huggingface/upload/temporary.metadata"
        cache.parent.mkdir(parents=True)
        cache.write_text("generated Hub cache")
        with patch.object(recorder, "inspect_hub", return_value=("verified-commit", info)):
            pruned = recorder.cleanup_local_after_upload(api, self.root, state)
        self.assertFalse(self.root.exists())
        self.assertTrue(pruned["local_pruned"])
        self.assertTrue(recorder.read_json(recorder.state_path(self.root))["local_pruned"])

        def download(repo_id, **kwargs):
            self.assertEqual(kwargs["revision"], "latest-commit")
            shutil.copytree(self.fixture_root, kwargs["local_dir"])

        with patch.object(recorder, "inspect_hub", return_value=("latest-commit", info)), \
                patch.object(recorder, "snapshot_download", side_effect=download):
            restored, resume = recorder.prepare_local(api, self.root)
        self.assertTrue(resume)
        self.assertEqual(restored["base_sha"], "latest-commit")
        self.assertFalse(restored.get("local_pruned", False))
        self.assertEqual(recorder.validate_saved_dataset(self.root)["total_episodes"], 2)

    def test_hub_checksum_mismatch_keeps_local_files(self):
        api, info, state = self._verified_hub_copy()
        item = api.dataset_info.return_value.siblings[0]
        if item.lfs:
            item.lfs.sha256 = "wrong"
        else:
            item.blob_id = "wrong"
        before = recorder.immutable_hashes(self.root)
        with patch.object(recorder, "inspect_hub", return_value=("verified-commit", info)):
            with self.assertRaisesRegex(RuntimeError, "checksum mismatch"):
                recorder.cleanup_local_after_upload(api, self.root, state)
        self.assertTrue(self.root.exists())
        self.assertEqual(recorder.immutable_hashes(self.root), before)

    def test_unuploaded_raw_frames_prevent_cleanup(self):
        api, info, state = self._verified_hub_copy()
        raw = self.root / "images/unsaved_frame.png"
        raw.parent.mkdir(parents=True, exist_ok=True)
        raw.write_bytes(b"unsaved raw frame")
        with patch.object(recorder, "inspect_hub", return_value=("verified-commit", info)):
            with self.assertRaisesRegex(RuntimeError, "Hub file is missing"):
                recorder.cleanup_local_after_upload(api, self.root, state)
        self.assertEqual(raw.read_bytes(), b"unsaved raw frame")
        self.assertTrue((self.root / "meta/info.json").exists())

    def test_upload_failure_never_runs_cleanup(self):
        with patch.object(recorder, "upload_pending", side_effect=OSError("network unavailable")), \
                patch.object(recorder, "cleanup_local_after_upload") as cleanup:
            with self.assertRaises(OSError):
                recorder.finish_session(Mock(), self.root, self.state)
        cleanup.assert_not_called()
        self.assertTrue((self.root / "meta/info.json").exists())

    def test_keep_local_disables_cleanup(self):
        with patch.object(recorder, "upload_pending", return_value=self.state), \
                patch.object(recorder, "cleanup_local_after_upload") as cleanup:
            recorder.finish_session(Mock(), self.root, self.state, keep_local=True)
        cleanup.assert_not_called()
        self.assertTrue(self.root.exists())

    def test_native_recorder_uploads_each_episode_with_metadata_only_resume(self):
        from lerobot.scripts import lerobot_record as native

        ram = Path(self.directory.name) / "ram_staging"
        revision = ["cloud-2"]
        original_hashes = recorder.immutable_hashes(self.root)
        api = Mock()

        def repo_info(*args, **kwargs):
            siblings = []
            for name, path in recorder.cleanup_files(self.root).items():
                content = path.read_bytes()
                blob = hashlib.sha1(
                    b"blob " + str(len(content)).encode() + b"\0" + content, usedforsecurity=False,
                ).hexdigest()
                siblings.append(SimpleNamespace(rfilename=name, size=len(content), blob_id=blob, lfs=None))
            return SimpleNamespace(sha=revision[0], siblings=siblings)

        api.dataset_info.side_effect = repo_info

        def upload(**kwargs):
            self.assertEqual(kwargs["parent_commit"], revision[0])
            self.assertIsNone(kwargs["delete_patterns"])
            for name, path in recorder.cleanup_files(ram).items():
                destination = self.root / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, destination)
            revision[0] = "cloud-" + str(recorder.read_json(self.root / "meta/info.json")["total_episodes"])
            return SimpleNamespace(oid=revision[0], commit_url="https://example.test/commit")

        api.upload_folder.side_effect = upload

        def download(repo_id, **kwargs):
            self.assertEqual(kwargs["allow_patterns"], ["meta/**"])
            shutil.copytree(self.root / "meta", Path(kwargs["local_dir"]) / "meta")

        robot = Mock()
        robot.name = robot.robot_type = "so_follower"
        robot.action_features = {name: float for name in recorder.JOINTS}
        robot.observation_features = {
            **robot.action_features, "top": (480, 640, 3), "left": (480, 640, 3),
        }
        robot.cameras = {"top": None, "left": None}
        robot.is_connected = False
        robot.connect.side_effect = lambda: setattr(robot, "is_connected", True)
        robot.disconnect.side_effect = lambda: setattr(robot, "is_connected", False)
        teleop = Mock(is_connected=False)
        args = SimpleNamespace(root=ram, episodes=2, episode_seconds=1, reset_seconds=0,
                               follower_port="unused", leader_port="unused")
        config = recorder.build_record_config(args, resume=True)
        config.manual_episode_control = False
        config.display_data = False
        config.play_sounds = False
        config.dataset.streaming_encoding = True
        config.dataset.num_image_writer_threads_per_camera = 0
        events = {"stop_recording": False, "rerecord_episode": False, "exit_early": False}
        counts = []

        def capture(**kwargs):
            dataset = kwargs.get("dataset")
            if dataset is None:
                return
            self.assertFalse(any(ram.glob("videos/**/*.mp4")))
            for _ in range(4):
                dataset.add_frame({
                    "observation.state": np.zeros(6, dtype=np.float32),
                    "action": np.ones(6, dtype=np.float32),
                    "observation.images.top": np.full((480, 640, 3), [80, 60, 40], dtype=np.uint8),
                    "observation.images.left": np.full((480, 640, 3), [40, 60, 80], dtype=np.uint8),
                    "task": recorder.TASK,
                })

        with patch.object(recorder, "inspect_hub", side_effect=lambda api: (
            revision[0], recorder.read_json(self.root / "meta/info.json")
        )), patch.object(direct, "snapshot_download", side_effect=download), \
                patch.object(recorder, "update_version_tag"), \
                patch.object(native, "make_robot_from_config", return_value=robot), \
                patch.object(native, "make_teleoperator_from_config", return_value=teleop), \
                patch.object(native, "init_keyboard_listener", return_value=(None, events)), \
                patch.object(native, "record_loop", side_effect=capture), \
                patch.object(native, "log_say"), patch.object(native, "init_logging"):
            state, _ = direct.prepare_ram_episode(api, ram)

            def ready(dataset, has_next):
                nonlocal state
                dataset.finalize()
                counts.append(dataset.num_episodes)
                state = direct.publish_ram_episode(api, ram, state)
                self.assertFalse(ram.exists())
                if has_next:
                    state, _ = direct.prepare_ram_episode(api, ram)
                    return direct.resume_ram_writer(config)
                return dataset

            native.record(config, episode_saved_callback=ready)
        self.assertEqual(counts, [3, 4])
        self.assertEqual(api.upload_folder.call_count, 2)
        self.assertEqual(recorder.validate_saved_dataset(self.root)["total_episodes"], 4)
        for name, digest in original_hashes.items():
            self.assertEqual(recorder.immutable_hashes(self.root)[name], digest)
        self.assertFalse(ram.exists())


if __name__ == "__main__":
    unittest.main()
