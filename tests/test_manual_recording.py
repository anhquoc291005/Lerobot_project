"""Exercise Space/Q/Esc against real dataset writers, without hardware or Hub access."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

import record_real_stack_3_box as collector
from lerobot.scripts import lerobot_record as native
from lerobot.utils import keyboard_input as keys


class ManualRecordingTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def new_events(self):
        with patch.object(keys, "create_key_listener", return_value=Mock()):
            return keys.init_manual_recording_listener()

    def test_key_mapping_is_sensitive_to_phase(self):
        _, events = self.new_events()
        keys.apply_manual_recording_control("q", events)
        self.assertFalse(events["stop_recording"])
        self.assertFalse(events["rerecord_episode"])
        keys.apply_manual_recording_control("space", events)
        self.assertTrue(events["start_episode"])
        self.assertEqual(events["phase"], "starting")
        events.update(phase="recording", exit_early=False, start_episode=False)
        keys.apply_manual_recording_control("space", events)
        self.assertTrue(events["exit_early"])
        self.assertEqual(events["phase"], "saving")
        self.assertFalse(events["stop_recording"])
        self.assertFalse(events["rerecord_episode"])
        keys.apply_manual_recording_control("space", events)
        self.assertFalse(events["start_episode"])
        events.update(phase="recording", exit_early=False)
        keys.apply_manual_recording_control("Q", events)
        self.assertTrue(events["rerecord_episode"])
        self.assertFalse(events["stop_recording"])
        keys.apply_manual_recording_control("esc", events)
        self.assertTrue(events["stop_recording"])

    def run_session(self, scenario, *, streaming=True):
        path = self.root / f"{scenario}_{streaming}"
        args = SimpleNamespace(root=path, episodes=10, episode_seconds=60, reset_seconds=30,
                               follower_port="unused", leader_port="unused")
        config = collector.build_record_config(args, resume=False)
        config.display_data = False
        config.play_sounds = False
        config.dataset.streaming_encoding = streaming
        robot = Mock()
        robot.name = robot.robot_type = "so_follower"
        robot.action_features = {joint: float for joint in collector.JOINTS}
        robot.observation_features = {
            **robot.action_features, "top": (480, 640, 3), "left": (480, 640, 3),
        }
        robot.cameras = {"top": None, "left": None}
        robot.is_connected = False
        listener, events = self.new_events()
        calls = []
        status_updates = []
        recording_count = 0

        def control_loop(**kwargs):
            nonlocal recording_count
            dataset = kwargs.get("dataset")
            calls.append("preview" if dataset is None else "record")
            if dataset is None:
                self.assertEqual(events["phase"], "waiting")
                self.assertFalse(status_updates[-1][0])
                self.assertEqual(kwargs["control_time_s"], float("inf"))
                keys.apply_manual_recording_control("esc" if scenario == "escape_waiting" else "space", events)
                return
            self.assertEqual(events["phase"], "recording")
            self.assertTrue(status_updates[-1][0])
            recording_count += 1
            if scenario == "escape_before_frame":
                keys.apply_manual_recording_control("esc", events)
                return
            for _ in range(4):
                dataset.add_frame({
                    "observation.state": np.zeros(6, dtype=np.float32),
                    "action": np.ones(6, dtype=np.float32),
                    "observation.images.top": np.full((480, 640, 3), [80, 60, 40], dtype=np.uint8),
                    "observation.images.left": np.full((480, 640, 3), [40, 60, 80], dtype=np.uint8),
                    "task": collector.TASK,
                })
            keys.apply_manual_recording_control(["q", "space", "esc"][recording_count - 1], events)

        def remember_status(active, episode, **kwargs):
            status_updates.append((active, episode, kwargs["detail"]))

        with patch.object(native, "make_robot_from_config", return_value=robot), \
                patch.object(native, "make_teleoperator_from_config", return_value=Mock(is_connected=False)), \
                patch.object(native, "init_manual_recording_listener", return_value=(listener, events)), \
                patch.object(native, "record_loop", side_effect=control_loop), \
                patch.object(native, "report_recording_status", side_effect=remember_status), \
                patch.object(native, "log_say"), patch.object(native, "init_logging"):
            native.record(config)
        listener.stop.assert_called_once()
        self.status_updates = status_updates
        self.assertFalse(status_updates[-1][0])
        return collector.validate_saved_dataset(path), calls

    def test_cancel_then_save_then_escape_saves_only_valid_episodes(self):
        for streaming in [False, True]:
            with self.subTest(streaming=streaming):
                info, calls = self.run_session("cancel_save_exit", streaming=streaming)
                self.assertEqual(info["total_episodes"], 2)
                self.assertEqual(info["total_frames"], 8)
                self.assertEqual(calls, ["preview", "record"] * 3)
                cancelled = [item for item in self.status_updates if "Đã hủy episode" in item[2]]
                self.assertEqual(len(cancelled), 1)
                self.assertFalse(cancelled[0][0])

    def test_escape_while_waiting_does_not_create_an_episode(self):
        info, calls = self.run_session("escape_waiting")
        self.assertEqual(info["total_episodes"], 0)
        self.assertEqual(info["total_frames"], 0)
        self.assertEqual(calls, ["preview"])

    def test_escape_before_first_frame_does_not_save_empty_episode(self):
        info, calls = self.run_session("escape_before_frame")
        self.assertEqual(info["total_episodes"], 0)
        self.assertEqual(info["total_frames"], 0)
        self.assertEqual(calls, ["preview", "record"])


if __name__ == "__main__":
    unittest.main()
