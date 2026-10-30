"""Check visible status text and the separate Rerun display channel."""

import unittest
from unittest.mock import patch

import rerun as rr

from lerobot.utils.recording_status import RECORDING_STATUS_PATH, report_recording_status


class RecordingStatusTests(unittest.TestCase):
    def test_recording_then_cancellation_updates_same_display_document(self):
        with patch.object(rr, "log") as log:
            report_recording_status(True, 0, display_mode="rerun", detail="Đang ghi dữ liệu.")
            report_recording_status(False, 0, display_mode="rerun", detail="Đã hủy episode. Chờ Space.")
        self.assertEqual(log.call_count, 2)
        for call in log.call_args_list:
            self.assertEqual(call.args[0], RECORDING_STATUS_PATH)
            self.assertTrue(call.kwargs["static"])
        active = log.call_args_list[0].args[1].text.as_arrow_array().to_pylist()[0]
        waiting = log.call_args_list[1].args[1].text.as_arrow_array().to_pylist()[0]
        self.assertIn("ĐANG THU DATA", active)
        self.assertIn("CHƯA THU DATA", waiting)
        self.assertIn("Đã hủy episode", waiting)
        self.assertIn("Episode 0", waiting)

    def test_headless_status_is_still_reported_to_terminal(self):
        with patch.object(rr, "log") as log, self.assertLogs(level="INFO") as terminal:
            report_recording_status(False, 2, detail="Chờ Space.")
        log.assert_not_called()
        self.assertIn("CHƯA THU DATA", terminal.output[0])


if __name__ == "__main__":
    unittest.main()
