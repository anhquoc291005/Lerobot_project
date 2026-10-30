"""Exercise real follower startup writes with a simulated SDK; never open a serial port."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from lerobot.motors.motors_bus import get_address
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig


class FollowerStartupRetryTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.robot = SO101Follower(SO101FollowerConfig(
            port="/dev/not-opened", calibration_dir=Path(directory.name), id="test_follower",
        ))
        self.bus = self.robot.bus
        self.bus.port_handler = Mock(is_open=True)
        self.bus.packet_handler = Mock()
        self.bus.packet_handler.getTxRxResult.return_value = "There is no status packet!"
        self.bus.packet_handler.getRxPacketError.return_value = "Motor reported an error"
        self.bus.configure_motors = Mock()

    def simulate_missing_replies(self, register, value, failures):
        address, _ = get_address(self.bus.model_ctrl_table, "sts3215", register)
        self.target_calls = 0

        def write_reply(port, motor_id, addr, length, data):
            if motor_id == 5 and addr == address and data == [value]:
                self.target_calls += 1
                if self.target_calls <= failures:
                    return -3001, 0
            return 0, 0

        self.bus.packet_handler.writeTxRx.side_effect = write_reply

    def test_two_lost_replies_recover_for_both_torque_transitions(self):
        for register in ["Torque_Enable", "Lock"]:
            for value in [0, 1]:
                with self.subTest(register=register, value=value):
                    self.simulate_missing_replies(register, value, failures=2)
                    self.robot.configure()
                    self.assertEqual(self.target_calls, 3)

    def test_persistent_missing_reply_still_stops_after_three_attempts(self):
        for register in ["Torque_Enable", "Lock"]:
            with self.subTest(register=register):
                self.simulate_missing_replies(register, 1, failures=100)
                with self.assertRaisesRegex(ConnectionError, "after 3 tries"):
                    self.robot.configure()
                self.assertEqual(self.target_calls, 3)

    def test_motor_error_is_not_retried_as_a_communication_failure(self):
        self.bus.packet_handler.writeTxRx.return_value = (0, 32)
        with self.assertRaises(RuntimeError):
            self.robot.configure()
        self.assertEqual(self.bus.packet_handler.writeTxRx.call_count, 1)


if __name__ == "__main__":
    unittest.main()
