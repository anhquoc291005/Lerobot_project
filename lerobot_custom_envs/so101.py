# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any

import cv2
import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np


# Danh sach o bi loai bo theo thiet lap thuc te
DEFAULT_STACK_EXCLUDED_CELLS: tuple[str, ...] = (
    "e1", "e2", "e3", "e4", "e5", "e6", "e7",
    "c1", "c7",
    "d1", "d2", "d6", "d7",
)

# Toa do tam chuan xac cua cac o luoi tren mat san (z=0.015m)
# Can chinh dong deu voi 4 canh bien cua o tren camera top (5x7)
DEFAULT_STACK_CELL_CENTERS: dict[str, tuple[float, float]] = {
    "b2": (0.1362, -0.2171),
    "b3": (0.1367, -0.1166),
    "b5": (0.1378, 0.0795),
    "b6": (0.1384, 0.1772),
    "c2": (0.2383, -0.2226),
    "c3": (0.2383, -0.1188),
    "c5": (0.2383, 0.0795),
    "c6": (0.2383, 0.1772),
    "d3": (0.3406, -0.1166),
    "d4": (0.3427, -0.0171),
    "d5": (0.3395, 0.0795),
}

DEFAULT_STACK_LEFT_CELLS: tuple[str, ...] = ("b2", "b3", "c2", "c3", "d3")
DEFAULT_STACK_RIGHT_CELLS: tuple[str, ...] = ("b5", "b6", "c5", "c6", "d5")

DEFAULT_TARGET_ZONE_POS: tuple[float, float] = (0.264, -0.0175)


def _resolve_default_scene_xml() -> Path:
    """Tim duong dan toi file scene.xml cua SO-101."""
    candidates = [
        Path("/home/anhquoc2910/Lerobot_project/github_export/so101-sim2real-policy-research/reference/sim_assets/so_arm100_simulation/SO101/scene.xml"),
        Path(__file__).resolve().parent.parent.parent.parent / "github_export/so101-sim2real-policy-research/reference/sim_assets/so_arm100_simulation/SO101/scene.xml",
        Path.cwd() / "github_export/so101-sim2real-policy-research/reference/sim_assets/so_arm100_simulation/SO101/scene.xml",
        Path.cwd() / "reference/sim_assets/so_arm100_simulation/SO101/scene.xml",
    ]
    for cand in candidates:
        if cand.exists():
            return cand.resolve()
    # Fallback ve candidate dau tien
    return candidates[0]


def _yaw_to_quat(yaw: float) -> np.ndarray:
    half = yaw * 0.5
    return np.array([math.cos(half), 0.0, 0.0, math.sin(half)], dtype=np.float64)


def draw_grid_5x7(
    image: np.ndarray,
    rows: int = 5,
    cols: int = 7,
    labels: bool = True,
    margin_left: int = 16,
    margin_top: int = 32,
    margin_right: int = 16,
    margin_bottom: int = 0,
) -> None:
    """Ve luoi 5x7 len anh BGR de can chinh giong preview thu thap data."""
    h, w = image.shape[:2]
    x_min = margin_left
    x_max = w - margin_right
    y_min = margin_top
    y_max = h - margin_bottom

    row_names = ("a", "b", "c", "d", "e")
    grid_w = x_max - x_min
    grid_h = y_max - y_min

    # Ve duong doc
    for c in range(cols + 1):
        x = int(x_min + round(c * grid_w / cols))
        cv2.line(image, (x, y_min), (x, y_max), (80, 80, 80), 1)

    # Ve duong ngang
    for r in range(rows + 1):
        y = int(y_min + round(r * grid_h / rows))
        cv2.line(image, (x_min, y), (x_max, y), (80, 80, 80), 1)

    if labels:
        for r in range(rows):
            for c in range(cols):
                rx = int(x_min + round(c * grid_w / cols))
                ry = int(y_min + round(r * grid_h / rows))
                lbl = f"{row_names[r]}{c + 1}"
                cv2.putText(
                    image,
                    lbl,
                    (rx + 4, ry + 18),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.38,
                    (130, 130, 130),
                    1,
                    cv2.LINE_AA,
                )


class SO101Stack3BlocksEnv(gym.Env):
    """Moi truong Gymnasium mo phong SO-101 xep chong 3 khoi hop ngau nhien (Stack 3 Blocks).
    
    Cac thong so duoc thiet lap dong bo 100% voi qua trinh thu thap du lieu:
    - MuJoCo Model: SO-101 moi calib (sts3215 gains, friction=1.2)
    - 3 Khoi hop: 30x30x30 mm (box_black, box_green, box_blue)
    - Dich xep chong: stack_target_zone o vi tri C4 sát D4 [0.264, -0.0175, 0.001]
    - Che do random: Grid-cell based (1 o ben trai, 1 o ben phai, 1 o giua, min dist >= 12cm)
    - Tan so: 30 FPS (timestep 0.001s, frame_skip 33)
    - Camera:
        + top: top_cam (640x480, fovy 55)
        + front: front_cam (640x480, fovy 52, brightness_gain 1.2)
    - Don vi State & Action: DO (DEGREES) de tuong thich truc tiep voi ACT Policy da train!
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(
        self,
        xml_path: str | Path | None = None,
        max_episode_steps: int = 1800,
        render_mode: str = "rgb_array",
        top_camera_name: str = "top_cam",
        front_camera_name: str = "front_cam",
        front_camera_brightness_gain: float = 1.2,
        image_height: int = 480,
        image_width: int = 640,
        frame_skip: int = 33,
        cell_jitter: float = 0.008,
        min_cell_distance: float = 0.12,
        render_composite: bool = True,
        render_grid_overlay: bool = True,
        **kwargs: Any,
    ):
        super().__init__()
        self._max_episode_steps = max_episode_steps
        self.render_mode = render_mode
        self.task_description = (
            "Stack 3 blocks (30x30x30 mm): white at the bottom, green in the middle, blue on top."
        )
        self.task = "so101_stack_3_blocks"

        if xml_path is None:
            resolved_xml = _resolve_default_scene_xml()
        else:
            resolved_xml = Path(xml_path)
            if not resolved_xml.is_absolute():
                candidates = [
                    Path.cwd() / resolved_xml,
                    Path("/home/anhquoc2910/Lerobot_project") / resolved_xml,
                    _resolve_default_scene_xml().parent / resolved_xml,
                ]
                for c in candidates:
                    if c.exists():
                        resolved_xml = c
                        break

        if not resolved_xml.exists():
            raise FileNotFoundError(f"MuJoCo XML scene not found at: {resolved_xml}")

        self.xml_path = str(resolved_xml)
        self.top_camera_name = top_camera_name
        self.front_camera_name = front_camera_name
        self.front_camera_brightness_gain = front_camera_brightness_gain
        self.image_height = image_height
        self.image_width = image_width
        self.frame_skip = frame_skip
        self.cell_jitter = cell_jitter
        self.min_cell_distance = min_cell_distance
        self.render_composite = render_composite
        self.render_grid_overlay = render_grid_overlay

        # Khoi tao MuJoCo Model & Data
        self.model = mujoco.MjModel.from_xml_path(self.xml_path)
        self.model.vis.global_.offwidth = max(int(self.model.vis.global_.offwidth), image_width * 2)
        self.model.vis.global_.offheight = max(int(self.model.vis.global_.offheight), image_height)
        self.data = mujoco.MjData(self.model)

        self._renderer = mujoco.Renderer(
            self.model,
            height=self.image_height,
            width=self.image_width,
        )

        # Gioi han dieu khien (Actuator control range)
        # MuJoCo luu goc theo radian, policy su dung do (degrees)
        self._ctrl_low_rad = self.model.actuator_ctrlrange[:, 0].copy().astype(np.float32)
        self._ctrl_high_rad = self.model.actuator_ctrlrange[:, 1].copy().astype(np.float32)
        self._ctrl_low_deg = np.rad2deg(self._ctrl_low_rad).astype(np.float32)
        self._ctrl_high_deg = np.rad2deg(self._ctrl_high_rad).astype(np.float32)

        # Action Space theo Do (DEGREES)
        self.action_space = spaces.Box(
            low=self._ctrl_low_deg,
            high=self._ctrl_high_deg,
            dtype=np.float32,
        )

        # Observation Space theo chuan LeRobot
        # agent_pos: 6 truc khop theo DO (degrees)
        # pixels: top (480x640x3) va front (480x640x3)
        self.observation_space = spaces.Dict(
            {
                "agent_pos": spaces.Box(
                    low=-180.0,
                    high=180.0,
                    shape=(self.model.nu,),
                    dtype=np.float32,
                ),
                "pixels": spaces.Dict(
                    {
                        "top": spaces.Box(
                            low=0,
                            high=255,
                            shape=(self.image_height, self.image_width, 3),
                            dtype=np.uint8,
                        ),
                        "front": spaces.Box(
                            low=0,
                            high=255,
                            shape=(self.image_height, self.image_width, 3),
                            dtype=np.uint8,
                        ),
                    }
                ),
            }
        )

        self._step_count = 0
        self._consecutive_success_steps = 0
        self._last_placements: dict[str, dict[str, Any]] = {}
        self._last_rendered_frame: np.ndarray | None = None

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        super().reset(seed=seed)
        episode_index = int((options or {}).get("episode_index", 0))

        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

        # Ap dung co che phan bo ngau nhien 3 khoi hop theo o luoi
        self._apply_stack_random_placements(episode_index)
        mujoco.mj_forward(self.model, self.data)

        self._step_count = 0
        self._consecutive_success_steps = 0

        obs = self._get_obs()
        info = {
            "is_success": False,
            "episode_index": episode_index,
            "stack_random_placements": dict(self._last_placements),
        }
        return obs, info

    def step(
        self, action: np.ndarray
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        self._step_count += 1
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.shape != (self.model.nu,):
            raise ValueError(
                f"Action shape mismatch: expected ({self.model.nu},), got {action.shape}."
            )

        # Chuyen action tu do (degrees) sang radian cho MuJoCo
        action_deg = np.clip(action, self._ctrl_low_deg, self._ctrl_high_deg)
        action_rad = np.deg2rad(action_deg).astype(np.float64)

        self.data.ctrl[:] = action_rad
        mujoco.mj_step(self.model, self.data, nstep=self.frame_skip)

        obs = self._get_obs()

        # Kiem tra tieu chi thanh cong xep chong 3 khoi hop
        is_stacked = self._check_stack_success()
        if is_stacked:
            self._consecutive_success_steps += 1
        else:
            self._consecutive_success_steps = 0

        # Can duy tri thap on dinh it nhat 15 buoc (0.5 giay o 30 FPS)
        is_success = self._consecutive_success_steps >= 15
        reward = 1.0 if is_success else 0.0

        terminated = is_success
        truncated = self._step_count >= self._max_episode_steps

        info = {
            "is_success": is_success,
            "is_stacked": is_stacked,
            "sim_time": float(self.data.time),
            "step_count": self._step_count,
            "stack_random_placements": dict(self._last_placements),
        }
        return obs, reward, terminated, truncated, info

    def render(self) -> np.ndarray:
        top_img = self._render_camera(self.top_camera_name)
        front_img = self._render_camera(self.front_camera_name)

        if not self.render_composite:
            return top_img

        top_view = top_img.copy()
        if self.render_grid_overlay:
            # Ve luoi len camera top
            top_bgr = cv2.cvtColor(top_view, cv2.COLOR_RGB2BGR)
            draw_grid_5x7(top_bgr, rows=5, cols=7, labels=True)
            top_view = cv2.cvtColor(top_bgr, cv2.COLOR_BGR2RGB)

        # Ghep song song 2 camera: Left = Top (co luoi), Right = Front
        composite = np.hstack([top_view, front_img])
        self._last_rendered_frame = composite
        return composite

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()

    def _get_obs(self) -> dict[str, Any]:
        # agent_pos tinh theo DO (degrees) de hop voi ACT policy da train tren stack_3_box_30x3030
        agent_pos_rad = self.data.qpos[: self.model.nu].copy().astype(np.float32)
        agent_pos_deg = np.rad2deg(agent_pos_rad).astype(np.float32)

        top_img = self._render_camera(self.top_camera_name)
        front_img = self._render_camera(self.front_camera_name)

        return {
            "agent_pos": agent_pos_deg,
            "pixels": {
                "top": top_img,
                "front": front_img,
            },
        }

    def _render_camera(self, camera_name: str) -> np.ndarray:
        self._renderer.update_scene(self.data, camera=camera_name)
        rgb = self._renderer.render()
        if (
            self.front_camera_brightness_gain != 1.0
            and camera_name == self.front_camera_name
        ):
            rgb = np.clip(
                rgb.astype(np.float32) * self.front_camera_brightness_gain,
                0,
                255,
            ).astype(np.uint8)
        return rgb

    def _apply_stack_random_placements(self, episode_index: int) -> None:
        """Phan bo 3 khoi hop theo o luoi rieng biet (chuan data collection)."""
        excluded_set = set(DEFAULT_STACK_EXCLUDED_CELLS)
        cell_centers = {
            k: v
            for k, v in DEFAULT_STACK_CELL_CENTERS.items()
            if k not in excluded_set and not k.startswith("e")
        }
        left_cells = [c for c in DEFAULT_STACK_LEFT_CELLS if c in cell_centers]
        right_cells = [c for c in DEFAULT_STACK_RIGHT_CELLS if c in cell_centers]
        all_cells = list(cell_centers.keys())

        c_left = str(self.np_random.choice(left_cells))
        c_right = str(self.np_random.choice(right_cells))
        p_left = np.array(cell_centers[c_left], dtype=np.float64)
        p_right = np.array(cell_centers[c_right], dtype=np.float64)

        candidates = [
            c
            for c in all_cells
            if c not in (c_left, c_right)
            and np.linalg.norm(np.array(cell_centers[c], dtype=np.float64) - p_left)
            >= self.min_cell_distance
            and np.linalg.norm(np.array(cell_centers[c], dtype=np.float64) - p_right)
            >= self.min_cell_distance
        ]
        if not candidates:
            candidates = [c for c in all_cells if c not in (c_left, c_right)]
        c_third = str(self.np_random.choice(candidates))

        chosen_cells = [c_left, c_right, c_third]
        self.np_random.shuffle(chosen_cells)

        body_bottom = "box_white" if mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "box_white") >= 0 else "box_black"
        bodies = [body_bottom, "box_green", "box_blue"]
        self._last_placements = {}

        for body_name, cell_name in zip(bodies, chosen_cells):
            base_x, base_y = cell_centers[cell_name]
            jx = float(self.np_random.uniform(-self.cell_jitter, self.cell_jitter))
            jy = float(self.np_random.uniform(-self.cell_jitter, self.cell_jitter))
            pos = np.array([base_x + jx, base_y + jy, 0.015], dtype=np.float64)
            yaw = float(self.np_random.uniform(-np.pi, np.pi))

            self._place_body(body_name, pos, quat=_yaw_to_quat(yaw))
            self._last_placements[body_name] = {
                "body": body_name,
                "episode_index": episode_index,
                "cell": cell_name,
                "x": float(pos[0]),
                "y": float(pos[1]),
                "z": float(pos[2]),
                "yaw": yaw,
            }

    def _place_body(
        self, body_name: str, pos: np.ndarray, quat: np.ndarray | None = None
    ) -> None:
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id < 0:
            return
        jnt_id = self.model.body_jntadr[body_id]
        if jnt_id < 0:
            return
        qpos_adr = self.model.jnt_qposadr[jnt_id]
        self.data.qpos[qpos_adr : qpos_adr + 3] = pos
        if quat is not None:
            self.data.qpos[qpos_adr + 3 : qpos_adr + 7] = quat

    def _check_stack_success(self) -> bool:
        """Kiem tra xem 3 khoi hop da duoc xep chong dung vi tri va dung thu tu chua:
        - Day: box_white (hoac box_black) tren vung stack_target_zone [0.264, -0.0175] (z ~ 0.015m)
        - Giua: box_green tren dinh khoi day (z ~ 0.045m)
        - Tren: box_blue tren dinh box_green (z ~ 0.075m)
        """
        id_bottom = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "box_white")
        if id_bottom < 0:
            id_bottom = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "box_black")
        id_green = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "box_green")
        id_blue = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "box_blue")

        if id_bottom < 0 or id_green < 0 or id_blue < 0:
            return False

        pos_bottom = self.data.xpos[id_bottom]
        pos_green = self.data.xpos[id_green]
        pos_blue = self.data.xpos[id_blue]

        # 1. Khối Đáy (Trắng / Đen) ở trong ô đích stack_target_zone và nằm trên mặt bàn
        dist_target = math.hypot(
            pos_bottom[0] - DEFAULT_TARGET_ZONE_POS[0],
            pos_bottom[1] - DEFAULT_TARGET_ZONE_POS[1],
        )
        if dist_target > 0.045 or pos_bottom[2] > 0.025:
            return False

        # 2. Khối Xanh lá nằm trên khối Đáy (khoảng cách tâm ngang < 25mm, chiều cao z ~ 0.045m)
        dist_bg = math.hypot(pos_green[0] - pos_bottom[0], pos_green[1] - pos_bottom[1])
        if dist_bg > 0.025 or not (0.035 <= pos_green[2] <= 0.055):
            return False

        # 3. Khối Xanh biển nằm trên khối Xanh lá (khoảng cách tâm ngang < 25mm, chiều cao z ~ 0.075m)
        dist_gb = math.hypot(pos_blue[0] - pos_green[0], pos_blue[1] - pos_green[1])
        if dist_gb > 0.025 or not (0.065 <= pos_blue[2] <= 0.088):
            return False

        return True


def create_so101_stack_3_blocks_envs(
    cfg: Any,
    n_envs: int,
    use_async_envs: bool = False,
) -> dict[str, dict[int, gym.vector.VectorEnv]]:
    """Ham tao VectorEnv cho LeRobot eval hoac training."""
    from lerobot.envs.configs import _make_vec_env_cls
    from lerobot.envs.utils import freeze_after_episode_end

    env_cls = _make_vec_env_cls(use_async_envs, n_envs)

    def _make_one():
        env = SO101Stack3BlocksEnv(
            xml_path=getattr(cfg, "xml_path", None),
            max_episode_steps=getattr(cfg, "episode_length", 1800),
            render_mode=getattr(cfg, "render_mode", "rgb_array"),
            image_height=getattr(cfg, "observation_height", 480),
            image_width=getattr(cfg, "observation_width", 640),
            fps=getattr(cfg, "fps", 30),
        )
        return env

    env_fn = freeze_after_episode_end(_make_one)
    vec = env_cls([env_fn for _ in range(n_envs)])
    return {getattr(cfg, "type", "so101_stack_3_blocks"): {0: vec}}


# Dang ky voi Gymnasium
try:
    gym.register(
        id="so101/SO101Stack3Blocks-v0",
        entry_point="lerobot.envs.so101:SO101Stack3BlocksEnv",
        max_episode_steps=1800,
    )
except Exception:
    pass

try:
    gym.register(
        id="SO101Stack3Blocks-v0",
        entry_point="lerobot.envs.so101:SO101Stack3BlocksEnv",
        max_episode_steps=1800,
    )
except Exception:
    pass
