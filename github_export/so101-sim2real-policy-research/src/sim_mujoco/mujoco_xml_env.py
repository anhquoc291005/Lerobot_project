from __future__ import annotations

from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

try:
    from .top_cam_grid_5x7 import GRID_CELLS, GridRegion
except ImportError:
    from top_cam_grid_5x7 import GRID_CELLS, GridRegion


@dataclass
class GridObjectPlacementConfig:
    cells: tuple[str, ...]
    z: float | None = None

    @classmethod
    def from_dict(cls, raw: str | list[str] | dict[str, Any]) -> "GridObjectPlacementConfig":
        if isinstance(raw, str):
            return cls(cells=(raw,))
        if isinstance(raw, list):
            return cls(cells=tuple(str(cell) for cell in raw))

        cell = raw.get("cell")
        cells = raw.get("cells")
        if cell is not None and cells is not None:
            raise ValueError("Use either `cell` or `cells` for a grid object placement, not both.")
        if cell is not None:
            resolved_cells = (str(cell),)
        elif cells is not None:
            resolved_cells = tuple(str(cell) for cell in cells)
        else:
            raise ValueError("Grid object placement requires `cell` or `cells`.")

        if not resolved_cells:
            raise ValueError("Grid object placement `cells` must not be empty.")
        z = raw.get("z")
        return cls(cells=resolved_cells, z=float(z) if z is not None else None)


@dataclass
class GridPlacementConfig:
    rows: int = 5
    cols: int = 7
    plane_z: float = 0.0
    camera_name: str | None = None
    objects: dict[str, GridObjectPlacementConfig] | None = None

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> "GridPlacementConfig | None":
        if not raw:
            return None

        rows = int(raw.get("rows", 5))
        cols = int(raw.get("cols", 7))
        if rows <= 0 or cols <= 0:
            raise ValueError("env.grid_placements rows and cols must be > 0.")

        raw_objects = raw.get("objects", {})
        if not isinstance(raw_objects, dict) or not raw_objects:
            raise ValueError("env.grid_placements.objects must be a non-empty object.")

        return cls(
            rows=rows,
            cols=cols,
            plane_z=float(raw.get("plane_z", 0.0)),
            camera_name=raw.get("camera_name"),
            objects={
                str(body_name): GridObjectPlacementConfig.from_dict(body_raw)
                for body_name, body_raw in raw_objects.items()
            },
        )


@dataclass
class MatrixRandomPlacementConfig:
    table_path: str = "src/sim_mujoco/matrix_5x7_selection_table.html"
    red_box_body: str = "pick_box"
    container_body: str = "container_box"
    red_box_z: float = 0.017
    container_z: float = 0.0
    red_box_region_margin: float = 0.0
    red_box_reach_center_x: float = 0.015
    red_box_reach_center_y: float = 0.0
    red_box_reach_radius: float = 0.44
    red_box_max_sample_attempts: int = 100
    red_box_random_yaw: bool = True

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | bool | None) -> "MatrixRandomPlacementConfig | None":
        if not raw:
            return None
        if isinstance(raw, bool):
            return cls()

        return cls(
            table_path=str(raw.get("table_path", cls.table_path)),
            red_box_body=str(raw.get("red_box_body", cls.red_box_body)),
            container_body=str(raw.get("container_body", cls.container_body)),
            red_box_z=float(raw.get("red_box_z", cls.red_box_z)),
            container_z=float(raw.get("container_z", cls.container_z)),
            red_box_region_margin=float(
                raw.get("red_box_region_margin", cls.red_box_region_margin)
            ),
            red_box_reach_center_x=float(
                raw.get("red_box_reach_center_x", cls.red_box_reach_center_x)
            ),
            red_box_reach_center_y=float(
                raw.get("red_box_reach_center_y", cls.red_box_reach_center_y)
            ),
            red_box_reach_radius=float(
                raw.get("red_box_reach_radius", cls.red_box_reach_radius)
            ),
            red_box_max_sample_attempts=int(
                raw.get("red_box_max_sample_attempts", cls.red_box_max_sample_attempts)
            ),
            red_box_random_yaw=bool(raw.get("red_box_random_yaw", cls.red_box_random_yaw)),
        )


DEFAULT_STACK_EXCLUDED_CELLS: tuple[str, ...] = (
    "e1", "e2", "e3", "e4", "e5", "e6", "e7",
    "c1", "c7",
    "d1", "d2", "d6", "d7",
)

# Toạ độ tâm chuẩn xác của các ô lưới trên mặt sàn (z=0.015m)
# Được căn chỉnh đồng đều với 4 cạnh biên của ô trên camera top (5x7)
# Đã loại bỏ hàng E, c1, c7, d1, d2, d6, d7 theo yêu cầu workspace thao tác thực tế
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


@dataclass
class StackRandomPlacementConfig:
    enabled: bool = True
    mode: str = "cell"  # "cell" (random theo o luoi rieng biet) hoac "uniform"
    bodies: tuple[str, ...] = ("box_white", "box_green", "box_blue")
    x_range: tuple[float, float] = (0.17, 0.28)
    y_range: tuple[float, float] = (-0.14, 0.14)
    cell_jitter: float = 0.008  # rung nhe quanh tam o de tranh dung le / duong bien
    min_cell_distance: float = 0.12  # khoang cach toi thieu giua cac o duoc chon
    z: float = 0.015
    min_distance: float = 0.05
    random_yaw: bool = True
    target_zone_pos: tuple[float, float] | None = (0.264, -0.0175)
    min_dist_to_target: float = 0.065
    max_attempts: int = 500
    excluded_cells: tuple[str, ...] = DEFAULT_STACK_EXCLUDED_CELLS

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | bool | None) -> "StackRandomPlacementConfig | None":
        if not raw:
            return None
        if isinstance(raw, bool):
            return cls(enabled=raw) if raw else None

        enabled = bool(raw.get("enabled", True))
        if not enabled:
            return None

        mode = str(raw.get("mode", "cell"))
        bodies_raw = raw.get("bodies", ("box_white", "box_green", "box_blue"))
        bodies = tuple(str(b) for b in bodies_raw)

        x_raw = raw.get("x_range", (0.17, 0.28))
        x_range = (float(x_raw[0]), float(x_raw[1]))

        y_raw = raw.get("y_range", (-0.14, 0.14))
        y_range = (float(y_raw[0]), float(y_raw[1]))

        target_zone_raw = raw.get("target_zone_pos")
        target_zone_pos = (
            (float(target_zone_raw[0]), float(target_zone_raw[1]))
            if target_zone_raw is not None
            else (0.264, -0.0175)
        )

        excluded_raw = raw.get("excluded_cells", DEFAULT_STACK_EXCLUDED_CELLS)
        if isinstance(excluded_raw, (list, tuple)):
            excluded_cells = tuple(str(c).strip().lower() for c in excluded_raw)
        else:
            excluded_cells = DEFAULT_STACK_EXCLUDED_CELLS

        return cls(
            enabled=enabled,
            mode=mode,
            bodies=bodies,
            x_range=x_range,
            y_range=y_range,
            cell_jitter=float(raw.get("cell_jitter", 0.008)),
            min_cell_distance=float(raw.get("min_cell_distance", 0.12)),
            z=float(raw.get("z", 0.015)),
            min_distance=float(raw.get("min_distance", 0.065)),
            random_yaw=bool(raw.get("random_yaw", True)),
            target_zone_pos=target_zone_pos,
            min_dist_to_target=float(raw.get("min_dist_to_target", 0.065)),
            max_attempts=int(raw.get("max_attempts", 500)),
            excluded_cells=excluded_cells,
        )


@dataclass
class MujocoXmlEnvConfig:
    xml_path: str
    top_camera_name: str = "cam_top"
    gripper_camera_name: str = "gripper_cam"
    image_height: int = 480
    image_width: int = 640
    frame_skip: int = 1
    state_source: str = "qpos"
    clip_actions_to_ctrlrange: bool = True
    qpos_reset_noise_std: float = 0.0
    auto_resize_offscreen_framebuffer: bool = True
    grid_placements: GridPlacementConfig | None = None
    matrix_random_placements: MatrixRandomPlacementConfig | None = None
    stack_random_placements: StackRandomPlacementConfig | None = None
    front_camera_brightness_gain: float = 1.2

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "MujocoXmlEnvConfig":
        xml_path = str(raw.get("xml_path", "")).strip()
        if not xml_path:
            raise ValueError("env.xml_path is required when env.backend='mujoco_xml'.")

        state_source = str(raw.get("state_source", "qpos")).strip().lower()
        if state_source not in {"qpos", "qpos_qvel"}:
            raise ValueError(
                f"Unsupported env.state_source='{state_source}'. Use 'qpos' or 'qpos_qvel'."
            )

        return cls(
            xml_path=xml_path,
            top_camera_name=str(raw.get("top_camera_name", "cam_top")),
            gripper_camera_name=str(raw.get("gripper_camera_name", "gripper_cam")),
            image_height=int(raw.get("image_height", 480)),
            image_width=int(raw.get("image_width", 640)),
            frame_skip=int(raw.get("frame_skip", 1)),
            state_source=state_source,
            clip_actions_to_ctrlrange=bool(raw.get("clip_actions_to_ctrlrange", True)),
            qpos_reset_noise_std=float(raw.get("qpos_reset_noise_std", 0.0)),
            auto_resize_offscreen_framebuffer=bool(
                raw.get("auto_resize_offscreen_framebuffer", True)
            ),
            grid_placements=GridPlacementConfig.from_dict(raw.get("grid_placements")),
            matrix_random_placements=MatrixRandomPlacementConfig.from_dict(
                raw.get("matrix_random_placements")
            ),
            stack_random_placements=StackRandomPlacementConfig.from_dict(
                raw.get("stack_random_placements")
            ),
            front_camera_brightness_gain=float(
                raw.get("front_camera_brightness_gain", 1.2)
            ),
        )


class MujocoXmlEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, cfg: MujocoXmlEnvConfig):
        super().__init__()
        self.cfg = cfg
        xml_path = Path(cfg.xml_path)
        if not xml_path.exists():
            raise FileNotFoundError(f"MuJoCo xml not found: {xml_path}")

        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        if cfg.auto_resize_offscreen_framebuffer:
            self.model.vis.global_.offwidth = max(
                int(self.model.vis.global_.offwidth),
                int(cfg.image_width),
            )
            self.model.vis.global_.offheight = max(
                int(self.model.vis.global_.offheight),
                int(cfg.image_height),
            )
        self.data = mujoco.MjData(self.model)
        self._renderer = mujoco.Renderer(
            self.model,
            height=cfg.image_height,
            width=cfg.image_width,
        )

        self._action_low, self._action_high = _build_action_bounds(self.model)
        self._last_grid_placements: dict[str, dict[str, float | str]] = {}
        self._last_matrix_random_placements: dict[str, dict[str, float | int | str]] = {}
        self._last_stack_random_placements: dict[str, dict[str, float | str]] = {}
        self._matrix_selection_plan = _load_matrix_selection_plan(
            Path(cfg.matrix_random_placements.table_path)
            if cfg.matrix_random_placements is not None
            else None
        )
        if cfg.matrix_random_placements is not None:
            _validate_reachable_red_cells(
                selection_plan=self._matrix_selection_plan,
                reach_center=(
                    cfg.matrix_random_placements.red_box_reach_center_x,
                    cfg.matrix_random_placements.red_box_reach_center_y,
                ),
                reach_radius=cfg.matrix_random_placements.red_box_reach_radius,
            )
        self.action_space = spaces.Box(
            low=self._action_low,
            high=self._action_high,
            dtype=np.float32,
        )

        state_dim = self.model.nu if cfg.state_source == "qpos" else self.model.nu * 2
        self.observation_space = spaces.Dict(
            {
                "agent_pos": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(state_dim,),
                    dtype=np.float32,
                ),
                "pixels": spaces.Dict(
                    {
                        "top": spaces.Box(
                            low=0,
                            high=255,
                            shape=(cfg.image_height, cfg.image_width, 3),
                            dtype=np.uint8,
                        ),
                        "gripper": spaces.Box(
                            low=0,
                            high=255,
                            shape=(cfg.image_height, cfg.image_width, 3),
                            dtype=np.uint8,
                        ),
                    }
                ),
            }
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        super().reset(seed=seed)
        episode_index = int((options or {}).get("episode_index", 0))
        mujoco.mj_resetData(self.model, self.data)

        if self.cfg.qpos_reset_noise_std > 0:
            noise = self.np_random.normal(
                0.0,
                self.cfg.qpos_reset_noise_std,
                size=self.model.nq,
            )
            self.data.qpos[:] = self.data.qpos[:] + noise.astype(np.float64)

        mujoco.mj_forward(self.model, self.data)
        self._apply_matrix_random_placements(episode_index)
        self._apply_grid_placements(episode_index)
        self._apply_stack_random_placements(episode_index)
        mujoco.mj_forward(self.model, self.data)
        return self._get_obs(), {
            "grid_placements": dict(self._last_grid_placements),
            "matrix_random_placements": dict(self._last_matrix_random_placements),
            "stack_random_placements": dict(self._last_stack_random_placements),
        }

    def step(
        self, action: np.ndarray
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        expected_dim = self.model.nu
        if action.shape != (expected_dim,):
            raise ValueError(
                f"Action shape mismatch: expected {(expected_dim,)}, got {tuple(action.shape)}."
            )

        if self.cfg.clip_actions_to_ctrlrange:
            action = np.clip(action, self._action_low, self._action_high)

        self.data.ctrl[:] = action.astype(np.float64)
        mujoco.mj_step(self.model, self.data, nstep=max(self.cfg.frame_skip, 1))

        obs = self._get_obs()
        terminated = False
        truncated = False
        reward = 0.0
        info = {
            "sim_time": float(self.data.time),
            "grid_placements": dict(self._last_grid_placements),
            "matrix_random_placements": dict(self._last_matrix_random_placements),
            "stack_random_placements": dict(self._last_stack_random_placements),
        }
        return obs, reward, terminated, truncated, info

    def close(self) -> None:
        self._renderer.close()

    def _get_obs(self) -> dict[str, Any]:
        if self.cfg.state_source == "qpos":
            state = self.data.qpos[: self.model.nu].copy().astype(np.float32)
        else:
            state = np.concatenate(
                [self.data.qpos[: self.model.nu], self.data.qvel[: self.model.nu]],
                axis=0,
            ).astype(np.float32)

        top = self._render_camera(self.cfg.top_camera_name)
        gripper = self._render_camera(self.cfg.gripper_camera_name)
        return {
            "agent_pos": state,
            "pixels": {
                "top": top,
                "gripper": gripper,
            },
        }

    def _render_camera(self, camera_name: str) -> np.ndarray:
        self._renderer.update_scene(self.data, camera=camera_name)
        rgb = self._renderer.render()
        if (
            self.cfg.front_camera_brightness_gain != 1.0
            and (
                camera_name == self.cfg.gripper_camera_name
                or "front" in camera_name.lower()
            )
        ):
            rgb = np.clip(
                rgb.astype(np.float32) * self.cfg.front_camera_brightness_gain,
                0,
                255,
            ).astype(np.uint8)
        return np.asarray(rgb, dtype=np.uint8)

    def _apply_matrix_random_placements(self, episode_index: int) -> None:
        matrix_cfg = self.cfg.matrix_random_placements
        self._last_matrix_random_placements = {}
        if matrix_cfg is None:
            return
        if not self._matrix_selection_plan:
            raise ValueError("matrix_random_placements table has no valid rows.")

        row_index = (episode_index // 4) % len(self._matrix_selection_plan)
        red_cell, container_cells = self._matrix_selection_plan[row_index]
        container_index = episode_index % len(container_cells)
        container_cell = container_cells[container_index]

        red_region = GRID_CELLS[red_cell]
        red_xy = _sample_region_xy(
            rng=self.np_random,
            region=red_region,
            margin=matrix_cfg.red_box_region_margin,
            reach_center=(
                matrix_cfg.red_box_reach_center_x,
                matrix_cfg.red_box_reach_center_y,
            ),
            reach_radius=matrix_cfg.red_box_reach_radius,
            max_attempts=matrix_cfg.red_box_max_sample_attempts,
        )
        red_pos = self._body_position_with_xy(
            body_name=matrix_cfg.red_box_body,
            xy=red_xy,
            z=matrix_cfg.red_box_z,
        )
        red_yaw = (
            float(self.np_random.uniform(-np.pi, np.pi))
            if matrix_cfg.red_box_random_yaw
            else 0.0
        )
        self._place_body(matrix_cfg.red_box_body, red_pos, quat=_yaw_to_quat(red_yaw))

        container_region = GRID_CELLS[container_cell]
        container_xy = np.asarray(container_region.center[:2], dtype=np.float64)
        container_pos = self._body_position_with_xy(
            body_name=matrix_cfg.container_body,
            xy=container_xy,
            z=matrix_cfg.container_z,
        )
        self._place_body(matrix_cfg.container_body, container_pos)

        self._last_matrix_random_placements = {
            matrix_cfg.red_box_body: {
                "cell": red_cell,
                "episode_index": episode_index,
                "table_row_index": row_index,
                "table_row_number": row_index + 1,
                "x": float(red_pos[0]),
                "y": float(red_pos[1]),
                "z": float(red_pos[2]),
                "yaw": red_yaw,
            },
            matrix_cfg.container_body: {
                "cell": container_cell,
                "source_row": red_cell,
                "episode_index": episode_index,
                "container_index": container_index,
                "container_number": container_index + 1,
                "x": float(container_pos[0]),
                "y": float(container_pos[1]),
                "z": float(container_pos[2]),
            },
        }

    def _apply_grid_placements(self, episode_index: int) -> None:
        grid_cfg = self.cfg.grid_placements
        self._last_grid_placements = {}
        if grid_cfg is None or not grid_cfg.objects:
            return

        camera_name = grid_cfg.camera_name or self.cfg.top_camera_name
        for body_name, placement in grid_cfg.objects.items():
            cell = placement.cells[episode_index % len(placement.cells)]
            xy = self._grid_cell_to_floor_xy(
                cell=cell,
                camera_name=camera_name,
                rows=grid_cfg.rows,
                cols=grid_cfg.cols,
                plane_z=grid_cfg.plane_z,
            )
            pos = self._body_position_with_xy(body_name, xy, placement.z)
            self._place_body(body_name, pos)
            self._last_grid_placements[body_name] = {
                "cell": cell,
                "x": float(pos[0]),
                "y": float(pos[1]),
                "z": float(pos[2]),
            }

    def _apply_stack_random_placements(self, episode_index: int) -> None:
        cfg = self.cfg.stack_random_placements
        self._last_stack_random_placements = {}
        if cfg is None or not cfg.enabled or not cfg.bodies:
            return

        if cfg.target_zone_pos is not None:
            site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "stack_target_zone")
            if site_id >= 0:
                self.model.site_pos[site_id][:2] = np.array(cfg.target_zone_pos, dtype=np.float64)

        if cfg.mode == "cell":
            # Phân phối theo ô lưới (Grid-cell based):
            # 1. Mỗi khối hộp nằm gọn bên trong 1 ô riêng biệt
            # 2. Hạn chế tối đa nằm đè lên vạch biên (tâm ô chuẩn xác + jitter nhỏ)
            # 3. Phân phối rộng ra 2 bên và phía trước thay vì cụm sát nhau
            excluded_set = set(cfg.excluded_cells)
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
                >= cfg.min_cell_distance
                and np.linalg.norm(np.array(cell_centers[c], dtype=np.float64) - p_right)
                >= cfg.min_cell_distance
            ]
            if not candidates:
                candidates = [c for c in all_cells if c not in (c_left, c_right)]
            c_third = str(self.np_random.choice(candidates))

            chosen_cells = [c_left, c_right, c_third]
            self.np_random.shuffle(chosen_cells)

            while len(chosen_cells) < len(cfg.bodies):
                rem = [c for c in all_cells if c not in chosen_cells]
                if not rem:
                    break
                chosen_cells.append(str(self.np_random.choice(rem)))

            placed_positions: list[np.ndarray] = []
            for body_name, cell_name in zip(cfg.bodies, chosen_cells):
                base_x, base_y = cell_centers[cell_name]
                jx = float(self.np_random.uniform(-cfg.cell_jitter, cfg.cell_jitter))
                jy = float(self.np_random.uniform(-cfg.cell_jitter, cfg.cell_jitter))
                candidate_xy = np.array([base_x + jx, base_y + jy], dtype=np.float64)

                yaw = (
                    float(self.np_random.uniform(-np.pi, np.pi))
                    if cfg.random_yaw
                    else 0.0
                )
                pos = self._body_position_with_xy(body_name, candidate_xy, cfg.z)
                self._place_body(body_name, pos, quat=_yaw_to_quat(yaw))
                placed_positions.append(pos.copy())

                self._last_stack_random_placements[body_name] = {
                    "body": body_name,
                    "episode_index": episode_index,
                    "cell": cell_name,
                    "x": float(pos[0]),
                    "y": float(pos[1]),
                    "z": float(pos[2]),
                    "yaw": yaw,
                }
            return

        placed_positions: list[np.ndarray] = []
        for body_name in cfg.bodies:
            candidate_xy: np.ndarray | None = None
            for _ in range(cfg.max_attempts):
                cx = float(self.np_random.uniform(cfg.x_range[0], cfg.x_range[1]))
                cy = float(self.np_random.uniform(cfg.y_range[0], cfg.y_range[1]))
                c_xy = np.array([cx, cy], dtype=np.float64)

                if cfg.target_zone_pos is not None and cfg.min_dist_to_target > 0:
                    tz = np.array(cfg.target_zone_pos, dtype=np.float64)
                    if np.linalg.norm(c_xy - tz) < cfg.min_dist_to_target:
                        continue

                if any(
                    np.linalg.norm(c_xy - prev_pos[:2]) < cfg.min_distance
                    for prev_pos in placed_positions
                ):
                    continue

                candidate_xy = c_xy
                break

            if candidate_xy is None:
                cx = float(self.np_random.uniform(cfg.x_range[0], cfg.x_range[1]))
                cy = float(self.np_random.uniform(cfg.y_range[0], cfg.y_range[1]))
                candidate_xy = np.array([cx, cy], dtype=np.float64)

            yaw = (
                float(self.np_random.uniform(-np.pi, np.pi))
                if cfg.random_yaw
                else 0.0
            )
            pos = self._body_position_with_xy(body_name, candidate_xy, cfg.z)
            self._place_body(body_name, pos, quat=_yaw_to_quat(yaw))
            placed_positions.append(pos.copy())

            self._last_stack_random_placements[body_name] = {
                "body": body_name,
                "episode_index": episode_index,
                "cell": "-",
                "x": float(pos[0]),
                "y": float(pos[1]),
                "z": float(pos[2]),
                "yaw": yaw,
            }

    def _grid_cell_to_floor_xy(
        self,
        cell: str,
        camera_name: str,
        rows: int,
        cols: int,
        plane_z: float,
    ) -> np.ndarray:
        row, col = _parse_grid_cell(cell, rows, cols)
        camera_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
        if camera_id < 0:
            raise ValueError(f"Camera '{camera_name}' was not found in MuJoCo model.")

        cam_pos = self.data.cam_xpos[camera_id]
        cam_xmat = self.data.cam_xmat[camera_id].reshape(3, 3)
        x_axis = cam_xmat[0]
        y_axis = cam_xmat[1]
        z_axis = cam_xmat[2]

        fovy = np.deg2rad(float(self.model.cam_fovy[camera_id]))
        aspect = float(self.cfg.image_width) / float(self.cfg.image_height)
        half_height = np.tan(fovy / 2.0)
        half_width = half_height * aspect

        u = (((col + 0.5) / cols) * 2.0 - 1.0) * half_width
        v = (1.0 - ((row + 0.5) / rows) * 2.0) * half_height
        ray = -z_axis + u * x_axis + v * y_axis

        if abs(float(ray[2])) < 1e-9:
            raise ValueError(f"Camera ray for cell '{cell}' is parallel to plane z={plane_z}.")

        t = (plane_z - float(cam_pos[2])) / float(ray[2])
        point = cam_pos + t * ray
        return point[:2].astype(np.float64)

    def _body_position_with_xy(
        self,
        body_name: str,
        xy: np.ndarray,
        z: float | None,
    ) -> np.ndarray:
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id < 0:
            raise ValueError(f"Body '{body_name}' was not found in MuJoCo model.")

        pos = self._current_body_position(body_id)
        pos[:2] = xy
        if z is not None:
            pos[2] = z
        return pos

    def _current_body_position(self, body_id: int) -> np.ndarray:
        free_joint_id = _free_joint_id_for_body(self.model, body_id)
        if free_joint_id is None:
            return self.model.body_pos[body_id].copy()

        qpos_adr = self.model.jnt_qposadr[free_joint_id]
        return self.data.qpos[qpos_adr : qpos_adr + 3].copy()

    def _place_body(
        self,
        body_name: str,
        pos: np.ndarray,
        quat: np.ndarray | None = None,
    ) -> None:
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        free_joint_id = _free_joint_id_for_body(self.model, body_id)
        if free_joint_id is None:
            self.model.body_pos[body_id, :] = pos
            return

        qpos_adr = self.model.jnt_qposadr[free_joint_id]
        qvel_adr = self.model.jnt_dofadr[free_joint_id]
        self.data.qpos[qpos_adr : qpos_adr + 3] = pos
        if quat is None:
            quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self.data.qpos[qpos_adr + 3 : qpos_adr + 7] = quat
        self.data.qvel[qvel_adr : qvel_adr + 6] = 0.0


def _build_action_bounds(model: mujoco.MjModel) -> tuple[np.ndarray, np.ndarray]:
    if model.nu <= 0:
        raise ValueError("Model has zero actuators (nu=0). Cannot build an action space.")

    ctrlrange = np.asarray(model.actuator_ctrlrange, dtype=np.float32)
    ctrllimited = np.asarray(model.actuator_ctrllimited, dtype=np.int32).astype(bool)

    low = np.full((model.nu,), -1.0, dtype=np.float32)
    high = np.full((model.nu,), 1.0, dtype=np.float32)

    if ctrlrange.shape == (model.nu, 2):
        low = np.where(ctrllimited, ctrlrange[:, 0], low).astype(np.float32)
        high = np.where(ctrllimited, ctrlrange[:, 1], high).astype(np.float32)

    eps = np.float32(1e-6)
    invalid = ~np.isfinite(low) | ~np.isfinite(high) | (high <= low)
    low = np.where(invalid, -1.0, low)
    high = np.where(invalid, 1.0, high + eps)
    return low, high


class _SelectionTableParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.rows: list[tuple[str, ...]] = []
        self._in_selection_table = False
        self._current_row: list[str] | None = None
        self._current_cell: list[str] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attrs_dict = dict(attrs)
        if tag == "table" and attrs_dict.get("id") == "selection-table":
            self._in_selection_table = True
            return
        if not self._in_selection_table:
            return
        if tag == "tr":
            self._current_row = []
        elif tag == "td" and self._current_row is not None:
            self._current_cell = []

    def handle_data(self, data: str) -> None:
        if self._current_cell is not None:
            self._current_cell.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == "td" and self._current_cell is not None and self._current_row is not None:
            cell = "".join(self._current_cell).strip().lower()
            if cell:
                self._current_row.append(cell)
            self._current_cell = None
            return
        if tag == "tr" and self._current_row is not None:
            if self._current_row:
                self.rows.append(tuple(self._current_row))
            self._current_row = None
            return
        if tag == "table" and self._in_selection_table:
            self._in_selection_table = False


def _load_matrix_selection_plan(
    table_path: Path | None,
) -> list[tuple[str, tuple[str, ...]]]:
    if table_path is None:
        return []
    if not table_path.exists():
        raise FileNotFoundError(f"Matrix selection table not found: {table_path}")

    parser = _SelectionTableParser()
    parser.feed(table_path.read_text(encoding="utf-8"))

    plan: list[tuple[str, tuple[str, ...]]] = []
    for row in parser.rows:
        if len(row) < 5:
            continue
        red_cell = row[0]
        container_cells = tuple(row[1:5])
        _validate_matrix_cell(red_cell, table_path)
        for cell in container_cells:
            _validate_matrix_cell(cell, table_path)
        plan.append((red_cell, container_cells))

    if not plan:
        raise ValueError(f"No placement rows found in matrix selection table: {table_path}")
    return plan


def _validate_matrix_cell(cell: str, table_path: Path) -> None:
    if cell not in GRID_CELLS:
        raise ValueError(f"Cell '{cell}' in {table_path} is not part of the 5x7 grid.")


def _validate_reachable_red_cells(
    selection_plan: list[tuple[str, tuple[str, ...]]],
    reach_center: tuple[float, float],
    reach_radius: float,
) -> None:
    unreachable: list[str] = []
    seen: set[str] = set()
    for red_cell, _ in selection_plan:
        if red_cell in seen:
            continue
        seen.add(red_cell)
        region = GRID_CELLS[red_cell]
        min_distance = _min_distance_to_region_xy(region, reach_center)
        if min_distance > reach_radius:
            unreachable.append(f"{red_cell} min_dist={min_distance:.6f}")

    if unreachable:
        raise ValueError(
            "Some red-box cells in matrix_random_placements are outside the robot reach "
            f"radius {reach_radius:.6f} from center {reach_center}: "
            f"{', '.join(unreachable)}. Update the HTML table, markers, or reach radius."
        )


def _min_distance_to_region_xy(
    region: GridRegion,
    reach_center: tuple[float, float],
) -> float:
    x = min(max(float(reach_center[0]), region.x_min), region.x_max)
    y = min(max(float(reach_center[1]), region.y_min), region.y_max)
    return float(np.hypot(x - float(reach_center[0]), y - float(reach_center[1])))


def _sample_region_xy(
    rng: np.random.Generator,
    region: GridRegion,
    margin: float,
    reach_center: tuple[float, float] | None = None,
    reach_radius: float | None = None,
    max_attempts: int = 100,
) -> np.ndarray:
    margin = max(float(margin), 0.0)
    width = max(
        _xy_distance(region.top_left, region.top_right),
        _xy_distance(region.bottom_left, region.bottom_right),
    )
    height = max(
        _xy_distance(region.top_left, region.bottom_left),
        _xy_distance(region.top_right, region.bottom_right),
    )
    row_margin_t = margin / height if height > 0 else 0.0
    col_margin_t = margin / width if width > 0 else 0.0
    if row_margin_t >= 0.5 or col_margin_t >= 0.5:
        raise ValueError(
            f"Margin {margin} is too large for grid region '{region.name}'."
        )

    attempts = max(int(max_attempts), 1)
    last_xy: np.ndarray | None = None
    for _ in range(attempts):
        point = region.point_at(
            row_t=float(rng.uniform(row_margin_t, 1.0 - row_margin_t)),
            col_t=float(rng.uniform(col_margin_t, 1.0 - col_margin_t)),
        )
        xy = np.asarray([point[0], point[1]], dtype=np.float64)
        last_xy = xy
        if reach_center is None or reach_radius is None:
            return xy
        if _is_within_reach(xy, reach_center, reach_radius):
            return xy

    raise ValueError(
        f"Could not sample a reachable point in grid region '{region.name}' "
        f"after {attempts} attempts. Last sampled xy={last_xy}, "
        f"reach_center={reach_center}, reach_radius={reach_radius}."
    )


def _xy_distance(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def _is_within_reach(
    xy: np.ndarray,
    reach_center: tuple[float, float],
    reach_radius: float,
) -> bool:
    dx = float(xy[0]) - float(reach_center[0])
    dy = float(xy[1]) - float(reach_center[1])
    return float(np.hypot(dx, dy)) <= float(reach_radius)


def _yaw_to_quat(yaw: float) -> np.ndarray:
    half = float(yaw) / 2.0
    return np.asarray([np.cos(half), 0.0, 0.0, np.sin(half)], dtype=np.float64)


def _parse_grid_cell(cell: str, rows: int, cols: int) -> tuple[int, int]:
    normalized = cell.strip().lower()
    if len(normalized) < 2 or not normalized[0].isalpha() or not normalized[1:].isdigit():
        raise ValueError(f"Invalid grid cell '{cell}'. Expected format like 'c4'.")

    row = ord(normalized[0]) - ord("a")
    col = int(normalized[1:]) - 1
    if row < 0 or row >= rows or col < 0 or col >= cols:
        raise ValueError(f"Grid cell '{cell}' is outside a {rows}x{cols} grid.")
    return row, col


def _free_joint_id_for_body(model: mujoco.MjModel, body_id: int) -> int | None:
    joint_start = int(model.body_jntadr[body_id])
    joint_count = int(model.body_jntnum[body_id])
    for joint_id in range(joint_start, joint_start + joint_count):
        if model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_FREE:
            return joint_id
    return None
