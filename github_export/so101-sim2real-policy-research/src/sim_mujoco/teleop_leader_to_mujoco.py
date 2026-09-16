import logging
import time
from dataclasses import asdict, dataclass
from pprint import pformat

import mujoco
import mujoco.viewer
import numpy as np

from lerobot.configs import parser
from lerobot.processor import make_default_processors
from lerobot.teleoperators import TeleoperatorConfig, make_teleoperator_from_config
from lerobot.teleoperators.so_leader.config_so_leader import SOLeaderTeleopConfig  # noqa: F401
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import init_logging


@dataclass
class TeleopMujocoConfig:
    teleop: TeleoperatorConfig
    fps: int = 30
    xml_path: str = "reference/sim_assets/so_arm100_simulation/SO101/scene.xml"
    relative_control: bool = False
    camera_preview: bool = True
    preview_camera_name: str = "top_cam"
    preview_width: int = 640
    preview_height: int = 480
    preview_scale: float = 1.0
    preview_grid: bool = True # Whether to draw a grid on the camera preview
    preview_grid_rows: int = 5 # Number of rows for the grid
    preview_grid_cols: int = 7 # Number of columns for the grid
    preview_grid_labels: bool = True # Whether to draw labels on the grid


def get_value_safe(d, *keys, default=None):
    for key in keys:
        if key in d:
            return d[key]
    return default


def teleop_value_to_mujoco_ctrl(actuator_name: str, value: float) -> float:
    if actuator_name == "gripper":
        value = value * 1.13 - 13.0
    return np.deg2rad(float(value))


def print_mujoco_names(model):
    print("\n=== MuJoCo actuators ===")
    for i in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        print(i, name)

    print("\n=== MuJoCo joints ===")
    for i in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        print(i, name)


def teleop_leader_to_mujoco_loop(
    teleop,
    fps,
    xml_path,
    camera_preview,
    preview_camera_name,
    preview_width,
    preview_height,
    preview_scale,
    preview_grid, # Whether to draw a grid on the camera preview
    preview_grid_rows, # Number of rows for the grid
    preview_grid_cols, # Number of columns for the grid
    preview_grid_labels, # Whether to draw labels on the grid
):
    print("[INFO] Loading MuJoCo model:", xml_path)

    model = mujoco.MjModel.from_xml_path(xml_path)
    if camera_preview:
        model.vis.global_.offwidth = max(int(model.vis.global_.offwidth), int(preview_width))
        model.vis.global_.offheight = max(int(model.vis.global_.offheight), int(preview_height))
    data = mujoco.MjData(model)

    renderer = None
    cv2 = None
    preview_window_name = f"mujoco_{preview_camera_name}_{preview_width}x{preview_height}"
    if camera_preview:
        try:
            import cv2 as cv2_module

            cv2 = cv2_module
            renderer = mujoco.Renderer(model, height=preview_height, width=preview_width)
            cv2.namedWindow(preview_window_name, cv2.WINDOW_NORMAL)
        except Exception as exc:
            print(f"[WARN] Camera preview disabled: {exc}")
            camera_preview = False

    print_mujoco_names(model)

    actuator_names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(model.nu)
    ]

    teleop_action_processor, _, _ = make_default_processors()

    print("\n[INFO] Starting leader -> MuJoCo teleoperation")
    print("[INFO] The OpenCV camera preview is fixed-aspect and matches dataset rendering.")
    print("[INFO] Close MuJoCo viewer to stop. Press q/Esc in preview to stop preview loop.")

    try:
        with mujoco.viewer.launch_passive(
            model,
            data,
            show_left_ui=False,
            show_right_ui=False,
        ) as viewer:
            while viewer.is_running():
                loop_start = time.perf_counter()

                raw_action = teleop.get_action()
                teleop_action = teleop_action_processor((raw_action, {}))

                for i, act_name in enumerate(actuator_names):
                    val_deg = get_value_safe(
                        teleop_action,
                        act_name,
                        f"{act_name}.pos",
                        default=None,
                    )
                    if val_deg is None:
                        continue
                    data.ctrl[i] = teleop_value_to_mujoco_ctrl(act_name, float(val_deg))

                target_sim_time = data.time + 1.0 / fps
                while data.time < target_sim_time:
                    mujoco.mj_step(model, data)

                viewer.sync()

                if camera_preview and renderer is not None and cv2 is not None:
                    if not _show_fixed_camera_preview(
                        cv2=cv2,
                        renderer=renderer,
                        data=data,
                        camera_name=preview_camera_name,
                        window_name=preview_window_name,
                        scale=preview_scale,
                        grid=preview_grid,
                        grid_rows=preview_grid_rows,
                        grid_cols=preview_grid_cols,
                        grid_labels=preview_grid_labels,
                    ):
                        break

                dt = time.perf_counter() - loop_start
                precise_sleep(max(1.0 / fps - dt, 0.0))
    finally:
        if renderer is not None:
            renderer.close()
        if cv2 is not None:
            cv2.destroyWindow(preview_window_name)


def _show_fixed_camera_preview(
    cv2,
    renderer,
    data,
    camera_name,
    window_name,
    scale,
    grid,
    grid_rows,
    grid_cols,
    grid_labels,
):
    renderer.update_scene(data, camera=camera_name)
    rgb = renderer.render()
    if "front" in camera_name.lower():
        rgb = np.clip(rgb.astype(np.float32) * 1.2, 0, 255).astype(np.uint8)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    if grid:
        _draw_camera_grid(cv2, bgr, rows=grid_rows, cols=grid_cols, labels=grid_labels)

    scale = max(float(scale), 0.1)
    if abs(scale - 1.0) > 1e-6:
        new_w = int(bgr.shape[1] * scale)
        new_h = int(bgr.shape[0] * scale)
        bgr = cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)

    cv2.imshow(window_name, bgr)
    wait_key = getattr(cv2, "waitKeyEx", cv2.waitKey)
    key = wait_key(1)
    return key not in (ord("q"), ord("Q"), 27)


def _draw_camera_grid(cv2, image, rows, cols, labels):
    h, w = image.shape[:2]
    line_color = (0, 255, 255)
    text_color = (0, 255, 255)
    shadow_color = (0, 0, 0)
    thickness = max(1, round(min(h, w) / 360))

    for col in range(1, cols):
        x = round(col * w / cols)
        cv2.line(image, (x, 0), (x, h - 1), line_color, thickness, cv2.LINE_AA)
    for row in range(1, rows):
        y = round(row * h / rows)
        cv2.line(image, (0, y), (w - 1, y), line_color, thickness, cv2.LINE_AA)

    if not labels:
        return

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.45, min(w / 640.0, h / 480.0) * 0.55)
    text_thickness = max(1, round(thickness))
    for row in range(rows):
        for col in range(cols):
            label = _camera_grid_label(row, col)
            (tw, th), baseline = cv2.getTextSize(label, font, font_scale, text_thickness)
            cx = round((col + 0.5) * w / cols)
            cy = round((row + 0.5) * h / rows)
            origin = (cx - tw // 2, cy + (th - baseline) // 2)
            cv2.putText(
                image,
                label,
                origin,
                font,
                font_scale,
                shadow_color,
                text_thickness + 2,
                cv2.LINE_AA,
            )
            cv2.putText(
                image,
                label,
                origin,
                font,
                font_scale,
                text_color,
                text_thickness,
                cv2.LINE_AA,
            )


def _camera_grid_label(row: int, col: int) -> str:
    row_name = chr(ord("a") + row) if row < 26 else f"r{row + 1}"
    return f"{row_name}{col + 1}"


@parser.wrap()
def main_cfg(cfg: TeleopMujocoConfig):
    init_logging()
    logging.info(pformat(asdict(cfg)))

    teleop = make_teleoperator_from_config(cfg.teleop)

    teleop.connect()

    try:
        teleop_leader_to_mujoco_loop(
            teleop=teleop,
            fps=cfg.fps,
            xml_path=cfg.xml_path,
            camera_preview=cfg.camera_preview,
            preview_camera_name=cfg.preview_camera_name,
            preview_width=cfg.preview_width,
            preview_height=cfg.preview_height,
            preview_scale=cfg.preview_scale,
            preview_grid=cfg.preview_grid,
            preview_grid_rows=cfg.preview_grid_rows,
            preview_grid_cols=cfg.preview_grid_cols,
            preview_grid_labels=cfg.preview_grid_labels,
        )
    except KeyboardInterrupt:
        pass
    finally:
        teleop.disconnect()


def main():
    register_third_party_plugins()
    main_cfg()


if __name__ == "__main__":
    main()
