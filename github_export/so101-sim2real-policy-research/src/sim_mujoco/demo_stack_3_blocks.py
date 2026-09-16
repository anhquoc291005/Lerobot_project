#!/usr/bin/env python3
"""Demo truc quan moi truong xep chong 3 khoi hop (Stack 3 Blocks) trong MuJoCo.

Khong can cam tay Leader that, ban co the chay demo nay de:
1. Quan sat robot SO-101 va 3 khoi hop (Den, Xanh la, Trang).
2. Xem co che random vi tri 3 khoi hop trong vung hoat dong (workspace) moi khi reset.
3. Xem goc nhin tu 2 camera (top_cam va front_cam) giong het khi thu thap dataset.
4. Dung chuot tuong tac 3D (keo tha vat the) trong che do 3D viewer.

Cach dung:
  # Che do 1: Xem 2 goc camera (top_cam + front_cam)
  python src/sim_mujoco/demo_stack_3_blocks.py

  # Che do 2: Xem 3D Viewer tuong tac
  python src/sim_mujoco/demo_stack_3_blocks.py --viewer
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import mujoco
import numpy as np

# Cho phep import cac module tuong doi trong sim_mujoco
sys.path.insert(0, str(Path(__file__).resolve().parent))
from mujoco_xml_env import MujocoXmlEnv, MujocoXmlEnvConfig
from top_camera_grid_overlay import draw_top_camera_grid


def run_camera_preview_demo(config_path: Path):
    print("=" * 65)
    print("  DEMO XEP CHONG 3 KHOI HOP - GOC NHIN 2 CAMERA (TOP & FRONT)")
    print("=" * 65)
    print("Huong dan phim tat tren cua so preview:")
    print("  - SPACE / ENTER / r : Reset va random lai vi tri 3 khoi hop")
    print("  - q / ESC           : Thoat chuong trinh")
    print("=" * 65)

    with open(config_path, "r", encoding="utf-8") as f:
        cfg_dict = json.load(f)["env"]

    env = MujocoXmlEnv(MujocoXmlEnvConfig.from_dict(cfg_dict))
    episode_index = 0
    obs, info = env.reset(seed=episode_index, options={"episode_index": episode_index})
    _print_placements(episode_index, info.get("stack_random_placements", {}))

    window_name = "MuJoCo Stack 3 Blocks Demo (Press SPACE to randomize, Q to quit)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    # Vi tri joint ban dau
    action_initial = np.array([0.0, -0.4, 0.8, 0.4, 0.0, 0.0], dtype=np.float32)
    try:
        while True:
            # Step nhe physics de on dinh tiep xuc
            obs, _, _, _, info = env.step(action_initial)

            top_bgr = cv2.cvtColor(obs["pixels"]["top"], cv2.COLOR_RGB2BGR)
            front_bgr = cv2.cvtColor(obs["pixels"]["gripper"], cv2.COLOR_RGB2BGR)

            # Ve luoi toa do tren camera top
            draw_top_camera_grid(
                cv2=cv2,
                image=top_bgr,
                rows=5,
                cols=7,
                labels=True,
                margin_left=16,
                margin_top=32,
                margin_right=16,
                margin_bottom=0,
            )

            # Ghep 2 camera thanh 1 man hinh kep
            combined = np.hstack([top_bgr, front_bgr])

            # Ve banner thong tin
            cv2.rectangle(combined, (0, 0), (combined.shape[1], 40), (25, 25, 25), -1)
            placements = info.get("stack_random_placements", {})
            bottom_block_name = "box_white" if ("box_white" in placements and "box_blue" in placements) or ("box_white" in placements and "box_black" not in placements) else "box_black"
            bottom_label = "White" if bottom_block_name == "box_white" else "Black"
            bottom_vn = "Trang" if bottom_block_name == "box_white" else "Den"
            b_info = placements.get(bottom_block_name, {})
            g_info = placements.get("box_green", {})
            top_block_name = "box_blue" if "box_blue" in placements else ("box_white" if bottom_block_name != "box_white" else "box_blue")
            w_info = placements.get(top_block_name, {})
            top_label = "Blue" if top_block_name == "box_blue" else "White"

            b_cell = f"[{b_info.get('cell')}] " if b_info.get("cell") and b_info.get("cell") != "-" else ""
            g_cell = f"[{g_info.get('cell')}] " if g_info.get("cell") and g_info.get("cell") != "-" else ""
            w_cell = f"[{w_info.get('cell')}] " if w_info.get("cell") and w_info.get("cell") != "-" else ""
            text_title = f"Episode {episode_index} | {bottom_label}: {b_cell}({b_info.get('x', 0):.2f}, {b_info.get('y', 0):.2f}) | Green: {g_cell}({g_info.get('x', 0):.2f}, {g_info.get('y', 0):.2f}) | {top_label}: {w_cell}({w_info.get('x', 0):.2f}, {w_info.get('y', 0):.2f})"
            cv2.putText(combined, text_title, (15, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

            # Labels goc camera
            cv2.putText(combined, "CAMERA 1: TOP CAM (Luo-i 5x7)", (15, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(combined, "CAMERA 2: FRONT CAM (Chinh dien)", (640 + 15, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            task_text = f"Nhiem vu: Xep {bottom_vn} (duoi) -> Xanh la (giua) -> {('Xanh nuoc bien' if top_block_name == 'box_blue' else 'Trang')} (tren)"
            cv2.putText(combined, task_text, (15, combined.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)

            cv2.imshow(window_name, combined)
            key = cv2.waitKey(33) & 0xFF

            if key in (ord("q"), 27):  # q or Esc
                print("\nThoat demo.")
                break
            elif key in (ord(" "), 13, 10, ord("r")):  # Space, Enter, r
                episode_index += 1
                obs, info = env.reset(seed=episode_index * 17 + 42, options={"episode_index": episode_index})
                _print_placements(episode_index, info.get("stack_random_placements", {}))
    finally:
        cv2.destroyAllWindows()
        env.close()


def run_3d_viewer_demo(config_path: Path):
    import mujoco.viewer

    print("=" * 65)
    print("  DEMO XEP CHONG 3 KHOI HOP - 3D INTERACTIVE VIEWER (MUJOCO)")
    print("=" * 65)
    print("Huong dan:")
    print("  - Chuot trai: Xoay goc nhin 3D")
    print("  - Chuot phai: Zoom in / Zoom out")
    print("  - Chuot giua: Di chuyen goc nhin (Pan)")
    print("  - Ctrl + Chuot phai: Tuong tac keo / day khoi hop va tay robot")
    print("  - SPACE: Tam dung / chay tiep simulation")
    print("  - Nhap vao terminal de doi vi tri random 3 khoi hop")
    print("=" * 65)

    with open(config_path, "r", encoding="utf-8") as f:
        cfg_dict = json.load(f)["env"]

    env = MujocoXmlEnv(MujocoXmlEnvConfig.from_dict(cfg_dict))
    episode_index = 0
    obs, info = env.reset(seed=42, options={"episode_index": 0})
    _print_placements(episode_index, info.get("stack_random_placements", {}))

    action_zero = np.zeros(env.action_space.shape, dtype=np.float32)

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        last_randomize_t = time.time()
        print("\nViewer dang chay! Mo cua so MuJoCo viewer de xem 3D.")
        print("[MEO] Cu sau 8 giay (hoac khi viewer reset), 3 khoi hop se duoc tu dong random vi tri moi.")

        while viewer.is_running():
            step_start = time.perf_counter()

            # Step physics
            mujoco.mj_step(env.model, env.data, nstep=max(cfg_dict.get("frame_skip", 1), 1))
            viewer.sync()

            # Auto re-randomize moi 8 giay de quan sat cac vi tri ngau nhien
            if time.time() - last_randomize_t > 8.0:
                episode_index += 1
                obs, info = env.reset(seed=episode_index * 13 + 7, options={"episode_index": episode_index})
                _print_placements(episode_index, info.get("stack_random_placements", {}))
                last_randomize_t = time.time()

            elapsed = time.perf_counter() - step_start
            time_left = (1.0 / 60.0) - elapsed
            if time_left > 0:
                time.sleep(time_left)

    env.close()
    print("Da dong 3D Viewer.")


def _print_placements(episode_index: int, placements: dict):
    print(f"\n[Episode {episode_index}] Vi tri 3 khoi hop:")
    bottom_name = "box_white" if ("box_white" in placements and "box_blue" in placements) or ("box_white" in placements and "box_black" not in placements) else "box_black"
    bottom_label = "TRANG   (day)" if bottom_name == "box_white" else "DEN     (day)"
    top_name = "box_blue" if "box_blue" in placements else ("box_white" if bottom_name != "box_white" else "box_blue")
    top_label = "XANH BIEN(tren)" if top_name == "box_blue" else "TRANG   (tren)"
    for name, label in [(bottom_name, bottom_label), ("box_green", "XANH LA (giua)"), (top_name, top_label)]:
        p = placements.get(name, {})
        cell_str = f"O = {p.get('cell'):4s} | " if p.get("cell") and p.get("cell") != "-" else ""
        x, y, yaw = p.get("x", 0.0), p.get("y", 0.0), p.get("yaw", 0.0)
        print(f"  * {label}: {cell_str}X = {x:.3f} m, Y = {y:.3f} m, Yaw = {np.rad2deg(yaw):+.1f} deg")


def main():
    parser = argparse.ArgumentParser(description="Demo moi truong stack 3 blocks trong MuJoCo (khong can tay leader)")
    parser.add_argument("--viewer", action="store_true", help="Chay o che do 3D interactive viewer MuJoCo")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("src/sim_mujoco/config_sim_collect_stack_3_blocks.json"),
        help="Duong dan file config",
    )
    args = parser.parse_args()

    if args.viewer:
        run_3d_viewer_demo(args.config)
    else:
        run_camera_preview_demo(args.config)


if __name__ == "__main__":
    main()