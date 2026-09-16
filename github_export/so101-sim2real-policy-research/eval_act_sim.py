#!/usr/bin/env python3
"""Chay va danh gia ACT Policy trong moi truong mo phong MuJoCo (SO-101 Stack 3 Blocks).

Script nay cho phep:
1. Load checkpoint ACT policy da train (cuc bo hoac tren Hugging Face Hub).
2. Chay truc tiep policy tren mo phong MuJoCo voi SO-101 Stack 3 Blocks.
3. Hien thi truc quan camera kep (Top Cam co luoi 5x7 + Front Cam) theo thoi gian thuc.
4. Tu dong random vi tri 3 khoi hop moi episode giong 100% thiet lap khi thu data.
5. Tinh toan ty le thanh cong (Success Rate) va ghi video MP4 danh gia neu can.

Cach su dung:
  # Chay voi checkpoint vua train xong:
  python eval_act_sim.py --policy outputs/train/act_stack_3_blocks/checkpoints/last/pretrained_model

  # Hoac tu thu muc goc cua repo:
  /home/anhquoc2910/miniforge3/envs/lerobot/bin/python eval_act_sim.py \
    --policy outputs/train/act_stack_3_blocks/checkpoints/last/pretrained_model

  # Chay khong hien GUI (Headless) va luu video:
  python eval_act_sim.py --policy outputs/train/act_stack_3_blocks/checkpoints/last/pretrained_model \
    --headless --save_video --num_episodes 10
"""

import argparse
import colorsys
import json
import os
import sys
import time
from pathlib import Path

import cv2
import mujoco
import numpy as np
import torch

# Cho phep import cac module tuong doi trong sim_mujoco va lerobot
CURRENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CURRENT_DIR))

try:
    from lerobot.envs.so101 import SO101Stack3BlocksEnv, draw_grid_5x7
except ImportError:
    # Fallback ve MujocoXmlEnv noi bo
    from mujoco_xml_env import MujocoXmlEnv as SO101Stack3BlocksEnv
    from top_camera_grid_overlay import draw_top_camera_grid as draw_grid_5x7

from lerobot.datasets import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.policies import make_pre_post_processors
from lerobot.policies.act import ACTPolicy
from lerobot.policies.utils import prepare_observation_for_inference


def resolve_policy_path(policy_path: str | Path, checkpoint_step: str | int | None = None) -> Path:
    """Tu dong nhan dien va resolve duong dan checkpoint (cuc bo hoac tu Hugging Face Hub)."""
    policy_str = str(policy_path)
    # 1. Kiem tra thu muc cuc bo
    if os.path.exists(policy_str):
        p = Path(policy_str)
        if (p / "config.json").exists():
            return p
        ckpt_dir = p / "checkpoints"
        if ckpt_dir.exists() and ckpt_dir.is_dir():
            subdirs = sorted([d for d in ckpt_dir.iterdir() if d.is_dir() and (d / "config.json").exists()], key=lambda x: x.name)
            if subdirs:
                selected = subdirs[-1]
                if checkpoint_step:
                    match = [d for d in subdirs if str(checkpoint_step) in d.name]
                    if match:
                        selected = match[0]
                print(f"[*] Chon checkpoint cuc bo: {selected}")
                return selected
        return p

    # 2. Repo tren Hugging Face Hub
    if "/" in policy_str:
        try:
            from huggingface_hub import HfApi, snapshot_download
            api = HfApi()
            repo_info = api.model_info(policy_str)
            files = [f.rfilename for f in repo_info.siblings]

            if "config.json" in files:
                print(f"[*] Tai model tu goc repo HF: {policy_str}")
                local_dir = snapshot_download(policy_str)
                return Path(local_dir)

            # Tim cac checkpoint theo subfolder checkpoints/<step>/config.json
            ckpt_steps = set()
            for f in files:
                parts = f.split("/")
                if len(parts) >= 3 and parts[0] == "checkpoints" and parts[2] == "config.json":
                    ckpt_steps.add(parts[1])

            if ckpt_steps:
                sorted_steps = sorted(list(ckpt_steps), key=lambda x: int(x) if x.isdigit() else x)
                target_step = sorted_steps[-1]
                if checkpoint_step:
                    for s in sorted_steps:
                        if str(checkpoint_step) in s:
                            target_step = s
                            break
                print(f"[*] Phat hien cac checkpoint tren HF: {sorted_steps}")
                print(f"[*] Chon checkpoint: {target_step}")
                local_dir = snapshot_download(policy_str, allow_patterns=[f"checkpoints/{target_step}/*"])
                return Path(local_dir) / "checkpoints" / target_step
        except Exception as e:
            print(f"[!] Canh bao khi kiem tra repo HF ({e}), se thu load truc tiep...")

    return Path(policy_str)


def load_policy_and_processors(policy_path: str | Path, dataset_id: str, device: torch.device, checkpoint_step: str | int | None = None):
    """Load policy checkpoint va khoi tao preprocessor / postprocessor."""
    actual_path = resolve_policy_path(policy_path, checkpoint_step)
    print(f"[*] Dang load ACT Policy tu: {actual_path}...")
    policy = ACTPolicy.from_pretrained(str(actual_path))
    policy.to(device)
    policy.eval()
    print("[*] Da load Policy thanh cong!")

    print(f"[*] Dang load metadata tu dataset: {dataset_id}...")
    try:
        ds_meta = LeRobotDatasetMetadata(dataset_id)
        dataset_stats = ds_meta.stats
    except Exception as e:
        print(f"[!] Canh bao: Khong the tai stats tu {dataset_id}: {e}")
        dataset_stats = None

    try:
        preprocessor, postprocessor = make_pre_post_processors(
            policy.config,
            pretrained_path=str(actual_path),
            dataset_stats=dataset_stats,
        )
    except Exception:
        preprocessor, postprocessor = make_pre_post_processors(
            policy.config,
            pretrained_path=None,
            dataset_stats=dataset_stats,
        )
    print("[*] Khoi tao Pre/Post Processors thanh cong!")
    return policy, preprocessor, postprocessor


def randomize_box_colors(model, mode: str = "full", rng: np.random.RandomState | None = None) -> dict[str, list[float]]:
    """Randomize mau sac 3 khoi hop giua cac lan thu (Domain Randomization)."""
    if rng is None:
        rng = np.random.RandomState()

    bottom_geom = "box_white_geom" if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "box_white_geom") >= 0 else "box_black_geom"
    top_geom = "box_blue_geom" if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "box_blue_geom") >= 0 else "box_white_geom"
    geom_names = [bottom_geom, "box_green_geom", top_geom]
    color_dict = {}

    if mode == "full":
        # Tao 3 mau sac tuoi sang, ngau nhien va tach biet ro rang theo vong tron HSV
        hues = [(rng.rand() + i / 3.0) % 1.0 for i in range(3)]
        rng.shuffle(hues)
        for gname, h in zip(geom_names, hues):
            gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, gname)
            if gid >= 0:
                matid = model.geom_matid[gid]
                if matid >= 0:
                    s = float(rng.uniform(0.65, 1.0))
                    v = float(rng.uniform(0.6, 1.0))
                    r, g, b = colorsys.hsv_to_rgb(h, s, v)
                    model.mat_rgba[matid][:3] = [r, g, b]
                    color_dict[gname] = [round(r, 2), round(g, 2), round(b, 2)]

    elif mode == "jitter":
        # Bien thien do sang / sac do nhe quanh 3 mau goc (Trang/Den, Xanh la, Xanh bien)
        base_colors = {
            "box_white_geom": [0.94, 0.94, 0.94],
            "box_black_geom": [0.12, 0.12, 0.12],
            "box_green_geom": [0.15, 0.75, 0.20],
            "box_blue_geom": [0.10, 0.55, 0.95],
        }
        for gname in geom_names:
            gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, gname)
            if gid >= 0:
                matid = model.geom_matid[gid]
                if matid >= 0:
                    base = np.array(base_colors.get(gname, [0.5, 0.5, 0.5]), dtype=np.float32)
                    jitter = rng.uniform(-0.15, 0.15, size=3)
                    rgb = np.clip(base + jitter, 0.03, 0.97)
                    model.mat_rgba[matid][:3] = rgb
                    color_dict[gname] = [round(float(c), 2) for c in rgb]

    elif mode == "shuffle":
        # Hoan vi ngau nhien 3 mau goc cho 3 khoi
        colors = [
            [0.12, 0.12, 0.12],
            [0.15, 0.75, 0.20],
            [0.10, 0.55, 0.95],
        ]
        rng.shuffle(colors)
        for gname, rgb in zip(geom_names, colors):
            gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, gname)
            if gid >= 0:
                matid = model.geom_matid[gid]
                if matid >= 0:
                    model.mat_rgba[matid][:3] = rgb
                    color_dict[gname] = [round(float(c), 2) for c in rgb]

    return color_dict


def run_eval_sim(args):
    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    print(f"[*] Thiet bi tinh toan: {device}")

    # 1. Load policy & processors
    policy, preprocessor, postprocessor = load_policy_and_processors(
        args.policy, args.dataset, device, getattr(args, "checkpoint", None)
    )

    # 2. Khoi tao Moi truong SO101 Stack 3 Blocks
    env = SO101Stack3BlocksEnv(
        max_episode_steps=args.max_steps,
        render_mode="rgb_array",
    )

    # Kiem tra che do hien thi GUI
    display_available = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    headless = args.headless or not display_available
    window_name = "MuJoCo SO-101 ACT Rollout (SPACE: Reset, P: Pause, Q: Quit)"

    if not headless:
        try:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        except Exception as e:
            print(f"[!] Canh bao: Khong mo duoc OpenCV window ({e}), chuyen sang che do headless.")
            headless = True

    # Thu muc luu video neu duoc yeu cau
    video_dir = None
    if args.save_video:
        video_dir = Path(args.video_dir)
        video_dir.mkdir(parents=True, exist_ok=True)
        print(f"[*] Video rollouts se duoc luu tai: {video_dir.resolve()}")

    print("\n" + "=" * 68)
    print("      BAT DAU CHAY EVALUATION ACT POLICY TREN SO-101 MUJOCO")
    print("=" * 68)
    print("Thong tin thiet lap:")
    print("  - Task        : Stack 3 blocks (Black day, Green giua, Blue tren cung)")
    print("  - Dieu kien   : Random vi tri 3 hop theo luoi o rieng biet (chuan data collection)")
    print("  - Tan so      : 30 FPS")
    print("  - Goc dieu khien: Do (Degrees)")
    if not headless:
        print("Phim tat:")
        print("  - SPACE / ENTER / r : Reset sang episode moi")
        print("  - p                 : Tam dung (Pause) / Tiep tuc")
        print("  - q / ESC           : Thoat chuong trinh")
    print("=" * 68 + "\n")

    target_fps = 30
    frame_interval = 1.0 / target_fps

    # Khoi tao LeRobotDataset neu co yeu cau luu data hoac day len Hub
    dataset = None
    dataset_root = None
    start_episodes_count = 0
    hub_repo_id = getattr(args, "hub_repo_id", "anhquoc29/rollout_stact_3_box_sim")
    if getattr(args, "save_dataset", False) or getattr(args, "push_to_hub", False):
        if "/" not in hub_repo_id:
            try:
                from huggingface_hub import HfApi
                user = HfApi().whoami()
                username = user.get("name", "anhquoc29")
                hub_repo_id = f"{username}/{hub_repo_id}"
            except Exception:
                hub_repo_id = f"anhquoc29/{hub_repo_id}"

        repo_suffix = hub_repo_id.split("/")[-1]
        dataset_root = Path(args.dataset_dir) / repo_suffix
        dataset_features = {
            "observation.state": {
                "dtype": "float32",
                "shape": (6,),
                "names": ["shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos", "wrist_flex.pos", "wrist_roll.pos", "gripper.pos"],
            },
            "action": {
                "dtype": "float32",
                "shape": (6,),
                "names": ["shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos", "wrist_flex.pos", "wrist_roll.pos", "gripper.pos"],
            },
            "observation.images.top": {
                "dtype": "video",
                "shape": (480, 640, 3),
                "names": ["height", "width", "channels"],
                "info": {"is_depth_map": False},
            },
            "observation.images.front": {
                "dtype": "video",
                "shape": (480, 640, 3),
                "names": ["height", "width", "channels"],
                "info": {"is_depth_map": False},
            },
        }
        # Kiem tra dataset da ton tai o cuc bo hoac tren Hub chua de ghi noi tiep (append)
        exists_locally = dataset_root.exists() and (dataset_root / "meta" / "info.json").exists()
        exists_on_hub = False
        if not exists_locally and not getattr(args, "overwrite", False):
            try:
                from huggingface_hub import HfApi
                exists_on_hub = HfApi().repo_exists(repo_id=hub_repo_id, repo_type="dataset")
            except Exception:
                exists_on_hub = False

        should_resume = (exists_locally or exists_on_hub) and not getattr(args, "overwrite", False)

        if should_resume:
            location_str = "cuc bo" if exists_locally else f"tren Hugging Face Hub ({hub_repo_id})"
            print(f"[*] Phat hien dataset da ton tai ({location_str}).")
            print(f"[*] Che do: GHI NOI TIEP (Append mode) - Khong ghi de du lieu cu!")
            dataset = LeRobotDataset.resume(repo_id=hub_repo_id, root=dataset_root)
            start_episodes_count = dataset.num_episodes
            print(f"[*] Da co san {start_episodes_count} tap trong dataset. Cac tap rollout moi se duoc ghi them tu tap {start_episodes_count + 1}!")
        else:
            if dataset_root.exists():
                import shutil
                shutil.rmtree(dataset_root, ignore_errors=True)
            dataset = LeRobotDataset.create(
                repo_id=hub_repo_id,
                fps=target_fps,
                features=dataset_features,
                root=dataset_root,
                robot_type="so_follower",
                use_videos=True,
            )
            print(f"[*] Da khoi tao Dataset moi cho Hub repo '{hub_repo_id}' tai: {dataset_root.resolve()}")

    episode_idx = 0
    total_episodes = args.num_episodes

    episode_results = []

    try:
        while episode_idx < total_episodes:
            episode_idx += 1
            global_ep_idx = start_episodes_count + episode_idx
            seed = args.seed + global_ep_idx * 17
            obs, info = env.reset(seed=seed, options={"episode_index": global_ep_idx})
            policy.reset()  # Reset action queue cua ACT

            chosen_colors = None
            if getattr(args, "random_colors", False):
                ep_rng = np.random.RandomState(seed)
                chosen_colors = randomize_box_colors(env.model, mode=getattr(args, "color_mode", "full"), rng=ep_rng)
                if hasattr(env, "data"):
                    mujoco.mj_forward(env.model, env.data)
                if hasattr(env, "_get_obs"):
                    obs = env._get_obs()

            placements = info.get("stack_random_placements", {})
            bottom_key = "box_white" if "box_white" in placements else "box_black"
            bottom_label = "Trang" if bottom_key == "box_white" else "Den"
            b_cell = placements.get(bottom_key, {}).get("cell", "-")
            g_cell = placements.get("box_green", {}).get("cell", "-")
            top_key = "box_blue" if "box_blue" in placements else "box_white"
            top_label = "Xanh bien" if top_key == "box_blue" else "Trang"
            top_cell = placements.get(top_key, {}).get("cell", "-")
            print(f"\n[Episode {episode_idx}/{total_episodes}] (Dataset Episode {global_ep_idx}) Bat dau! {bottom_label}: [{b_cell}] | Xanh la: [{g_cell}] | {top_label}: [{top_cell}]")
            if chosen_colors:
                c_bottom = chosen_colors.get(f"{bottom_key}_geom", [])
                c_green = chosen_colors.get("box_green_geom", [])
                c_blue = chosen_colors.get(f"{top_key}_geom", [])
                print(f"   🎨 Random mau ({args.color_mode}): {bottom_label}={c_bottom} | Xanh la={c_green} | {top_label}={c_blue}")

            step_count = 0
            paused = False
            ep_frames = []
            ep_success = False

            while step_count < args.max_steps:
                step_start_time = time.perf_counter()

                # Trich xuat camera quan sat
                top_img = obs["pixels"]["top"]
                front_img = obs["pixels"].get("front", obs["pixels"].get("gripper"))

                # Chuyen doi BGR phuc vu render
                top_bgr = cv2.cvtColor(top_img, cv2.COLOR_RGB2BGR)
                front_bgr = cv2.cvtColor(front_img, cv2.COLOR_RGB2BGR)

                # Ve luoi toa do 5x7 len camera Top
                draw_grid_5x7(
                    top_bgr,
                    rows=5,
                    cols=7,
                    labels=True,
                    margin_left=16,
                    margin_top=32,
                    margin_right=16,
                    margin_bottom=0,
                )

                combined = np.hstack([top_bgr, front_bgr])

                # Top banner
                cv2.rectangle(combined, (0, 0), (combined.shape[1], 45), (20, 20, 20), -1)
                title_txt = f"EPISODE {episode_idx}/{total_episodes} | STEP {step_count}/{args.max_steps} | FPS: {target_fps}"
                cv2.putText(combined, title_txt, (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                # Camera labels
                cv2.putText(combined, "TOP CAMERA (Grid 5x7)", (15, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 1)
                cv2.putText(combined, "FRONT CAMERA", (640 + 15, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 1)

                if paused:
                    cv2.putText(combined, "** TAM DUNG (PAUSED) - BAM P DE TIEP TUC **", (250, 250),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

                if args.save_video:
                    ep_frames.append(cv2.cvtColor(combined, cv2.COLOR_BGR2RGB))

                if not headless:
                    cv2.imshow(window_name, combined)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), 27):
                        print("\nNguoi dung yeu cau thoat.")
                        return
                    elif key in (ord(" "), 13, 10, ord("r")):
                        print("-> Reset som sang episode moi.")
                        break
                    elif key == ord("p"):
                        paused = not paused

                if paused:
                    time.sleep(0.05)
                    continue

                # --- Policy Inference ---
                # State theo Do (Degrees) giong khi thu dataset
                state_deg = obs["agent_pos"].astype(np.float32)

                raw_obs = {
                    "observation.state": state_deg,
                    "observation.images.top": top_img,
                    "observation.images.front": front_img,
                }

                inf_obs = prepare_observation_for_inference(raw_obs, device=device)
                processed_obs = preprocessor(inf_obs)

                with torch.no_grad():
                    action_out = policy.select_action(processed_obs)
                    action_out = postprocessor(action_out)

                action_deg = action_out.squeeze(0).cpu().numpy().astype(np.float32)

                # Step mo phong SO101 (nhan action theo DO)
                obs, reward, terminated, truncated, step_info = env.step(action_deg)
                step_count += 1

                # Ghi frame vao LeRobotDataset
                if dataset is not None:
                    dataset.add_frame({
                        "observation.state": state_deg,
                        "action": action_deg,
                        "observation.images.top": top_img,
                        "observation.images.front": front_img,
                        "task": "Stack 3 blocks: black on bottom, green in middle, blue on top",
                    })

                if step_info.get("is_success", False):
                    ep_success = True

                # Gioi han toc do ~30 FPS thuc te
                if not headless:
                    elapsed = time.perf_counter() - step_start_time
                    sleep_time = frame_interval - elapsed
                    if sleep_time > 0:
                        time.sleep(sleep_time)

                if terminated or truncated:
                    status_str = "THANH CONG (STACKED)" if ep_success else "KET THUC"
                    print(f"-> Episode {episode_idx} [{status_str}] o step {step_count}.")
                    break

            episode_results.append({
                "episode": episode_idx,
                "steps": step_count,
                "success": ep_success,
            })

            # Luu tap vao LeRobotDataset
            if dataset is not None:
                if getattr(args, "save_successful_only", False) and not ep_success:
                    print(f"   [-] Bo qua luu tap {episode_idx} vao dataset do chua thanh cong (--save_successful_only).")
                    dataset.clear_episode_buffer()
                else:
                    dataset.save_episode()
                    print(f"   [+] Da luu tap {episode_idx} ({step_count} frames) vao dataset buffer.")

            # Luu video episode neu duoc yeu cau
            if args.save_video and ep_frames and video_dir is not None:
                video_file = video_dir / f"eval_episode_{episode_idx}.mp4"
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                h_out, w_out = ep_frames[0].shape[:2]
                writer = cv2.VideoWriter(str(video_file), fourcc, target_fps, (w_out, h_out))
                for f in ep_frames:
                    writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
                writer.release()
                print(f"   [+] Da luu video: {video_file}")

            time.sleep(0.3)

    finally:
        if not headless:
            cv2.destroyAllWindows()
        env.close()
        print("\nDa dong moi truong mo phong.")

        if dataset is not None:
            new_episodes = dataset.num_episodes - start_episodes_count
            if new_episodes > 0:
                print("\n[*] Dang dong goi dataset (finalize)...")
                dataset.finalize()
                print(f"[*] Da ghi them thanh cong {new_episodes} tap moi vao: {dataset_root.resolve()}!")
                print(f"[*] Tong so tap hien co trong dataset: {dataset.num_episodes} tap.")

                if getattr(args, "push_to_hub", False):
                    print("\n" + "=" * 70)
                    print(f"  DANG DAY DU LIEU ROLLOUT MOI LEN HUGGING FACE: {hub_repo_id}...")
                    print("=" * 70)
                    dataset.push_to_hub(
                        tags=["lerobot", "act", "so101", "simulation", "rollout"],
                        private=getattr(args, "private", False),
                    )
                    print(f"\n>>> DA DAY THANH CONG {new_episodes} TAP MOI LEN HUGGING FACE! <<<")
                    print(f">>> TONG SO TAP HIEN CO TREN CLOUD: {dataset.num_episodes} TAP! <<<")
                    print(f">>> Link Dataset: https://huggingface.co/datasets/{hub_repo_id} <<<\n")

                    if getattr(args, "cleanup_local_after_push", False):
                        import shutil
                        shutil.rmtree(dataset_root, ignore_errors=True)
                        print(f">>> DA XOA DU LIEU CUC BO TAI {dataset_root} DE TIET KIEM O CUNG! <<<")
            else:
                print(f"[*] Khong co tap moi nao duoc luu vao dataset (Dataset giu nguyen {dataset.num_episodes} tap ban dau).")

    # In bao cao tong ket
    if episode_results:
        success_count = sum(1 for r in episode_results if r["success"])
        success_rate = (success_count / len(episode_results)) * 100.0
        avg_steps = np.mean([r["steps"] for r in episode_results])
        print("\n" + "=" * 50)
        print("          KET QUA EVALUATION TONG KET")
        print("=" * 50)
        print(f"Tong so episode       : {len(episode_results)}")
        print(f"So tap thanh cong     : {success_count} / {len(episode_results)}")
        print(f"Ty le thanh cong      : {success_rate:.1f}%")
        print(f"So buoc trung binh    : {avg_steps:.1f} steps")
        print("=" * 50 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Chay demo ACT Policy tren mo phong SO-101 MuJoCo")
    parser.add_argument(
        "--policy",
        type=str,
        default="outputs/train/act_stack_3_blocks/checkpoints/last/pretrained_model",
        help="Duong dan thu muc checkpoint policy hoac repo ID Hugging Face (vi du: vasco281204/act_stack_3_box_30x3030)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Buoc checkpoint cu the neu repo tren HF co nhieu checkpoint (vi du: 100000, 080000). Mac dinh tu chon checkpoint moi nhat.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="anhquoc29/stack_3_box_30x3030",
        help="Dataset ID de load normalization stats",
    )
    parser.add_argument("--num_episodes", type=int, default=20, help="So episode muon chay eval")
    parser.add_argument("--max_steps", type=int, default=1800, help="So step toi da moi episode (1800 step = 60s)")
    parser.add_argument("--device", type=str, default="cuda", help="Thiet bi: cuda hoac cpu")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--headless", action="store_true", help="Chay khong hien thi cua so OpenCV GUI")
    parser.add_argument("--save_video", action="store_true", help="Ghi lai video MP4 cac episode")
    parser.add_argument("--video_dir", type=str, default="outputs/eval_videos", help="Thu muc luu video")

    # Cac tham so thu thap va day Dataset len Hugging Face
    parser.add_argument("--save_dataset", action="store_true", help="Ghi lai toan bo du lieu rollout vao LeRobotDataset")
    parser.add_argument("--push_to_hub", action="store_true", help="Tu dong day dataset len Hugging Face Hub sau khi rollout xong")
    parser.add_argument(
        "--hub_repo_id",
        type=str,
        default="anhquoc29/rollout_stact_3_box_sim",
        help="Ten repo dataset tren Hugging Face Hub (mac dinh: anhquoc29/rollout_stact_3_box_sim)",
    )
    parser.add_argument("--dataset_dir", type=str, default="datasets", help="Thu muc goc luu dataset cuc bo")
    parser.add_argument("--save_successful_only", action="store_true", help="Chi luu cac episode dat thanh cong vao dataset")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Ghi de va tao moi dataset tu dau (xoa bo cac tap cu neu co). Mac dinh LUON LA GHI NOI TIEP (append) khong ghi de.",
    )
    parser.add_argument("--private", action="store_true", help="Dat dataset o che do rieng tu tren Hugging Face")
    parser.add_argument("--cleanup_local_after_push", action="store_true", help="Tu dong xoa dataset cuc bo sau khi day len Hub thanh cong")

    # Domain Randomization (Random mau sac giua cac lan thu)
    parser.add_argument(
        "--random_colors",
        action="store_true",
        help="Tu dong random mau sac 3 khoi hop giua cac lan thu (Domain Randomization)",
    )
    parser.add_argument(
        "--color_mode",
        type=str,
        default="full",
        choices=["full", "jitter", "shuffle"],
        help="Che do random mau: 'full' (mau ngau nhien hoan toan), 'jitter' (bien thien do sang/sac do nhe quanh mau goc), 'shuffle' (hoan vi 3 mau goc)",
    )
    args = parser.parse_args()

    run_eval_sim(args)


if __name__ == "__main__":
    main()
