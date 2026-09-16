import os
import cv2
import pandas as pd
import subprocess
from pathlib import Path
from lerobot.datasets.lerobot_dataset import LeRobotDataset

# === CẤU HÌNH ===
DATASET_ID = "vasco281204/pick_red_20260527_133805"
# ===============

def fix_dataset():
    dataset = LeRobotDataset(DATASET_ID)
    root_dir = dataset.root
    print(f"Đang xử lý dataset local tại: {root_dir}")

    # Đọc metadata của episodes
    episodes_dir = root_dir / "meta" / "episodes"
    if episodes_dir.is_dir():
        df = pd.concat([pd.read_parquet(f) for f in episodes_dir.glob("**/*.parquet")], ignore_index=True)
    else:
        df = pd.read_parquet(root_dir / "meta" / "episodes.parquet")
        
    camera_keys = [k for k, v in dataset.features.items() if v["dtype"] == "video"]

    # Xử lý từng episode
    for idx, row in df.iterrows():
        expected_frames = int(row["length"])
        episode_index = int(row["episode_index"])
        
        for cam in camera_keys:
            chunk_col = f"{cam}/chunk_index" if f"{cam}/chunk_index" in row else f"videos/{cam}/chunk_index"
            file_col = f"{cam}/file_index" if f"{cam}/file_index" in row else f"videos/{cam}/file_index"
            
            if chunk_col not in row or file_col not in row:
                continue

            chunk_idx = int(row[chunk_col])
            file_idx = int(row[file_col])

            video_path = root_dir / f"videos/{cam}/chunk-{chunk_idx:03d}/file-{file_idx:03d}.mp4"
            if not video_path.exists():
                continue

            cap = cv2.VideoCapture(str(video_path))
            actual_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()

            if actual_frames > expected_frames:
                diff = actual_frames - expected_frames
                print(f"[Episode {episode_index}] {cam}: Cắt bỏ {diff} frame thừa ở đầu...")
                
                temp_path = video_path.with_suffix(".temp.mp4")
                cmd = [
                    "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                    "-i", str(video_path),
                    "-vf", f"select='gte(n\\,{diff})',setpts=PTS-STARTPTS",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    str(temp_path)
                ]
                subprocess.run(cmd, check=True)
                os.replace(temp_path, video_path)
                print(f" -> Đã cắt xong.")

    print("\nHoàn tất quá trình cắt frame!")

if __name__ == "__main__":
    fix_dataset()