import torch
import numpy as np
import shutil
from pathlib import Path
from tqdm import tqdm
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.video_utils import decode_video_frames
import warnings
warnings.filterwarnings("ignore")

# ================= CẤU HÌNH =================
OLD_DATASET_ID = "vasco281204/pick_red_fixed_new"
NEW_DATASET_ID = "vasco281204/pick_red_labeled"

# Ngưỡng sai số của kẹp (Gripper) để nhận diện đang gắp vật.
# Nếu độ lệch giữa lệnh (action) và thực tế (state) lớn hơn mức này -> Đang gắp.
ERROR_THRESHOLD_DEG = 5.0  # Độ (Degrees)
# ============================================

def main():
    print(f"Đang tải dataset gốc: {OLD_DATASET_ID}...")
    old_dataset = LeRobotDataset(OLD_DATASET_ID)
    
    # Dọn dẹp cache cũ nếu có
    new_dataset_dir = Path(f"~/.cache/huggingface/lerobot/{NEW_DATASET_ID}").expanduser()
    if new_dataset_dir.exists():
        print(f"Thư mục {new_dataset_dir} đã tồn tại. Đang tiến hành xoá để tạo lại...")
        shutil.rmtree(new_dataset_dir)

    print(f"Đang tạo dataset mới (có dán nhãn tự động): {NEW_DATASET_ID}...")
    new_dataset = LeRobotDataset.create(
        repo_id=NEW_DATASET_ID,
        fps=old_dataset.fps,
        features=old_dataset.features,
        use_videos=True,
        streaming_encoding=False 
    )
    
    camera_keys = old_dataset.meta.camera_keys

    for ep_idx in tqdm(range(old_dataset.num_episodes), desc="Đánh nhãn & Copy"):
        ep_meta = old_dataset.meta.episodes[ep_idx]
        from_idx = ep_meta["dataset_from_index"]
        length = ep_meta["length"]
        
        # --- 1. TỰ ĐỘNG NHẬN DIỆN GIAI ĐOẠN ---
        # Trích xuất toàn bộ state và action của tập hiện tại
        ep_dict = old_dataset.hf_dataset.select(range(from_idx, from_idx + length))
        states = np.array(ep_dict["observation.state"])
        actions = np.array(ep_dict["action"])
        
        # Lấy dữ liệu khớp Gripper (Khớp cuối cùng - index 5)
        gripper_state = states[:, 5]
        gripper_action = actions[:, 5]
        
        # Tính độ lệch (Tracking Error)
        error = np.abs(gripper_state - gripper_action)
        
        # Tự động điều chỉnh ngưỡng nếu dữ liệu đang lưu ở hệ Radian
        threshold = ERROR_THRESHOLD_DEG
        if np.max(np.abs(gripper_state)) <= 10.0:
            threshold = np.deg2rad(ERROR_THRESHOLD_DEG)
            
        # Tìm các frame mà độ lệch vượt ngưỡng
        is_grasping = error > threshold
        grasp_indices = np.where(is_grasping)[0]
        
        if len(grasp_indices) > 5:  # Tránh nhiễu (phải gắp liên tục > 5 frame)
            start_grasp = grasp_indices[0]
            end_grasp = grasp_indices[-1]
            print(f"\n[Episode {ep_idx}] Phân dải: Trước gắp (0->{start_grasp}), Gắp ({start_grasp}->{end_grasp}), Sau gắp ({end_grasp}->{length})")
        else:
            start_grasp = length
            end_grasp = length
            print(f"\n[Episode {ep_idx}] Không phát hiện vật cản kẹp. Gán toàn bộ là 'trước gắp'.")

        # --- 2. ĐỌC VIDEO VÀ COPY DỮ LIỆU ---
        video_frames = {}
        for cam in camera_keys:
            chunk_idx = ep_meta[f"videos/{cam}/chunk_index"]
            file_idx = ep_meta[f"videos/{cam}/file_index"]
            video_path = old_dataset.root / f"videos/{cam}/chunk-{chunk_idx:03d}/file-{file_idx:03d}.mp4"
            
            start_frame_in_video = round(ep_meta[f"videos/{cam}/from_timestamp"] * old_dataset.fps)
            timestamps = [(start_frame_in_video + i) / old_dataset.fps for i in range(length)]
            frames = decode_video_frames(video_path, timestamps, tolerance_s=1.5/old_dataset.fps, return_uint8=True)
            video_frames[cam] = frames

        for frame_idx in range(length):
            raw_item = ep_dict[frame_idx]
            
            # Xác định nhãn (Task) cho frame hiện tại
            if frame_idx < start_grasp:
                current_task = "trước gắp"
            elif frame_idx <= end_grasp:
                current_task = "gắp"
            else:
                current_task = "sau gắp"
                
            new_frame = {"task": current_task}
            
            for key, ft in old_dataset.features.items():
                if key in ["task_index", "frame_index", "episode_index", "timestamp", "index", "task"]:
                    continue
                if ft["dtype"] not in ["video", "image"]:
                    new_frame[key] = torch.tensor(raw_item[key])
            
            for cam in camera_keys:
                new_frame[cam] = video_frames[cam][frame_idx].permute(1, 2, 0)
                
            new_dataset.add_frame(new_frame)
        new_dataset.save_episode()
        
    new_dataset.finalize()
    print(f"\nĐang tải bộ dữ liệu đã đánh nhãn lên Hugging Face: {NEW_DATASET_ID} ...")
    new_dataset.push_to_hub()

if __name__ == "__main__":
    main()