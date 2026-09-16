import torch
import cv2
import numpy as np
import shutil
from pathlib import Path
from tqdm import tqdm
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.video_utils import decode_video_frames

# ================= CẤU HÌNH =================
OLD_DATASET_ID = "vasco281204/pick_red_20260527_133805"
OLD_DATASET_DIR = "/home/vietanh/pick_red_20260527_133805"
NEW_DATASET_ID = "vasco281204/pick_red_fixed_new"

# 1. BỎ QUA EPISODE LỖI
# Không cần xoá tay nữa, kịch bản sẽ tự động phát hiện frame thừa (tĩnh) ở cuối và cắt.
EPISODES_TO_SKIP = [] 

# 2. SỬA LỖI LỆCH FRAME VIDEO
# Nếu lúc quay episode 3 bạn lỡ ấn nút <- sau khi quay được 45 frame, thì 45 frame rác đó
# đã bị đẩy vào video. Kết quả là Episode 3 và TẤT CẢ các episode sau nó sẽ bị trễ 45 frame.
# Bạn có thể khai báo độ lệch (offset) để script tự động lấy hình ảnh tiến về phía sau.
# Cú pháp: {episode_index: số_frame_bị_lệch}
VIDEO_FRAME_OFFSET = {
    # 3: 45,  # Từ episode 3 bù 45 frame
    # 4: 45,  # Tập 4 cũng nằm chung file nên cũng phải bù 45 frame
}

# 3. CẮT FRAME Ở CUỐI EPISODE
# Nếu đoạn cuối của episode bị hỏng/thừa động tác, bạn có thể cắt bớt đi.
# Cú pháp: {episode_index: số_frame_muốn_cắt_bỏ_ở_cuối}
TRIM_END_FRAMES = {
    # 0: 30, # Ví dụ: Bỏ đi 30 frame cuối cùng của Episode 0
}
# ============================================

def main():
    print(f"Đang tải dataset cũ trực tiếp từ ổ cứng: {OLD_DATASET_DIR}...")
    old_dataset = LeRobotDataset(OLD_DATASET_ID, root=OLD_DATASET_DIR)
    
    # Xoá cache của dataset mới nếu nó đã tồn tại từ lần chạy trước
    new_dataset_dir = Path(f"~/.cache/huggingface/lerobot/{NEW_DATASET_ID}").expanduser()
    if new_dataset_dir.exists():
        print(f"Thư mục {new_dataset_dir} đã tồn tại. Đang tiến hành xoá để tạo lại...")
        shutil.rmtree(new_dataset_dir)

    print(f"Đang tạo dataset mới: {NEW_DATASET_ID}...")
    new_dataset = LeRobotDataset.create(
        repo_id=NEW_DATASET_ID,
        fps=old_dataset.fps,
        features=old_dataset.features,
        use_videos=True,
        # Tắt streaming_encoding để đảm bảo an toàn tuyệt đối cho data mới
        streaming_encoding=False 
    )
    
    camera_keys = old_dataset.meta.camera_keys

    for ep_idx in tqdm(range(old_dataset.num_episodes), desc="Copying episodes"):
        if ep_idx in EPISODES_TO_SKIP:
            print(f"\n[Bỏ qua] Episode {ep_idx}")
            continue
            
        ep_meta = old_dataset.meta.episodes[ep_idx]
        from_idx = ep_meta["dataset_from_index"]
        length = ep_meta["length"]
        task = ep_meta["tasks"][0]
        
        offset = VIDEO_FRAME_OFFSET.get(ep_idx, 0)
        trim_end = TRIM_END_FRAMES.get(ep_idx, 0)
        new_length = length - trim_end
        
        if new_length <= 0:
            print(f"\n[Bỏ qua] Episode {ep_idx} vì bị cắt mất hết frame!")
            continue
        
        # -- TỰ ĐỘNG CẮT FRAME THỪA --
        # 1. Kiểm tra video gốc để đảm bảo không đọc quá số frame thực tế
        min_video_frames = new_length
        for cam in camera_keys:
            chunk_idx = ep_meta[f"videos/{cam}/chunk_index"]
            file_idx = ep_meta[f"videos/{cam}/file_index"]
            video_path = old_dataset.root / f"videos/{cam}/chunk-{chunk_idx:03d}/file-{file_idx:03d}.mp4"
            
            if video_path.exists():
                cap = cv2.VideoCapture(str(video_path))
                actual_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()
                
                start_frame_in_video = round(ep_meta[f"videos/{cam}/from_timestamp"] * old_dataset.fps)
                available_frames = actual_frames - start_frame_in_video - offset
                if 0 < available_frames < min_video_frames:
                    min_video_frames = available_frames

        if min_video_frames < new_length:
            print(f"\n[Tự động cắt] Episode {ep_idx}: Video gốc ngắn hơn metadata. Cắt từ {new_length} xuống {min_video_frames}.")
            new_length = min_video_frames
            
        # 2. Tự động cắt các frame cuối nếu robot không còn di chuyển (dư thừa khi bấm dừng quay trễ)
        states = []
        for i in range(new_length):
            states.append(old_dataset.hf_dataset[from_idx + i]["observation.state"])
        states = torch.tensor(np.array(states)) if not isinstance(states[0], torch.Tensor) else torch.stack(states)
        
        diffs = torch.norm(states[1:] - states[:-1], dim=1)
        
        # Tự động điều chỉnh ngưỡng phát hiện chuyển động nếu dữ liệu là Độ hay Radian
        max_val = torch.max(torch.abs(states))
        threshold = 1e-1 if max_val > 10.0 else 1e-3
        
        moving_indices = torch.where(diffs > threshold)[0]
        if len(moving_indices) > 0:
            last_move_idx = moving_indices[-1].item()
            auto_length = min(new_length, last_move_idx + 1 + 5)  # Giữ thêm 5 frame padding để video mượt mà
            if auto_length < new_length:
                print(f"\n[Tự động cắt] Episode {ep_idx}: Phát hiện {new_length - auto_length} frame tĩnh ở cuối. Cắt từ {new_length} xuống {auto_length}.")
                new_length = auto_length

        # Đọc toàn bộ frame của episode này cho các camera bằng hàm chuẩn của LeRobot (an toàn với mọi codec)
        video_frames = {}
        for cam in camera_keys:
            chunk_idx = ep_meta[f"videos/{cam}/chunk_index"]
            file_idx = ep_meta[f"videos/{cam}/file_index"]
            video_path = old_dataset.root / f"videos/{cam}/chunk-{chunk_idx:03d}/file-{file_idx:03d}.mp4"
            
            start_frame_in_video = round(ep_meta[f"videos/{cam}/from_timestamp"] * old_dataset.fps)
            timestamps = [(start_frame_in_video + offset + i) / old_dataset.fps for i in range(new_length)]
            
            try:
                # return_uint8=True trả về tensor [N, C, H, W]
                frames = decode_video_frames(video_path, timestamps, tolerance_s=1.5/old_dataset.fps, return_uint8=True)
                video_frames[cam] = frames
            except Exception as e:
                raise RuntimeError(f"Lỗi khi đọc video {cam} tại episode {ep_idx}:\n{e}")

        # Trích xuất và đồng bộ lại từng frame
        for frame_idx in range(new_length):
            global_idx = from_idx + frame_idx
            raw_item = old_dataset.hf_dataset[global_idx]
            
            new_frame = {"task": task}
            for key, ft in old_dataset.features.items():
                if key in ["task_index", "frame_index", "episode_index", "timestamp", "index"]:
                    continue
                if ft["dtype"] not in ["video", "image"]:
                    # Vá lỗi UserWarning: Khuyên dùng clone().detach() nếu dữ liệu đã là tensor
                    if isinstance(raw_item[key], torch.Tensor):
                        new_frame[key] = raw_item[key].clone().detach()
                    else:
                        new_frame[key] = torch.tensor(raw_item[key])
            
            for cam in camera_keys:
                # Chuyển shape ảnh từ (C, H, W) sang (H, W, C) để LeRobot ghi nhận đúng
                new_frame[cam] = video_frames[cam][frame_idx].permute(1, 2, 0)
                
            new_dataset.add_frame(new_frame)
            
        new_dataset.save_episode()
        
    new_dataset.finalize()
    print(f"\nHoàn tất! Dataset mới đã sạch sẽ tại: {new_dataset.root}")

    # Tự động đẩy lên Hugging Face Hub
    print("\nĐang tải dataset lên Hugging Face Hub, vui lòng đợi...")
    new_dataset.push_to_hub()

if __name__ == "__main__":
    main()