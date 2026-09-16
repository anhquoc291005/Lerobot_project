# Hướng Dẫn Thiết Lập Mô Hình MuJoCo & Thu Thập Dữ Liệu Teleop Robot SO-101

Tài liệu này hướng dẫn chi tiết quy trình:
1. Sử dụng mô hình mô phỏng MuJoCo của cánh tay robot **SO-101** (Follower ảo).
2. Kết nối và hiệu chuẩn cánh tay **SO-101 Leader thật** qua cổng Serial/USB.
3. Điều khiển mô phỏng thời gian thực (Teleoperation: Leader thật $\rightarrow$ Follower ảo).
4. Thu thập dataset mô phỏng theo định dạng chuẩn `LeRobotDataset` phục vụ huấn luyện Policy (ACT, Diffusion Policy).

---

## 1. Cấu Trúc Thư Mục Dự Án

```text
Lerobot_project/
├── SO-ARM100/                             # Repository gốc SO-ARM100 (CAD, STL, MJCF)
│   └── Simulation/
│       └── SO101/
│           ├── scene.xml                  # File cảnh mô phỏng cơ bản
│           ├── so101_new_calib.xml        # Robot XML (chuẩn calib giữa dải góc)
│           ├── so101_old_calib.xml        # Robot XML (chuẩn calib duỗi thẳng)
│           ├── joints_properties.xml      # Thông số động cơ STS3215
│           └── assets/                    # File 3D mesh (.stl, .part)
│
├── github_export/
│   └── so101-sim2real-policy-research/    # Framework nghiên cứu Sim2Real SO-101
│       ├── src/
│       │   └── sim_mujoco/
│       │       ├── teleop_leader_to_mujoco.py      # Teleop trực tiếp
│       │       ├── collect_sim_dataset.py          # Script thu thập dataset
│       │       ├── probe_env.py                    # Kiểm tra môi trường & camera
│       │       └── config_sim_collect_real_like.json # Config thu thập dữ liệu
│       ├── calibration/                   # Thư mục lưu file calib của Leader
│       ├── datasets/                      # Thư mục lưu dữ liệu thu thập được
│       └── reference/sim_assets/...       # Scene mô phỏng mở rộng (có camera, bàn gắp)
└── lerobot/                               # Thư viện LeRobot lõi
```

---

## 2. Chuẩn Bị Môi Trường & Phần Cứng

### 2.1. Kích hoạt môi trường Python (Conda)
Môi trường Conda đã được cài đặt sẵn đầy đủ thư viện `lerobot`, `mujoco`, `opencv-python`:

```bash
conda activate lerobot
```

### 2.2. Kết nối và cấp quyền cổng Serial cho cánh tay Leader
Cắm cáp USB của mạch điều khiển cánh tay Leader (ví dụ Waveshare Bus Servo Board) vào máy tính Linux.

Kiểm tra tên cổng kết nối:
```bash
ls /dev/ttyACM* || ls /dev/ttyUSB*
```
*(Thường cổng sẽ có dạng `/dev/ttyACM0` hoặc `/dev/ttyUSB0`)*

Cấp quyền truy cập cổng Serial cho user hiện tại:
```bash
sudo chmod 666 /dev/ttyACM0
```
*(Hoặc thêm vĩnh viễn user vào nhóm dialout: `sudo usermod -a -G dialout $USER` và đăng nhập lại).*

---

## 3. Hiệu Chuẩn (Calibration) Cánh Tay Leader Thật

Nếu lần đầu sử dụng hoặc sau khi tháo lắp lại cánh tay Leader, cần chạy lệnh calibrate để lưu dải góc của 6 động cơ:

```bash
cd /home/anhquoc2910/Lerobot_project/github_export/so101-sim2real-policy-research

python -m lerobot.scripts.lerobot_calibrate \
  --teleop.type=so101_leader \
  --teleop.port=/dev/ttyACM0 \
  --teleop.id=my_so101_leader \
  --teleop.calibration_dir="$PWD/calibration"
```

> **Cách thực hiện khi calibrate:**
> - Di chuyển từng khớp của cánh tay Leader từ điểm cực tiểu (min) đến cực đại (max) theo hướng dẫn trên terminal.
> - Sau khi hoàn tất, file calibration JSON sẽ được lưu vào thư mục `calibration/my_so101_leader.json`.

---

## 4. Kiểm Tra Teleoperation Trực Tiếp (Leader Thật $\rightarrow$ MuJoCo Ảo)

Trước khi thu data, hãy chạy script test teleop để kiểm tra độ trễ và độ chính xác của chuyển động giữa tay thật và tay ảo:

```bash
cd /home/anhquoc2910/Lerobot_project/github_export/so101-sim2real-policy-research

python src/sim_mujoco/teleop_leader_to_mujoco.py \
  --teleop.type=so101_leader \
  --teleop.port=/dev/ttyACM0 \
  --teleop.id=my_so101_leader \
  --teleop.calibration_dir="$PWD/calibration" \
  --teleop.use_degrees=true \
  --xml_path="reference/sim_assets/so_arm100_simulation/SO101/scene.xml"
```

- **Màn hình hiển thị**: Cửa sổ tương tác 3D MuJoCo và cửa sổ xem trước camera góc trên (`top_cam`) có lưới căn chỉnh.
- **Thao tác**: Di chuyển tay Leader thật, cánh tay Follower trong MuJoCo sẽ di chuyển đồng bộ theo thời gian thực.
- Nhấn `Q` hoặc `Esc` trên cửa sổ camera để thoát.

---

## 5. Thu Thập Dataset Mô Phỏng (Dataset Collection)

Quy trình thu thập dữ liệu hoạt động theo cơ chế **`real-like` (Continuous Physics)**:
- Môi trường chỉ khởi tạo physics 1 lần khi bắt đầu session.
- Không tự động reset đột ngột vật thể giữa các episode, giúp thao tác gắp - thả diễn ra mượt mà và tự nhiên tương tự quy trình thu trên robot thật.

### 5.1. Probe kiểm tra cấu hình trước khi thu
Kiểm tra xem các camera và kích thước action 6 khớp đã sẵn sàng chưa:

```bash
cd /home/anhquoc2910/Lerobot_project/github_export/so101-sim2real-policy-research

python src/sim_mujoco/probe_env.py \
  --config src/sim_mujoco/config_sim_collect_real_like.json
```

### 5.2. Chạy lệnh thu thập Episode

```bash
cd /home/anhquoc2910/Lerobot_project/github_export/so101-sim2real-policy-research

python src/sim_mujoco/collect_sim_dataset.py \
  --config=src/sim_mujoco/config_sim_collect_real_like.json \
  --teleop.type=so101_leader \
  --teleop.port=/dev/ttyACM0 \
  --teleop.id=my_so101_leader \
  --teleop.calibration_dir="$PWD/calibration" \
  --teleop.use_degrees=true \
  --dataset.repo_id=local/so101_mujoco_dataset \
  --dataset.root="$PWD/datasets/so101_mujoco_dataset" \
  --dataset.single_task="Stack 3 blocks: black at the bottom, green in the middle, white on top." \
  --dataset.fps=30 \
  --recording.control_fps=30 \
  --dataset.num_episodes=30 \
  --dataset.episode_time_s=180 \
  --env.reset_between_episodes=false \
  --display_data=true \
  --no-push
```

---

## 6. Phím Tắt Điều Khiển Trong Khi Thu Thập

Trong lúc chương trình thu dữ liệu đang chạy, bạn tương tác trực tiếp qua cửa sổ Camera Preview:

| Trạng thái | Phím bấm | Chức năng |
| :--- | :--- | :--- |
| **Preparation Mode** (Chế độ chuẩn bị) | **`Enter`** / **`Space`** / **`Mũi tên phải`** | Bắt đầu ghi hình episode |
| | **`Q`** / **`Esc`** | Thoát chương trình (không tạo episode trống) |
| **Recording Mode** (Đang ghi episode) | **`Enter`** / **`Space`** / **`Mũi tên phải`** | **Lưu episode thành công** và chuyển về chế độ chuẩn bị |
| | **`Backspace`** / **`R`** / **`Mũi tên trái`** | **Hủy bỏ episode hiện tại** (khi gắp trượt/rơi vật) và chuẩn bị ghi lại |
| | **`Q`** / **`Esc`** | Lưu phần đã ghi và dừng toàn bộ phiên làm việc |

---

## 7. Kiểm Tra và Tiếp Tục Thu Dữ Liệu (Resume)

### 7.1. Kiểm tra cấu trúc Dataset đã lưu
Dữ liệu được lưu tại thư mục `datasets/so101_mujoco_dataset/`:
- `meta/info.json`: Chứa thông tin camera, fps, số lượng episode và frame.
- `meta/episodes.jsonl`: Thông tin chi tiết từng episode.
- `data/chunk-*.parquet`: Dữ liệu góc khớp và hành động (state & action).
- `videos/`: Video quan sát chất lượng cao từ các camera góc nhìn (`top`, `front`).

Chạy lệnh kiểm tra tính toàn vẹn:
```bash
python src/sim_mujoco/validate_preupload.py \
  --config src/sim_mujoco/config_sim_collect_real_like.json
```

### 7.2. Thu nối tiếp vào dataset cũ (Resume)
Nếu bạn đã thu 10 episodes và muốn thu thêm 20 episodes nữa vào cùng dataset:

Thêm tham số `--resume=true` và truyền số episode muốn thu thêm qua `--dataset.num_episodes=20`:

```bash
python src/sim_mujoco/collect_sim_dataset.py \
  --config=src/sim_mujoco/config_sim_collect_real_like.json \
  --teleop.type=so101_leader \
  --teleop.port=/dev/ttyACM0 \
  --teleop.id=my_so101_leader \
  --teleop.calibration_dir="$PWD/calibration" \
  --teleop.use_degrees=true \
  --dataset.repo_id=local/so101_mujoco_dataset \
  --dataset.root="$PWD/datasets/so101_mujoco_dataset" \
  --dataset.num_episodes=20 \
  --resume=true \
  --no-push
```

---

## 8. Xử Lý Các Vấn Đề Thường Gặp (Troubleshooting)

1. **Lỗi `Permission denied: '/dev/ttyACM0'`:**
   - Chạy lệnh: `sudo chmod 666 /dev/ttyACM0`

2. **Lỗi `Calibration file not found`:**
   - Kiểm tra xem ID truyền vào (`--teleop.id=my_so101_leader`) có đúng với tên file trong thư mục `calibration/` hay chưa.

3. **Cánh tay ảo bị rung hoặc phản hồi chậm:**
   - Đảm bảo tham số `--dataset.fps=30` và `--recording.control_fps=30` khớp nhau.
   - Kiểm tra tải CPU hoặc GPU khi render video preview.

4. **Khớp kẹp (Gripper) không gắp dính vật thể:**
   - Trong file `reference/sim_assets/so_arm100_simulation/SO101/scene.xml`, bề mặt tiếp xúc của má kẹp (`gripper_collision`) đã được tối ưu hệ số ma sát cao (`friction="1.0 0.01 0.001"`). Đảm bảo sử dụng đúng file scene này.
