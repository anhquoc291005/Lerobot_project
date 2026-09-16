# Thu dataset SO-101 trong MuJoCo theo quy trình giống robot thật

Hướng dẫn này dùng SO-101 leader thật để điều khiển follower trong MuJoCo và
ghi dữ liệu theo định dạng `LeRobotDataset`. Chế độ `real-like` chỉ khởi tạo
MuJoCo một lần khi bắt đầu chương trình, sau đó giữ liên tục trạng thái robot,
vật thể và physics giữa các episode.

Config mẫu:

```text
src/sim_mujoco/config_sim_collect_real_like.json
```

Script chính:

```text
src/sim_mujoco/collect_sim_dataset.py
```

> [!NOTE]
> Nếu bạn muốn thực hiện nhiệm vụ **xếp chồng 3 khối hộp (Stack 3 Blocks)** với cơ chế tự động random vị trí mỗi episode và chạy demo không cần tay leader, xem tài liệu chuyên biệt tại: [`docs/so101/so101_stack_3_blocks_sim.md`](so101_stack_3_blocks_sim.md).

## 1. Hành vi của chế độ real-like

Với:

```json
"reset_between_episodes": false
```

quy trình là:

```text
Khởi tạo MuJoCo một lần
        ↓
Preparation mode: leader hoạt động, chưa ghi dữ liệu
        ↓ Enter/Space/Right
Ghi episode
        ↓ Enter/Space/Right
Lưu episode, giữ nguyên trạng thái MuJoCo
        ↓
Preparation mode cho episode tiếp theo
```

Trong preparation mode, leader vẫn điều khiển robot MuJoCo nhưng các frame
không được ghi. Có thể dùng giai đoạn này để đưa tay robot hoặc vật thể về trạng
thái bắt đầu.

Không có `env.reset()` giữa các episode, vì vậy chương trình không tự:

- đưa robot về home;
- đặt lại vật và container;
- random lại vị trí;
- đặt lại vận tốc hoặc thời gian physics.

MuJoCo vẫn phải được khởi tạo một lần khi chương trình bắt đầu. Khi đóng chương
trình rồi chạy resume, MuJoCo cũng được khởi tạo lại một lần; resume chỉ tiếp tục
dataset, không thể phục hồi trạng thái physics của process cũ.

## 2. Chuẩn bị môi trường

Repo LeRobot cần được cài trong environment đang dùng:

```bash
cd /home/vietanh/lerobot
uv pip install -e ".[core_scripts,training,feetech,diffusion]"

cd /home/vietanh/lerobot/github_export/so101-sim2real-policy-research
uv pip install -r requirements.txt
```

Nếu máy đang lỗi `uv` do Snap/AppArmor nhưng Conda environment `lerobot` đã có
đủ package, có thể chạy các lệnh Python bên dưới trực tiếp bằng `python`.

Kiểm tra leader port:

```bash
lerobot-find-port
```

Hoặc:

```bash
ls /dev/ttyACM*
```

Khai báo leader trước khi chạy collector:

```bash
export REAL_TELEOP_PORT=/dev/ttyACM0
export REAL_TELEOP_ID=my_so101_leader
export REAL_TELEOP_USE_DEGREES=true
export REAL_TELEOP_CALIBRATION_DIR=/home/vietanh/lerobot/github_export/so101-sim2real-policy-research/calibration
```

Thay `/dev/ttyACM0` và `REAL_TELEOP_ID` theo leader/calibration thực tế.

## 3. Kiểm tra config trước khi thu

Mở:

```text
src/sim_mujoco/config_sim_collect_real_like.json
```

Các trường cần kiểm tra:

```json
{
  "dataset": {
    "repo_id": "vasco281204/so101_mujoco_real_like",
    "root": "datasets/so101_mujoco_real_like",
    "robot_type": "so_follower",
    "fps": 20,
    "num_episodes": 30,
    "episode_time_s": 120,
    "video_backend": "pyav",
    "vcodec": "libsvtav1",
    "data_files_size_in_mb": 100,
    "video_files_size_in_mb": 200,
    "push_to_hub": true
  }
}
```

Ý nghĩa:

- `repo_id`: dataset được upload lên
  `vasco281204/so101_mujoco_real_like`.
- `root`: thư mục dataset local; phải dành riêng cho dữ liệu sim.
- `robot_type`, `fps` và camera keys phải giống dataset real nếu hai dataset sẽ
  được aggregate hoặc dùng chung một pipeline train.
- `num_episodes`: số episode mới cần ghi trong lần chạy hiện tại.
- `episode_time_s`: thời lượng tối đa một episode.
- `vcodec=libsvtav1` tạo video AV1; backend đọc video là `pyav`.
- `push_to_hub=true`: tự upload sau khi collection được finalize thành công.

Dataset real tham chiếu có 44.821 frame, 30 episode ở 20 FPS, tương đương trung
bình khoảng 74,7 giây mỗi episode. Config đặt giới hạn 120 giây và cho phép kết
thúc sớm bằng `Enter/Space/Right`, thay vì ép mọi episode dài đúng 120 giây.

Các trường `codebase_version`, `total_episodes`, `total_frames`, `splits`,
`data_path` và `video_path` được LeRobot sinh tự động khi tạo/finalize dataset;
không chép chúng trực tiếp vào config collector.

Không dùng repo ID hoặc root của dataset real. Nếu dùng chung root và feature
tương thích, dữ liệu sim có thể bị nối vào dataset real.

Config mẫu dùng grid preview `5x7`, phù hợp với model/grid MuJoCo hiện tại. Không
đổi riêng preview thành `5x9` nếu chưa hiệu chỉnh lại tọa độ MuJoCo tương ứng.

## 4. Probe môi trường trước khi ghi

Từ thư mục gốc của repo:

```bash
cd /home/vietanh/lerobot/github_export/so101-sim2real-policy-research

python src/sim_mujoco/probe_env.py \
  --config src/sim_mujoco/config_sim_collect_real_like.json
```

Nếu chạy bằng `uv`:

```bash
uv run python src/sim_mujoco/probe_env.py \
  --config src/sim_mujoco/config_sim_collect_real_like.json
```

Probe phải nhận đúng:

- action shape gồm 6 khớp;
- `agent_pos`;
- camera `pixels.top`;
- camera `pixels.gripper`;
- ảnh `480x640x3`.

Trong model XML, camera vật lý thứ hai có tên `front_cam`. Environment đưa ảnh
này ra qua key nội bộ `pixels.gripper`, còn adapter đổi tên feature dataset
thành `observation.images.front`:

```json
"gripper_camera_name": "front_cam",
"gripper_image_key": "pixels.gripper",
"secondary_dataset_camera_key": "front"
```

Nhờ vậy dataset sim có camera keys `top` và `front`, giống lệnh thu dataset real
trong `docs/real_dataset_collection_guide.md`.

## 5. Thu dataset mới

Đảm bảo `dataset.root` chưa tồn tại, sau đó chạy:

```bash
cd /home/vietanh/lerobot/github_export/so101-sim2real-policy-research

python src/sim_mujoco/collect_sim_dataset.py \
  --config=src/sim_mujoco/config_sim_collect_real_like.json \
  --teleop.type=so101_leader \
  --teleop.port=/dev/ttyACM0 \
  --teleop.id=my_so101_leader \
  --teleop.calibration_dir="$PWD/calibration" \
  --teleop.use_degrees=true \
  --dataset.repo_id=vasco281204/so101_mujoco_real_like \
  --dataset.root="$PWD/datasets/so101_mujoco_real_like" \
  --dataset.single_task="Pick the red block and place it into the container." \
  --dataset.fps=30 \
  --recording.control_fps=30 \
  --dataset.num_episodes=200 \
  --dataset.episode_time_s=12000 \
  --dataset.reset_time_s=1 \
  --dataset.push_to_hub=true \
  --dataset.private=false \
  --env.reset_between_episodes=false \
  --display_data=true
```

Nếu muốn tự nhấn Enter trước mỗi episode thay vì đếm ngược 10 giây, thay:

```bash
--dataset.reset_time_s=10
```

bằng:

```bash
--recording.require_enter_before_episode=true
```

Collector tự push lên `vasco281204/so101_mujoco_real_like` sau khi session kết
thúc thành công. Nếu chỉ muốn test local, thêm `--no-push`; cờ này ghi đè cấu
hình upload trong lần chạy đó.

### Trước mỗi episode

Collector mở preparation mode:

```text
Leader control: đang hoạt động
Dataset recording: chưa hoạt động
MuJoCo reset: không
```

Chuẩn bị robot/vật thể rồi nhấn:

- `Enter`, `Space` hoặc `Right`: bắt đầu ghi episode;
- `Q` hoặc `Esc`: kết thúc session mà không tạo episode rỗng.

### Trong khi ghi

- `Enter`, `Space` hoặc `Right`: kết thúc sớm và lưu episode;
- `Left`, `R` hoặc `Backspace`: xóa buffer episode hiện tại và ghi lại;
- `Q` hoặc `Esc`: lưu phần episode đã ghi rồi kết thúc session.

Khi chọn ghi lại trong chế độ continuous, buffer dataset bị xóa nhưng trạng thái
MuJoCo không quay về đầu episode. Chương trình trở lại preparation mode để người
dùng tự chuẩn bị lại trạng thái.

Tránh dùng `Ctrl+C` khi có thể. Config mẫu đặt:

```json
"save_partial_episode_on_interrupt": false
```

nên episode đang ghi sẽ bị bỏ nếu bị `KeyboardInterrupt`.

## 6. Kiểm tra dataset local

Sau khi dừng collector:

```bash
python src/sim_mujoco/validate_preupload.py \
  --config src/sim_mujoco/config_sim_collect_real_like.json
```

Kiểm tra thêm số episode và trạng thái file:

```bash
find datasets/so101_mujoco_real_like/meta -maxdepth 2 -type f
find datasets/so101_mujoco_real_like/data -type f | head
```

Không sửa trực tiếp file Parquet/video khi collector hoặc encoder vẫn đang chạy.

## 7. Resume dataset

Resume cần giữ nguyên các trường sau so với lần thu đầu:

- `dataset.repo_id`;
- `dataset.root`;
- `dataset.robot_type`;
- `dataset.fps`;
- camera names và kích thước ảnh;
- tên và số chiều state/action;
- `use_videos`.

Chỉnh `dataset.num_episodes` thành số episode **muốn ghi thêm** trong lần resume.
Ví dụ dataset đã có 10 episode và muốn ghi thêm 5:

```json
"num_episodes": 5
```

Sau đó chạy:

```bash
cd /home/vietanh/lerobot/github_export/so101-sim2real-policy-research

python src/sim_mujoco/collect_sim_dataset.py \
  --config=src/sim_mujoco/config_sim_collect_real_like.json \
  --teleop.type=so101_leader \
  --teleop.port=/dev/ttyACM0 \
  --teleop.id=my_so101_leader \
  --teleop.calibration_dir="$PWD/calibration" \
  --dataset.repo_id=vasco281204/so101_mujoco_real_like \
  --dataset.root="$PWD/datasets/so101_mujoco_real_like" \
  --dataset.fps=20 \
  --recording.control_fps=20 \
  --dataset.num_episodes=5 \
  --dataset.episode_time_s=120 \
  --dataset.reset_time_s=10 \
  --dataset.push_to_hub=true \
  --env.reset_between_episodes=false \
  --display_data=true \
  --resume=true
```

Collector gọi `LeRobotDataset.resume()`, kiểm tra feature compatibility và dùng
`dataset.num_episodes` hiện có làm index cho episode tiếp theo.

Ví dụ:

```text
Dataset hiện có: 10 episode
dataset.num_episodes trong config: 5
Episode mới: 10, 11, 12, 13, 14
Tổng sau lần chạy: 15 episode
```

Sau khi restart để resume, MuJoCo khởi tạo một lần ở episode kế tiếp. Trạng thái
robot/vật thể từ process trước không được phục hồi. Từ đó trở đi, các episode
trong cùng session resume tiếp tục chạy mà không reset.

Nếu resume báo feature mismatch, không ép ghi vào dataset cũ. Kiểm tra lại config
hoặc tạo `dataset.root` và `repo_id` mới.

## 8. Upload Hugging Face

Config mẫu đã bật:

```json
"repo_id": "vasco281204/so101_mujoco_real_like",
"push_to_hub": true
```

Đăng nhập và xác nhận token có quyền ghi vào namespace `vasco281204`:

```bash
hf auth login
hf auth whoami
```

Chạy collector không có `--no-push`. Dataset được finalize và upload khi session
kết thúc bình thường. Collector bỏ qua push nếu collection gặp exception bất
ngờ.

Nếu chỉ muốn test local, thêm:

```text
--no-push
```

Cờ này chỉ tắt upload trong lần chạy, không sửa `push_to_hub=true` trong file
config.

## 9. Các lỗi thường gặp

### Không tìm thấy leader port

```text
Set REAL_TELEOP_PORT to your leader serial port
```

Khai báo lại:

```bash
export REAL_TELEOP_PORT=/dev/ttyACM0
```

### Không tìm thấy calibration

Kiểm tra:

```bash
ls calibration
```

`REAL_TELEOP_ID` phải khớp tên calibration hoặc calibration đã lưu trong thư mục
được truyền qua `REAL_TELEOP_CALIBRATION_DIR`.

### Dataset root đã tồn tại

Nếu muốn tiếp tục dataset cũ, dùng `--resume`. Nếu muốn tạo thí nghiệm mới, đổi
cả `dataset.repo_id` và `dataset.root`; không xóa dataset cũ khi chưa sao lưu.

### Preview không xuất hiện

Kiểm tra `opencv-python`, display session và:

```json
"display_camera_preview": true
```

Không nên thu real-like mà thiếu preview vì preview chứa phím bắt đầu/kết thúc
episode và giúp xác nhận camera trước khi ghi.
