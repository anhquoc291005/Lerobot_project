# Chuyển setup camera và robot sang máy lab

Repo GitHub lưu calibration, cấu hình camera, vùng thao tác, script và bản vá
source LeRobot. Trọng số policy tải riêng từ Hugging Face. Không cần đưa dataset
huấn luyện lên GitHub để chạy rollout với checkpoint đã có đủ processor/stats.

## Trên máy hiện tại

```bash
cd /home/anhquoc2910/Lerobot_project
python3 scripts/export_lab_setup.py
git add scripts/ patches/ calibration/ \
  my_awesome_follower_arm.json my_awesome_leader_arm.json \
  teleop_2cam_workspace.sh workspace_boundary.json edit_workspace_boundary.py \
  record_real_stack_3_box.py record_real_stack_3_box.sh record_real_stack_hub.py \
  tests/ SO-ARM101_camera_wrist_mount.stl \
  HUONG_DAN_CHUYEN_SETUP_SANG_LAB.md HUONG_DAN_VUNG_GIOI_HAN_CAMERA.md \
  HUONG_DAN_THU_DATA_REAL_WHITE_GREEN_BLUE.md
git diff --cached --stat
git commit -m "Save portable camera and robot setup for lab rollout"
git push -u origin main
```

Chạy lại export trước mỗi lần lưu thay đổi source LeRobot. Export lấy thay đổi
trong `lerobot/src` và `lerobot/tests`, gồm cả file mới chưa track. Phiên bản gốc
được ghi trong `patches/lerobot-base-commit.txt`. Nếu thay đổi dependency trong
`lerobot/pyproject.toml` hoặc `uv.lock`, cần cập nhật quy trình export trước khi
chuyển môi trường; script hiện chỉ xuất source và tests.

## Trên máy lab lần đầu (Linux)

```bash
git clone https://github.com/anhquoc291005/Lerobot_project.git
cd Lerobot_project
bash scripts/restore_lab_setup.sh
```

Repo private cần đăng nhập GitHub bằng tài khoản có quyền truy cập; có thể clone
bằng SSH nếu máy lab đã cấu hình SSH key.

Cài `uv` nếu máy lab chưa có, rồi tạo môi trường Python 3.12 từ lockfile:

```bash
uv python install 3.12
cd lerobot
uv sync --locked --python 3.12 --extra core_scripts --extra feetech
# Source mô phỏng tùy chỉnh được import cùng envs, nên cần MuJoCo cả khi chạy thật.
uv pip install --python .venv/bin/python 'mujoco==3.12.0'
cd ..
export LEROBOT_PYTHON="$PWD/lerobot/.venv/bin/python"
"$LEROBOT_PYTHON" -c 'import torch; print("CUDA:", torch.cuda.is_available())'
```

Lockfile cố định dependency trong phần chính; MuJoCo được cài bổ sung riêng.
CUDA còn phụ thuộc GPU và driver máy lab. Nếu thiếu FFmpeg hoặc quyền truy cập
USB/serial, cần cấu hình trên máy lab. Git không chuyển driver/quyền hệ thống.

## Kiểm tra phần cứng và policy

```bash
ls /dev/serial/by-id/
ls /dev/v4l/by-id/
"$LEROBOT_PYTHON" -m lerobot.scripts.lerobot_find_cameras opencv
```

Cấu hình hiện tại: follower `/dev/ttyACM0`, top `/dev/video4`, left `/dev/video2`,
640×480, 30 FPS, MJPG. Cổng có thể đổi trên máy lab; chọn đúng thiết bị trước khi
chạy. Ưu tiên đường dẫn `/dev/serial/by-id/` và `/dev/v4l/by-id/` nếu có.

Calibration chuẩn dùng trong rollout là `calibration/my_awesome_follower_arm.json`.
Chỉ dùng lại calibration nếu đó là chính cánh tay đã calibrate. Nếu lab dùng một
cánh tay khác cùng mẫu, cần calibrate riêng. Giữ đúng vị trí camera, góc nhìn và
đơn vị khớp như lúc thu data; JSON không khôi phục vị trí vật lý của camera.

## Rollout policy từ Hugging Face

Thay `TAI_KHOAN/TEN_POLICY` và mô tả task bằng policy thực tế. Policy private cần
đăng nhập Hugging Face trên máy lab (`hf auth login`). Nếu checkpoint ở thư mục
con trên Hub, tải thư mục `pretrained_model` đầy đủ và dùng đường dẫn local.

```bash
export LEROBOT_PYTHON="$PWD/lerobot/.venv/bin/python"
"$LEROBOT_PYTHON" -m lerobot.scripts.lerobot_rollout \
  --strategy.type=base \
  --policy.path=TAI_KHOAN/TEN_POLICY \
  --device=cuda \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM0 \
  --robot.id=my_awesome_follower_arm \
  --robot.calibration_dir="$PWD/calibration" \
  --robot.use_degrees=true \
  --robot.cameras="{top: {type: opencv, index_or_path: /dev/video4, width: 640, height: 480, fps: 30, fourcc: MJPG, backend: V4L2}, left: {type: opencv, index_or_path: /dev/video2, width: 640, height: 480, fps: 30, fourcc: MJPG, backend: V4L2}}" \
  --fps=30 \
  --duration=30 \
  --task="MO_TA_TASK_LUC_TRAIN" \
  --display_data=true
```

Lệnh trên bắt đầu điều khiển robot thật. Bản này dùng `lerobot-rollout`; không
truyền `--policy.path` vào `lerobot-record`. Chế độ `base` không ghi dataset.
`use_degrees=true` khớp script thu data hiện tại; kiểm tra lại nếu dùng policy
được train từ dataset khác. Policy khác ACT có thể cần thêm extra dependency.

Camera policy phải nhận đúng key `observation.images.top` và
`observation.images.left`. Nếu policy dùng `front`, chỉ đổi tên hoặc dùng
`--rename_map` khi đã xác nhận góc nhìn tương ứng; đổi tên không sửa được khác
biệt góc nhìn. Vùng `workspace_boundary.json` là đường biên hiển thị, không giới
hạn chuyển động; CLI rollout hiện chưa có cờ `--display_workspace`.

## Những lần kéo tiếp theo

```bash
git pull --ff-only
bash scripts/restore_lab_setup.sh
```

Nếu patch không đổi, script báo đã khôi phục. Khi patch mới làm thay đổi source
đang có, script dừng nếu checkout LeRobot có sửa đổi để giữ nguyên dữ liệu trên
máy lab. Cần lưu các sửa đổi đó và dùng checkout sạch trước khi áp bản mới.
