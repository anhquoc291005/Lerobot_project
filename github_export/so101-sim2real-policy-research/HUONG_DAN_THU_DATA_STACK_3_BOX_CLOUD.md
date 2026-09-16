# Hướng Dẫn Thu Thập Dữ Liệu Mô Phỏng SO-101 (Dataset: stack_3_box_white_green_blue_simulation)

Tài liệu này hướng dẫn chi tiết quy trình điều khiển **SO-101 Leader thật** để thao tác gắp và xếp chồng 3 khối hộp trong môi trường mô phỏng **MuJoCo** (cấu hình khối **TRẮNG** ở đáy, khối **XANH LÁ** ở giữa, khối **XANH DƯƠNG** ở trên cùng, camera **Front Cam tại ô E9**).

Dữ liệu được lưu vào thư mục cục bộ **`datasets/stack_3_box_white_green_blue_simulation`** và tự động đồng bộ lên Hugging Face Cloud. **Mỗi lần chạy lệnh, hệ thống sẽ tự động GHI NỐI TIẾP (Resume) các episode mới mà tuyệt đối KHÔNG ghi đè hay thay thế các tập đã thu trước đó**.

---

## 1. Thông Tin Dataset & Cấu Hình Đã Thiết Lập

- **Thư mục lưu trên máy**: `datasets/stack_3_box_white_green_blue_simulation`
- **Kho lưu trữ trên Cloud**: [**anhquoc29/stack_3_box_white_green_blue_simulation**](https://huggingface.co/datasets/anhquoc29/stack_3_box_white_green_blue_simulation)
- **Chuẩn định dạng**: LeRobot Dataset **`v3.0`**
- **Tốc độ lấy mẫu & ghi hình**: **`30 FPS`** (`fps=30`, `control_fps=30`, `frame_skip=33`)
- **Vật thể thao tác**: 3 khối hộp kích thước **$30 \times 30 \times 30$ mm**
  - Khối **TRẮNG**: Đặt đáy tháp (trên ô đích `stack_target_zone` ở ô C4 sát biên D4).
  - Khối **XANH LÁ**: Đặt ở giữa (chồng lên khối Trắng).
  - Khối **XANH NƯỚC BIỂN**: Đặt ở trên cùng (chồng lên khối Xanh lá).
- **Góc nhìn Camera**:
  - `observation.images.top`: Camera trên trần nhìn vuông góc xuống mặt bàn ($480 \times 640 \times 3$, video `av1`).
  - `observation.images.front`: Camera góc nghiêng phía trước đặt tại **ô E9** nhìn chéo vào bàn ($480 \times 640 \times 3$, video `av1`).
- **Cơ chế nối tiếp & bảo vệ dữ liệu**:
  - `auto_resume_if_root_exists = True`: Tự động nhận diện số tập đã có và ghi tiếp (ví dụ đã có 5 tập thì lần chạy sau sẽ ghi tiếp tập 6, 7, 8...).
  - `cleanup_local_after_push = False`: **Không xóa thư mục cục bộ** sau khi đẩy lên Cloud, bảo đảm thư mục luôn còn nguyên để các lần chạy sau tiếp tục nối thêm tập mới.
  - `push_to_hub = True`: Mỗi khi kết thúc một phiên thu thập, toàn bộ dữ liệu (cũ + mới) được tự động tải lên Hugging Face.

---

## 2. Chuẩn Bị Trước Khi Chạy

### 2.1. Cắm cáp USB mạch Leader & Cấp quyền cổng kết nối
```bash
# Kiểm tra cổng kết nối (thường là /dev/ttyACM0 hoặc /dev/ttyUSB0)
ls /dev/ttyACM* /dev/ttyUSB*

# Cấp quyền đọc/ghi cho cổng
sudo chmod 666 /dev/ttyACM0
```

### 2.2. Kích hoạt môi trường Conda & Thiết lập biến môi trường
```bash
# 1. Chuyển vào thư mục mã nguồn
cd /home/anhquoc2910/Lerobot_project/github_export/so101-sim2real-policy-research

# 2. Kích hoạt môi trường conda lerobot
conda activate lerobot

# 3. Khai báo thông tin tay Leader
export REAL_TELEOP_PORT=/dev/ttyACM0
export REAL_TELEOP_ID=my_so101_leader
export REAL_TELEOP_USE_DEGREES=true
export REAL_TELEOP_CALIBRATION_DIR=/home/anhquoc2910/Lerobot_project/github_export/so101-sim2real-policy-research/calibration
```

---

## 3. Lệnh Chạy Thu Thập Dữ Liệu (Lưu thêm Episode, không thay thế)

### Lệnh khuyến nghị (File config đã thiết lập sẵn toàn bộ):

```bash
/home/anhquoc2910/miniforge3/envs/lerobot/bin/python src/sim_mujoco/collect_sim_dataset.py \
  --config src/sim_mujoco/config_sim_collect_stack_3_blocks.json
```

> **Cơ chế hoạt động:**
> - **Lần chạy đầu tiên**: Tạo mới thư mục `datasets/stack_3_box_white_green_blue_simulation` và bắt đầu từ Episode 0.
> - **Các lần chạy tiếp theo**: Hệ thống sẽ thông báo:
>   `Dataset root already exists with valid meta/info.json and auto_resume_if_root_exists=true. Switching to resume mode.`
>   Và tự động thu nối tiếp từ Episode tiếp theo (ví dụ: `Số tập hiện có trên máy: 10` -> tập thu mới sẽ là Episode 10, 11...).

*(Mẹo: Bạn có thể thêm `--dataset.num_episodes 10` nếu trong phiên đó bạn chỉ muốn thu thêm 10 tập rồi nghỉ).*

---

## 4. Lệnh Chạy Tường Minh Bằng Tham Số CLI (Tùy chọn)

Nếu bạn không muốn phụ thuộc vào file config mà muốn chỉ định tường minh mọi tham số trên dòng lệnh:

```bash
/home/anhquoc2910/miniforge3/envs/lerobot/bin/python src/sim_mujoco/collect_sim_dataset.py \
  --config src/sim_mujoco/config_sim_collect_stack_3_blocks.json \
  --dataset.root datasets/stack_3_box_white_green_blue_simulation \
  --dataset.repo_id anhquoc29/stack_3_box_white_green_blue_simulation \
  --dataset.cleanup_local_after_push false \
  --resume
```

---

## 5. Quy Trình Thao Tác & Phím Tắt Trong Khi Thu

Khi lệnh khởi chạy, cửa sổ OpenCV Preview hiển thị 2 camera (`top` có lưới tọa độ và `front` góc E9) sẽ mở ra:

1. **Chuẩn bị (Preparation Mode)**:
   - Cầm tay Leader di chuyển cánh tay robot ảo đến tư thế xuất phát mong muốn.
   - Nhấp chuột vào cửa sổ Preview và bấm **`ENTER`** (hoặc **`SPACE`**, phím **`->`**) để **BẮT ĐẦU GHI**.
2. **Thao tác gắp & xếp chồng (Recording Mode)**:
   - Gắp khối **TRẮNG** đặt vào vị trí đích ô C4.
   - Gắp khối **XANH LÁ** đặt chồng lên khối TRẮNG.
   - Gắp khối **XANH DƯƠNG** đặt chồng lên khối XANH LÁ.
3. **Lưu hoặc Hủy Episode**:
   - **Xếp thành công**: Bấm **`ENTER`** (hoặc **`SPACE`**, **`->`**) để **LƯU EPISODE**.
   - **Bị trượt/đổ**: Bấm **`r`** (hoặc **`Backspace`**, **`<-`**) để **HỦY VÀ GHI LẠI** tập đó.
4. **Dừng phiên thu thập**:
   - Bấm **`q`** hoặc **`ESC`** trên cửa sổ preview để kết thúc phiên thu thập.
   - Hệ thống sẽ đóng gói dữ liệu và tự động đẩy toàn bộ các tập (bao gồm cả tập cũ và tập mới) lên Hugging Face.

---

## 6. Xem & Kiểm Tra Dataset Trực Tiếp

1. **Xem trực tiếp trên Hugging Face Hub**:
   👉 [https://huggingface.co/datasets/anhquoc29/stack_3_box_white_green_blue_simulation](https://huggingface.co/datasets/anhquoc29/stack_3_box_white_green_blue_simulation)

2. **Xem trực quan video và đồ thị hành động (Visualize Online)**:
   👉 [https://huggingface.co/spaces/lerobot/visualize_dataset?path=%2Fanhquoc29%2Fstack_3_box_white_green_blue_simulation%2Fepisode_0](https://huggingface.co/spaces/lerobot/visualize_dataset?path=%2Fanhquoc29%2Fstack_3_box_white_green_blue_simulation%2Fepisode_0)
