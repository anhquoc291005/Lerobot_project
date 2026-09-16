# Hướng Dẫn Chạy & Đánh Giá Mô Hình ACT: `vasco281204/act_stack_3_box_30x3030`

Tài liệu này tổng hợp hướng dẫn chi tiết quy trình chạy, đánh giá và kiểm thử mô hình **ACT Policy** (`vasco281204/act_stack_3_box_30x3030`) trên môi trường mô phỏng **MuJoCo SO-101** và định hướng triển khai thực tế.

---

## 1. Thông Tin Mô Hình & Môi Trường Đã Được Kiểm Tra

Hệ thống đã kiểm tra tự động và xác nhận môi trường đã sẵn sàng:

* **Môi trường Conda (`lerobot`)**:
  - Python: `3.12`
  - PyTorch: `2.11.0+cu128` (CUDA khả dụng: **True** - Hỗ trợ tính toán trực tiếp trên GPU).
  - Thư viện điều khiển & mô phỏng: **LeRobot 0.6.2**, **MuJoCo 3.12.0**, **Gymnasium 1.3.0**.
  - Tên môi trường Gym đã đăng ký: `so101_stack_3_blocks` (Gym ID: `so101/SO101Stack3Blocks-v0`).
* **Thông tin mô hình Hugging Face**:
  - Repo ID: [**`vasco281204/act_stack_3_box_30x3030`**](https://huggingface.co/vasco281204/act_stack_3_box_30x3030)
  - Dataset đối chiếu: [**`anhquoc29/stack_3_box_30x3030`**](https://huggingface.co/datasets/anhquoc29/stack_3_box_30x3030)
  - Kiến trúc: **ACT Policy** (Chunk size: 100, n_action_steps: 100, Backbone: ResNet18).
  - Dữ liệu đầu vào: Góc 6 khớp (`observation.state`) + 2 Camera RGB (`observation.images.top`, `observation.images.front` kích thước $480 \times 640$).
  - Các mốc checkpoint có sẵn trong repo: `020000`, `040000`, `060000`, `080000`, `100000`.
  - Checkpoint cao nhất (`100000`) **đã được tải và lưu trong bộ nhớ cache cục bộ**.

---

## 2. Bước Chuẩn Bị Môi Trường

Mở cửa sổ Terminal và kích hoạt môi trường conda `lerobot`:

```bash
# 1. Di chuyển vào thư mục dự án
cd /home/anhquoc2910/Lerobot_project

# 2. Kích hoạt môi trường conda
conda activate lerobot
```

---

## 3. Các Cách Chạy & Đánh Giá Mô Hình Trong Mô Phỏng MuJoCo

Script điều khiển chính: [`eval_act_sim.py`](./eval_act_sim.py) (đã được tối ưu tự động tìm kiếm checkpoint từ Hugging Face Hub).

### Cách 1: Chạy có giao diện trực quan thời gian thực (GUI Mode)
*(Khuyên dùng khi máy tính có màn hình desktop hoặc kết nối có hỗ trợ X11/GUI)*

Script sẽ mở cửa sổ OpenCV hiển thị trực quan đồng thời cả 2 camera: **Top Cam (có lưới tọa độ $5 \times 7$)** và **Front Cam**.

```bash
python eval_act_sim.py \
  --policy vasco281204/act_stack_3_box_30x3030 \
  --num_episodes 10
```

> **Các phím tắt điều khiển khi cửa sổ giao diện đang mở:**
> - `SPACE` hoặc `ENTER` hoặc `r`: Bỏ qua episode hiện tại, reset ngay lập tức sang episode mới với vị trí 3 khối hộp ngẫu nhiên mới.
> - `p`: Tạm dừng (Pause) / Tiếp tục mô phỏng để quan sát vị trí gắp của robot.
> - `q` hoặc `ESC`: Dừng chương trình ngay lập tức và in bảng thống kê kết quả.

---

### Cách 2: Chạy ngầm không cần màn hình (Headless) & Tự động xuất Video MP4
*(Khuyên dùng khi chạy từ xa qua SSH hoặc máy chủ không có giao diện đồ họa)*

Mô hình sẽ thực hiện rollout ngầm với tốc độ tính toán tối đa của GPU và xuất từng episode thành video MP4 chất lượng cao.

```bash
python eval_act_sim.py \
  --policy vasco281204/act_stack_3_box_30x3030 \
  --headless \
  --save_video \
  --video_dir outputs/eval_videos \
  --num_episodes 20
```

Video quay lại toàn bộ góc nhìn kép sẽ được lưu tại:
📁 `outputs/eval_videos/eval_episode_1.mp4`, `eval_episode_2.mp4`, ...

---

### Cách 3: Chọn checkpoint cụ thể (Ví dụ so sánh step 80,000 vs 100,000)
Mặc định script sẽ chọn checkpoint cao nhất (`100000`). Nếu bạn muốn chạy thử nghiệm các checkpoint ở bước huấn luyện trước:

```bash
# Chạy với checkpoint 80,000 steps
python eval_act_sim.py \
  --policy vasco281204/act_stack_3_box_30x3030 \
  --checkpoint 080000 \
  --num_episodes 10
```

### Cách 4: Chạy Rollout & Tự Động Đẩy Dữ Liệu Lên Hugging Face Hub
Nếu bạn muốn lưu lại toàn bộ dữ liệu rollout (bao gồm góc khớp `observation.state`, lệnh `action`, cùng 2 camera `top` và `front` chuẩn LeRobot Dataset v3.0) và tự động tải lên Hugging Face với tên repo **`anhquoc29/rollout_stact_3_box_sim`**:

```bash
python eval_act_sim.py \
  --policy vasco281204/act_stack_3_box_30x3030 \
  --num_episodes 20 \
  --save_dataset \
  --push_to_hub \
  --hub_repo_id anhquoc29/rollout_stact_3_box_sim
```

#### Các tùy chọn hữu ích khi thu thập rollout:
- **Chỉ lưu các tập thành công (`--save_successful_only`)**:
  Loại bỏ các tập gắp trượt hoặc làm đổ tháp, chỉ đóng gói và đẩy các episode hoàn thành nhiệm vụ lên Hub:
  ```bash
  python eval_act_sim.py \
    --policy vasco281204/act_stack_3_box_30x3030 \
    --num_episodes 20 \
    --save_dataset \
    --push_to_hub \
    --hub_repo_id anhquoc29/rollout_stact_3_box_sim \
    --save_successful_only
  ```

- **Tự động xóa dữ liệu cục bộ sau khi đẩy xong (`--cleanup_local_after_push`)**:
  Giải phóng 100% dung lượng ổ cứng trên máy tính sau khi dữ liệu đã được lưu an toàn trên Cloud Hugging Face:
  ```bash
  python eval_act_sim.py \
    --policy vasco281204/act_stack_3_box_30x3030 \
    --num_episodes 20 \
    --save_dataset \
    --push_to_hub \
    --cleanup_local_after_push
  ```

- **Chạy ngầm tốc độ cao (Headless) + Thu thập data + Đẩy lên Hub**:
  ```bash
  python eval_act_sim.py \
    --policy vasco281204/act_stack_3_box_30x3030 \
    --num_episodes 50 \
    --headless \
    --save_dataset \
    --push_to_hub \
    --hub_repo_id anhquoc29/rollout_stact_3_box_sim
  ```

- **Cơ chế ghi nối tiếp (Mặc định - Append Mode)**:
  Mỗi lần bạn chạy rollout, hệ thống **tự động kiểm tra dữ liệu đã có (cả trên Cloud Hugging Face lẫn cục bộ)** và **ghi thêm các tập mới vào phía sau, tuyệt đối không ghi đè**:
  *(Ví dụ: Dataset trên Cloud hiện có 6 tập, chạy thêm 10 tập sẽ tự động ghi từ tập 7 đến tập 16, tổng cộng sau khi đẩy lên Cloud sẽ là 16 tập).*

- **Chỉ ghi đè nếu bạn muốn xóa làm lại từ đầu (`--overwrite`)**:
  Chỉ khi bạn muốn xóa sạch các tập cũ trên dataset và tạo lại từ episode 1:
  ```bash
  python eval_act_sim.py \
    --policy vasco281204/act_stack_3_box_30x3030 \
    --num_episodes 10 \
    --save_dataset \
    --push_to_hub \
    --overwrite
  ```

---

### Bảng tổng hợp các tham số dòng lệnh của `eval_act_sim.py`

| Tham số | Giá trị mặc định | Giải thích |
| :--- | :--- | :--- |
| `--policy` | `.../pretrained_model` | Đường dẫn thư mục checkpoint cục bộ hoặc Repo ID trên Hugging Face (`vasco281204/act_stack_3_box_30x3030`). |
| `--checkpoint` | `None` (chọn cao nhất) | Chỉ định checkpoint cụ thể nếu repo có nhiều checkpoint (ví dụ: `100000`, `080000`, `060000`). |
| `--num_episodes` | `20` | Tổng số episode sẽ chạy đánh giá / rollout. |
| `--max_steps` | `1800` | Số bước tối đa của một episode (1800 bước $\approx$ 60 giây ở 30 FPS). |
| `--device` | `cuda` | Thiết bị chạy model: `cuda` (GPU) hoặc `cpu`. |
| `--headless` | `False` | Bật chế độ chạy ngầm không hiển thị cửa sổ OpenCV GUI. |
| `--save_video` | `False` | Bật cờ xuất video MP4 các episode đã chạy vào `video_dir`. |
| `--save_dataset` | `False` | Ghi lại toàn bộ dữ liệu quan sát và điều khiển thành chuẩn LeRobot Dataset v3.0. |
| `--push_to_hub` | `False` | **Tự động tải Dataset lên Hugging Face Hub** sau khi kết thúc. |
| `--hub_repo_id` | `anhquoc29/rollout_stact_3_box_sim` | Tên repository dataset trên Hugging Face Cloud. |
| `--save_successful_only`| `False` | Chỉ lưu các episode gắp và xếp tháp thành công. |
| `--cleanup_local_after_push` | `False` | Tự động xóa thư mục dataset cục bộ sau khi đẩy lên Cloud thành công để tiết kiệm ổ cứng. |
| `--overwrite` | `False` | **Mặc định tắt**. Chỉ bật khi bạn muốn xóa sạch dữ liệu cũ và ghi mới lại từ episode 1. Mặc định luôn là **GHI NỐI TIẾP (Append)**. |
| `--random_colors` | `False` | Tự động random màu sắc 3 khối hộp giữa các lần thử (Domain Randomization). |
| `--color_mode` | `full` | Chế độ đổi màu: `full` (màu ngẫu nhiên hoàn toàn), `jitter` (biến thiên sắc độ nhẹ quanh màu gốc), `shuffle` (hoán vị 3 màu). |

---

## 4. Cách Đọc Kết Quả Đánh Giá

Sau khi hoàn thành số episode đã chỉ định, Terminal sẽ hiển thị bảng kết quả:

```text
==================================================
          KET QUA EVALUATION TONG KET
==================================================
Tong so episode       : 20
So tap thanh cong     : 18 / 20
Ty le thanh cong      : 90.0%
So buoc trung binh    : 412.3 steps
==================================================
```

- **Tiêu chí tính thành công (`is_success`)**:
  1. Tháp 3 tầng xếp đúng thứ tự: Khối **Đen** ở đáy $\to$ Khối **Xanh lá** ở giữa $\to$ Khối **Xanh biển** ở đỉnh.
  2. Khối Đen nằm chuẩn trong vùng đích **C4** (sai số $< 45$ mm).
  3. Tháp đứng vững liên tục trong ít nhất 15 bước mô phỏng ($0.5$ giây).

---

## 5. Hướng Dẫn Mở Rộng: Triển Khai Trên Cánh Tay Thật (SO-101 Real Robot)

Nếu bạn muốn nạp model này chạy trên cánh tay robot thật SO-101:

1. **Chuẩn bị phần cứng**:
   - Cánh tay SO-101 Follower Arm kết nối qua mạch điều khiển Feetech/Waveshare Bus Servo Board.
   - 2 Camera USB góc rộng: Top Camera (nhìn vuông góc bàn $5 \times 7$) và Front Camera (nhìn chéo phía trước).
   - 3 khối hộp kích thước đúng $30 \times 30 \times 30$ mm (Đen, Xanh lá, Xanh biển) và bàn cờ chia ô $5 \times 7$ tương ứng.
2. **Cấp quyền cổng kết nối cánh tay thật**:
   ```bash
   sudo chmod 666 /dev/ttyACM* 2>/dev/null || sudo chmod 666 /dev/ttyUSB* 2>/dev/null
   ```
3. **Cơ chế hoạt động**:
   - Input khớp của model `vasco281204/act_stack_3_box_30x3030` nhận góc khớp theo **Độ (Degrees)** và xuất ra action cũng theo **Độ (Degrees)**, đồng bộ trực tiếp với hệ tọa độ calib của cánh tay SO-101.
