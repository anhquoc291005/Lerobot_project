# Hướng Dẫn Chạy & Đánh Giá Mô Hình ACT Trong Môi Trường Mô Phỏng SO-101 (Stack 3 Blocks)

Tài liệu này hướng dẫn chi tiết quy trình chạy và đánh giá mô hình **ACT Policy** trong môi trường mô phỏng **MuJoCo SO-101** với nhiệm vụ gắp và xếp chồng 3 khối hộp ngẫu nhiên. Môi trường được thiết lập **đồng bộ 100% với điều kiện vật lý, góc đặt camera, quy tắc random ô lưới và đơn vị góc khi thu thập dataset**.

---

## 1. Thông Tin Cấu Hình Môi Trường Mô Phỏng Đã Thiết Lập

Môi trường đã được tích hợp và đăng ký chuẩn vào LeRobot với tên: **`so101_stack_3_blocks`** (Gym ID: `so101/SO101Stack3Blocks-v0`).

- **Dataset đối chiếu chuẩn**: [**anhquoc29/stack_3_box_30x3030**](https://huggingface.co/datasets/anhquoc29/stack_3_box_30x3030)
- **Tốc độ lấy mẫu & điều khiển**: **`30 FPS`** (`frame_skip=33`, physics timestep $0.001\text{s}$).
- **Vật thể thao tác**: 3 khối hộp kích thước **$30 \times 30 \times 30$ mm** (khối lượng 25g):
  - Khối **TRẮNG** (`box_white`): Đặt ở đáy tháp (trên ô đích `stack_target_zone`).
  - Khối **XANH LÁ** (`box_green`): Đặt ở giữa (chồng lên khối Trắng).
  - Khối **XANH BIỂN** (`box_blue`): Đặt ở đỉnh trên cùng (chồng lên khối Xanh lá).
- **Khu vực đích xếp chồng (`stack_target_zone`)**:
  - Tọa độ tâm: `[0.264, -0.0175, 0.001]` (nằm trọn trong ô **C4**, mép dưới sát vạch biên **D4**).
- **Cơ chế phân bố ngẫu nhiên (Grid-cell Randomization)**:
  - Tự động lấy **1 ô cánh trái** (`b2, b3, c2, c3, d3`), **1 ô cánh phải** (`b5, b6, c5, c6, d5`), và **1 ô khu vực giữa** (khoảng cách tối thiểu giữa các ô $\ge 12$ cm).
  - Độ rung ngẫu nhiên nhỏ ($\pm 8$ mm) và xoay góc Yaw tự do ($\pm 180^\circ$), đảm bảo 3 khối hộp luôn nằm gọn trong ô và không đè lên vạch biên.
- **Dữ liệu quan sát & điều khiển (Đồng bộ chuẩn)**:
  - `observation.state`: 6 góc trục khớp cánh tay robot (đơn vị: **Độ / Degrees**).
  - `action`: 6 lệnh góc tương ứng (đơn vị: **Độ / Degrees**).
  - `observation.images.top`: Camera trên trần nhìn xuống ($480 \times 640 \times 3$, góc nhìn rộng $55^\circ$).
  - `observation.images.front`: Camera góc nghiêng đặt tại ô **E9** ($480 \times 640 \times 3$, gain độ sáng $1.2$).
- **Tiêu chí tự động phát hiện thành công (`is_success`)**:
  - Tháp 3 tầng xếp đúng thứ tự: Trắng $\to$ Xanh lá $\to$ Xanh biển.
  - Vị trí khối Trắng nằm trong ô đích C4 ($< 45$ mm so với tâm đích).
  - Tháp đứng vững và ổn định liên tục ít nhất $15$ bước ($0.5$ giây).

---

## 2. Chuẩn Bị Trước Khi Chạy

Mở Terminal và kích hoạt môi trường Conda đã cài đặt sẵn thư viện:

```bash
# 1. Chuyển vào thư mục dự án
cd /home/anhquoc2910/Lerobot_project

# 2. Kích hoạt môi trường conda
conda activate lerobot
```

*(Lưu ý: Bạn không cần cắm cánh tay Leader thật khi chạy mô hình ACT, mô phỏng sẽ tự động đưa góc khớp từ model vào MuJoCo).*

---

## 3. Các Cách Chạy & Đánh Giá Mô Hình ACT

### Cách 1: Chạy trực quan tương tác thời gian thực (Khuyên dùng)

Sử dụng script [`eval_act_sim.py`](file:///home/anhquoc2910/Lerobot_project/eval_act_sim.py). Script này sẽ mở một cửa sổ OpenCV trực quan hiển thị đồng thời cả 2 camera (Top Cam kèm lưới tọa độ $5 \times 7$ và Front Cam):

```bash
# Chạy với checkpoint vừa train xong
python eval_act_sim.py \
  --policy outputs/train/act_stack_3_blocks/checkpoints/last/pretrained_model
```

*(Nếu bạn lưu checkpoint ở vị trí khác hoặc tải từ Hugging Face Hub, chỉ cần truyền đường dẫn hoặc repo ID vào tham số `--policy`).*

#### Các tham số mở rộng hữu ích:
- `--save_video`: Tự động lưu video MP4 của từng episode đã rollout vào thư mục `outputs/eval_videos`.
- `--num_episodes 50`: Chạy 50 episode liên tục để thống kê tỷ lệ thành công.
- `--max_steps 1800`: Thời gian tối đa mỗi episode (1800 bước = 60 giây ở 30 FPS).
- `--device cuda`: Chạy model trên GPU (hoặc `--device cpu`).
- `--headless`: Tắt hiển thị cửa sổ OpenCV (phù hợp khi chạy trên máy chủ/server không có màn hình).

**Ví dụ chạy 20 episodes đánh giá tự động và lưu video:**
```bash
python eval_act_sim.py \
  --policy outputs/train/act_stack_3_blocks/checkpoints/last/pretrained_model \
  --num_episodes 20 \
  --save_video \
  --video_dir outputs/eval_videos
```

---

### Cách 2: Chạy đánh giá tự động bằng lệnh chuẩn `lerobot-eval`

Môi trường đã được đăng ký chính thức vào hệ thống LeRobot, bạn có thể gọi trực tiếp lệnh CLI chuẩn hóa:

```bash
python -m lerobot.scripts.lerobot_eval \
  --policy.path=outputs/train/act_stack_3_blocks/checkpoints/last/pretrained_model \
  --env.type=so101_stack_3_blocks \
  --eval.n_episodes=20 \
  --eval.batch_size=1
```

Lệnh trên sẽ tự động:
1. Khởi tạo môi trường mô phỏng `so101_stack_3_blocks`.
2. Tải normalization stats từ dataset `anhquoc29/stack_3_box_30x3030`.
3. Rollout 20 episodes liên tiếp và tự động ghi video MP4 vào thư mục `outputs/eval/`.
4. Tính toán và in bảng báo cáo tổng kết chi tiết: tỷ lệ thành công (`pc_success`), phần thưởng trung bình (`avg_sum_reward`), thời gian đánh giá.

---

### Cách 3: Nhúng trực tiếp vào Python Code (Gymnasium & LeRobot API)

Nếu bạn muốn viết script tùy chỉnh riêng hoặc tích hợp vào pipeline kiểm thử tự động:

```python
import numpy as np
import torch
from lerobot.envs import make_env, make_env_config, preprocess_observation
from lerobot.policies.act import ACTPolicy

# 1. Khởi tạo môi trường
env_cfg = make_env_config("so101_stack_3_blocks")
envs = make_env(env_cfg, n_envs=1)
vec_env = envs[env_cfg.type][0]

# 2. Reset episode (tự động random vị trí 3 khối hộp)
obs, info = vec_env.reset(seed=42)
print("Vị trí ngẫu nhiên 3 hộp:", info["stack_random_placements"])

# 3. Tiền xử lý observation cho ACT Policy
policy_obs = preprocess_observation(obs)
# policy_obs['observation.state']: tensor góc 6 khớp (Độ/Degrees)
# policy_obs['observation.images.top']: tensor camera trần (1, 3, 480, 640)
# policy_obs['observation.images.front']: tensor camera trước (1, 3, 480, 640)

# 4. Step môi trường với action (Độ/Degrees)
# action shape: (1, 6)
dummy_action = np.zeros((1, 6), dtype=np.float32)
next_obs, reward, terminated, truncated, step_info = vec_env.step(dummy_action)

if step_info.get("is_success", [False])[0]:
    print("Xếp chồng 3 khối hộp thành công!")

vec_env.close()
```

---

## 4. Phím Tắt Điều Khiển Trong Cửa Sổ Preview (GUI Mode)

Khi chạy qua `eval_act_sim.py`, bạn có thể theo dõi trực tiếp camera kép và tương tác bàn phím:

| Phím bấm | Chức năng |
| :--- | :--- |
| **`SPACE`** / **`ENTER`** / **`r`** | **Reset sớm sang episode mới**: Môi trường sẽ reset ngay lập tức và random 3 khối hộp vào 3 ô vị trí mới. |
| **`p`** | **Tạm dừng (Pause) / Tiếp tục**: Đóng băng mô phỏng để quan sát chi tiết tư thế gắp của robot. |
| **`q`** / **`ESC`** | **Thoát chương trình**: Đóng cửa sổ và in bảng kết quả thống kê các episode đã thực hiện. |

---

## 5. Báo Cáo Đánh Giá & Đo Lường Kết Quả

Sau khi kết thúc phiên đánh giá, terminal sẽ tự động hiển thị bảng tổng kết:

```text
==================================================
          KET QUA EVALUATION TONG KET
==================================================
Tong so episode       : 20
So tap thanh cong     : 17 / 20
Ty le thanh cong      : 85.0%
So buoc trung binh    : 428.5 steps
==================================================
```

- **Số tập thành công**: Đếm số lần robot hoàn thành việc gắp và đặt vững chắc cả 3 khối hộp lên vị trí đích C4.
- **Tỷ lệ thành công**: Phần trăm hoàn thành nhiệm vụ của mô hình ACT.
- **Số bước trung bình**: Thời gian trung bình để robot hoàn thành thao tác (bước càng ít chứng tỏ robot gắp càng dứt khoát và tối ưu quỹ đạo).

---

## 6. Xử Lý Các Vấn Đề Thường Gặp (Troubleshooting)

1. **Lỗi `Cannot connect to X server` khi chạy qua SSH:**
   - Nếu bạn đang kết nối từ xa không có giao diện đồ họa X11, hãy thêm cờ `--headless` khi chạy:
     ```bash
     python eval_act_sim.py --policy outputs/train/act_stack_3_blocks/checkpoints/last/pretrained_model --headless --save_video
     ```
   - Video quay lại toàn bộ quá trình vẫn sẽ được ghi và lưu vào thư mục `outputs/eval_videos/`.

2. **Lỗi `Model weights not found`:**
   - Đảm bảo đường dẫn `--policy` trỏ đến thư mục chứa đầy đủ hai file: `config.json` và `model.safetensors` (thường là thư mục con `pretrained_model` trong checkpoint).

3. **Tải normalization stats chậm khi không có internet:**
   - Mặc định script sẽ kiểm tra stats từ dataset [anhquoc29/stack_3_box_30x3030](https://huggingface.co/datasets/anhquoc29/stack_3_box_30x3030).
   - Dữ liệu metadata này đã được tải và cache tại `~/.cache/huggingface/`, do đó các lần chạy sau sẽ load ngay lập tức mà không cần tải lại.
