# Ghi Chú Thiết Lập Góc Camera & Quản Lý Branch GitHub Dự Án LeRobot SO-101

Tài liệu này tổng hợp toàn bộ thông số kỹ thuật, góc đặt camera khớp với dataset **`anhquoc29/rollout_stact_3_box_sim`**, các lệnh chạy demo/đánh giá policy, và quy trình quản lý các thiết lập thử nghiệm trên GitHub.

---

## 1. Thông Số Góc Đặt Camera Chuẩn (Dataset `anhquoc29/rollout_stact_3_box_sim`)

Toàn bộ thông số 3D của 2 camera đã được đo đạc, đối chiếu ảnh trích xuất từ video rollout và đồng bộ vào file mô phỏng MuJoCo:

### 1.1. Top Camera (`top_cam`)
* **Vị trí (`pos`)**: `0.22 0.0 0.54`
  * $X = 0.22\text{ m}$ ($22\text{ cm}$ trước đế robot, đặt ngay tâm bàn thao tác).
  * $Y = 0.0\text{ m}$ (ngay chính giữa tim robot).
  * $Z = 0.54\text{ m}$ (cao $54\text{ cm}$ trên nóc trần).
* **Định hướng (`xyaxes`)**: `"0 0.999391 -0.0349 -1 0 0"`
  * Camera nhìn từ trên xuống vuông góc với mặt bàn thao tác.
* **Góc mở (`fovy`)**: `55°`
* **Khai báo XML**:
  ```xml
  <camera name="top_cam" pos="0.22 0 0.54" xyaxes="0 0.999391 -0.0349 -1 0 0" fovy="55" />
  ```

---

### 1.2. Front Camera (`front_cam`)
* **Vị trí (`pos`)**: `0.65 0.0 0.14`
  * $X = 0.65\text{ m}$ ($65\text{ cm}$ trước đế robot, gắn giữa thanh nhôm ngang trước).
  * $Y = 0.0\text{ m}$ (thẳng hàng tim giữa robot).
  * $Z = 0.14\text{ m}$ (cao $14\text{ cm}$ so với mặt sàn).
* **Định hướng (`xyaxes`)**: `"0 1 0 -0.233445 0 0.972370"`
  * Trục quang học hướng chéo chúi xuống mặt bàn với góc nghiêng $\approx 13.5^\circ$, hiển thị rõ 3 khối hộp, target zone và thân robot.
* **Góc mở (`fovy`)**: `52°`
* **Khai báo XML**:
  ```xml
  <camera name="front_cam" pos="0.65 0 0.14" xyaxes="0 1 0 -0.233445 0 0.972370" fovy="52" />
  ```

---

## 2. Các Lệnh Chạy Demo & Đánh Giá Mô Hình

### Cách 1: Chạy Policy ACT tự động gắp xếp 3 khối hộp
```bash
cd /home/anhquoc2910/Lerobot_project
conda activate lerobot

python eval_act_sim.py \
  --policy vasco281204/act_stack_3_box_30x3030 \
  --num_episodes 10
```
* Cửa sổ trực quan kép sẽ hiển thị: **Top Cam (kèm lưới $5 \times 7$)** và **Front Cam**.
* Bấm **`q`** hoặc **`ESC`** để thoát sớm.
* Thêm cờ `--save_video` nếu muốn tự động lưu video MP4 các lượt gắp vào thư mục `outputs/eval_videos`.

---

### Cách 2: Xem trực quan 2 Camera & Test Random vị trí (Không cần tay Leader)
```bash
cd /home/anhquoc2910/Lerobot_project/github_export/so101-sim2real-policy-research
conda activate lerobot

python src/sim_mujoco/demo_stack_3_blocks.py
```
* **`SPACE`** / **`ENTER`** / **`r`**: Random lại vị trí 3 khối hộp trong không gian làm việc.
* **`q`** / **`ESC`**: Thoát demo.

---

### Cách 3: Mở 3D Interactive Viewer (MuJoCo Native)
```bash
cd /home/anhquoc2910/Lerobot_project/github_export/so101-sim2real-policy-research
conda activate lerobot

python src/sim_mujoco/demo_stack_3_blocks.py --viewer
```
* **Chuột trái**: Xoay 3D.
* **Chuột phải**: Zoom in/out.
* **Chuột giữa**: Di chuyển khung hình (Pan).
* **`Ctrl` + Chuột phải kéo**: Tác dụng lực vật lý trực tiếp kéo/thả khối hộp hoặc tay robot.

---

## 3. Lỗi Đã Được Khắc Phục

### Lỗi `UnboundLocalError: start_episodes_count`
* **Hiện tượng**: Khi chạy `eval_act_sim.py` ở chế độ thông thường (không truyền cờ `--save_dataset`), biến `start_episodes_count` không được định nghĩa, gây crash khi bắt đầu episode 1.
* **Đã xử lý**: Khởi tạo mặc định `start_episodes_count = 0` tại dòng 264 của file [`eval_act_sim.py`](eval_act_sim.py). Hiện script chạy ổn định cả 2 chế độ: chỉ đánh giá trực quan hoặc lưu tập rollout.

---

## 4. Quản Lý Repository Trên GitHub

* **Địa chỉ Repository**: [**https://github.com/anhquoc291005/Lerobot_project.git**](https://github.com/anhquoc291005/Lerobot_project)
* **Giao thức kết nối**: SSH (`git@github.com:anhquoc291005/Lerobot_project.git`).

### Danh mục các thư mục và file cấu hình đã được đẩy lên:
1. **Calibration (`calibration/`)**:
   - `my_awesome_leader_arm.json`
   - `my_awesome_follower_arm.json`
   - `my_so101_leader.json`
2. **Cấu hình mô phỏng & thu thập (`github_export/.../src/sim_mujoco/`)**:
   - `config_sim_collect_stack_3_blocks.json`
   - `config_sim_collect_stack_3_blocks_demo.json`
   - `config_sim_collect_real_like.json`
   - `config_sim_collect_mujoco_xml_local_test.json`
   - `config_sim_collect_mujoco_xml_matrix_hf_test.json`
3. **Môi trường 3D XML (`github_export/.../reference/sim_assets/.../SO101/`)**:
   - `scene.xml` (Góc camera chuẩn của dataset)
   - `test_scene.xml`, `so101_new_calib.xml`, `so101_old_calib.xml`
   - Toàn bộ mesh 3D `.stl`, `.part` trong thư mục `assets/`
4. **Môi trường Gym LeRobot (`lerobot_custom_envs/`)**:
   - `so101.py`, `configs.py`, `__init__.py`
5. **Scripts thực thi**:
   - `eval_act_sim.py` (Script chạy ACT policy tại thư mục gốc)
   - Toàn bộ source code trong `github_export/.../src/`
6. **Tài liệu hướng dẫn**:
   - `TOA_DO_CAMERA_VA_VUNG_MUC_TIEU.md`
   - `HUONG_DAN_CHAY_ACT_STACK_3_BOX.md`
   - `HUONG_DAN_CHAY_MODEL_VASCO281204.md`
   - `HUONG_DAN_SO101_MUJOCO_TELEOP.md`
   - `HUONG_DAN_THU_DATA_STACK_3_BOX_CLOUD.md`
   - `v1_schematic.pdf`

---

## 5. Quy Trình Tạo Nhánh (Branching Workflow) Cho Các Thiết Lập Mới

Mỗi lần bạn muốn thay đổi hoặc thử nghiệm cấu hình khác (góc camera mới, vị trí ô lưới khác, màu sắc hộp khác):

### Bước 1: Tạo nhánh mới
```bash
cd /home/anhquoc2910/Lerobot_project
git checkout -b <ten-nhanh-moi>
```
*Ví dụ*: `git checkout -b setup-cam-e9-box-white`

### Bước 2: Chỉnh sửa cấu hình
Sửa file `scene.xml` hoặc file cấu hình `json` theo mong muốn của bạn.

### Bước 3: Lưu lại và đẩy lên GitHub
```bash
git add .
git commit -m "feat: mo ta thiet lap moi cua ban"
git push -u origin <ten-nhanh-moi>
```

### Bước 4: Quay lại cấu hình chuẩn hiện tại
Bất cứ lúc nào muốn khôi phục lại góc camera và thiết lập của `rollout_stact_3_box_sim`, bạn chỉ cần gõ:
```bash
git checkout setup-rollout-stact-3-box-sim
```
Mọi file cấu hình, XML và camera sẽ tự động trở về đúng trạng thái ban đầu mà không bị mất dữ liệu.
