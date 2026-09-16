# Thu thập dữ liệu mô phỏng MuJoCo: Xếp chồng 3 khối hộp (Stack 3 Blocks)

Tài liệu hướng dẫn quy trình điều khiển **SO-101 Leader thật (teleoperation)** để gắp và xếp chồng 3 khối hộp trong môi trường mô phỏng **MuJoCo**, tự động **random vị trí 3 hộp sau mỗi episode**, và **ghi lưu dataset theo chuẩn LeRobotDataset**.

---

## 1. Quy cách nhiệm vụ (Task Specification)

- **Vật thể thao tác**: 3 khối hộp kích thước **30 x 30 x 30 mm** (khối lượng 25g):
  1. `box_white` (Trắng): Đặt ở **dưới cùng** (đáy tháp hoặc trên ô `stack_target_zone`).
  2. `box_green` (Xanh lá): Đặt ở **giữa** (trên đỉnh khối trắng).
  3. `box_blue` (Xanh nước biển): Đặt ở **trên cùng** (trên đỉnh khối xanh lá).
- **Khu vực đích xếp chồng (`stack_target_zone`)**:
  - Nằm gọn trong ô **C4**, mép dưới nằm sát với đường phân chia giữa ô **C4** và **D4**.
  - Kích thước thu nhỏ còn **2/3** ($53.33 \times 53.33$ mm, half_size = $0.0267$ m).
- **Quy tắc random vị trí (Grid-cell based)**:
  - **Mỗi khối hộp nằm trọn trong 1 ô riêng biệt**: Tọa độ tâm ô được căn chỉnh chính xác trên camera top (5x7) kết hợp độ rung nhẹ $\pm 8$ mm, đảm bảo hộp không bao giờ đè lên hoặc vượt qua vạch biên của ô.
  - **Phân phối an toàn trong tầm với (Spread Out & Reachable)**: Tự động lấy 1 ô bên cánh trái (`b2, b3, c2, c3, d3`), 1 ô bên cánh phải (`b5, b6, c5, c6, d5`), và 1 ô còn lại (kèm ô trung tâm `d4`) với khoảng cách tối thiểu $\ge 12$ cm, giúp 3 hộp phân bố đều không gian làm việc. Toàn bộ hàng E, các ô biên ngoài `c1, c7, d1, d7` và các góc khó với `d2, d6` đều được loại bỏ triệt để.
  - Xoay ngẫu nhiên góc Yaw ($\pm 180^\circ$).
- **Độ phân giải & Chất lượng Camera**:
  - Độ phân giải: **$640 \times 480$** (chuẩn huấn luyện policy LeRobot: nhẹ, tốc độ cao, tiết kiệm dung lượng).
  - Hiển thị Preview: Tỷ lệ 1:1 (`preview_scale = 1.0`, kích thước cửa sổ $1280 \times 480$), hiển thị đúng điểm ảnh gốc không bị co bóp mờ nhòe.

---

## 2. File cấu hình & Script chính

- **File cấu hình**: [`src/sim_mujoco/config_sim_collect_stack_3_blocks.json`](file:///home/anhquoc2910/Lerobot_project/github_export/so101-sim2real-policy-research/src/sim_mujoco/config_sim_collect_stack_3_blocks.json)
- **Script thu thập**: [`src/sim_mujoco/collect_sim_dataset.py`](file:///home/anhquoc2910/Lerobot_project/github_export/so101-sim2real-policy-research/src/sim_mujoco/collect_sim_dataset.py)
- **Môi trường MuJoCo**: [`src/sim_mujoco/mujoco_xml_env.py`](file:///home/anhquoc2910/Lerobot_project/github_export/so101-sim2real-policy-research/src/sim_mujoco/mujoco_xml_env.py)
- **Scene XML**: [`reference/sim_assets/so_arm100_simulation/SO101/scene.xml`](file:///home/anhquoc2910/Lerobot_project/github_export/so101-sim2real-policy-research/reference/sim_assets/so_arm100_simulation/SO101/scene.xml)

---

## 3. Kết nối tay Leader thật

1. **Xác định cổng Serial của tay Leader**:
   ```bash
   ls /dev/ttyACM* /dev/ttyUSB*
   # hoặc dùng tool LeRobot:
   lerobot-find-port
   ```

2. **Cấp quyền truy cập cổng USB (nếu cần)**:
   ```bash
   sudo chmod 666 /dev/ttyACM0
   ```

3. **Thiết lập biến môi trường Leader**:
   ```bash
   export REAL_TELEOP_PORT=/dev/ttyACM0
   export REAL_TELEOP_ID=my_so101_leader
   export REAL_TELEOP_USE_DEGREES=true
   export REAL_TELEOP_CALIBRATION_DIR=/home/anhquoc2910/Lerobot_project/github_export/so101-sim2real-policy-research/calibration
   ```

---

## 3.5. Chạy Demo thử nghiệm (Không cần cắm tay Leader)

Nếu bạn chưa cắm tay Leader hoặc muốn kiểm tra trước trực quan mô hình, góc nhìn 2 camera và cơ chế random vị trí 3 khối hộp:

### Cách 1: Xem trực quan 2 camera (Top & Front) và cơ chế Random
```bash
python src/sim_mujoco/demo_stack_3_blocks.py
```
- Màn hình hiển thị 2 camera song song cùng lưới tọa độ.
- Nhấn **`Space`** hoặc **`Enter`**: Môi trường tự động random lại vị trí 3 khối hộp trong workspace và in tọa độ lên terminal.
- Nhấn **`q`** hoặc **`Esc`**: Thoát demo.

### Cách 2: Mở 3D Interactive Viewer (MuJoCo Native)
```bash
python src/sim_mujoco/demo_stack_3_blocks.py --viewer
```
- Dùng chuột xoay, zoom, pan quanh cánh tay robot SO-101.
- Dùng **`Ctrl + Chuột phải kéo`** để tương tác vật lý trực tiếp với các khối hộp và tay robot.
- Tự động đổi vị trí random sau mỗi 8 giây hoặc khi reset.

### Cách 3: Chạy thử quy trình thu thập Dataset (Dry-run Demo)
```bash
python src/sim_mujoco/collect_sim_dataset.py \
  --config src/sim_mujoco/config_sim_collect_stack_3_blocks_demo.json
```
- Thử nghiệm toàn bộ chu trình thu thập: Preparation mode $\to$ Nhấn `Enter` để ghi frame $\to$ Nhấn `Enter` để lưu episode $\to$ Tự động reset và random vị trí cho episode tiếp theo $\to$ Nhấn `q` để đóng gói dataset.

---

## 4. Chạy thu thập dữ liệu (Data Collection)

Chạy lệnh sau từ thư mục gốc của repository:

```bash
python src/sim_mujoco/collect_sim_dataset.py \
  --config src/sim_mujoco/config_sim_collect_stack_3_blocks.json
```

Hoặc truyền trực tiếp cổng leader qua CLI:

```bash
python src/sim_mujoco/collect_sim_dataset.py \
  --config src/sim_mujoco/config_sim_collect_stack_3_blocks.json \
  --teleop.port /dev/ttyACM0 \
  --dataset.num_episodes 30
```

---

## 5. Quy trình điều khiển trong mỗi Episode

1. **Preparation Mode (Chế độ chuẩn bị)**:
   - Cửa sổ OpenCV preview mở ra hiển thị 2 camera: `pixels.top` (camera trên nhìn xuống) và `pixels.gripper`/`front` (camera góc nhìn nghiêng phía trước).
   - Tay leader đã có thể điều khiển cánh tay trong MuJoCo, nhưng **chưa ghi dữ liệu vào dataset**.
   - Người điều khiển di chuyển tay leader về tư thế xuất phát thoải mái.
   - Nhấn **`Enter`**, **`Space`** hoặc phím **`Mũi tên phải (->)`** trên cửa sổ preview để **BẮT ĐẦU GHI**.

2. **Recording Mode (Chế độ ghi dữ liệu)**:
   - Điều khiển tay robot gắp khối **ĐEN** đặt vào vị trí đích (hoặc giữ nguyên làm đế trên ô `stack_target_zone`).
   - Điều khiển gắp khối **XANH LÁ** đặt chồng lên khối ĐEN.
   - Điều khiển gắp khối **XANH BIỂN** đặt chồng lên đỉnh khối XANH LÁ.

3. **Hoàn thành & Lưu Episode**:
   - Khi tháp 3 khối đã ổn định, nhấn **`Enter`**, **`Space`** hoặc **`Mũi tên phải (->)`** để kết thúc sớm và **LƯU EPISODE**.
   - Nếu xảy ra lỗi (ví dụ làm đổ tháp), nhấn **`r`**, **`Backspace`** hoặc **`Mũi tên trái (<-)`** để **HỦY VÀ GHI LẠI** episode đó.

4. **Tự động chuyển sang Episode tiếp theo**:
   - Môi trường MuJoCo tự động reset về vị trí ban đầu.
   - Cả 3 khối hộp tự động được **xếp vào vị trí random mới** (chia theo ô lưới riêng biệt, phân bố rộng 2 bên và phía trước).
   - Toàn bộ tọa độ khởi đầu của 3 khối được lưu tự động vào `datasets/stack_3_box_30x3030/meta/stack_random_placements.jsonl`.
   - Lặp lại bước 1 cho đến khi đủ số episode mục tiêu.
   - Nhấn **`q`** hoặc **`Esc`** bất kỳ lúc nào để dừng phiên thu thập, đóng gói dataset và tự động đẩy lên Hugging Face.

---

## 6. Kiểm tra dữ liệu thu được

Sau khi hoàn thành phiên thu thập, dataset được lưu tại `datasets/stack_3_box_30x3030` và trên Hugging Face tại [anhquoc29/stack_3_box_30x3030](https://huggingface.co/datasets/anhquoc29/stack_3_box_30x3030).

- **Xem nhật ký tọa độ random của từng episode**:
  ```bash
  cat datasets/stack_3_box_30x3030/meta/stack_random_placements.jsonl
  ```

- **Phát lại và kiểm tra video/quỹ đạo đã thu**:
  ```bash
  lerobot-visualize-dataset --repo-id datasets/stack_3_box_30x3030 --episode-index 0
  ```