# Notice — nguồn gốc code trong repo này

Repo này là **nghiên cứu cá nhân** (đánh giá sim-real gap trên SO-101, dùng ACT
Policy và Diffusion Policy trong LeRobot), nhưng có vay mượn/vendor một phần code
từ các dự án mã nguồn mở khác để phần phân tích tự chứa và dễ đọc hơn. Toàn bộ
đều là Apache-2.0, tương thích với `LICENSE` ở gốc repo (copy nguyên văn từ
`huggingface/lerobot`, đã bao gồm các thông báo bên thứ ba liên quan — ví dụ
Diffusion Policy/DETR).

## Code vay mượn nguyên vẹn (`reference/`)

- `reference/policies/act/`, `reference/policies/diffusion/` — copy không sửa từ
  [`huggingface/lerobot`](https://github.com/huggingface/lerobot) (Apache-2.0).
- `reference/robots_so101/` — copy không sửa từ `huggingface/lerobot`
  (`lerobot.robots.so_follower`, `lerobot.teleoperators.so_leader`), Apache-2.0.
- `reference/sim_assets/so_arm100_simulation/` — từ
  [`TheRobotStudio/SO-ARM100`](https://github.com/TheRobotStudio/SO-ARM100)
  (Apache-2.0), lấy qua fork `thuongtran21112004-star/realtosim`. Chỉ phần
  `Simulation/` (model MuJoCo + URDF SO-100/SO-101) được copy — không copy phần
  CAD/3D-print (`STL/`, `STEP/`, `Mini/`, `Optional/`, `media/`) vì không cần cho
  chạy policy/mô phỏng.

## Code chỉnh sửa/thích nghi

- `src/sim_mujoco/` — từ `thuongtran21112004-star/realtosim` (fork của
  `huggingface/lerobot`, commit "Add real-to-sim teleoperation integration"),
  copy gần như nguyên vẹn, chỉ kiểm tra lại không còn đường dẫn/thông tin cá nhân
  hardcode.
- `src/act/act_training_example.py`, `src/act/act_using_example.py`,
  `src/diffusion/diffusion_training_example.py`,
  `src/diffusion/diffusion_using_example.py` — từ tutorial examples của
  `huggingface/lerobot` (Apache-2.0).
- `src/act/analyze.py` — dựa trên script gốc của tác giả repo này, đã viết lại
  phần CLI (argparse) để bỏ đường dẫn cá nhân hardcode.

## Code gốc

Mọi thứ khác — `src/act/shap_analysis.py`, `src/act/latent_space_analysis.py`,
`src/sim_real/*`, `src/dataset_tools/*`, `src/utils/*`, toàn bộ `docs/`, `reports/`
— là nghiên cứu/kết quả gốc của tác giả repo này.

## Không đưa vào repo

`outputs/` (checkpoint model), `datasets/` (raw dataset), thư mục `shap/`
(third-party library — cài qua `pip install shap`, không vendor), phần CAD của
SO-ARM100 nêu trên, và mọi hình ảnh kết quả (`.png`/`.svg`/`.pdf`) — theo yêu cầu
của tác giả khi tổng hợp repo này.
