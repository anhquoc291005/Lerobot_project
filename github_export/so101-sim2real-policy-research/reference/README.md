# reference/

Code trong thư mục này **không phải code tự viết** — đây là bản sao gần như
nguyên vẹn của core code từ các repo upstream, được đưa vào để đọc/tra cứu cùng
với các script phân tích ở `src/` mà không cần checkout riêng `lerobot`. Xem
[`NOTICE.md`](../NOTICE.md) ở gốc repo để biết nguồn gốc và giấy phép chi tiết.

Code ở đây **không tự chạy độc lập** trong thư mục này — nó import từ package
`lerobot` (ví dụ `lerobot.utils.constants`, `lerobot.processor`, ...), nên vẫn cần
`pip install lerobot` / checkout `lerobot` đầy đủ để chạy training/inference thật.
Mục đích của `reference/` là để review code (kiến trúc model, cách xử lý hardware)
mà không phải đào bới trong một checkout `lerobot` đầy đủ (hàng nghìn file).

## Nội dung

- `policies/act/` — implementation đầy đủ của ACT Policy (`configuration_act.py`,
  `modeling_act.py`, `processor_act.py`). So sánh với các script phân tích ở
  `src/act/` (SHAP, latent-space) dùng chính các class này.
- `policies/diffusion/` — implementation đầy đủ của Diffusion Policy.
- `robots_so101/follower/` — class điều khiển SO-101 follower thật
  (`so_follower.py`), config (`config_so_follower.py`), và xử lý động học
  (`robot_kinematic_processor.py`, forward/inverse kinematics dùng khi điều khiển
  bằng end-effector thay vì góc khớp).
- `robots_so101/leader/` — class đọc leader arm thật (`so_leader.py`), dùng bởi cả
  `lerobot-teleoperate`/`lerobot-record` và `src/sim_mujoco/teleop_leader_to_mujoco.py`.
- `sim_assets/so_arm100_simulation/` — model MuJoCo (MJCF) + URDF của SO-100/SO-101,
  dùng bởi `src/sim_mujoco/`. Xem `docs/so101/so101_mujoco_sim.md`.
