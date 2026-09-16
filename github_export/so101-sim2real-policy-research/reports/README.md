# Reports

Thư mục này chứa **dữ liệu** (CSV/MD/TEX/TXT) tổng hợp từ các script trong
`../src/sim_real/` và `../src/act/`. Không chứa hình ảnh kết quả (`.png`/`.svg`/`.pdf`)
— theo yêu cầu, repo này chỉ giữ số liệu, không giữ figure. Muốn tái tạo hình, chạy
lại script tương ứng (xem `docs/pipeline/end_to_end_pipeline.md`).

- `sim_real_gap/`: chỉ số & bảng cho phân tích sim-real gap (RMSE theo lag, action
  envelope theo quantile, tốc độ phản hồi, bảng LaTeX, verification stats) — từ
  `src/sim_real/paper_quality_sim_real_gap.py` và `compare_sim_real.py`.
- `hardware_command/`: `comprehensive_metrics.csv` — độ trễ lệnh, tracking error —
  từ `src/sim_real/hardware_command_eval.py`.
- `kinematics/`: `kinematic_metrics.csv` — phân phối vị trí/vận tốc/gia tốc/jerk —
  từ `src/sim_real/sim_real_kinematic_analysis.py`.
- `rollout_metrics/`: kết quả rollout (`lerobot-rollout --strategy.type=sentry`) —
  tracking theo lag, độ mượt, tương quan delta, báo cáo tổng hợp.

Không lưu dataset raw, checkpoint model, video, hoặc sample CSV lớn trong repo này
(xem `.gitignore`).
