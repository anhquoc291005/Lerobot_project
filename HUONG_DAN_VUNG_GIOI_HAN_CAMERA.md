# Vùng thao tác ảo trên camera

Đa giác xanh trong `workspace_boundary.json` nằm phía trong các dụng cụ bao quanh
bàn trong ảnh ngày 26/10/2026. Đây là mốc trực quan để thu data, không tự giới hạn
chuyển động hoặc dừng robot. Đường biên được log thành đối tượng 2D riêng trong
Rerun; ảnh lưu vào dataset không chứa đường vẽ.

Camera hiện tại:

| Tên trong lệnh | Thiết bị | Góc nhìn |
| --- | --- | --- |
| `top` | `/dev/video4` | Nhìn từ trên xuống, có đường biên |
| `left` | `/dev/video2` | Nhìn ngang từ bên cạnh |

Tên camera đã thống nhất thành `top` và `left`. Nếu đổi tên hoặc đổi camera,
phải chỉnh key tương ứng trong JSON. Cần chỉnh lại đa giác khi di chuyển camera
hoặc thay đổi góc nhìn; tọa độ ảnh 2D không xác định giới hạn chiều cao của robot.

## Teleop

```bash
cd /home/anhquoc2910/Lerobot_project
bash teleop_2cam_workspace.sh
```

Script dùng leader `/dev/ttyACM1`, follower `/dev/ttyACM0`, calibration trong
`calibration/`, và Python từ môi trường `miniforge3/envs/lerobot`. Có thể chọn
Python khác bằng biến `LEROBOT_PYTHON`.

## Thu dataset

Thêm hai cờ này vào lệnh `python -m lerobot.scripts.lerobot_record` hiện có:

```bash
--display_data=true \
--display_workspace=/home/anhquoc2910/Lerobot_project/workspace_boundary.json
```

Chạy với source đã chỉnh trong dự án:

```bash
export PYTHONPATH=/home/anhquoc2910/Lerobot_project/lerobot/src${PYTHONPATH:+:$PYTHONPATH}
```

Đường biên hỗ trợ `--display_mode=rerun` (mặc định). Khởi động lại phiên đang mở
để nạp cấu hình mới.

## Xem và chỉnh đường biên mà không điều khiển robot

Dừng teleop/record trước để nhường camera, rồi chạy:

```bash
cd /home/anhquoc2910/Lerobot_project
conda activate lerobot
PYTHONPATH="$PWD/lerobot/src" python edit_workspace_boundary.py
```

- Kéo các đỉnh xanh để sửa đường biên.
- Nhấn `R` để vẽ lại, sau đó click các đỉnh theo thứ tự quanh vùng làm việc.
- Nhấn `U` để bỏ đỉnh cuối, `S` để lưu, `Q` hoặc `Esc` để thoát.
- Khởi động lại teleop/record sau khi lưu.

Điểm được lưu theo tỉ lệ `[x / (width - 1), y / (height - 1)]`, nên tự co giãn
khi đổi độ phân giải với cùng góc nhìn và khung hình.
