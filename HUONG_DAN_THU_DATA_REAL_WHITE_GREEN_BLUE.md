# Thu data real: xếp 3 khối trắng, xanh lá, xanh dương

Dataset đích: https://huggingface.co/datasets/anhquoc29/stack_3_box_white_green_blue_simulation

Giữ nguyên repo theo yêu cầu, dù tên có chữ `simulation`. Các episode thu bằng
script này là dữ liệu robot thật. Khi kiểm tra ngày 27/10/2026, repo có 0 episode,
chỉ có metadata khởi tạo. Phiên thu thành công đầu tiên sẽ thiết lập schema camera
`top` và `left`.

Task từ dưới lên: **trắng → xanh lá → xanh dương**. Đưa khối trắng đến vị trí đích,
đặt khối xanh lá lên trắng, rồi đặt khối xanh dương lên xanh lá.

## Chạy thu data

Dừng teleop hoặc công cụ chỉnh đường biên trước khi chạy để nhường robot/camera.

```bash
cd /home/anhquoc2910/Lerobot_project
bash record_real_stack_3_box.sh --episodes 10
```

Mỗi lần chạy thu tối đa **10 episode mới**, mỗi episode tối đa **60 giây**.
Chương trình chờ **Space** trước episode đầu tiên và trước từng episode tiếp theo;
bạn có thể đặt lại các khối trong lúc chờ. Muốn thu từng episode một:

```bash
bash record_real_stack_3_box.sh --episodes 1
```

Muốn đổi thời gian:

```bash
bash record_real_stack_3_box.sh --episodes 10 --episode-seconds 90
```

Script dùng Python trong `~/miniforge3/envs/lerobot`, không cần activate conda.

Theo lựa chọn hiện tại, dữ liệu được lưu tạm trên ổ đĩa trong thư mục `datasets/`.
Kết thúc phiên thu, script upload các episode đã lưu lên Hub, xác minh đầy đủ rồi
tự xóa bản dataset local. Nếu upload hoặc xác minh lỗi, dữ liệu local được giữ lại
để thử lại. Dùng lệnh không có `--hub-direct` để chạy chế độ này.

| Thiết bị | Cổng / tên dữ liệu |
| --- | --- |
| Leader | USB serial `5A7C122417`, hiện là `/dev/ttyACM1` |
| Follower | USB serial `5AB0158112`, hiện là `/dev/ttyACM0` |
| Camera trên | `/dev/video4`, `observation.images.top` |
| Camera bên cạnh | `/dev/video2`, `observation.images.left` |
| Calibration | `calibration/` trong dự án |

Robot được chọn theo mã USB cố định để tránh nhầm tay nếu số ACM đổi sau khi cắm
lại. Hai camera thu RGB 640×480 ở 30 fps, video H.264, CRF 18. Năm khớp quay dùng
đơn vị độ; độ mở gripper được chuẩn hóa trong khoảng 0–100.

## Điều khiển từng episode

- **`Space`**: khi đang chờ, bắt đầu thu một episode; khi đang thu, kết thúc và
  lưu episode ngay, không cần chờ đủ 60 giây, rồi quay lại chờ Space cho episode
  tiếp theo. Camera, viền xanh và teleop vẫn hoạt động trong lúc chờ, nhưng chưa
  ghi frame vào dataset.
- **`Q`**: hủy episode đang thu, bỏ các frame của episode đó rồi trở về trạng thái
  chờ Space. Episode đã lưu trước đó được giữ nguyên; Q không thoát chương trình.
- **`Esc`**: lưu episode đang thu nếu đã có frame, dừng phiên và upload các episode
  đã lưu. Nếu đang chờ, thoát và upload dữ liệu đã lưu, không tạo episode rỗng.
- **Mũi tên phải `→`**: kết thúc/lưu episode sớm khi đã xếp xong, rồi chờ Space.

Nếu không nhấn phím kết thúc, episode tự lưu khi đạt giới hạn 60 giây (hoặc giá
trị `--episode-seconds`), rồi chờ Space. Thời gian đặt lại khối không giới hạn;
`--reset-seconds` không được dùng trong chế độ điều khiển thủ công này.

Trên Wayland, cần giữ focus ở **terminal chạy lệnh** để các phím được nhận.
Không đóng terminal khi đang encode hoặc upload. Upload diễn ra khi kết thúc
phiên thu, sau khi các file video/parquet đã được đóng.

## Thông báo trạng thái đang thu / chưa thu

Phía trên các camera trong Rerun có bảng **Trạng thái thu data**:

- **🔴 ĐANG THU DATA**: đã bấm Space và đang ghi frame của episode.
- **🟢 CHƯA THU DATA**: đang chờ, đang lưu, hoặc đã hủy episode.

Bấm **Q** sẽ chuyển bảng về **CHƯA THU DATA**, kèm thông báo đã hủy episode,
rồi tiếp tục chờ Space. Bảng cũng hiển thị số episode và gợi ý phím điều khiển.
Terminal in cùng trạng thái để dễ theo dõi. Bảng thông báo là đối tượng hiển thị
riêng, không xuất hiện trong ảnh/video dataset.

## Viền xanh và ảnh dataset

Camera `top` hiển thị một đường viền xanh khép kín từ `workspace_boundary.json`,
không chữ, không tô nền. Đường viền là một đối tượng 2D riêng trong Rerun,
không vẽ lên ảnh gốc. Video `top`/`left` trong dataset không chứa đường viền.
Đường này là mốc nhìn để thao tác, không tự chặn chuyển động robot.

Để chỉnh lại đường biên sau khi đổi góc camera, dừng phiên thu rồi chạy:

```bash
conda activate lerobot
PYTHONPATH="$PWD/lerobot/src" python edit_workspace_boundary.py
```

Kéo các đỉnh, nhấn `S` lưu, `Q` thoát. Phiên thu sau sẽ dùng cấu hình mới.

## Lưu tạm trên máy và nối tiếp dữ liệu

Dữ liệu local:

```text
/home/anhquoc2910/Lerobot_project/datasets/stack_3_box_white_green_blue_real
```

Chạy lại **cùng lệnh** để thu tiếp. Ví dụ: đã có episode 0–9 thì lần chạy thêm
10 episode sẽ tạo 10–19. Script dùng `resume` khi đã có dataset và tắt cơ chế
tự thêm timestamp vào tên repo.

Script tải đầy đủ dữ liệu cũ từ Hub nếu bắt đầu bằng một thư mục local mới.
Các shard data/video/metadata episode cũ được kiểm tra SHA-256 và giữ nguyên;
thống kê tổng, task và `meta/info.json` được cập nhật để bao gồm episode mới.
LeRobot v3 có thể gộp nhiều episode vào cùng shard trong một phiên, nhưng
`episode_index` vẫn tách riêng từng lần thu.

Upload không gửi lệnh xóa file trên Hub. Commit chỉ được chấp nhận nếu phiên bản
Hub vẫn trùng với phiên bản đã đọc trước khi thu (`parent_commit`). Nếu Hub bị
cập nhật từ một máy khác trong lúc thu, upload dừng và giữ nguyên dữ liệu local.
Script cập nhật tag định dạng `v3.0` đến commit mới để LeRobot đọc được dữ liệu mới.

**Mặc định mới: tự xóa bản dataset local sau khi upload và xác minh thành công.**
Script đối chiếu số episode/frame và checksum từng file với commit đã upload
(Git blob SHA-1 hoặc LFS SHA-256). Nếu upload lỗi, thiếu file, sai checksum hoặc
còn ảnh/frame local chưa có trên Hub, script giữ dữ liệu local. Chỉ dọn thư mục
dataset đang dùng, bao gồm cache tải/upload nằm bên trong thư mục đó.

Một file trạng thái nhỏ được giữ lại cạnh thư mục dataset:

```text
datasets/.stack_3_box_white_green_blue_real.hub_state.json
```

Lần sau chạy **cùng lệnh, cùng đường dẫn**, script tự tải dữ liệu đã có từ Hub
rồi tiếp tục episode mới. Vì vậy, sau mỗi lần dọn sẽ cần tải lại toàn bộ dataset
ở phiên tiếp theo. Dữ liệu trên Hub được giữ nguyên khi dọn bản local.

Nếu muốn giữ dataset local để thu tiếp nhanh hơn hoặc train ngay trên máy:

```bash
bash record_real_stack_3_box.sh --episodes 10 --keep-local
```

Nếu upload lỗi mạng, thử lại bằng lệnh này; lệnh không kết nối robot:

```bash
bash record_real_stack_3_box.sh --upload-only
```

Lệnh này cũng tự dọn local sau khi xác minh thành công. Thêm `--keep-local` nếu
muốn upload mà giữ dữ liệu trên máy.

Nếu Hub đã đổi từ máy khác, dùng một thư mục mới để tải phiên bản mới nhất rồi
thu tiếp, ví dụ `--root "$PWD/datasets/stack_real_synced"`. Các episode local chưa
upload trong thư mục cũ vẫn được giữ; không tự gộp một bản local bị lệch phiên bản.

## Kiểm tra mà không thu

```bash
bash record_real_stack_3_box.sh --check
```

Lệnh chỉ đọc trạng thái đăng nhập, metadata Hub, cấu hình và sự tồn tại của cổng.
Không kết nối motor, không thu episode, không upload. Nếu chưa đăng nhập Hugging
Face, chạy `hf auth login` trong môi trường `lerobot` với tài khoản có quyền ghi repo.
