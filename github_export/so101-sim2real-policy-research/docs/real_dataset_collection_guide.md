# Huong Dan Thu Dataset Real SO101

Huong dan nay dung cho setup SO101 leader/follower, 2 camera `top` va `left`, co man hinh setup grid truoc moi episode de dat vat va hop theo plan.

## 1. Kiem Tra Port Va Camera

Kiem tra arm serial ports:

```bash
ls /dev/ttyACM*
```

Trong lenh mau hien tai:

```text
Follower arm: /dev/ttyACM1
Leader arm:   /dev/ttyACM0
Top camera:   /dev/video5
Left camera:  /dev/video4
```

Kiem tra camera mode:

```bash
v4l2-ctl --device=/dev/video5 --list-formats-ext
v4l2-ctl --device=/dev/video4 --list-formats-ext
```

Nen dung `640x480`, `30 fps`, `MJPG` neu camera ho tro.

## 2. Calibration Dang Dung

Calibration follower:

```text
/home/aira/.cache/huggingface/lerobot/calibration/robots/so_follower/my_awesome_follower_arm.json
```

Calibration leader:

```text
/home/aira/.cache/huggingface/lerobot/calibration/teleoperators/so_leader/my_awesome_leader_arm.json
```

Hai file nay da duoc copy tu:

```text
/home/aira/VietAnh-THUONG/my_awesome_follower_arm.json
/home/aira/VietAnh-THUONG/my_awesome_leader_arm.json
```

## 3. Plan Setup Vat Va Hop

Plan mac dinh:

```text
/home/aira/VietAnh-THUONG/SO101_sim2real_policy-main/datasets/real_setup_plan_reachable_5x9_200_canh_sat.jsonl
```

Plan nay co:

```text
200 episode
18 o hop le cho vat va hop
Khong trung cap object_cell -> container_cell
Co them truong hop canh_sat
```

Vung hop le nam trong bien:

```text
a1, b2, c3, d4, d5, d6, c7, b8, a9
```

Loai rieng:

```text
a3, a4, a5, a6, b4, b5
```

Tao lai plan neu can:

```bash
cd /home/aira/VietAnh-THUONG/SO101_sim2real_policy-main

python src/sim_mujoco/real_dataset_setup_plan.py \
  --num-episodes 200 \
  --output datasets/real_setup_plan_reachable_5x9_200_canh_sat.jsonl
```

## 4. Lenh Thu Dataset Real

Dung lenh nay de chac chan chay dung code trong `/home/aira/VietAnh-THUONG/lerobot`, vi moi truong hien tai co the dang tro `lerobot-record` sang mot folder khac.

```bash
cd /home/aira/VietAnh-THUONG/lerobot

PYTHONPATH=/home/aira/VietAnh-THUONG/lerobot/src \
python -m lerobot.scripts.lerobot_record \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM1 \
  --robot.id=my_awesome_follower_arm \
  --robot.calibration_dir=/home/aira/.cache/huggingface/lerobot/calibration/robots/so_follower \
  --robot.cameras="{ top: {type: opencv, index_or_path: /dev/video0, width: 640, height: 480, fps: 30, fourcc: MJPG, backend: V4L2}, front: {type: opencv, index_or_path: /dev/video2, width: 640, height: 480, fps: 30, fourcc: MJPG, backend: V4L2} }" \
  --teleop.type=so101_leader \
  --teleop.port=/dev/ttyACM0 \
  --teleop.id=my_awesome_leader_arm \
  --teleop.calibration_dir=/home/aira/.cache/huggingface/lerobot/calibration/teleoperators/so_leader \
  --dataset.repo_id=vasco281204/so101_pick_red_block_real_5x9_reachable \
  --dataset.root=/home/aira/VietAnh-THUONG/SO101_sim2real_policy-main/datasets/so101_pick_red_block_real_5x9_reachable \
  --dataset.num_episodes=200 \
  --dataset.fps=30 \
  --dataset.episode_time_s=30 \
  --dataset.reset_time_s=5 \
  --dataset.single_task="Pick the object and place it into the box." \
  --dataset.video=true \
  --dataset.streaming_encoding=true \
  --dataset.encoder_threads=2 \
  --dataset.encoder_queue_maxsize=30 \
  --dataset.push_to_hub=true \
  --dataset.private=false \
  --display_data=true \
  --setup_enabled=true \
  --setup_camera_key=top \
  --setup_plan=/home/aira/VietAnh-THUONG/SO101_sim2real_policy-main/datasets/real_setup_plan_reachable_5x9_200_canh_sat.jsonl \
  --setup_confirm_log=/home/aira/VietAnh-THUONG/SO101_sim2real_policy-main/datasets/real_setup_confirmed.jsonl
```

Dataset se upload len Hugging Face:

```text
vasco281204/so101_pick_red_block_real_5x9_reachable
```

Neu chua login Hugging Face:

```bash
huggingface-cli login
```

## 5. Quy Trinh Thu Moi Episode

Truoc moi episode, cua so setup top-camera se hien:

```text
OBJECT <cell>
BOX <cell>
relation
```

Lam theo cac buoc:

1. Dat vat vao o `OBJECT`.
2. Dat hop vao o `BOX`.
3. Nhin camera top de chac vat va hop nam dung o.
4. Nhan `Enter` hoac `Space` de bat dau thu episode.
5. Dieu khien leader arm de follower thuc hien pick/place.
6. Khi xong episode, bam phim chuyen nhanh neu khong muon doi het 30 giay.

## 6. Phim Dieu Khien

Trong man hinh setup truoc episode:

```text
Enter / Space : bat dau thu episode
N             : skip vi tri setup hien tai
P             : quay lai vi tri truoc
Q / Esc       : thoat luon
```

Khi dang thu episode:

```text
Right arrow / n : ket thuc nhanh episode hien tai, save va qua episode moi
Left arrow / r  : bo episode hien tai, thu lai
Esc / q         : thoat ngay, encode video va upload Hugging Face
```

## 7. Loi Camera Thuong Gap

Neu gap loi:

```text
failed to set capture_width=640
actual_width=1920
```

Kiem tra camera co ho tro mode khong:

```bash
v4l2-ctl --device=/dev/video5 --list-formats-ext
```

Neu camera ho tro `640x480 MJPG 30fps` ma OpenCV van loi, giu `backend: V4L2` trong camera config.

Neu camera `left` khong nhan `MJPG`, bo rieng `fourcc: MJPG` cua camera `left`:

```text
left: {type: opencv, index_or_path: /dev/video4, width: 640, height: 480, fps: 30, backend: V4L2}
```

## 8. Resume Dataset

Neu dang thu bi dung giua chung va muon tiep tuc dataset cu, them:

```bash
--resume=true
```

Lenh se doc dataset root cu va tiep tuc tu episode da co. Vi du da co 37 episode thi lan resume se thu tiep episode 38. Man hinh setup grid cung se nhay toi vi tri setup tuong ung, khong quay lai episode 0.

Lenh resume mau cho dataset `vasco281204/so101_pick_green_block` voi camera `top=/dev/video0` va `front=/dev/video2`:

```bash
cd /home/aira/VietAnh-THUONG/lerobot

PYTHONPATH=/home/aira/VietAnh-THUONG/lerobot/src \
python -m lerobot.scripts.lerobot_record \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM1 \
  --robot.id=my_awesome_follower_arm \
  --robot.calibration_dir=/home/aira/.cache/huggingface/lerobot/calibration/robots/so_follower \
  --robot.cameras="{ top: {type: opencv, index_or_path: /dev/video0, width: 640, height: 480, fps: 30, fourcc: MJPG, backend: V4L2}, front: {type: opencv, index_or_path: /dev/video2, width: 640, height: 480, fps: 30, fourcc: MJPG, backend: V4L2} }" \
  --teleop.type=so101_leader \
  --teleop.port=/dev/ttyACM0 \
  --teleop.id=my_awesome_leader_arm \
  --teleop.calibration_dir=/home/aira/.cache/huggingface/lerobot/calibration/teleoperators/so_leader \
  --dataset.repo_id=vasco281204/so101_pick_green_block \
  --dataset.root=/home/aira/VietAnh-THUONG/SO101_sim2real_policy-main/datasets/so101_pick_red_block_real_5x9_reachable \
  --dataset.num_episodes=200 \
  --dataset.fps=30 \
  --dataset.episode_time_s=3000 \
  --dataset.reset_time_s=5 \
  --dataset.single_task="Pick the object and place it into the box." \
  --dataset.video=true \
  --dataset.streaming_encoding=true \
  --dataset.encoder_threads=2 \
  --dataset.encoder_queue_maxsize=30 \
  --dataset.push_to_hub=true \
  --dataset.private=false \
  --display_data=true \
  --setup_enabled=true \
  --setup_camera_key=top \
  --setup_plan=/home/aira/VietAnh-THUONG/SO101_sim2real_policy-main/datasets/real_setup_plan_reachable_5x9_200_canh_sat.jsonl \
  --setup_confirm_log=/home/aira/VietAnh-THUONG/SO101_sim2real_policy-main/datasets/real_setup_confirmed.jsonl \
  --resume=true
```

Khi muon dung va upload len Hugging Face, nhan `q` hoac `Esc`. Khong nen dung `Ctrl+C` neu muon LeRobot encode video va push gon gang.

## 9. File Log Sau Khi Thu

Dataset:

```text
/home/aira/VietAnh-THUONG/SO101_sim2real_policy-main/datasets/so101_pick_red_block_real_5x9_reachable
```

Log setup da xac nhan:

```text
/home/aira/VietAnh-THUONG/SO101_sim2real_policy-main/datasets/real_setup_confirmed.jsonl
```

Plan setup:

```text
/home/aira/VietAnh-THUONG/SO101_sim2real_policy-main/datasets/real_setup_plan_reachable_5x9_200_canh_sat.jsonl
```
