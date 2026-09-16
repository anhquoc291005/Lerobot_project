# Bảng Thông Số & Tọa Độ Camera, Vùng Mục Tiêu (SO-101 MuJoCo Simulation)

Tài liệu này tổng hợp chi tiết hệ trục tọa độ, vị trí, góc xoay của 2 camera (`top_cam`, `front_cam`), vùng mục tiêu xếp chồng (`stack_target_zone`), và các khối hộp trong môi trường mô phỏng MuJoCo.

---

## 1. Quy Ước Hệ Trục Tọa Độ $(X, Y, Z)$

Hệ tọa độ tuân theo quy tắc bàn tay phải (Right-Handed System) chuẩn của MuJoCo:

```text
                  +Z (Chiều cao - Hướng lên trần nhà)
                  ▲
                  │   +X (Trước - Sau: Hướng từ đế robot ra phía trước bàn)
                  │  ▲
                  │ /
                  │/
  (-Y: Bên Phải) ─┼────────► +Y (Trái - Phải: Hướng sang bên trái robot)
                 (0,0,0) [Tâm đế Robot SO-101 trên mặt bàn]
```

* **Gốc tọa độ $(0, 0, 0)$**: Nằm tại **tâm đế (base)** của cánh tay robot SO-101 ngay trên mặt bàn (`floor`).
* **Trục $+X$**: Hướng thẳng từ đế robot ra phía trước mặt bàn (vùng thao tác gắp hộp $X \in [0.15\text{ m}, 0.35\text{ m}]$).
* **Trục $+Y$**: Hướng sang **bên trái** của robot (khi đứng từ sau lưng robot nhìn về phía trước).
* **Trục $-Y$**: Hướng sang **bên phải** của robot.
* **Trục $+Z$**: Hướng thẳng đứng lên trời ($Z = 0$ là mặt phẳng bàn làm việc).

---

## 2. Sơ Đồ Bố Trí Không Gian (Mặt Phẳng $XY$)

```text
       +Y (Bên Trái)
 +0.08 m ┼              [Khối Xanh Biển]
         │
   0.0 m ┼── [Robot SO-101] ─── [top_cam] (X=0.22, Z=0.54) ─── [Vùng Đích Vàng C4] ◄── [front_cam]
         │    (0, 0, 0)         [Khối Xanh Lá] (X=0.22, Y=0)  (X=0.264, Y=-0.0175)   (X=0.65, Y=0, Z=0.14)
         │                      (Nhìn thẳng xuống mặt bàn)                           (Nghiêng ~13.5°, nhìn chéo)
 -0.08 m ┼               [Khối Đen / Trắng]
         │
         └─────────────┼───────────────┼───────────────┼───────────────┼────────► +X (Phía trước)
                     0.0 m          +0.22 m         +0.264 m        +0.65 m
```

---

## 3. Thông Số Chi Tiết Các Đối Tượng

### 3.1. Camera Top Trên Trần (`top_cam`)
* **Tọa độ vị trí (`pos`)**: `(X=0.22, Y=0.0, Z=0.54)`
  * $X = 0.22\text{ m}$ ($22\text{ cm}$ trước đế robot, đặt ngay trung tâm vùng gắp hộp).
  * $Y = 0.0\text{ m}$ (chính giữa tâm trục robot).
  * $Z = 0.54\text{ m}$ ($54\text{ cm}$ thẳng đứng trên nóc trần nhìn xuống bàn).
* **Định hướng (`xyaxes`)**: `"0 0.999391 -0.0349 -1 0 0"`
  * Camera nhìn từ trên xuống vuông góc với mặt sàn, bao quát toàn vẹn lưới $5 \times 7$.
* **Góc mở (`fovy`)**: `55°`
* **Khai báo XML**:
  ```xml
  <camera name="top_cam" pos="0.22 0 0.54" xyaxes="0 0.999391 -0.0349 -1 0 0" fovy="55" />
  ```

---

### 3.2. Camera Phía Trước Nhìn Chính Diện (`front_cam`)
* **Tọa độ vị trí (`pos`)**: `(X=0.65, Y=0.0, Z=0.14)`
  * $X = 0.65\text{ m}$ ($65\text{ cm}$ trước đế robot, nằm trên vách khung trước).
  * $Y = 0.0\text{ m}$ (ngay chính giữa tim robot, nhìn trực diện).
  * $Z = 0.14\text{ m}$ (cao hơn mặt bàn $14\text{ cm}$).
* **Định hướng (`xyaxes`)**: `"0 1 0 -0.233445 0 0.972370"`
  * Trục quang học của camera hướng chéo chúi xuống mặt bàn với góc nghiêng $\approx 13.5^\circ$ ($\arcsin(0.2334) \approx 13.5^\circ$).
* **Góc mở (`fovy`)**: `52°`
* **Khai báo XML**:
  ```xml
  <camera name="front_cam" pos="0.65 0 0.14" xyaxes="0 1 0 -0.233445 0 0.972370" fovy="52" />
  ```

---

### 3.3. Vùng Mục Tiêu Đặt Vật - Tấm Vàng (`stack_target_zone`)
* **Tọa độ tâm (`pos`)**: `(X=0.264, Y=-0.0175, Z=0.001)`
  * Cách đế robot $26.4\text{ cm}$ về phía trước.
  * Lệch nhẹ sang bên phải robot $1.75\text{ cm}$.
  * Nổi sát trên mặt sàn $1\text{ mm}$ ($Z=0.001\text{ m}$).
* **Vị trí theo ô lưới**: Nằm gọn trong **ô C4**, sát với đường biên ngăn cách giữa ô **C4** và **D4**.
* **Kích thước khai báo (`size`)**: `0.0267 0.0267 0.001` (bán kính / half-size).
* **Kích thước thực tế**: Dài $53.4\text{ mm} \times$ Rộng $53.4\text{ mm} \times$ Dày $2\text{ mm}$.
* **Khai báo XML**:
  ```xml
  <site name="stack_target_zone" pos="0.264 -0.0175 0.001" size="0.0267 0.0267 0.001" type="box" material="target_zone_mat" />
  ```

---

### 3.4. Tọa Độ Khởi Tạo Mặc Định Của 3 Khối Hộp ($30 \times 30 \times 30\text{ mm}$)
* **Khối TRẮNG (`box_white`)**: `(X=0.22, Y=-0.08, Z=0.015)` (lệch phải $8\text{ cm}$)
* **Khối XANH LÁ (`box_green`)**: `(X=0.20, Y=0.0, Z=0.015)` hoặc `(X=0.25, Y=0.0, Z=0.015)` (chính giữa)
* **Khối XANH BIỂN (`box_blue`)**: `(X=0.22, Y=0.08, Z=0.015)` (lệch trái $8\text{ cm}$)

---

## 4. Các File Cần Chỉnh Sửa Trong Dự Án

| Mục đích | Tên file & Đường dẫn | Dòng cần sửa |
| :--- | :--- | :--- |
| **Sửa tọa độ 3D camera & tấm vàng** | `reference/sim_assets/so_arm100_simulation/SO101/scene.xml` | Dòng 41, 42, 47 |
| **Sửa đồng bộ file mô phỏng phụ** | `SO-ARM100/Simulation/SO101/scene.xml` | Dòng 41, 42, 47 |
| **Sửa tọa độ đích trong logic thu thập** | `src/sim_mujoco/config_sim_collect_stack_3_blocks.json` | `target_zone_pos` (dòng 39-42) |
| **Sửa độ phân giải & độ sáng camera** | `src/sim_mujoco/config_sim_collect_stack_3_blocks.json` | `image_height`, `image_width` (dòng 7-9) |

---

## 5. Vùng Hoạt Động (Workspace) Của Robot SO-101

### 5.1. Giới Hạn Động Học Vật Lý (Kinematic Limits)
* **Tầm với tối đa ($R_{max}$)**: $\approx 0.38\text{ m}$ ($38\text{ cm}$ tính từ tâm đế khi các khớp duỗi thẳng).
* **Vùng chết / Cận đế ($R_{min}$)**: $< 0.12\text{ m}$ ($12\text{ cm}$ - tay robot không gập quá sát thân để tránh tự va chạm).
* **Vùng thao tác tối ưu (Sweet Spot)**: Bán kính $R \in [0.14\text{ m}, 0.35\text{ m}]$, góc xoay $\pm 60^\circ$ quanh trục giữa bàn.
* **Độ cao thao tác ($Z$)**: Mặt bàn $Z = 0.0\text{ m}$ đến độ cao nhấc vật an toàn $Z \approx 0.15\text{ m} - 0.25\text{ m}$.

### 5.2. Bản Đồ Lưới Ma Trận 5x7 (Matrix 5x7 Table)

Dự án chia mặt bàn thao tác thành lưới 5 hàng (A - E) và 7 cột (1 - 7). Để đảm bảo robot thao tác mượt mà và không vượt quá giới hạn cánh tay, hệ thống chia thành các vùng:

```text
       Cột 1     Cột 2     Cột 3     Cột 4     Cột 5     Cột 6     Cột 7
    (Y ~ -0.3m)                                                    (Y ~ +0.3m)
  ┌─────────┬─────────┬─────────┬─────────┬─────────┬─────────┬─────────┐
A │   [X]   │   [X]   │   [X]   │   [X]   │   [X]   │   [X]   │   [X]   │ (X ~ 0.50m - Quá xa tầm với)
  ├─────────┼─────────┼─────────┼─────────┼─────────┼─────────┼─────────┤
B │   [X]   │   [O]   │   [O]   │   [O]   │   [O]   │   [O]   │   [X]   │ (X ~ 0.14m - Tầm với gần)
  ├─────────┼─────────┼─────────┼─────────┼─────────┼─────────┼─────────┤
C │   [X]   │   [O]   │   [O]   │   [★]   │   [O]   │   [O]   │   [X]   │ (X ~ 0.24m - VÙNG LÝ TƯỞNG NHẤT)
  ├─────────┼─────────┼─────────┼─────────┼─────────┼─────────┼─────────┤
D │   [X]   │   [X]   │   [O]   │   [O]   │   [O]   │   [X]   │   [X]   │ (X ~ 0.34m - Tầm với xa)
  ├─────────┼─────────┼─────────┼─────────┼─────────┼─────────┼─────────┤
E │   [X]   │   [X]   │   [X]   │   [X]   │   [X]   │   [X]   │   [X]   │ (X ~ 0.05m - Sát mép/Khuất cam)
  └─────────┴─────────┴─────────┴─────────┴─────────┴─────────┴─────────┘
                                   ▲
                             [Đế Robot (0,0)]
```

* **`[★]` (Ô C4)**: Vị trí đặt tấm vàng mục tiêu (`stack_target_zone`, tọa độ $X=0.264, Y=-0.0175$).
* **`[O]` (13 ô an toàn)**: Các ô nằm hoàn hảo trong tầm với của SO-101 dùng để random vị trí ban đầu của 3 khối hộp:
  * Hàng B: `b2`, `b3`, `b4`, `b5`, `b6`
  * Hàng C: `c2`, `c3`, `c4`, `c5`, `c6`
  * Hàng D: `d3`, `d4`, `d5`
* **`[X]` (Vùng loại bỏ - Excluded)**: Các ô quá xa cánh tay (hàng A, mép ngoài d1, d2, d6, d7), quá sát robot (hàng E) hoặc ngoài góc nhìn ổn định.

---

## 6. Quy Chuẩn Bố Trí Trong Hộp Khung Nhôm ($60 \times 100 \times 75\text{ cm}$)

Kích thước không gian bao quanh chuẩn:
* **Chiều sâu (Trục X - Trước/Sau)**: $60\text{ cm} = 0.60\text{ m}$
* **Chiều rộng (Trục Y - Trái/Phải)**: $100\text{ cm} = 1.00\text{ m}$
* **Chiều cao (Trục Z - Độ cao trần)**: $75\text{ cm} = 0.75\text{ m}$

### 6.1. Sơ Đồ Bố Trí Trong Khung Nhôm (Nhìn từ trên xuống)

```text
 ┌───────────────────────── Chiều Rộng: 100 cm ─────────────────────────┐
 │                                                                       │
 │  [-50 cm]                      [Y = 0]                       [+50 cm] │
 │  (Vách Phải)                                              (Vách Trái) │
 │                                                                       │
 │  ┌─────────────────────────────────────────────────────────────────┐  │ ▲
 │  │                         [THANH NHÔM SAU]                        │  │ │
 │  │                                                                 │  │ │
 │  │                         [Robot SO-101]                          │  │ │
 │  │                            (0, 0, 0)                            │  │ │
 │  │                     (Cách mép sau: 10 cm)                       │  │ │
 │  │                                                                 │  │ │
 │  │                                              [top_cam]        │  │ │
 │  │                                           (Gắn giữa nóc trần)   │  │ │
 │  │                                          (X=0.22, Y=0.0, Z=0.54) │  │ │
 │  │                                          (Nhìn thẳng xuống bàn)  │  │ │
 │  │                                                    │             │  │ │
 │  │                                                    ▼             │  │ │
 │  │                   [Tấm Vàng Mục Tiêu C4]                         │  │ │
 │  │                     (X=0.264, Y=-0.0175)                         │  │ │
 │  │                                                                  │  │ │
 │  │                              ▲                                   │  │ │
 │  │                              │                                   │  │ │
 │  │                         [front_cam]                              │  │ │
 │  │                 (Gắn giữa thanh nhôm trước)                      │  │ │
 │  │                   (X=0.65, Y=0.0, Z=0.14)                        │  │ │
 │  │                   (Nghiêng ~13.5° nhìn chéo)                     │  │ │
 │  │                        [THANH NHÔM TRƯỚC]                        │  │ │
 │  └──────────────────────────────────────────────────────────────────┘  │ ▼
 └────────────────────────────────────────────────────────────────────────┘
```

### 6.2. Bảng Tọa Độ Định Vị Chuẩn Trong Khung

| Đối tượng | Tọa độ chuẩn $(X, Y, Z)$ | Vị trí lắp đặt cơ khí thực tế | Khoảng cách an toàn |
| :--- | :--- | :--- | :--- |
| **Đế Robot SO-101** | `(0.0, 0.0, 0.0)` | Bắt cố định lên mặt bàn, cách mép thanh nhôm sau $10\text{ cm}$, chính giữa bề ngang $100\text{ cm}$. | Cách vách trái: $50\text{ cm}$<br>Cách vách phải: $50\text{ cm}$ |
| **Tầm Với Max ($R_{max}$)** | Bán kính $38\text{ cm}$ quanh đế | Vùng quét tối đa của đầu gắp khi duỗi thẳng. | Cách vách trước: $\ge 12\text{ cm}$<br>Cách 2 vách bên: $\ge 12\text{ cm}$ (Không lo va đập) |
| **Tấm Vàng Mục Tiêu (`C4`)** | `(0.264, -0.0175, 0.001)` | Dán cố định trên mặt bàn/tấm nền. | Cách mép sau: $36.4\text{ cm}$<br>Cách vách trước: $23.6\text{ cm}$ |
| **Top Camera (`top_cam`)** | `(0.22, 0.0, 0.54)` | Bắt gá kẹp trên thanh nhôm nóc khung, chính giữa trục $Y=0$. | Cao $54\text{ cm}$ (thẳng đứng trên mặt bàn), nhìn vuông góc trực diện xuống lưới $5 \times 7$. |
| **Front Camera (`front_cam`)** | `(0.65, 0.0, 0.14)` | Gá kẹp tại chính giữa thanh nhôm ngang phía trước, nhìn chính diện vào robot. | Nằm tại tim giữa $Y=0$, cao $14\text{ cm}$ so với mặt sàn, nghiêng $\approx 13.5^\circ$ so với phương ngang chúi nhẹ vào robot. |


