# Công thức nên bổ sung vào Mục 4.4

File này tổng hợp các công thức nên thêm vào Mục 4.4: **Phân tích độ mượt, độ trễ và khả năng bám lệnh**. Mục tiêu là giải thích rõ các chỉ số đang xuất hiện trong Bảng 4.5, Bảng 4.6, Bảng 4.7 và Hình 4.9.

## 1. Đoạn nên thêm vào đầu Mục 4.4

Nên thêm sau đoạn giới thiệu hai chuỗi tín hiệu `policy_action[t]` và `observation.state[t]`.

```latex
Gọi $N$ là số frame trong một rollout và $J=6$ là số khớp của robot.
Tại frame $t$, lệnh hành động do policy sinh ra được ký hiệu là
$a_t \in \mathbb{R}^{J}$, còn trạng thái khớp thực tế đo từ robot được ký hiệu là
$s_t \in \mathbb{R}^{J}$. Với khớp thứ $j$, hai giá trị tương ứng được viết là
$a_{t,j}$ và $s_{t,j}$.

Trong các phân tích sau, các chỉ số được tính trên toàn bộ rollout, sau đó lấy trung bình
hoặc tổng hợp trên cả 6 khớp, trừ khi có ghi rõ là đánh giá theo từng khớp.
```

## 2. Công thức cho Mục 4.4.1: Độ mượt

Hiện tại mục 4.4.1 đã có công thức:

```latex
\Delta x_t = x_t - x_{t-1}.
```

Nên giữ công thức này, sau đó thêm các công thức dưới đây để giải thích các cột trong Bảng 4.5.

### 2.1. Mean absolute delta

```latex
Mean absolute delta được dùng để đo mức thay đổi trung bình giữa hai frame liên tiếp:
\[
\mathrm{MeanAbsDelta}
=
\frac{1}{(N-1)J}
\sum_{t=2}^{N}
\sum_{j=1}^{J}
\left|x_{t,j} - x_{t-1,j}\right|.
\]
Giá trị này càng nhỏ thì tín hiệu điều khiển hoặc chuyển động thực tế càng mượt.
```

### 2.2. P95 absolute delta

```latex
Bên cạnh giá trị trung bình, đồ án sử dụng phân vị 95 của độ thay đổi tuyệt đối:
\[
\mathrm{P95AbsDelta}
=
Q_{0.95}\left(\left|\Delta x_{t,j}\right|\right),
\]
trong đó $Q_{0.95}(\cdot)$ là toán tử lấy phân vị 95 trên toàn bộ các frame và các khớp.
Chỉ số này cho biết mức thay đổi lớn nhưng vẫn đại diện cho phần lớn dữ liệu,
ít bị chi phối bởi các ngoại lệ đơn lẻ hơn so với giá trị lớn nhất.
```

### 2.3. Max absolute delta

```latex
Giá trị thay đổi lớn nhất trong rollout được xác định bởi:
\[
\mathrm{MaxAbsDelta}
=
\max_{t,j}\left|\Delta x_{t,j}\right|.
\]
Chỉ số này giúp phát hiện các thời điểm xuất hiện lệnh điều khiển hoặc chuyển động
đột ngột.
```

### 2.4. Vận tốc, gia tốc và jerk

Nên thêm đoạn này trước khi nói về RMS jerk.

```latex
Để đánh giá mức độ giật của tín hiệu, vận tốc, gia tốc và jerk được xấp xỉ
bằng sai phân hữu hạn theo thời gian:
\[
v_t = \frac{x_t - x_{t-1}}{\Delta t},
\]
\[
a_t = \frac{x_t - 2x_{t-1} + x_{t-2}}{\Delta t^2},
\]
\[
j_t =
\frac{x_t - 3x_{t-1} + 3x_{t-2} - x_{t-3}}{\Delta t^3}.
\]
Trong thí nghiệm này, dữ liệu được ghi ở tần số 30 FPS nên
\[
\Delta t = \frac{1}{30}\ \mathrm{s}.
\]
```

### 2.5. RMS jerk

```latex
RMS jerk được tính bằng:
\[
\mathrm{RMSJerk}
=
\sqrt{
\frac{1}{(N-3)J}
\sum_{t=4}^{N}
\sum_{j=1}^{J}
j_{t,j}^{2}
}.
\]
RMS jerk càng lớn thì tín hiệu càng có xu hướng thay đổi đột ngột,
thể hiện chuyển động kém mượt hơn.
```

## 3. Đoạn hoàn chỉnh có thể thay vào Mục 4.4.1

Nếu muốn thay trực tiếp đoạn hiện tại trong mục 4.4.1, có thể dùng bản dưới đây:

```latex
Độ mượt được đánh giá thông qua mức thay đổi giữa hai frame liên tiếp:
\[
\Delta x_t = x_t - x_{t-1},
\]
trong đó $x_t$ có thể là action do policy sinh ra hoặc state thực tế của robot.

Mean absolute delta được dùng để đo mức thay đổi trung bình trên toàn bộ rollout:
\[
\mathrm{MeanAbsDelta}
=
\frac{1}{(N-1)J}
\sum_{t=2}^{N}
\sum_{j=1}^{J}
\left|x_{t,j} - x_{t-1,j}\right|.
\]
Ngoài giá trị trung bình, đồ án sử dụng thêm phân vị 95:
\[
\mathrm{P95AbsDelta}
=
Q_{0.95}\left(\left|\Delta x_{t,j}\right|\right),
\]
và giá trị thay đổi lớn nhất:
\[
\mathrm{MaxAbsDelta}
=
\max_{t,j}\left|\Delta x_{t,j}\right|.
\]

Để đánh giá mức độ giật của tín hiệu, vận tốc, gia tốc và jerk được xấp xỉ bằng:
\[
v_t = \frac{x_t - x_{t-1}}{\Delta t},
\]
\[
a_t = \frac{x_t - 2x_{t-1} + x_{t-2}}{\Delta t^2},
\]
\[
j_t =
\frac{x_t - 3x_{t-1} + 3x_{t-2} - x_{t-3}}{\Delta t^3}.
\]
Với tần số ghi dữ liệu 30 FPS, ta có $\Delta t = 1/30$ s.
RMS jerk được tính bởi:
\[
\mathrm{RMSJerk}
=
\sqrt{
\frac{1}{(N-3)J}
\sum_{t=4}^{N}
\sum_{j=1}^{J}
j_{t,j}^{2}
}.
\]
Các chỉ số này được dùng để so sánh độ mượt giữa lệnh action của policy
và trạng thái chuyển động thực tế của robot.
```

## 4. Công thức cho Mục 4.4.2: Độ trễ và tracking

Hiện tại mục 4.4.2 đã có công thức:

```latex
e_t(\tau) = state_{t+\tau} - action_t.
```

Nên chỉnh ký hiệu lại đồng nhất với phần đầu:

```latex
e_t(\tau) = s_{t+\tau} - a_t.
```

Sau đó thêm các công thức sau.

### 4.1. Tracking error theo lag

```latex
Với một độ trễ $\tau$, sai số tracking giữa action và state được xác định bởi:
\[
e_t(\tau) = s_{t+\tau} - a_t,
\]
trong đó $a_t$ là action do policy sinh tại frame $t$ và $s_{t+\tau}$ là state thực tế
của robot sau $\tau$ frame. Khi đó, số mẫu hợp lệ còn lại là $M=N-\tau$.
```

### 4.2. MAE

```latex
Sai số tuyệt đối trung bình được tính bởi:
\[
\mathrm{MAE}(\tau)
=
\frac{1}{MJ}
\sum_{t=1}^{M}
\sum_{j=1}^{J}
\left|e_{t,j}(\tau)\right|.
\]
MAE cho biết mức sai lệch trung bình giữa lệnh policy và trạng thái robot sau khi
đã xét tới độ trễ phản hồi.
```

### 4.3. RMSE

```latex
Sai số căn trung bình bình phương được tính bởi:
\[
\mathrm{RMSE}(\tau)
=
\sqrt{
\frac{1}{MJ}
\sum_{t=1}^{M}
\sum_{j=1}^{J}
e_{t,j}^{2}(\tau)
}.
\]
So với MAE, RMSE nhạy hơn với các sai số lớn, do đó phù hợp để phát hiện
các giai đoạn robot bám lệnh kém.
```

### 4.4. P95 error và Max error

```latex
Phân vị 95 của sai số tracking được định nghĩa là:
\[
\mathrm{P95Error}(\tau)
=
Q_{0.95}\left(\left|e_{t,j}(\tau)\right|\right).
\]
Sai số lớn nhất trong rollout được xác định bởi:
\[
\mathrm{MaxError}(\tau)
=
\max_{t,j}\left|e_{t,j}(\tau)\right|.
\]
```

### 4.5. Tỉ lệ trong ngưỡng ±5 đơn vị

```latex
Tỉ lệ mẫu có sai số nằm trong ngưỡng $\pm 5$ đơn vị được tính bởi:
\[
R_{\pm 5}(\tau)
=
\frac{1}{MJ}
\sum_{t=1}^{M}
\sum_{j=1}^{J}
\mathbf{1}
\left(
\left|e_{t,j}(\tau)\right| \leq 5
\right)
\times 100\%,
\]
trong đó $\mathbf{1}(\cdot)$ là hàm chỉ thị, nhận giá trị 1 nếu điều kiện đúng
và 0 nếu điều kiện sai.
```

### 4.6. Chọn lag tốt nhất

```latex
Độ trễ phản hồi tốt nhất được chọn theo tiêu chí cực tiểu hóa RMSE toàn cục:
\[
\tau^{*}
=
\arg\min_{\tau \in \{0,1,\ldots,15\}}
\mathrm{RMSE}(\tau).
\]
Trong thí nghiệm, $\tau^{*}=5$ frame. Với tần số 30 FPS, độ trễ này tương ứng:
\[
T_{\mathrm{lag}}
=
\frac{\tau^{*}}{30}
=
\frac{5}{30}
\approx 0{,}167\ \mathrm{s}
=
166{,}7\ \mathrm{ms}.
\]
```

## 5. Đoạn hoàn chỉnh có thể thay vào Mục 4.4.2

```latex
Để ước lượng độ trễ phản hồi của robot, đồ án so sánh action tại thời điểm $t$
với state tại thời điểm $t+\tau$, trong đó $\tau$ là độ trễ tính theo số frame:
\[
e_t(\tau) = s_{t+\tau} - a_t.
\]
Ở đây, $a_t$ là action do policy sinh ra tại frame $t$, còn $s_{t+\tau}$ là state
thực tế của robot sau $\tau$ frame. Với mỗi giá trị $\tau$, số mẫu hợp lệ còn lại là
$M=N-\tau$.

Các chỉ số tracking toàn cục được tính như sau:
\[
\mathrm{MAE}(\tau)
=
\frac{1}{MJ}
\sum_{t=1}^{M}
\sum_{j=1}^{J}
\left|e_{t,j}(\tau)\right|,
\]
\[
\mathrm{RMSE}(\tau)
=
\sqrt{
\frac{1}{MJ}
\sum_{t=1}^{M}
\sum_{j=1}^{J}
e_{t,j}^{2}(\tau)
}.
\]
Ngoài ra, phân vị 95 và sai số lớn nhất được xác định bởi:
\[
\mathrm{P95Error}(\tau)
=
Q_{0.95}\left(\left|e_{t,j}(\tau)\right|\right),
\]
\[
\mathrm{MaxError}(\tau)
=
\max_{t,j}\left|e_{t,j}(\tau)\right|.
\]
Tỉ lệ mẫu có sai số nằm trong ngưỡng $\pm 5$ đơn vị được tính bởi:
\[
R_{\pm 5}(\tau)
=
\frac{1}{MJ}
\sum_{t=1}^{M}
\sum_{j=1}^{J}
\mathbf{1}
\left(
\left|e_{t,j}(\tau)\right| \leq 5
\right)
\times 100\%.
\]

Độ trễ tốt nhất được xác định bằng cách quét $\tau = 0,1,\ldots,15$ frame
và chọn giá trị làm nhỏ nhất RMSE toàn cục:
\[
\tau^{*}
=
\arg\min_{\tau \in \{0,1,\ldots,15\}}
\mathrm{RMSE}(\tau).
\]
```

## 6. Công thức cho Mục 4.4.3: Sai số theo từng khớp

Mục 4.4.3 đang có Bảng 4.7 theo từng khớp. Nên thêm một đoạn ngắn trước bảng để nói rằng các metric được tính riêng cho từng khớp.

```latex
Đối với đánh giá theo từng khớp, các chỉ số MAE và RMSE được tính riêng cho
mỗi khớp $j$:
\[
\mathrm{MAE}_j(\tau)
=
\frac{1}{M}
\sum_{t=1}^{M}
\left|e_{t,j}(\tau)\right|,
\]
\[
\mathrm{RMSE}_j(\tau)
=
\sqrt{
\frac{1}{M}
\sum_{t=1}^{M}
e_{t,j}^{2}(\tau)
}.
\]
Tương tự, P95 error, Max error và tỉ lệ trong ngưỡng $\pm 5$ cũng được tính
trên chuỗi sai số riêng của từng khớp.
```

## 7. Công thức cho Hình 4.9: Normalized tracking summary

Nếu Hình 4.9 có dùng chỉ số chuẩn hóa, nên thêm đoạn này vào cuối mục 4.4.3.

```latex
Do mỗi khớp có miền giá trị hoạt động khác nhau, đồ án sử dụng thêm RMSE chuẩn hóa
để so sánh tương đối giữa các khớp:
\[
\mathrm{NRMSE}_j
=
\frac{\mathrm{RMSE}_j}
{\max(a_j) - \min(a_j)}.
\]
Trong đó $\max(a_j)$ và $\min(a_j)$ lần lượt là giá trị lớn nhất và nhỏ nhất
của action ở khớp $j$ trong rollout đang xét. Chỉ số này giúp đánh giá sai số tracking
tương đối so với biên độ hoạt động của từng khớp.
```

## 8. Gợi ý vị trí chèn trong bản hiện tại

Theo bản `main (2).pdf`, có thể chèn như sau:

| Vị trí trong đồ án | Nội dung nên thêm |
|---|---|
| Sau đoạn mở đầu Mục 4.4 | Ký hiệu \(N\), \(J\), \(a_t\), \(s_t\) |
| Sau công thức (4.1) | MeanAbsDelta, P95AbsDelta, MaxAbsDelta, vận tốc, gia tốc, jerk, RMSJerk |
| Sau công thức (4.2) | MAE, RMSE, P95Error, MaxError, \(R_{\pm 5}\), \(\tau^\*\) |
| Trước Bảng 4.7 | MAE/RMSE theo từng khớp |
| Trước hoặc sau Hình 4.9 | NRMSE nếu hình có dùng chuẩn hóa |

## 9. Lưu ý để tránh làm mục 4.4 quá nặng

Không nên đưa toàn bộ diễn giải dài vào luận văn nếu trang bị hạn chế. Bản tối ưu nên gồm:

1. Một đoạn ký hiệu chung.
2. Công thức độ mượt: MeanAbsDelta, RMSJerk.
3. Công thức tracking: MAE, RMSE, P95, \(R_{\pm 5}\).
4. Công thức chọn lag tốt nhất.
5. Công thức NRMSE nếu Hình 4.9 có chuẩn hóa.

Các phần giải thích chi tiết hơn có thể để trong phụ lục hoặc bỏ bớt nếu chương 4 quá dài.
