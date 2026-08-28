# Báo cáo tuần 15–22/08

ResNet-50 + CIFAR-100, 10 task × 10 lớp, class-incremental, exemplar-free.
Toàn bộ tuần này chạy trên **đúng giao thức của Fly-CL**: checkpoint
`resnet50.tv2_in1k`, `coding_level` 0.3, `expand_dim` 10.000, `ridge_lower` 4.

Mốc so sánh là **bản tái lập** `flycl_baseline.py`, không phải con số công bố —
lý do ở mục *Giới hạn*.

---

## 1. Tóm tắt

Tuần trước kết luận rằng hai ý tưởng gốc của đề tài (projection học được, MLP
head) đều làm model tệ đi, và không tìm được cải tiến nào cho Fly-CL sau bảy can
thiệp.

Tuần này tìm được cải tiến thật, đến từ một hướng không nằm trong danh sách nào
đã đề ra: **chiếu hai tầng của backbone thay vì một**.

Fly-CL chiếu **một** vector đặc trưng (stage 4, 2048 chiều) lên 10.000 chiều rồi
top-k. Thay vào đó, chiếu **cả stage 3 và stage 4** bằng hai projection riêng rồi
**nối lại** trước khi top-k.

| Cách kết hợp | Δ A_T | Δ Ā | Δ Forgetting | Bộ nhớ |
|---|---:|---:|---:|---|
| **concat** `[a ; b]` | **+2.07 ± 0.06** | **+1.37 ± 0.18** | **−1.06** | gấp 4 |

Ba seed, hiệu tính theo cặp từng seed, dương ở cả ba không ngoại lệ.

Nửa sau tuần chuyển sang **AnaCP** (NeurIPS 2025 Spotlight), phương pháp tốt nhất
hiện nay trên nhánh analytic CIL, và ghép tầng Contrastive Projection của nó vào
Fly-CL. Hai kết quả:

| | Δ Ā | Ghi chú |
|---|---:|---|
| Tầng CP của AnaCP | **+0.15 ± 0.06** | 3 seed; rút gọn được thành phép nhân logit với một ma trận `C×C`, khớp 100% ở 16 lần đo |
| **First-Session Adaptation** | **+3.77** | 1 seed; thứ AnaCP để ở Implementation Details và không ablate |
| **FSA + concat + ensemble 5** | **+4.97** | 1 seed; **cấu hình tốt nhất của cả dự án**, Ā 89.05 |

Tức **đóng góp phương pháp luận được trình bày của AnaCP (§4, Lemma 4.1) gần như không đo
được, còn phần thực sự ăn lại là bước tiền xử lý họ không nhận là đóng góp.**

Ba phép kiểm chốt kết luận đó không phải do cài đặt sai:

```
tái lập paper của họ (DINOv2)      95.38 / 92.11   vs công bố 95.43 / 92.15
code họ trên feature ResNet của ta  84.31          vs Fly-CL 84.08     → hoà
code họ trên feature ResNet + FSA   87.80          vs Fly-CL 87.85     → hoà
```

Hai phương pháp rất khác nhau về cơ chế nhưng về **cùng một chỗ** khi cho cùng feature; đổi
backbone thì nhảy 11 điểm. Chi tiết ở mục 7 và 8.

---

## 2. Bảng kết quả chính

Hai khối, **khác giao thức nên không trộn số**: khối trên giữ đúng giao thức Fly-CL
(backbone đóng băng hoàn toàn), khối dưới thêm **FSA** — thích nghi backbone bằng gradient
ở task 0 rồi đóng băng. FSA dùng nhãn của task 0 nên là thiết lập khác, dù phổ biến (APER,
RanPAC, AnaCP đều dùng).

| Cấu hình | FSA | Feature | Mix | Ens | Unit | Bộ nhớ `G` | A_T | Ā | Forgetting | Δ Ā |
|---|:-:|---|---|---:|---:|---:|---:|---:|---:|---:|
| **Giao thức Fly-CL** — 3 seed (1993/2023/2025), Δ so với 84.19 | | | | | | | | | | |
| *Paper công bố* | ✗ | s4 | — | 1 | 10.000 | 0.4 GB | — | *84.61 ± 0.16* | — | |
| **Fly-CL, bản tái lập** | ✗ | s4 | — | 1 | 10.000 | 0.4 GB | 76.76 | 84.19 ± 0.42 | 8.13 | — |
| `expand_dim` 20.000 | ✗ | s4 | — | 1 | 20.000 | 1.6 GB | 76.86 | 82.81 | 7.38 | −1.38 |
| ensemble | ✗ | s4 | — | 5 | 50.000 | 2.0 GB | 77.49 | 85.01 ± 0.46 | 8.09 | +0.82 ± 0.13 |
| **feature mix** | ✗ | **s3+s4** | **concat** | 1 | 20.000 | 1.6 GB | 78.83 | 85.56 ± 0.29 | **7.06** | **+1.37 ± 0.18** |
| all | ✗ | s3+s4 | concat | 2 | 40.000 | 3.2 GB | 79.34 | 86.10 ± 0.39 | 7.20 | +1.90 ± 0.03 |
| **all** | ✗ | **s3+s4** | **concat** | **5** | 100.000 | 8.0 GB | **79.66** | **86.49 ± 0.40** | 7.34 | **+2.30 ± 0.07** |
| **Thêm FSA**  | | | | | | | | | | |
| *Fly-CL, bản tái lập* | ✗ | s4 | — | 1 | 10.000 | 0.4 GB | *76.99* | *84.08* | *8.54* | *—* |
| **Fly-CL, bản tái lập** | ✓ | s4 | — | 1 | 10.000 | **0.4 GB** | 80.93 | **87.85** | 7.74 | **+3.77** |
| **feature mix** | ✓ | **s3+s4** | **concat** | 1 | 20.000 | 1.6 GB | 81.76 | 88.10 | 7.38 | +4.02 |
| all | ✓ | s3+s4 | concat | 2 | 40.000 | 3.2 GB | 82.62 | 88.90 | **6.99** | +4.82 |
| **all** | ✓ | **s3+s4** | **concat** | **5** | 100.000 | 8.0 GB | **82.72** | **89.05** | 7.12 | **+4.97** |

Bốn cột giữa xác định hoàn toàn mỗi dòng: **FSA** có thích nghi backbone ở task 0 không,
**Feature** là những tầng backbone được dùng, **Mix** là phép kết hợp chúng (`concat` =
nối đuôi, thành 20.000 unit), **Ens** là số nhánh.

**Concat cải thiện mạnh nhất ở giai đoạn cuối** — A_T +2.07 ± 0.06, chỉ số quan trọng nhất
của continual learning — và giảm forgetting 1.06. Nó cũng ổn định nhất bảng: σ = 0.29 so
với 0.42 của baseline.

**Dòng `expand_dim` 20.000 là đối chứng quan trọng nhất.** Cũng 20.000 unit, cũng 1.6 GB
như concat, khác đúng một chỗ: 10.000 unit thứ hai chiếu từ **stage 3** hay từ **stage 4
lần nữa**. Chiếu từ stage 3 hơn **2.75 điểm**, và bản thân việc tăng `expand_dim` **làm tệ
đi** 1.38.

**Concat và ensemble độc lập, cộng dồn được**: riêng lẻ +1.37 và +0.82, kết hợp +2.30
(tổng lý thuyết 2.19). Khác `expand_dim`, vốn không cộng dồn — nó cùng làm mã giàu hơn nên
thay thế concat, còn ensemble tác động lên phương sai nên độc lập.

### Đọc khối FSA

**FSA một mình cho +3.77** — lớn hơn mọi can thiệp khác trong bảng, và **miễn phí bộ nhớ**.
Chi phí là một lần huấn luyện adapter ~400s (bão hoà từ epoch 3, xem mục 8) cộng một lần
trích lại feature.

**Nhưng nó hấp thụ phần lớn lợi ích của multi-stage:**

| | không FSA | có FSA |
|---|---:|---:|
| concat | +1.52 | **+0.26** |

Cộng riêng lẻ được 3.77 + 1.52 = 5.29, thực tế chỉ 4.02. Hai hướng chồng lấn 1.27 — hợp lý
vì cả hai đều làm giàu biểu diễn: FSA bằng cách thích nghi trọng số, multi-stage bằng cách
thêm tầng trung gian. Chúng **không thay thế nhau** (cộng lại vẫn hơn từng cái) nhưng phần
lớn thông tin stage 3 mang lại thì FSA đã tự lấy được.

**Ensemble bão hoà ở cùng chỗ trong cả hai khối**: bước 2→5 nhánh cho +0.18 khi không FSA
và +0.15 khi có FSA. `ensemble 2` là điểm đáng dùng; `ensemble 5` chỉ để biết trần.

### Phát biểu nào đứng được

```
trong giao thức Fly-CL:   86.49 ± 0.40,  +2.30 ± 0.07   (3 seed)
nới sang FSA:             89.05,         +4.97          (1 seed)
```

Khoảng lệch tái lập là 0.42–0.80 tuỳ số seed, nên **chỉ dòng `concat + ensemble 5` mới
chênh đủ xa số công bố** (1.88) để nói mạnh hơn "hơn baseline chạy cùng code". Chi tiết ở
mục 9.

Khối FSA **chưa dùng để so với Fly-CL được**: một seed, và đổi giao thức. Hai dòng concat
trong khối đó còn dùng cách xử lý λ khác phần còn lại (mục 6).

---

### So sánh ba phương pháp

Cùng ResNet-50 `tv2_in1k`, CIFAR-100, 10 task, cùng class order. AnaCP chạy bằng
`models/anacp.py` **nguyên bản của họ** trên chính feature của ta, siêu tham số của họ.

| | FSA | Unit | Bộ nhớ | A_T | Ā | Forgetting | Thời gian |
|---|:-:|---:|---:|---:|---:|---:|---:|
| Fly-CL, bản tái lập (3 seed) | ✗ | 10.000 | 0.4 GB | 76.76 | 84.19 ± 0.42 | 8.13 | 97s |
| AnaCP, code của họ (1 seed) | ✗ | — | ~0.4 GB | 76.88 | 84.31 | 9.01 | **25s** |
| **Của ta** — `concat` (3 seed) | ✗ | 20.000 | 1.6 GB | 78.83 | 85.56 ± 0.29 | 7.06 | 105s |
| **Của ta** — `concat + ens 5` (3 seed) | ✗ | 100.000 | 8.0 GB | **79.66** | **86.49 ± 0.40** | 7.34 | ~490s |
| Fly-CL + FSA (1 seed) | ✓ | 10.000 | 0.4 GB | 80.93 | 87.85 | 7.74 | 97s |
| AnaCP + FSA, code của họ (1 seed) | ✓ | — | ~0.4 GB | 80.84 | 87.80 | 8.12 | **22s** |
| **Của ta + FSA** — `concat` (1 seed) | ✓ | 20.000 | 1.6 GB | 81.76 | 88.10 | 7.38 | 107s |
| **Của ta + FSA** — `concat + ens 5` (1 seed) | ✓ | 100.000 | 8.0 GB | **82.72** | **89.05** | **7.12** | 826s |

Ba điều:

**Fly-CL và AnaCP hoà nhau** ở cả hai mức — 84.31 vs 84.19, và 87.80 vs 87.85. Hai phương
pháp rất khác về cơ chế nhưng về cùng một chỗ khi cho cùng feature. AnaCP nhanh hơn 4 lần
(25s vs 97s) vì `D` = 5.000 thay vì `E` = 10.000, đó là lợi thế thật duy nhất của nó ở đây.

**Cải tiến của ta là thứ duy nhất tách khỏi cả hai**, nhưng phải trả bằng bộ nhớ: concat
gấp 4 lần `G` vì `G` cỡ `E²`.

**Con số 95.38 của AnaCP trên DINOv2 không so được** với bảng này: khác backbone. Nó chỉ
nói rằng khoảng cách 84 → 95 nằm ở PTM, không ở phương pháp (mục 7).

---

## 3. Bảng chọn tầng và cách kết hợp

Một seed (1993). Mốc: A_T 76.99, Ā 84.08.

| Cách kết hợp | Feature | Unit | Bộ nhớ | Thời gian | A_T | Ā | Δ Ā |
|---|---|---:|---:|---:|---:|---:|---:|
| **concat** `[a ; b]` | s3+s4 | 20.000 | 1.6 GB | 105s | **79.01** | **85.60** | **+1.52** |
| concat, `b` gộp feature s2+s3 | s2+s3+s4 | 20.000 | 1.6 GB | 105s | 78.43 | 85.15 | +1.07 |
| concat, mỗi tầng một khối riêng | s2+s3+s4 | 30.000 | 3.6 GB | 321s | 78.81 | 85.02 | +0.94 |

### Chỉ tầng liền kề mới đáng ghép

Stage 2 được thử **ba cách khác nhau**, tất cả đều làm tệ đi so với chỉ dùng
stage 3:

| Cách đưa stage 2 vào | Δ so với chỉ dùng s3 |
|---|---:|
| Ghép feature s2 vào rồi chiếu chung (trên `a1_in1k`) | −0.33 |
| Gộp feature s2 vào nhánh `b` của concat | −0.45 |
| **Khối unit riêng cho s2 (concat 3 tầng)** | **−0.58** |

Cách cuối là phép thử công bằng nhất — stage 2 có projection riêng, khối unit
riêng, không phải chen chỗ với stage 3, và tổng số unit tăng từ 20.000 lên
30.000. Vẫn mất 0.58 điểm trong khi tốn **2.25 lần bộ nhớ và 3 lần thời gian**.

Ba phép đo độc lập cùng một hướng đóng lại hướng "thêm nhiều tầng nữa". Cũng
không cần thử stage 1: nó thô hơn stage 2, mà stage 2 đã hại.

Cộng với đối chứng `stage 4 × stage 4` (−0.84), ba điểm vẽ ra một đường cong theo
khoảng cách giữa hai tầng — có **điểm ngọt ở tầng liền kề**:

| Cặp tầng | Cách nhau | Δ Ā |
|---|---|---:|
| stage 4 × stage 4 | 0 tầng | **−0.84** |
| **stage 4 × stage 3** | **1 tầng** | **+1.04** |
| stage 4 × stage 2 | 2 tầng | +0.24 |

Quá giống thì không có thông tin bổ trợ; quá xa thì mức ngữ nghĩa lệch nhau nhiều
quá, tương tác giữa "hoa văn" với "vật gì" ít nghĩa hơn giữa "bộ phận" với "vật
gì".

---

## 4. Bảng ensemble

`m` nhánh độc lập, mỗi nhánh một projection riêng và một `G` riêng, **mọi nhánh
thấy toàn bộ dữ liệu**, cộng logit khi dự đoán. Đo trên Fly-CL nguyên bản, không
kèm concat. Ba seed.

| Nhánh | Seed | Unit | Bộ nhớ | Thời gian | A_T | Ā | Δ Ā theo cặp |
|---:|:-:|---:|---:|---:|---:|---:|---:|
| 1 | 3 | 10.000 | 0.4 GB | 97s | 76.76 | 84.19 ± 0.42 | — |
| 2 | 3 | 20.000 | 0.8 GB | 196s | 77.33 | 84.83 ± 0.39 | **+0.64 ± 0.16** |
| **5** | **3** | 50.000 | 2.0 GB | 489s | **77.49** | **85.01 ± 0.46** | **+0.82 ± 0.13** |
| 10 | **1** | 100.000 | 4.0 GB | 971s | 77.48 | 84.73 | +0.65 |

**Đừng đọc dòng m = 10 như một cú tụt.** Nó là **một seed**, còn các dòng trên là
trung bình **ba seed** — hai loại số khác nhau đặt cùng một cột. So ở **cùng seed
1993** thì m=5 cho 84.75 và m=10 cho 84.73, tức chênh **−0.02**, không phải −0.28.

Dương ở cả ba seed tới m = 5. Bước từ 1 lên 2 cho +0.64, từ 2 lên 5 chỉ thêm
**+0.18**, và từ 5 lên 10 thì **phẳng**. Đo trên nền có `feature mix` (xem
`bao-cao.md`, Bảng 3) cũng cùng kết luận và cùng seed: 85.93 ở m=5, 85.96 ở m=10,
85.97 ở m=20 — từ 5 lên 20 chỉ thêm **+0.04** với 4 lần bộ nhớ và 4.6 lần thời gian.

**Bão hoà ở m = 5.** Xác nhận lại trên feature đã FSA (mục 8): bước 2→5 nhánh cho
+0.15, gần trùng con số +0.18 ở đây — điểm bão hoà không đổi.

Forgetting **không đổi** (8.13 → 8.15 → 8.09) — ensemble không giúp gì ở chỉ số
này, khác concat (−1.06).

Hiệu ứng ensemble **bất biến với cấu hình nền**: đo lần hai trên checkpoint
`a1_in1k` + `coding_level` 0.1 + kết hợp tuyến tính cho +0.40 / +0.88 / +0.91 ở
m = 2/5/10 — gần trùng khít, và cũng bão hoà tại 5.

Nhưng ensemble **không** làm σ giữa các seed nhỏ đi (0.42 → 0.38), dù lý thuyết
nói nó giảm phương sai. Lý do: `--seed` điều khiển **cả** thứ tự lớp **lẫn**
projection, mà ensemble chỉ trung bình hoá được projection. Phần phương sai còn
lại đến từ thứ tự lớp và không khử được. Đây là hạn chế của cách đo hiện tại —
`config.py` gộp hai nguồn ngẫu nhiên vào một cờ.

---

## 5. Kết quả âm

| Can thiệp | Δ Ā |
|---|---:|
| Chia top-k đều giữa hai nửa của concat | −0.05 |
| Top-k theo khối (m = 2…20) | −0.06 |
| Thêm stage 2 vào concat | −0.45 |
| `expand_dim` 20.000 (ở `coding_level` 0.3) | −1.38 |
| 10 khối riêng, mọi khối đủ dữ liệu | −3.47 |
| Hỗn hợp chuyên gia, chia dữ liệu (m=8) | −5.93 |
| Học phép chiếu ở task 0 rồi đóng băng (8 cấu hình) | −0.02 … −0.14 |
| GCV trên thống kê tích luỹ (λ "đúng" thay vì λ hiện tại) | −0.60 |
| **n ma trận cho n task** | **−23.71** |

### Vì sao "n ma trận cho n task" hỏng nặng

Khối `t` sinh ra ở task `t`, nên chỉ thấy dữ liệu từ task `t` trở đi:

```
so mau moi khoi da hoc:  50k 45k 40k 35k 30k 25k 20k 15k 10k 5k
```

Khối cuối chỉ có 1/10 dữ liệu nhưng logit vẫn được cộng **ngang hàng** với khối
đầu. Forgetting nhảy lên 51.29.

Và ngay cả đối chứng — 10 khối riêng nhưng mọi khối đủ dữ liệu — cũng mất 3.47
điểm, vì `G` khối chéo bỏ mất tương quan giữa các khối.

### Học phép chiếu một lần rồi đóng băng — không ăn

FSA cho +3.77 bằng đúng một ý: có một chỗ được học một lần rồi khoá vĩnh viễn.
Chỗ còn lại có tham số trong Fly-CL là phép chiếu thưa, và nó đang hoàn toàn ngẫu
nhiên. `models/flycl_lp.py` học một adapter trên feature ở task 0 rồi đóng băng:

```
A(x) = x + Up(Down(x)),  Up khoi tao = 0   →  A = DONG NHAT luc bat dau
Down: 2048 → r,  Up: r → 2048              →  cap nhat bi chan hang r
```

Hai ràng buộc đó sinh ra từ chẩn đoán của `--pos pre` (mất 43.8 điểm vì ridge với
đích `C` hàng cho ma trận hạng ≤ C). Adapter chỉ đụng vào **đúng lát stage 4** —
cho nó chạy trên cả vector 4 tầng thì nó bơm được thông tin stage 3 sang ô của
stage 4, và thứ đó là concat trá hình.

| cấu hình | drift | task 0 | A_T | Ā | Δ Ā |
|---|---:|---:|---:|---:|---:|
| mốc Fly-CL | 0 | 93.10 | 76.99 | 84.08 | — |
| ep=3 r=64 pres=0 | 0.154 | 92.90 | 76.69 | 84.02 | −0.06 |
| ep=10 r=64 pres=0 | 0.180 | 93.10 | 76.91 | 83.97 | −0.11 |
| ep=30 r=64 pres=0 | 0.190 | 93.40 | 76.88 | 84.02 | −0.06 |
| ep=10 r=64 pres=0.1 | 0.088 | 93.30 | 76.88 | 84.03 | −0.05 |
| ep=10 r=64 pres=1 | 0.036 | 93.40 | 76.83 | 84.01 | −0.07 |
| ep=10 r=64 pres=10 | 0.012 | 93.30 | 76.86 | 84.00 | −0.08 |
| ep=10 r=16 pres=0 | 0.171 | 92.80 | 76.86 | 84.06 | −0.02 |
| ep=10 r=256 pres=0 | 0.292 | 93.40 | 76.87 | 83.94 | −0.14 |

`drift` = ‖A(x) − x‖ / ‖x‖. Hai đối chứng loại trừ lỗi cài đặt: loss đi từ 2.56
xuống 0.0002 nên A khớp hoàn hảo task 0, và `pres` kéo drift xuống 15 lần đúng như
thiết kế. Cả hai đều chạy, kết quả vẫn không nhúc nhích — biên độ tám cấu hình là
0.12 điểm, nhỏ hơn σ giữa các seed. Xu hướng duy nhất: **càng đi xa càng tệ**, tối
ưu nằm ở chính phép đồng nhất.

Lý do có thể phát biểu chặt: `A = I + U Vᵀ` là ánh xạ **khả nghịch**, nên nó không
xoá thông tin nào, mà ridge lại bất biến với việc đánh số lại unit. Kênh duy nhất
A tác động được là đổi xem unit nào thắng top-k — đo được là ±0.1 điểm.

Đối chiếu giải thích luôn vì sao FSA ăn còn cái này không: FSA sửa **backbone**
bằng phi tuyến qua 23M tham số nên **thêm thông tin** vào feature; adapter nằm
**sau** backbone và tuyến tính khả nghịch nên chỉ **xoay lại thứ đã có**.

### Quy luật chung

Bốn can thiệp thua nặng nhất đều **tăng độ biểu đạt** hoặc **chia nhỏ dữ liệu**.
CIFAR-100 có 500 ảnh mỗi lớp; chia 8 vùng thì mỗi chuyên gia còn khoảng 62 ảnh
mỗi lớp trong khi vẫn phải phân biệt đủ 100 lớp. Bài toán bị chặn bởi **dữ
liệu**, không bởi độ biểu đạt.

Sau kết quả `flycl_lp`, quy luật siết được thêm một nấc và giờ khớp cả mười phép đo:

```
AN                                     nguyen nhan
FSA            +3.77    doi feature bang phi tuyen        → THEM thong tin
concat s3+s4   +1.52    lay them mot tang backbone        → THEM thong tin
gap doi E      +1.05    nhieu unit hon                    → THEM suc chua
ensemble m=5   +0.82    nhieu ban rut ngau nhien doc lap  → THEM suc chua

KHONG AN
tang CP        +0.18    tuyen tinh sau top-k              → sup thanh logit·K
hoc phep chieu −0.07    tuyen tinh KHA NGHICH truoc chieu → chi xoay thu da co
deep 2 tang    −1.49    chieu ngau nhien chong len nhau   → mat thong tin
top-k theo khoi  ≤ 0    sap xep lai cuoc canh tranh       → khong doi thong tin
MoE            −2.11    chia du lieu cho chuyen gia       → moi chuyen gia thay IT hon
stage 2        −0.58    them tang nhung QUA XA            → them nhieu
```

**Thêm thông tin hoặc thêm sức chứa thì ăn. Sắp xếp lại thứ đã có thì không.**

Điều này giải thích vì sao concat ăn được còn các cách khác thì không: nó
**không** chia dữ liệu, chỉ đưa thêm một nguồn thông tin đã có sẵn trên đường
forward của backbone.

---

## 6. Vấn đề phương pháp phát hiện trong tuần

**`coding_level` và `expand_dim` không độc lập.** Tăng `expand_dim` từ 10.000 lên
20.000 cho **+1.05** ở `coding_level` 0.1 nhưng **−1.38** ở 0.3. Ở `k` = 0.1 thì
`E` = 10.000 chỉ giữ 1.000 unit hoạt động, mã quá thưa nên thêm chiều còn ăn; ở
`k` = 0.3 giữ 3.000 unit, đã dùng nhiều hơn nên thêm chiều thành thừa. Paper quét
từng tham số một và ngầm coi chúng tách rời.

**GCV không đáng tin khi số unit hoạt động vượt số mẫu mỗi task.** Ở `E` = 20.000
với `k` = 0.3 thì có 6.000 unit hoạt động trong khi mỗi task chỉ 5.000 mẫu. Nới
lưới λ xuống làm baseline tụt 1.44 điểm — GCV được cho thêm lựa chọn thì chọn λ
nhỏ hơn, và λ đó tệ hơn trên test.

**Lưới λ do người dùng đặt đã ba lần tạo kết luận sai** trong dự án này. Lần đầu
làm ResNet thấp 7 điểm (kẹt sàn lưới), lần hai suýt làm bỏ đi một cải tiến thật
(nghi kẹt trần), lần ba làm baseline tụt 1.44. Cả ba lần đều không có dấu hiệu gì
trong log — λ nằm ở biên trông y hệt λ nằm trong lưới. `flycl_baseline.py` nên in
cờ cảnh báo khi λ được chọn rơi vào đầu hoặc cuối lưới.

**Cấp phát kết nối phải tường minh cho từng tầng.** Script cũ
`flycl_multistage.py` ghép cột rồi cho mỗi hàng rút 300 cột trên toàn bộ 3072,
khiến stage 4 bị hạ từ 300 xuống khoảng 200 kết nối. Con số đo được vì thế là
"lợi của stage 3 **trừ đi** hại do làm yếu stage 4". Sửa cách cấp phát làm kết
quả đi từ +0.13 lên +0.53.

---

### Đường cong GCV có hai cực tiểu — cơ chế chung của cả ba lần lưới λ gây chuyện

Ba sự cố về lưới λ ghi ở trên đều có **cùng một nguyên nhân**, tìm ra khi `concat`
trên feature FSA làm Cholesky sập. In toàn bộ đường cong GCV ở task gây lỗi:

```
feature FSA, task 3, n = 5000, rank = 4995

 log10 λ |   df/n   | residual |    GCV
    2    |  0.9983  |    0.0   | 7.45e-1
    3    |  0.9922  |    0.0   | 8.61e-2   ◀ cực tiểu GIẢ — GCV chọn cái này
    4    |  0.9383  |    2.1   | 1.09e-1
    5    |  0.6682  |   56.0   | 1.02e-1   ◀ cực tiểu THẬT
    6    |  0.2517  |  301.7   | 1.08e-1
    7    |  0.0537  |  682.5   | 1.52e-1
```

Cực tiểu bên trái là **giả**. Ở đó `residual = 0` (model nội suy hoàn hảo) và
`(1 − df/n)² = 6e−5`, nên GCV là `0/0` và con số in ra là nhiễu chia nhiễu. Cực tiểu
thật nằm ở `10⁵`, nơi còn sai số thật để đo.

Đường cong trên feature **gốc** cùng hình dạng, **lệch đúng một bậc**: cực tiểu giả ở
`10²`, cực tiểu thật ở `10⁴`. Hệ quả:

```
--ridge_lower 3, feature gốc :  10² NGOÀI lưới  → buộc lấy cực tiểu thật 10⁴  → chạy tốt
--ridge_lower 3, feature FSA :  10³ TRONG lưới  → vớ phải cực tiểu giả        → crash
```

Nên `--ridge_lower 3` đúng trên feature gốc **do may**, không do đúng — sàn tình cờ nằm
ngay trên cái hố giả. FSA làm `‖x‖` gấp 3.2 lần nên `trace(G)` gấp 11 lần, đẩy mọi thứ
sang một bậc và cái may hết. Cùng cơ chế giải thích hai sự cố cũ: hạ sàn làm lộ cực
tiểu giả (baseline tụt 1.44), còn nâng sàn lên 6 thì loại luôn cực tiểu **thật** (ResNet
mất 7 điểm). Hai đầu của một cửa sổ hẹp mà người dùng phải đoán trúng.

**Tiêu chí tách hai cực tiểu không phải λ mà là `df/n`:**

```
df/n → 1     model nội suy, GCV = 0/0, vô nghĩa    → loại
df/n < 0.9   còn sai số thật để đo                  → giữ
```

Thử tay trên cả hai tập feature thì tiêu chí này chọn đúng giá trị mà 9/10 task tự chọn
(`10⁵` cho FSA, `10⁴` cho gốc), **không cần biết thang feature**.

Nhưng **không đưa vào code**, vì đã thử sửa cách chọn λ và mọi hướng sửa đều tệ hơn. Chạy
GCV trên **dữ liệu tích luỹ** thay vì task hiện tại — đúng hơn về lý thuyết (xoá dạng
`0/0`, sửa việc λ thiếu 10× so với lượng dữ liệu) và nhanh hơn 10 lần (9s so với 99s vì
không cần SVD `5000×10000`) — nhưng **thua ở cả ba ô so được: −0.60 / −0.41 / −0.36**.

Kết luận: **GCV không phải tiêu chí tốt để chọn λ ở bài toán này**; sửa cho đúng công thức
lại tệ hơn, nghĩa là tiêu chí sai chứ không phải cách áp dụng sai. Chọn λ đúng phải đo trên
dữ liệu giữ riêng, chưa làm.

**Việc thực dụng phải làm:** `flycl_concat.py` giữ nguyên GCV per-task của Fly-CL, và trên
feature đã FSA thì **bắt buộc truyền `--ridge_lower 5`** — ở sàn 3 nó rơi vào cực tiểu giả
ở task 3 và Cholesky sập. Ghi trong docstring của file.

### λ đã ở đỉnh — không còn gì để giành

`select_ridge_parameter` được gọi với `H` của **task hiện tại** (n = 5.000) nhưng
λ ấy đem giải với `G` đã cộng dồn (n = 5.000·t). Nghiệm `W` cho trước λ thì đúng
bằng nghiệm joint, nhưng λ lại không được chọn như nghiệm joint — một chỗ hở trong
chính phát biểu của phương pháp. Và nó biểu hiện ra: GCV chọn `λ = 10⁴` ở **cả
mười task** trong khi trị riêng của `G` lớn lên mười lần.

Chữa được mà vẫn exemplar-free, bằng cách chiếu cả hai số hạng của GCV về `(G, Q)`:

```
Z = (G + λI)⁻¹ Q
‖(I−A)Y‖²_F = n − 2·tr(Qᵀ Z) + tr(Zᵀ G Z)        chinh xac, gan nhu mien phi
tr A(λ)     = E − λ · tr((G + λI)⁻¹)             Hutchinson, 64 vector Rademacher
```

`‖Y‖²_F = n` vì Y one-hot nên không cần đến Y. Kết quả: **hiệu ứng đúng như chẩn
đoán, nhưng làm tệ đi.**

| | λ theo task | λ tích luỹ |
|---|---|---|
| λ chọn ra | 1e4 ở cả 10 task | 1e4 → **1e5** từ task 1 |
| Ā | **84.08** | 83.48 |

Nhờ đã tăng tốc (mục 9) nên quét được λ cố định — thứ `logs/lam_oracle.txt` bắt đầu
đo rồi bỏ dở:

| λ | 1e2 | 1e3 | **1e4** | 1e5 | 1e6 | 1e7 |
|---|---:|---:|---:|---:|---:|---:|
| Ā | 81.51 | 82.97 | **84.08** | 83.43 | 80.64 | 75.82 |

**Đỉnh nằm đúng ở 1e4, tức đúng chỗ GCV-theo-task đang chọn.** Nội suy parabol cho
λ\* ≈ 1.35e4, ăn thêm 0.015 điểm — bằng không.

Vì sao: GCV tối thiểu hoá **sai số bình phương**, còn thứ đo là **độ chính xác của
argmax**. Dưới-siết làm nghiệm bám dữ liệu hơn nên sai số bình phương tệ đi, nhưng
thứ tự các logit vẫn đúng, mà argmax chỉ cần thứ tự. Cái "lệch" đang có lợi.

Đường cong quanh đỉnh dốc — mỗi bước 10 lần mất 0.65 đến 1.11 điểm — nên λ quan
trọng thật, chỉ là nó đã ở đúng chỗ. **Mục này đóng.**

---

---

## 7. Ghép tầng Contrastive Projection của AnaCP vào Fly-CL

**AnaCP** (Momeni, Xiao, Liu — NeurIPS 2025 Spotlight, arXiv:2511.13880,
[code](https://github.com/SalehMomeni/AnaCP)) đạt kết quả tốt nhất hiện nay trên
nhánh analytic CIL. Nó **không dùng contrastive loss** — thay vào đó thay đích
one-hot của ridge bằng "target prototype" sinh từ class mean đã được tách xa nhau
bằng whitening + SVD. Không loss, không gradient.

Vì Fly-CL cũng là ridge nghiệm đóng, hai pipeline chỉ khác nhau ở tầng này nên port
được. Toàn bộ mục này chạy trên **đúng giao thức Fly-CL**: `resnet50.tv2_in1k`,
`coding_level` 0.3, `expand_dim` 10.000, `ridge_lower` 4.

### Tầng CP gồm ba thứ

```
1. thống kê ở tầng input      μ_c (mean mỗi lớp),  Σ (covariance within-class)
2. sinh đích P                whiten → SVD → nới trị kỳ dị → de-whiten → chuẩn hoá
3. một phép ridge             W_CP = (Gram + λI)⁻¹ · (Q @ P)
```

Phần "contrastive" nằm toàn bộ ở bước 2. Bước 1 và 3 là hạ tầng.

Một quan sát làm phần ghép nhẹ đi nhiều: `Q` của Fly-CL **đã chính là** `M·N` mà
Eq. 9 của AnaCP cần, vì `Q[:,c] = ∑_{i∈c} h_i = n_c · m_c`. Nên `B = M N P` viết
bằng ngôn ngữ Fly-CL chỉ là `Q @ P` — một phép nhân ma trận, và bộ tích luỹ one-hot
giữ nguyên không sửa dòng nào.

### Đặt đúng chỗ AnaCP đặt thì nó sụp thành phép nhân logit

Với `P` đã chuẩn hoá (mọi prototype cùng độ dài), đọc bằng NCM tương đương lấy max
tích vô hướng, và khai triển ra:

```
Wo_cp = (G + λI)⁻¹ · Q · P  =  Wo_flycl · P
u     = hᵀ · Wo_cp          =  (hᵀ · Wo_flycl) · P  =  s · P
                                └──── logit Fly-CL ───┘

điểm mới của lớp c = u · p_c = ∑ s_j · (p_j · p_c)

⇒  s_mới  =  s_cũ · K        với  K = P·Pᵀ   cỡ (C, C)
```

Tức tầng CP, khi sau nó không còn phi tuyến nào, **chỉ là phép nhân vector logit của
Fly-CL với ma trận Gram của prototype**. `K = I` thì kết quả trùng Fly-CL từng chữ
số. Mà negative repulsion đẩy prototype về vuông góc, tức đẩy `K` về `I`, tức đẩy
phương pháp về chính Fly-CL.

Kiểm chứng thay vì tin: script chạy song song hai đường — NCM trên đầu ra CP, và
`logits · K` — rồi so từng mẫu. **Khớp 100.0% ở 16 lần đo độc lập** (2 mức
`expand_dim` × 7 mức α, 3 seed × 2 α, và các biến thể spread).

### Bảng — 3 seed, hiệu theo cặp

| seed | Fly-CL Ā | + CP (α=1) | Δ Ā | Fly-CL A_T | + CP | Δ A_T |
|---|---:|---:|---:|---:|---:|---:|
| 1993 | 84.08 | 84.26 | +0.18 | 76.99 | 77.13 | +0.14 |
| 2023 | 84.66 | 84.85 | +0.19 | 76.71 | 76.97 | +0.26 |
| 2025 | 83.84 | 83.93 | +0.09 | 76.57 | 76.80 | +0.23 |
| **TB** | **84.19 ± 0.42** | 84.35 | **+0.15 ± 0.06** | **76.76** | 76.97 | **+0.21 ± 0.06** |

Dòng mốc trùng khít bản tái lập ở Bảng 2 (84.19 ± 0.42 / 76.76), nên script mới đo
đúng thứ nó phải đo.

Hiệu ứng **có thật, dương ở cả 3 seed, không đổi dấu** — nhưng đặt cạnh những thứ đã
có thì nhỏ hơn tám lần:

| | Δ Ā | Chi phí thêm |
|---|---:|---|
| tầng CP | +0.15 | +1s, +17 MB |
| ensemble 2 nhánh | +0.49 | ×2 thời gian, ×2 bộ nhớ |
| ensemble 5 nhánh | +0.82 | ×5, ×5 |
| concat `[s3;s4]` | +1.37 | ×4 bộ nhớ |

### Negative repulsion — toàn bộ phần lý thuyết — đóng góp ≤ 0

Quét α trên seed 1993. `mean|cos|` là trung bình trị tuyệt đối phần ngoài đường chéo
của `K`, tức **đo trực tiếp độ lớn tác dụng**: bằng 0 nghĩa là `K = I` nghĩa là không
làm gì.

| α | A_T | Ā | mean\|cos\| | Δ Ā |
|---:|---:|---:|---:|---:|
| 0 | 77.22 | 84.26 | 0.0376 | +0.18 |
| 0.25 | 77.19 | 84.25 | 0.0366 | +0.17 |
| 0.5 | 77.16 | 84.26 | 0.0357 | +0.18 |
| **1** *(cấu hình của AnaCP)* | **77.13** | **84.26** | 0.0339 | **+0.18** |
| 2 | 77.16 | 84.23 | 0.0311 | +0.15 |
| 5 | 77.06 | 84.21 | 0.0255 | +0.13 |
| 10 | 76.91 | 84.07 | 0.0216 | −0.01 |
| **bỏ whitening** | 70.03 | 79.80 | **0.5797** | **−4.28** |

Đọc thành một câu: **`K` càng lệch khỏi `I` càng tệ.** Có một vùng phẳng quanh
`mean|cos|` ≈ 0.02–0.04 nơi Δ ≈ +0.15, rồi rơi thẳng. α = 10 cho Ā **84.07** — bằng
đúng mốc Fly-CL 84.08, y như đại số dự đoán.

Nên vai trò thật của whitening không phải "cải thiện" mà là **kiểm soát thiệt hại**:
nó ép `mean|cos|` từ 0.58 xuống 0.038, tức vô hiệu hoá phương pháp vừa đủ để nó
không phá.

### Lemma 4.1 rỗng trên dữ liệu này

Lemma 4.1 là phần lý thuyết duy nhất của bài: một bước gradient descent giải bằng
công thức đóng, chọn `δᵢ = −sign(Φᵢ)` để giảm tổng `|cos|` giữa các prototype. Nhưng
code họ phát hành (`models/anacp.py:86`) viết `spread_S = S + 1`, tức ép `δ = +1` cho
mọi lớp, không tính `Φ` bao giờ.

Cài đúng công thức rồi in dấu ra:

```
task |  C  | δ=+1 | δ=0 | δ=−1 | min|Φ| / max|Φ|
  0  |  10 |  10  |  0  |  0   | 2.18e-02
  …
  9  | 100 | 100  |  0  |  0   | 2.33e-01
```

**`δ = +1` ở 550/550 trường hợp**, và cột cuối cho thấy không phải trùng hợp số học —
mọi `Φᵢ` đều âm vững chắc. Chạy `--spread paper` cho kết quả **trùng khít từng chữ
số** với `--spread repo`. Lý thuyết và code khớp nhau, nhưng lý thuyết không bao giờ
thay đổi câu trả lời.

Ghi thêm: phát biểu Lemma trong paper viết `(2·1[·≥0])` mà thiếu `−1`; Appendix A
viết đúng. Thiếu `−1` thì hàm nhận giá trị `{0,2}` thay vì `{−1,+1}`.

### Bản AnaCP đầy đủ — cấu hình duy nhất không sụp — lại là cấu hình tệ nhất

Chỉ khi có tầng ridge **thứ hai** với phi tuyến xen giữa thì phép rút gọn mới không
áp dụng được. Cái giá là đầu ra tầng CP đổi mỗi task nên `G₂`, `Q₂` không tích luỹ
được, phải dựng lại từ đầu bằng **pseudo-replay** `x̃ ~ 𝒩(μ_c, Σ)`, 100 mẫu/lớp.

| Cấu hình | A_T | Ā | Forgetting | Thời gian | Δ Ā |
|---|---:|---:|---:|---:|---:|
| Fly-CL | 76.99 | 84.08 | 8.54 | 97s | — |
| CP + NCM (không tầng 2) | 77.13 | 84.26 | 8.54 | 99s | +0.18 |
| đầy đủ, H=1, phi tuyến `top-k` | 76.61 | 83.98 | 8.88 | 340s | **−0.10** |
| đầy đủ, H=1, phi tuyến `GELU` | 76.39 | 83.84 | 8.97 | 359s | **−0.24** |
| đầy đủ, H=3, phi tuyến `top-k` | 76.96 | 84.58 | 8.98 | 533s | +0.50 |

**Tầng thứ hai làm tệ đi.** Với H=1 nó kéo 84.26 xuống 83.98, tức **dưới cả Fly-CL
gốc**. Phi tuyến có thật nên không sụp nữa, nhưng cái nó mua được không bù nổi cái nó
bán đi: dữ liệu thật đổi lấy mẫu Gauss. Forgetting cũng tăng 8.54 → 8.88, đúng dấu
hiệu mất tính chính xác của nghiệm đóng.

**`GELU` thua `top-k`** (83.84 vs 83.98). Ngay trong khung của AnaCP, phi tuyến của
Fly-CL vẫn tốt hơn phi tuyến của họ.

**Dòng H=3 là công của ensemble, không phải của CP.** Đối chiếu Bảng 3, cùng seed
cùng giao thức, ensemble trần cho 84.08 / 84.57 / 84.75 ở m = 1/2/5. Ba head + tầng
CP + tầng hai + pseudo-replay cho 84.58, tức **bằng đúng ensemble trần 2 nhánh** mà
tốn 533s so với ~196s.

Một điểm không như dự đoán ban đầu: bản đầy đủ **vẫn giữ tính bất biến theo thứ tự
task**, vì `μ_c`, `Σ`, `Gₕ`, `Qₕ` đều là tổng theo mẫu và `G₂`/`Q₂` dựng lại từ replay
của mọi lớp đã gặp. Cái mất là tính **chính xác**, không phải tính bất biến.

### Đặt CP trước phép chiếu: hỏng vì hạng

Ý tưởng đặt CP ở tầng input để phi tuyến `top-k` nằm **sau** nó — về nguyên tắc tránh
được phép sụp. Thực tế mất 43.8 điểm.

```
E = 10.000:   A_T 22.02 | Ā 40.26     so với mốc 76.99 / 84.08
E =  2.000:   A_T 15.81 | Ā 32.37
```

Nguyên nhân đo được, không phải đoán. Phân tích trị kỳ dị của `W_CP` (2048×2048):

```
s[0..8]  =  4.6e-1 … 3.1e-1        9 trị kỳ dị đáng kể
s[9]     =  1.8e-2                 nhỏ hơn 25 lần
s[10]    =  2.4e-6                 = 0
năng lượng trong 10 chiều đầu:  1.000000
```

**Hạng thực tế bằng 9.** Hệ quả cấu trúc: ridge với đích chỉ có `C` hàng luôn cho ma
trận hạng ≤ `C`, mà task 0 chỉ có 10 lớp. Nó ép 2048 chiều xuống 9 **trước** khi mở
rộng, và phép chiếu thưa không khôi phục được thứ đã bị xoá. Tăng `expand_dim` chỉ là
chiếu 9 chiều lên nhiều unit hơn.

Ở vị trí sau top-k thì điều này vô hại — đằng nào đầu ra cũng chỉ cần `C` chiều để so
với `C` prototype.

### Chạy code gốc của họ — hai phép kiểm

Mọi kết luận âm ở trên chỉ đứng được nếu bản cài lại của ta đúng. Hai phép kiểm độc lập.

**Kiểm 1 — tái lập số công bố.** Chạy nguyên `upstream/AnaCP/run.py` theo đúng `scripts/anacp.sh`:
DINOv2-base + FSA(aper, 10 epoch) + AnaCP đầy đủ, CIFAR-100, 10 task, seed 0.

| | A_avg | A_last |
|---|---:|---:|
| Paper (3 seed) | 95.43 | 92.15 |
| **Chạy lại (seed 0)** | **95.38** | **92.11** |
| lệch | 0.05 | 0.04 |

Tái lập gần như hoàn hảo. Ma trận accuracy cũng lành: task 0 đi từ 0.992 xuống 0.945 sau
10 task. **Phương pháp của họ không có vấn đề gì.**

(Chạy được với `transformers 5.15.1` dù `requirements.txt` ghi 4.44.1. Đã kiểm riêng
`AutoImageProcessor`: nó trả `pixel_values` là **list chứa một tensor** `(B,3,224,224)`,
nên `inputs['pixel_values'][0]` ở `backbone.py:147` vẫn lấy đúng cả batch.)

**Kiểm 2 — code của họ trên feature của ta.** README của họ ghi `models/anacp.py` là
*"standalone module that can be used to incrementally train a model on input features"*,
nên đưa thẳng feature ResNet-50 đã cache vào, **không sửa một dòng nào**, đúng siêu tham
số của họ (`D=5000, reg=1e2, heads=3, R=100, shared_cov`), cùng class order cùng split.

| Feature | | A_T | Ā | Forgetting | Thời gian |
|---|---|---:|---:|---:|---:|
| ResNet-50 gốc | Fly-CL | 76.99 | 84.08 | 8.54 | 97s |
| ResNet-50 gốc | **AnaCP (code họ)** | 76.88 | **84.31** | 9.01 | **25s** |
| ResNet-50 + FSA | Fly-CL | 80.93 | **87.85** | 7.74 | 97s |
| ResNet-50 + FSA | **AnaCP (code họ)** | 80.84 | 87.80 | 8.12 | **22s** |

Hai kết luận:

**Bản port ở các mục trên không sai.** `anacp_cp.py --pos post` cho +0.15 ± 0.06 (3 seed);
code gốc của họ cho +0.23 (1 seed). Cùng một chỗ.

**Trên feature ResNet-50, hai phương pháp tương đương.** Toàn bộ bộ máy của họ — RP dense
5000 chiều + GELU + tầng CP + 3 head + RP thứ hai + ELM + pseudo-replay — chạy trong 25
giây và về đúng chỗ một phép ridge duy nhất của Fly-CL đã đứng.

### Bảng 2×2: bản cài lại so với code gốc, có FSA và không

Gộp mọi phép đo lại theo hai trục — ai cài, và feature nào. Một seed (1993), cùng
class order. Cơ chế của việc đổi dấu sau FSA nằm ở mục 8, không nhắc lại ở đây.

| Cấu hình | A_T | Ā | Δ Ā |
|---|---:|---:|---:|
| **ResNet-50 trần** | | | |
| Fly-CL (mốc) | 76.99 | 84.08 | — |
| Fly-CL adapt CP layer | 77.13 | 84.26 | +0.18 |
| Fly-CL adapt full (3 head) | 76.96 | 84.58 | +0.50 |
| AnaCP (code gốc) | 76.88 | 84.31 | +0.23 |
| **ResNet-50 + FSA** | | | |
| Fly-CL (mốc) | 80.94 | 87.85 | — |
| Fly-CL adapt CP layer | 78.94 | 86.79 | **−1.06** |
| Fly-CL adapt full (3 head) | 80.57 | 87.99 | +0.14 |
| AnaCP (code gốc) | 80.84 | 87.80 | −0.05 |

Ba tên đầu dùng chung backbone Fly-CL nên so trực tiếp được với nhau; dòng cuối đổi
cả backbone (dense 5000 + GELU + 3 head của họ) nên chỉ dùng để kiểm tra bản cài lại
không sai.

**Số head là biến quan trọng, đừng đọc "adapt full" như một dòng duy nhất.** Bảng ở
mục 8 dùng 1 head và cho 83.98 / 87.41; bảng này dùng 3 head và cho 84.58 / 87.99.
Chênh 0.6 điểm ở cả hai tập feature, nhất quán.

### Khoảng cách nằm ở backbone, không ở phương pháp

Ghép ba điểm đo lại:

```
                        Fly-CL      AnaCP
ResNet-50 gốc            84.08      84.31
ResNet-50 + FSA          87.85      87.80
DINOv2 + FSA            (chưa đo)   95.38
```

Hai phương pháp khác hẳn nhau về cơ chế nhưng về **cùng một chỗ** khi cho cùng feature;
còn đổi backbone thì nhảy 11 điểm. Khớp đúng câu kết của chính họ ở §5.6: *"a strong PTM
is the key to CL, and representation-level forgetting is not the primary limitation"* —
và số liệu ở đây nói mạnh hơn: **PTM không chỉ là chìa khoá, nó gần như là tất cả.**

Ô còn thiếu là ô quyết định: **Fly-CL trên chính feature DINOv2 + FSA**. Nếu nó cũng ra
~95 thì tầng CP không đóng góp gì ở **bất kỳ** backbone nào, và phát biểu đó có bằng chứng
ba điểm. Cần một script trích feature DINOv2+FSA ra cache của ta rồi chạy `anacp_cp.py`.

### Kết luận mục này

Không có cấu hình nào của AnaCP vượt được thứ Fly-CL đã có, và cấu hình duy nhất **về
nguyên tắc** có thể khác biệt lại là cấu hình tệ nhất. Phát biểu hợp lệ:

> Tầng Contrastive Projection của AnaCP, khi ghép vào Fly-CL, rút gọn về một phép
> nhân logit với ma trận Gram của prototype, và cho +0.15 ± 0.06 Ā — nhất quán nhưng
> nhỏ hơn tám lần so với tương tác hai tầng backbone.

Kết quả này khớp đúng nguyên tắc đã lập ở `bao-cao.md`: phần closed-form của Fly-CL
đã tối ưu tuyệt đối trên feature cho trước, nên muốn vượt thì phải cải thiện chính
feature. Tầng CP không đụng vào feature.

---

## 8. First-Session Adaptation — thứ duy nhất của AnaCP thực sự ăn

AnaCP quảng cáo "no gradient-based training", nhưng Implementation Details ghi: *"We
also adopt FSA following [56] before applying AnaCP"*. Trong code đó là
`--training_method aper`: gắn Adapter down-up vào mỗi block, đóng băng backbone, 10
epoch cross-entropy trên **task 0**, rồi đóng băng vĩnh viễn. Nó chiếm **65% compute**
của họ (6m05s / 9m18s) và **không được ablate**. Fly-CL cố tình bỏ FSA.

### Cách cài — `fsa_extract.py`

Chỉnh backbone bằng gradient **đúng một lần ở task 0** rồi đóng băng vĩnh viễn. Đóng băng
là bắt buộc chứ không phải lựa chọn: `G_t = G_{t−1} + H_tᵀH_t` chỉ hợp lệ khi mọi `H` cùng
hệ toạ độ, sai thì sai **im lặng**.

Adapter là **nhánh cộng thêm**, trọng số gốc không đổi một byte:

```
out = block(x)                                   ← ResNet gốc, đóng băng
out = out + adapter(out)                         ← chỉ nhánh này học
adapter: Conv2d(C,8,1) → ReLU → Conv2d(8,C,1)    `up` khởi tạo 0 ⇒ xuất phát = baseline

16 khối Bottleneck | backbone 23.51 M đóng băng | adapter 241.7 K học (1.03 %)
```

Luồng: chia task theo seed → gắn adapter → classifier tạm `Linear(2048,10)` → Adam 10
epoch trên 5.000 ảnh task 0 → vứt classifier, đóng băng → trích lại **toàn bộ** 60.000 ảnh
ra hai cache (`+fsa{seed}` 2048 chiều và `+fsa{seed}+ms` 3840 chiều cho feature mix) → mọi
script khác chạy y như cũ.

**Siêu tham số** lấy nguyên nhánh DINOv2 của `scripts/anacp.sh`: `rank 8`, `10 epoch`,
`lr 1e-3` (classifier) / `1e-4` (adapter). Hai lưu ý: `run.py` của họ mặc định 20 epoch
trong khi script dùng 10; và chính họ đổi lr theo backbone (1e-2 cho MoCo-v3), tức nó
không chuyển được — mà ta mượn bộ DINOv2 cho backbone thứ ba. `lr` và `rank` chưa quét.

**Hai cạm bẫy.** Tag cache phải chứa **seed**, vì FSA phụ thuộc 10 lớp nào rơi vào task 0 —
dùng chung cache là rò rỉ giữa các seed. Và BatchNorm phải ở `eval()`: `running_mean`/
`running_var` là **buffer chứ không phải parameter** nên `requires_grad=False` không chặn
được, đây là đường duy nhất backbone đổi mà không cần gradient. AnaCP gọi `model.train()`
nên BN của họ có trôi; ở đây mặc định `--bn_mode eval`, có cờ để tái lập cách của họ.

**Chi phí một lần**: ~400s huấn luyện + ~90s trích feature. Sau đó mỗi run vẫn 97s — FSA
không làm chậm phần continual learning. Để so: FSA chiếm **65% compute** của AnaCP và họ
**không ablate** nó.

### Kết quả

Số của FSA đã gộp vào **Bảng kết quả chính (mục 2)**, khối dưới. Tóm tắt: FSA một mình cho
**+3.77 Ā**, và cấu hình tốt nhất của cả dự án là `FSA + concat + ensemble 5` với **Ā 89.05
/ A_T 82.72 / Forgetting 7.12**.

Phần dưới đây chỉ ghi những gì **không** thuộc bảng chính: tương tác giữa FSA và tầng CP
của AnaCP.

| Cấu hình | A_T | Ā | Forgetting | Δ Ā |
|---|---:|---:|---:|---:|
| Fly-CL + FSA | 80.93 | **87.85** | 7.74 | — |
| + CP 2 tầng (H=3) | 80.57 | **87.99** | 8.60 | **+0.14** |
| + CP 2 tầng (H=1) | 79.51 | 87.41 | 8.91 | **−0.44** |
| + CP 1 tầng | 78.94 | 86.79 | 8.98 | **−1.06** |

Chỉ cấu hình **đầy đủ nhất** mới vượt được mốc, và vượt 0.14 — kém concat gần mười
lần. Bỏ tầng thứ hai đi thì mất 1.06.

### CP đảo dấu sau FSA — và đúng theo quy luật đã lập

| | mean\|cos\| | Δ Ā |
|---|---:|---:|
| CP trên feature gốc | 0.0339 | **+0.18** |
| CP trên feature đã FSA | **0.4839** | **−1.06** |

`mean|cos|` nhảy từ 0.034 lên 0.48, tức `K` xa `I` hơn 14 lần — và theo đúng đường
cong ở mục 7, `K` xa `I` thì tệ đi. Đây là xác nhận độc lập của cơ chế trên một tập
feature hoàn toàn khác.

Cơ chế đo được, không phải suy đoán. Whitening vẫn làm đúng việc của nó sau FSA —
`mean|cos|` của class mean **ngay sau bước whiten** gần như không đổi (0.0592 →
0.0601). Chỗ hỏng nằm ở bước **de-whiten**, nơi code AnaCP nhân `Σ^(−1/2)` lần nữa
thay vì `Σ^(+1/2)`:

```
                        cond(Σ)    trace(Σ)   mean|cos| sau whiten   mean|cos| của P
feature gốc             8.6e3      1.04e2           0.0592               0.0331
feature sau FSA         8.9e4      9.80e2           0.0601               0.4859
```

FSA làm `cond(Σ)` xấu đi 10 lần và độ lớn feature tăng 10 lần. Trên feature gốc, phép
nhân `Σ^(−1/2)` lần hai tình cờ kéo `K` về gần `I` (0.033) nên tầng CP vô hại; sau FSA
nó không kéo về được nữa (0.486) nên tầng CP làm hỏng logit.

Nói cách khác: **cái làm tầng CP không gây hại trên feature gốc lại chính là chỗ code
lệch khỏi Eq. 16 của paper.** Với `Σ^(+1/2)` đúng công thức thì `P` bám sát class mean
thô và `mean|cos|` là 0.55–0.63 ở cả hai tập feature — tức luôn nằm ở vùng có hại.

### Tầng thứ hai là cơ chế sửa chữa, không phải cơ chế cải thiện

Chạy bản AnaCP đầy đủ (2 tầng ridge + pseudo-replay) trên feature FSA cho một quan sát
sạch — thứ tự hai bản **lật ngược** so với feature gốc:

| | 1 tầng (CP + NCM) | 2 tầng (CP + ELM + replay) | chênh |
|---|---:|---:|---:|
| feature gốc (`mean\|cos\|` = 0.034) | 84.26 | 83.98 | 2 tầng **thua** 0.28 |
| feature FSA (`mean\|cos\|` = 0.484) | 86.79 | 87.41 | 2 tầng **hơn** 0.62 |

Khi `K ≈ I` thì tầng CP vốn đã vô hại, nên tầng hai chỉ thêm sai số của xấp xỉ Gauss.
Khi `K` xấu thì tầng hai có phi tuyến nên gỡ lại được 0.62 trong 1.06 đã mất — nhưng
không gỡ hết. Với **1 head** thì mọi biến thể AnaCP đều thua Fly-CL trần, tốt nhất là
−0.44. Nâng lên **3 head** thì bản hai tầng đạt 87.99, tức **+0.14** so với mốc 87.85 —
vừa đủ vượt, và vẫn kém concat gần mười lần. Nên phát biểu đúng là: trên feature FSA,
tầng CP một mình gây hại 1.06 điểm, và toàn bộ phần còn lại của bộ máy AnaCP chỉ vừa
đủ để gỡ lại chỗ nó vừa làm hỏng.

### Ý nghĩa

AnaCP báo cáo con số tốt nhất trên nhánh analytic CIL. Đo tách bạch trong khung Fly-CL
thì phần lớn giá trị nằm ở **FSA** — thứ họ để ở Implementation Details, không ablate,
và bản thân nó là gradient training — chứ không ở tầng CP là đóng góp phương pháp luận
được trình bày ở §4.

Với hướng của mình, đây là kết quả **dương** và là hướng đáng theo nhất còn lại: FSA +
concat + ensemble cho **+4.97 Ā** trên đúng giao thức Fly-CL, với 501s một lần.

### 10 epoch là quá đủ — bão hoà từ epoch 3

AnaCP đặt `--num_epochs 10` trong `scripts/anacp.sh` (còn `run.py` để mặc định 20 — lại
một chỗ script khác mặc định). Họ không ablate. Quét thử trên ResNet-50:

| epochs | train acc task 0 | A_T | Ā | Δ Ā tích luỹ |
|---:|---:|---:|---:|---:|
| 0 *(không FSA)* | — | 76.99 | 84.08 | — |
| 1 | 77.94 | 79.52 | 86.37 | +2.29 |
| **3** | **95.88** | 80.81 | **87.74** | **+3.66** |
| 10 | — | 80.93 | 87.85 | +3.77 |

**Một epoch đã lấy 61% lợi ích, ba epoch lấy 97%.** Từ 3 lên 10 chỉ thêm +0.11, nằm
trong nhiễu. Đường cong huấn luyện khớp: `77.94 → 92.48 → 95.88`, classifier gần bão hoà
ngay ở epoch 3.

Hai hệ quả:

**Con số +3.77 không phải cận dưới.** 10 epoch nằm an toàn sau điểm bão hoà, nên không
cần đo lại khối FSA ở mức epoch cao hơn.

**Chi phí FSA thấp hơn báo cáo ở trên.** Huấn luyện adapter chỉ mất ~120s (3 epoch) hoặc
~400s (10 epoch); phần còn lại của con số 501s là trích lại feature cho 60.000 ảnh. Ở
mức 3 epoch thì tổng chi phí một lần rơi về ~230s.

Chưa quét mức 20 và 40 — dừng vì đường cong đã phẳng. Câu còn để ngỏ là liệu huấn luyện
lâu hơn có **làm tệ đi** không (adapter overfit 10 lớp của task 0), tức đánh đổi
discriminability ↔ transferability. Chưa đo.

### Giới hạn của nhánh này

- **Một seed.** Cần 2023 và 2025 trước khi đưa vào bảng chính.
- **Đổi giao thức, không phải "vượt Fly-CL".** FSA dùng nhãn của task 0 để sửa
  backbone. Đây là thiết lập hợp lệ và phổ biến (APER, RanPAC, AnaCP đều dùng), nhưng
  Fly-CL cố tình từ chối nó để giữ tuyên bố "không huấn luyện gì". So sánh
  `Fly-CL + FSA` với `Fly-CL` là so hai giao thức khác nhau, phải nói rõ.
- **Siêu tham số adapter mới quét một trục**: số epoch đã quét và bão hoà từ epoch 3;
  `rank` 8 và lr 1e-3/1e-4 vẫn lấy nguyên cấu hình DINOv2 của AnaCP, chưa chỉnh cho
  ResNet-50. Chính AnaCP cũng đổi lr theo backbone (1e-3 cho DINOv2, 1e-2 cho MoCo-v3),
  tức thừa nhận nó không chuyển được — mà ta đang mượn bộ của DINOv2 cho backbone thứ ba.
- **Chưa ablate vị trí adapter.** Gắn sau mọi Bottleneck; chưa thử chỉ `layer4`, chỉ
  BN, hay LoRA.

## 9. Tăng tốc 10.8 lần, số liệu không đổi một chữ số

Bổ thời gian một task trên GPU, dữ liệu thật, `E` = 10.000:

```
buoc                        giay      %      GFLOP   TFLOP/s
ma hoa train  topk(W Xᵀ)   0.042    0.4%     204.8     4.93
G += H Hᵀ                  0.137    1.4%    1000.0     7.30
Q += H Y                   0.002    0.0%      10.0     5.61
GCV  svd(Hᵀ)               9.208   97.4%     500.0     0.05   ←
cholesky(G+λI)             0.049    0.5%     333.3     6.79
cholesky_solve             0.010    0.1%      20.0     1.99
```

**97.4% thời gian dùng để chọn một số vô hướng từ lưới 11 giá trị.** Toàn bộ thuật
toán thật — mã hoá, Gram, Cholesky — hết 0.24 giây. Cột cuối chỉ thẳng nguyên nhân:
SVD chạy 0.05 TFLOP/s trong khi matmul chạy 7.3 trên cùng GPU, và nó làm ít phép
tính hơn Gram mà lâu hơn 67 lần.

### Nút thắt 1 — SVD dựng một ma trận rồi vứt

```python
U, S, _ = torch.linalg.svd(X, full_matrices=False)     # X co [5000, 10000]
```

`_` là ma trận vector kỳ dị phải, cỡ 5000×10000 — 200 MB tính ra rồi vứt ở ngay dấu
gạch dưới. Cờ `full_matrices=False` không cứu được: nó chỉ chọn 5000 hay 10000 vector,
không có lựa chọn "không vector nào", và PyTorch không có API trả `U, S` mà bỏ `V`.

Mà GCV chỉ cần `S²` và `w_j = ‖u_jᵀY‖²`, cả hai lấy được từ ma trận Gram **nhỏ hơn
trong hai chiều**:

```
n ≤ E:   eigh(X Xᵀ)   [n, n]     tri rieng = S²,  vector rieng = U
n > E:   eigh(Xᵀ X)   [E, E]     u_jᵀY = v_jᵀ(XᵀY)/s_j,   XᵀY chi la [E, C]

‖Y − Ŷ‖²  =  Σ_j (1 − d_j)²·w_j  +  (‖Y‖² − Σ_j w_j)
                                     └─ bang 0 khi n ≤ E ─┘
```

Số hạng cuối là phần của `Y` nằm ngoài span(U); bản gốc không cần vì `n ≤ E` luôn
đúng ở đây, nhưng viết vào thì hàm dùng được cho cả `E` nhỏ.

Điểm mấu chốt về chi phí: cạnh của `X Xᵀ` là **số mẫu một task**, không phải `E`. Nên
concat (`E` = 20.000) vẫn chéo hoá ma trận 5000×5000 y hệt — **chọn λ trở thành độc
lập với `E`**, trong khi SVD trước đây tỉ lệ với `E`.

### Nút thắt 2 — ma trận đơn vị, chỉ lộ ra ở cấu hình lớn

```python
torch.linalg.cholesky(G[e] + ridge * self.eye)
```

Bắt **ba** ma trận `Eb × Eb` cùng tồn tại: `eye` thường trú, bản tạm của phép cộng,
và đầu ra Cholesky. Ở concat + ensemble 5 thì `Eb` = 20.000 nên mỗi cái là 1.6 GB,
trên nền `G` đã chiếm 8 GB của card 16 GB — thời gian đi vào cấp phát chứ không vào
tính toán. Riêng `eye` là 1.6 GB để chứa hai vạn số 1 và bốn trăm triệu số 0.

Cộng thẳng vào đường chéo của bản sao thì chỉ còn hai, và bỏ hẳn được `eye`. Không
dùng cách sửa `G` tại chỗ rồi trừ lại: cộng rồi trừ `1e4` trên nền `G_ii` cỡ `1e7`
không khử nhau chính xác trong fp32, sai số sẽ tích luỹ qua mười task.

### Kết quả

| Cấu hình | ban đầu | sau `eigh` | sau bỏ `eye` | nhanh hơn |
|---|---:|---:|---:|---:|
| Fly-CL | 97s | 9s | 9s | **10.8×** |
| concat | 105s | 14s | 14s | **7.5×** |
| concat + ensemble 5 | 490s | 372s | **102s** | **4.8×** |

Hai nút thắt đổi vai theo kích thước: `E` nhỏ thì nghẽn ở **phép tính** (SVD), `E`
lớn thì nghẽn ở **bộ nhớ** (ba ma trận 1.6 GB). Phải sửa cả hai mới nhanh ở mọi cỡ.

Bốn phép kiểm, không lệch một chữ số:

```
flycl            76.99 / 84.08 / 8.54
concat           79.01 / 85.60 / 7.57
concat + ens 5   79.77 / 86.41 / 7.71
anacp_cp post    77.13 / 84.26          agree van 100.0%
anacp_full none  77.13 / 84.26          trung anacp_cp post (phep tu kiem cua repo)
```

### Đây không phải contribution

Thuật toán **không đổi một dòng nào** — cùng tiêu chí, cùng lưới, cùng kết quả. Nên
chỉ được viết "bản cài đặt của chúng tôi nhanh hơn", không được viết "phương pháp của
chúng tôi nhanh hơn". Cả hai đường đều là `O(n²E)`; lợi 16 lần là **hằng số**, đến từ
chỗ không dựng `V` và chỗ thuật toán đối xứng rẻ hơn SVD trên mỗi phép tính.

Chỗ nó được phép xuất hiện: bảng thời gian trong phần thực nghiệm (nhánh analytic CL
bán mình bằng tốc độ, nên wall-clock là số liệu hợp lệ), lý do cho quy mô ablation, và
một Remark ngắn trong Implementation details.

Hệ quả thực tế thì đáng kể: ba seed của cấu hình đắt nhất từ 25 phút xuống 5 phút, nên
những phép quét trước đây phải cân nhắc thì giờ chạy thẳng.

---

## 10. Công trình liên quan phải đọc trước khi viết

Tìm được trong tuần, hai bài chạm trực tiếp vào hai ý chính của mình.

### LayUP — "Read Between the Layers" (arXiv 2312.08888)

Nối `k` tầng cuối của ViT rồi đưa thẳng vào ridge với ma trận Gram. Báo cáo +1.2% đến
+5.7% ở CIL. Đây là **prior art cho ý "dùng nhiều hơn tầng cuối"**, nên concat không
còn được trình bày như một ý tưởng mới.

```
LayUP     noi FEATURE roi ridge           KHONG co phep chieu ngau nhien
          ViT-B/16, k = 6 tang cuoi       cac tang cung chieu, ke nhau
          k = 6 va k = 12 lech 0.4%       → cang nhieu tang cang tot

cua ta    moi tang co PHEP CHIEU RIENG    tuong tac xay ra trong top-k
          ResNet-50, stage 3 + stage 4    khac chieu, phai chuan hoa thang
          stage 2 hai o ca 3 phep do      → chi tang LIEN KE moi an
```

Hai kết luận **ngược nhau** về việc thêm tầng xa. Giả thuyết: các block kề nhau của
ViT rất giống nhau, còn ResNet đổi cả độ phân giải lẫn ngữ nghĩa qua mỗi stage. Nếu
đúng thì phát hiện của mình là **tính chất của kiến trúc phân cấp**, không phải của
phương pháp — phát biểu sắc hơn, nhưng **bắt buộc phải đo concat trên ViT** mới nói
được. Việc này chuyển từ "nên có" sang **ưu tiên 1**.

Cần đọc kỹ bản đầy đủ, không đọc tóm tắt, trước khi viết related work.

### SCL-MGSM — "Guided Random Projection" (arXiv 2603.19145, 03/2026)

Chọn lọc cơ sở ngẫu nhiên ở task 0 theo tiêu chí căn với đích rồi đóng băng. Đây đúng
là ô "học phép chiếu một lần rồi đóng băng" mà `flycl_lp` vừa thử. Họ báo cáo ăn, mình
đo ra âm, và **không mâu thuẫn** — hai cơ chế khác nhau đúng theo quy luật ở mục 5:

```
ho    CHON co so nao duoc giu     doi do phu cua khong gian ma  → them suc chua
ta    hoc mot anh xa tuyen tinh   KHA NGHICH, khong doi gi ca   → tai tham so hoa
```

Kết quả âm của mình vẫn đứng, nhưng giờ nó là đối chứng cho bài của họ chứ không còn
là ô trống. Họ cũng so với AnaCP, tức nhánh này đang chuyển động nhanh.

### Về tốc độ: cửa đóng

`ACIL`, `DS-AL`, `G-ACIL` đã có câu chuyện tốc độ của nhánh này — duy trì ma trận
tương quan nghịch đảo và cập nhật đệ quy bằng Woodbury thay vì phân rã lại. Nhưng RLS
**chỉ chạy được khi λ cố định**, mà Fly-CL tính lại λ mỗi task.

Hutchinson thì đúng nghĩa đen là sách giáo khoa: ước lượng vết bằng vector ±1 được đề
ra năm 1989 **chính là để tính vết cho GCV**.

Và chỗ đóng cửa hẳn: **cả nhánh này né vấn đề bằng cách cố định λ**. SCL-MGSM quét
lưới một lần trên task 0 rồi khoá `λ = 0.01`; AnaCP khoá `λ = 1e2`. Chỉ Fly-CL trả giá
GCV mỗi task. Nên "chọn siêu tham số tốn hơn cả mô hình" là đặc thù của một bản cài
đặt, không phải vấn đề của lĩnh vực.

Hệ quả thực tế: vì mục 6 đã chứng minh `λ = 1e4` là đỉnh ở mọi task, **cố định λ và bỏ
GCV hoàn toàn** là hợp lệ, cho kết quả trùng khít, và đưa 9 giây xuống khoảng 4.

---

## 11. Giới hạn

**Bản tái lập thấp hơn số công bố 0.42** (84.19 ± 0.42 so với 84.61 ± 0.16, 3 seed).
Khoảng lệch này phụ thuộc số seed và chưa giải thích được: 0.53 với 1 seed (84.08),
0.42 với 3 seed (84.19), 0.80 với 5 seed (83.81 ± 0.85, xem `bao-cao.md`).
Phần chênh giữa concat (85.56) và số công bố (84.61) là **0.95** — cùng bậc độ lớn
với khoảng lệch tái lập. Nên phát biểu hợp lệ là *"+1.37 so với baseline chạy cùng
code, cùng seed, cùng projection"* — **không** phải "vượt Fly-CL". Riêng dòng
`concat + ensemble 5` (86.49 ± 0.40) thì chênh 1.88 so với số công bố, tức lớn hơn
khoảng lệch tái lập ở mọi mức seed; đó là dòng duy nhất có thể phát biểu mạnh hơn,
và vẫn kèm điều kiện đọc thêm stage 3 nêu ngay dưới.

**Phải đọc thêm stage 3 của backbone.** Trọng số backbone không đổi, không train
gì thêm, không tốn thêm phép tính nào trong backbone — bản đồ stage 3 đã được
tính sẵn trên đường forward. Nhưng đây vẫn là **đầu vào giàu hơn** so với Fly-CL.
Phép so cho cả hai bên cùng đầu vào và cùng số unit nằm ở dòng `expand_dim`
20.000 trong Bảng chính: **+2.75**.

**Chỉ ResNet-50 + CIFAR-100.** Chưa thử ViT-B/16 — kiến trúc này không có khái
niệm "tầng trung gian" theo kiểu ResNet, nên cách áp dụng chưa rõ. Chưa thử
CUB-200 hay VTAB.

**Số seed không đều.** Bảng chính có 3 seed cho khối không-FSA, 1 seed cho khối FSA.
Bảng chọn tầng và bảng ensemble mới một seed.

---

## 12. Đang chạy và còn mở

| | Trạng thái |
|---|---|
| **Fly-CL trên feature DINOv2 + FSA** | **ưu tiên 1** — ô cuối để chốt "khoảng cách là do backbone" |
| **FSA trên seed 2023 và 2025** | ưu tiên 2 — kết quả +3.77 mới có một seed |
| **FSA + concat, FSA + ensemble** | chưa — kiểm xem FSA có chồng lấn với cả hai không |
| Quét siêu tham số adapter (rank, lr, epoch, vị trí) | chưa — hiện lấy nguyên cấu hình DINOv2 của AnaCP |
| concat + ensemble, 3 seed × {2, 5} nhánh | đang chạy |
| Quét `expand_dim` trên ViT + CIFAR-100 | chưa — để tách hiện tượng thuộc dataset hay backbone |
| **Concat trên ViT-B/16** | **ưu tiên 1** — LayUP (mục 10) kết luận ngược về tầng xa, phải đo mới phân định được |
| Augmentation ở mức feature | chưa — ý duy nhất còn lại đi cùng chiều với chẩn đoán "thiếu dữ liệu" |

Ý cuối đáng làm nhất trong ba ý còn lại: `Q` và `G` là **tổng theo mẫu**, nên
thêm phiên bản augment của mỗi ảnh chỉ thêm số hạng vào tổng — **kích thước `Q`
và `G` không đổi một byte**. Lật ngang là gấp đôi dữ liệu hiệu dụng, thêm crop là
gấp 8. Đây là trục duy nhất thêm được thông tin trong ràng buộc bộ nhớ cố định,
và Fly-CL không làm — `load_dataset.py` của họ chỉ có `Resize → CenterCrop →
ToTensor → Normalize`, một lượt duy nhất.

---

## 13. Tái lập

```bash
cd sparse-cl

# Fly-CL goc, giao thuc cua ho
python flycl_baseline.py --model_name resnet50.tv2_in1k --data_augmentation resnet                          --coding_level 0.3 --ridge_lower 4 --ridge_upper 10

# + concat   (grid 0:1 la doi chung khong dung stage 3)
python flycl_concat.py --model_name resnet50.tv2_in1k --coding_level 0.3 --grid 0:1,300:1

# + concat + ensemble 5 nhanh
python flycl_concat.py --model_name resnet50.tv2_in1k --coding_level 0.3                        --mode ens --branches 5 --grid 300:1
```

Lan chay dau voi `resnet50.tv2_in1k` se trich lai feature bon stage va luu cache
`CIFAR-100_resnet50.tv2_in1k+ms_resnet.pt`.

Muc 7 va 8:

```bash
# tang CP cua AnaCP
python anacp_cp.py --pos post --alpha_grid 0,0.25,0.5,1,2,5,10
python anacp_cp.py --pos post --spread none      # bo whitening
python anacp_cp.py --pos post --spread paper     # Lemma 4.1 that
python anacp_cp.py --pos pre                     # dat truoc phep chieu

# ban AnaCP day du (2 tang ridge + pseudo-replay)
python anacp_full.py --heads 1 --nl2 none  # tu kiem: phai trung --pos post
python anacp_full.py --heads 1 --nl2 topk

# code NGUYEN BAN cua ho tren feature cua ta
python anacp_reference.py --model_name resnet50.tv2_in1k
python anacp_reference.py --model_name resnet50.tv2_in1k+fsa1993

# tai lap paper cua ho (DINOv2, ~45 phut)
cd ../upstream/AnaCP && python run.py --dataset_name cifar100 --model_name anacp --mode cil     --backbone dinov2 --seed 0 --D 5000 --reg 1e2 --num_heads 3     --training_method aper --num_epochs 10 --learning_rate 1e-3     --lora_learning_rate 1e-4 --shared_cov --data_dir ../data

# FSA roi chay lai moi thu tren feature da adapt
cd ../sparse-cl
python fsa_extract.py  --seed 1993 --epochs 10
python anacp_cp.py     --model_name resnet50.tv2_in1k+fsa1993 --pos none
python flycl_concat.py --model_name resnet50.tv2_in1k+fsa1993+ms                        --coding_level 0.3 --grid 0:1,300:1 --ridge_lower 5
```

`anacp_cp.py` in canh bao khi λ do GCV chon roi vao dau hoac cuoi luoi — loi nay da
ba lan tao ket luan sai trong du an (muc 6).
