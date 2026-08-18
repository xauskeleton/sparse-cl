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
top-k. Thay vào đó, chiếu **cả stage 3 và stage 4** bằng hai projection riêng,
rồi kết hợp trước khi top-k.

| Cách kết hợp | Δ A_T | Δ Ā | Δ Forgetting | Bộ nhớ |
|---|---:|---:|---:|---|
| **product** `a ⊙ b` | +1.31 ± 0.15 | +1.18 ± 0.14 | −0.26 | **không đổi** |
| **concat** `[a ; b]` | **+2.07 ± 0.06** | **+1.37 ± 0.18** | **−1.06** | gấp 4 |

Ba seed, hiệu tính theo cặp từng seed, dương ở cả ba không ngoại lệ.

---

## 2. Bảng kết quả chính

Ba seed (1993 / 2023 / 2025).

| Cấu hình | Feature | Cách mix | Ensemble | Unit | Bộ nhớ `G` | A_T | Ā | Forgetting | Δ Ā |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| *Paper công bố* | s4 | — | 1 | 10.000 | 0.4 GB | — | *84.61 ± 0.16* | — | |
| **Fly-CL, bản tái lập** | s4 | — | 1 | 10.000 | 0.4 GB | 76.76 | 84.19 ± 0.42 | 8.13 | — |
| `expand_dim` 20.000 | s4 | — | 1 | 20.000 | 1.6 GB | 76.86 | 82.81 | 7.38 | **−1.38** |
| ensemble | s4 | — | 5 | 50.000 | 2.0 GB | 77.49 | 85.01 ± 0.46 | 8.09 | +0.82 ± 0.13 |
| **feature mix** | **s3+s4** | **product** | 1 | 10.000 | **0.4 GB** | 78.07 | 85.38 ± 0.42 | 7.87 | **+1.18 ± 0.14** |
| **feature mix** | **s3+s4** | **concat** | 1 | 20.000 | 1.6 GB | 78.83 | 85.56 ± 0.29 | **7.06** | **+1.37 ± 0.18** |
| all | s3+s4 | product | 5 | 50.000 | 2.0 GB | 78.84 | 86.11 ± 0.38 | 7.80 | +1.92 ± 0.08 |
| all | s3+s4 | concat | 2 | 40.000 | 3.2 GB | 79.34 | 86.10 ± 0.39 | 7.20 | +1.90 ± 0.03 |
| **all** | **s3+s4** | **concat** | **5** | 100.000 | 8.0 GB | **79.66** | **86.49 ± 0.40** | 7.34 | **+2.30 ± 0.07** |

Ba cột giữa xác định hoàn toàn mỗi dòng: **Feature** là những tầng backbone được
dùng, **Cách mix** là phép kết hợp chúng (`product` = nhân từng phần tử, giữ
10.000 unit; `concat` = nối đuôi, thành 20.000 unit), **Ensemble** là số nhánh.

Bốn nhóm: `expand_dim` chỉ tăng số chiều, `ensemble` chỉ nhân bản nhánh,
**`feature mix`** kết hợp hai tầng, và `all` gộp cả hai hướng.

**Product** là cải tiến duy nhất không tốn thêm bộ nhớ — cùng 10.000 unit, cùng
`G` cỡ 10.000², chỉ thêm một projection thưa (3 triệu giá trị khác 0).

**Concat** cho A_T cao hơn product **0.76 điểm** và forgetting thấp hơn **0.81**,
nhưng ở Ā chỉ hơn 0.18 (nằm trong nhiễu). Nó tốt hơn rõ **ở giai đoạn cuối** —
chỉ số quan trọng nhất của continual learning — còn Ā bị kéo lại vì giai đoạn đầu
hai bên ngang nhau. Concat cũng ổn định nhất bảng: σ = 0.29 so với 0.42.

**Dòng `expand_dim` 20.000 là đối chứng quan trọng nhất.** Nó cũng 20.000 unit,
cũng 1.6 GB như concat, khác đúng một chỗ: 10.000 unit thứ hai chiếu từ **stage
3** hay từ **stage 4 lần nữa**. Chiếu từ stage 3 hơn **2.75 điểm**, và bản thân
việc tăng `expand_dim` **làm tệ đi** 1.38.

**Product và ensemble độc lập, cộng dồn được**: riêng lẻ +1.18 và +0.82, kết hợp
+1.92 (tổng lý thuyết 2.00). Khác `expand_dim`, vốn không cộng dồn với product —
cả hai cùng làm mã giàu hơn nên thay thế nhau, còn ensemble tác động lên phương
sai nên độc lập.

---

## 3. Bảng chọn tầng và cách kết hợp

Một seed (1993). Mốc: A_T 76.99, Ā 84.08.

| Cách kết hợp | Feature | Unit | Bộ nhớ | Thời gian | A_T | Ā | Δ Ā |
|---|---|---:|---:|---:|---:|---:|---:|
| **product** `a ⊙ b` | s3+s4 | 10.000 | 0.4 GB | 97s | 78.13 | 85.12 | +1.04 |
| **concat** `[a ; b]` | s3+s4 | 20.000 | 1.6 GB | 105s | **79.01** | **85.60** | **+1.52** |
| concat, `b` gộp feature s2+s3 | s2+s3+s4 | 20.000 | 1.6 GB | 105s | 78.43 | 85.15 | +1.07 |
| concat, mỗi tầng một khối riêng | s2+s3+s4 | 30.000 | 3.6 GB | 321s | 78.81 | 85.02 | +0.94 |

### Cơ chế nằm ở đâu

Hai đối chứng chạy riêng, cùng seed và cùng cấu hình:

| Đối chứng | Ā | Δ | Kết luận |
|---|---:|---:|---|
| Kết hợp bằng **phép cộng** | 84.35 | +0.27 | giá trị nằm ở **phi tuyến bậc hai** |
| Nhân hai projection của **cùng stage 4** | 83.25 | **−0.84** | phải là **hai tầng khác nhau** |

Dòng đầu là phép so sạch nhất: cùng đặc trưng, cùng projection, cùng số unit,
cùng bộ nhớ — khác **đúng một dấu phép toán**. Cộng cho +0.27, nhân cho +1.04.

Dòng thứ hai loại bỏ cách giải thích "đặc trưng bậc hai nói chung thì tốt". Nếu
đúng vậy thì nhân hai projection độc lập của cùng stage 4 phải giúp, nhưng nó
**có hại**.

### Chỉ tầng liền kề mới đáng ghép

Stage 2 được thử **bốn cách khác nhau**, tất cả đều làm tệ đi so với chỉ dùng
stage 3:

| Cách đưa stage 2 vào | Δ so với chỉ dùng s3 |
|---|---:|
| Thay stage 3 bằng stage 2 (product) | −0.80 |
| Ghép feature s2 vào rồi chiếu chung (trên `a1_in1k`) | −0.33 |
| Gộp feature s2 vào nhánh `b` của concat | −0.45 |
| **Khối unit riêng cho s2 (concat 3 tầng)** | **−0.58** |

Cách cuối là phép thử công bằng nhất — stage 2 có projection riêng, khối unit
riêng, không phải chen chỗ với stage 3, và tổng số unit tăng từ 20.000 lên
30.000. Vẫn mất 0.58 điểm trong khi tốn **2.25 lần bộ nhớ và 3 lần thời gian**.

Bốn phép đo độc lập cùng một hướng đóng lại hướng "thêm nhiều tầng nữa". Cũng
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
kèm product. Ba seed.

| Nhánh | Unit | Bộ nhớ | Thời gian | A_T | Ā | Δ Ā theo cặp |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 10.000 | 0.4 GB | 97s | 76.76 | 84.19 ± 0.42 | — |
| 2 | 20.000 | 0.8 GB | 196s | 77.33 | 84.83 ± 0.39 | **+0.64 ± 0.16** |
| **5** | 50.000 | 2.0 GB | 489s | **77.49** | **85.01 ± 0.46** | **+0.82 ± 0.13** |
| 10 | 100.000 | 4.0 GB | 971s | 77.48 | 84.73 * | +0.65 * |

\* một seed (1993). Các dòng trên là trung bình ba seed.

Dương ở cả ba seed tới m = 5. Nhưng bước từ 1 lên 2 cho +0.64 còn từ 2 lên 5 chỉ
thêm **+0.18**, và từ 5 lên 10 thì **không thêm gì** — ở cùng seed 1993, m=5 cho
84.75 còn m=10 cho 84.73.

**Bão hoà ở m = 5.** Đường cong đo lần hai trên nền có product xác nhận cùng điểm
bão hoà và kéo dài hơn: 85.12 / 85.50 / 85.93 / 85.96 / 85.97 ở m = 1/2/5/10/20 —
từ 5 lên 20 chỉ thêm 0.04 điểm với 4 lần bộ nhớ và 4.6 lần thời gian.

Forgetting **không đổi** (8.13 → 8.15 → 8.09) — ensemble không giúp gì ở chỉ số
này, khác product (−0.26) và concat (−1.06).

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

### Quy luật chung

Bốn can thiệp thua nặng nhất đều **tăng độ biểu đạt** hoặc **chia nhỏ dữ liệu**.
CIFAR-100 có 500 ảnh mỗi lớp; chia 8 vùng thì mỗi chuyên gia còn khoảng 62 ảnh
mỗi lớp trong khi vẫn phải phân biệt đủ 100 lớp. Bài toán bị chặn bởi **dữ
liệu**, không bởi độ biểu đạt.

Điều này giải thích vì sao product ăn được còn các cách khác thì không: nó
**không** thêm tham số cũng **không** chia dữ liệu, mà đổi **lớp hàm** từ tuyến
tính sang bậc hai.

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

## 7. Giới hạn

**Bản tái lập thấp hơn số công bố 0.80** (84.19 ± 0.42 so với 84.61 ± 0.16).
Khoảng lệch này chưa giải thích được và **lớn hơn** phần chênh giữa 85.56 và
84.61. Nên phát biểu hợp lệ là *"+1.37 so với baseline chạy cùng code, cùng seed,
cùng projection"* — **không** phải "vượt Fly-CL".

**Phải đọc thêm stage 3 của backbone.** Trọng số backbone không đổi, không train
gì thêm, không tốn thêm phép tính nào trong backbone — bản đồ stage 3 đã được
tính sẵn trên đường forward. Nhưng đây vẫn là **đầu vào giàu hơn** so với Fly-CL.
Phép so cho cả hai bên cùng đầu vào và cùng số unit nằm ở dòng `expand_dim`
20.000 trong Bảng chính: **+2.75**.

**Chỉ ResNet-50 + CIFAR-100.** Chưa thử ViT-B/16 — kiến trúc này không có khái
niệm "tầng trung gian" theo kiểu ResNet, nên cách áp dụng chưa rõ. Chưa thử
CUB-200 hay VTAB.

**Số seed không đều.** Bảng chính có 3 seed cho mọi dòng (dòng product có 5).
Bảng chọn tầng và bảng ensemble mới một seed.

---

## 8. Đang chạy và còn mở

| | Trạng thái |
|---|---|
| concat + ensemble, 3 seed × {2, 5} nhánh | đang chạy |
| Quét `expand_dim` trên ViT + CIFAR-100 | chưa — để tách hiện tượng thuộc dataset hay backbone |
| Product/concat trên ViT-B/16 | chưa — cần định nghĩa "tầng liền kề" cho transformer |
| Augmentation ở mức feature | chưa — ý duy nhất còn lại đi cùng chiều với chẩn đoán "thiếu dữ liệu" |

Ý cuối đáng làm nhất trong ba ý còn lại: `Q` và `G` là **tổng theo mẫu**, nên
thêm phiên bản augment của mỗi ảnh chỉ thêm số hạng vào tổng — **kích thước `Q`
và `G` không đổi một byte**. Lật ngang là gấp đôi dữ liệu hiệu dụng, thêm crop là
gấp 8. Đây là trục duy nhất thêm được thông tin trong ràng buộc bộ nhớ cố định,
và Fly-CL không làm — `load_dataset.py` của họ chỉ có `Resize → CenterCrop →
ToTensor → Normalize`, một lượt duy nhất.

---

## 9. Tái lập

```bash
cd sparse-cl

# Fly-CL goc, giao thuc cua ho
python flycl_baseline.py --model_name resnet50.tv2_in1k --data_augmentation resnet \
                         --coding_level 0.3 --ridge_lower 4 --ridge_upper 10

# + product
python flycl_stagemix.py --model_name resnet50.tv2_in1k --coding_level 0.3 \
                         --combine prod --grid 300:1

# + concat
python flycl_stagemix.py --model_name resnet50.tv2_in1k --coding_level 0.3 \
                         --combine concat --grid 300:1

# + ensemble 5 nhanh
python flycl_stagemix.py --model_name resnet50.tv2_in1k --coding_level 0.3 \
                         --combine prod --mode ens --branches 5 --grid 300:1
```

`--grid 0:1` cho dòng đối chứng (không dùng stage 3). Lần chạy đầu với
`resnet50.tv2_in1k` sẽ trích lại feature bốn stage và lưu cache
`CIFAR-100_resnet50.tv2_in1k+ms_resnet.pt`.
