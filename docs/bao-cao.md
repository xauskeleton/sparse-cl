# Sparse expansion học được cho continual learning — báo cáo kết quả

CIFAR-100, 10 tasks × 10 classes, class-incremental, exemplar-free.
Hai backbone: ViT-B/16 và ResNet-50.

## Tóm tắt

Ý tưởng xuất phát là lấy "nửa trái" của Fly-CL — phép chiếu ngẫu nhiên thưa rồi top-k
winner-take-all — và biến nó thành bài toán học được, có thể kèm EWC-DR để kiểm soát trôi.
Ba biến thể được đề ra:

1. `(backbone → chiếu thưa)` đóng băng → MLP(ReLU), chỉ học MLP
2. `backbone` đóng băng → chiếu thưa → MLP(ReLU), học cả chiếu và MLP
3. `backbone` đóng băng → chiếu thưa → classifier tuyến tính, học chiếu

Kết quả: **cả ba đều kém hơn giữ nguyên thiết kế gốc của Fly-CL.** Cho phép chiếu học được
làm mất 10.6–11.8 điểm Ā; thêm MLP làm mất 6.1–15.2 điểm; hai tác hại cộng dồn chứ không bù
trừ.

Đóng góp dương duy nhất tìm được là **bản thân phép chiếu thưa cố định**: +2.24 Ā trên ViT
và +2.42 trên ResNet, tái lập được ở cả hai backbone.

EWC-DR có tác dụng đo được nhưng chỉ ở các cấu hình vốn đã kém, và không đưa cấu hình nào
vượt qua được `frozen projection + Linear head`. Fine-tune backbone làm tệ đi 7 điểm với chi
phí gấp 170–200 lần.

## Thiết lập

| | |
|---|---|
| Dataset | CIFAR-100, 10 task × 10 lớp, chia đều |
| Backbone | ViT-B/16 (`augreg2_in21k_ft_in1k`) hoặc ResNet-50 (`a1_in1k`), đóng băng |
| Projection | `W ∈ R^{10000×d}`, 300 non-zeros mỗi hàng, `d` = 768 hoặc 2048 |
| Activation | top-k winner-take-all, coding level 0.1 |
| Optimizer | AdamW, lr 1e-3, weight decay 1e-4, cosine schedule |
| Ngân sách | 100 epochs mỗi task, early stopping patience 20 |
| Validation | 10% cắt từ task hiện tại |
| Batch | 256 |
| Seeds | 1993 / 2023 / 2025; `±` là độ lệch chuẩn |

Cách chia task tái lập đúng Fly-CL: `set_seed(seed)` rồi `random.sample(range(100), 100)`,
để số liệu so sánh trực tiếp được với log có sẵn của họ.

Feature được trích **một lần cho cả dataset** rồi cắt theo task. Điều này hợp lệ vì feature
không phụ thuộc seed — chỉ cách chia task mới phụ thuộc. Nhờ đó một run chỉ mất 23 giây
(ViT) hoặc 47 giây (ResNet), và toàn bộ 120 run trong báo cáo này chạy hết khoảng 75 phút.

### Metrics

Sau mỗi task, đo accuracy trên **mọi lớp đã học từ đầu**, được dãy `A_t`:

```
after task    1      2      3      4      5      6      7      8      9     10
            97.70  95.95  93.27  91.50  89.90  88.80  88.06  85.27  84.20  83.33
```

| Metric | Định nghĩa | |
|---|---|---|
| `A_T` | phần tử cuối của dãy — accuracy trên cả 100 lớp sau khi học xong | cao hơn tốt hơn |
| `Ā` | trung bình cả dãy — model tốt tới đâu xuyên suốt quá trình | cao hơn tốt hơn |
| `Forgetting` | trung bình mức sụt từ đỉnh của mỗi task cũ | thấp hơn tốt hơn |

## Bảng tổng hợp — ResNet-50, hai chế độ backbone

| Backbone | # | Projection | Head | A_T | Ā | Forgetting | A_T | Ā | Forgetting |
|---|:-:|---|---|---:|---:|---:|---:|---:|---:|
| | | | | **no regularizer** | | | **+ EWC-DR (λ=100)** | | |
| **đóng băng** | 1 | none | Linear | 58.61 ± 1.23 | 71.97 ± 1.16 | 14.86 ± 2.04 | n/a | n/a | n/a |
| | 2 | none | MLP | 52.90 ± 1.40 | 65.89 ± 1.64 | 28.10 ± 2.27 | 57.07 ± 0.56 | 71.27 ± 1.71 | 17.92 ± 5.95 |
| | 3 | frozen | Linear | **62.46 ± 1.80** | **74.39 ± 1.65** | 15.00 ± 3.33 | n/a | n/a | n/a |
| | 4 | frozen | MLP | 55.14 ± 0.86 | 68.08 ± 2.21 | 28.52 ± 2.55 | 54.48 ± 2.47 | 70.28 ± 1.80 | 26.57 ± 5.76 |
| | 5 | learnable | Linear | 44.86 ± 2.86 | 63.79 ± 2.97 | 43.07 ± 10.55 | 45.13 ± 2.12 | 64.09 ± 0.44 | 37.84 ± 5.85 |
| | 6 | learnable | MLP | 32.07 ± 3.71 | 51.98 ± 2.23 | 64.19 ± 4.55 | 32.88 ± 3.57 | 59.11 ± 1.74 | 53.87 ± 3.98 |
| **fine-tune** | 1 | none | Linear | 48.26 | 64.89 | 4.56 | 49.11 | 64.89 | 4.20 |
| | 2 | none | MLP | — | — | — | — | — | — |
| | 3 | frozen | Linear | 50.87 | 67.38 | 4.96 | 54.84 | 68.80 | 3.51 |
| | 4 | frozen | MLP | — | — | — | — | — | — |
| | 5 | learnable | Linear | 24.92 | 46.55 | 27.57 | 33.91 | 50.06 | 31.33 |
| | 6 | learnable | MLP | — | — | — | — | — | — |

Hàng `đóng băng`: 3 seeds, batch 256, một run 47 giây.
Hàng `fine-tune`: 1 seed, batch 128, một run ~2.3 giờ. Batch khác nhau vì giới hạn VRAM.

`—` = chưa chạy. Ba cấu hình MLP ở chế độ fine-tune còn trống, lấp đủ mất khoảng 14 giờ.
Đây là chỗ duy nhất còn có thể bất ngờ: EWC-DR ở chế độ fine-tune tăng tác dụng theo lượng
tham số trôi, mà MLP thêm 1.05 M tham số trôi nữa.

**Kết luận về projection giữ nguyên qua cả hai chế độ**, và giữ gần như y hệt về độ lớn:

| | Δ Ā khi thêm frozen projection | Δ Ā khi cho projection học |
|---|---:|---:|
| backbone đóng băng | **+2.42** | −10.60 |
| backbone fine-tune | **+2.49** | **−20.83** |

Thứ tự `frozen > none > learnable` không đổi. Đây là xác nhận mạnh nhất cho kết quả dương duy
nhất của nghiên cứu — nó độc lập với việc backbone có học hay không.

Ngược lại, tác hại của việc cho phép chiếu học được **nặng gấp đôi** khi backbone cũng học.
Hai nguồn trôi cộng dồn chứ không bù trừ.

**Mở backbone thua đóng băng ở mọi cấu hình**, kể cả khi có EWC-DR: config 3 mất 5.59 điểm Ā
(68.80 so với 74.39) và tốn 8000 giây thay vì 47.

## Bảng kết quả chính — ViT-B/16

| # | Projection | Head | A_T | Ā | Forgetting | A_T | Ā | Forgetting |
|:-:|---|---|---:|---:|---:|---:|---:|---:|
| | | | **no regularizer** | | | **+ EWC-DR (λ=100)** | | |
| 1 | none | Linear | 80.20 ± 1.03 | 87.25 ± 1.33 | 9.53 ± 1.07 | n/a | n/a | n/a |
| 2 | none | MLP | 76.99 ± 1.11 | 86.24 ± 0.90 | 16.37 ± 0.84 | 74.56 | 85.19 | 18.39 |
| 3 | frozen | Linear | **83.17 ± 0.23** | **89.49 ± 0.97** | **7.49 ± 0.63** | n/a | n/a | n/a |
| 4 | frozen | MLP | 53.48 ± 1.40 | 74.25 ± 1.57 | 41.08 ± 3.12 | 51.73 | 74.17 | 43.70 |
| 5 | learnable | Linear | 59.14 ± 1.65 | 77.73 ± 2.77 | 39.55 ± 1.93 | 71.24 | 83.14 | 22.98 |
| 6 | learnable | MLP | 30.18 ± 0.47 | 59.61 ± 3.69 | 72.04 ± 1.18 | 35.63 | 61.52 | 60.88 |
| — | *Fly-CL (mốc)* | *ridge, closed-form* | *88.68* | *92.99* | *—* | | | |

## Bảng kết quả chính — ResNet-50

| # | Projection | Head | A_T | Ā | Forgetting | A_T | Ā | Forgetting |
|:-:|---|---|---:|---:|---:|---:|---:|---:|
| | | | **no regularizer** | | | **+ EWC-DR (λ=100)** | | |
| 1 | none | Linear | 58.61 ± 1.23 | 71.97 ± 1.16 | 14.86 ± 2.04 | n/a | n/a | n/a |
| 2 | none | MLP | 52.90 ± 1.40 | 65.89 ± 1.64 | 28.10 ± 2.27 | 57.07 ± 0.56 | 71.27 ± 1.71 | 17.92 ± 5.95 |
| 3 | frozen | Linear | **62.46 ± 1.80** | **74.39 ± 1.65** | 15.00 ± 3.33 | n/a | n/a | n/a |
| 4 | frozen | MLP | 55.14 ± 0.86 | 68.08 ± 2.21 | 28.52 ± 2.55 | 54.48 ± 2.47 | 70.28 ± 1.80 | 26.57 ± 5.76 |
| 5 | learnable | Linear | 44.86 ± 2.86 | 63.79 ± 2.97 | 43.07 ± 10.55 | 45.13 ± 2.12 | 64.09 ± 0.44 | 37.84 ± 5.85 |
| 6 | learnable | MLP | 32.07 ± 3.71 | 51.98 ± 2.23 | 64.19 ± 4.55 | 32.88 ± 3.57 | 59.11 ± 1.74 | 53.87 ± 3.98 |
| — | *Fly-CL (mốc)* | *ridge, closed-form* | *72.87* | *81.03* | *9.07* | | | |

Mốc Fly-CL chạy trên cùng checkpoint `a1_in1k` và cùng `coding_level` 0.1 với bảng trên. Ở
`coding_level` 0.3 của họ thì Ā lên 82.21; ở checkpoint `tv2_in1k` của họ thì lên 84.08.

### Cách đọc

`Projection: none` = bỏ hẳn phép chiếu và top-k, đưa feature backbone thẳng vào head.
`frozen` = sparse random projection giữ nguyên từ lúc khởi tạo. `learnable` = cùng phép
chiếu nhưng cập nhật bằng gradient qua các task (`projection_lr` 5e-3).
`Head: MLP` = `Linear(d→512) → ReLU → Linear(512→100)`.

Config 4, 5, 6 là ba phương án đề ra. Config 1, 2, 3 là control để tách bạch từng yếu tố.

`n/a` — cấu hình không có tham số nào học liên tục qua các task. Phép chiếu đóng băng,
không MLP, và classifier chỉ **thêm hàng mới**: hàng cũ được giữ nguyên trong
`IncrementalLinear.expand`, còn `--ce_scope new` chỉ tính cross-entropy trên logit lớp mới
nên hàng cũ không nhận gradient. Không có gì trôi, hình phạt EWC luôn bằng 0, và `validate()`
chặn tổ hợp này thay vì để chạy ra số vô nghĩa.

Cột `EWC-DR` trên ResNet chạy 3 seeds; trên ViT chạy 1 seed.

Không có mốc Fly-CL cho ResNet-50 — repo gốc chỉ kèm một log duy nhất, ViT-B/16 trên
CIFAR-100.

## Ablation theo cặp

| So sánh | Yếu tố được cô lập | Δ Ā (ViT) | Δ Ā (ResNet) |
|---|---|---:|---:|
| 1 → 3 | thêm sparse projection (Linear head) | **+2.24** | **+2.42** |
| 2 → 4 | thêm sparse projection (MLP head) | −11.99 | **+2.19** |
| 1 → 2 | thêm MLP head (không projection) | −1.01 | −6.08 |
| 3 → 4 | thêm MLP head (frozen projection) | −15.24 | −6.31 |
| 5 → 6 | thêm MLP head (learnable projection) | −18.12 | −11.81 |
| 3 → 5 | cho projection học (Linear head) | −11.76 | −10.60 |
| 4 → 6 | cho projection học (MLP head) | −14.64 | −16.10 |

### Sparse projection có ích

+2.24 và +2.42 với Linear head, tái lập được trên hai backbone hoàn toàn khác nhau về kiến
trúc, dữ liệu tiền huấn luyện và chiều feature. Forgetting cũng giảm trên ViT (9.53 → 7.49).

Đây là kết quả dương duy nhất trong toàn bộ nghiên cứu, và nó ủng hộ đúng cơ chế mà Fly-CL
đề xuất: mở rộng lên chiều cao rồi giữ thưa làm giảm multicollinearity giữa các feature, nên
bài toán similarity-matching ở tầng cuối dễ hơn.

Nhưng hiệu ứng chỉ vào khoảng 1.7–2σ với ba seed. Là dấu hiệu, chưa phải bằng chứng chắc.

### MLP head luôn có hại, và hại khác nhau giữa hai backbone

Trên ViT, thêm MLP sau phép chiếu làm sập 15.24 điểm. Trên ResNet chỉ 6.31. Khác biệt giải
thích được bằng số tham số của lớp đầu MLP:

| Backbone | d | `Linear(d→512)` | Với projection `Linear(10000→512)` | Tỉ lệ |
|---|---:|---:|---:|---:|
| ViT-B/16 | 768 | 0.39 M | 5.12 M | **13×** |
| ResNet-50 | 2048 | 1.05 M | 5.12 M | 4.9× |

Trên 4050 mẫu train mỗi task, phình 13 lần thì overfit nặng; phình 4.9 lần thì chịu được.
Tác hại của MLP không đến từ projection mà từ **tỉ lệ phình tham số**, và ViT chịu nặng hơn
chỉ vì feature gốc của nó nhỏ hơn.

Hệ quả thực dụng: nếu buộc phải dùng MLP thì trên ViT nên bỏ hẳn projection (86.24 so với
74.25).

Có một lý do cơ chế nữa. Kiến trúc này chống quên một phần nhờ **tính cục bộ của gradient**:
top-k chỉ cho 10% unit đi qua, nên mỗi mẫu chỉ cập nhật một phần nhỏ trọng số. Với
`mlp_act=relu`, gradient ở tầng cuối trở thành dense và tính cục bộ đó mất đi ở đúng lớp
phân loại.

### Cho phép chiếu học được thì hại nặng

−11.76 trên ViT và −10.60 trên ResNet, kèm forgetting tăng gấp ba tới năm lần. Đây là bác bỏ
trực tiếp giả thuyết ban đầu.

Giả thuyết phòng vệ "cho học chỉ giúp khi feature yếu" cũng không đứng: ResNet-50 có feature
yếu hơn ViT-B/16 rõ rệt (Ā tốt nhất 74.39 so với 89.49), mà kết quả vẫn cùng chiều và cùng
độ lớn.

Đã kiểm rằng đây không phải lỗi chọn hyperparameter. `projection_lr` được quét qua ba bậc độ
lớn (1e-5 đến 5e-3) và ba chế độ bias cho lớp chiếu (`none` / `fixed` / `learn`) — không chế
độ nào thắng được phép chiếu đóng băng. Ở `projection_lr` 5e-3, `proj_drift` đạt 1.82, tức
trọng số đã đi xa hơn cả độ lớn ban đầu của chính nó.

## EWC-DR

EWC-DR chỉ áp dụng được ở bốn trong sáu cấu hình.

| Config | Δ Ā (ViT) | Δ Ā (ResNet) |
|---|---:|---:|
| 2 none + MLP | −1.05 | **+5.38** |
| 4 frozen + MLP | −0.08 | +2.20 |
| 5 learnable + Linear | **+5.41** | +0.30 |
| 6 learnable + MLP | +1.91 | **+7.13** |

Hiệu ứng có thật nhưng **không nhất quán giữa hai backbone**: trên ViT nó chỉ giúp config 5,
trên ResNet nó giúp mạnh nhất ở config 2 và 6. Điểm chung là các cấu hình có nhiều tham số
học liên tục nhất — đúng nơi có nhiều thứ để bảo vệ nhất.

Nhưng **không cấu hình nào vượt được `frozen + Linear`**. Bản tốt nhất có EWC-DR là config 5
trên ViT với Ā 83.14, vẫn kém 6.35 điểm so với 89.49 của một cấu hình không dùng regularizer
nào.

Lý do có tính cấu trúc: `frozen + Linear` đạt được **bằng thiết kế** cái mà EWC-DR phải đạt
bằng hình phạt. Nó không có tham số nào trôi, nên không cần chống trôi.

### λ = 100, không phải 10000

Repo gốc EWC-DR đặt `lamda: 10000`. Quét qua {1, 10, 100, 1000, 10000} trên ba cấu hình của
ViT cho thấy λ=100 mới đúng ở quy mô này — thấp hơn hai bậc độ lớn, và nhất quán ở cả ba.

| λ | 1 | 10 | 100 | 1000 | 10000 |
|---|---:|---:|---:|---:|---:|
| Ā, config 4 | 73.78 | 72.68 | **74.17** | 72.60 | 44.32 |
| Ā, config 5 | 78.48 | 79.49 | **83.14** | 72.54 | 70.27 |
| Ā, config 6 | 54.66 | 57.22 | **61.52** | 56.59 | 46.89 |

Forgetting giảm đơn điệu theo λ ở cả ba, nhưng Ā thì không — nó đạt đỉnh ở giữa rồi sụp.
Ở λ=10000, `pen/clf` đạt 2.0–5.2, tức hình phạt áp đảo loss phân loại và mạng gần như bị
đóng băng.

### λ không chuyển được giữa các chế độ

Cùng λ=100, tỉ lệ `penalty / loss_clf` lệch cả bậc độ lớn khi đổi backbone, đổi tập tham số
được bảo vệ, hoặc đổi từ backbone đóng băng sang fine-tune.

Đó là hệ quả trực tiếp của việc hình phạt **cộng dồn trên mọi tham số được bảo vệ**:

```
penalty = λ · Σᵢ ωᵢ (θᵢ − θ*ᵢ)² / 2
```

Độ lớn của nó phụ thuộc số tham số và thang gradient — hai thứ đổi mỗi khi ta đổi cấu hình.
Chuẩn hoá ω theo trung bình trước khi nhân λ sẽ làm λ trở thành đại lượng chuyển được; chưa
thử.

### Suy λ từ lý thuyết

EWC xuất phát từ Laplace approximation, nên về nguyên tắc **không có λ** — hệ số đã được ấn
định. λ chỉ xuất hiện vì cách cài đặt làm lệch thang đo, và truy ngược chỗ lệch cho ra một
dự đoán kiểm chứng được.

`Regularizer.estimate` lấy gradient của loss **trung bình theo batch** rồi bình phương, nên
`ω ≈ F_mẫu / B`. Loss phân loại cũng là trung bình. Cân hai vế với hậu nghiệm Laplace
`F_total = N_cũ · F_mẫu` cho:

```
λ = B · N_cũ / N_mới
```

Với B=128 và mỗi task 4500 mẫu, λ nằm trong khoảng 128 tới 1152 tuỳ task — đúng bậc độ lớn
quan sát được.

Công thức này cũng chỉ ra một lệch so với EWC gốc: `F_total` là **tổng** Fisher qua các task,
trong khi cả repo EWC-DR lẫn bản cài ở đây đều lấy **trung bình có trọng số**
(`alpha = known/total`). Trung bình giữ độ lớn ω không đổi, nên hình phạt không mạnh lên khi
lượng kiến thức cần bảo vệ tăng. Điều này khớp với quan sát trong ma trận accuracy: EWC-DR
bảo vệ các task **đầu** tốt hơn nhưng phá các task **giữa** nặng hơn, tổng bằng nhau.

## Fine-tune backbone

Câu hỏi: mở backbone có đáng không. Chạy trên ResNet-50, 100 epochs, batch 128, một seed.

| Config | Backbone | A_T | Ā | Forgetting | Thời gian một run |
|---|---|---:|---:|---:|---:|
| 1 none + Linear | đóng băng | 58.61 | 71.97 | 14.86 | 47 s |
| 1 none + Linear | fine-tune | 48.26 | **64.89** | 4.56 | 9392 s |
| 3 frozen + Linear | đóng băng | 62.46 | 74.39 | 15.00 | 47 s |
| 3 frozen + Linear | fine-tune | 50.87 | **67.38** | 4.96 | 8049 s |
| 5 learnable + Linear | đóng băng | 44.86 | 63.79 | 43.07 | 47 s |
| 5 learnable + Linear | fine-tune | 24.92 | **46.55** | 27.57 | 7434 s |

Config 1 và 3 mất đúng **7 điểm Ā**, config 5 mất **17.24**, với chi phí gấp **160–200 lần**.
Kết luận đứng trên cấu hình tốt nhất chứ không chỉ trên cấu hình yếu.

### Forgetting thấp không phải thành tích

Điểm đáng chú ý nhất của nhánh này: fine-tune cho forgetting **thấp hơn ba lần** nhưng
accuracy tệ hơn hẳn. Nghịch lý được giải bằng đường chéo ma trận accuracy — accuracy ngay
sau khi học xong mỗi task, config 3:

```
task    0      1      2      3      4      5      6      7      8      9
      87.50  82.40  68.60  55.60  49.20  41.50  39.70  36.70  37.70  39.30
```

Từ task 3 trở đi mô hình chỉ đạt 37–56% trên **chính task nó vừa học**. Nó không quên — nó
chưa bao giờ học được.

Forgetting đo mức sụt **từ đỉnh**. Đỉnh đã thấp thì không còn gì để mất. Đây là bằng chứng
cụ thể rằng forgetting đọc một mình là chỉ số gây hiểu lầm: một mô hình không học gì có
forgetting bằng 0.

Nguyên nhân nằm ở `--ce_scope new`. Loss chỉ đẩy logit lớp mới lên, không bao giờ đẩy logit
lớp cũ xuống. Tới task 7 thì 70 lớp cũ đã có logit cao ổn định, còn 10 lớp mới phải trèo lên
từ đầu. Chỉ số `val_acc` xác nhận: nó rơi từ 90.6 xuống 32–41 qua các task, dù validation
chỉ chứa mẫu của task hiện tại.

### EWC-DR ở chế độ fine-tune

Ở đây EWC-DR **có tác dụng**, và tác dụng tăng theo lượng tham số trôi:

| Config | Tham số trôi | Δ A_T | Δ Ā |
|---|---|---:|---:|
| 1 none | backbone 23.5 M | +0.85 | +0.00 |
| 3 frozen | backbone 23.5 M | +3.97 | +1.42 |
| 5 learnable | backbone 23.5 M + projection 20.5 M | **+8.99** | **+3.51** |

Đây là quan hệ đơn điệu sạch nhất tìm được cho EWC-DR trong toàn bộ nghiên cứu, và nó khớp
đúng cơ chế: hình phạt chống trôi thì càng có nhiều thứ trôi càng có ích.

Nhưng nó vẫn không đủ để bù cho việc mở backbone. Config 3 với EWC-DR đạt Ā 68.80, vẫn kém
**5.59 điểm** so với 74.39 của chính cấu hình đó khi backbone đóng băng.

### Một quét λ cho kết quả sai, và vì sao

Trước khi có số liệu trên, một sweep λ được chạy ở 64×64 với 20 epoch cho rẻ:

| λ | 0 | 10 | 30 | 100 | 1000 |
|---|---:|---:|---:|---:|---:|
| Ā | 57.62 | 57.78 | 57.71 | 57.65 | 57.08 |

Toàn bộ dải nằm trong 0.7 điểm, không có đỉnh, và kết luận rút ra khi đó là "EWC-DR không làm
gì ở chế độ fine-tune". Kết luận đó **sai**.

Hai lý do, và cả hai đều là bài học về cách rút gọn thí nghiệm:

`pen/clf` ở 64px chỉ 0.0045, so với 0.05–0.085 ở 224px. Ảnh nhỏ làm feature yếu đi và thang
gradient đổi theo.

Quan trọng hơn: **20 epoch thì trọng số chưa trôi đủ để có gì mà phạt**. Hình phạt tỉ lệ với
`(θ−θ*)²`, nên cắt ngân sách epoch không phải phép sàng lọc trung tính cho λ — nó bóp chính
đại lượng mà λ tác động lên. Rút gọn theo chiều nào cũng được, trừ chiều đó.

`omega_saturated ≈ 0.52` ở mọi λ — hơn một nửa tham số chạm trần `omegamax = 1e-4`, nên ω mất
khả năng phân biệt và **EWC-DR thoái hoá gần thành L2 thuần**. Hạn chế này vẫn còn ở các run
224px (0.54–0.64), nên con số +3.51 nhiều khả năng là cận dưới.

## Đối chiếu với hai repo gốc

### Fly-CL

Thuật toán của họ được cài lại trong `sparse-cl/flycl_baseline.py` để chạy trên **cùng
feature, cùng class order, cùng split** với mọi bảng ở đây. Bản cài lại đã kiểm chứng trên cả
hai backbone:

| | Bản cài lại | Nguồn gốc | Lệch |
|---|---:|---:|---:|
| ViT-B/16 (log seed 1993 của họ) | 88.77 / 93.11 | 88.68 / 92.99 | 0.09 / 0.12 |
| ResNet-50 (Table 2 của paper) | 76.99 / **84.08** | — / 84.61 ± 0.16 | — / 0.53 |

#### Giao thức chuẩn của Fly-CL

Ba tham số dưới đây **bắt buộc** phải đúng; sai bất kỳ cái nào cũng làm con số ResNet sụt
nhiều điểm mà không có dấu hiệu gì bất thường trong log.

| Tham số | Giá trị | Ghi chú |
|---|---|---|
| `--ridge_lower` | **4** (mặc định `main.py`) | `scripts/test_cifar.sh` ghi 6 nhưng script đó **chỉ dành cho ViT**. Repo không có script ResNet. |
| `--coding_level` | **0.3** | `main.py` mặc định 0.01, script CIFAR ghi đè thành 0.3. |
| Checkpoint ResNet-50 | **`resnet50.tv2_in1k`** | Tương đương `resnet50-11ad3fa6.pth` (torchvision IMAGENET1K_V2) mà `load_model.py` nạp. Khác `resnet50` mặc định của timm (`a1_in1k`) dùng cho mọi bảng ở đây. |

```bash
# tai lap dung Fly-CL tren ResNet-50
python flycl_baseline.py --model_name resnet50.tv2_in1k --data_augmentation resnet \
                         --coding_level 0.3 --ridge_lower 4 --ridge_upper 10
```

Bóc tách khoảng cách giữa lần đo đầu tiên và số của paper (ResNet-50, Ā):

| Bước | Ā | Δ |
|---|---:|---:|
| Đo lần đầu (`ridge_lower 6`, `k` 0.1, `a1_in1k`) | 74.04 | |
| Hạ sàn λ xuống 1e3 | 81.03 | **+6.99** |
| `coding_level` 0.1 → 0.3 | 82.21 | +1.18 |
| Checkpoint `a1_in1k` → `tv2_in1k` | 84.08 | +1.87 |
| Paper | 84.61 ± 0.16 | *còn 0.53* |

Sàn λ là yếu tố lớn nhất, gấp bốn lần hai yếu tố còn lại cộng lại. GCV chọn λ trong một lưới
log do người dùng đặt; nếu λ tối ưu nằm dưới sàn thì nó im lặng chọn sàn. Trên ResNet-50 λ tối
ưu là 1e4, dưới sàn 1e6 mà `test_cifar.sh` dùng cho ViT.

#### Khoảng cách với cấu hình tốt nhất ở đây

So sánh phải dùng **cùng checkpoint** (`a1_in1k`, như mọi bảng ở đây):

| | Fly-CL (`k`=0.1) | Fly-CL (`k`=0.3) | config 3 ở đây | Chênh |
|---|---:|---:|---:|---:|
| ViT-B/16, Ā | 92.85 | 93.11 | 89.49 | **−3.36** |
| ResNet-50, Ā | 81.03 | 82.21 | 74.39 | **−6.64 … −7.82** |

Chênh lệch nằm ở **head chứ không ở representation**. Ở task 1 hai bên đạt giống hệt nhau —
cùng backbone, cùng projection, cùng top-k. Khoảng cách chỉ mở ra từ task 2 và tăng đều.

Lý do có tính cấu trúc: `Q = Σ H Y` và `G = Σ H Hᵀ` là **sufficient statistics** của bài toán
ridge, nên `(G + λI)⁻¹Q` bằng **đúng** nghiệm khi gộp toàn bộ dữ liệu mọi task và giải một
lần — không xấp xỉ, và không phụ thuộc thứ tự task. Phần "continual" của Fly-CL vì thế đã tối
ưu tuyệt đối và không còn gì để cải tiến; forgetting 4.79 của nó không phải quên mà là mức một
mô hình tuyến tính train chung trên 100 lớp cũng gặp. Cấu hình ở đây dùng SGD chỉ nhìn task
hiện tại. Đó là toàn bộ khoảng cách.

Khoảng cách trên ResNet-50 **lớn gấp đôi** trên ViT-B/16. Feature ResNet yếu hơn nên có nhiều
dư địa tối ưu hơn, và nghiệm đóng khai thác hết dư địa đó còn SGD thì không.

### EWC-DR

Bảng của paper, CIFAR-100, equally split, T=10:

| Method | A_last | A_avg |
|---|---:|---:|
| EWC | 10.62 | 27.90 |
| online EWC | 17.45 | 33.04 |
| SI | 9.91 | 23.58 |
| MAS | 23.41 | 37.84 |
| EWC-DR | 29.41 | 46.01 |
| **Cấu hình tốt nhất ở đây (ViT)** | **83.17** | **89.49** |

Chênh lệch **không nói phương pháp ở đây tốt hơn** — nó nói backbone pretrained tốt hơn train
from scratch. Paper huấn luyện ResNet-18 từ đầu ở 32×32 với lr 0.1.

Nhưng đúng chỗ đó giải thích vì sao EWC-DR yếu trong thiết lập này. Trong bảng của họ, EWC-DR
nâng A_last từ 10.62 lên 29.41, gần gấp ba so với EWC thường. Ở đây mức tốt nhất là +7.13 Ā.
EWC-DR chống trôi trọng số; train từ đầu thì biểu diễn thay đổi hoàn toàn mỗi task nên có rất
nhiều thứ để bảo vệ, còn backbone đóng băng thì rất ít.

Bản cài đặt ở đây khớp repo gốc ở mọi chi tiết: ước lượng ω từ gradient trung bình theo batch,
chia cho số batch, cắt trần bằng `omegamax`, trộn qua task bằng `alpha = known/total`,
Logits Reversal `logits*-1` trước cross-entropy, và CE chỉ trên logit lớp mới.

## Ablation phụ

**`expand_dim`** — giúp tới khoảng 5000 rồi không phân biệt được nữa. Chi phí tuyến tính theo
tham số này nên mở rộng là nút rẻ. Có một dấu hiệu dương chưa được xác nhận: khi giữ `k` cố
định thay vì giữ tỉ lệ, forgetting giảm đơn điệu 10.16 → 7.32 qua năm điểm. Cần thêm seed.

**`coding_level`** — phẳng trong dải đo được (0.02–0.30) **khi head học bằng SGD**, tức dưới
độ phân giải của ba seed. Nhưng với head nghiệm đóng (không có nhiễu tối ưu hoá) thì hiệu ứng
hiện rõ: 0.1 → 0.3 cho +1.18 Ā trên ResNet-50 và +0.26 trên ViT-B/16. Kết luận "phẳng" chỉ
đúng trong phạm vi head SGD, nơi nhiễu seed lớn hơn hiệu ứng.

**`proj_bias`** — ba chế độ `none` / `fixed` / `learn` không khác nhau về accuracy. Học bias
liên tục qua các task làm forgetting tăng 1.49 điểm (3.3 SE), tức có hại nhẹ nhưng đo được.

**Nhiễu seed** — đo trên năm seed ở cấu hình tốt nhất: Ā có σ = 0.94, forgetting có σ = 0.57.
Mọi chênh lệch dưới khoảng 2 điểm Ā trong báo cáo này nên đọc là chưa kết luận được.

## Vấn đề phương pháp đã phát hiện

Ghi lại vì chúng đã trực tiếp tạo ra kết luận sai trong quá trình làm.

**Sai normalization ở 15 run.** Sweep λ trên ResNet chạy với `--data_augmentation vit`, tức
chuẩn hoá feature ResNet bằng mean/std của ViT (0.5) thay vì thống kê ImageNet. Vì
`data_augmentation` nằm trong tag cache, chúng còn đọc từ **một file cache hoàn toàn khác**.
Kết luận "EWC-DR không giúp gì trên ResNet" rút từ chúng đã phải rút lại — sau khi chạy lại
đúng giao thức, EWC-DR cho +5.38 và +7.13 ở hai cấu hình.

**Lệch giao thức early stopping.** Cùng sweep đó chạy patience 10 trong khi baseline chạy 20,
nên nhánh EWC được train ít hơn, làm phần "giảm forgetting" bị thổi lên và phần "giảm
accuracy" bị phạt oan.

**Tên file không mã hoá đủ flag.** Cả hai lỗi trên tồn tại âm thầm vì `data_augmentation`,
`early_stop_patience`, `epochs`, `lr` và `image_size` đều không nằm trong tên file kết quả,
nên các run khác giao thức ghi đè lên nhau mà không báo gì. Một sweep `projection_lr` đã bị
mất trắng theo cách này trước đó. Đã đưa tất cả vào tên và đổi tên 37 file cũ cho khớp.

**Resize trên GPU không khớp PIL.** Khi cài đường ảnh cho fine-tune, `F.interpolate(bicubic)`
cho cosine similarity chỉ 0.981 so với feature đã cache. Nguyên nhân ba tầng: PIL dùng kernel
Keys `a = −0.5` còn PyTorch dùng `−0.75`; PIL resize hai lượt ngang/dọc riêng và làm tròn về
uint8 sau **mỗi** lượt; và làm tròn của PIL là half-up. Cài lại đúng cả ba cho cosine
1.000000 trên cả hai backbone.

**Chọn nhầm dtype trên GPU không hỗ trợ bf16.** `torch.cuda.is_bf16_supported()` trả về `True`
trên Tesla T4 vì mặc định nó tính cả trường hợp emulation. T4 không có tensor core bf16, nên
lựa chọn đó chạy chậm hơn khoảng 8 lần so với fp16. Phải hỏi chỉ số SM thay vì hỏi "có hỗ trợ
không".

**Trần `omegamax` nén dải động của ω.** `omega_saturated` đạt 0.52–0.64 ở chế độ fine-tune,
tức quá nửa tham số bị cắt bằng nhau. Giá trị `1e-4` lấy từ repo gốc, nơi thang gradient hoàn
toàn khác. Mọi kết luận về EWC-DR ở nhánh fine-tune còn dính hạn chế này.

**Early stopping nhìn sai thứ.** Validation chỉ chứa lớp của task hiện tại, nên tiêu chí dừng
đo mức khớp task mới chứ không đo quên task cũ. Ở nhánh fine-tune, `val_acc` giữ 95–99% trong
khi `A_t` rơi từ 99.3 xuống 76.3. Với backbone đóng băng thì ít hại vì head cạn khả năng
nhanh; với 86M tham số tự do thì checkpoint được chọn chính là checkpoint đã dịch chuyển xa
nhất khỏi task cũ.

## Kết luận

Cấu hình tốt nhất là **frozen sparse random projection + Linear head** — đúng thiết kế của
Fly-CL. Cả hai ý tưởng đề ra ban đầu đều làm model tệ đi, và hai tác hại cộng dồn chứ không
bù trừ.

Đóng góp dương duy nhất là bản thân phép chiếu thưa: +2.24 Ā trên ViT và +2.42 trên ResNet,
tái lập được trên hai backbone.

EWC-DR có tác dụng đo được nhưng không nhất quán giữa hai backbone khi backbone đóng băng.
Ở chế độ fine-tune thì nó nhất quán và đơn điệu theo lượng tham số trôi (+0.00 / +1.42 /
+3.51 Ā cho config 1 / 3 / 5) — nhưng vẫn không đưa cấu hình nào vượt qua được
`frozen + Linear` với backbone đóng băng.

Fine-tune backbone làm tệ đi 7 điểm ở hai cấu hình tốt và 17 điểm ở cấu hình có projection
học được, với chi phí gấp 160–200 lần.

Khoảng cách còn lại so với Fly-CL nằm hoàn toàn ở **cách giải head**, không ở representation —
bằng chứng là task 1 hai bên trùng khít. Trên cùng checkpoint và cùng `coding_level`, khoảng
cách là **3.36 Ā trên ViT-B/16 và 6.64 trên ResNet-50** — tức lớn gấp đôi ở backbone yếu hơn,
vì feature yếu để lại nhiều dư địa tối ưu hơn và nghiệm đóng khai thác hết còn SGD thì không.

Hướng đáng theo tiếp là thay SGD head bằng nghiệm closed-form tích luỹ sufficient statistics,
chứ không phải làm cho nửa trái học được. Nhưng cần nhận thức rõ giới hạn của hướng đó: phần
closed-form của Fly-CL **đã tối ưu tuyệt đối** trên feature cho trước (xem phần đối chiếu), nên
áp dụng nó chỉ đưa ta *bằng* Fly-CL chứ không vượt. Muốn vượt thì phải cải thiện chính feature,
mà mọi thao tác lên feature đều phải giữ được tính bất biến theo thứ tự task — nếu không sẽ mất
đúng thứ làm nên giá trị của nghiệm đóng.

Hai cải tiến đã thử trên chính phần head của Fly-CL đều **không ăn thua**, đo trên ResNet-50:

| Cải tiến | Δ Ā |
|---|---:|
| GCV trên toàn bộ dữ liệu tích luỹ thay vì từng task | +0.00 |
| Top-k theo trị tuyệt đối (giữ cả đuôi âm) thay vì chỉ đuôi dương | −0.29 |
| Cả hai | −1.27 |

Cái thứ nhất là một lệch có thật về mặt logic trong `main.py:112` — λ chọn trên một task rồi
áp lên `G` tích luỹ — nhưng vô hại về mặt số, vì các task đồng dạng nên λ tối ưu không đổi
theo quy mô. Cái thứ hai bác bỏ giả thiết "hai đuôi của phân bố mang lượng thông tin tương
đương": đuôi dương mang nhiều hơn.

## Giới hạn

Chỉ CIFAR-100. Chưa thử CUB-200-2011 hay VTAB, là hai dataset có domain shift so với
ImageNet, nơi backbone đóng băng có thể không còn đủ và kết luận có thể đổi.

Nhánh fine-tune mới có một seed và chưa đủ cả sáu cấu hình.

Mốc Fly-CL nay chạy lại trong cùng môi trường trên cùng feature (`flycl_baseline.py`), nhưng
vẫn **chỉ một seed** (1993). Phép chiếu là ngẫu nhiên nên con số có nhiễu; muốn so chặt chẽ
cần ít nhất ba seed cho mỗi ô.

Cột EWC-DR của bảng ViT chỉ có một seed.

Với ba seed, mọi chênh lệch dưới khoảng 2 điểm Ā chưa kết luận được. Điều này áp cho chính
kết quả dương chính (+2.24 và +2.42). Thêm seed rất rẻ — mỗi run 23–47 giây.

## Tái lập

```bash
git clone <repo> && cd sparse-cl
pip install -r requirements.txt

# cau hinh tot nhat
python train.py --train_projection False --projection_schedule task0 --cl_reg none

# ca luoi
python run_grid.py --backbone resnet --freeze_backbone True --seeds 1993,2023,2025
python make_table.py --backbone resnet
```

Lần chạy đầu tự tải CIFAR-100 và trọng số backbone rồi cache feature (~5 phút). Sau đó mỗi
run mất 23 giây (ViT) hoặc 47 giây (ResNet).

`make_table.py` **lọc theo giao thức** — chỉ gộp các run cùng `data_augmentation`, `epochs`,
`patience`, `lamda` — và cảnh báo khi số seed không đều giữa các ô. Đọc `runs/` bằng tay là
cách đã tạo ra hai kết luận sai trong quá trình làm.
