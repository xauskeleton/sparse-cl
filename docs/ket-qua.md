# Kết quả thí nghiệm — sparse-cl

CIFAR-100, 10 tasks × 10 classes, class-incremental learning, exemplar-free.
Backbone ViT-B/16 (`augreg2_in21k_ft_in1k`) frozen, features pre-extracted.
Sparse random projection `W ∈ R^{10000×768}`, 300 non-zeros per row,
top-k activation với coding level 0.1.
AdamW lr 1e-3, tối đa 100 epochs/task, early-stop patience 20, val split 15%,
batch size 256. Không dùng regularizer. Mỗi config chạy 3 seeds (1993/2023/2025).

## Metrics

Sau mỗi task, đo average accuracy trên mọi class đã học từ đầu:

```
after task    1      2      3      4      5      6      7      8      9     10
            97.70  95.95  93.27  91.50  89.90  88.80  88.06  85.27  84.20  83.33
```

| Metric | Định nghĩa | |
|---|---|---|
| **A_T** (last-stage accuracy) | Phần tử cuối dãy — accuracy trên toàn bộ 100 classes sau khi học xong | cao hơn = tốt hơn |
| **Ā** (overall / accumulated accuracy) | Trung bình cả dãy — model tốt tới đâu xuyên suốt quá trình | cao hơn = tốt hơn |
| **Forgetting** | Task cũ mất trung bình bao nhiêu điểm so với peak accuracy của nó | thấp hơn = tốt hơn |

## Kết quả

| # | Projection | Head | A_T | Ā | Forgetting |
|:-:|---|---|---:|---:|---:|
| 1 | none | Linear | 80.35 ± 1.31 | 87.30 ± 1.32 | 9.08 ± 1.70 |
| 2 | none | MLP | 76.51 ± 1.05 | 86.28 ± 0.22 | 17.25 ± 0.87 |
| 3 | frozen | Linear | **83.17 ± 0.23** | **89.49 ± 0.97** | **7.49 ± 0.63** |
| 4 | frozen | MLP | 53.48 ± 1.40 | 74.25 ± 1.57 | 41.08 ± 3.12 |
| 5 | learnable | Linear | 59.14 ± 1.65 | 77.73 ± 2.77 | 39.55 ± 1.93 |
| 6 | learnable | MLP | 30.18 ± 0.47 | 59.61 ± 3.69 | 72.04 ± 1.18 |
| — | *Fly-CL (baseline)* | *ridge, closed-form* | *88.68* | *92.99* | *—* |

- `Projection: none` = bỏ hẳn projection và top-k, đưa feature 768-d của backbone thẳng vào head.
- `Projection: frozen` = sparse random projection, giữ nguyên từ lúc khởi tạo.
- `Projection: learnable` = cùng projection nhưng được cập nhật bằng gradient qua các task.
- `Head: MLP` = `Linear(d→512) → ReLU → Linear(512→100)`, với `d` = 768 hoặc 10000.
- Config 4, 5, 6 là ba phương án đề ra. Config 1, 2, 3 là control để tách bạch các yếu tố.

## Ablation theo cặp

| So sánh | Yếu tố được cô lập | Δ Ā |
|---|---|---:|
| 1 → 3 | thêm sparse random projection (Linear head) | **+2.19** |
| 2 → 4 | thêm sparse random projection (MLP head) | −12.03 |
| 1 → 2 | thêm MLP head (không projection) | −1.02 |
| 3 → 4 | thêm MLP head (frozen projection) | −15.24 |
| 5 → 6 | thêm MLP head (learnable projection) | −18.12 |
| 3 → 5 | cho projection học (Linear head) | −11.76 |
| 4 → 6 | cho projection học (MLP head) | −14.64 |

Ba nhận xét:

**Sparse random projection có ích, nhưng chỉ khi head là Linear** (+2.19 Ā, và
forgetting giảm 9.08 → 7.49). Đây là kết quả dương duy nhất trong bảng.

**MLP head luôn có hại, và hại nặng hơn hẳn khi đặt sau projection**
(−1.02 khi không có projection, nhưng −15.24 khi có). Nguyên nhân là số
parameters: `Linear(768→512)` có 393 K, còn `Linear(10000→512)` có 5.12 M —
gấp 13 lần trên cùng 4250 mẫu mỗi task. Nếu buộc phải dùng MLP thì bỏ hẳn
projection còn tốt hơn (86.28 so với 74.25).

**Learnable projection luôn có hại**, ở cả hai loại head (−11.76 và −14.64).

## Kết luận

Cấu hình tốt nhất là **frozen sparse random projection + Linear head** — đúng
thiết kế của Fly-CL. Cả hai ý tưởng đề ra đều làm model tệ đi, và hai tác hại
cộng dồn chứ không bù trừ.

Không phải do chọn hyperparameter: đã sweep `projection_lr` qua 3 orders of
magnitude (1e-5 đến 5e-3) và thử 3 chế độ bias cho projection layer — không
chế độ nào thắng được frozen projection.

Chênh lệch với Fly-CL nằm ở head chứ không ở representation. Ở task 1 hai bên
đạt giống hệt 97.70 (cùng backbone, cùng projection, cùng top-k); khoảng cách
chỉ mở ra từ task 2 và tăng đều tới 5.35 điểm — đúng lúc bắt đầu có forgetting.
Fly-CL dùng closed-form ridge solution, tích luỹ sufficient statistics qua các
task nên nghiệm luôn optimal trên toàn bộ dữ liệu đã thấy; config 3 dùng SGD
chỉ nhìn task hiện tại.

## Giới hạn

Baseline Fly-CL lấy từ `Fly-CL-main/log_cifar_seed1993.txt` — 1 seed, môi
trường khác, ViT checkpoint có thể khác. Muốn so chặt chẽ phải chạy lại trong
cùng môi trường với ít nhất 3 seeds.

Chênh lệch +2.19 của sparse projection chỉ khoảng 2σ — cần thêm seeds để chắc chắn.

Chỉ thử trên CIFAR-100 + ViT-B/16. Features của ViT-B/16 IN21k vốn đã rất mạnh
cho CIFAR-100 (linear probe khoảng 90%), nên có thể không còn gì để học. Kết
quả có thể khác với backbone yếu hơn (ResNet-50) hoặc dataset có domain shift
(VTAB).
