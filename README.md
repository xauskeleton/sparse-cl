# CL — Continual Learning

Nghiên cứu về class-incremental learning trên backbone pretrained đóng băng, xoay quanh
**Fly-CL** (chiếu ngẫu nhiên thưa + top-k + ridge nghiệm đóng).

```
docs/          tài liệu và báo cáo   ← đọc từ đây
sparse-cl/     code của mình
upstream/      ba repo của người khác, giữ nguyên trạng để đối chứng
data/          dataset (không track)
```

## Đọc từ đâu

| | |
|---|---|
| [`docs/README.md`](docs/README.md) | bản đồ tài liệu |
| [`docs/bao-cao.md`](docs/bao-cao.md) | **báo cáo kết quả tổng thể** |
| [`docs/15-22 report.md`](docs/15-22%20report.md) | báo cáo tuần 15–22/08 — mới nhất |
| [`docs/so-sanh-va-luu-y.md`](docs/so-sanh-va-luu-y.md) | cạm bẫy đã phát hiện trong code — **đọc trước khi chạy** |
| [`sparse-cl/README.md`](sparse-cl/README.md) | cách chạy code của mình |

Báo cáo viết theo tuần, mỗi tuần một file trong `docs/`.

## Kết quả hiện tại

CIFAR-100, 10 task, ResNet-50 `tv2_in1k`, exemplar-free.

| | Ā | Δ so với bản tái lập |
|---|---:|---:|
| Fly-CL công bố | 84.61 ± 0.16 | |
| Fly-CL, bản tái lập | 84.19 ± 0.42 | — |
| `concat` stage 3 + stage 4 | 85.56 ± 0.29 | +1.37 ± 0.18 |
| `concat` + ensemble 5 nhánh | 86.49 ± 0.40 | +2.30 ± 0.07 |
| + First-Session Adaptation | **89.05** | +4.97 *(1 seed, đổi giao thức)* |

Dòng cuối dùng nhãn của task 0 để thích nghi backbone, nên **khác giao thức** với Fly-CL —
xem mục 9 của báo cáo tuần.

## `upstream/`

Ba repo giữ nguyên trạng, không sửa, chỉ dùng để đối chứng và trích dẫn:

| | Paper |
|---|---|
| `upstream/Fly-CL-main/` | *Fly-CL: A Fly-Inspired Framework…* — ICLR 2026 |
| `upstream/AnaCP/` | *AnaCP: Toward Upper-Bound Continual Learning…* — NeurIPS 2025 Spotlight |
| `upstream/EWC-DR/` | *Elastic Weight Consolidation Done Right…* — CVPR 2026 |

`EWC-DR/` có `.git` riêng nên không track ở đây (xem `.gitignore`).
