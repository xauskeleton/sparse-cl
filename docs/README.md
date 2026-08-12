# Tài liệu workspace `CL` (Continual Learning)

Workspace này chứa **hai repo nghiên cứu độc lập** về Continual Learning (học liên tục), không dùng chung code:

| Thư mục | Paper | Hướng tiếp cận | Có huấn luyện backbone? |
|---|---|---|---|
| [`EWC-DR/`](./EWC-DR.md) | *Elastic Weight Consolidation Done Right for Continual Learning* (CVPR 2026) | Regularization-based, train from scratch | Có (SGD, 180–200 epoch/task) |
| [`Fly-CL-main/`](./Fly-CL.md) | *Fly-CL: A Fly-Inspired Framework...* (ICLR 2026) | Pre-trained model + closed-form ridge regression | Không (backbone đóng băng) |

## Bản đồ tài liệu

- **[EWC-DR.md](./EWC-DR.md)** — kiến trúc, luồng chạy, từng file, config, thuật toán EWC / EWC-DR.
- **[Fly-CL.md](./Fly-CL.md)** — kiến trúc, thuật toán, tham số, cách chạy.
- **[so-sanh-va-luu-y.md](./so-sanh-va-luu-y.md)** — so sánh hai phương pháp + **danh sách lỗi/cạm bẫy đã phát hiện trong code** (đọc phần này trước khi chạy).

## Các file `.md` sẵn có trong repo

| File | Nội dung |
|---|---|
| `EWC-DR/README.md` | README gốc của tác giả: yêu cầu môi trường, cách tải dataset, lệnh training, citation. |
| `Fly-CL-main/readme.md` | README gốc: abstract, setup conda, tải pretrained model, lệnh chạy script, citation, liên hệ. |

Tài liệu trong `docs/` **bổ sung** chứ không thay thế hai README trên: README gốc nói *chạy thế nào*, `docs/` giải thích *code hoạt động ra sao và chỗ nào cần sửa*.

## Cấu trúc cây thư mục

```
CL/
├── docs/                       <- tài liệu này
├── EWC-DR/
│   ├── main.py                 <- entry point: đọc JSON config -> train()
│   ├── trainer.py              <- vòng lặp qua các task, logging, lưu kết quả
│   ├── models/                 <- các thuật toán CL (BaseLearner, EWC, EWCDR, Finetune)
│   ├── convs/                  <- các backbone CNN (ResNet các biến thể)
│   ├── utils/                  <- data manager, network wrapper, tiện ích
│   ├── exps/                   <- 10 file JSON cấu hình thí nghiệm
│   ├── *.ipynb                 <- notebook chạy trên Kaggle
│   └── data/                   <- dataset (cifar-100-python.tar.gz đã có sẵn)
└── Fly-CL-main/
    ├── main.py                 <- toàn bộ thuật toán nằm ở đây
    ├── utils.py                <- seed, trích xuất đặc trưng, one-hot
    ├── models/load_model.py    <- nạp ViT / ResNet-50 pretrained qua timm
    ├── datasets/load_dataset.py<- chia dataset thành các task
    ├── scripts/                <- 3 script bash chạy CIFAR / CUB / VTAB
    └── pretrained_model/       <- script tải checkpoint ViT
```
