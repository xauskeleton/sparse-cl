# sparse-cl

Model riêng. **Không phụ thuộc và không sửa gì trong `EWC-DR/` hay `Fly-CL-main/`** — hai thư mục đó giữ nguyên trạng để đối chứng.

Kết quả đầy đủ: [`../docs/bao-cao.md`](../docs/bao-cao.md).

## Kiến trúc

```
ảnh → backbone (đóng băng hoặc fine-tune) → feat        [B, D]
    → sparse projection                    → expanded    [B, expand_dim]
    → top-k winner-take-all                → sparse code [B, expand_dim]
    → (tuỳ chọn) MLP
    → classifier                           → logits      [B, C]
```

Train end-to-end bằng cross-entropy. Hai cơ chế chống quên: **sparsity** (mỗi lớp
chiếm một tập con khác nhau của không gian ẩn) và **EWC-DR** (hình phạt bậc hai
lên các tham số học liên tục).

## File

| File | Nội dung |
|---|---|
| `config.py` | Toàn bộ flag + `validate()` chặn các tổ hợp mâu thuẫn + `_auto_name()` sinh tên run |
| `model.py` | `SparseProjection`, `TopK`, `IncrementalLinear`, `SparseExpandCL`, `Regularizer` |
| `data.py` | `TaskData` (feature đã cache) và `ImageTaskData` (ảnh thô, cho fine-tune) |
| `train.py` | Vòng lặp task, early stopping, EWC-DR, metrics + chẩn đoán |
| `run_grid.py` | Chạy nhiều cấu hình / seed / λ trong một lệnh, ghi log gộp |
| `make_table.py` | Dựng bảng kết quả từ `runs/`, **lọc theo giao thức** |
| `flycl_baseline.py` | Thuật toán Fly-CL (closed-form ridge) chạy trên feature của ta — **bản đối chứng, không sửa** |
| `flycl_*.py` (improved, multistage, moe, blocktopk, ensemble) | Các biến thể đề xuất cho Fly-CL — xem mục cuối |
| `backbone_run.ipynb` | Bản Kaggle: clone repo → chạy lưới → bảng + zip kết quả |

## Chạy

```bash
# một cấu hình
python train.py --model_name resnet50 --data_augmentation resnet \
                --train_projection False --projection_schedule task0 --use_mlp False

# cả lưới 6 cấu hình × {không reg, EWC-DR}, backbone đóng băng, 3 seed
python run_grid.py --backbone resnet --configs all --regs none,ewc_dr --seeds 1993,2023,2025

# fine-tune backbone
python run_grid.py --backbone resnet --configs 1,3,5 --freeze_backbone False --epochs 100

# bảng kết quả
python make_table.py --backbone resnet

# mốc so sánh Fly-CL trên chính checkpoint của ta
python flycl_baseline.py --model_name resnet50 --data_augmentation resnet --coding_level 0.1
```

### Giao thức chuẩn của Fly-CL

Ba tham số này phải đúng, sai cái nào cũng làm số ResNet sụt nhiều điểm mà log không báo gì:

| Tham số | Giá trị | Vì sao dễ sai |
|---|---|---|
| `--ridge_lower` | **4** | Mặc định của `main.py`. `scripts/test_cifar.sh` ghi 6 nhưng script đó chỉ dành cho ViT; repo không có script ResNet. Đặt 6 làm ResNet mất **7 điểm Ā**. |
| `--coding_level` | **0.3** | `main.py` mặc định 0.01, script CIFAR ghi đè thành 0.3. |
| Checkpoint ResNet | **`resnet50.tv2_in1k`** | Tương đương `resnet50-11ad3fa6.pth` họ nạp. Khác `resnet50` mặc định của timm (`a1_in1k`). |

```bash
# tai lap dung so cua paper: A_T 76.99, A_bar 84.08 (paper: 84.61 +- 0.16)
python flycl_baseline.py --model_name resnet50.tv2_in1k --data_augmentation resnet \
                         --coding_level 0.3 --ridge_lower 4 --ridge_upper 10
```

Nhưng để **so với các bảng ở đây** thì phải dùng `--model_name resnet50` (checkpoint
`a1_in1k`, giống mọi run khác), nếu không là so hai backbone khác nhau.

Sáu cấu hình trong `run_grid.py`:

| # | Projection | Head |
|:-:|---|---|
| 1 | none | Linear |
| 2 | none | MLP |
| 3 | frozen | Linear |
| 4 | frozen | MLP |
| 5 | learnable | Linear |
| 6 | learnable | MLP |

Cấu hình 1 và 3 không có tham số nào học liên tục ở chế độ đóng băng, nên
`run_grid.py` tự bỏ qua tổ hợp `1/3 + EWC` — hình phạt sẽ luôn bằng 0.

## Thư mục sinh ra khi chạy

| | |
|---|---|
| `cache/` | Feature đã trích, khoá theo `{dataset}_{model_name}_{data_augmentation}` |
| `runs/` | Một JSON mỗi run: `acc_matrix`, `metrics`, `per_task`, `args` |
| `runs/_invalid/` | Run chạy lệch giao thức, giữ lại để tra cứu, `make_table.py` không đọc |
| `logs/` | stdout của `run_grid.py` |

Cả bốn đều nằm trong `.gitignore`.

## Đọc kết quả

Ngoài accuracy, luôn nhìn các cột chẩn đoán trong `runs/<name>.json`:

| Cột | Ý nghĩa |
|---|---|
| `pen_over_clf` | `penalty / loss_clf`. **Không có ngưỡng cố định** — λ tối ưu đo được nằm ở `pen/clf` ≈ 0.96–2.38 trên ViT, phải quét chứ đừng nhắm vào một khoảng |
| `omega_saturated` | Tỉ lệ `ω` chạm trần `omegamax`. Đang ở 0.52–0.64, tức EWC-DR đã thoái hoá một phần thành L2 thuần |
| `proj_drift` | `‖W − W₀‖ / ‖W₀‖` của sparse projection |
| `dead_frac` | Tỉ lệ unit không bao giờ thắng top-k |

Accuracy **không** phản ánh ba vấn đề đầu — model vẫn chạy bình thường trong cả ba trường hợp.

## Cạm bẫy đã gặp

- **Cache khoá theo `data_augmentation`.** Đổi normalization sinh file cache khác
  một cách im lặng. 15 run ResNet đầu tiên chạy nhầm `--data_augmentation vit` và
  phải bỏ (nay ở `runs/_invalid/`).
- **Resize phải khớp PIL**, không phải bicubic của PyTorch. `data.py` tái lập
  đúng kernel Keys `a = −0.5`, hai lượt riêng biệt, làm tròn uint8 sau mỗi lượt.
  Lệch thì cosine giữa feature GPU và feature PIL chỉ còn 0.981 và mọi số
  fine-tune vô nghĩa.
- **bf16 cần SM ≥ 8.0.** `torch.cuda.is_bf16_supported()` trả True trên T4 nhờ
  emulation, chậm hơn ~8 lần. `train.py` kiểm tra compute capability trực tiếp.
- **Backbone đóng băng phải ở `eval()` kể cả trong `model.train()`**, nếu không
  BatchNorm của ResNet vẫn cập nhật running stats và feature trôi âm thầm.
- **`_auto_name()` phải chứa mọi flag đã quét**, nếu không hai run khác nhau ghi
  đè lên cùng một JSON.

## Tổ hợp flag bị chặn cứng

- `cache_features=True` + `freeze_backbone=False` → feature đổi mỗi epoch,
  cache thành rác. `config.py` tự tắt cache và cảnh báo.
- `cl_reg != none` mà không tham số nào học liên tục → regularizer luôn bằng 0.
- `coding_level >= 1.0` → không còn phi tuyến, `W_head · W_proj` sụp thành một
  ma trận, tầng mở rộng vô nghĩa.
- `--image_size` khác 224 với ViT → patch embedding không khớp.

## Thử nghiệm cải tiến Fly-CL

Mỗi file chạy độc lập, đều `import` lại từ `flycl_baseline.py` và tái lập đúng
baseline ở cấu hình trung tính (m=1, s4, khối=1) làm phép tự kiểm tra.

| File | Thử gì | Kết quả |
|---|---|---|
| `flycl_improved.py` | GCV trên dữ liệu tích luỹ; top-k theo trị tuyệt đối | +0.00 / −0.29 |
| `flycl_multistage.py` | Ghép feature nhiều stage của backbone | +0.16 (`s3+s4`) |
| `flycl_moe.py` | Hỗn hợp chuyên gia, cổng đóng băng | −5.93 (m=8) |
| `flycl_blocktopk.py` | Top-k theo khối thay vì toàn cục | −0.06 |
| `flycl_ensemble.py` | m phép chiếu độc lập, cộng logit | +0.68 (m=10) |

Kết quả dương duy nhất đáng kể không nằm trong các file trên: **`--expand_dim 20000`
cho +1.05 ± 0.04** so với mặc định 10000 (3 seed, ResNet-50, CIFAR-100). Xem
`../docs/bao-cao.md`.

### Cạm bẫy khi viết script mới dùng chung ma trận chiếu

```python
# SAI - Python danh gia ve PHAI truoc, nen randn chay truoc randperm
W[r, torch.randperm(d)[:n]] = torch.randn(n)

# DUNG - giong flycl_baseline
pick = torch.randperm(d)[:n]
W[r, pick] = torch.randn(n)
```

Viết gộp một dòng thì thứ tự tiêu thụ RNG bị đảo và ma trận chiếu **khác hẳn**
dù cùng `--seed`. Không có lỗi nào được báo; chỉ lộ ra khi so hai script lẽ ra
phải cho cùng kết quả. Đã mất một lần đo vì lỗi này.
