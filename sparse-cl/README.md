# sparse-cl

Model riêng. **Không phụ thuộc và không sửa gì trong `upstream/EWC-DR/` hay `upstream/Fly-CL-main/`** — hai thư mục đó giữ nguyên trạng để đối chứng.

Báo cáo kết quả đầy đủ được giữ trong workspace nghiên cứu, không nằm trong repo này.

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

## Cấu trúc

```
src/sparse_cl/            thu vien, cai bang `pip install -e .`
  config.py               toan bo flag + validate() + _auto_name()
  data.py                 TaskData (feature cache) / ImageTaskData (anh tho)
  backbones.py            load_backbone, cache_exists
  ridge.py                select_ridge_parameter, topk_rows - sao NGUYEN VAN tu Fly-CL
  model.py                SparseExpandCL, Regularizer - chi nhanh SGD dung

experiments/              moi file la mot thi nghiem chay duoc, khong import lan nhau
  flycl.py                Fly-CL nghiem dong;  --grid 0:1 la moc goc,
                          300:1 la noi stage 3 + stage 4;  --mode ens de ensemble
  fsa.py                  First-Session Adaptation - sinh cache feature moi
  anacp_cp.py             tang Contrastive Projection cua AnaCP, 4 vi tri dat
  anacp_full.py           AnaCP day du: 2 tang ridge + pseudo-replay
  anacp_reference.py      chay code NGUYEN BAN cua AnaCP tren feature cua ta
  train_sgd.py            nhanh SGD: projection hoc duoc, MLP head, EWC-DR
  run_grid.py             chay luoi cau hinh/seed/lambda cho nhanh SGD
  make_table.py           dung bang tu runs/, LOC THEO GIAO THUC

archive/                  cac bien the da bi bac bo, giu de tra cuu
notebooks/                backbone_run.ipynb (ban Kaggle)
cache/ runs/ logs/ data/  sinh ra khi chay, deu trong .gitignore
```

`src/sparse_cl/ridge.py` giữ hai hàm `select_ridge_parameter` và `topk_rows`, sao **nguyên
văn** từ `main.py` của Fly-CL. Nhờ đó không file nào trong `experiments/` phải import lẫn nhau.

Không có file baseline riêng: `flycl.py --grid 0:1` (tức `deg_s3 = 0`) tái lập **chính
xác** Fly-CL gốc — cùng thứ tự tiêu thụ RNG, cùng kết quả đến từng chữ số.

Hai pipeline độc lập trong repo này: **nghiệm đóng** (`flycl`, `fsa`, `anacp_*`) là nơi có
mọi kết quả dương, và **SGD** (`train_sgd`, `run_grid`, `make_table`, `model.py`) là hướng
ban đầu — kết quả toàn âm, giữ lại để đối chứng.

## Cài

```bash
pip install -r requirements.txt
pip install -e .          # cho `import sparse_cl` chay duoc tu bat ky dau
```

## Chạy

```bash
# cai tien chinh: noi stage 3 + stage 4  (grid 0:1 la doi chung khong dung stage 3)
python experiments/flycl.py --model_name resnet50.tv2_in1k        --coding_level 0.3 --grid 0:1,300:1

# + ensemble 5 nhanh
python experiments/flycl.py --model_name resnet50.tv2_in1k        --coding_level 0.3 --mode ens --branches 5 --grid 300:1

# First-Session Adaptation roi chay lai tren feature da adapt
python experiments/fsa.py  --seed 1993 --epochs 10
python experiments/flycl.py --model_name resnet50.tv2_in1k+fsa1993+ms        --coding_level 0.3 --grid 0:1,300:1 --ridge_lower 5


# nhanh SGD
python experiments/run_grid.py --backbone resnet --configs all        --regs none,ewc_dr --seeds 1993,2023,2025
python experiments/make_table.py --backbone resnet
```

`--ridge_lower 5` là **bắt buộc** trên feature đã FSA: ở sàn 3, GCV rơi vào một cực tiểu
giả ở task 3 (`residual = 0`, `df/n = 0.9922` → GCV = 0/0) và Cholesky sập.

Các file trong `archive/` chạy bằng module: `python -m archive.flycl_moe --help`.

`anacp_reference.py` cần repo AnaCP của tác giả:

```bash
git clone https://github.com/SalehMomeni/AnaCP
python experiments/anacp_reference.py --anacp_path ./AnaCP
```

Lần chạy đầu tự tải CIFAR-100 vào `./data` và trọng số backbone, rồi cache feature
(~5 phút). Sau đó mỗi run mất 23 giây (ViT) hoặc 47–100 giây (ResNet).

### Giao thức chuẩn của Fly-CL

Ba tham số này phải đúng, sai cái nào cũng làm số ResNet sụt nhiều điểm mà log không báo gì:

| Tham số | Giá trị | Vì sao dễ sai |
|---|---|---|
| `--ridge_lower` | **4** | Mặc định của `main.py`. `scripts/test_cifar.sh` ghi 6 nhưng script đó chỉ dành cho ViT; repo không có script ResNet. Đặt 6 làm ResNet mất **7 điểm Ā**. |
| `--coding_level` | **0.3** | `main.py` mặc định 0.01, script CIFAR ghi đè thành 0.3. |
| Checkpoint ResNet | **`resnet50.tv2_in1k`** | Tương đương `resnet50-11ad3fa6.pth` họ nạp. Khác `resnet50` mặc định của timm (`a1_in1k`). |

```bash
# tai lap dung so cua paper: A_T 76.99, A_bar 84.08 (paper: 84.61 +- 0.16)
python experiments/flycl.py --model_name resnet50.tv2_in1k --grid 0:1 \
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

Mỗi file chạy độc lập và tái lập đúng
baseline ở cấu hình trung tính (m=1, s4, khối=1) làm phép tự kiểm tra.

| File (trong `archive/`) | Thử gì | Kết quả |
|---|---|---|
| `flycl_improved.py` | GCV trên dữ liệu tích luỹ; top-k theo trị tuyệt đối | +0.00 / −0.29 |
| `flycl_multistage.py` | Ghép feature nhiều stage (cấp phát kết nối sai — xem `experiments/flycl.py`) | +0.16 |
| `flycl_moe.py` | Hỗn hợp chuyên gia, cổng đóng băng | −5.93 (m=8) |
| `flycl_blocktopk.py` | Top-k theo khối thay vì toàn cục | −0.06 |
| `flycl_ensemble.py` | m phép chiếu độc lập, cộng logit | +0.68 (m=10) |
| `flycl_deep.py`, `flycl_pertask.py` | tầng sâu hơn; n ma trận cho n task | −, −23.71 |

Hai kết quả dương không nằm trong các file trên, và chúng là nội dung chính của
`experiments/flycl.py` và `experiments/fsa.py`:

| | Δ Ā | Ghi chú |
|---|---:|---|
| `concat` stage 3 + stage 4 | **+1.37 ± 0.18** | 3 seed; `+ ensemble 5` cho +2.30 ± 0.07 |
| **First-Session Adaptation** | **+3.77** | 1 seed; cộng cả hai được **89.05 Ā** |

Ngoài ra `--expand_dim 20000` cho +1.05 ± 0.04 ở `coding_level` 0.1 nhưng **−1.38** ở
0.3 — hai tham số này không độc lập.

### Cạm bẫy khi viết script mới dùng chung ma trận chiếu

```python
# SAI - Python danh gia ve PHAI truoc, nen randn chay truoc randperm
W[r, torch.randperm(d)[:n]] = torch.randn(n)

# DUNG - giong Fly-CL goc
pick = torch.randperm(d)[:n]
W[r, pick] = torch.randn(n)
```

Viết gộp một dòng thì thứ tự tiêu thụ RNG bị đảo và ma trận chiếu **khác hẳn**
dù cùng `--seed`. Không có lỗi nào được báo; chỉ lộ ra khi so hai script lẽ ra
phải cho cùng kết quả. Đã mất một lần đo vì lỗi này.
