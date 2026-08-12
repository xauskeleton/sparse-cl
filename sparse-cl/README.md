# sparse-cl

Model riêng. **Không phụ thuộc và không sửa gì trong `EWC-DR/` hay `Fly-CL-main/`** — hai thư mục đó giữ nguyên trạng để đối chứng.

## Kiến trúc

```
ảnh → backbone (đóng băng)  → feat        [B, 768]
    → chiếu thưa            → expanded    [B, expand_dim]
    → top-k winner-take-all → sparse code [B, expand_dim]
    → (tuỳ chọn) MLP
    → classifier            → logits      [B, C]
```

Train end-to-end bằng cross-entropy. Hai cơ chế chống quên:
1. **Sparsity** — mỗi lớp chiếm một tập con khác nhau của không gian ẩn, gradient chỉ chảy vào các unit thắng nên nhiễu loạn giữa các task giảm.
2. **EWC-DR** — hình phạt bậc hai kiểm soát mức trôi của các tham số học liên tục.

## File

| File | Nội dung |
|---|---|
| `config.py` | 38 flag + `validate()` chặn các tổ hợp mâu thuẫn |
| `model.py` | `SparseProjection`, `TopK`, `IncrementalLinear`, `SparseExpandCL`, `Regularizer` |
| `data.py` | Trích feature 1 lần → cache → cắt theo task (đúng cách chia của Fly-CL) |
| `train.py` | Vòng lặp task, early stopping, EWC-DR, chỉ số + chẩn đoán |
| `run.sh` | Ba cấu hình đầu tiên, chạy trên server |
| `kaggle_run.ipynb` | Bản Kaggle: cache feature → chạy 3 cấu hình → bảng kết quả |

## Giai đoạn 1 — ba cấu hình đầu, **chưa dùng EWC**

Cả ba chạy với `--cl_reg none`. Hiệu số giữa chúng khi đó đo đúng một thứ: **sparsity tự nó chống quên được bao nhiêu**. Bật regularizer ngay sẽ làm lẫn hai nguồn đóng góp.

| # | Kiến trúc | Học cái gì |
|---|---|---|
| 1 | `(backbone → chiếu thưa)` ❄ → MLP(ReLU) 🔥 | chỉ MLP |
| 2 | backbone ❄ → chiếu thưa 🔥 → MLP(ReLU) 🔥 | chiếu + MLP |
| 3 | backbone ❄ → chiếu thưa 🔥 → cls tuyến tính 🔥 | chiếu |

```bash
bash run.sh                 # cả ba
SEED=2023 GPU=1 bash run.sh # đổi seed / gpu
```

Hoặc từng cái:

```bash
# 1
python train.py --cl_reg none --train_projection False --projection_schedule task0 \
                --use_mlp True --mlp_act relu
# 2
python train.py --cl_reg none --train_projection True --projection_schedule continual \
                --use_mlp True --mlp_act relu
# 3
python train.py --cl_reg none --train_projection True --projection_schedule continual \
                --use_mlp False
```

`2 ↔ 3` cho biết MLP mang lại gì (phép chiếu đều được học).

**Lưu ý về `1 ↔ 3`**: hai cấu hình này khác nhau *hai* thứ cùng lúc — chiếu đóng băng/học được, **và** có/không MLP. Muốn tách bạch thì chạy thêm cấu hình thứ tư:

```bash
# 0 — chiếu đóng băng + cls tuyến tính (không MLP)
python train.py --cl_reg none --train_projection False --projection_schedule task0 \
                --use_mlp False
```

Khi đó `0 ↔ 3` cô lập đúng ảnh hưởng của việc học phép chiếu, `0 ↔ 1` cô lập ảnh hưởng của MLP.

## Giai đoạn 2 — bật EWC-DR

Chỉ chạy sau khi có số liệu giai đoạn 1, trên cấu hình tốt nhất, để đo phần tăng thêm:

```bash
python train.py --train_projection True --projection_schedule continual \
                --use_mlp False --cl_reg ewc_dr --lamda 10000
```

## Mốc so sánh

Fly-CL chạy trên cùng máy, cùng seed 1993 (`Fly-CL-main/log_cifar_seed1993.txt`):

| | |
|---|---|
| `A_T` (sau task cuối) | **88.68** |
| `Ā` (accumulated) | **92.99** |

Dùng số này, **không** dùng 93.89 trong paper — cùng môi trường mới so sánh được.

Ba thứ phải khớp để phép so sánh hợp lệ, và `data.py` đã đảm bảo cả ba: cách chia task (`random.sample` sau khi seed, y hệt Fly-CL), định nghĩa `A_t`/`Ā`, và normalization (`--data_augmentation vit`).

## Đọc kết quả

`runs/<exp_name>.json` chứa `acc_matrix`, `metrics`, `per_task`. Ngoài accuracy, luôn nhìn ba cột chẩn đoán:

| Cột | Ý nghĩa | Ngưỡng |
|---|---|---|
| `pen_over_clf` | `penalty / loss_clf` | mục tiêu **0.1–1**. Nhỏ hơn → regularizer vô hình. Lớn hơn → mạng bị đóng băng |
| `omega_saturated` | tỉ lệ `ω` chạm trần `omegamax` | **> 0.5** → EWC-DR đã thoái hoá thành L2 thuần, hạ `--omegamax` và nâng `--lamda` |
| `dead_frac` | tỉ lệ unit không bao giờ thắng top-k | tăng dần → bật `--adaptive_threshold` / `--load_balance_coef` |

Accuracy **không** cho biết ba vấn đề này — model vẫn chạy bình thường trong cả ba trường hợp.

## Tổ hợp flag bị chặn cứng

- `cache_features=True` + `freeze_backbone=False` → feature đổi mỗi epoch, cache thành rác im lặng.
- `cl_reg != none` mà không tham số nào học liên tục → regularizer luôn bằng 0, dễ hiểu nhầm thành "EWC-DR không giúp gì".
- `coding_level >= 1.0` → không còn phi tuyến; `W_head · W_proj` sụp thành một ma trận, tầng mở rộng vô nghĩa.
- `projection_schedule=offline` mà thiếu `--offline_data`.

## Cần biết trước

- **Trích feature 1 lần** (~10 phút), mỗi run sau đó vài phút. Cache theo *dataset*, không theo task — nên quét seed / số task / cấu hình đều miễn phí. Có cache rồi thì không cần `timm`/`torchvision`, chạy được cả trên session CPU.
- **Rủi ro chính là overfit, không phải compute**: 3–8 M tham số trên ~5.000 mẫu/task, và feature đã cache nên **không augment được**.
- `--lamda 10000` lấy từ EWC-DR (ResNet from scratch). Quy mô ở đây khác hẳn — chỉnh theo `pen_over_clf` chứ đừng tin con số.
- `--coding_level 0.1` thấp hơn Fly-CL (0.3) vì nhắm vào chống quên, nhưng đó là vùng **chưa được kiểm chứng**. Phải quét.

## Việc nên làm trước khi tin bất kỳ con số nào

1. **Linear probe** trên feature đóng băng — trần trên của mọi thứ phía sau (~88–90% với ViT-B/16 + CIFAR-100). Viết riêng ~5 dòng, model này không có chế độ "không mở rộng".
2. **Chạy Fly-CL với 3 seed** để có error bar. Nếu nó dao động ±1% thì cải thiện 0.5% của bạn vô nghĩa.
3. **Quét `coding_level`** 0.02 → 0.30, vẽ accuracy **và** forgetting trên cùng trục. Fly-CL chỉ tối ưu accuracy và chốt 0.3; vế forgetting chưa ai đo — đây là chỗ có khả năng thành đóng góp thật.
