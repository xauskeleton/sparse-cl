# Fly-CL — Tài liệu code

> **Paper**: *Fly-CL: A Fly-Inspired Framework for Enhancing Efficient Decorrelation and Reduced Training Time in Pre-trained Model-based Continual Representation Learning* (Zou, Zang, Xu, Ji — ICLR 2026)
> **PDF trong repo**: `12717_Fly_CL_A_Fly_Inspired_Fr.pdf`
> **Bài toán**: Class-Incremental Learning trên backbone pretrained **đóng băng hoàn toàn**.

---

## 1. Ý tưởng thuật toán

Fly-CL **không huấn luyện mạng nơ-ron nào**. Không có optimizer, không có backward pass, không có epoch. Toàn bộ "học" là giải một bài toán bình phương tối thiểu có regularization ở dạng đóng.

Cảm hứng sinh học: mạch khứu giác của ruồi giấm chiếu ~50 nơ-ron nhận cảm sang ~2000 tế bào Kenyon qua kết nối **thưa và ngẫu nhiên**, rồi một nơ-ron ức chế chỉ giữ lại ~5% tế bào hoạt động mạnh nhất (winner-take-all). Mã hóa thưa chiều cao này làm giảm tương quan giữa các đặc trưng — chính là thứ chữa được **đa cộng tuyến** (multicollinearity) khiến ridge regression trên đặc trưng ViT thô hoạt động kém.

### Ba bước

**Bước 1 — Trích xuất đặc trưng (đóng băng).**
```python
train_embeddings, train_labels = feature_extract(pretrained_model, train_loader[task], device)
# shape: [N, 768] với ViT-B/16
```

**Bước 2 — Chiếu ngẫu nhiên thưa + top-k winner-take-all.**
```python
# main.py:81  ma trận chiếu dựng MỘT LẦN, dùng chung cho mọi task
projection_matrix = torch.zeros(args.expand_dim, args.embedding_dim)   # 10000 x 768
for row in range(args.expand_dim):
    selected_cols = torch.randperm(args.embedding_dim)[:synaptic_degree]  # 300 cột ngẫu nhiên
    projection_matrix[row, selected_cols] = torch.randn(synaptic_degree)
projection_matrix = projection_matrix.to(device).to_sparse_csc()

# main.py:103  chiếu rồi giữ top-k
train_embeddings = torch.sparse.mm(projection_matrix, train_embeddings.T)   # [10000, N]
values, indices = train_embeddings.topk(int(expand_dim * coding_level), dim=0)
output = torch.zeros_like(train_embeddings)
output.scatter_(0, indices, values)     # phần còn lại = 0
train_embeddings = output
```
`coding_level = 0.3` → giữ 3000/10000 chiều mỗi mẫu. Đây chính là "Kenyon cell + APL inhibition".

**Bước 3 — Ridge regression đệ quy (chống quên tuyệt đối).**
```python
Y = target2onehot(train_labels, num_classes)         # [N, C]
Q = Q + train_embeddings @ Y                          # [D, C]  <- tích lũy
G = G + train_embeddings @ train_embeddings.T         # [D, D]  <- tích lũy
ridge = select_ridge_parameter(train_embeddings.T, Y, ridge_lower, ridge_upper)
L  = torch.linalg.cholesky(G + ridge * I)             # nhanh hơn ~40% so với solve trực tiếp
Wo = torch.cholesky_solve(Q, L)                       # [D, C]
```

`G` và `Q` là **thống kê đủ** (sufficient statistics) của toàn bộ dữ liệu đã thấy. Vì chúng cộng dồn qua các task, nghiệm `Wo` ở task `t` **đồng nhất** với nghiệm khi huấn luyện một lần trên toàn bộ dữ liệu task 1..t. Do đó phần classifier **không quên gì cả** — mọi sụt giảm độ chính xác chỉ đến từ việc bài toán phân loại ngày càng khó (nhiều lớp hơn) và từ chất lượng đặc trưng của backbone.

### Chọn hệ số ridge bằng GCV

`select_ridge_parameter` (`main.py:43`) dùng Generalized Cross-Validation trên phân rã SVD, quét `ridge ∈ {10^4 … 10^9}` (mặc định; script dùng `10^6 … 10^9`):

```python
U, S, Vh = torch.linalg.svd(X, full_matrices=False)
diag = S² / (S² + ridge)          # hệ số thu nhỏ
df   = diag.sum()                 # bậc tự do hiệu dụng
gcv  = (‖Y − Ŷ‖² / n) / (1 − df/n)²
```
Chọn `ridge` cho `gcv` nhỏ nhất. Được tính lại mỗi task, **chỉ trên dữ liệu task hiện tại**.

### Suy luận

```python
test_embeddings = topk(projection @ test_features)   # cùng phép biến đổi
output = test_embeddings @ Wo                        # [N, C]
predicts = output.argmax(dim=1)
```

---

## 2. Mô tả từng file

### `main.py` (171 dòng) — toàn bộ thuật toán

| Phần | Dòng | Nội dung |
|---|---|---|
| `get_parser()` | 16–40 | Khai báo 15 tham số dòng lệnh. |
| `select_ridge_parameter()` | 43–61 | GCV chọn hệ số ridge. |
| Khởi tạo | 64–93 | Chọn device, set seed, nạp dataset + model, dựng ma trận chiếu, khởi tạo `Q`/`G` bằng 0. |
| Vòng lặp task | 95–128 | Trích đặc trưng → chiếu+topk → cập nhật `Q`,`G` → Cholesky → **đánh giá lại toàn bộ sub-task từ 0 đến task hiện tại**. |
| Báo cáo | 130–172 | In ma trận độ chính xác, Average Accuracy `A_t`, Accumulated Accuracy (`mean(A_t)`), thời gian huấn luyện và thời gian trích đặc trưng. |

### `utils.py` (38 dòng)

| Hàm | Nội dung |
|---|---|
| `random_initialization(seed)` | Set seed cho `torch`, `cuda`, `numpy`, `random`; bật `cudnn.deterministic`; **tắt `cudnn.enabled`** (đảm bảo tái lập, nhưng làm forward pass chậm hẳn). |
| `get_parameters(model)` | Trả các tham số `requires_grad` — không được gọi ở đâu. |
| `feature_extract(model, loader, device)` | Vòng lặp `@torch.no_grad()` gom embedding + nhãn, có thanh `tqdm`. |
| `target2onehot(targets, n_classes)` | One-hot bằng `scatter_`. |

### `models/load_model.py` (26 dòng)

| Nhánh | Hành vi |
|---|---|
| `vit_base_patch16_224` | `timm.create_model(..., pretrained=True, num_classes=0)` — tự tải trọng số từ HuggingFace Hub. `num_classes=0` → model trả feature 768 chiều thay vì logits. |
| `resnet-50` | Nạp từ file cục bộ `./pretrained_model/resnet50-11ad3fa6.pth`, xóa key chứa `"classifier"`, nạp lại vào model `num_classes=0`. **File .pth này không có trong repo và `download.sh` không tải nó.** |

### `datasets/load_dataset.py` (105 dòng)

| Thành phần | Nội dung |
|---|---|
| `CustomDataset` | `Dataset` đọc ảnh theo đường dẫn. **Được định nghĩa nhưng không dùng** — cả 3 dataset đều đi qua `datasets.CIFAR100` / `datasets.ImageFolder`. |
| `build_transform(is_cifar, data_augmentation)` | Resize→CenterCrop 224→ToTensor→Normalize. CIFAR resize thẳng lên 224; các dataset khác resize lên 256 rồi crop 224. `data_augmentation`: `None` (không normalize), `"resnet"` (mean/std ImageNet), `"vit"` (0.5/0.5). |
| `load_dataset(args)` | Nạp full dataset, **xáo thứ tự lớp** bằng `random.sample`, chia đều thành `num_tasks` nhóm, tạo `Subset` + `DataLoader` cho từng task. Trả 2 dict `{task_idx: DataLoader}`. |

Lưu ý: `train_transform` và `test_transform` được xây **giống hệt nhau** — không có augmentation ngẫu nhiên nào, hợp lý vì backbone đóng băng và mỗi mẫu chỉ được nhìn một lần.

Đường dẫn dataset mong đợi (`--root`, mặc định `../data`):
```
../data/cifar-100-python/       (tự tải)
../data/cub/train/  ../data/cub/test/
../data/vtab/train/ ../data/vtab/test/
```

### `scripts/`

Cả 3 script đều `cd ..` rồi gọi `main.py`:

| Script | dataset | num_classes | num_tasks | seed | gpu |
|---|---|---|---|---|---|
| `test_cifar.sh` | CIFAR-100 | 100 | 10 | 1993 | **5** |
| `test_cub.sh` | CUB-200-2011 | 200 | 10 | 2023 | **1** |
| `test_vtab.sh` | VTAB | 50 | 5 | 2023 | **6** |

Tham số chung: `--model_name vit_base_patch16_224 --embedding_dim 768 --expand_dim 10000 --synaptic_degree 300 --coding_level 0.3 --batch_size 128 --data_augmentation vit --ridge_lower 6 --ridge_upper 10`.

> ⚠️ Chỉ số GPU (5, 1, 6) là của máy tác giả. Trên máy 1 GPU phải sửa thành `--gpu 0`.
> ⚠️ Giá trị trong script **khác mặc định trong `main.py`** ở 3 chỗ: `synaptic_degree` 300 vs 100, `coding_level` 0.3 vs 0.01, `ridge_lower` 6 vs 4. Dùng script, đừng dùng mặc định.

### `pretrained_model/download.sh`

Tải checkpoint ViT-B/16 (bản AugReg, pretrain ImageNet-21k) từ Google Storage và đổi tên thành `vit_base_patch16_224_in21k.npz`. **Nhưng `load_model.py` không đọc file này** — nó gọi `timm.create_model(pretrained=True)` tải trọng số riêng. File `.npz` hiện là dư thừa trừ khi bạn sửa `load_model.py` để dùng `checkpoint_path`.

### Các file khác

| File | Nội dung |
|---|---|
| `log_cifar_seed1993.txt` | Kết quả chạy CIFAR-100 seed 1993 (UTF-16, xem mục 3). |
| `assets/Fig1.png` | Hình kiến trúc trong README. |
| `2.7` | **Rác** — output của `conda search pytorch` bị redirect nhầm vào file tên `2.7` (67 KB). Có thể xóa. |
| `.gitignore` | `__pycache__/`, `.vscode/`, `datasets/data/`, `.npz`. |
| `LICENSE` | Giấy phép của repo. |

---

## 3. Kết quả tham chiếu (`log_cifar_seed1993.txt`)

CIFAR-100, 10 task, ViT-B/16:

| Sau task | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| Average Accuracy | 97.70 | 97.25 | 95.53 | 94.40 | 93.00 | 92.40 | 91.80 | 90.04 | 89.10 | **88.68** |

- **Accumulated Accuracy**: 92.99
- **Thời gian huấn luyện trung bình**: 53.86 s/task — trong đó **43.92 s là trích đặc trưng**, tức phần "học" thực sự (chiếu + Cholesky) chỉ ~10 s.
- Đọc ma trận độ chính xác trong log: hàng `i` = task `i`, cột `j` = độ chính xác của task `i` sau khi đã học đến task `j`. Ví dụ task 0 đi từ 97.7 → 89.1 sau 10 task: mức suy giảm ~8.6 điểm này **không phải quên trọng số** (classifier là nghiệm chính xác trên toàn dữ liệu) mà là do phải phân biệt 100 lớp thay vì 10.

File được lưu ở **UTF-16 LE**, nên các công cụ đọc UTF-8 sẽ thấy ký tự xen null. Đọc bằng:
```python
open('log_cifar_seed1993.txt', encoding='utf-16').read()
```

---

## 4. Chạy

```bash
conda create -n FlyCL python=3.9
conda activate FlyCL
conda install pytorch==1.13.1 torchvision==0.14.1 pytorch-cuda=11.7 -c pytorch -c nvidia
conda install "numpy<2.0.0" && conda install timm==0.9.16 tqdm scipy

cd upstream/Fly-CL-main/scripts
# sửa --gpu 5 -> --gpu 0 trước khi chạy
./test_cifar.sh
```

Hoặc gọi trực tiếp:
```bash
cd Fly-CL-main
python main.py --dataset CIFAR-100 --num_classes 100 --num_tasks 10 \
  --model_name vit_base_patch16_224 --embedding_dim 768 \
  --expand_dim 10000 --synaptic_degree 300 --coding_level 0.3 \
  --seed 1993 --batch_size 128 --gpu 0 --data_augmentation vit \
  --ridge_lower 6 --ridge_upper 10 --root ../data
```

**Yêu cầu bộ nhớ**: `G` có kích thước `expand_dim²` = 10000² float32 ≈ **400 MB trên GPU**, cộng thêm bản sao `G + ridge*I` và thừa số Cholesky khi giải → dự trù ~1.5 GB chỉ cho phần này. Tăng `expand_dim` lên 20000 sẽ nhân 4 con số đó.
