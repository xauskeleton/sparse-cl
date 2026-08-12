# EWC-DR — Tài liệu code

> **Paper**: *Elastic Weight Consolidation Done Right for Continual Learning* (Liu & Chang, CVPR 2026)
> **Nền tảng code**: fork/rút gọn từ [PyCIL](https://github.com/LAMDA-CL/PyCIL).
> **Bài toán**: Exemplar-Free Class-Incremental Learning (EFCIL) — học lần lượt từng nhóm lớp, **không lưu ảnh cũ** (`memory_size = 0`).

---

## 1. Ý tưởng thuật toán

### 1.1 EWC gốc

EWC chống quên bằng cách thêm một hình phạt kéo trọng số về gần giá trị cũ, có trọng số theo **độ quan trọng** (Fisher information):

```
L = L_CE(task mới)  +  λ · Σ_i  F_i · (θ_i − θ*_i)²  / 2
```

trong đó `F_i` là đường chéo ma trận Fisher, ước lượng bằng bình phương gradient trung bình trên tập dữ liệu của task vừa học:

```python
# models/ewc.py:234  getFisherDiagonal()
logits = self._network(inputs)["logits"]
loss = F.cross_entropy(logits, targets)
loss.backward()
fisher[n] += p.grad.pow(2)
```

**Vấn đề mà paper chỉ ra**: sau khi model đã hội tụ trên task đó, `p(y=target)` gần 1 nên gradient của cross-entropy **tiêu biến** → `F_i ≈ 0` cho hầu hết trọng số → hình phạt gần như biến mất → quên thảm khốc. Đây là *weight importance misalignment*.

### 1.2 EWC-DR — Logits Reversal (LR)

Sửa chữa nằm gọn trong **một dòng**:

```python
# models/ewc_dr.py:246  getImportance()
logits = self._network(inputs)["logits"]
logits = logits * -1          # <-- Logits Reversal
loss = F.cross_entropy(logits, targets)
loss.backward()
omega[n] += p.grad.pow(2)
```

Đảo dấu logits biến mẫu "đã học đúng" thành mẫu "sai hoàn toàn" dưới góc nhìn của softmax → gradient **không** tiêu biến → `ω_i` phản ánh đúng độ nhạy của loss theo từng trọng số. Phần còn lại của thuật toán giữ nguyên như EWC.

### 1.3 Tích lũy độ quan trọng qua các task

Cả EWC và EWC-DR đều trộn Fisher/omega cũ với mới theo tỉ lệ số lớp:

```python
alpha = self._known_classes / self._total_classes
new_omega[n][:len(old)] = alpha * old[n] + (1 - alpha) * new_omega[n][:len(old)]
```

`[:len(old)]` cần thiết vì lớp `fc` **nở ra** sau mỗi task (thêm neuron cho lớp mới), nên tensor mới dài hơn tensor cũ ở chiều 0.

Sau đó `ω` bị chặn trên bằng `omegamax` (mặc định `1e-4`) để tránh một vài trọng số chiếm hết hình phạt:

```python
omega[n] = torch.min(omega[n], torch.tensor(self.omegamax).to(self._device))
```

### 1.4 Loss khi học task mới

Điểm đáng chú ý: cross-entropy chỉ tính trên **các logit của lớp mới**, không phải toàn bộ:

```python
# models/ewc_dr.py:173
loss_clf = F.cross_entropy(
    logits[:, self._known_classes:],      # chỉ cột của lớp mới
    targets - self._known_classes         # nhãn dời về 0-based
)
loss = loss_clf + self.lamda * self.compute_ewc()
```

Nhưng lúc **đánh giá** thì dùng toàn bộ logit (`_eval_cnn`). Đây là thiết lập chuẩn của PyCIL cho EFCIL.

---

## 2. Luồng thực thi

```
main.py
  └─ đọc --config (JSON) -> merge vào argparse Namespace -> dict
     └─ trainer.train(args)
        └─ lặp qua từng seed -> trainer._train(args)
           ├─ tạo logs/{model}/{dataset}/{init_cls}/{increment}/...
           ├─ _set_random()          (seed cứng = 1)
           ├─ _set_device(args)      (ép cuda)
           ├─ DataManager(...)       (tải + chia lớp thành task)
           ├─ factory.get_model(...) (chọn learner)
           └─ for task in range(nb_tasks):
                ├─ model.incremental_train(data_manager)
                │    ├─ _network.update_fc(total_classes)   nở classifier
                │    ├─ tạo train_loader / test_loader
                │    ├─ _train(...)   -> _init_train (task 0) | _update_representation
                │    ├─ getImportance / getFisherDiagonal
                │    └─ lưu self.mean = snapshot trọng số
                ├─ model.eval_task()  -> (cnn_accy, nme_accy)
                └─ model.after_task() -> _known_classes = _total_classes
             └─ np.save pre_dict / target_dict
```

### Chi tiết `incremental_train`

1. `_cur_task += 1`, tính `_total_classes = _known_classes + task_size`.
2. `_network.update_fc(_total_classes)` — tạo `SimpleLinear` mới lớn hơn, **copy weight/bias cũ** vào phần đầu:
   ```python
   # utils/inc_net.py:140
   fc.weight.data[:nb_output] = weight
   fc.bias.data[:nb_output]   = bias
   ```
3. Dựng DataLoader: train chỉ trên lớp mới `[known, total)`, test trên **tất cả** lớp đã thấy `[0, total)`.
4. Task 0 dùng lịch `init_*` (200 epoch, lr 0.1, milestones [60,120,170]); task ≥1 dùng `epochs`/`lr`/`milestones` (180 epoch, milestones [70,120,150]).
5. Tính importance, chụp `self.mean` (bản sao trọng số hiện tại — chính là `θ*` trong công thức).

---

## 3. Mô tả từng file

### 3.1 Gốc

| File | Vai trò |
|---|---|
| `main.py` | 30 dòng. Parse `--config`, `json.load`, merge, gọi `trainer.train`. |
| `trainer.py` | Thiết lập logging (file + stdout), seed, device; vòng lặp task; in `CNN top1 curve` và `Average Accuracy`; lưu `pre_dict_{seed}.npy` / `target_dict_{seed}.npy`. |
| `requirements.txt` | 179 dòng — pin toàn bộ môi trường (torch 2.0.1, torchvision 0.15.2, timm 0.6.7, python 3.11.4). |
| `LICENSE.txt` | Apache 2.0. |

### 3.2 `models/` — các thuật toán CL

| File | Class | Ghi chú |
|---|---|---|
| `base.py` | `BaseLearner` | Lớp cha 507 dòng. Chứa: `eval_task`, `_eval_cnn` (top-k argmax), `_eval_nme` (nearest-class-mean bằng khoảng cách Euclid trên feature đã chuẩn hóa), `_eval_maha`/`_mahalanobis` (khoảng cách Mahalanobis, dùng cho FeCAM), `_extract_vectors`, và bộ ba **herding exemplar** `_construct_exemplar` / `_reduce_exemplar` / `_construct_exemplar_unified`. |
| `ewc.py` | `EWC` | EWC gốc. Hyperparameter **hardcode ở cấp module** (`init_epoch=200`, `lamda=1000`, `fishermax=1e-4`, `batch_size=128`...). Có thanh tiến trình `tqdm`. |
| `ewc_dr.py` | `EWCDR` | Phương pháp của paper. Hyperparameter **đọc từ JSON** qua `self.args.get(...)`. Không có `tqdm`, ghi log mỗi epoch bằng `logging.info`. `lamda` là **bắt buộc** trong JSON (`self.args["lamda"]`, không có default). |
| `finetune.py` | `Finetune` | Baseline: chỉ cross-entropy, không regularization. Dùng để đo mức quên tối đa. |
| `__init__.py` | — | Rỗng. |

**Khác biệt EWC vs EWC-DR ngoài dòng `logits*-1`:**

| | `ewc.py` | `ewc_dr.py` |
|---|---|---|
| Nguồn hyperparameter | hằng số module | JSON config |
| `lamda` mặc định | 1000 | bắt buộc khai báo (config dùng 10000) |
| Nhân `lamda` | `loss_clf + lamda * loss_ewc` | `loss_clf + (lamda * compute_ewc())` — tương đương |
| Clamp importance | `torch.tensor(fishermax)` (CPU scalar) | `torch.tensor(omegamax).to(device)` |
| Thứ tự `DataParallel` | bọc → train → gỡ | bọc rồi gỡ **ngay lập tức** (vô hiệu, xem `docs/so-sanh-va-luu-y.md`) |
| Hiển thị | `tqdm` | `logging` |

### 3.3 `utils/`

| File | Nội dung |
|---|---|
| `factory.py` | `get_model(name, args)` → dispatch chuỗi tên sang class. Hỗ trợ `ewc`, `finetune`, `ewc_dr`, và **3 tên không tồn tại file**: `ewc_mas`, `ewc_online`, `ewc_si`. |
| `data.py` | Định nghĩa `iCIFAR10`, `iCIFAR100`, `iImageNet100`, `iImageNet1000`, `iTinyImageNet200`. Mỗi class khai báo `train_trsf` / `test_trsf` / `common_trsf` (augmentation), `class_order`, `use_path`, và `download_data()`. **Có patch riêng**: `_ensure_cifar100()` tải CIFAR-100 từ mirror HuggingFace kèm kiểm tra MD5, vì `toronto.edu` bị chặn/chậm trên Kaggle. |
| `data_manager.py` | `DataManager` — trái tim của phần dữ liệu. Chia `class_order` thành list `_increments` (`init_cls` rồi lặp `increment`), cung cấp `get_dataset(indices, source, mode, ...)` trả về `DummyDataset`. `DummyDataset.__getitem__` trả tuple **3 phần tử** `(idx, image, label)` — lý do mọi vòng lặp train đều viết `for i, (_, inputs, targets) in enumerate(loader)`. `_map_new_class_index` ánh xạ nhãn gốc sang thứ tự lớp đã xáo. |
| `inc_net.py` | 845 dòng. `get_convnet()` dispatch tên backbone. `BaseNet` (bọc convnet + fc, trả dict `{fmaps, features, logits}`). Các biến thể: **`IncrementalNet`** (dùng bởi cả 3 learner ở đây), `IncrementalNetWithBias` (BiC), `CosineIncrementalNet` (UCIR), `SimpleCosineIncrementalNet`, `DERNet`, `FOSTERNet`, `BEEFISONet`, `AdaptiveNet` (MEMO), `IL2ANet`. Phần lớn là **code thừa kế từ PyCIL, không dùng trong repo này**. |
| `toolkit.py` | `count_parameters`, `tensor2numpy`, `target2onehot`, `accuracy()` (tính top1 tổng + độ chính xác theo nhóm 10 lớp + `old`/`new`), `split_images_labels`, `save_fc`, `save_model`, `ConfigEncoder`. |
| `ops.py` | Các phép augmentation nguyên thủy dựa trên PIL: `Cutout`, `ShearX/Y`, `TranslateX/Y`, `Rotate`, `Color`, `Posterize`, `Solarize`, `Contrast`, `Sharpness`, `Brightness`, `AutoContrast`, `Equalize`, `Invert`. |
| `autoaugment.py` | `ImageNetPolicy`, `CIFAR10Policy`, `SVHNPolicy`, `SubPolicy` — AutoAugment. **Không được gọi** trong luồng hiện tại (`data.py` chỉ dùng RandomCrop/Flip/ColorJitter). |
| `rl_utils/ddpg.py` | `PolicyNet`, `RMMPolicyNet`, `QValueNet`, `TwoLayerFC`, `DDPG` — dùng cho thuật toán RMM của PyCIL. **Không dùng ở đây.** |
| `rl_utils/rl_utils.py` | `ReplayBuffer` cho DDPG. **Không dùng ở đây.** |

### 3.4 `convs/` — thư viện backbone

Tất cả backbone đều trả về **dict** `{'fmaps': [x1..x4], 'features': vector}` thay vì tensor, và có thuộc tính `out_dim` để `BaseNet.feature_dim` đọc.

| File | Nội dung | Có dùng? |
|---|---|---|
| `resnet.py` | ResNet chuẩn (torchvision-style) đã sửa để trả dict: `resnet10/18/26/34/50/101/152`, `resnext`, `wide_resnet`. **Stem `conv1` thay đổi theo config** — xem bảng bên dưới. | ✅ `resnet18` là backbone của mọi config |
| `cifar_resnet.py` | `CifarResNet` / `resnet32` và các `DownsampleA–D`. | Có sẵn (`convnet_type: resnet32`) |
| `linears.py` | `SimpleLinear` (trả `{'logits': ...}`), `CosineLinear`, `SplitCosineLinear`, `reduce_proxies`. | ✅ `SimpleLinear` |
| `ucir_resnet.py`, `ucir_cifar_resnet.py` | Backbone cho UCIR (cosine normalization). | ❌ |
| `modified_represnet.py` | ResNet có `conv_block` tách rời cho phương pháp rep-based. | ❌ |
| `resnet_cbam.py` | ResNet + attention CBAM (`ChannelAttention`, `SpatialAttention`). | ❌ |
| `memo_resnet.py`, `memo_cifar_resnet.py` | Backbone tách đôi Generalized/Specialized cho MEMO. | ❌ |
| `conv_cifar.py`, `conv_imagenet.py` | ConvNet nông (2/4 lớp) cho benchmark nhỏ. | ❌ |

#### Stem của `resnet.py` phụ thuộc config (`convs/resnet.py:150–179`)

Đây là chỗ dễ bỏ sót nhất: kiến trúc **thay đổi theo `dataset` và theo việc `init_cls` có bằng `increment` hay không**.

| Điều kiện | Stem |
|---|---|
| `'cifar' in dataset` (và `model_name != "memo"`) | Conv 3×3 stride 1 + BN + ReLU (không MaxPool) |
| `'imagenet' in dataset` **và** `init_cls == increment` | Conv 7×7 stride 2 + BN + ReLU + MaxPool 3×3 stride 2 |
| `'imagenet' in dataset` **và** `init_cls != increment` | Conv 3×3 stride 1 + BN + ReLU + MaxPool 3×3 stride 2 |

Hệ quả thực tế:
- `'imagenet' in 'tinyimagenet'` là `True` → **Tiny-ImageNet đi theo nhánh imagenet**, không có nhánh riêng.
- Config `ewcdr_imagesub_bigstart.json` (`init_cls=50 ≠ increment=5`) dùng stem **3×3 stride 1 trên ảnh 224×224** → feature map tầng đầu là 224×224×64 thay vì 56×56×64, chi phí tính toán và VRAM cao hơn khoảng **16 lần** so với nhánh equal-split. Đây là hành vi kế thừa từ PyCIL, không phải lỗi cú pháp, nhưng cần biết khi ước lượng thời gian chạy.
- `assert args is not None, "you should pass args to resnet"` — không thể tạo backbone này ngoài luồng config.

---

## 4. Cấu hình (`exps/*.json`)

### Các khóa

| Khóa | Ý nghĩa |
|---|---|
| `prefix` | Tiền tố tên file log. |
| `dataset` | `cifar100` \| `imagenet100` \| `tinyimagenet` \| `cifar10` \| `imagenet1000`. |
| `model_name` | `ewc_dr` \| `ewc` \| `finetune` — quyết định class trong `factory.py`. |
| `convnet_type` | `resnet18` trong mọi config sẵn có. |
| `init_cls` / `increment` | Số lớp task đầu / mỗi task sau. `init_cls == increment` → *equal-split*; `init_cls` lớn → *big-start*. |
| `shuffle` | `true` → xáo thứ tự lớp bằng `np.random.seed(seed)`. |
| `memory_size`, `memory_per_class`, `fixed_memory` | Đều 0/true → **không dùng exemplar** (EFCIL). |
| `lamda` | Trọng số hình phạt EWC. Config dùng `10000`. Chỉ `ewc_dr` đọc khóa này. |
| `init_epochs`, `init_lr`, `init_milestones`, `init_lr_decay`, `init_weight_decay` | Lịch huấn luyện task 0. |
| `epochs`, `lr`, `milestones`, `lr_decay`, `weight_decay` | Lịch huấn luyện task ≥1. |
| `batch_size`, `device`, `seed` | `seed` là **list** (`[1993]`) — chạy nhiều seed liên tiếp. |

### Bảng các config sẵn có

| File | Dataset | init_cls / increment | Số task | Model |
|---|---|---|---|---|
| `ewcdr_cifar_bigstart.json` | cifar100 | 50 / 5 | 11 | ewc_dr |
| `ewcdr_cifar_equalsplit.json` | cifar100 | 10 / 10 | 10 | ewc_dr |
| `ewcdr_imagesub_bigstart.json` | imagenet100 | 50 / 5 | 11 | ewc_dr |
| `ewcdr_imagesub_equalsplit.json` | imagenet100 | 10 / 10 | 10 | ewc_dr |
| `ewcdr_tinyimg_bigstart.json` | tinyimagenet | 100 / 10 | 11 | ewc_dr |
| `ewcdr_tinyimg_equalsplit.json` | tinyimagenet | 20 / 20 | 10 | ewc_dr |
| `ewc.json` | cifar100 | 5 / 5 | 20 | ewc |
| `ewc_equalsplit.json` | cifar100 | 10 / 10 | 10 | ewc |
| `finetune.json` | cifar100 | 50 / 2 | 26 | finetune |
| `finetune_equalsplit.json` | cifar100 | 10 / 10 | 10 | finetune |

> README gốc ghi "preconfigured for 10-task", đúng với các file `*_equalsplit`; các file `bigstart` thực ra là 11 task (1 task lớn + 10 task nhỏ). `ewc.json` và `finetune.json` lệch hẳn khỏi mô tả này.

---

## 5. Notebook

| File | Nội dung |
|---|---|
| `ewc-dr.ipynb` | `git clone` repo từ GitHub về `/kaggle/working`, kiểm tra GPU, in config, chạy `main.py`, parse log lấy `CNN top1 curve` + `Average Accuracy`. |
| `ewc.ipynb`, `ewcdr.ipynb`, `kaggle_ewc_run.ipynb`, `kaggle_ewcdr_run.ipynb` | Cùng cấu trúc nhưng **copy từ `/kaggle/input`** thay vì clone, và tự tải sẵn `cifar-100-python.tar.gz` (kiểm MD5). Khác nhau chỉ ở config được chạy (`ewc_equalsplit` vs `ewcdr_cifar_equalsplit`). |
| `notebook65c0822b27.ipynb` | Notebook tự sinh từ Kaggle, cùng mục đích. |

Cả 6 notebook về cơ bản **trùng lặp nhau**; giữ 1–2 file là đủ.

---

## 6. Đầu ra

```
logs/{model_name}/{dataset}/{init_cls}/{increment}/
├── {prefix}_{seed}_{convnet_type}.log   <- log text đầy đủ
├── pre_dict_{seed}.npy                  <- dict {task: y_pred top-5} 
└── target_dict_{seed}.npy               <- dict {task: y_true}
```

Trong log, dòng quan trọng:
- `CNN top1 curve: [...]` — độ chính xác sau mỗi task.
- `Average Accuracy (CNN): x` — trung bình của curve trên, chính là chỉ số báo cáo trong paper.
- `CNN: {'total':..., '00-09':..., 'old':..., 'new':...}` — phân rã theo nhóm lớp.

`init_cls` trong đường dẫn được đặt về `0` khi `init_cls == increment` (xem `trainer.py:24`), nên config equal-split sẽ ghi vào `logs/ewc_dr/cifar100/0/10/`.

---

## 7. Chạy

```bash
cd EWC-DR
python main.py --config=./exps/ewcdr_cifar_equalsplit.json
```

Yêu cầu: GPU CUDA (code **không** có nhánh CPU — xem lưu ý #2 trong `so-sanh-va-luu-y.md`), dataset trong `./data/`.
