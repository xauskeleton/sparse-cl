# So sánh hai phương pháp & danh sách cạm bẫy trong code

---

## Phần A — So sánh EWC-DR vs Fly-CL

| Tiêu chí | EWC-DR | Fly-CL |
|---|---|---|
| Họ phương pháp | Regularization-based | Pre-trained + closed-form (analytic) |
| Backbone | ResNet-18 **train from scratch** | ViT-B/16 pretrained, **đóng băng** |
| Cơ chế học | SGD + momentum, 180–200 epoch/task | Ridge regression, không có gradient descent |
| Cơ chế chống quên | Hình phạt bậc hai có trọng số `ω_i` trên độ lệch trọng số | Tích lũy thống kê đủ `G`, `Q` → nghiệm chính xác trên toàn dữ liệu |
| Có quên không? | Có, chỉ giảm nhẹ | Phần classifier **không quên** về mặt toán học |
| Lưu ảnh cũ (exemplar) | Không | Không |
| Lưu gì giữa các task | `ω` (1 tensor/tham số) + `θ*` (1 bản sao model) | `G` (D×D) + `Q` (D×C) |
| Thời gian/task | Hàng giờ (GPU) | ~54 s, trong đó 44 s là trích đặc trưng |
| Đóng góp chính | Đảo dấu logits khi ước lượng độ quan trọng | Chiếu ngẫu nhiên thưa + top-k để khử đa cộng tuyến |
| Độ chính xác tham chiếu | Xem `figs/result.png` | 88.68 % cuối cùng / 92.99 % tích lũy trên CIFAR-100 |

**Hai bài toán không so sánh trực tiếp được**: EWC-DR học biểu diễn từ đầu (không có kiến thức ngoài), Fly-CL dựa trên ViT đã pretrain trên ImageNet-21k — vốn đã "biết" phần lớn khái niệm trong CIFAR-100. Con số 88.68 % của Fly-CL không cùng thang đo với các con số EFCIL của EWC-DR.

---

## Phần B — Cạm bẫy & vấn đề đã phát hiện

### B1. EWC-DR

**1. `factory.py` trỏ tới 3 file không tồn tại**
```python
elif name == "ewc_mas":     from models.ewc_mas import EWC_MAS       # models/ewc_mas.py KHÔNG có
elif name == "ewc_online":  from models.ewc_online import OnlineEWC  # KHÔNG có
elif name == "ewc_si":      from models.ewc_si import EWC_SI         # KHÔNG có
```
Chỉ `ewc`, `ewc_dr`, `finetune` chạy được. Ba baseline so sánh trong paper (MAS, Online EWC, SI) **không có trong bản phát hành này**.

**2. `_set_device` bỏ qua config, ép CUDA cứng** — `trainer.py:119`
```python
def _set_device(args):
    device_type = args["device"]     # đọc rồi... bỏ đi
    gpus = []
    device = torch.device("cuda")    # hardcode
    gpus.append(device)
    args["device"] = gpus
```
Nhánh xử lý `-1 → cpu` và multi-GPU đã bị comment. Hệ quả: **không chạy được trên máy không có CUDA**, và `"device": ["0"]` trong JSON hoàn toàn vô nghĩa. `len(args["device"])` luôn = 1, nên mọi nhánh `if len(self._multiple_gpus) > 1` đều là code chết.

**3. Seed trong config chỉ ảnh hưởng thứ tự lớp, không ảnh hưởng khởi tạo trọng số** — `trainer.py:137`
```python
def _set_random():
    torch.manual_seed(1)   # hằng số, không phải args["seed"]
```
`args["seed"]` chỉ được truyền vào `DataManager` để `np.random.seed(seed)` xáo `class_order`. Khi báo cáo "mean across three independent trials", ba lần chạy có **cùng khởi tạo mạng**, chỉ khác thứ tự lớp.

**4. `models/ewc.py` bỏ qua hoàn toàn hyperparameter trong JSON**
Toàn bộ `init_epoch`, `epochs`, `lrate`, `batch_size`, `lamda=1000`, `fishermax` là hằng số cấp module. `"batch_size": 128` trong `ewc.json` không có tác dụng gì (trùng giá trị nên vô hại), nhưng nếu bạn sửa JSON để đổi `lamda` hay `epochs` cho EWC thì **sẽ không có gì thay đổi**. Chỉ `ewc_dr.py` đọc JSON.

**5. `omegamax` / `fishermax` không có trong bất kỳ file JSON nào**
`ewc_dr.py` dùng `self.args.get("omegamax", 0.0001)` → luôn rơi về `1e-4`. Đây là ngưỡng chặn trên rất chặt: kết hợp với `lamda=10000`, hình phạt tối đa cho mỗi trọng số là `10000 * 1e-4 / 2 = 0.5 * (Δθ)²`. Nếu muốn thí nghiệm với độ mạnh regularization, đây là khóa cần thêm vào JSON.

**6. `ewc_dr.py` bọc rồi gỡ `DataParallel` ngay lập tức** — dòng 70–74
```python
if len(self._multiple_gpus) > 1:
    self._network = nn.DataParallel(...)
if len(self._multiple_gpus) > 1:          # gỡ ngay, chưa train gì
    self._network = self._network.module
self._train(...)
```
So với `ewc.py` (bọc → train → gỡ). Vì lỗi #2 khiến `len == 1` nên không gây sai kết quả, nhưng nếu ai đó sửa `_set_device` để bật multi-GPU thì `ewc_dr` sẽ **không** dùng nhiều GPU còn `ewc` thì có → so sánh không công bằng.

**7. `_eval_nme` luôn trả `None`**
`_class_means` chỉ được tạo trong `build_rehearsal_memory()`, mà không learner nào ở đây gọi (đúng với thiết lập exemplar-free). Nên `eval_task` luôn đi nhánh `nme_accy = None`, log luôn in `"No NME accuracy."`. Toàn bộ code herding exemplar trong `base.py` (khoảng 200 dòng) là code chết.

**8. Nhánh `fecam` trong `base.py:106` là code chết và sẽ crash nếu kích hoạt**
```python
if self.args["model_name"] == "fecam":
    y_pred, y_true = self._eval_maha(self.test_loader, self._init_protos, self._protos)
```
`self._init_protos` và `self._protos` không được định nghĩa ở bất kỳ đâu trong repo. `factory.py` cũng không có `fecam`, nên nhánh không bao giờ chạy — nhưng đừng đặt `"model_name": "fecam"` vào JSON.

**9. Độ chính xác theo nhóm luôn chia theo bước 10 lớp**
`toolkit.accuracy(y_pred, y_true, nb_old, increment=10)` — `base.py:75` gọi mà không truyền `increment`. Với config `increment=5` (`ewc.json`, `ewcdr_*_bigstart`), các nhãn `'00-09'`, `'10-19'`… trong log **không tương ứng ranh giới task**. Chỉ `top1`, `old`, `new` là đáng tin.

**10. `shuffle: false` sẽ hỏng với ImageNet-100 / Tiny-ImageNet**
`iImageNet100.class_order = np.arange(1000).tolist()` và `iTinyImageNet200.class_order = np.arange(200)` — cái đầu sai (dataset chỉ có 100 lớp). Mọi config hiện tại đặt `shuffle: true` nên `class_order` bị thay bằng `np.random.permutation(100)`, che mất lỗi. Đừng đặt `shuffle: false` cho `imagenet100`.

**11. Ước lượng độ quan trọng chạy ở chế độ `train()` → cập nhật BatchNorm**
`getFisherDiagonal` / `getImportance` gọi `self._network.train()` rồi forward toàn bộ train_loader. Trọng số không đổi (không có `optimizer.step()`), nhưng **running_mean / running_var của mọi lớp BatchNorm bị cập nhật thêm một lượt** sau khi huấn luyện đã xong. Ảnh hưởng nhỏ, đồng đều giữa EWC và EWC-DR nên không phá so sánh, nhưng là một tác dụng phụ ngoài ý muốn.

**12. `data.py` chứa patch riêng cho môi trường Kaggle**
`_ensure_cifar100()` (dòng 19–37) tải CIFAR-100 từ mirror HuggingFace kèm kiểm MD5, với comment tiếng Việt không dấu. Đây là **sửa đổi cục bộ**, không có trong repo gốc của tác giả. Nếu đồng bộ lại từ upstream, đoạn này sẽ mất — giữ lại nếu vẫn chạy trên Kaggle. File `data/cifar-100-python.tar.gz` đã có sẵn nên hàm sẽ thoát sớm khi MD5 khớp.

**13. Sáu notebook gần như trùng lặp**
`ewc-dr.ipynb`, `ewc.ipynb`, `ewcdr.ipynb`, `kaggle_ewc_run.ipynb`, `kaggle_ewcdr_run.ipynb`, `notebook65c0822b27.ipynb` chỉ khác nhau ở tên config và cách lấy source (clone từ GitHub vs copy từ `/kaggle/input`). Nên gộp còn một notebook có tham số.

---

### B2. Fly-CL

**1. Script hardcode chỉ số GPU của máy tác giả**
`test_cifar.sh --gpu 5`, `test_cub.sh --gpu 1`, `test_vtab.sh --gpu 6`. Trên máy 1 GPU, `torch.device("cuda:5")` sẽ lỗi khi cấp phát. **Sửa thành `--gpu 0` trước khi chạy.**

**2. Giá trị mặc định trong `main.py` khác hẳn giá trị dùng thật**

| Tham số | Mặc định `main.py` | Script (giá trị thật) |
|---|---|---|
| `synaptic_degree` | 100 | **300** |
| `coding_level` | 0.01 | **0.3** |
| `ridge_lower` | 4 | **6** |

Chạy `python main.py` không tham số sẽ ra kết quả kém hơn nhiều so với paper. Luôn dùng script hoặc copy đầy đủ tham số.

**3. `pretrained_model/download.sh` tải file không ai dùng**
Script tải `vit_base_patch16_224_in21k.npz`, nhưng `load_model.py` gọi `timm.create_model("vit_base_patch16_224", pretrained=True)` — timm tự tải trọng số riêng từ HuggingFace Hub. File `.npz` bị bỏ không. Muốn dùng đúng checkpoint đã tải, phải sửa thành `checkpoint_path='./pretrained_model/vit_base_patch16_224_in21k.npz'`.

**4. Nhánh `resnet-50` cần file không có trong repo**
`load_model.py:17` yêu cầu `./pretrained_model/resnet50-11ad3fa6.pth`. Không có file này và không script nào tải nó → `--model_name resnet-50` sẽ lỗi. Ngoài ra `keys_to_remove = [k for k in state_dict if "classifier" in k]` không khớp gì cả — ResNet của timm đặt tên head là `fc.*`, không phải `classifier.*`; nhưng `load_state_dict(strict=False)` nuốt lỗi nên vẫn chạy được nếu có file.

**5. Trích đặc trưng test lặp lại O(T²) lần**
```python
for task in range(num_tasks):
    ...
    for sub_task in range(task + 1):
        test_embeddings, test_labels = feature_extract(pretrained_model, test_loader[sub_task], device)
```
Đặc trưng test của task 0 được tính lại ở **cả 10 vòng**. Với ViT trên CIFAR-100 điều này chiếm phần lớn wall-clock. Cache một lần là tối ưu hiển nhiên (giảm khoảng 5 lần thời gian tổng), nhưng lưu ý con số "Feature Extract Time" báo cáo trong log **chỉ tính phần train**, nên tối ưu này không làm thay đổi số liệu trong paper.

**6. `cudnn.enabled = False`** — `utils.py:14`
Được đặt để đảm bảo tái lập, nhưng tắt cuDNN làm mọi phép convolution chậm đi đáng kể. Với ViT (chủ yếu là matmul) ảnh hưởng nhỏ; với `--model_name resnet-50` thì rất lớn.

**7. Bộ nhớ GPU tăng theo bình phương `expand_dim`**
`G` là `expand_dim × expand_dim` float32 ≈ 400 MB ở `expand_dim=10000`. Cộng `G + ridge*I` và thừa số Cholesky → ~1.2 GB. `expand_dim=20000` → ~4.8 GB. Ngoài ra `train_embeddings` sau top-k được giữ **dạng dense** `[10000, N]`; với N ≈ 5000 mẫu/task đó là thêm ~200 MB dù 70 % là số 0.

**8. `CustomDataset` được định nghĩa nhưng không dùng**
`datasets/load_dataset.py:9–24`. Cả ba dataset đều đi qua `datasets.CIFAR100` hoặc `datasets.ImageFolder`. Code chết.

**9. `log_cifar_seed1993.txt` mã hóa UTF-16**
Đọc bằng `encoding='utf-16'`; công cụ mặc định UTF-8 sẽ thấy ký tự null xen kẽ.

**10. File `2.7` là rác**
67 KB output của `conda search pytorch` bị redirect nhầm (`conda search pytorch > 2.7` thay vì `--python 2.7`). Xóa được.

**11. `random.sample` chia task phụ thuộc seed toàn cục**
`load_dataset` gọi `random.sample(...)` sau `random_initialization(seed)`, nên thứ tự lớp tái lập được — nhưng nếu thêm bất kỳ lời gọi `random` nào trước đó, cách chia task sẽ đổi.

---

## Phần C — Việc nên làm nếu tiếp tục phát triển

Sắp theo mức độ ảnh hưởng:

1. **Sửa `_set_device` trong `EWC-DR/trainer.py`** để tôn trọng config và có nhánh CPU — hiện là rào cản cứng khi chạy thử trên máy không GPU.
2. **Sửa `_set_random` để dùng `args["seed"]`** — nếu không, "3 independent trials" không thực sự độc lập ở phần khởi tạo.
3. **Thêm 3 file thiếu (`ewc_mas`, `ewc_online`, `ewc_si`)** hoặc bỏ chúng khỏi `factory.py` để tránh lỗi import khó hiểu.
4. **Sửa `--gpu` trong 3 script Fly-CL** về `0`.
5. **Cache đặc trưng test trong `Fly-CL/main.py`** — giảm khoảng 5 lần thời gian đánh giá, không đổi kết quả.
6. **Đưa `omegamax` và `lamda` vào JSON làm biến khảo sát** — hiện `omegamax` bị khóa cứng ở mặc định.
7. Dọn: gộp 6 notebook EWC-DR, xóa `Fly-CL-main/2.7`, cân nhắc xóa phần kế thừa PyCIL không dùng — 9/11 file trong `convs/` (thực tế chỉ `resnet.py` + `linears.py` được khởi tạo), `utils/rl_utils/`, `utils/autoaugment.py`, và ~200 dòng herding exemplar trong `models/base.py`.
8. Truyền `increment` thật vào `toolkit.accuracy()` để nhãn nhóm trong log khớp ranh giới task.
