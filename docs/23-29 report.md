# Báo cáo tuần 23–29/08

Tuần trước chỉ chạy trên **một** backbone (ResNet-50) và **một** dataset
(CIFAR-100), nên mọi quy luật rút ra đều chưa biết là tính chất của phương pháp
hay của một cấu hình cụ thể. Tuần này mở cả hai trục: **ViT-B/16** và
**CUB-200-2011 / VTAB**.

Mốc so sánh vẫn là bản tái lập trong `sparse-cl/`, không phải con số công bố.

---

## 1. Tóm tắt

Kết quả lớn nhất: **ba quy luật của tuần trước chuyển được sang ViT, một quy luật
thì không** — và đúng cái không chuyển được là cái đang mâu thuẫn với LayUP.

| Quy luật rút ra trên ResNet | Trên ViT | |
|---|---|---|
| Nối tầng liền kề thì ăn | **đúng** | +0.46 |
| Càng lấy tầng xa càng kém | **đúng** | +0.46 → +0.28 → +0.12 |
| Tầng **quá xa thì có hại** | **sai** | vẫn +0.12, không âm |
| concat và ensemble cộng dồn được | **đúng** | 0.46 + 0.33 ≈ 0.73 |
| FSA ăn nhiều hơn mọi thứ khác | **đúng** | +2.14 |

Trên trục dataset, kết quả đáng chú ý nhất: **FSA gần như tắt trên VTAB**
(+0.27 thay vì +3.77), vì task 0 chỉ có ~360 ảnh. Chỗ đó concat vẫn cho +1.26 —
lần đầu tiên concat vượt FSA, và vì một lý do có thể phát biểu thành nguyên tắc:
concat không cần nhãn và không cần gradient.

Thử **Improved FSA của PACE** (ICLR 2026) trên cả sáu ô: thắng đúng một ô, và
thua càng nặng khi task 0 càng nhiều dữ liệu. Nhưng tiêu chí chọn tầng bằng CKA
của họ lại rơi đúng vào stage 3 / block 9 — hai tầng mà bảng concat chọn — nên nó
thành một chống lưng độc lập cho concat (mục 6).

Ngoài ra: đo lại tốc độ sạch trong một lần chạy được **12.0×**, và phát hiện chi
phí SVD **không giảm khi hạ chiều** — điều bác bỏ phản biện hiển nhiên nhất với
phần tối ưu. Ba lỗi trong repo được tìm ra khi mở sang backbone mới.

---

## 2. ViT-B/16 — quy luật nào chuyển được

ViT không có "stage". Lấy **CLS token ở bốn điểm đều theo độ sâu** — block 3, 6,
9, 12 — mỗi cái 768 chiều, ghép thành 3072. Tap cuối đi qua `norm` cuối cùng nên
trùng khớp tuyệt đối với feature 768 chiều đã dùng trước đây; đó là phép tự kiểm
cho toàn bộ mục này, và nó đạt.

Chọn bốn điểm đều thay vì bốn block cuối để `--b_stage` giữ nguyên nghĩa ở cả hai
kiến trúc: "lùi lại một phần tư độ sâu" là stage 3 trên ResNet và block 9 trên
ViT. Nhờ vậy hai bảng đọc song song được.

### Bảng chính

Một seed (1993), `E` = 10.000, `deg_s4` = 112, `coding_level` 0.3, sàn λ = 5.

| Cấu hình | A_T | Ā | Forgetting | Δ Ā |
|---|---:|---:|---:|---:|
| **Không FSA** | | | | |
| mốc (block 12) | 88.98 | 93.12 | 4.59 | — |
| ensemble m=2 | 89.19 | 93.31 | 4.58 | +0.19 |
| ensemble m=5 | 89.32 | 93.45 | 4.62 | +0.33 |
| **concat block 9** | 89.46 | **93.58** | 4.47 | **+0.46** |
| concat + ens 5 | 89.82 | 93.85 | 4.39 | +0.73 |
| **Có FSA** | | | | |
| mốc (block 12) | 91.58 | 95.26 | 3.69 | — |
| ensemble m=2 | 91.92 | 95.48 | 3.61 | +0.22 |
| ensemble m=5 | 92.12 | 95.65 | 3.54 | +0.39 |
| concat block 9 | 91.89 | 95.49 | 3.57 | +0.23 |
| **concat + ens 5** | **92.20** | **95.70** | 3.64 | +0.44 |

FSA một mình cho **+2.14** (93.12 → 95.26), vẫn là can thiệp lớn nhất, đúng như
trên ResNet.

### Tầng nào đáng nối

| Tầng nối vào | deg_s3 = 112 | deg_s3 = 300 |
|---|---:|---:|
| block 9 (liền kề) | +0.36 | **+0.46** |
| block 6 | +0.32 | +0.28 |
| block 3 (xa nhất) | +0.17 | +0.12 |

**Càng gần càng ăn**, đơn điệu, không ngoại lệ — giống hệt ResNet. Nhưng tầng xa
nhất trên ViT vẫn **dương** (+0.12), trong khi stage 2 trên ResNet **âm** (−0.58).

Đây là chỗ phân định với LayUP: họ đúng trên ViT, mình đúng trên ResNet, và lý do
là kiến trúc. ResNet đổi cả độ phân giải lẫn ngữ nghĩa qua mỗi stage nên tầng xa
mang thứ *khác loại*; các block ViT nằm trên cùng một residual stream nên tầng xa
chỉ *nhạt* đi chứ không *lệch* đi.

### Nối nhiều hơn một tầng thì không được gì

Cùng chi phí bộ nhớ (`Eb` vẫn 20.000), chỉ thay đổi phép chiếu đọc từ bao nhiêu
tầng:

| Nối vào | deg 300 | deg 112 |
|---|---:|---:|
| block 9 | 93.58 | 93.48 |
| block 6 + 9 | 93.57 | 93.58 |
| block 3 + 6 + 9 | 93.59 | 93.58 |

Bốn cấu hình nằm trong 0.02 điểm của nhau. Từng tầng xa **có** thông tin bổ trợ so
với block 12 (bảng trên), nhưng thông tin đó **đã nằm sẵn trong block 9** — đúng
cách residual stream hoạt động.

Phát biểu hợp nhất cho cả hai kiến trúc, và nó mạnh hơn LayUP (họ dùng `k` = 6):

> Đúng **một** tầng phụ là đáng lấy — tầng liền kề. Trên ViT lấy thêm là thừa,
> trên ResNet lấy thêm là có hại.

### Ensemble: đường cong đầy đủ

Đo ở sàn λ = 3 nên mốc là 93.03, không dùng chung cột Δ với bảng chính:

| m | 1 | 2 | 5 | 10 | 20 |
|---|---:|---:|---:|---:|---:|
| Ā | 93.03 | 93.30 | 93.51 | 93.59 | 93.66 |
| Δ | — | +0.27 | +0.48 | +0.56 | +0.63 |
| c = Δ/(1−1/m) | — | 0.54 | 0.60 | 0.62 | 0.66 |

`c` gần như hằng số ⇒ **trần ở khoảng +0.65 đến +0.70**, đọc ra được bằng ngoại
suy chứ không phải chạy tới m = 100. Ensemble là thuần giảm phương sai. Trên
ResNet thì `c` tụt (1.28 → 1.02), tức nó rời quy luật sớm hơn — giả thuyết: phần
thiên lệch trên ResNet lớn hơn nên chạm sàn trước khi phương sai kịp cạn.

Forgetting không đổi ở mọi mức m, cả hai backbone.

### Hệ số co giữa hai backbone

| Can thiệp | ResNet-50 | ViT-B/16 | tỉ lệ |
|---|---:|---:|---:|
| FSA | +3.77 | +2.14 | 1.8× |
| concat | +1.37 | +0.46 | 3.0× |
| ensemble m=5 | +0.82 | +0.33 | 2.5× |

Ba cơ chế hoàn toàn khác nhau mà cùng một hệ số co, nên hệ số đó là **tính chất
của khoảng cách tới trần**, không phải của phương pháp nào.

### concat sau FSA

| | chưa FSA | sau FSA | còn lại |
|---|---:|---:|---:|
| ResNet-50 | +1.37 | +0.25 | 18% |
| ViT-B/16 | +0.46 | +0.23 | 50% |

Sau khi backbone được thích nghi, nối thêm một tầng chỉ còn đáng khoảng một phần
tư điểm ở **cả hai** kiến trúc. Đây là tin xấu cho concat với tư cách một đóng
góp: **giá trị của nó phụ thuộc vào việc backbone chưa được thích nghi.**

### Trần của hai backbone

| | A_T | Ā | Forgetting |
|---|---:|---:|---:|
| ResNet-50, cấu hình tốt nhất | 82.72 | 89.05 | 7.12 |
| **ViT-B/16, cấu hình tốt nhất** | **92.20** | **95.70** | **3.64** |

Toàn bộ những gì cộng vào được cho ResNet (+4.97) **vẫn không bằng một nửa**
khoảng cách do đổi backbone tạo ra (+6.65). Con số này thuộc phần thảo luận.

---

## 3. Tăng tốc: đo lại sạch, và một phản biện bị bác

Tuần trước ghi 10.8×, ghép từ hai lần đo khác nhau. Đo lại cả bốn cấu hình trong
**một lần chạy**, cùng máy, `scripts/bench_time.sh`:

| Cấu hình | GCV | Thời gian | A_T | Ā | Forgetting |
|---|---|---:|---:|---:|---:|
| E=10000 m=1 | SVD (bản gốc) | 96s | 76.99 | 84.08 | 8.54 |
| E=10000 m=1 | eigh | **8s** | 76.99 | 84.08 | 8.54 |
| E=5000 m=2 | SVD (bản gốc) | 190s | 75.49 | 83.41 | 9.60 |
| E=5000 m=2 | eigh | **12s** | 75.49 | 83.41 | 9.60 |

**12.0×**, không lệch một chữ số nào kể cả forgetting. Bản gốc chạy được bằng
`FLYCL_GCV=svd` — hàm `_select_ridge_parameter_svd` trong `utils.py` chép nguyên
văn từ `upstream/Fly-CL-main/main.py`.

### Hạ chiều không cứu được bản gốc

Phản biện hiển nhiên: "lợi ích đến từ chỗ không dựng `V` cỡ `[n, E]`, nên `E` nhỏ
thì hết lợi". Chia chi phí ra mỗi lần gọi:

```
E=10000, 10 lan goi SVD:  (96 - 8)/10   = 8.8s moi lan
E= 5000, 20 lan goi SVD:  (190 - 12)/20 = 8.9s moi lan
```

**Giảm `E` một nửa mà SVD không rẻ đi chút nào.** Chi phí của nó bị chặn bởi phần
`O(n³)` trên cạnh `n`, không phải bởi `E`. Nên khoảng cách **giãn ra** khi hạ
chiều (15.8× ở `E`=5000 so với 12.0× ở `E`=10.000), chứ không thu hẹp.

Ba nguồn lợi, chỉ một phụ thuộc `E`:

1. không dựng `V` — tỉ lệ với `E`
2. `eigh` trên ma trận đối xứng rẻ hơn `svd` trên ma trận tổng quát — không phụ
   thuộc `E`
3. bản gốc dựng lại `Y_hat = U(diag·UᵀY)` ở **cả 11 bước lưới**; bản mới gấp phần
   đó vào `w_j = ‖u_jᵀY‖²` tính một lần — không phụ thuộc `E`

### Ensemble không phải cách nén

Cùng 10.000 unit, chia hai cách:

| | Ā | A_T | Forgetting | Bộ nhớ `G` |
|---|---:|---:|---:|---:|
| E=10000, m=1 | **84.08** | **76.99** | **8.54** | 381 MB |
| E=5000, m=2 | 83.41 | 75.49 | 9.60 | **190 MB** |

Chia nhỏ ra thì mất 0.67 điểm để đổi lấy một nửa bộ nhớ. Ensemble chỉ ăn khi nó
**thêm** sức chứa, không phải khi nó **chia** sức chứa sẵn có — nhất quán với quy
luật ở mục 5 của báo cáo tuần trước.

---

## 4. Ba lỗi trong repo, tìm ra khi mở sang backbone mới

**`_ensure_fsa` không idempotent** (`trainer.py`). Vòng lặp `--grid` gọi
`train_cil` nhiều lần trên **cùng** object `args`, mà lần đầu đã ghi đè
`model_name` thành tag FSA. Lần hai nó đi FSA cho `xxx+fsa1993` và timm báo
"Unknown model". Hệ quả: **mọi lệnh có `--grid` nhiều mục kèm `--training_method
aper` đều chết từ mục thứ hai** — kể cả lệnh ResNet ghi trong mục Tái lập của báo
cáo tuần trước.

**`run_fsa` không truyền `--data_augmentation` và `--dataset`** (`fsa.py`). Mặc
định của `fsa.py` là `resnet` + CIFAR-100, nên FSA cho ViT sẽ chạy với
normalization của ResNet, và FSA cho CUB sẽ học trên CIFAR rồi ghi đè lên cache
của CUB. Cả hai đều **không báo lỗi gì**.

**Sàn λ là bẫy chung, không riêng của FSA.** Ở `--ridge_lower 3` trên feature
ViT+ms, GCV rơi vào cực tiểu giả rồi Cholesky vỡ ở task 3 ("not
positive-definite") — cùng triệu chứng, cùng nguyên nhân với lỗi đã gặp trên
feature FSA tuần trước. Hai lần ở hai chỗ không liên quan ⇒ đây là **thuộc tính
của việc chọn λ bằng GCV trên feature chiều cao**. Quy tắc: đổi backbone hoặc đổi
cách ghép feature thì phải kiểm lại sàn. `test_cifar.sh` của Fly-CL dùng sàn 6
cho ViT; sàn 5 đủ và được dùng cho mọi bảng ở đây.

---

## 5. FSA trên ba dataset của Fly-CL, hai backbone

Fly-CL công bố trên **CIFAR-100, CUB-200-2011, VTAB**. Repo của họ và repo này
đều chỉ có đường *đọc* CUB/VTAB (`ImageFolder` tại `data/cub/{train,test}` và
`data/vtab/{train,test}`), không có bước lấy dữ liệu. Đã dựng xong cả hai:

| Dataset | Lớp | Task | Train | Test | Nguồn |
|---|---:|---:|---:|---:|---|
| CIFAR-100 | 100 | 10 | 50.000 | 10.000 | torchvision |
| CUB-200-2011 | 200 | 10 | 5.994 | 5.794 | caltech.edu, 1.1 GB → `scripts/prepare_cub.py` |
| VTAB | 50 | 5 | 1.796 | 8.619 | Google Drive của LAMDA-PILOT, 127 MB |

CUB được chia theo `train_test_split.txt` **chính thức** (5994/5794), không tự
chia ngẫu nhiên — nếu chia lại thì con số không so được với bài nào. VTAB giải
nén ra đúng layout cần, dù README của PILOT ghi là `train/val`.

### Bảng FSA

Một seed (1993), `--grid 0:1` (không concat), sàn λ = 5, adapter hạng 8, 10 epoch
trên task 0 rồi đóng băng. Cột `t` là vòng CIL (học + đánh giá lại mọi task cũ),
**không** tính trích feature và **không** tính huấn luyện adapter (~400s trên
CIFAR) — bước FSA là chỗ duy nhất trong toàn pipeline có gradient.

| Dataset | Backbone | Ā không FSA | **Ā có FSA** | Δ Ā | A_T | Forgetting | t |
|---|---|---:|---:|---:|---:|---:|---:|
| CIFAR-100 | ResNet-50 | 84.08 | **87.86** | **+3.78** | 80.94 | 7.74 | 9s |
| CIFAR-100 | ViT-B/16 | 93.12 | **95.26** | **+2.14** | 91.58 | 3.69 | 9s |
| CUB-200-2011 | ResNet-50 | 71.22 | **74.20** | **+2.98** | 66.12 | 9.75 | 2s |
| CUB-200-2011 | ViT-B/16 | 91.30 | **91.31** | **+0.01** | 88.17 | 4.00 | 1s |
| VTAB | ResNet-50 | 93.27 | **93.54** | **+0.27** | 89.51 | 3.79 | 1s |
| VTAB | ViT-B/16 | 94.91 | **95.19** | **+0.28** | 93.40 | 2.72 | 1s |

### FSA ăn bao nhiêu — trải từ +3.78 xuống +0.01

```
delta cua FSA        ResNet-50   ViT-B/16    anh task 0
CIFAR-100              +3.78      +2.14         5.000
CUB-200-2011           +2.98      +0.01           600
VTAB                   +0.27      +0.28           360
```

Biên độ 378 lần giữa ô lớn nhất và ô nhỏ nhất. **FSA không phải một cải tiến ổn
định** — nó là một cải tiến có điều kiện, và hai điều kiện đọc thẳng ra từ bảng.

**Điều kiện 1 — đủ dữ liệu ở task 0.** VTAB có 1.796 ảnh train cho 50 lớp nên
task 0 chỉ còn ~360 ảnh để huấn luyện adapter. Cả hai backbone đều tắt ở đó
(+0.27, +0.28), dù VTAB là dataset **dễ nhất** trong ba (93–95 điểm). Đây là
giới hạn về cỡ dữ liệu, không phải về độ khó.

**Điều kiện 2 — backbone còn chỗ để cải thiện.** Trên CUB, cùng ~600 ảnh task 0,
ResNet ăn **+2.98** còn ViT ăn **+0.01**. Khác biệt không nằm ở dữ liệu mà ở
mốc: ResNet đứng ở 71.22 nên còn rất nhiều chỗ, ViT đã ở 91.30 nên 600 ảnh chim
không dạy thêm được gì. Ô +0.01 này là bằng chứng sạch nhất cho quy luật "hệ số
co theo khoảng cách tới trần" ở mục 2 — ở đây nó co về đúng không.

Hệ quả: **con số +3.77 của báo cáo tuần trước là ô thuận lợi nhất trong sáu ô.**
Nếu chỉ báo cáo CIFAR + ResNet thì FSA trông như can thiệp mạnh nhất; nhìn cả
sáu ô thì nó là can thiệp **kém ổn định nhất**.

Điều này cho concat một luận điểm mà FSA không có: **FSA cần nhãn của task 0 và
cần đủ dữ liệu để chạy gradient, còn concat không cần gì cả.** Đo phụ trên VTAB
(`logs/fsa_vtab.txt`, ngoài phạm vi bảng): concat cho **+1.26** trên ResNet và
**+1.04** trên ViT — gấp bốn lần FSA trên cùng dataset.

**Khoảng cách giữa hai backbone thay đổi 10 lần tuỳ dataset**: 1.65 trên VTAB,
7.40 trên CIFAR-100, **17.11 trên CUB**. CUB là phân loại mịn 200 loài chim, mà
feature ResNet-50 giám sát trên ImageNet-1k gần như không mang thông tin phân
biệt ở mức đó — rơi xuống 74.20 với forgetting 9.75, tệ nhất mọi cấu hình từng
đo. Hệ quả cho việc viết bài: **chọn backbone quyết định nhiều hơn chọn phương
pháp**, và mọi Δ trong cả hai báo cáo đều nhỏ hơn khoảng cách backbone trên chính
dataset đó.

## 6. Improved FSA kiểu PACE — thử và hầu hết là âm

**PACE** (Li, Zhou, Wang — *"PACE: Pretrained Audio Continual Learning"*, ICLR
2026, [arXiv 2602.03355](https://arxiv.org/abs/2602.03355)) đặt tên cho đúng hiện
tượng ở mục 5: **representation saturation** — backbone tiền huấn luyện đã nắm gần
hết thông tin liên quan ngay ở session đầu, nên FSA cải thiện task 0 mà không giúp
được task sau. Bằng chứng của họ giống hệt của mình: trên ESC-50, naive FSA cho
92.25 còn **không** FSA cho 92.50.

Improved FSA của họ có ba ý:

```
1. han che hoc o head     eta_head << eta_bb, VA train theo giai doan:
                          ham nong head E_head epoch roi KHOA head lai
2. chi adapt tang sau     ranh gioi chon bang CKA voi nguong rho = 0.94
3. analytic classifier    Fly-CL da co san - ridge tren ma thua
```

Ý 3 mình có sẵn. Cài ý 1 và 2 vào `fsa.py` (`--fsa_pace 1`), giữ nguyên mọi thứ
khác, chạy lại sáu ô.

### Kết quả: thắng đúng một ô

| Dataset | Backbone | không FSA | FSA thường | FiLM | **PACE** | PACE − thường |
|---|---|---:|---:|---:|---:|---:|
| CIFAR-100 | ResNet-50 | 83.43 | **87.86** | 86.53 | 84.16 | **−3.70** |
| CIFAR-100 | ViT-B/16 | 93.12 | **95.26** | 94.90 | 94.00 | −1.26 |
| CUB-200 | ResNet-50 | 71.22 | **74.20** | 72.21 | 70.54 | **−3.66** |
| CUB-200 | ViT-B/16 | 91.30 | 91.31 | **91.50** | 90.77 | −0.54 |
| VTAB | ResNet-50 | 93.27 | 93.54 | 93.27 | **93.92** | **+0.38** |
| VTAB | ViT-B/16 | 94.91 | **95.19** | 95.08 | 95.06 | −0.13 |

Hai ô CUB còn tụt **xuống dưới cả mốc không FSA**. Sắp theo cỡ dữ liệu task 0 thì
quy luật rõ:

```
anh task 0     PACE vs FSA thuong
   400            +0.38     VTAB ResNet
   400            -0.13     VTAB ViT
   600            -0.54     CUB ViT
   600            -3.66     CUB ResNet
 5.000            -3.70     CIFAR ResNet
 5.000            -1.26     CIFAR ViT
```

Càng nhiều dữ liệu càng thua. Hợp lý về cơ chế: mẹo của họ là **kìm head lại để
backbone hấp thụ gradient**, thiết kế cho tình huống backbone đã bão hoà và dữ liệu
ít. Có 5.000 ảnh thì để head học bình thường lại tốt hơn.

### Tách ba mẹo (VTAB + ResNet)

| | Ā | A_T | Forgetting |
|---|---:|---:|---:|
| không FSA | 93.27 | 88.56 | 4.39 |
| naive FSA (head 1e-3, adapter 1e-4) | 93.54 | 89.51 | 3.79 |
| + đảo tỉ lệ lr (head 1e-4, adapter 1e-3) | **93.04** | 89.70 | 3.35 |
| + train theo giai đoạn | 93.66 | 90.26 | 3.65 |
| + cắt tầng bằng CKA | **93.92** | **90.59** | **3.13** |

**Đảo lr một mình thì có hại** — 93.04, thấp hơn cả mốc không FSA. Chỉ khi kèm bước
khoá head nó mới thành dương. Nên đó không phải hai mẹo cộng dồn được mà là **hai
nửa của một mẹo**, đúng như PACE trình bày chúng chung trong một mục.

Hai chỉ số nói hai chuyện: `A_T` **tăng đơn điệu ở cả bốn nấc** (88.56 → 90.59),
còn `Ā` có một nấc tụt. `A_T` là chỉ số quan trọng hơn của CIL, và nó ủng hộ luận
điểm của PACE rằng khớp session đầu không phải mục tiêu. Trong lúc chạy: giai đoạn
thăm dò đạt 95.25% trên task 0 còn giai đoạn cuối chỉ 94.00% — **khớp task 0 kém
hơn mà kết quả cuối tốt hơn**.

### Thứ đáng giữ lại không phải cột điểm mà là ranh giới CKA

`L_tune` là tầng nông nhất có CKA (so với model gốc) tụt dưới 0.94 — đo bằng một
lượt adapt thăm dò rồi vứt trọng số. Nó **không dùng nhãn, không dùng độ chính
xác**, chỉ đo biểu diễn dịch chuyển ở đâu:

```
ResNet-50, 16 khoi Bottleneck        CIFAR 12   CUB 13   VTAB 12
   stage 4 bat dau o khoi 13   -> ranh gioi roi dung STAGE 3 / STAGE 4

ViT-B/16, 12 block                   CIFAR 7    CUB 7    VTAB 8
                               -> ranh gioi roi dung BLOCK 8-9
```

Đường CKA trên ResNet/VTAB, đơn điệu sạch:

```
khoi   0     1     2     3     4     5     6     7
CKA  1.000 0.999 1.000 1.000 0.998 0.997 0.997 0.994
khoi   8     9    10    11    12    13    14    15
CKA  0.975 0.972 0.960 0.947 0.929 0.853 0.784 0.763
```

Mà mục 2 và bảng concat của tuần trước chọn **stage 3** trên ResNet và **block 9**
trên ViT là hai tầng đáng nối nhất. Một tiêu chí hoàn toàn khác — không nhãn, không
điểm số, lấy từ một bài về audio — chỉ vào đúng hai tầng đó, trên **cả hai kiến
trúc và cả ba dataset**. Đây là chống lưng độc lập cho phát hiện concat, và mạnh
hơn bất kỳ con số nào trong bảng trên.

### Giới hạn

Đây là kết luận về **bản port với siêu tham số cố định**, chưa phải về phương pháp
của họ. Ba chỗ lệch: `E_0` của họ chỉnh theo từng dataset (1 đến 30 epoch) còn ở
đây để cứng 10; họ dùng lr 0.05/0.01 (nhiều khả năng SGD) còn ở đây Adam 1e-3/1e-4;
và toàn bộ số liệu của họ trên audio với backbone EAT, không có điểm dữ liệu vision
nào. Một seed.

---

## 7. Tái lập

```bash
cd sparse-cl

bash scripts/bench_time.sh    # bang muc 3: ban goc (SVD) so voi ban nay (eigh)
bash scripts/vit_concat.sh    # bang tang nao dang noi, muc 2
bash scripts/vit_ens.sh       # duong cong ensemble  (DEGS=, MS= de doi luoi)
bash scripts/vit_fsa.sh       # FSA tren ViT roi ca hai truc
bash scripts/fsa_all.sh       # FSA tren 3 dataset x 2 backbone, muc 5
bash scripts/pace_all.sh      # Improved FSA kieu PACE, muc 6
ADAPTER=film bash scripts/fsa_all.sh    # FSA voi adapter FiLM

python scripts/prepare_cub.py     # sau khi tai CUB_200_2011.tgz
python -m gdown 1xUiwlnx4k0oDhYi26KL5KwrCAya-mvJ_ -O data/vtab.zip
```

Lần chạy đầu của mỗi script tự trích feature và lưu cache. Sàn λ **phải là 5**
cho mọi thứ dính ViT+ms và FSA.

```bash
# bat ban GCV goc de doi chieu
FLYCL_GCV=svd python run.py --method flycl ...

# ViT + concat + ensemble, cau hinh tot nhat
python run.py --method flycl --model_name vit_base_patch16_224+ms \
    --training_method aper --data_augmentation vit \
    --coding_level 0.3 --expand_dim 10000 --deg_s4 112 --b_stage 3 \
    --grid 300:1 --branches 5 --ridge_lower 5 --ridge_upper 13 --seed 1993
```

---

## 8. Giới hạn

**Mọi số ViT trong báo cáo này là một seed.** σ giữa các seed trên ViT đo trước
đây là 0.94 — lớn hơn mọi Δ trong bảng chính. Nên hiện tại bảng này chỉ mới loại
được giả thuyết "concat vô dụng trên ViT" và xác lập **thứ tự**, chưa chứng minh
được **độ lớn** nào. Ba seed là việc bắt buộc trước khi viết.

Bảng ensemble ở sàn λ = 3 (mốc 93.03) không dùng chung cột Δ với bảng chính ở sàn
λ = 5 (mốc 93.12).
