# Improved First-Session Adaptation (PACE, ICLR 2026)

**Nguồn**: Chang Li, Kanglei Zhou, Liyuan Wang — *"PACE: Pretrained Audio Continual Learning"*, arXiv 2602.03355, **Accepted at ICLR 2026**. Department of Psychological and Cognitive Sciences, Tsinghua University.

Tài liệu này chỉ trích phần **Improved FSA** (Sec. 3.2 + Algorithm 1 + Table 3 của paper), bỏ qua MSA và boundary-aware regularization.

---

## 1. Động cơ: hai phát hiện dẫn tới Improved FSA

### Finding 1 — Statistics-based thắng PEFT-based trong audio CL

Các phương pháp PEFT chuyển từ vision sang audio đều tệ: L2P, DualPrompt, S-Prompt++ suy giảm gần **gấp 3 lần** so với ở vision. Lý do: chúng dựa vào biểu diễn dùng chung (shared representation) cho prompt-key matching, kém hiệu quả với cấu trúc phổ chi tiết (fine-grained spectral) của audio.

Ngược lại, phương pháp statistics-based (backbone chỉnh một lần + analytic classifier dựa trên thống kê bậc hai) — RanPAC, ACL — cho kết quả mạnh và ổn định hơn hẳn.

> Kết luận của paper: chọn **analytic classifier + thống kê bậc hai trên backbone đóng băng** làm *foundational technical route* cho pretrained audio CL.

### Finding 2 — Representation saturation

Ba quan sát:

1. RanPAC đạt accuracy session đầu cao **ngay cả khi không có FSA** trên coarse-grained → domain gap giữa pretrain và downstream nhỏ.
2. FSA cải thiện thêm accuracy session đầu nhưng **không cải thiện được accuracy các task tương lai** một cách có ý nghĩa, kể cả khi train kéo dài. Backbone pretrained đã nắm gần hết thông tin liên quan ngay ở session đầu → hết khả năng trích thêm đặc trưng phân biệt cho task sau. Paper gọi hiện tượng này là **representation saturation**.
3. **Đóng băng tầng nông** trong lúc FSA cải thiện kết quả, còn **tune toàn bộ tầng thường làm tệ đi, thậm chí tụt xuống dưới cả mốc không FSA**.

Bằng chứng số trên fine-grained (Table 1):

| Method | TIMIT-2 | VocalSet |
|---|---:|---:|
| w/o FSA | 75.87 | 61.51 |
| Naive FSA | 89.92 | 62.85 |
| **Extended FSA** | **83.25** ↓ | **61.18** ↓ |
| Joint Training | 95.22 | 76.65 |

Extended FSA (train session đầu lâu hơn) làm TIMIT-2 tụt **89.92 → 83.25** — dấu hiệu overfit mạnh vào session đầu.

---

## 2. Ba ý tưởng của Improved FSA

Nguyên văn: *"our improved FSA incorporates three key ideas: (1) restricting updates to the output head so that gradients flow primarily into the backbone; (2) adapting only deeper, semantic-relevant layers; and (3) replacing the trainable head with an analytic classifier after adaptation to ensure stability in later sessions."*

### 2.1 Restricted Head Learning

**Vấn đề**: FSA hiện có train đồng thời linear head với backbone gần như đóng băng → head overfit trong khi backbone không được adapt đủ để tinh chỉnh có ý nghĩa.

**Hai sửa đổi**:

1. **Tối ưu bất đối xứng**: đặt learning rate của head thấp hơn hẳn backbone
   $$\eta_{\text{head}} \ll \eta_{\text{bb}}$$
   Thực tế: $\eta_{bb} = 0.05$, $\eta_{head} = 0.01$ cho mọi task.

2. **Train theo giai đoạn**: train head trước $E_{\text{head}}$ epoch với backbone đóng băng, rồi **cố định head** và fine-tune backbone $E_0$ epoch, với $E_{\text{head}} \ll E_0$.
   Thực tế: $E_{\text{head}} = 1$; $E_0$ tuỳ dataset (ESC-50: 10, US8K: 15, SC2: 1, TIMIT-2/3: 30, VocalSet: 6).

**Ý nghĩa**: sơ đồ bất đối xứng này ép backbone hấp thụ phần lớn tín hiệu gradient.

> **Điểm đáng chú ý**: paper nhấn mạnh chiến lược này **ngược hẳn** với LAE (Gao et al. 2023) và SLCA (Zhang et al. 2023) — hai bài kìm hãm cập nhật backbone để chống quên. Với audio backbone thì phải chủ động *khuyến khích* backbone adapt.

### 2.2 Later Layer LoRA (chọn ranh giới bằng CKA)

**Cơ sở**: cấu trúc phân tầng của audio model — tầng nông mã hoá pattern thời-tần và âm học chung của domain, tầng sâu mã hoá trừu tượng ngữ nghĩa cao hơn, đặc thù task.

**Cách chọn ranh giới $L_{\text{tune}}$**: phân tích representation shift trong lúc full fine-tune, chọn **tầng nông nhất có độ lệch CKA so với model pretrained vượt ngưỡng $\rho_{\text{layer}}$**.

**Áp LoRA cho các tầng $l \in [L_{\text{tune}}, L]$**:
$$W_1^l = W_0^l + A_1^l B_1^l, \quad L_{\text{tune}} \le l \le L$$
với $W_0^l \in \mathbb{R}^{d\times d}$ là trọng số pretrained, $A_1^l \in \mathbb{R}^{d\times r}$, $B_1^l \in \mathbb{R}^{r\times d}$ là ma trận low-rank học được, $r \ll d$.

Tầng $1$ đến $L_{\text{tune}}-1$ bị đóng băng.

### 2.3 Analytic Classifier

Sau khi adapt xong, **vứt head trainable** $h^t(\cdot)$, thay bằng analytic classifier đệ quy exemplar-free (RanPAC, McDonnell et al. 2023) để tránh bias tích luỹ từ head.

Với random projector $W_{\text{proj}} \in \mathbb{R}^{D\times D_{\text{proj}}}$, feature chiếu $\hat Z_t = W_{\text{proj}}Z_t \in \mathbb{R}^{N_t\times D_{\text{proj}}}$, one-hot label $Y_t$, ma trận tự tương quan khởi tạo với regularization $\gamma > 0$:
$$R_t = (\hat Z_t^\top \hat Z_t + \gamma I)^{-1}$$

Cập nhật đệ quy bằng **Woodbury identity** (Eq. 2):
$$R_t = R_{t-1} - R_{t-1}\hat Z_t^\top (I + \hat Z_t R_{t-1}\hat Z_t^\top)^{-1}\hat Z_t R_{t-1}$$

Trọng số classifier cập nhật bằng công thức đóng (Eq. 3):
$$\hat W_t = \hat W_{t-1} - R_t\hat Z_t^\top \hat Z_t \hat W_{t-1} + R_t\hat Z_t^\top Y_t$$

Inference: $\hat y_{i,t} = \phi_t(W_{\text{proj}}z_{i,t}) = \hat z_{i,t}\hat W_t$.

Thực tế: $D_{\text{proj}} = 8192$.

---

## 3. Algorithm 1 — Improved First Session Adaptation

```
Require: backbone pretrained f₀, E₀, E_head, ngưỡng CKA ρ_layer,
         learning rate η_bb > η_head

Stage A (dò ranh giới tầng):
  θ₁^pre ← θ₀
  for epoch = 1..E₀ do
      θ₁^pre ← θ₁^pre − η_bb ∇_θ L_ce(h_ce(f₁^pre(X₁)), Y₁)   {PEFT trên MỌI tầng}
  end for
  for k = 1..L do
      s^k ← CKA(f₁^pre(X₁)^k, f₀(X₁)^k)
  end for
  k* ← argmax_k [ s^k < ρ_layer ]

Stage B (head warm-up, η_head):
  for epoch = 1..E_head do
      h₁ ← h₁ − η_head ∇_h L_ce(h₁(f₀(X₁)), Y₁)
  end for

Stage C (adapt backbone, η_bb):
  Đóng băng h_ce, θ₀;  θ₁ ← θ₀ + θ₁^LoRA
  for epoch = 1..E₀ do
      θ₁ ← θ₁ − η_bb ∇_θ L_ce(h₁(f₁(X₁)), Y₁)
      {chỉ tầng k*+1 (= L_tune) đến L trong f_θ₁ được train}
  end for

Stage D (giai đoạn analytic):
  Đóng băng θ₁;  Vứt h₁;  Khởi tạo thống kê cho φ(·) = f₁(·)
```

**Lưu ý về Stage A**: đây là một lượt PEFT "thăm dò" (probe sensitivity) trên toàn bộ tầng, **chỉ để đo CKA**, không giữ lại trọng số. $L_{\text{tune}} = k^* + 1$ là điểm mà độ tương đồng biểu diễn bắt đầu tụt xuống dưới ngưỡng — nơi việc adapt nên bắt đầu.

---

## 4. Ablation (Table 3) — coarse-grained

| Method | ESC-50 | US8K | SC2 |
|---|---:|---:|---:|
| w/o FSA | 92.50 | 96.49 | 81.22 |
| Naive FSA | 92.25 | 97.08 (+0.61%) | 90.53 |
| Low Learning Rate | 93.75 (+1.35%) | 97.35 (+0.89%) | 90.95 (+11.98%) |
| Learning & Freeze | 94.50 (+2.16%) | 97.38 (+0.92%) | 91.30 (+12.41%) |
| **Our FSA** | **95.75 (+3.51%)** | **97.49 (+1.04%)** | **91.87 (+13.11%)** |

Phần trăm tính so với mốc w/o FSA.

**Đọc bảng này**: mức tăng dồn từng mẹo một — LR bất đối xứng đóng góp phần lớn, staged freeze thêm chút, later-layer LoRA khép lại. Chú ý ESC-50: **Naive FSA (92.25) còn thấp hơn w/o FSA (92.50)** — bằng chứng trực tiếp cho representation saturation.

### Độ nhạy $\rho_{\text{layer}}$ (Table 12)

| $\rho_{\text{layer}}$ | TIMIT-3 | SC2 | VocalSet | ESC-50 | US8K | Avg |
|---|---:|---:|---:|---:|---:|---:|
| 0.90 | 83.41 | 90.96 | 62.82 | 94.50 | 97.38 | 85.01 |
| 0.92 | 85.63 | 90.96 | 62.82 | 94.50 | 97.38 | 86.26 |
| **0.94** | **85.63** | **90.96** | **62.82** | **94.50** | **97.38** | **86.26** |
| 0.96 | 85.63 | 90.54 | 62.82 | 92.25 | 97.38 | 85.72 |
| 0.99 | 85.63 | 90.53 | 60.23 | 92.25 | 97.08 | 85.14 |
| w/o | 85.63 | 90.53 | 60.23 | 92.25 | 97.08 | 85.14 |

Ổn định trong khoảng **0.92–0.95**, tụt dần từ 0.96 trở lên. Chọn $\rho_{\text{layer}} = 0.94$ bằng grid search trên fine-grained.

---

## 5. Thiết lập thực nghiệm liên quan

- **Backbone**: EAT (Chen et al. 2024) — self-supervised audio model, pretrain trên AudioSet-2M (~5000 giờ), kiến trúc ViT với **12 Transformer block**, dựa trên spectrogram masked prediction.
- **Hardware**: 1× NVIDIA A800. Input $512\times128$, clip cắt 5.12 giây, batch size 24.
- **Hyperparameter Improved FSA**: $E_{\text{head}} = 1$, $\rho_{\text{layer}} = 0.94$, $D_{\text{proj}} = 8192$, $\eta_{bb} = 0.05$, $\eta_{head} = 0.01$.
- **Chi phí**: trên coarse-grained, Improved FSA đủ, **không cần cập nhật backbone thêm** → chi phí gần bằng fine-tune thường. Trên fine-grained (khi có thêm MSA), PACE/RanPAC = 1.22× (VocalSet), 2.96× (TIMIT-3), 3.13× (TIMIT-2) — so với HiDe-Prompt/RanPAC = 5.44× đến 146.98×.

---

## 6. Ý nghĩa cho pipeline ResNet-50 / CIFAR-100

**Port được ngay**:

- **Mẹo 1 (restricted head learning)**: hoàn toàn không đặc thù audio. Áp thẳng vào vòng train FSA hiện tại — cho classifier học 1 epoch với LR thấp, đóng băng, rồi mới train adapter. Gần như miễn phí.
- **Mẹo 3 (analytic classifier)**: pipeline Fly-CL đã có sẵn dạng này (sparse projection + ridge), không cần đổi.

**Cần điều chỉnh**:

- **Mẹo 2 (CKA layer selection)**: ResNet-50 không có 12 transformer block mà có 4 stage. Có thể dùng CKA để chọn nên gắn adapter vào stage nào, thay vì gắn đều 16 block như hiện tại. Đây cũng là cách kiểm chứng có nguyên tắc cho phát hiện concat stage 3 + stage 4.

**Khái niệm mang đi được**: "representation saturation" cho một cái tên và một công cụ đo (CKA giữa các tầng trước/sau adapt) để giải thích chính thức hiện tượng FSA gain dao động mạnh giữa các dataset, thay vì chỉ ghi nhận như quan sát thực nghiệm.

**Lưu ý khi trích dẫn**: mọi số liệu trong tài liệu này đều trên audio benchmark (ESC-50, US8K, SC2, TIMIT, VocalSet) với backbone EAT. Paper không có điểm dữ liệu vision nào — nên không thể trích số trực tiếp cho setup ResNet-50/CIFAR-100, chỉ mượn được phương pháp.

**Code**: paper ghi *"We have included the source code with clear instructions, and will release them upon acceptance"* — đã accepted nhưng chưa tìm thấy repo công khai tính đến 8/2026.
