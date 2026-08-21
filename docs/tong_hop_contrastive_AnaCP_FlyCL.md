# Tổng hợp: Contrastive Learning trong CIL → AnaCP → Kế hoạch ghép với Fly-CL

*(Tổng hợp toàn bộ cuộc trao đổi, 20/08/2026)*

---

## Phần I — Nền tảng khái niệm

### 1. EWC và giới hạn của regularization

**EWC vanilla so với fine-tune trần:** chỉ nhỉnh hơn vài điểm phần trăm, cả hai đều suy giảm nặng.

Số liệu từ survey CIL (TPAMI 2024):

| Setup | Finetune | EWC |
|---|---|---|
| CIFAR-100 Base50-Inc25 (session cuối) | 20.88 | 23.85 |
| ImageNet-100 Base0-Inc5 (session cuối) | 4.70 | 6.14 |

*(LwF cùng setup ImageNet-100 đạt 17.74)*

**Nguyên nhân sâu (theo EWC-DR):** khi model hội tụ và tự tin cao ($p_c \to 1$), gradient $\to 0$ → FIM và importance $\Omega$ gần 0 → regularization gần như "tắt", EWC thoái hóa thành fine-tune trần ở đúng những weight quan trọng nhất.

**Trường hợp EWC còn tệ hơn không regularize:**
- **Progress & Compress** (DeepMind, RL/Atari): Finetune 117/125/117/105/98 vs EWC 55/53/53/50/54 — EWC bóp nghẹt khả năng học task mới
- $\lambda$ quá lớn → model đóng băng, task mới cũng học kém
- RL / non-stationary: Fisher noisy, $\Omega$ không phản ánh đúng importance
- Long sequences: $\sum_t \Omega^t$ tích lũy → mất plasticity hoàn toàn
- Domain-IL: Fisher từ task cũ không transfer sang task mới

---

### 2. Contrastive loss — bản chất

**Định nghĩa:** hàm mất mát học biểu diễn, kéo positive gần, đẩy negative xa trong không gian embedding. **Không phân loại** — chỉ định hình cấu trúc feature space.

**Dạng cơ bản (Chopra 2005):**
$$L = y \cdot D^2 + (1-y)\max(0, m - D)^2$$

**InfoNCE (SimCLR, CLIP, MoCo):**
$$L = -\log\frac{\exp(\mathrm{sim}(x,x^+)/\tau)}{\sum_i \exp(\mathrm{sim}(x,x_i)/\tau)}$$

Về bản chất là cross-entropy trên bài toán "đâu là positive trong N+1 candidates". Càng nhiều negative càng tốt → cần batch lớn.

**SupCon (Khosla 2020):**
$$L_{scl} = \sum_i \frac{-1}{|P(i)|}\sum_{p\in P(i)}\log\frac{\exp(z_i\cdot z_p/\tau)}{\sum_{a\in A(i)}\exp(z_i\cdot z_a/\tau)}$$

Positive định nghĩa bằng **label** thay vì augmentation.

**Ứng dụng:** SSL (SimCLR/MoCo/BYOL), metric learning (FaceNet, re-ID), multimodal (CLIP), sentence embedding (SimCSE), recommendation, graph, continual learning.

**Điểm quan trọng:** contrastive loss **luôn cần gradient**. Không có biến thể "training-free contrastive loss". Muốn tách class mà không train thì phải đổi cơ chế hoàn toàn (random projection, ETF hình học).

---

### 3. Hai trường phái tách class

| | (a) Hard/training-free | (b) Learned contrastive |
|---|---|---|
| Đại diện | Fly-CL, RanPAC | Co2L, GPLASC, SupCon |
| Cơ chế | Random projection chiều cao + top-k | Loss + gradient descent |
| Nền toán | Johnson-Lindenstrauss, xác suất chiều cao | Học có giám sát |
| Cần data? | Không | Có |
| Cần label? | Không | Có |
| Risk forgetting | Không | Có |
| Chất lượng tách | "Đủ tốt" theo xác suất | Tối ưu theo data |

**Điểm giao:** first-session adaptation (train task 0 rồi đóng băng), guided random projection, và **AnaCP** — analytic contrastive.

---

### 4. Taxonomy representation-based CL

| Nhánh | Train ở đâu | Đại diện |
|---|---|---|
| Frozen backbone | 0 lần | Fly-CL, RanDumb |
| First-session adaptation | Chỉ task 0 | FSA, APER, RanPAC, AnaCP |
| Continuously training | Mọi task | Co2L, GPLASC, ProNC |

**"Training-free" không có nghĩa toàn bộ pipeline không train** — mà là **giai đoạn incremental (task 1→T) không train**. Task 0 vẫn có thể train.

---

### 5. Vì sao W của Fly-CL phải cố định

Không phải lựa chọn tùy ý mà là **điều kiện bắt buộc**:

1. **Phép tích lũy $G_t = G_{t-1} + H_t^\top H_t$ chỉ hợp lệ nếu mọi $H_t$ cùng không gian.** Nếu mỗi task có $W_t$ riêng, chiều thứ $i$ ở task 1 và task 2 là hai tổ hợp tuyến tính khác nhau → cộng dồn vô nghĩa.

2. **Inference cần so sánh across-task.** Class ở task 1 và task 2 phải nằm chung hệ tọa độ mới so được.

**Cùng kích thước KHÔNG đủ** — phải cùng $W$ cụ thể. Đây là bẫy nguy hiểm: code chạy không lỗi nhưng kết quả sai.

---

## Phần II — Khảo sát paper (2025–2026)

### Nhóm "contrastive full" (train xuyên suốt)

| Paper | Venue | Code | Ghi chú |
|---|---|---|---|
| **GPLASC** | FCS 2025 (Q1, IF 4.11) | ❌ | ETF liên-task cố định + R2SCL nội-task; +5% vs Co2L |
| **FNC²/MNC³L** | WACV 2025 | ❌ | Focal Neural Collapse Contrastive, không buffer |
| **Projection-Enhanced CL** | ICLR 2026 | ❌ | **BỊ REJECT** |
| **ProNC** | ICLR 2026 | ✅ | ETF mở rộng dần; CE-based, không phải contrastive |
| **IPAL** | 2025 | — | Continual graph learning |

### Nhóm "contrastive adapt" (chỉ task đầu)

| Paper | Venue | Code | Ghi chú |
|---|---|---|---|
| **AnaCP** | NeurIPS 2025 Spotlight | ✅ | **Ưu tiên số 1** |
| **MoTiC** | 2025 | ✅ | FSCIL |
| **FACL** | 2025 | — | FSCIL |
| **BR-FSCIL/HierCon** | J.Imaging 2025 | — | FSCIL |

### Dòng chảy Co2L

Co2L (ICCV 2021) → CILA (ICML 2024) → CCLIS (AAAI 2024) → GPLASC (2025) → FNC² (WACV 2025)

Xu hướng: chuyển sang **ETF/neural collapse** để không cần replay buffer.

---

### Bài học từ ICLR 2026 bị reject (OpenReview yCXNIgV6ce)

Tác giả: Jeevan Thapa, Ruochen Shi, Rui Li. Điểm: 2, 2, 4, 6. **Decision: Reject** (26/01/2026).

**Bảng forgetting do chính tác giả cung cấp (CIFAR100-10):**

| Setting | Forgetting |
|---|---|
| LR only | 12.17 |
| LR + Contrastive | 15.26 |
| LR + Contrastive + Old-class Rep | **15.82** |

→ **Thêm contrastive làm forgetting TĂNG.**

Reviewer 232d chốt: *"the performance gain comes largely from a significant boost in the initial accuracy of each task instead of reducing forgetting... It feels more like a representation learning technique applied to a continual learning setting."*

**Ý nghĩa với hướng của mình:** contrastive là **kỹ thuật representation learning**, không phải kỹ thuật continual learning. Nhưng trong pipeline analytic (Fly-CL/AnaCP), forgetting đã được giải bằng cơ chế khác → vai trò "chỉ cải thiện representation" **lại đúng là điều cần**. Đây có thể là luận điểm framing tốt.

**Hai paper phát sinh từ review (đáng đọc):**
- **DPCR** — He et al., ICML 2025, closed-form linear regression + SVD trên uncentered covariance
- **REAL** — He et al., arXiv:2403.13522, analytic learning cho EFCIL

Cùng nhóm Run He, nằm đúng giao điểm analytic + EFCIL.

---

## Phần III — Kiến thức toán nền

### Covariance

$$\Sigma = \mathrm{Cov}(x) = E[(x-\mu)(x-\mu)^\top]$$

- Đường chéo $\Sigma_{ii}$ = phương sai chiều $i$
- Ngoài đường chéo $\Sigma_{ij}$ = tương quan giữa chiều $i$ và $j$
- Luôn đối xứng, nửa xác định dương
- $\Sigma$ đường chéo → các chiều độc lập; $\Sigma = I$ → đã whiten

### Whitening

$$z = W(x-\mu), \quad \mathrm{Cov}(z) = I \iff W^\top W = \Sigma^{-1}$$

**Không duy nhất** — $QW$ cũng là nghiệm với mọi $Q$ trực giao.

| Loại | Công thức | Đặc điểm |
|---|---|---|
| **ZCA** (AnaCP dùng) | $\Sigma^{-1/2} = U\Lambda^{-1/2}U^\top$ | Đối xứng, giữ hướng gốc tối đa |
| PCA | $\Lambda^{-1/2}U^\top$ | Xoay về hệ trục PCA |
| Cholesky | $L^{-1}$ | Rẻ nhất, phụ thuộc thứ tự chiều |

**Ý nghĩa hình học:** elip nghiêng → xoay (khử tương quan) → co giãn (chuẩn hóa phương sai) → hình cầu.

**Quan hệ Mahalanobis:**
$$\|\Sigma^{-1/2}(x-y)\|^2 = (x-y)^\top\Sigma^{-1}(x-y) = d_M(x,y)^2$$

Euclid **sau** whitening = Mahalanobis **trước** whitening.

**Vấn đề số học:** $\Sigma$ suy biến khi $n < d$. Xử lý bằng clamp eigenvalue hoặc Ledoit-Wolf shrinkage.

```python
Sigma = 0.5 * (Sigma + Sigma.T)
lam, U = torch.linalg.eigh(Sigma)
lam = torch.clamp(lam, min=1e-6)
Sig_inv_half = U @ torch.diag(lam.rsqrt()) @ U.T
Sig_half     = U @ torch.diag(lam.sqrt())  @ U.T
```

### SVD

$$A = U S V^\top$$

**Hình học:** mọi biến đổi tuyến tính = xoay ($V^\top$) → giãn ($S$) → xoay ($U$).

**Dạng tổng:** $A = \sum_k \sigma_k u_k v_k^\top$ (tổng các ma trận rank-1, sắp theo độ quan trọng)

**Với ma trận đối xứng nửa xác định dương:** SVD ≡ eigendecomposition ($U = V$, $\sigma_i = \lambda_i$)

**Tìm cơ sở trực giao:** QR (rẻ nhất) / SVD (thêm singular values) / eigh (ma trận đối xứng). Gram-Schmidt không ổn định số học, không dùng thực tế.

### Neural Collapse & ETF

**4 hiện tượng NC** (Papyan, Han, Donoho, PNAS 2020) khi train quá điểm zero error:
1. Within-class variance → 0
2. Class mean → simplex ETF
3. Classifier weight ∥ class mean
4. Classifier ≡ NCM

**Simplex ETF:** $K$ vector cùng norm, $\cos(m_i, m_j) = -\frac{1}{K-1}$ (nhỏ nhất có thể)

$$E = \sqrt{\tfrac{K}{K-1}}\, U\left(I_K - \tfrac{1}{K}\mathbf{1}\mathbf{1}^\top\right)$$

```python
def simplex_etf(d, K):
    A = torch.randn(d, K)
    U, _ = torch.linalg.qr(A)
    P = torch.eye(K) - torch.ones(K,K)/K
    M = math.sqrt(K/(K-1)) * (U @ P)
    return M / M.norm(dim=0, keepdim=True)
```

**Vì sao hấp dẫn trong CL:** prototype cố định → không drift → không cần buffer để tạo negative.

**Nhược:** phải biết trước tổng số class. ProNC giải bằng cách mở rộng ETF dần.

---

## Phần IV — AnaCP chi tiết

**Saleh Momeni, Changnan Xiao, Bing Liu — NeurIPS 2025 Spotlight — arXiv:2511.13880 — [code](https://github.com/SalehMomeni/AnaCP)**

### Điểm mấu chốt

**AnaCP KHÔNG dùng contrastive loss.** Nguyên văn: *"applying this directly to CIL would lead to CF due to iterative gradient updates. We achieve similar effects via an analytic CP layer."*

| Contrastive loss thật | AnaCP thay bằng |
|---|---|
| Kéo positive gần | Ridge regression về target prototype (§4.1) |
| Đẩy negative xa | Whitening + SVD + nhiễu loạn $S$ (§4.2) |

Không anchor, không positive/negative pair, không $\tau$, không backward pass.

### Kiến trúc

```
PTM đóng băng (input layer, d chiều)
  → Random Projection: Z = GELU(XR), D = 5000
  → Contrastive Projection (CP) layer          ← ĐÓNG GÓP MỚI
  → Classifier (NCM hoặc ELM ridge)
```

### §3 — Background

Ridge: $W = (Z^\top Z + \lambda I)^{-1}Z^\top Y$

Nguồn gốc: $\min_W \|ZW-Y\|^2 + \lambda\|W\|^2$, đạo hàm = 0.

- $G = Z^\top Z$ (Gram matrix)
- $H = Z^\top Y$ (cross matrix)

Tích lũy: $G_t = G_{t-1} + X_t^\top X_t$, $H_t = H_{t-1} + X_t^\top Y_t$ (cần zero-pad khi số class tăng)

Random projection: $Z = \phi(XR)$, hoạt động như kernel, mở rộng không gian và tạo tương tác phi tuyến.

### §4.1 — Positive Alignment

**Quan sát then chốt:** khi $T$ là one-hot,

$$H = Z^\top T = MN \qquad (8)$$

Chứng minh: cột $c$ của $Z^\top T$ = $\sum_{i\in c} z_i$ = $n_c m_c$

- $M \in \mathbb{R}^{D\times C}$ — class prototype (mean của random feature)
- $N = \mathrm{diag}(n_1,\ldots,n_C)$ — số mẫu mỗi class

**Mở rộng:** thay one-hot bằng target prototype $P \in \mathbb{R}^{C\times d}$:

$$H = Z^\top T = \sum_c\left(\sum_{i\in c}z_i\right)p_c^\top = \sum_c m_c n_c p_c^\top = MNP \qquad (9)$$

**Vì sao được phép thay:** ridge regression **không biết $T$ là nhãn**. Nó chỉ tìm $W$ sao cho $ZW \approx T$. $T$ chứa gì cũng được — giá nhà, tọa độ, prototype. One-hot chỉ là **một lựa chọn**, không phải yêu cầu.

**Trong code:**
```python
# thay vì
T = one_hot(y, C)        # (N, C)
# dùng
T = P[y]                 # (N, d) — tra bảng theo nhãn
```

**Vì sao đây là "positive alignment":** mọi mẫu class $c$ có chung đích $p_c$ → ridge ép chúng về cùng điểm → co cụm.

**Vì sao chiều $P$ = $d$:** *"to preserve the geometric structure of the representations"* — giữ cấu trúc ngữ nghĩa PTM đã học (mèo gần hổ, xa xe hơi). One-hot vứt bỏ điều này (mọi class cách đều nhau).

### §4.2 — Negative Repulsion

**Vấn đề:** nếu 2 class mean vốn đã gần nhau, kéo về đó vẫn nhầm.

**Eq. 10 — tích lũy covariance:**
$$\Sigma_t = \frac{N_{t-1}}{N_t}\Sigma_{t-1} + \frac{1}{N_t}\sum_{c\in C_t}\sum_{i\in c}(x_i-\mu_c)(x_i-\mu_c)^\top$$

Trung bình có trọng số → đúng bằng covariance trên toàn bộ data, không cần lưu data cũ. Trừ $\mu_c$ (mean của class) → **within-class covariance**.

**Eq. 11 — whitening:**
$$\hat{C} = \Sigma^{-1/2}C$$

*"a large variance in some dimensions can give a false impression of class separability"* — chiều phương sai cao cần khoảng cách lớn hơn mới có ý nghĩa tương đương.

**Eq. 12 — SVD:**
$$\hat{C} = USV^\top$$

- $U$ $(d\times C)$ — hướng các class mean trải theo
- $S$ — độ lớn trải theo từng hướng
- $V$ — cách từng class phân bổ

**Quan trọng — vì sao KHÔNG dùng ETF:**

> *"In principle, maximum separation would be achieved if the means were arranged uniformly on the surface of a hypersphere. However, this strict configuration would significantly alter the original data geometry, distorting the feature representations."*

→ Tác giả **biết** về ETF, **cân nhắc**, và **cố tình từ chối**. Đây là khác biệt triết lý với GPLASC/ProNC/NC-FSCIL.

**Eq. 13 — nhiễu loạn:**
$$\tilde{S} = S + \alpha\,\mathrm{Diag}\{\delta_1,\ldots,\delta_C\}$$

Vì $\delta_i \in \{-1,0,1\}$: $\tilde\sigma_i \in \{\sigma_i-\alpha,\ \sigma_i,\ \sigma_i+\alpha\}$

**Lemma 4.1:**

Đặt $SV^\top = (w_1,\ldots,w_C)$, $V^\top = (e_1,\ldots,e_C)$.

Hàm mục tiêu: $\mathcal{J} = \sum_i\sum_{j\neq i}|\theta(w_i,w_j)|$

Xét $f_i(\alpha) = \sum_{j\neq i}\dfrac{|\langle w_j, w_i+\alpha\delta_i e_i\rangle|}{\|w_j\|\cdot\|w_i+\alpha\delta_i e_i\|}$

Đạo hàm tại $\alpha=0$ (dùng $\frac{d|x|}{dx} = 2\cdot\mathbf{1}_{x\geq0}-1$):

$$f'_i(0) = \frac{\delta_i}{\|w_i\|^3}\Phi_i$$

$$\Phi_i = \sum_{j\neq i}\frac{1}{\|w_j\|}\Big[(2\cdot\mathbf{1}_{\langle w_i,w_j\rangle\geq0}-1)\langle e_i,w_j\rangle\|w_i\|^2 - \langle e_i,w_i\rangle|\langle w_i,w_j\rangle|\Big]$$

Chọn $\delta_i = -\mathrm{sign}(\Phi_i)$ để $f'_i(0) \leq 0$.

**Bản chất: một bước gradient descent, tính bằng công thức đóng, chỉ lấy dấu.**

Hai hạng tử trong $\Phi_i$: (A) ảnh hưởng tử số cosine, (B) ảnh hưởng mẫu số ($\|w_i\|$).

Dùng $|\theta|$ vì cosine âm cũng là "gần" (cùng đường thẳng) — muốn về vuông góc.

**Lỗi trong paper:** phát biểu Lemma viết $(2\cdot\mathbf{1}_{\cdot\geq0})$ thiếu $-1$; Appendix A viết đúng.

**Về $\alpha$:** *"if too small, the shift becomes negligible; if too large, the reduction may no longer be valid"* — vì đạo hàm chỉ đúng cục bộ. Đặt $\alpha=1$, **không ablate**.

**Eq. 15 — bảo toàn qua $U$:**
$$\langle Uw_i, Uw_j\rangle = w_i^\top U^\top U w_j = \langle w_i,w_j\rangle$$

→ $\theta(Uw_i,Uw_j) = \theta(w_i,w_j)$. Nhờ đó kết luận trong $\mathbb{R}^C$ chuyển lên $\mathbb{R}^d$.

**Eq. 16 — de-whiten:**
$$\hat{\tilde{C}} = U\tilde{S}V^\top, \qquad \tilde{C} = \Sigma^{1/2}\hat{\tilde{C}}, \qquad P = \tilde{C}^\top$$

**Vì sao phải de-whiten:** $P$ dùng làm đích cho ridge, mà ridge làm việc trên feature **chưa whiten**. Nếu để nguyên → hai bên khác hệ tọa độ, **code chạy không lỗi nhưng sai**.

**Chuỗi biến đổi:**
$$C \xrightarrow{\Sigma^{-1/2}} \hat{C} \xrightarrow{\text{SVD}} USV^\top \xrightarrow{S\to S+\alpha\Delta} U\tilde{S}V^\top \xrightarrow{\Sigma^{1/2}} \tilde{C} = P^\top$$

**Về Mahalanobis:** đo cosine trên data đã whiten ≡ đo Mahalanobis — dẫn FeCAM.

### §4.3 — Classifier

**Multi-head (Eq. 17):**
$$u^{(h)} = \phi(xR^{(h)})W^{(h)}, \qquad u = \frac{1}{H}\sum_h u^{(h)}$$

Các head **chia sẻ chung $P$** → output cùng không gian → trung bình có nghĩa.

$H=1$: 91.99 | $H=3$: 92.15 | $H=5$: 92.20 → lợi ích giảm nhanh. *"even with H=1, AnaCP outperforms all baselines"*

**NCM:** $\hat{y} = \arg\min_c \|u - p_c\|$. Không tham số, không cần train.

**ELM:** mạng 1 lớp ẩn, weight random cố định, output giải least squares. Chính là thứ Fly-CL đang dùng.

NCM 91.86 → ELM 92.15 (trung bình +0.66%).

**Vấn đề CF của classifier:**

> *"Unlike PTM features and their random projections, which remain fixed, the CP outputs evolve as new tasks are introduced... Directly updating the ELM classifier incrementally may therefore lead to CF."*

| Tầng | Input | Cố định? | Tích lũy được? |
|---|---|---|---|
| $\mu_c$, $\Sigma$ | feature PTM | ✅ | ✅ |
| $M$, $N$, $G$ | $Z$ (sau RP) | ✅ | ✅ |
| $W_{CP}$ | từ $M,N,P$ | — | ✅ |
| **Classifier** | $u = ZW_{CP}$ | ❌ | ❌ |

**Quy tắc: input của tầng nào cố định thì tầng đó tích lũy được.**

**Giải pháp — Gaussian pseudo-replay:**

Sau mỗi task, train lại classifier từ đầu:
1. Mọi class đã gặp: sinh mẫu giả $\tilde{x} \sim \mathcal{N}(\mu_c, \Sigma)$, $R=100$ mẫu
2. Đưa qua RP + CP **phiên bản mới nhất**
3. Giải ELM từ đầu

**Vì sao được:** $\mu_c$, $\Sigma$ ở input layer, PTM đóng băng thật → **không bao giờ lỗi thời**.

Covariance chung thay vì riêng từng class: giảm bộ nhớ 100 lần, chênh accuracy chỉ 0.00–0.36%.

### §5 — Thực nghiệm

**Kết quả (DINO-v2, 10 task, 3 seed):**

| Dataset | AnaCP ($A_{avg}$/$A_{last}$) | RanPAC | Joint FT |
|---|---|---|---|
| CIFAR100 | 95.43 / **92.15** | 94.10 / 90.09 | 93.35 |
| ImageNet-R | 90.37 / **86.60** | 87.96 / 84.41 | 88.79 |
| CUB | 93.84 / **90.57** | 92.37 / 88.44 | 89.87 |
| TinyImageNet | 91.57 / **87.35** | 89.55 / 84.67 | 88.81 |
| Cars | 94.16 / **90.65** | 91.48 / 87.48 | 89.92 |

Vượt cận trên joint fine-tuning trên CIFAR100 và CUB. Giảm lỗi tương đối 4.8–31.4%.

**MoCo-v3 (PTM yếu hơn):** CIFAR100 83.70, khoảng cách tới joint giãn lên ~6%. Nhưng *"AnaCP outperforms the baselines by a larger margin with MoCo-v3... as the weaker features make the CP layer more impactful"*.

**Ablation (Table 3):**

| So sánh | Chênh lệch |
|---|---|
| Bỏ CP layer (row 1 vs 3) | **−2.93% đến −19.97%** |
| Tắt NR (row 2 vs 3) | −0.38% đến −3.23% |
| NCM → ELM (row 3 vs 6) | +0.66% trung bình |
| $R$ = 20/50/100 | 92.14/92.13/92.15 (gần như không đổi) |
| $H$ = 1/3/5 | 91.99/92.15/92.20 |
| $D$ = 1000/2000/5000/10000 | 90.64/91.35/92.15/92.75 |

**Phân tích CF (§5.3):** dùng setting TIL, in ma trận $A[t][i]$. Cột task 1 qua 10 bước: 0.987 → 0.987 → 0.989 → ... → 0.986. Gần như phẳng.

Lý do dùng TIL: trong CIL accuracy tất yếu giảm khi thêm class (bài toán khó hơn), không phân biệt được với forgetting thật.

**Thời gian (§5.5) — CIFAR100 + DINO-v2:**

| Bước | Thời gian |
|---|---|
| **FSA** | 6 phút 5 giây (65%) |
| Extract feature | 3 phút 8 giây |
| **Lõi AnaCP** | **5 giây** |
| Tổng | 9 phút 18 giây |

So sánh: CODA-Prompt 3h47m, joint fine-tuning 1h58m.

**Bộ nhớ:** $C=200$, $d=768$, $D=5000$, $H=3$ → ~106.6M tham số, phần lớn là Gram matrices $D\times D$ (không tăng theo số class). Float8 → ~102 MiB.

**Discussion (§5.6):** *"a strong PTM is the key to CL, and representation-level forgetting is not the primary limitation"* + liên hệ neuroscience (não giữ biểu diễn ổn định, PTM ≈ kết quả tiến hóa).

### FSA — điểm dễ bỏ sót

**FSA = First Session Adaptation** (Panos et al., ICCV 2023, arXiv:2303.13199): fine-tune backbone chỉ ở task 1 bằng gradient, rồi đóng băng vĩnh viễn.

AnaCP có dùng: *"We also adopt FSA following [56] before applying AnaCP to the frozen PTM as it helps improve accuracy."* — nhưng chỉ ghi ở Implementation Details, không phải Methodology.

**Vì sao đáng chú ý:**
- Paper quảng cáo "no gradient-based training", nhưng FSA **là** gradient training và chiếm 65% compute
- **Không ablate FSA** — lỗ hổng rõ nhất của bài
- Fly-CL cố tình bỏ FSA (phê phán RanPAC tốn compute)

**Vì sao bắt buộc đóng băng sau FSA:** analytic tích lũy $G_t = G_{t-1} + Z_t^\top Z_t$ chỉ hợp lệ nếu mọi $Z$ cùng không gian.

### Caveat

- Dùng **DINO-v2 và MoCo-v3** (self-supervised) để tránh information leakage, **không phải** ViT-B/16-IN21K supervised
- **Không có ResNet** — $d=2048$ chưa được test
- Test trên CIFAR100, ImageNet-R, CUB, TinyImageNet, Cars

---

## Phần V — Kế hoạch ghép với Fly-CL

### So sánh pipeline

| Tầng | AnaCP | Fly-CL |
|---|---|---|
| Input | PTM đóng băng, $d$ chiều | Giống |
| Projection | $Z = \mathrm{GELU}(XR)$, dense | $h' = \text{top-}k(Wv)$, sparse |
| **CP layer** | **Có** | **Không** |
| Chọn $\lambda$ | Cố định 100 | **GCV** |
| Giải hệ | Nghịch đảo | **Cholesky** |
| Classifier | NCM / ELM | Streaming ridge |

**Hai pipeline giống nhau trừ tầng CP** → port khả thi.

### Pipeline đầy đủ sau khi ghép

**Giai đoạn 0:**
```
1. PTM pretrained, đóng băng (không FSA — giữ tinh thần Fly-CL)
2. Sample W_proj sparse (m × d), mỗi hàng p phần tử ≠ 0, cố định
```

**Mỗi task t:**
```python
# Bước 1: extract
x = PTM(images)                      # (n, d)

# Bước 2: tích lũy ở INPUT LAYER  [MỚI]
for c in classes_in_task_t:
    Xc = x[y == c]
    mu[c] = Xc.mean(0)
    outer += (Xc - mu[c]).T @ (Xc - mu[c])
N_t = N_prev + n_task
Sigma = (N_prev/N_t)*Sigma + outer/N_t

# Bước 3: sparse projection + top-k  [GIỮ NGUYÊN]
h_prime = top_k(W_proj @ x.T, k).T   # (n, m)

# Bước 4: tích lũy sau projection
G = G + h_prime.T @ h_prime
M[:, c] = mean theo class
N_diag[c] = số mẫu

# Bước 5: tính P  [MỚI]
Sig_inv_half, Sig_half = whiten_matrices(Sigma)
C_hat = Sig_inv_half @ C_mat
U, S, Vh = svd(C_hat, full_matrices=False)
delta = compute_delta(diag(S) @ Vh, Vh)
S_tilde = diag(S) + alpha * diag(delta)
P = (Sig_half @ (U @ S_tilde @ Vh)).T   # (C, d)

# Bước 6: giải CP layer  [LAI]
lam_star = GCV(G, M @ N_diag @ P)        # ← Fly-CL
W_CP = cho_solve(cholesky(G + lam_star*I), M @ N_diag @ P)   # (m, d)

# Bước 7: classifier
# Phương án (a): NCM — không cần gì thêm
# Phương án (b): ELM + pseudo-replay — phải tính lại mỗi task
```

**Inference:**
```python
h = top_k(W_proj @ x, k)
u = h @ W_CP
pred = argmin_c ||u - P[c]||        # NCM
```

### Bảng thay đổi

| Thành phần | Fly-CL gốc | Sau khi ghép |
|---|---|---|
| Backbone, $W_{proj}$, top-k | Giữ nguyên | Giữ nguyên |
| Tích lũy $G$ | Có | Có |
| Tích lũy $S$ (one-hot) | Có | **→ $MNP$** |
| GCV + Cholesky | Có | Có (dùng 2 lần) |
| $\mu_c$, $\Sigma$ | ✗ | **Thêm** |
| CP layer | ✗ | **Thêm** $(m\times d)$ |
| Classifier | Tích lũy vĩnh viễn | Tùy phương án |

### Bê được phần nào

**Bê nguyên:**
- Whitening + SVD + Lemma 4.1 → $P$ (ở input layer, độc lập cách chiếu)
- Tích lũy $\Sigma$, $\mu_c$

**Bê nhưng sửa chiều:**
- Đổi target ridge sang $P$: $S_t$ từ $(m\times C)$ → $(m\times d)$
- NCM classifier

**Phải thiết kế lại:**
- Cơ chế tích lũy classifier (xem 2 phương án dưới)

**Không nên bê:**
- Multi-head ($H=1$ đủ), ELM (kéo theo pseudo-replay), GELU (đã có top-k), $\lambda$ cố định, nghịch đảo trực tiếp

### Hai phương án cho classifier

**(a) Đóng băng CP sau task 1**
- Giữ nguyên toàn bộ cơ chế Fly-CL (tích lũy vĩnh viễn, không cần pseudo-replay)
- Giữ tính order-independent
- Đơn giản nhất
- Mất khả năng thích nghi liên tục

**(b) CP cập nhật mỗi task + pseudo-replay**
- Giống AnaCP
- Phải bỏ tích lũy, tính lại classifier mỗi task
- Phải lưu $\mu_c$ + $\Sigma$

→ Đây là **ablation đáng chạy**: so sánh xem "adaptation liên tục" đáng giá bao nhiêu.

### Ba rủi ro khi implement

**1. Quên tính lại classifier** — code chạy không lỗi, kết quả tệ, khó debug. Kiểm tra: in $W_{CP}$ sau task 1 và 2, nếu khác → bắt buộc tính lại.

**2. $M$ không còn sparse** — sau top-k mỗi mẫu có $k$ vị trí ≠ 0 nhưng **vị trí khác nhau**, nên mean là union → có thể dày đặc. Kiểm tra: `(M != 0).float().mean()`

**3. $\Sigma$ suy biến với ResNet-50** — $d=2048$, CIFAR-100 mỗi task ~5000 mẫu. Kiểm tra: `torch.linalg.cond(Sigma)`, nếu > 1e6 cần shrinkage mạnh.

### Code cần thêm

```python
def whiten_matrices(Sigma, eps=1e-6):
    S = 0.5 * (Sigma + Sigma.T)
    lam, U = torch.linalg.eigh(S)
    lam = torch.clamp(lam, min=eps)
    return (U @ torch.diag(lam.rsqrt()) @ U.T,
            U @ torch.diag(lam.sqrt())  @ U.T)

def compute_delta(W, E):
    """W, E: (C, C), cột i là w_i, e_i"""
    norms = W.norm(dim=0)
    G  = W.T @ W
    EW = E.T @ W
    sign = torch.where(G >= 0, 1.0, -1.0)
    A = sign * EW * norms.pow(2).unsqueeze(1)
    B = EW.diagonal().unsqueeze(1) * G.abs()
    Mmat = (A - B) / norms.unsqueeze(0)
    Mmat.fill_diagonal_(0)
    return torch.sign(-Mmat.sum(dim=1))

def compute_target_prototypes(class_means, Sigma, alpha=1.0):
    """class_means: (d, C) → P: (C, d)"""
    Sig_inv_half, Sig_half = whiten_matrices(Sigma)
    C_hat = Sig_inv_half @ class_means
    U, S, Vh = torch.linalg.svd(C_hat, full_matrices=False)
    delta = compute_delta(torch.diag(S) @ Vh, Vh)
    S_tilde = torch.diag(S) + alpha * torch.diag(delta)
    return (Sig_half @ (U @ S_tilde @ Vh)).T
```

**Metric kiểm chứng:**
```python
def mean_abs_cos(M):
    Mn = M / M.norm(dim=0, keepdim=True)
    G = (Mn.T @ Mn).abs()
    C = M.shape[1]
    return (G.sum() - C) / (C*C - C)

# PHẢI giảm sau khi tách
print(mean_abs_cos(C_mat), "→", mean_abs_cos(C_sep))
```

---

## Phần VI — Kế hoạch ablation

### Nhóm A — Đóng góp từng thành phần (bắt chước Table 3 AnaCP)

| # | Target | Pseudo-replay | Mô tả |
|---|---|---|---|
| A0 | one-hot | ✗ | Fly-CL gốc (baseline) |
| A1 | class mean thô | ✗ | Chỉ positive alignment |
| A2 | class mean đã tách | ✗ | + negative repulsion |
| A3 | class mean đã tách | ✓ | Full |

A2 về mặt toán học **sai** (tích lũy trên feature đã đổi) nhưng chạy vẫn ra số → cho biết sai lệch đó tốn bao nhiêu.

### Nhóm B — Tần suất cập nhật CP *(đóng góp riêng, AnaCP không có)*

| # | CP cập nhật khi nào | Pseudo-replay? |
|---|---|---|
| B1 | Không có CP | ✗ |
| B2 | Chỉ task 1 rồi đóng băng | ✗ |
| B3 | Mỗi 2 task | ✓ |
| B4 | Mỗi task | ✓ |

Nếu B2 ≈ B4 → kết luận mạnh: *"adaptation liên tục không cần thiết khi backbone đủ tốt"* — phản biện chính AnaCP.

### Nhóm C — Tương tác với đặc thù Fly-CL *(chưa paper nào có)*

| Biến | Giá trị | Câu hỏi |
|---|---|---|
| $k$ (top-k) | 1%, 5%, 10%, 100% của $m$ | CP bù được cho top-k gắt không? |
| $m$ | 2000, 5000, 10000 | Chiều cao rồi thì CP còn cần không? |
| Mật độ $W_{proj}$ | sparse vs dense | Sparse làm CP kém đi không? |

**Giả thuyết:** $m$ lớn → RP đã tách đủ → CP đóng góp ít. $m$ nhỏ → CP quan trọng hơn. Nếu đúng: *"CP và chiều chiếu là hai cách thay thế nhau"*.

### Nhóm D — Siêu tham số

| Biến | Giá trị | Ghi chú |
|---|---|---|
| $\alpha$ | 0, 0.5, 1, 2, 5 | **AnaCP không ablate — lỗ hổng rõ** |
| $H$ | 1, 3 | Bắt đầu $H=1$ |
| $R$ | 20, 50, 100 | |
| shrinkage $\epsilon$ | 1e-6, 1e-4, 1e-2 | Quan trọng với ResNet-50 |

### Metric bổ sung

Ngoài $A_{last}$, $\bar{A}$, Forgetting:

- **Mean pairwise cosine giữa prototype** (trước/sau CP) — đại lượng Lemma 4.1 tối ưu, **phải giảm**, nếu không thì implement sai
- **Tỉ lệ intra/inter class distance** — đo cả hai hiệu ứng trong một số

### Thứ tự chạy

1. **Nhóm A** — xác nhận pipeline đúng
2. **Nhóm D** ($\alpha$) — tìm giá trị tốt trước
3. **Nhóm B** — đóng góp riêng
4. **Nhóm C** — tốn nhất, để cuối

Ngân sách hạn chế: A + B là đủ cho một paper.

---

## Phần VII — Lộ trình thực hiện

### Bước 1 — Reproduce AnaCP
Clone [github.com/SalehMomeni/AnaCP](https://github.com/SalehMomeni/AnaCP), chạy CIFAR-100 + DINO-v2.
**Mốc: $A_{last}$ = 92.15**

Nếu không reproduce được thì đừng port — không có mốc so sánh.

### Bước 2 — Đọc `anacp.py`
Đối chiếu implementation thật với công thức. Chú ý:
- Thứ tự cập nhật $P$ và $W_{CP}$ trong cùng task
- $w_i$, $e_i$ là cột hay hàng
- Cách nhiễu loạn singular value chính xác

### Bước 3 — Port sang Fly-CL
Phiên bản tối giản: chỉ §4.1 + §4.2 + NCM + phương án (a).
~60-80 dòng code thêm. Ước lượng 1-2 ngày nếu quen codebase.

### Bước 4 — Chạy ablation A, B

### Bước 5 (tùy chọn) — Thử ETF
Nếu muốn thử ETF của ProNC, dùng nội suy thay vì thay thẳng:

$$P = (1-\beta)\tilde{C}^\top + \beta E$$

$\beta=0$ → AnaCP, $\beta=1$ → ETF. Chạy ablation trên $\beta$. Nếu tối ưu ở giữa → cả hai paper đều sai vì chọn cực đoan.

Không cần code ProNC — chỉ cần hàm ETF (~10 dòng, lấy từ NC-FSCIL) + phép nội suy.

### Ngưỡng quyết định

- Không vượt Fly-CL gốc ≥1% $A_{last}$ → lợi ích phụ thuộc backbone, pivot sang biến thể contrastive-chỉ-first-session
- Mean pairwise cosine không giảm → implement sai, dừng lại debug

### Với ResNet-50

AnaCP chưa test CNN backbone. Đây là **đóng góp phụ có giá trị**: *"AnaCP có generalize sang CNN không?"*

Lưu ý: $d=2048$ → $\Sigma$ lớn hơn 7 lần, eigendecomposition tốn hơn, condition number có thể xấu. Kiểm tra sớm.

---

## Phần VIII — Danh sách paper

### Có code (ưu tiên)

| Paper | Link |
|---|---|
| **AnaCP** | github.com/SalehMomeni/AnaCP |
| **RanPAC** | github.com/McDonnell-Research-Lab/RanPAC |
| **ProNC** | github.com/Continue-Edge-AI-Lab/ProNC |
| **NC-FSCIL** | github.com/NeuralCollapseApplications/FSCIL |
| **MoTiC** | github.com/huangshuai0605/MoTiC |
| **Co2L** | github.com/chaht01/Co2L |
| **PCR** | github.com/FelixHuiweiLin/PCR |
| **RAPF** | github.com/linlany/RAPF |

### Không có code (bỏ qua)
GPLASC, CILA, CCLIS, FNC²/MNC³L, Projection-Enhanced CL (đã reject)

### Đáng đọc thêm
- **DPCR** — He et al., ICML 2025
- **REAL** — He et al., arXiv:2403.13522
- **FSA** — Panos et al., ICCV 2023, arXiv:2303.13199
- **FeCAM** — Goswami et al., NeurIPS 2023 (nguồn Eq. 10 của AnaCP)
- **SupCon** — Khosla et al., NeurIPS 2020

### Negative results (đọc để tránh cạm bẫy)
- arXiv:2304.00933 — SupCon quên nhanh dù transferability cao
- Davari et al., CVPR 2022, arXiv:2203.13381 — Probing representation forgetting
- NeurIPS 2021 — Contrastive learning và shortcut solutions
- arXiv:2503.17024 — SupCon suy giảm khi mất cân bằng lớp

---

## Phụ lục — Ghi chú nhanh

**Ký hiệu dễ nhầm:**
- $C$ vừa là ma trận class mean $(d\times C)$, vừa là số class — paper viết ẩu
- PyTorch `svd` trả `Vh` = $V^\top$, không phải $V$
- PyTorch `svd` trả `S` là **vector**, cần `torch.diag(S)`

**Hàm PyTorch:**
- `torch.linalg.eigh` — eigendecomposition ma trận đối xứng, eigenvalue tăng dần, eigenvector trực chuẩn. Chỉ đọc tam giác dưới → phải ép đối xứng trước.
- `torch.linalg.svd(A, full_matrices=False)` — thin SVD
- `torch.linalg.qr` — cơ sở trực chuẩn (rẻ nhất)

**Chi phí:**
- `eigh` trên $\Sigma$: $O(d^3)$ — ViT 768 → <1s, ResNet 2048 → vài giây
- SVD trên $\hat{C}$ $(d\times C)$: $O(dC^2)$ — rẻ hơn nhiều

**Kết luận cốt lõi:**
> Contrastive loss là kỹ thuật **representation learning**, không phải kỹ thuật **continual learning**. Trong pipeline analytic, forgetting đã được giải bằng cơ chế khác — nên vai trò "chỉ cải thiện representation" lại đúng là điều cần.
