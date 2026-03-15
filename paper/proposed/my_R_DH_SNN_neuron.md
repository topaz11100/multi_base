# my_R_DH_SNN_neuron — Reverse DH-SNN + soma-dense 입력(KM) + dendrite-broadcast + IF soma + 가변 가지 수

## 0. 목적

Reverse DH-SNN 은 다중 수상돌기 필터와 세포체 집계를 분리하되, 레이어 연결은 일반 dense 처럼 두고 뉴런 내부 가지에는 동일한 입력을 broadcast 하는 구조를 뜻한다.

본 명세의 핵심 변경점은 세포체 동역학을 LIF 가 아니라 IF + soft reset 으로 두는 것이다. 따라서 세포체는 사실상 집계 전류 메모리와 스파이크 생성기의 역할만 담당한다.

정리하면 본 구조는 다음 세 규칙을 따른다.

- 레이어 연결은 세포체 기준 dense 입력 투영 $W_{\mathrm{in}}\in\mathbb{R}^{M\times K}$ 로 계산한다.
- 같은 뉴런 내부의 모든 가지는 동일한 세포체 기준 입력을 broadcast 로 공유한다.
- 수상돌기 전류는 가지별 EMA 로 유지하고, 세포체는 누설 없이 적분한 뒤 soft reset 으로 스파이크를 생성한다.

가변 가지 수 $s$ , 소프트 마스크, 상관 정규화의 공통 이론은 `varidble_dendric.md` 를 따른다.

---

## 1. soma-dense 입력 투영(KM) + dendrite-broadcast

이전 레이어 세포체 수를 $K$ , 현재 레이어 세포체 수를 $M$ , 사전 할당 가지 수를 $D$ 라 둔다.

### 1.1 fully-dense branch 입력과의 비교

수상돌기 단위를 $(m,d)$ 로 펼친 완전 branch-dense 연결은

- 입력 $O[t]\in\mathbb{R}^{K}$
- 펼친 branch 입력 $O_{\mathrm{branch,flat}}[t]\in\mathbb{R}^{MD}$
- 가중치 $W_{\mathrm{in,flat}}\in\mathbb{R}^{(MD)\times K}$

로 표현되며 파라미터 수는 $KMD$ 이다.

### 1.2 dendrite-broadcast 제약

본 구조에서는 레이어 연결을 먼저 세포체 기준으로 계산하고, 같은 뉴런 내부 가지 방향으로만 가중치를 공유한다.

$$
W_{\mathrm{in,flat}}[(m,d),k] = W_{\mathrm{in}}[m,k],
\qquad
\forall m, d, k
$$

여기서 $W_{\mathrm{in}}\in\mathbb{R}^{M\times K}$ 는 일반 dense 레이어 연결 가중치다.

이 제약으로 파라미터 수는 $KMD \rightarrow KM$ 으로 줄어든다.

### 1.3 branch 입력 정의

먼저 세포체 기준 입력을

$$
O_{\mathrm{soma}}[t] = W_{\mathrm{in}}\,O[t]\in\mathbb{R}^{M}
$$

로 정의한다.

그 다음 각 뉴런 $m$ 의 모든 가지에 대해

$$
O_{\mathrm{branch}}[t]_{m,d} = O_{\mathrm{soma}}[t]_m
$$

로 broadcast 한다.

---

## 2. Reverse DH-SNN 뉴런 동역학

단일 뉴런 $m$ 기준으로 사전 할당 가지 수는 $D$ 이고, 유효 가지 수는 연속 파라미터 $s$ 로 제어된다.

### 2.1 뉴런별 입력

단일 뉴런 $m$ 의 입력 row 벡터를

$$
\mathbf{w}_{\mathrm{in},m} = W_{\mathrm{in}}[m,:]\in\mathbb{R}^{K}
$$

라고 하면

$$
O_{\mathrm{soma},m}[t] = \mathbf{w}_{\mathrm{in},m}^{\top} O[t]
$$

이고, 모든 가지는 동일한 $O_{\mathrm{soma},m}[t]$ 를 입력으로 사용한다.

### 2.2 가지 전류(EMA 필터)

$$
i_d[t]
=
\alpha_d\,i_d[t-1]
+
(1-\alpha_d)\,O_{\mathrm{soma},m}[t],
\qquad 0 < \alpha_d < 1
$$

각 가지는 서로 다른 $\alpha_d$ 를 가질 수 있으며, 이 값이 가지별 상태전이계수다.

### 2.3 세포체 입력(가지 전류의 가중합 + $1/s$ 평균)

가지 전류 벡터를 $I[t]=[i_1[t],\ldots,i_D[t]]^{\top}$ 로 두고, 학습 가능한 mixing 가중치 $w_{\mathrm{mix}}\in\mathbb{R}^{D}$ 로 세포체 입력을 정의한다.

$$
H[t]
=
\frac{1}{s}\,w_{\mathrm{mix}}^{\top} I[t]
$$

$1/D$ 가 아니라 $1/s$ 를 사용하는 이유와 소프트 마스크 해석은 `varidble_dendric.md` 를 따른다.

### 2.4 세포체 IF 누적 및 soft reset 스파이크

세포체는 leak 를 두지 않고 집계 입력을 누적한다.

$$
u[t]
=
u[t-1]
+
H[t]
-
u_{\mathrm{th}}\,o[t-1]
$$

$$
o[t] = \Theta\!\left(u[t]-u_{\mathrm{th}}\right)
$$

이 명세에서 세포체는 별도의 leak 계수 $\beta$ 를 갖지 않는다. 따라서 세포체의 역할은 다음 두 가지로 요약된다.

- 가지에서 집계된 입력 $H[t]$ 를 누적하는 전류 메모리
- 임계값 $u_{\mathrm{th}}$ 를 넘을 때 스파이크를 발생시키는 발생기

---

## 3. 레이어 형태(벡터화, 사전 할당 + 마스킹)

뉴런 수 $M$ 인 레이어에 대해

- 사전 할당 가지 수 $D$
- 레이어 입력 가중치 $W_{\mathrm{in}}\in\mathbb{R}^{M\times K}$
- 가지 전류 $I[t]\in\mathbb{R}^{M\times D}$
- 세포체 mixing 가중치 $W_{\mathrm{mix}}\in\mathbb{R}^{M\times D}$
- 세포체 누적 상태 $u[t]\in\mathbb{R}^{M}$
- 출력 spike $o[t]\in\{0,1\}^{M}$

를 사용한다.

### 3.1 branch 입력

$$
O_{\mathrm{soma}}[t] = W_{\mathrm{in}}\,O[t]\in\mathbb{R}^{M}
$$

$$
O_{\mathrm{branch}}[t]_{m,d} = O_{\mathrm{soma}}[t]_m
$$

### 3.2 가변 가지 수 $s$ 는 뉴런 단위

뉴런 $m$ 별 연속 구조 파라미터 $s_m\in[S_{\min},S_{\max}]$ 에 대해

$$
\tilde{s}_m = \min(D,s_m)
$$

$$
D_{m,\mathrm{full}} = \lfloor \tilde{s}_m \rfloor
$$

$$
\rho_m = \tilde{s}_m - D_{m,\mathrm{full}}\in[0,1)
$$

$$
D_{m,\mathrm{int}} = D_{m,\mathrm{full}} + \mathbf{1}[\rho_m > 0]
$$

를 정의한다.

마스크는

$$
G_{m,d}(s_m)=
\begin{cases}
1, & d\le D_{m,\mathrm{full}} \\
\rho_m, & d=D_{m,\mathrm{full}}+1 \\
0, & \text{otherwise}
\end{cases}
$$

로 둔다.

### 3.3 branch 업데이트(마스크 포함)

$$
I[t]
=
G\odot\Bigl(
A\odot I[t-1] + (1-A)\odot O_{\mathrm{branch}}[t]
\Bigr)
$$

여기서 $A\in(0,1)^{M\times D}$ 는 가지별 $\alpha$ 텐서다.

### 3.4 세포체 입력과 IF 업데이트

각 뉴런 $m$ 에 대해 세포체 입력은

$$
H[t]_m
=
\frac{1}{s_m}
\sum_{d=1}^{D}
\Bigl(
W_{\mathrm{mix}}[m,d] \; I[t]_{m,d}
\Bigr)
$$

이고, 세포체 누적 상태와 스파이크는

$$
u[t] = u[t-1] + H[t] - u_{\mathrm{th}}\,o[t-1]
$$

$$
o[t] = \Theta\!\left(u[t]-u_{\mathrm{th}}\right)
$$

로 계산한다.

---

## 4. 파라미터화 요약

### 4.1 $s$ 범위 제약

$$
s_m = S_{\min} + (S_{\max}-S_{\min})\,\sigma(\hat{s}_m)
$$

- $S_{\max}$ 는 구현상 $D$ 보다 클 수 있다.
- 이 경우에도 활성 가지 수는 $D$ 로 clamp 되지만, $1/s_m$ 로 인해 세포체 입력 스케일은 계속 변한다.
- `s_init` 을 따로 주지 않으면 초기값은 $S_{\max}$ 근처에서 시작한다.

### 4.2 가지 상관 억제 정규화

EMA 커널

$$
h_d[n] = (1-\alpha_d)\alpha_d^n
$$

에 대해 코사인 유사도는

$$
\cos(h_i,h_j)
=
\frac{\sqrt{(1-\alpha_i^2)(1-\alpha_j^2)}}{1-\alpha_i\alpha_j}
$$

이고, 활성 가지에 대해서만

$$
\mathcal{L}_{\mathrm{ortho}}
=
\sum_{m=1}^{M}
\sum_{1\le i<j\le D_{m,\mathrm{int}}}
\cos(h_i,h_j)^2
$$

를 사용한다.

---

## 5. 총 손실

task loss 를 $\mathcal{L}_{\mathrm{task}}$ 라 하면

$$
\mathcal{L}_{\mathrm{total}}
=
\mathcal{L}_{\mathrm{task}}
+
\lambda_{\mathrm{ortho}}\mathcal{L}_{\mathrm{ortho}}
+
\lambda_s\mathcal{L}_s
$$

$$
\mathcal{L}_s = \frac{1}{M}\sum_{m=1}^{M} s_m
$$

### 5.1 AdamW weight decay 적용 범위

- 적용 대상: 레이어 연결 가중치 $W_{\mathrm{in}}$, 내부 mixing 가중치 $W_{\mathrm{mix}}$
- 비적용 대상: 구조 파라미터 $s$, 가지 상태전이계수 $\alpha$

R-DH 에서는 세포체 leak 계수 $\beta$ 가 존재하지 않으므로, 세포체 timing factor 분포나 weight decay 예외 항목도 $\beta$ 를 포함하지 않는다.

---

## 6. 학습 안정화: 가중치 초기화

같은 뉴런 내부 가지가 동일한 입력을 받으므로, $1/s$ 평균화만 적용하면 초기 세포체 입력이 과도하게 작아질 수 있다.

따라서 권장 초기화는 다음과 같다.

- $W_{\mathrm{mix}}$ 는 **Kaiming 균등분포 초기화** 를 사용한 뒤, 필요하면 초기 세포체 입력 스케일 보정을 위해 $D$ 배 스케일 업
- $W_{\mathrm{in}}$ 은 일반 dense 와 동일한 fan-in 기반 초기화 사용

즉,

$$
W_{\mathrm{mix}} \leftarrow D\,W_{\mathrm{mix}}
$$

를 권장한다.

---

## 7. 구현 체크리스트

- 입력 라우팅 마스크는 사용하지 않고 레이어 연결은 일반 dense 로 둔다.
- 레이어 입력 가중치는 $W_{\mathrm{in}}\in\mathbb{R}^{M\times K}$ 다.
- 같은 뉴런 내부 가지는 동일 입력을 공유한다.
- $s$ 는 뉴런 단위 파라미터다.
- 구현은 가지 append/pop 이 아니라 사전 할당 + 소프트 마스크를 사용한다.
- 세포체는 IF + soft reset 이며, leak 계수 $\beta$ 는 없다.
- timing factor 는 가지별 $\alpha$ 만 기록한다.
- 세포체는 집계 전류 메모리와 스파이크 생성기의 역할만 맡는다.
