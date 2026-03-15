# my_DH_SNN_neuron — DH-SNN(다중 수상돌기 LIF) 제안 구조 및 총 손실

## 0. 범위

- 기본 뉴런은 Zheng *et al.* (Nature Communications, 2024) 의 DH-LIF / DH-SNN 계열(다중 수상돌기 + 세포체) 을 따른다.
- 본 문서는 **DH 계열 뉴런의 제안 구조(동역학) 와 총 손실** 만 정리한다.
- "가변 가지 수 $s$ " 및 "수상돌기 LTI 해석 + 상관 정규화" 는 별도 문서 **varidble_dendric.md** 에서 공통 이론으로 정리한다.

---

## 1. 표기

- 시간: $t=0,1,2,\ldots$
- 레이어: $l$
- 뉴런(출력 채널) 인덱스: $n\in\{1,\ldots,N^{l}\}$
- 가지(수상돌기) 인덱스: $d\in\{1,\ldots,D\}$
  - **$D$ 는 텐서 shape 을 위한 "사전 할당(max) 가지 수"** 이며, 구현에서는 `branch` 로 고정된다.
- 입력 스파이크(이전 레이어): $\mathbf{o}^{t,l-1}\in\{0,1\}^{N^{l-1}}$
- 출력 스파이크(현재 레이어): $\mathbf{o}^{t,l}\in\{0,1\}^{N^{l}}$
- 가지 전류(상태): $\mathbf{i}^{t,l}\in\mathbb{R}^{N^{l}\times D}$, 원소는 $i_{n,d}^{t,l}$
- 세포체 막전위: $\mathbf{u}^{t,l}\in\mathbb{R}^{N^{l}}$, 원소는 $u_{n}^{t,l}$
- 임계값: $u_{\mathrm{th}}$

학습 파라미터(기본형):

- 가지별 feedforward 가중치: $\mathbf{W}_{d}^{l}\in\mathbb{R}^{N^{l}\times N^{l-1}}$
- 가지별 recurrent 가중치(선택): $\mathbf{U}_{d}^{l}\in\mathbb{R}^{N^{l}\times N^{l}}$
- 가지별 timing factor: $\boldsymbol{\alpha}^{l}\in(0,1)^{N^{l}\times D}$, 원소는 $\alpha_{n,d}^{l}$
- 세포체 timing factor: $\boldsymbol{\beta}^{l}\in(0,1)^{N^{l}}$, 원소는 $\beta_{n}^{l}$

Optimizer regularization(AdamW weight decay) 적용 범위(프로젝트 구현):

- weight decay 적용: 레이어 연결(시냅스) 가중치(예: $\mathbf{W}_{d}^{l}$ )
- weight decay 비적용: 구조 파라미터 $s$, 타이밍 파라미터($\boldsymbol{\alpha}^{l}$ , $\boldsymbol{\beta}^{l}$ )

구조(가변 가지) 파라미터:

- **뉴런 단위 연속 파라미터:** $\mathbf{s}^{l}\in[S_{\min},S_{\max}]^{N^{l}}$, 원소는 $s_{n}^{l}$
- **가지 "추가/삭제" 해석을 위한 분해(소프트 마스크):** - clamp: $\tilde{s}_{n}^{l}=\min(D,s_{n}^{l})$
  - 완전 통과 가지 수(정수): $D_{n,\mathrm{full}}^{l}=\lfloor \tilde{s}_{n}^{l}\rfloor$
  - 소프트 마스킹 잔여분: $\rho_{n}^{l}=\tilde{s}_{n}^{l}-D_{n,\mathrm{full}}^{l}\in[0,1)$
  - 마스크가 0이 아닌 가지 개수: $D_{n,\mathrm{int}}^{l}=D_{n,\mathrm{full}}^{l}+\mathbf{1}[\rho_{n}^{l}>0]$
- **마스크(사전 할당 후 게이팅, $M\in[0,1]$):** $$
  M_{n,d}^{l}(s_{n}^{l})=
  \begin{cases}
  1, & d\le D_{n,\mathrm{full}}^{l}\\
  \rho_{n}^{l}, & d=D_{n,\mathrm{full}}^{l}+1\\
  0, & \text{otherwise}
  \end{cases}
  $$

> 이 프로젝트에서는 가지를 실제로 append/pop 하지 않고, 항상 $D$ 개를 사전 할당한 뒤 $M\in[0,1]$ 로 게이팅한다(비활성/약활성 가지는 0 또는 작은 값으로 유지).

---

## 2. 제안 DH-SNN 뉴런 동역학

### 2.1 가지별 시냅스 입력 전류

feedforward + recurrent 를 가지별로 분리해 갖는 형태를 쓴다.

$$
\mathbf{I}_{d}^{t+1,l}
=
\mathbf{W}_{d}^{l}\mathbf{o}^{t+1,l-1}
+
\mathbf{U}_{d}^{l}\mathbf{o}^{t,l}
$$

- feedforward-only 설정이면 $\mathbf{U}_d^l=\mathbf{0}$ 로 둔다.

### 2.2 가지별 수상돌기 전류(누설 적분)

$$
 i_{n,d}^{t+1,l}
=
M_{n,d}^{l}\Bigl(
\alpha_{n,d}^{l}\, i_{n,d}^{t,l}
+
(1-\alpha_{n,d}^{l})\, I_{n,d}^{t+1,l}
\Bigr)
$$

- $\alpha_{n,d}^{l}$ 는 가지별 시간상수(다중 시간스케일) 를 결정한다.
- 마스크 $M\in[0,1]$ 로 인해 비활성/약활성 가지는 0 또는 작은 값으로 유지된다.

### 2.3 세포체 집계 + 막전위 업데이트

세포체로 들어갈 때 가지 전류를 **$1/s$ 로 평균화** 한다. (가변 가지 학습을 위한 미분 경로 확보)

$$
H_{n}^{t+1,l}
=
\frac{1}{s_{n}^{l}}\sum_{d=1}^{D} i_{n,d}^{t+1,l}
$$

soft reset 을 포함하면

$$
 u_{n}^{t+1,l}
=
\beta_{n}^{l}u_{n}^{t,l}
+
(1-\beta_{n}^{l})H_{n}^{t+1,l}
-
 o_{n}^{t,l}u_{\mathrm{th}}
$$

스파이크 출력은

$$
 o_{n}^{t+1,l}=H\!\left(u_{n}^{t+1,l}-u_{\mathrm{th}}\right)
$$

이다.

> $D_{n}^{l}=\lfloor s_{n}^{l}\rfloor$ 자체는 비미분이지만, $H$ 의 $1/s$ 항을 통해 task loss가 $s$ 를 직접 업데이트할 수 있다.

---

## 3. 파라미터 제약 및 재매개화(권장)

### 3.1 timing factor

$$
\alpha_{n,d}^{l}=\sigma(\hat{\alpha}_{n,d}^{l}),
\qquad
\beta_{n}^{l}=\sigma(\hat{\beta}_{n}^{l})
$$

- 필요하면 $\alpha$ 에 상한을 두어 $\alpha\to 1$ 근방 민감도를 줄인다( varidble_dendric.md 권장).

### 3.2 구조 파라미터 $s$: $S_{\min},S_{\max}$ 기반

본 프로젝트에서는 $s$ 범위를 **$S_{\min},S_{\max}$** 로만 정의한다.

$$
 s_{n}^{l}
=
S_{\min}+(S_{\max}-S_{\min})\,\sigma(\hat{s}_{n}^{l})
$$

- $s$ 는 **레이어 단위가 아니라 뉴런 단위 파라미터** 이다.
- $S_{\max}$ 는 구현상 $D$ 보다 클 수도 있다.
  - 이때도 $D_{n}^{l}\le D$ 로 clamp 되지만, $s$ 가 커질수록 $1/s$ 로 인해 soma drive 가 감소하므로 여전히 의미가 있다.

- **기본 초기화 정책:** `s_init` 을 별도로 지정하지 않으면 **S_max 근처에서 시작** 한다.
  - 구현은 sigmoid 포화를 피하기 위해 정규화 비율을 $[\epsilon,1-\epsilon]$ 로 clamp 하므로, 실제 초기 $s$ 는
    $S_{\max}-\epsilon(S_{\max}-S_{\min})$ 처럼 S_max보다 약간 작은 값이 된다(자세한 근거는 varidble_dendric.md §2.3).

---

## 4. 희소 연결 제약

DH-SNN 원 논문의 희소화 마스크(입력 라우팅)는 사용하지 않고, **dense 연결** 을 기본으로 한다.

---

## 5. 학습(직접학습, surrogate gradient)

- 시간 전개 후 BPTT 로 학습
- 스텝 함수 $H(\cdot)$ 의 미분 불가능성은 surrogate gradient $\tilde{H}'(\cdot)$ 로 대체

일반적으로 레이어 $l$ 의 가중치 그래디언트는(요지)

$$
\nabla_{\mathbf{W}_{d}^{l}}\mathcal{L}
=
\sum_{t}
\left(\frac{\partial \mathcal{L}}{\partial \mathbf{I}_{d}^{t+1,l}}\right)
\left(\mathbf{o}^{t+1,l-1}\right)^{\top}
$$

로 계산되며, $\tfrac{\partial \mathcal{L}}{\partial \mathbf{I}_{d}^{t+1,l}}$ 는 $\mathbf{i},\mathbf{u}$ 및 surrogate 를 통해 시간 역전파로 얻는다.

---

## 6. 총 손실(제안)

task loss 를 $\mathcal{L}_{\text{task}}$ 라고 하면, 총 손실은

$$
\mathcal{L}_{\text{total}}
=
\mathcal{L}_{\text{task}}
+
\lambda_{\text{ortho}}\,\mathcal{L}_{\text{ortho}}
+
\lambda_{s}\,\mathcal{L}_{s}
$$

처럼 구성한다(필요 항만 사용) .

- $\mathcal{L}_{\text{ortho}}$ : 가지 시간필터($\alpha$) 중복 억제
  - **마스크가 0이 아닌 가지($d\le D_{n,\mathrm{int}}^{l}$)** 에 대해서만 합산한다.
  - 닫힌형은 varidble_dendric.md 참조
- $\mathcal{L}_{s}$ : 가지 복잡도 비용(전체 뉴런 기준 평균)

  $$
  \mathcal{L}_{s}
  =
  \frac{1}{N_{\text{neuron}}}\sum_{\forall \text{neuron}} s
  $$

---

## 7. 학습 안정화: 가중치 초기화(중요)

본 구조는 soma 집계에서 $1/s$ 로 평균을 내기 때문에, **기본 Kaiming 초기화만 그대로 쓰면 초기 soma drive 분산이 과도하게 감소** 할 수 있다.

- 가지 입력이 "서로 독립적인 랜덤 변수"에 가까운 경우(본 문서의 dense $\mathbf{W}_d$ )
  - $\sum_{d=1}^{D} i_d$ 의 표준편차는 $\sqrt{D}$ 에 비례
  - 그런데 $1/s\approx 1/D$ 로 평균을 내면 표준편차가 $1/\sqrt{D}$ 만큼 줄어든다.

따라서 구현에서는 **초기화 직후 $\mathbf{W}$ 를 $\sqrt{D}$ 만큼 스케일 업** 하여, 평균화로 줄어드는 분산을 보정한다.

- 권장(본 프로젝트 구현):
  - Kaiming 초기화 후 $\mathbf{W}\leftarrow \sqrt{D}\,\mathbf{W}$
  - 임계값 $u_{\mathrm{th}}$ 를 변경했다면, 초기 발화율을 맞추기 위해 선형 비율로 보정할 수 있다(예: $u_{\mathrm{th}}$ 기준으로 스케일링).

---

## 8. 구현 체크리스트

- reset: soft reset(위 식)
- $s$: **뉴런 단위** $s_{n}^{l}$ 를 사용
- 활성 가지: $D_{n,\mathrm{int}}^{l}=D_{n,\mathrm{full}}^{l}+\mathbf{1}[\rho_{n}^{l}>0]$ ($\rho$ 는 소프트 마스크 잔여분)
- 구현: 가지는 사전 할당 후 **소프트 마스크로 게이팅** (append/pop 하지 않음)
- 세포체 집계: $1/s_{n}^{l}$ 적용
- 규제항: $\lambda_{\text{ortho}},\lambda_s$ (varidble_dendric.md)
- 학습: surrogate 종류, BPTT 길이, gradient clipping
