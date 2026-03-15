# my_D_RF_neuron — D-RF(다중 수상돌기 Resonate-and-Fire) 제안 구조 및 총 손실

## 0. 범위

- 기본 아이디어는 "Dendritic Resonate-and-Fire (D-RF) " 구조(다중 가지 공진 동역학 + soma 집계 + adaptive threshold) 를 따른다.
- 본 문서는 구현 관점에서 **직접 시뮬레이션 가능한 재귀식 형태** 와, 원 D-RF 논문에서의 **시간축 LTI 컨볼루션/FFT 기반 병렬화 형태** 를 동치로 정리한다. 
- varidble_dendric.md 방식(가변 가지 수, LTI 해석 및 상관 정규화) 을 적용하기 위해 다음을 선택한다.
  - (1) 가지(수상돌기) 공진 동역학을 **재귀식(직접 시뮬레이션)** 과 **컨볼루션(병렬화)** 두 관점으로 모두 정의한다. 
  - (2) soma 로 들어갈 때 가지별 중요도 가중합(예: $C$ 또는 $\gamma$ 역할) 을 제거하고 **단순합/평균** 으로 통일한다. 
  - (3) 유동가지(soft mask) 는 "사전 할당 $D$ 가지 + 마스크 $M\in[0,1]$ 게이팅"으로 유지하며, 컨볼루션/FFT 계산 이후에 $M$ 을 곱해도 동치가 되도록 정식화한다. 
- 병렬화를 최대한 살리기 위해, adaptive threshold 는 기본적으로 "출력 스파이크 $S$ 히스토리" 대신 "pre-indicator $p$ 히스토리" 기반(원 논문 스타일) 을 기본값으로 둔다. (출력 스파이크 히스토리 기반은 옵션으로 부록에 분리) 

---

## 1. 표기

- 시간: $t=0,1,2,\ldots$
- 뉴런(출력 채널) 인덱스: $n\in\{1,\ldots,N\}$
- 가지(수상돌기) 인덱스: $d\in\{1,\ldots,D\}$
  - **$D$ 는 텐서 shape 을 위한 "사전 할당(max) 가지 수"** 이며, 구현에서는 `branch` 로 고정된다. 
- 입력(실수 또는 spike 집계): $I_n[t]\in\mathbb{R}$ 또는 $I_n[t]\in\mathbb{R}^{K}$
- 가지 복소 상태: $z_{n,d}[t]\in\mathbb{C}$, $z_{n,d}[t]=u_{n,d}[t]+i v_{n,d}[t]$
- 시간간격: $\delta>0$
- 가지 감쇠/주파수: $\tau_{n,d}>0,\ \omega_{n,d}>0$
- soma drive(실수부 평균): $H_n[t]\in\mathbb{R}$
- pre-indicator: $p_n[t]\in\{0,1\}$
- 출력 스파이크: $S_n[t]\in\{0,1\}$
- 기본 임계값: $V_{\text{pre}}$
- adaptive threshold 커널: $A=[a_1,\ldots,a_K]$, $a_k>0$

구조(가변 가지) 파라미터:

- **뉴런 단위 연속 파라미터:** $s_n\in[S_{\min},S_{\max}]$ 

  - (구현) $s$ 는 제약 없는 변수 $\hat{s}\in\mathbb{R}$ 를 두고

    $$
    s_n = S_{\min} + (S_{\max}-S_{\min})\,\sigma(\hat{s}_n)
    $$

    로 재매개화하여 학습 중 항상 $[S_{\min},S_{\max}]$ 를 유지한다.

  - **기본 초기화 정책:** `s_init` 을 별도로 지정하지 않으면 **S_max 근처에서 시작** 한다.
    - sigmoid 포화 방지를 위해 정규화 비율을 $[\epsilon,1-\epsilon]$ 로 clamp 하므로, 실제 초기 $s$ 는
      $S_{\max}-\epsilon(S_{\max}-S_{\min})$ 처럼 S_max보다 약간 작은 값이 된다(근거: varidble_dendric.md §2.3).
- **가지 "추가/삭제" 해석을 위한 분해(소프트 마스크):** - clamp: $\tilde{s}_{n}=\min(D,s_{n})$
  - 완전 통과 가지 수(정수): $D_{n,\mathrm{full}}=\lfloor \tilde{s}_{n}\rfloor$
  - 소프트 마스킹 잔여분: $\rho_{n}=\tilde{s}_{n}-D_{n,\mathrm{full}}\in[0,1)$
  - 마스크가 0이 아닌 가지 개수: $D_{n,\mathrm{int}}=D_{n,\mathrm{full}}+\mathbf{1}[\rho_{n}>0]$
- **마스크(사전 할당 후 게이팅, $M\in[0,1]$):** $$
  M_{n,d}(s_{n})=
  \begin{cases}
  1, & d\le D_{n,\mathrm{full}}\\
  \rho_{n}, & d=D_{n,\mathrm{full}}+1\\
  0, & \text{otherwise}
  \end{cases}
  $$

> 이 프로젝트에서는 가지를 실제로 append/pop 하지 않고, 항상 $D$ 개를 사전 할당한 뒤 $M\in[0,1]$ 로 게이팅한다(비활성/약활성 가지는 0 또는 작은 값으로 유지). 

---

## 2. 가지(수상돌기) 공진 동역학

D-RF 의 한 가지는 RF 뉴런과 유사하게 "감쇠 진동" 1차 선형계로 모델링된다.

### 2.1 연속시간 형태(개념)

$$
\frac{d}{dt}z_{n,d}(t)
=
\left(-\frac{1}{\tau_{n,d}}+i\omega_{n,d}\right)z_{n,d}(t)
+
\gamma_{n,d} I_n(t)
$$

- 본 프로젝트는 soma 로 들어갈 때의 가지별 가중합($\gamma$) 을 제거하는 방향이므로, 구현에서는 $\gamma_{n,d}$ 를 입력 스케일에 흡수하거나(혹은 $\gamma_{n,d}=1$ 로 고정) 한다. 

### 2.2 이산시간 재귀(직접 시뮬레이션)

가장 단순한 구현은 exponential Euler(또는 근사 Euler) 로 다음처럼 둔다.

$$
z_{n,d}[t]
=
\rho_{n,d}\, z_{n,d}[t-1]
+
\kappa_{n,d}\, I_n[t]
$$

여기서

$$
\rho_{n,d}=\exp\!\left(\delta\left(-\frac{1}{\tau_{n,d}}+i\omega_{n,d}\right)\right)
$$

이며, $\kappa_{n,d}$ 는

- 근사 Euler: $\kappa_{n,d}=\delta$

로 둔다.

> 구현 팁: 복소 연산이 부담이면, $(u_{n,d},v_{n,d})$ 2차원 실수 상태로 펼쳐서 동일한 선형 재귀로 구현한다. 

### 2.3 시간축 LTI 컨볼루션(병렬화 가능한 동치 표현)

위 재귀는 $z_{n,d}[-1]=0$ 일 때, 과거 입력의 가중합으로 풀어쓸 수 있다.

$$
z_{n,d}[t]
=
\sum_{k=0}^{t} \kappa_{n,d}\,\rho_{n,d}^{\,k}\, I_n[t-k]
$$

따라서 가지 $d$ 의 임펄스 응답(커널) 을

$$
h_{n,d}[k] := \kappa_{n,d}\,\rho_{n,d}^{\,k}\quad (k\ge 0)
$$

로 정의하면, 길이 $T$ 시퀀스에 대해(관측 구간 $t=0,\ldots,T-1$ ):

$$
z_{n,d}[t] = (h_{n,d} * I_n)[t]
\;:=\;
\sum_{k=0}^{t} h_{n,d}[k] I_n[t-k]
$$

- 즉, "재귀 업데이트" 는 "고정 커널로 시간축 causal convolution" 과 동치이다.
- 실전 구현에서는 $t\le T-1$ 에서 필요한 탭은 $k\le T-1$ 뿐이므로, $h_{n,d}[k]$ 를 $k=0,\ldots,T-1$ 로 절단(truncate) 해도 forward 출력은 정확히 동일하다. 

### 2.4 FFT 기반 빠른 컨볼루션(선택)

컨볼루션은 FFT 를 통해 다음처럼 계산할 수 있다.  
길이 $T$ 입력과 길이 $T$ 커널을 선형 컨볼루션으로 계산하려면, 제로패딩 길이 $L\ge 2T-1$ 를 잡는다(보통 $L$ 은 $2$ 의 거듭제곱으로 선택). 

- 제로패딩된 시퀀스를 $\mathrm{pad}_L(\cdot)$ 로 표기하면,

$$
\mathbf{z}_{n,d}
=
\mathrm{IFFT}\left(
\mathrm{FFT}(\mathrm{pad}_L(\mathbf{h}_{n,d}))
\odot
\mathrm{FFT}(\mathrm{pad}_L(\mathbf{I}_{n}))
\right)_{0:T-1}
$$

- 여기서 $\mathbf{I}_n=[I_n[0],\ldots,I_n[T-1]]$ 이고, $\mathbf{h}_{n,d}=[h_{n,d}[0],\ldots,h_{n,d}[T-1]]$ 이다.
- 입력은 실수이지만 커널/출력은 복소이므로, 구현에서는 복소 FFT 를 그대로 쓰는 것이 안전하다. 

> 구현 메모: $\tau,\omega$ 를 고정하면 $\mathrm{FFT}(\mathbf{h}_{n,d})$ 를 미리 캐시할 수 있다. 반대로 $\tau,\omega$ 를 학습한다면 forward 마다 커널을 재생성/FFT 해야 하지만, 시간축 루프 제거 자체는 유지된다. 

---

## 3. soma 집계 및 발화(유동가지 유지, 병렬화 우선)

### 3.1 soma 입력(단순 평균, $1/s$)

가지 상태의 실수부를 단순 평균(또는 합) 하여 soma drive 를 만든다.
varidble_dendric.md 를 적용하여, $D$ 대신 연속 변수 $s_n$ 로 정규화한다.

$$
H_n[t]
=
\frac{1}{s_n}\sum_{d=1}^{D} M_{n,d}(s_n)\,\Re\{z_{n,d}[t]\}
$$

- 컨볼루션/FFT 로 $z_{n,d}[t]$ 를 먼저 계산한 뒤, $M_{n,d}$ 를 곱하고 합쳐도 동치이다(마스크가 시간 불변이므로). 

### 3.2 adaptive threshold(원 D-RF 스타일: pre-indicator 히스토리 기반)

병렬화를 위해, 임계값은 "출력 스파이크 $S$ " 가 아니라 "pre-indicator $p$ " 의 과거에 의해 상승한다고 둔다.

- pre-indicator:
$$
p_n[t]=\Theta\!\left(H_n[t]-V_{\text{pre}}\right)
$$

- adaptive threshold:
$$
V_{\text{th},n}[t]
=
V_{\text{pre}}
+
\sum_{k=1}^{K} a_k\, p_n[t-k]
$$

이는 시간축 causal convolution 으로도 해석된다.

$$
V_{\text{th},n}[t]=V_{\text{pre}}+\left(\mathrm{Conv1d}(p_n,A)\right)[t]
$$

- $p$ 와 $V_{\text{th}}$ 사이에는 순환 의존성이 없으므로( $p$ 는 $H$ 로부터 즉시 계산), 전체 $t=0,\ldots,T-1$ 을 "시퀀스 단위"로 병렬 계산할 수 있다.

### 3.3 스파이크 생성

$$
S_n[t]=\Theta\!\left(H_n[t]-V_{\text{th},n}[t]\right)
$$

- $\Theta(\cdot)$ : Heaviside step
- 학습 시에는 surrogate gradient 를 사용한다. 
- $p$ 와 $S$ 모두 surrogate 를 적용할지 여부는 구현 선택이다. (병렬 forward 자체는 둘 다 step 함수여도 가능) 

### 3.4 (정리) 완전 병렬 forward 파이프라인

길이 $T$ 시퀀스 입력에 대해, 아래 연산들은 모두 "시간 루프 없이" 병렬로 구성할 수 있다.

1) 시냅스 입력 만들기(레이어 입력 결합):  
   $$I^{l}[t] = \mathbf{W}^{l}\mathbf{S}^{l-1}[t]$$
2) 가지 공진 상태: 재귀 대신 컨볼루션(Conv1d/FFT) 로 $z_{n,d}[0:T-1]$ 를 한 번에 계산  
3) soma 집계: 마스크 $M(s)$ 를 곱해 가지를 게이팅하고, $1/s$ 로 정규화해 $H[0:T-1]$ 계산  
4) pre-indicator $p[0:T-1]$ 계산  
5) threshold: $V_{\text{th}} = V_{\text{pre}} + \mathrm{Conv1d}(p,A)$  
6) 스파이크 $S[0:T-1]$ 계산  

---

## 4. 네트워크로 확장(레이어 형태)

레이어 $l$ 의 입력을 이전 레이어 spike 로부터 선형 결합한다고 두면

$$
I^{l}[t] = \mathbf{W}^{l}\mathbf{S}^{l-1}[t]
$$

이고, 각 뉴런 $n$ 은 자신의 가지 상태 $z_{n,d}[t]$ 를 위 컨볼루션/재귀 정의 중 하나로 계산한다.

> soma-가중합($C$) 을 제거했으므로, "가지의 역할" 은 주로 $(\tau_{n,d},\omega_{n,d})$ 의 다양성으로 결정된다. (이 다양성 정규화는 varidble_dendric.md 의 상관 정규화를 그대로 확장 적용 가능) 

---

## 5. 학습(직접학습, surrogate gradient)

- $\Theta(\cdot)$ 의 미분 불가능성은 surrogate $\tilde{\Theta}'(\cdot)$ 로 대체
- D-RF 논문에서는 double-Gaussian 형태 surrogate 를 사용하지만, 구현은 다른 surrogate 도 가능

학습 가능한 파라미터 예시:

- 시냅스 가중치: $\mathbf{W}^{l}$
- 가지 파라미터: $\tau_{n,d},\omega_{n,d}$ (선택: 고정/학습)
- adaptive threshold 커널: $a_1,\ldots,a_K$ (선택)
- (가변 가지) $s_n$ 및 가지 마스킹 규칙(varidble_dendric.md)

Optimizer regularization(AdamW weight decay) 적용 범위(프로젝트 구현):

- weight decay 적용: 레이어 연결(시냅스) 가중치 $\mathbf{W}^{l}$
- weight decay 비적용: $\tau/\omega$, $a_1,\ldots,a_K$, 구조 파라미터 $s$

> 병렬형 컨볼루션으로 forward 를 구성해도, 역전파는 자동미분으로 동일하게 수행된다(컨볼루션/FFT 연산을 통과하는 gradient). 

---

## 6. 총 손실(제안)

task loss $\mathcal{L}_{\text{task}}$ 에 더해 다음을 조합한다.

$$
\mathcal{L}_{\text{total}}
=
\mathcal{L}_{\text{task}}
+
\lambda_s\,\mathcal{L}_{s}
+
\lambda_{\text{ortho}}\mathcal{L}_{\text{ortho}}
$$

- $\mathcal{L}_{s}$ : 가지 복잡도 비용(전체 뉴런 기준 평균)

$$
\mathcal{L}_{s}
=
\frac{1}{N_{\text{neuron}}}\sum_{\forall \text{neuron}} s
$$

- $\mathcal{L}_{\text{ortho}}$ : 가지 다양성(시간필터/응답) 중복 억제
  - varidble_dendric.md 의 "임펄스 응답 커널 내적 → 코사인 유사도 → 코사인 제곱 정규화"를 D-RF 가지(복소 공진 IIR)로 확장한다.

---

## 7. D-RF 가지 커널 직교 정규화(닫힌형 요약)

### 7.1 D-RF(Euler) 가지 임펄스 응답 커널

가지 재귀(구현형):

$$
z_d[t]
=
\rho_d\, z_d[t-1]
+
\delta\, I[t],
\qquad
\rho_d=\exp\!\left(\delta\left(-\frac{1}{\tau_d}+i\omega_d\right)\right)
$$

- $\tau_d>0$ 이면 $|\rho_d|=\exp(-\delta/\tau_d)<1$ 이므로 아래 무한합(내적/노름)이 수렴한다.

크로네커 델타 입력 $I[t]=\mathbf{1}_{t=0}$ 및 $z_d[-1]=0$ 에 대한 가지의 **임펄스 응답(커널)** 을

$$
h_d[n] := z_d[n]\quad (I[t]=\mathbf{1}_{t=0})
$$

로 정의하면,

$$
h_d[n] = \delta\,\rho_d^{\,n}\quad (n\ge 0)
$$

---

### 7.2 커널 내적(무한 지평)과 코사인 제곱 항의 닫힌형

복소 커널이므로, 표준 복소 $\ell_2$ 내적을 사용한다(켤레 포함).

- 내적
$$
\langle h_i,h_j\rangle
:=
\sum_{n=0}^{\infty} h_i[n]\,\overline{h_j[n]}
$$

- 노름
$$
\|h_i\|^2 := \langle h_i,h_i\rangle
$$

여기에 $h_d[n]=\delta\rho_d^n$ 를 대입하면, 기하급수 합으로 즉시 닫힌형이 된다.

$$
\langle h_i,h_j\rangle
=
\delta^2\sum_{n=0}^{\infty}(\rho_i\overline{\rho_j})^n
=
\frac{\delta^2}{1-\rho_i\overline{\rho_j}}
$$

$$
\|h_i\|^2
=
\delta^2\sum_{n=0}^{\infty}|\rho_i|^{2n}
=
\frac{\delta^2}{1-|\rho_i|^2}
$$

이제 varidble_dendric.md와 동일한 "코사인 유사도 → 제곱"을 정의한다. (복소이므로 분자에 절댓값을 사용) 

$$
\cos(h_i,h_j)
:=
\frac{|\langle h_i,h_j\rangle|}{\|h_i\|\,\|h_j\|}
$$

따라서

$$
\cos(h_i,h_j)^2
=
\frac{|\langle h_i,h_j\rangle|^2}{\|h_i\|^2\,\|h_j\|^2}
=
\frac{(1-|\rho_i|^2)(1-|\rho_j|^2)}{|1-\rho_i\overline{\rho_j}|^2}
$$

- Euler 이산화에서는 $\delta$ 가 완전히 소거되므로, 정규화는 **가지 파라미터 $\tau,\omega$ 의 상대적 차이** 만 반영한다. 

---

### 7.3 $(\tau,\omega)$ 로 전개한 닫힌형

$\rho_d$ 를 크기/위상으로 쓰면

$$
\rho_d = r_d e^{i\theta_d},
\qquad
r_d=|\rho_d|=\exp(-\delta/\tau_d),
\qquad
\theta_d=\omega_d\delta
$$

또한

$$
\rho_i\overline{\rho_j}
=
r_ir_j\exp\!\left(i\delta(\omega_i-\omega_j)\right)
$$

이므로

$$
|1-\rho_i\overline{\rho_j}|^2
=
1+r_i^2r_j^2-2r_ir_j\cos\!\left(\delta(\omega_i-\omega_j)\right)
$$

따라서 코사인 제곱 항은 최종적으로

$$
\cos(h_i,h_j)^2
=
\frac{(1-r_i^2)(1-r_j^2)}
{1+r_i^2r_j^2-2r_ir_j\cos\!\left(\delta(\omega_i-\omega_j)\right)}
\quad,\;
r_d=\exp(-\delta/\tau_d)
$$

---

### 7.4 직교(비중복) 유도 정규화 손실

한 뉴런 $n$ 에서 마스크가 0이 아닌 가지가 $d=1,\ldots,D_{n,\mathrm{int}}$ 일 때

$$
\mathcal{L}_{\text{ortho},n}
=
\sum_{1\le i<j\le D_{n,\mathrm{int}}}\cos(h_i,h_j)^2
$$

로 두어, 서로 유사한 가지 커널(= 유사한 시간상수/주파수 조합)이 중복되는 것을 억제한다.

---

### 7.5 수치 안정화(권장)

- 분모 안정화:
$$
|1-\rho_i\overline{\rho_j}|^2 \leftarrow |1-\rho_i\overline{\rho_j}|^2 + \varepsilon
$$
- 파라미터 제약: $\tau_d>0$ (softplus 등), 필요 시 $\tau$ 상한/하한을 두어 $r\to 1$ 근방 민감도를 완화한다.

---

## 8. 구현 체크리스트(병렬화 포함)

- 이산화 선택: $\kappa_d$ (Euler 적분) 및 $\rho_d$ 계산
- 파라미터 범위: $\tau_d>0,\omega_d>0$ (softplus 파라미터화 권장)
- 길이 $T$ 시퀀스에 대해 커널 탭 생성: $h_{n,d}[k]=\kappa_{n,d}\rho_{n,d}^{\,k}$, $k=0,\ldots,T-1$
- 컨볼루션 구현 선택:
  - (1) `Conv1d` (직접 conv) 
  - (2) FFT 기반(제로패딩 길이 $L\ge 2T-1$ , crop 으로 causal 부분 $0:T-1$ 유지) 
- 유동가지: $M_{n,d}(s_n)$ 계산 후, 컨볼루션 출력의 실수부에 곱해 게이팅
- soma 집계: $H_n[t]=\frac{1}{s_n}\sum_d M_{n,d}\Re\{z_{n,d}[t]\}$ 
- threshold: pre-indicator $p$ 기반 $V_{\text{th}}=V_{\text{pre}}+\mathrm{Conv1d}(p,A)$ 
- adaptive threshold 길이 $K$ 및 $a_k$ 초기화(양수 제약)
- $s$: 뉴런 단위, $1/s$ 집계 적용
- surrogate 선택, gradient clipping, 긴 시퀀스에서의 안정화

---

## 부록 A. (옵션) 출력 스파이크 히스토리 기반 threshold(순차 의존성 발생)

출력 스파이크 히스토리로 임계값을 정의하면,

$$
V_{\text{th},n}[t]
=
V_{\text{pre}}
+
\sum_{k=1}^{K} a_k\, S_n[t-k]
$$

이 역시 수식적으로는 컨볼루션이지만, $S$ 자체가 $V_{\text{th}}$ 에 의존하므로( $S\leftrightarrow V_{\text{th}}$ ) 시간축 순환 의존성이 생긴다.

$$
S_n[t]=\Theta\!\left(H_n[t]-V_{\text{th},n}[t]\right)
$$

- 이 경우 $t=0\to T-1$ 순서의 time-loop 가 필요해져, 원 D-RF 논문에서 주장하는 "완전 병렬" 형태는 깨진다.
- 유동가지 자체와는 독립이므로(마스크는 시간 불변), 유동가지를 유지하면서도 병렬화를 최대로 살리고 싶다면 본문(3.2) 의 $p$ 기반 정의를 우선 권장한다.
