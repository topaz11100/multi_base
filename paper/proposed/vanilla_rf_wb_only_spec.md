# Vanilla Resonate-and-Fire (RF) 뉴런 구현/이론 명세서

## 0. 문서 목적

이 문서는 **학습 가능한 파라미터를 각 뉴런의 고유 동역학 파라미터 \$\omega, b\$로 제한한 vanilla Resonate-and-Fire (RF) 뉴런** 의 구현 및 분석 명세를 정의한다. 목적은 다음 두 가지다.

1. **구현 목적** : PyTorch만으로 안정적으로 학습 가능한 RF 뉴런을 정의한다.
2. **연구 목적** : 학습이 끝난 뒤 각 뉴런이 수렴한 고유 주파수 및 대역폭 통계를 분석하여, RF 집단이 **주파수 선택적 필터 뱅크** 로 자기조직화되는지 해석한다.

이 명세는 **"주파수 도메인 해석 가능성"** 을 최우선 목표로 두며, 따라서 학습 성능 자체보다 **해석 가능하고 비교 가능한 설정** 을 우선한다.

---

## 1. 연구 가설과 설계 원칙

### 1.1 핵심 가설

RF 뉴런의 선형 서브스레시홀드(subthreshold) 동역학은 본질적으로 **감쇠 진동계** 이며, 입력 전류에 대해 **주파수 선택적 응답** 을 보인다. 따라서 RF 뉴런 집단만으로 구성된 모델을 학습시키면, 각 뉴런은 과제 및 데이터의 시간 구조에 맞춰 **서로 다른 중심 주파수와 대역폭** 을 갖는 방향으로 수렴할 가능성이 있다.

### 1.2 이 명세의 기본 철학

이 해석을 성립시키려면 뉴런마다 학습되는 자유도를 최소화해야 한다. 따라서 다음을 기본 원칙으로 둔다.

- 뉴런 내부 trainable parameter는 **오직 \$\omega_i, b_i\$** 뿐이다.
- threshold, reset, gain, refractory 등은 **고정값** 으로 둔다.
- 입력 투영(input projection)은 **고정** 하거나, 최소한 RF의 고유 동역학과 분리하여 해석한다.
- 학습 후 분석은 반드시 **비선형 spike 생성과 분리된 선형화된 서브스레시홀드 응답** 을 기준으로 수행한다.

즉, 이 연구에서 해석 대상은 "스파이크 발생 메커니즘 전체"가 아니라, **뉴런 내부의 공명 동역학 자체** 다.

---

## 2. Canonical vanilla RF 정의

### 2.1 연속시간 동역학

뉴런 \$i\$의 상태를 실수 상태 \$x_i(t), y_i(t)\$로 두고, 입력 전류를 \$I_i(t)\$라 하면, 기본 RF 뉴런은 다음으로 정의한다.

\$$
\dot{x}_i = b_i x_i - \omega_i y_i + I_i(t)
\$$
\$$
\dot{y}_i = \omega_i x_i + b_i y_i
\$$

여기서

- \$\omega_i > 0\$: 고유 각주파수(intrinsic angular frequency)
- \$b_i < 0\$: 감쇠 계수(damping)

이다. 복소 상태 \$z_i = x_i + j y_i\$를 쓰면

\$$
\dot{z}_i = (b_i + j\omega_i) z_i + I_i(t)
\$$

로 쓸 수 있다. 따라서 이 뉴런의 연속시간 고유치는

\$$
\lambda_i = b_i \pm j\omega_i
\$$

이며, \$b_i < 0\$이면 서브스레시홀드 상태는 안정하다.

### 2.2 스파이크 정의

본 명세의 **기본 스파이크 규칙** 은 다음과 같다.

\$$
s_i(t) = H(x_i(t) - \theta)
\$$

- \$H(\cdot)\$: Heaviside step
- \$\theta\$: 고정 임계값, 기본값 1.0

### 2.3 reset 정책

문헌마다 RF의 reset 정의가 다르므로, 본 연구에서는 **해석 우선 설정** 을 위해 아래를 표준으로 채택한다.

#### 표준 설정: `no-reset`

스파이크는 readout에만 사용하고 내부 상태는 reset하지 않는다.

\$$
(x_i, y_i) \leftarrow (x_i, y_i)
\$$

이 설정의 장점은 다음과 같다.

- 서브스레시홀드 동역학이 가장 직접적으로 유지된다.
- 학습 후 주파수 응답을 선형계 해석으로 복원하기 쉽다.
- bandwidth 해석이 spike 후 비선형 state jump에 덜 오염된다.

#### 보조 실험(ablation): `soft-reset`

선택적으로 아래를 추가할 수 있다.

\$$
x_i \leftarrow x_i - \theta s_i
\$$

이 경우도 \$y_i\$는 유지한다. `hard-reset`은 동역학 해석을 크게 오염시키므로 본 motivation study의 기본 실험에서는 권장하지 않는다.

---

## 3. 연구용 아키텍처 명세

### 3.1 가장 해석 가능한 기본 구조: RF filter bank

입력 시계열 \$u_t \in \mathbb{R}^{d_{in}}\$에 대해, RF 뱅크 입력은 **고정 투영** 으로 만든다.

\$$
I_t = P u_t, \qquad P \in \mathbb{R}^{N \times d_{in}}
\$$

- \$P\$는 **학습하지 않는다** .
- 각 행은 unit norm으로 정규화한다.
- 가능하면 orthogonal/random Gaussian projection을 사용한다.

이때 뉴런 \$i\$는 오직 \$I_{i,t}\$와 \$\omega_i, b_i\$만으로 진화한다.

### 3.2 출력 헤드

출력 헤드는 연구 목적에 따라 둘 중 하나를 선택한다.

#### A. Intrinsic analysis 우선

- RF bank의 \$\omega, b\$만 학습
- 출력은 fixed decoder 또는 간단한 pooled statistic 사용

이 설정은 가장 깔끔하지만 과제 성능이 낮을 수 있다.

#### B. 실용적 권장 설정

- RF bank 내부에서는 \$\omega, b\$만 학습
- 최종 linear readout \$W_{out}\$는 학습 가능

이 경우에도 **RF 뉴런 내부의 해석 대상은 여전히 \$\omega, b\$** 이므로, 로그와 분석에서는 readout parameter를 분리해야 한다.

### 3.3 금지 또는 비권장 요소

본 motivation study의 1차 실험에서는 아래를 넣지 않는 편이 좋다.

- trainable threshold
- trainable input gain per neuron
- trainable recurrent coupling among RF neurons
- adaptive refractory
- learnable reset strength

이 요소들은 모두 "뉴런의 대역 통계"를 다른 자유도와 혼합시킨다.

---

## 4. 이산시간 구현 명세

### 4.1 왜 Euler가 아니라 exact discretization인가

단순 Euler 이산화는

\$$
z_{t+1} = \big(1 + (b_i + j\omega_i)\Delta\big) z_t + \Delta I_{i,t}
\$$

가 되며, 안정 조건은

\$$
\left|1 + (b_i + j\omega_i)\Delta\right| < 1
\$$

즉,

\$$
(1 + b_i\Delta)^2 + (\omega_i\Delta)^2 < 1
\$$

이다. 이 조건은 \$\omega\$, \$b\$, \$\Delta\$ 조합에 강하게 의존하므로, 학습 중에 파라미터가 조금만 움직여도 수치적 발산이나 의미 왜곡이 생길 수 있다.

따라서 본 명세의 **규범적 구현(normative implementation)** 은 **exact zero-order hold (ZOH) discretization** 이다.

### 4.2 exact ZOH 이산화

한 스텝 동안 입력 \$I_{i,t}\$가 상수라고 두면,

\$$
z_{i,t+1} = \alpha_i z_{i,t} + \beta_i I_{i,t}
\$$

이며,

\$$
\alpha_i = e^{(b_i + j\omega_i)\Delta} = \rho_i e^{j\phi_i}
\$$

\$$
\beta_i = \frac{e^{(b_i + j\omega_i)\Delta} - 1}{b_i + j\omega_i}
\$$

이다.

여기서

\$$
\rho_i = e^{b_i\Delta}, \qquad \phi_i = \omega_i\Delta
\$$

이므로, 실수 상태로 풀면

\$$
\begin{bmatrix}
x_{i,t+1} \\
y_{i,t+1}
\end{bmatrix}
=
\rho_i
\begin{bmatrix}
\cos\phi_i & -\sin\phi_i \\
\sin\phi_i & \cos\phi_i
\end{bmatrix}
\begin{bmatrix}
x_{i,t} \\
y_{i,t}
\end{bmatrix}
+
\begin{bmatrix}
\beta_{r,i} \\
\beta_{\ell,i}
\end{bmatrix} I_{i,t}
\$$

로 쓸 수 있다. 여기서 실수부/허수부는

\$$
\beta_{r,i} = \frac{b_i(\rho_i\cos\phi_i - 1) + \omega_i\rho_i\sin\phi_i}{b_i^2 + \omega_i^2}
\$$

\$$
\beta_{\ell,i} = \frac{b_i\rho_i\sin\phi_i - \omega_i(\rho_i\cos\phi_i - 1)}{b_i^2 + \omega_i^2}
\$$

이다. (`\ell`은 imaginary-like second channel 표기용이다. 구현에서는 `beta_i` 또는 `beta_y` 같은 이름을 써도 된다.)

### 4.3 이산시간 안정 조건

exact discretization에서는 안정성 판단이 단순하다.

\$$
|\alpha_i| = \rho_i = e^{b_i\Delta} < 1 \iff b_i < 0
\$$

따라서 연속시간 의미와 이산시간 의미가 일치한다. 즉,

- \$b_i < 0\$이면 안정
- \$\omega_i\$는 pole angle만 정함

이므로 해석이 훨씬 깔끔하다.

---

## 5. 파라미터화 명세

사용자는 분석 결과를 \$\omega_i, b_i\$로 보고 싶지만, 구현에서는 안정성을 위해 내부 raw parameter를 쓰는 것이 바람직하다.

### 5.1 의미론적(trainable concept) 파라미터

- 의미론적으로 학습되는 파라미터: \$\omega_i, b_i\$
- 분석 및 저장: 반드시 \$\omega_i, b_i\$로 기록

### 5.2 내부 구현용 제약 파라미터화

권장 방식은 다음 둘 중 하나다.

#### 방식 A. 직접 제약 파라미터화

\$$
\omega_i = \omega_{min} + \text{softplus}(u^{(\omega)}_i)
\$$

\$$
b_i = -\text{softplus}(u^{(b)}_i) - \varepsilon
\$$

장점: 구현이 단순하다.

#### 방식 B. 이산 파라미터 기반 안정 파라미터화 **(권장)** 먼저 이산시간 변수

\$$
\rho_i \in (\rho_{min}, \rho_{max}) \subset (0,1)
\$$
\$$
\phi_i \in (\phi_{min}, \phi_{max}) \subset (0,\pi)
\$$

를 내부 raw parameter로 학습하고,

\$$
b_i = \frac{\log \rho_i}{\Delta}, \qquad \omega_i = \frac{\phi_i}{\Delta}
\$$

로 복원한다.

이 방식의 장점은 다음과 같다.

- \$b_i < 0\$가 자동 보장된다.
- \$\omega_i < \pi / \Delta\$가 자동 보장되어 aliasing 위험을 줄인다.
- exact discretization과 가장 잘 맞는다.

### 5.3 권장 범위

- \$\phi_{min} > 0\$, \$\phi_{max} < \pi\$
- \$\rho_{min}\$은 너무 작지 않게, \$\rho_{max}\$는 1보다 약간 작게

실험 시작점 예시:

- \$\rho \in [0.95, 0.999]\$
- \$\phi\$는 관심 주파수 대역에 대해 log-uniform 초기화

---

## 6. forward 규약

뉴런 \$i\$의 한 step은 다음과 같이 정의한다.

1. 외부에서 입력 전류 \$I_{i,t}\$를 받는다.
2. exact ZOH로 \$x_{i,t+1}, y_{i,t+1}\$를 업데이트한다.
3. 스파이크를 생성한다.
4. reset 정책이 있으면 적용한다.

### 6.1 기준 수식

\$$
s_{i,t+1} = H(x_{i,t+1} - \theta)
\$$

기본 설정인 `no-reset`에서는 여기서 끝난다.

보조 실험 `soft-reset`에서는

\$$
x_{i,t+1} \leftarrow x_{i,t+1} - \theta s_{i,t+1}
\$$

를 추가한다.

### 6.2 surrogate gradient

학습 시에는

\$$
H(x) \approx \tilde{H}(x)
\$$

로 surrogate gradient를 사용한다. 단, 본 연구의 해석 대상은 surrogate 자체가 아니라 RF 동역학이므로, surrogate는 **가급적 단순한 fast-sigmoid류** 를 쓴다. threshold 주변에서만 gradient를 주고, 넓은 범위에서 과도한 gradient를 주지 않는 형태가 적합하다.

예:

\$$
\frac{\partial \tilde{H}}{\partial x} = \frac{1}{(1 + \alpha |x|)^2}
\$$

---

## 7. PyTorch 모듈 인터페이스 명세

### 7.1 RF cell 수준

```python
class VanillaRFCell(nn.Module):
    def __init__(
        self,
        n_neurons: int,
        dt: float,
        threshold: float = 1.0,
        reset_mode: str = "none",   # "none" or "soft"
        rho_range=(0.95, 0.999),
        phi_range=(phi_min, phi_max),
    ):
        ...

    def init_state(self, batch_size: int, device=None, dtype=None):
        # returns x, y, s_prev (optional)
        ...

    def forward(self, I_t, state):
        # I_t: [B, N]
        # state: (x, y)
        # returns spike, new_state, aux_dict
        ...
```

### 7.2 layer/bank 수준

```python
class FixedProjectionRFBank(nn.Module):
    def __init__(self, d_in, n_neurons, dt, learn_readout=True):
        # P: fixed random projection
        # rf: VanillaRFCell
        # head: optional linear readout
        ...
```

### 7.3 trainable parameter 규약

`model.named_parameters()` 기준으로 RF 내부에서 optimizer에 들어가는 것은 다음뿐이어야 한다.

- `raw_phi` 또는 `raw_omega`
- `raw_rho` 또는 `raw_b`

threshold, projection matrix, reset coefficient 등은 optimizer parameter list에 포함되지 않아야 한다.

---

## 8. 구현용 의사코드

```python
# state: x, y
# parameters per neuron: omega_i, b_i
# exact discrete params:
# rho_i = exp(b_i * dt)
# phi_i = omega_i * dt

for t in range(T):
    I_t = fixed_projection(u_t)              # [B, N], not trainable

    c = cos(phi)
    s = sin(phi)
    den = b**2 + omega** 2 + eps

    beta_x = (b * (rho * c - 1.0) + omega * rho * s) / den
    beta_y = (b * rho * s - omega * (rho * c - 1.0)) / den

    x_new = rho * (c * x - s * y) + beta_x * I_t
    y_new = rho * (s * x + c * y) + beta_y * I_t

    spike = surrogate_heaviside(x_new - theta)

    if reset_mode == "soft":
        x_new = x_new - theta * spike

    x, y = x_new, y_new
```

---

## 9. 초기화 명세

### 9.1 \$\omega\$ 초기화

관심 주파수 범위가 넓으면 선형 간격보다 **log-uniform 초기화** 가 낫다.

예를 들어 입력의 유효 대역이 \$[f_{min}, f_{max}]\$이면,

\$$
\omega_i \sim \text{LogUniform}(2\pi f_{min}, 2\pi f_{max})
\$$

로 둔다.

### 9.2 \$b\$ 초기화

너무 음수면 감쇠가 심해져 공명성이 죽고, 0에 너무 가까우면 지나치게 long-memory가 된다. 따라서 초기에는 중간 정도의 감쇠를 주고, 학습이 조정하게 한다.

실용적으로는 \$\rho\$를 먼저 초기화하는 편이 안정적이다.

### 9.3 threshold

- 고정값 1.0 권장
- 모든 뉴런 동일
- 학습 금지

threshold까지 학습시키면 "대역 선택성"과 "발화 민감도"가 혼합된다.

---

## 10. 학습 설정 명세

### 10.1 손실 함수

연구 목적에 따라 아래를 쓴다.

#### 지도학습

\$$
\mathcal{L} = \mathcal{L}_{task} + \lambda_{reg}\mathcal{L}_{reg}
\$$

#### 자기지도/분석 중심 실험

- reconstruction loss
- contrastive temporal objective
- spectral alignment objective

등을 사용할 수 있다. 중요한 점은 **RF 내부 파라미터 해석 가능성** 이 유지되도록 자유도를 제한하는 것이다.

### 10.2 optimizer

권장:

- Adam 또는 AdamW
- \$\omega, b\$ 계열에는 작은 learning rate 사용
- gradient clipping 사용

실용적 예시:

- RF parameter group: `lr = 1e-3` 전후
- readout이 있다면 readout group이 RF보다 약간 큰 lr 가능
- global grad clip: 1.0

### 10.3 regularization

#### 필수는 아님

기본 baseline에서는 regularizer 없이 시작하는 것이 좋다. 그래야 정말로 데이터/과제 때문에 대역이 분화되는지 확인할 수 있다.

#### 선택적 regularizer

1. **주파수 분산 유도(repulsion on log-frequency)** 2. **과도한 저감쇠 방지** 3. **너무 넓은/너무 좁은 bandwidth 방지** 예시:

\$$
\mathcal{L}_{repel} = \frac{1}{N(N-1)}\sum_{i \neq j} \exp\left(-\frac{|\log \omega_i - \log \omega_j|}{\tau}\right)
\$$

단, 이 regularizer는 뉴런 대역 분포를 **유도** 하므로, baseline과 분리된 보조 실험으로만 써야 한다.

---

## 11. 분석 명세: 대역 통계를 어떻게 정의할 것인가

스파이크 기반 뉴런은 비선형이므로, "뉴런의 대역폭"을 정의할 때 반드시 **분석 수준(level of analysis)** 을 나눠야 한다.

### 11.1 Level A: intrinsic subthreshold band

학습된 \$\omega_i, b_i\$를 고정한 뒤, spike/reset를 끄고 **선형 서브스레시홀드 시스템** 의 주파수 응답을 계산한다. 이것이 본 연구의 주 해석 대상이다.

### 11.2 Level B: membrane response band

실제 task input을 넣었을 때 \$x_i(t)\$의 power spectrum을 측정한다. 이는 intrinsic band와 데이터 스펙트럼의 상호작용 결과다.

### 11.3 Level C: spike response band

스파이크열 \$s_i(t)\$의 spectrum 또는 rate modulation spectrum을 측정한다. 비선형 threshold 때문에 harmonic이 생길 수 있으므로, 이것을 intrinsic band와 혼동하면 안 된다.

### 11.4 권장 해석 순서

1. **A를 먼저 본다.**
2. 다음으로 A와 B의 정렬 정도를 본다.
3. 마지막으로 C가 과제와 어떤 관계를 갖는지 본다.

---

## 12. 선형 주파수 응답 계산 명세

### 12.1 이산시간 상태공간 표현

뉴런 \$i\$에 대해

\$$
\mathbf{h}_{i,t} =
\begin{bmatrix}
x_{i,t} \\
y_{i,t}
\end{bmatrix}
\$$

라고 두면,

\$$
\mathbf{h}_{i,t+1} = A_i \mathbf{h}_{i,t} + B_i I_{i,t}
\$$

이며,

\$$
A_i = \rho_i
\begin{bmatrix}
\cos\phi_i & -\sin\phi_i \\
\sin\phi_i & \cos\phi_i
\end{bmatrix}
\$$

\$$
B_i =
\begin{bmatrix}
\beta_{r,i} \\
\beta_{\ell,i}
\end{bmatrix},
\qquad
C = \begin{bmatrix}1 & 0\end{bmatrix},
\qquad D = 0
\$$

이다.

### 12.2 전달함수

주파수 \$\Omega\$에서의 이산 주파수 응답은

\$$
H_i(e^{j\Omega\Delta}) = C \left(e^{j\Omega\Delta}I - A_i\right)^{-1} B_i
\$$

로 계산한다.

구현상으로는 주파수 grid \$\{\Omega_k\}\$를 만들고, 각 점에서 복소 응답의 크기를 측정하면 된다.

\$$
M_i(\Omega_k) = \left| H_i(e^{j\Omega_k\Delta}) \right|
\$$

### 12.3 중심 주파수와 대역폭 정의

닫힌형 근사식을 쓰지 말고, **수치적 정의** 를 쓴다.

- **중심 주파수** \\
  \$$\Omega^*_i = \arg\max_{\Omega_k} M_i(\Omega_k)\$$

- **3 dB bandwidth** \\
  \$$M_i(\Omega_{lo}) = M_i(\Omega_{hi}) = M_i(\Omega^*_i) / \sqrt{2}\$$
  \\
  \$$BW_i = \Omega_{hi} - \Omega_{lo}\$$

- **Q-factor** \\
  \$$Q_i = \Omega^*_i / BW_i\$$

- **peak gain** \\
  \$$G_i = M_i(\Omega^*_i)\$$

이 정의는 reset, surrogate, output nonlinearity와 무관하게 **뉴런 내부 동역학만으로 정의된 passband descriptor** 를 준다.

---

## 13. 분석할 통계량

뉴런 집단 \$i = 1, \dots, N\$에 대해 다음을 저장한다.

### 13.1 1차 통계

- \$\omega_i\$ 히스토그램
- \$b_i\$ 히스토그램
- \$\Omega^*_i\$ 히스토그램
- \$BW_i\$ 히스토그램
- \$Q_i\$ 히스토그램

### 13.2 집단 커버리지

주파수 축에서 뉴런 뱅크가 어느 정도를 덮는지 본다.

- passband union length
- 평균 overlap count
- redundancy index

예를 들어 thresholded passband set을

\$$
\mathcal{B}_i = \{\Omega : M_i(\Omega) \ge \eta M_i(\Omega^*_i)\}
\$$

로 정의하고, \$\cup_i \mathcal{B}_i\$의 길이와 겹침 빈도를 측정할 수 있다.

### 13.3 분포 엔트로피

- center frequency entropy
- bandwidth entropy
- joint entropy of \$(\Omega^*, BW)\$

### 13.4 학습 과정 추적

학습 epoch마다

- \$\omega_i\$, \$b_i\$
- \$\Omega^*_i\$, \$BW_i\$
- activation rate / spike rate

를 저장해 **대역이 언제 갈라지기 시작하는지** 본다.

### 13.5 입력/과제와의 정렬

- 입력 데이터의 PSD와 뉴런 \$\Omega^*_i\$ 분포 비교
- 클래스별 또는 조건별 스펙트럼과 뉴런 분포 비교
- 특정 주파수 대역의 입력 에너지가 높을수록 해당 대역 뉴런이 더 활성화되는지 측정

이 분석이 motivation study의 핵심이다.

---

## 14. 권장 실험 설계

### 14.1 최소 baseline

- fixed random projection RF bank
- learnable \$\omega, b\$
- no-reset
- exact discretization
- optional linear readout

### 14.2 필수 ablation

1. `no-reset` vs `soft-reset`
2. exact discretization vs Euler
3. \$\omega\$-only 학습 vs \$b\$-only 학습 vs 둘 다 학습
4. fixed projection vs trainable projection

### 14.3 해석용 synthetic probe

학습 후 각 뉴런에 대해 아래 synthetic input도 넣는다.

#### sinusoidal sweep

\$$
I_t = A \sin(\Omega t \Delta)
\$$

여러 \$\Omega\$에 대해 steady-state amplitude와 spike rate를 측정한다.

#### chirp

시간에 따라 주파수가 증가하는 입력을 넣어, 뉴런이 어느 구간에서 강하게 반응하는지 본다.

#### white / pink noise

실제 데이터 외에 표준 입력에 대한 응답을 측정해 intrinsic tuning을 비교한다.

---

## 15. 재현성 체크리스트

- random seed 고정
- fixed projection matrix 저장
- \$\Delta\$ 고정
- frequency grid 해상도 고정
- reset policy 명시
- threshold 고정값 명시
- parameterization 방식 명시 (`direct` vs `rho-phi`)
- 학습 후 \$\omega, b\$ 원값 저장
- 분석 시 spike on/off 여부 명시

---

## 16. 구현 검증 기준

구현이 맞는지 확인하려면 최소한 아래를 통과해야 한다.

### 16.1 안정성 테스트

- zero input에서 장시간 시뮬레이션 시 state가 발산하지 않는다.
- \$b < 0\$이면 \$\rho < 1\$가 항상 성립한다.

### 16.2 응답 테스트

- 단일 사인 입력 sweep 시 특정 주파수 근방에서 membrane amplitude가 최대가 된다.
- 학습된 \$\Omega^*_i\$와 실측 sweep 최대응답 주파수가 가까워야 한다.

### 16.3 gradient 테스트

- `raw_phi`, `raw_rho` 혹은 `raw_omega`, `raw_b`에 gradient가 흐른다.
- threshold나 projection matrix에 gradient가 흐르지 않는다.

### 16.4 serialization 테스트

- 저장/로드 후 \$\omega, b\$ 값이 보존된다.
- 주파수 응답 곡선이 동일하게 재현된다.

---

## 17. 권장 보고 지표

논문/노트에서 최소한 아래 그림은 넣는 것이 좋다.

1. 학습 전후 \$\omega\$ 분포
2. 학습 전후 \$b\$ 분포
3. 학습 전후 \$\Omega^*\$-\$BW\$ scatter
4. 전체 뉴런의 frequency response heatmap
5. epoch에 따른 passband migration plot
6. input PSD vs neuron center-frequency histogram overlay
7. 특정 샘플/클래스에서 활성 뉴런의 passband 강조 그림

---

## 18. 이 명세의 권장 결론 문장 템플릿

실험이 잘 되면 해석은 대체로 아래 질문으로 정리할 수 있다.

- RF 뉴런 집단은 학습 후 **서로 다른 중심 주파수와 bandwidth를 갖는 필터 뱅크** 로 분화되는가?
- 이 분화는 입력 데이터의 스펙트럼과 정렬되는가?
- 과제 난이도나 클래스 구조에 따라 passband 분포가 달라지는가?
- reset 여부가 intrinsic band organization을 약화 또는 강화하는가?
- \$\omega\$와 \$b\$를 함께 학습할 때, 주파수 선택성과 memory span이 어떤 trade-off를 보이는가?

---

## 19. 최종 권고안

이 motivation study의 1차 구현은 아래 설정으로 고정하는 것이 가장 좋다.

| 항목 | 권장 설정 | 이유 |
|---|---|---|
| RF variant | vanilla RF | 해석 단순성 |
| trainable intrinsic params | \$\omega, b\$ only | 대역 해석의 순도 유지 |
| discretization | exact ZOH | 수치 안정성, 의미 보존 |
| spike rule | fixed threshold | 자유도 최소화 |
| reset | no-reset | 선형 동역학 해석 보존 |
| input projection | fixed | 대역 분화를 intrinsic dynamics에 귀속 |
| readout | optional linear head | 과제 수행 가능성 확보 |
| analysis target | subthreshold frequency response | 가장 직접적인 passband descriptor |

즉, **"학습은 스파이킹 네트워크로 하고, 해석은 선형화된 공명 동역학으로 한다"** 가 이 명세의 핵심이다.

---

## 20. 참고문헌

1. Eugene M. Izhikevich, *Resonate-and-Fire Neurons*, Neural Networks, 2001.  
   https://www.izhikevich.org/publications/resfire.pdf

2. Saya Higuchi, Sebastian Kairat, Sander Bohte, Sebastian Otte, *Balanced Resonate-and-Fire Neurons*, ICML 2024.  
   https://proceedings.mlr.press/v235/higuchi24a.html

