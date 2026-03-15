# Variable Dendritic Multiplicity (가변 수상돌기 수) — 공통 이론 정리

## 0. 목적

본 문서는 "수상돌기(가지) 개수" 를 하이퍼파라미터로 고정하지 않고, **연속 변수로 두어 학습 과정에서 자연스럽게 결정** 되도록 만드는 일반적 설계를 정리한다. 또한 수상돌기 동역학을 **LTI 필터** 로 해석하고, 가지 간 중복(상관) 을 억제하는 정규화 항을 정의한다.

> 적용 대상: DH-LIF / DH-SNN, D-RF, Reverse-DH 계열 등 "다중 수상돌기 + 세포체 집계" 구조 전반

---

## 1. 핵심 아이디어 요약

### 1.1 "가지 수" 가 학습되려면 출력 예측식에 직접 들어가야 함

가지 수가 구조적으로만 바뀌고(예: $D_n=\lfloor s_n\rfloor$) 예측식에 미분 가능한 경로로 들어가지 않으면, 일반적인 경사하강 기반 학습에서 $s$ 에 대한 그래디언트가 약해지거나 0이 된다.

따라서 다음 중 하나를 반드시 포함한다.

- 세포체 집계에서 $1/D_n$ 대신 $1/s_n$ 를 사용
- $s$ 를 이용한 게이팅/스케일링을 예측식에 포함
- 그리고 $s$ 자체에 대한 복잡도 패널티를 목적함수에 포함

---

## 2. 연속 가지-복잡도 파라미터 $s$ 와 정수 활성 가지 수 $D_n$

### 2.1 표기(중요: $s$ 는 뉴런 단위)

- 레이어 $l$, 뉴런 $n$ 별 연속 파라미터: $s^{(l,n)}$ (또는 벡터 $\mathbf{s}^{l}\in\mathbb{R}^{N^{l}}$)
- 텐서 shape 을 위한 **사전 할당(max) 가지 수:** $D$ (코드 `branch`)
- 뉴런 $n$ 의 가지 "추가/삭제" 는 $s_n$ 의 정수부/소수부로 해석한다.
  - clamp: $\tilde{s}_n=\min(D, s_n)$
  - (소프트) 완전 통과 가지 수(정수): $D_n=\lfloor \tilde{s}_n\rfloor$
  - 소프트 마스킹 잔여분: $\rho_n=\tilde{s}_n-D_n\in[0,1)$
  - (하드 전환) 정수 활성 가지 수: $D_n^{\text{hard}}=\left\lfloor \tilde{s}_n+\frac{1}{2}\right\rfloor$ (가장 가까운 정수, §4.4)

> $D$ 는 "허용 가능한 최대 가지 수(텐서 차원)"일 뿐이며, 학습되는 구조 파라미터는 $s_n$ 이다. 소프트 단계 forward 에서는 $(D_n,\rho_n)$ 로부터 $M_{n,d}(s_n)\in[0,1]$ 마스크를 만든다(§4.2). 하드 전환 이후에는 $D_n^{\text{hard}}$ 로부터 $M_{n,d}^{\text{hard}}\in\{0,1\}$ 마스크를 만든다(§4.4).

### 2.2 범위 제약을 위한 재매개화(권장, $S_{\min},S_{\max}$)

본 프로젝트에서는 $s$ 범위를 **$S_{\min},S_{\max}$** 로만 제약한다.

- 제약 없는 최적화 변수: $\hat{s}_n\in\mathbb{R}$
- 범위 사상:

  $$
  s_n = S_{\min} + (S_{\max}-S_{\min})\,\sigma(\hat{s}_n)
  $$
- 이렇게 하면 학습 중 $s_n$ 이 항상 $[S_{\min},S_{\max}]$ 를 유지한다.
- $S_{\max}$ 는 구현상 $D$ 보다 클 수도 있다.

  - 이때도 $D_n\le D$ 로 clamp 되지만, $s_n>D$ 구간은 $1/s_n$ 로 인해 soma drive 를 더 약화시키는 연속 제어로 작동한다.

### 2.3 (구현 주의) sigmoid 포화로 인한 $s$ 업데이트 정지

- $s_n$ 를 정확히 $S_{\max}$ 또는 $S_{\min}$ 에서 초기화하면, $\sigma(\hat{s})$ 가 0 또는 1 근처로 포화되어 $\partial s/\partial \hat{s}$ 가 거의 0이 될 수 있다.
- 따라서 초기화 시에는 정규화 비율

  $$
  r=\frac{s_{\text{init}}-S_{\min}}{S_{\max}-S_{\min}}
  $$

  을 $r\in[\epsilon,1-\epsilon]$ (예: $\epsilon=0.01$) 로 clamp 한 뒤 logit 을 취하는 방식이 안정적이다.

**(본 저장소의 기본 정책)** proposed(my_*) 계열 구현에서는 `s_init` 을 별도로 지정하지 않으면 **S_max 근처에서 시작** 하도록 둔다.

- 다만 위에서 설명한 sigmoid 포화 방지 정책 때문에, 실제 초기 $s$ 는 정확히 $S_{\max}$ 가 아니라

  $$
  s_{\text{init,actual}} = S_{\max} - \epsilon (S_{\max}-S_{\min})
  $$

  처럼 $S_{\max}$ 보다 약간 작은 값이 된다.
- 목적: 학습 초기에 거의 최대 가지 구조에서 시작한 뒤, 필요하면 정규화와 그래디언트로 가지 수가 줄어들도록 하여 구조 복잡도를 자연스럽게 결정한다.

### 2.4 (구현 주의) $S_{\max}$ 가 정수일 때 floor 경계 문제

sigmoid 기반 파라미터화에서는 $\sigma(\hat{s})<1$ 이므로 $s$ 가 $S_{\max}$ 에 "정확히" 도달하지 않는다. 이때

- $S_{\max}=D$ 처럼 정수 상한을 두면,
- $s\approx D-10^{-6}$ 형태가 되어 $\lfloor s\rfloor = D-1$ 이 되는 현상이 생길 수 있다.

소프트 마스크($\rho_n$ 사용)에서는 이 경우에도 마지막 가지 $d=D$ 가 $\rho_n\approx 1$ 로 **거의 완전 통과** 하므로, 실제 forward 동작은 크게 문제되지 않는다.

실무적 권장 해결책:

- 가장 단순/안전: $S_{\max}$ 를 $D+0.5$ 처럼 정수보다 약간 크게 둔다.
  그러면 학습 중 $s_n>D$ 영역에 들어갈 수 있고, clamp $\tilde{s}_n=\min(D,s_n)$ 로 인해 $D_n=\lfloor\tilde{s}_n\rfloor=D$ 및 $\rho_n=0$ 을 자연스럽게 얻을 수 있다.

---

## 3. "$1/s$ 집계" 를 통한 미분 경로 만들기

다중 가지 구조에서 세포체로 들어오는 집계를 일반형으로 쓰면 다음과 같다.

- 가지 상태(전류/막전위 등) 를 $x_{n,d}[t]$ 라 두고
- 가지 활성/가중 마스크를 $M_{n,d}(s_n)\in[0,1]$ 라 두면(§4),
- 세포체 drive 를

$$
g_n[t] = \frac{1}{s_n}\sum_{d=1}^{D} M_{n,d}(s_n)\,x_{n,d}[t]
$$

로 정의한다. (전통적으로는 $1/D_n$ 를 많이 사용)

이때 $g_n[t]$ 의 $s_n$ 에 대한 도함수는, $M$ 이 $s$ 에 의존하는 일반 경우에

$$
\frac{\partial g_n[t]}{\partial s_n}
=
-\frac{1}{s_n^2}\sum_{d=1}^{D} M_{n,d}(s_n)\,x_{n,d}[t]
+
\frac{1}{s_n}\sum_{d=1}^{D}\frac{\partial M_{n,d}(s_n)}{\partial s_n}\,x_{n,d}[t]
$$

로 쓸 수 있다.

> 하드 마스킹($M\in\{0,1\}$, 그리고 $M$ 을 $s$ 와 분리)인 경우 두 번째 항은 0이다. §4 의 소프트 마스크를 쓰면 "마지막 인덱스" 에서만(구간 내부에서) $\partial M/\partial s = 1$ 이 되어, $s$ 변화가 출력에 연속적으로 반영된다.

---

## 4. 구현: 사전 할당 + 소프트 마스킹(append/pop 대신)

이 프로젝트의 기본 구현은 "가변 가지 수" 를 위해 가지를 동적으로 append/pop 하지 않는다. 대신

1. 항상 $D$ 개 가지 상태/파라미터를 **사전 할당** 하고,
2. 뉴런별 $s_n$ 로부터 $(D_n,\rho_n)$ 를 계산한 뒤,
3. $[0,1]$ 마스크 $M_{n,d}(s_n)$ 로 가지를 게이팅한다.

### 4.1 목표: 하드 마스킹 충격 완화

기존의 이진 마스크 $M_{n,d}\in\{0,1\}$ 는 $s_n$ 이 정수 경계를 넘을 때 마지막 가지가 **즉시 0→1(또는 1→0)** 로 바뀌어, soma drive 와 내부 상태가 불연속적으로 변할 수 있다.

따라서 본 프로젝트는 (1) 학습 중에는 §4.2의 소프트 마스크로 연속성을 확보하고, (2) 소프트→하드 전환 시에는 §4.4의 "가장 가까운 정수" 정수화 및 §4.5의 전환 직전 STE 실행으로 낙폭을 줄인다.

### 4.2 규칙: 마지막 인덱스만 소프트하게 추가/삭제

$D_n=\lfloor\tilde{s}_n\rfloor$, $\rho_n=\tilde{s}_n-D_n$ 를 두면, 각 가지 인덱스 $d\in\{1,\ldots,D\}$ 에 대한 마스크를

$$
M_{n,d}(s_n)=
\begin{cases}
1, & d \le D_n \\
\rho_n, & d = D_n+1 \ \text{and}\ D_n < D \\
0, & d \ge D_n+2
\end{cases}
$$

로 둔다.

- $d\le D_n$ : 마스크 1(완전 통과)
- $d=D_n+1$ : 마스크 $\rho_n=\tilde{s}_n-\lfloor \tilde{s}_n\rfloor$(소프트 마스킹)
- 그 외 : 0(차단)

이 정의는 $s_n$ 이 연속적으로 변할 때, "마지막 인덱스" 의 가지 하나만 $[0,1]$ 연속 게이트로 켜지고/꺼지도록 만든다. 또한 $\sum_{d=1}^{D} M_{n,d}(s_n)=\tilde{s}_n=\min(D,s_n)$ 이라서, $s_n$ 을 **soft branch count** 로 해석할 수 있다.

### 4.3 적용 위치 (중요: 마스크는 1회만 적용)

구현은 기존과 동일하게 텐서 shape 을 고정한 채 마스크로 게이팅한다.
단, forward 에서 **마스크 $M_{n,d}(s_n)$는 한 번만** 적용한다.

- 가지 입력/상태 갱신으로 얻은 $\tilde{x}_{n,d}$에 대해

  $$
  x_{n,d}\leftarrow M_{n,d}(s_n) \cdot \tilde{x}_{n,d}
  $$
- soma 집계는 **이미 마스킹된 $x$** 를 그대로 합산한다:

  $$
  g_n[t]=\frac{1}{s_n}\sum_{d=1}^{D} x_{n,d}[t]
  $$

장점:

- 텐서 shape 이 고정되어 GPU 병렬화/벡터화가 쉬움
- 가지 수 변화가 연속적으로 반영되어 안정적(상태 초기화/불연속 drive 문제 완화)

### 4.4 하드 전환(hardening): 가장 가까운 정수로 정수 가지 수 결정

소프트 마스킹 단계에서는 $\rho_n$ 로 인해 마지막 가지가 연속적으로 켜지고/꺼지지만, 배포/측정/연산량 산정을 위해서는 최종적으로 $M\in\{0,1\}$ 의 하드 구조가 필요하다. 이때 단순히 $D_n=\lfloor\tilde{s}_n\rfloor$ 로 down-rounding 하면, $\rho_n\approx 1$ 인 "거의 활성" 가지가 전환 순간에 통째로 삭제되어 큰 성능 낙폭이 발생할 수 있다.

따라서 하드 전환 시 정수 활성 가지 수는 "가장 가까운 정수"로 정의한다.

$$
D_n^{\text{hard}} = \left\lfloor \tilde{s}_n + \frac{1}{2} \right\rfloor,
\qquad
1 \le D_n^{\text{hard}} \le D
$$

이에 따른 하드 마스크는

$$
M_{n,d}^{\text{hard}} = \mathbb{1}\left[d \le D_n^{\text{hard}}\right]
$$

로 둔다.

- $\rho_n<0.5$ 인 경우: 해당 "마지막 가지"는 삭제된다.
- $\rho_n\ge 0.5$ 인 경우: 해당 가지는 유지된다.
- 구현에서는 `round` 의 banker rounding(0.5가 짝수로 가는 규칙)을 피하기 위해, $\lfloor x+1/2\rfloor$ 형태(half-up)를 권장한다.

### 4.5 전환 충격 완화: 전환 직전 STE(직진 추정기) 실행

하드 마스크로의 전환을 완화하기 위해, 하드 전환 직전 일부 epoch 동안 forward 는 하드 구조를 사용하되 backward 는 소프트 그래디언트를 사용하는 STE 를 적용한다.

- CLI 인수: `--ste_epochs` (기본 0)
- 정의: 하드 전환 시점이 $t_{\text{hard}}$ (epoch index)일 때, $t\in[t_{\text{hard}}-T_{\text{ste}},\,t_{\text{hard}})$ 구간에서 STE 를 사용한다. 여기서 $T_{\text{ste}}$ 는 `ste_epochs` 이다.

구현 개념은 다음과 같다. 소프트 출력을 $y_{\text{soft}}$, 하드 출력을 $y_{\text{hard}}$ 라 하면,

$$
y = y_{\text{soft}} + \operatorname{stopgrad}\left(y_{\text{hard}} - y_{\text{soft}}\right)
$$

- forward 값은 $y_{\text{hard}}$ 와 동일하게 동작한다.
- backward 는 $y_{\text{soft}}$ 경로의 도함수를 사용하므로, 연속 파라미터 $s$ 및 가지 파라미터들이 계속 학습된다.

전환 이후(하드 프루닝 단계)에는 $D_n^{\text{hard}}$ 를 고정하고, `stabilize_epochs` 동안 하드 구조로 미세조정(fine-tuning)한다.

---

## 5. 수상돌기 동역학을 LTI 필터로 해석하기(EMA 예)

많은 다중-수상돌기 뉴런에서 가지 상태는 1차 누설 적분 형태(EMA) 로 주어진다.

$$
x_{n,d}[t+1] = \alpha_{n,d}\,x_{n,d}[t] + (1-\alpha_{n,d})\,u_{n,d}[t+1],
\qquad 0<\alpha_{n,d}<1
$$

- $u_{n,d}[t]$ : 가지 입력(시냅스 전류, spike 집계, 실수 입력 등)

이 갱신은 **1차 IIR 저역통과(LTI) 필터** 와 동치이며, 임펄스 응답은

$$
h_d[n] = (1-\alpha_d)\,\alpha_d^{n},
\qquad n\ge 0
$$

이다. 즉, 각 가지는 서로 다른 $\alpha_d$ 를 가지면 서로 다른 시간상수(기억 길이) 를 갖는 필터 뱅크로 해석할 수 있다.

### 5.1 전달함수 $H_d(z)$ (EMA 1차 IIR)

가지 $d$ 의 입력을 $u[t]$, 출력을 $x_d[t]$ 라 하고

$$
x_d[t] = \alpha_d x_d[t-1] + (1-\alpha_d)u[t]
$$

로 두면, $z$-변환에서

$$
H_d(z)=\frac{X_d(z)}{U(z)}=\frac{1-\alpha_d}{1-\alpha_d z^{-1}}
$$

이다.

주파수 응답은 $z=e^{j\omega}$ 로 두면

$$
H_d(e^{j\omega})=\frac{1-\alpha_d}{1-\alpha_d e^{-j\omega}}
$$

이며, 크기 제곱은

$$
\lvert H_d(e^{j\omega})\rvert^2
=
\frac{(1-\alpha_d)^2}{1+\alpha_d^2-2\alpha_d\cos\omega}
$$

이다.

- $\omega=0$ 에서 $\lvert H_d(e^{j\cdot 0})\rvert=1$ (DC 이득 1)이다.

### 5.2 $-3\,\mathrm{dB}$ 컷오프 주파수 $\omega_{c,d}$ 의 닫힌형

DC 대비 $-3\,\mathrm{dB}$ 점을

$$
\lvert H_d(e^{j\omega_{c,d}})\rvert^2 = \frac{1}{2}
$$

로 정의하면(DC 이득 1 기준), 위 식을 풀어

$$
\cos\omega_{c,d}
=
\frac{4\alpha_d-\alpha_d^2-1}{2\alpha_d}
$$

를 얻는다.

따라서

$$
\omega_{c,d}
=
\arccos\left(\operatorname{clip}\left(\frac{4\alpha_d-\alpha_d^2-1}{2\alpha_d},-1,1\right)\right)
$$

로 계산할 수 있다.

- $\alpha_d$ 가 매우 작으면(매우 빠른 가지), 필터가 거의 평탄해져 $-3\,\mathrm{dB}$ 점이 $[0,\pi]$ 내에 존재하지 않을 수 있다.
  - 이 경우 위 clip 으로 인해 $\omega_{c,d}\approx \pi$ 로 포화된다.

### 5.3 $\omega$ 를 Hz 로 바꾸기

시뮬레이션 시간 간격을 $\Delta t$ 초로 두면, rad/sample 주파수 $\omega$ 를 Hz 로 바꿔

$$
f = \frac{\omega}{2\pi\Delta t}
$$

로 쓸 수 있다.

또한 일부 구현에서는 시간상수 $\tau_d$ 로부터

$$
\alpha_d = \exp\left(-\frac{\Delta t}{\tau_d}\right)
$$

처럼 $\alpha_d$ 를 정의한다. 이때 $\omega_{c,d}(\alpha_d)$ 공식을 그대로 대입하면 $\tau_d$ 의 닫힌형 함수로 $f_{c,d}$ 를 얻는다.

---

## 6. 가지 중복 억제: 지수 커널 상관(코사인) 정규화

### 6.1 동기

$D$ 를 늘려도 $\alpha_d$ 들이 서로 비슷하면, 실질적으로 같은 필터를 중복 추가한 것과 유사해진다.
따라서 가지 간 필터 커널 $h_d$ 들이 **서로 덜 상관** 되도록 하는 정규화가 필요하다.

### 6.2 지수 커널 코사인 유사도(무한 지평, 닫힌형)

두 가지 $i,j$ 의 커널 $h_i,h_j$ 에 대해(무한 길이 $n=0\ldots\infty$):

- 내적

$$
\langle h_i, h_j\rangle
=
\sum_{n=0}^{\infty} h_i[n]h_j[n]
=
\frac{(1-\alpha_i)(1-\alpha_j)}{1-\alpha_i\alpha_j}
$$

- 노름

$$
\|h_i\|^2
=
\sum_{n=0}^{\infty} h_i[n]^2
=
\frac{(1-\alpha_i)^2}{1-\alpha_i^2}
=
\frac{1-\alpha_i}{1+\alpha_i}
$$

따라서 코사인 유사도는

$$
\cos(h_i,h_j)
=
\frac{\langle h_i,h_j\rangle}{\|h_i\|\,\|h_j\|}
=
\frac{\sqrt{(1-\alpha_i^2)(1-\alpha_j^2)}}{1-\alpha_i\alpha_j}
$$

이다. $\alpha_i\approx\alpha_j$ 일수록 $\cos(h_i,h_j)\to 1$ 에 가까워진다.

### 6.3 직교(비상관) 유도 정규화 항(활성 가지에 대해서만)

소프트 마스킹을 쓰면 "활성 가지" 를 $M_{n,d}(s_n)>0$ 인 가지로 보는 것이 자연스럽다. 가장 단순한 구현은 마스크로 가중한 합으로 쓰는 것이다.

$$
\mathcal{L}_{\text{ortho},n}
=
\sum_{1\le i<j\le D} M_{n,i}(s_n)\,M_{n,j}(s_n)\,\cos(h_i,h_j)^2
$$

- 하드 마스킹($M\in\{0,1\}$)에서는 기존의 $\sum_{1\le i<j\le D_n}$ 와 동일한 효과를 갖는다.
- 소프트 마스킹에서는 마지막 가지($d=D_n+1$)가 $\rho_n$ 비율로만 패널티에 기여한다.

### 6.4 수치 안정화

- 분모 $1-\alpha_i\alpha_j$ 에 $\varepsilon$ 를 더해

$$
1-\alpha_i\alpha_j+\varepsilon
$$

로 구현한다.

- $\alpha\to 1$ 근방에서 민감해질 수 있으므로, 다음처럼 상한을 제한하는 파라미터화가 안정적이다.

$$
\alpha_d
=
\alpha_{\min}+(\alpha_{\max}-\alpha_{\min})\,\sigma(\hat{\alpha}_d),
\qquad 0<\alpha_{\min}<\alpha_{\max}<1
$$

---

## 7. 학습 안정화: $1/s$ 로 인한 분산 붕괴(핵심 이슈)

$1/s$ 평균화는 $s$ 학습에 필요하지만, **초기화 분산(variance) 관점에서는 신호를 지나치게 약화** 시킬 수 있다.
즉, "스파이크가 전혀 안 나오는 dead neuron" 고정점으로 빠질 위험이 커진다.

### 7.1 분산 스케일 mismatch 의 정성적 설명

- 가지 출력 $x_d$ 들이 서로 독립이라면:
  - $\sum_{d=1}^{D} x_d$ 의 표준편차는 $\sqrt{D}$ 에 비례
  - 평균 $D^{-1}\sum x_d$ 의 표준편차는 $1/\sqrt{D}$ 로 감소
- $s\approx D$ 일 때 $1/s$ 평균화도 동일한 약화를 만든다.

따라서 "기본 Kaiming 초기화"만 쓰면, 설계자가 기대한 발화 스케일보다 입력이 작아져 스파이크가 사라질 수 있다.

### 7.2 권장 초기화 스케일 보정(구조별)

아래는 "초기 soma drive 크기를 보존"하기 위한 실무 규칙이다.

| 구조                                                                                         | soma 집계 형태                   | 권장 스케일 보정                                                                                                                                          |
| -------------------------------------------------------------------------------------------- | -------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- |
| DH-SNN (가지별 독립$\mathbf{W}_d$)                                                         | $H=\frac{1}{s}\sum_d i_d$      | 초기$\mathbf{W}\leftarrow \sqrt{D}\,\mathbf{W}$                                                                                                         |
| Reverse DH-SNN (soma-dense$W_{\text{in}}$ + mixing $W_{\text{mix}}$, dendrite-broadcast) | $H=\frac{1}{s}\sum_d w_d\,i_d$ | 초기$W_{\text{mix}}\leftarrow D\,W_{\text{mix}}$ + $W_{\text{in}}$ 은 fan-in($=K$) 기반 초기화(예: $\mathrm{Var}(W_{\text{in}}[m,k])\propto 1/K$) |

> 요지: "sum 후 $1/s$" 로 인해 줄어드는 스케일을, 초기 가중치에 미리 반영한다.

Reverse DH-SNN의 soma-dense 입력은 다음 형태를 갖는다.

$$
O_{\text{soma}}[t] = W_{\text{in}}\, O[t],\qquad W_{\text{in}}\in\mathbb{R}^{M\times K}
$$

(뉴런 $m$ 단위로 쓰면 $O_{\text{soma},m}[t] = \mathbf{w}_{\text{in},m}^{\top} O[t]$ 이고, 이 값이 해당 뉴런의 모든 가지로 broadcast 된다.)

따라서 $K$ 가 변해도 각 뉴런의 $\mathrm{Var}(O_{\text{soma},m})$ (입력 스케일)이 크게 흔들리지 않도록, $W_{\text{in}}$ 의 초기화를

$$
W_{\text{in}}[m,k] \sim \mathcal{N}(0,\,1/K)\quad (\text{즉 } \mathrm{std}=1/\sqrt{K})
$$

처럼 설정하는 것이 안전하다.

### 7.3 임계값 $u_{\mathrm{th}}$ 변경 시

임계값을 원 논문/Origin 코드와 다르게 바꾸면(예: $u_{\mathrm{th}}=1.0$ ), 동일한 입력에서도 초기 발화율이 크게 변한다.
따라서 필요하면 **가중치 초기 스케일을 $u_{\mathrm{th}}$ 에 비례하게 재보정** 하는 것이 안정적이다.

---

## 8. 목적함수 템플릿(총 손실)

task loss를 $\mathcal{L}_{\text{task}}$ , s 자체에 대한 손실을 $\mathcal{L}_{\text{s}}$ 라고 하면, 일반적 총 손실은

$$
\mathcal{L}
=
\mathcal{L}_{\text{task}}
+
\lambda_{\text{ortho}}\sum_{n}\mathcal{L}_{\text{ortho},n}
+
\lambda_{\text{s}}\sum_{n} \mathcal{L}_{\text{s}}
$$

로 둘 수 있다.

- $\lambda_{\text{ortho}}$ : 시간필터 중복 억제
- $\lambda_s$ : $s$ 자체에 대한 복잡도 비용(과도한 가지 증가 억제)

또한 $\mathcal{L}_{\text{s}} = \frac{1}{\text{총 뉴런 개수}}\sum_{\text{모든 뉴런}}{s_n}$ 으로 정의, 고정한다

---

## 9. 구현 체크리스트

- $s$ 의 범위: $S_{\min},S_{\max}$
- $s$ 는 **뉴런 단위**
- 텐서 사전 할당: 최대 가지 수 $D$
- 소프트 마스크: $M_{n,d}(s_n)\in[0,1]$
  - $d\le\lfloor\tilde{s}_n\rfloor$ 는 1, $d=\lfloor\tilde{s}_n\rfloor+1$ 은 $\tilde{s}_n-\lfloor\tilde{s}_n\rfloor$, 그 외 0
- 구현: 사전 할당 + 마스크로 비활성 가지 0 고정(마지막 가지는 소프트 게이트)
- 세포체 집계: $1/s$
- 하드 전환(hardening): $D_n^{\text{hard}}=\left\lfloor \tilde{s}_n+\frac{1}{2}\right\rfloor$ 및 $M_{n,d}^{\text{hard}}\in\{0,1\}$
- 전환 직전 STE(옵션): `--ste_epochs` 동안 $y = y_{\text{soft}} + \operatorname{stopgrad}(y_{\text{hard}}-y_{\text{soft}})$
- $\alpha$ 파라미터화 및 안정화($\alpha_{\max}$, $\varepsilon$)
- 정규화 비용: $\lambda_{\text{ortho}},\lambda_s$
