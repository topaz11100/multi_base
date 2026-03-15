# Vanilla RF ordered band clipping motivation study 명세서

## 0. 목적과 범위

이 문서는 `paper/proposed/vanilla_rf_wb_only_spec.md` 의 vanilla RF 뉴런 동역학을 바탕으로, 공진주파수 해석축인 $\omega$ 를 직접 구조화하는 motivation study 를 정의한다. 비교 대상은 아래 4개 실험군이다.

1. 자유형 vanilla RF 모델
2. 뉴런 인덱스 순서에 따라 $\omega$ 대역을 지정하고 그 구간 안에서만 학습되도록 제한한 ordered band clipping 모델
3. 주파수 clipping 은 하지 않지만, 같은 band assignment 집합끼리만 hidden-to-hidden dense connection 을 허용하고 나머지 연결을 고정 마스크로 차단한 structural same-group dense mask 모델
4. 2번 모델에 더해, 같은 band assignment 집합끼리만 hidden-to-hidden dense connection 을 허용하는 ordered band clipping + same-group dense mask 모델

본 study 는 **SHD 데이터셋 하나를 기준으로 위 4개 시나리오를 병렬 실행** 하고, 모든 주파수 통계를 discrete normalized frequency 기준으로 비교한다.

본 문서의 목적은 아래 두 가지다.

- train/test accuracy 의 epoch 추이를 비교한다.
- `paper/proposed/r_dh_snn_transfer_analysis.md` 의 공통 규격을 vanilla RF 에 맞게 확장하여, epoch 단위 대역 통계와 band assignment 정합도를 비교한다.
- clipping 자체의 효과와 구조적 연결 차단 자체의 효과를 분리해서 본다.
- **중요** : ablation study 를 통해 주파수 대역을 강제로 나누거나, 또는 그 대역에 맞춘 구조적 연결만 남겼을 때 정확도와 통계가 어떻게 달라지는지 살핀다.

본 문서는 실험 설계와 저장 규약을 정의한다. 뉴런의 exact ZOH 이산화와 안정 파라미터화의 세부 수식은 vanilla RF 기본 문서를 그대로 따른다.

## 0.1 표기 규칙

이 문서에서는 공진주파수는 $\omega$ 로, 덴스 연결 가중치는 $W$ 로 표기한다. 다만 Bash/CLI 이름은 사용 편의를 위해 `w_clip_edges` 라는 키를 사용한다. 즉, `w_clip_edges` 는 덴스 가중치가 아니라 공진주파수 축의 clip 경계를 뜻한다.

ordered band clipping 은 raw continuous-time $\omega$ 자체가 아니라, 아래의 discrete normalized frequency 로 정의한다.

$$
f^{(\omega)} = \frac{\phi}{2\pi} = \frac{\omega \Delta}{2\pi} \in [0, 0.5]
$$

여기서 $\Delta$ 는 데이터 timestep 이고, $\phi = \omega \Delta$ 는 discrete pole angle 이다. 사용자가 주는 clip 경계는 모두 $f^{(\omega)}$ 기준으로 해석한다. 따라서 예시 `0.0 0.2 0.35 0.5` 는 cycles/step 축의 band edge 다.

---

## 1. 핵심 연구 질문

이 motivation study 는 아래 질문에 답하는 것을 목표로 한다.

1. vanilla RF 에서 $\omega$ 를 자유롭게 학습시키는 경우와, 뉴런 인덱스 기반 band clipping 을 강제하는 경우의 accuracy 차이는 얼마나 큰가.
2. ordered band clipping 이 실제 필터 통계에서도 유지되는가. 즉, 학습 후 수치적으로 계산한 $f_{\mathrm{peak}}$ 와 $bw_{3\mathrm{dB}}$ 가 지정 band 와 정합되는가.
3. clipping 없이 같은 group 끼리의 hidden-to-hidden dense connection 만 남겨도, band purity 는 증가하는가. 반대로 accuracy 는 얼마나 변하는가.
4. clipping 과 structural mask 를 동시에 적용하면, 두 효과가 additive 하게 나타나는가.
5. 구조적 band 분리가 생기면 `lp`, `bp`, `hp`, `mixed` 분포와 band coverage 가 어떻게 달라지는가.

---

## 2. 공정 비교 원칙

세 실험군은 아래 항목을 동일하게 유지한다.

- SHD 데이터셋과 전처리
- timestep $\Delta$
- hidden layer 개수와 각 layer width
- threshold, surrogate gradient, reset 정책
- optimizer, scheduler, batch size, epoch 수, seed
- dense weight 초기화 방식
- $\rho$ 또는 $b$ 의 허용 범위와 초기화 분포
- filter_property exact grid 크기와 userbin 계산 방식
- 모든 주파수 통계의 계산 축(discrete normalized frequency)
- 출력층 뉴런 마지막 타임스텝 막전위에 대한 softmax 예측 규칙

변하는 것은 아래 두 가지뿐이다.

- hidden neuron 별 $\omega$ band 제약 유무
- hidden-to-hidden dense connection 에 same-group mask 를 적용하는지 여부

이 연구에서는 parameter matching 을 위해 hidden width 를 보정하지 않는다. 즉, 실험군 3은 활성 dense parameter 수가 감소할 수 있으며, 이것 자체를 구조 변화의 일부로 간주한다. 대신 결과 보고 시 총 파라미터 수와 활성 파라미터 수를 함께 기록한다.

---

## 3. 공통 모델 아키텍처

## 3.1 전체 구조

기본 분류기는 SHD 입력 시퀀스를 처리하는 $L \ge 2$ 개의 hidden vanilla RF layer 와, 마지막 타임스텝 막전위를 logits 로 사용하는 비스파이킹 output membrane layer 로 구성한다.

$$
x_t^{(0)} = u_t
$$

$$
I_t^{(1)} = W^{(1)} u_t
$$

$$
I_t^{(\ell)} = W_{\mathrm{eff}}^{(\ell)} o_t^{(\ell-1)}, \qquad \ell = 2, \dots, L
$$

$$
I_t^{(\mathrm{out})} = W_{\mathrm{out}} o_t^{(L)} + b_{\mathrm{out}}
$$

$$
\nu_{t+1}^{(\mathrm{out})} = \rho_{\mathrm{out}} \, \nu_t^{(\mathrm{out})} + I_t^{(\mathrm{out})}
$$

$$
\mathrm{logits} = \nu_T^{(\mathrm{out})}
$$

$$
\hat{\mathbf{p}} = \mathrm{softmax}(\mathrm{logits})
$$

여기서

- $u_t$ 는 SHD 입력 시점 $t$ 의 입력 벡터
- $o_t^{(\ell)}$ 는 hidden layer $\ell$ 의 spike output
- $x_t^{(\ell)}$ 는 hidden layer $\ell$ 의 실수부 resonant state
- $W_{\mathrm{eff}}^{(\ell)}$ 는 실험군에 따라 마스크가 적용될 수 있는 hidden-to-hidden weight matrix
- $\nu_t^{(\mathrm{out})}$ 는 class 수 차원의 비스파이킹 output membrane vector
- $\rho_{\mathrm{out}} \in (0,1]$ 는 네 실험군에서 공통으로 고정하는 output membrane leak 계수

이다.

입력에서 첫 번째 hidden layer 로 가는 $W^{(1)}$ 와 output layer 입력 가중치 $W_{\mathrm{out}}$ 은 네 실험군 모두 fully dense 로 유지한다. same-group mask 는 hidden-to-hidden connection 에만 적용한다.

## 3.2 hidden vanilla RF cell

hidden layer $\ell$ 의 뉴런 $m$ 에 대해 상태를

$$
h_{\ell,m,t} = \begin{bmatrix} x_{\ell,m,t} \\ y_{\ell,m,t} \end{bmatrix}
$$

로 두면 exact ZOH 업데이트는

$$
h_{\ell,m,t+1} = A_{\ell,m} h_{\ell,m,t} + B_{\ell,m} I_{\ell,m,t}
$$

이다. 여기서

$$
A_{\ell,m} = \rho_{\ell,m}
\begin{bmatrix}
\cos \phi_{\ell,m} & -\sin \phi_{\ell,m} \\
\sin \phi_{\ell,m} & \cos \phi_{\ell,m}
\end{bmatrix}
$$

$$
\rho_{\ell,m} = e^{b_{\ell,m}\Delta}, \qquad \phi_{\ell,m} = \omega_{\ell,m}\Delta
$$

이고 $B_{\ell,m} = [\beta_{x,\ell,m}, \beta_{y,\ell,m}]^\top$ 는 `vanilla_rf_wb_only_spec.md` 의 exact ZOH 수식을 그대로 사용한다.

스파이크는

$$
o_{\ell,m,t+1} = H\bigl(x_{\ell,m,t+1} - \theta\bigr)
$$

로 정의한다. 본 study 의 기본 설정은 `no-reset` 이다. 따라서 spike 생성 뒤 내부 상태 jump 는 없다.

## 3.3 trainable parameter 범위

이 study 에서 trainable parameter 는 아래로 한정한다.

- hidden layer dense weight $W^{(1)}, \dots, W^{(L)}$
- output membrane layer 입력 가중치 $W_{\mathrm{out}}, b_{\mathrm{out}}$
- hidden neuron 별 raw frequency parameter
- hidden neuron 별 raw damping parameter

threshold, output membrane leak $\rho_{\mathrm{out}}$, clip band 경계, dense mask 는 모두 non-trainable 이다.

## 3.4 출력층과 layer 수에 대한 제약

same-group dense mask 는 hidden-to-hidden connection 에 정의되므로, 실험군 3은 hidden layer 개수가 최소 2개여야 한다. 따라서 본 study 의 기본 hidden 설정은 단일 layer 가 아니라 2개 이상이다.

`hidden=[N_1, N_2]` 가 최소 권장값이다.

---

## 4. 네 실험군 정의

## 4.1 실험군 1: 자유형 vanilla RF

이 실험군의 hidden neuron 은 공통 global support 안에서 자유롭게 $\omega$ 를 학습한다. hidden-to-hidden connection 은 fully dense 다.

effective normalized frequency 는

$$
f^{(\omega)}_{\ell,m} = f_{\min} + (f_{\max} - f_{\min}) \sigma\bigl(u^{(\omega)}_{\ell,m}\bigr)
$$

로 둔다. 여기서 기본 global support 는

$$
[f_{\min}, f_{\max}] = [e_0, e_G]
$$

이며, $e_0, \dots, e_G$ 는 band assignment 실험에서 사용할 전체 경계 집합의 양 끝값과 동일하게 둔다.

이 실험군은 비교 기준선 역할을 한다.

## 4.2 실험군 2: ordered band clipping vanilla RF

이 실험군은 hidden layer 내부 뉴런 인덱스 순서에 따라 band group 을 미리 할당하고, 각 뉴런의 $f^{(\omega)}$ 가 자신에게 지정된 band 안에서만 학습되도록 제한한다. dense connection 은 여전히 fully dense 다.

band edge 를

$$
0 \le e_0 < e_1 < \cdots < e_G \le 0.5
$$

로 두고, band $g$ 를

$$
\mathcal{B}_g = [e_{g-1}, e_g), \qquad g = 1, \dots, G-1
$$

$$
\mathcal{B}_G = [e_{G-1}, e_G]
$$

로 정의한다.

layer $\ell$ 의 hidden width 가 $N_\ell$ 일 때, cumulative neuron end index 를

$$
0 = c_{\ell,0} < c_{\ell,1} < \cdots < c_{\ell,G-1} < c_{\ell,G} = N_\ell
$$

로 둔다. 그러면 뉴런 $m$ 의 group id 는

$$
g_\ell(m) = g \iff c_{\ell,g-1} \le m < c_{\ell,g}
$$

이다.

이때 effective normalized frequency 는

$$
f^{(\omega)}_{\ell,m} = e_{g_\ell(m)-1} + \bigl(e_{g_\ell(m)} - e_{g_\ell(m)-1}\bigr) \sigma\bigl(u^{(\omega)}_{\ell,m}\bigr)
$$

로 정의한다.

즉, 본 명세의 규범 구현은 hard post-step clip 이 아니라 band-bounded parameterization 이다. 필요하면 optimizer step 뒤 아래 투영으로 동치 구현할 수 있다.

$$
f^{(\omega)}_{\ell,m} \leftarrow \operatorname{clip}\bigl(f^{(\omega)}_{\ell,m}, e_{g_\ell(m)-1}, e_{g_\ell(m)}\bigr)
$$

하지만 보고 및 저장되는 유효 주파수는 항상 위 band 내부 값이어야 한다.

## 4.3 실험군 3: structural same-group dense mask without clipping

이 실험군은 모든 hidden neuron 의 $f^{(\omega)}$ 를 실험군 1과 동일한 global support 안에서 자유롭게 학습시킨다. 다만 각 뉴런에 대해 실험군 2와 동일한 band assignment $g_\ell(m)$ 를 미리 정의하고, 이 assignment 를 **구조적 연결 마스크에만** 사용한다.

즉, frequency parameterization 은

$$
f^{(\omega)}_{\ell,m} = f_{\min} + (f_{\max} - f_{\min}) \sigma\bigl(u^{(\omega)}_{\ell,m}\bigr)
$$

로 유지하되, hidden-to-hidden connection mask 는

$$
M^{(\ell)}_{ij} = \mathbb{1}\bigl[g_\ell(i) = g_{\ell-1}(j)\bigr], \qquad \ell = 2, \dots, L
$$

로 정의한다.

effective dense matrix 는

$$
W_{\mathrm{eff}}^{(\ell)} = W^{(\ell)} \odot M^{(\ell)}
$$

로 적용한다.

따라서 이 실험군은 **clipping 없음 + 구조적 same-group 연결만 유지** 라는 효과를 분리해서 측정한다.

## 4.4 실험군 4: ordered band clipping + same-group dense mask

이 실험군은 실험군 2의 band-bounded parameterization 과 실험군 3의 same-group dense mask 를 동시에 적용한다.

즉,

$$
f^{(\omega)}_{\ell,m} = e_{g_\ell(m)-1} + \bigl(e_{g_\ell(m)} - e_{g_\ell(m)-1}\bigr) \sigma\bigl(u^{(\omega)}_{\ell,m}\bigr)
$$

와

$$
W_{\mathrm{eff}}^{(\ell)} = W^{(\ell)} \odot M^{(\ell)}
$$

를 함께 사용한다.

입력층과 output membrane layer 는 group 개념이 없으므로 아래 두 연결은 마스크하지 않는다.

- $u_t \to$ hidden1
- hidden$L \to$ output membrane layer

이 실험군의 핵심 목적은 clipping 과 cross-band 연결 차단을 동시에 적용했을 때 accuracy 와 filter_property 가 어떻게 변하는지 측정하는 것이다.


---

## 5. band 배치와 CLI 규격

## 5.1 band 경계 입력 규격

공진주파수 band assignment 경계는 Python CLI 에서 아래처럼 받는다.

```text
--w_clip_edges 0.0 0.2 0.35 0.5
```

의미는 아래와 같다.

- 총 band 수는 $G = \text{len}(w\_clip\_edges) - 1$
- 각 값은 cycles/step 단위다.
- 마지막 값은 기본적으로 0.5 이하여야 한다.
- 내부 구현은 필요하면 $[\varepsilon_f, 0.5-\varepsilon_f]$ 로 수치 clamp 할 수 있지만, config 에 저장되는 사용자 값은 원본을 유지한다.

## 5.2 layer 별 뉴런 구간 입력 규격

각 hidden layer 의 cumulative group end index 는 문자열 리스트로 받는다.

```text
--band_neuron_ends 10,20 10,20
```

위 예시는 hidden layer 가 2개이고 각 layer width 가 최소 21 이상일 때 아래를 의미한다.

- hidden1: $[0,10)$, $[10,20)$, $[20,N_1)$
- hidden2: $[0,10)$, $[10,20)$, $[20,N_2)$

즉, 사용자 예시인

```text
레이어 1의 10번 뉴런까진 [0,0.2), 20번 뉴런까진 [0.2,0.35), 나머지는 [0.35,0.5]
```

를 그대로 표현한다.

입력 검증 규칙은 아래와 같다.

1. `len(w_clip_edges) - 1 = G`
2. 각 hidden layer 마다 cumulative end index 개수는 정확히 $G-1$ 개여야 한다.
3. 각 cumulative end index 는 정수여야 하며 strictly increasing 해야 한다.
4. 마지막 group 의 끝은 자동으로 해당 layer width 로 해석한다.
5. 실험군 3과 4에서는 hidden layer 개수가 최소 2개여야 한다.

## 5.3 권장 Python CLI 추가 인수

본 study 의 runner 는 기존 `freq_analysis` 계열 CLI 에 아래 인수를 추가하는 것을 표준으로 한다.

```text
--models vanilla_RF_free vanilla_RF_bandclip vanilla_RF_structmask vanilla_RF_bandmask
--hidden 128 128
--epochs 50
--w_clip_edges 0.0 0.2 0.35 0.5
--band_neuron_ends 10,20 10,20
--tracked_per_band 3
--filter_F 512
```

추가 인수의 의미는 아래와 같다.

- `models`: 네 실험군을 개별 model alias 로 병렬 실행한다.
- `hidden`: hidden layer width 리스트다. 실험군 3과 4 때문에 최소 길이 2를 권장한다.
- `epochs`: 고정 training budget 이다.
- `w_clip_edges`: ordered band clipping 경계다.
- `band_neuron_ends`: hidden layer 별 cumulative end index 다.
- `tracked_per_band`: 각 hidden layer 각 band 에서 추적할 뉴런 수다.
- `filter_F`: exact filter_property 계산용 주파수 grid 개수다.

`tracked_per_band` 기본값은 3으로 둔다. 각 band 에서 첫 번째, 중앙, 마지막 뉴런을 우선 선택한다. band 크기가 3보다 작으면 가능한 인덱스만 고른다.

## 5.4 권장 Bash wrapper 환경변수

Bash wrapper 는 아래 환경변수를 우선 지원한다.

```bash
MODELS="vanilla_RF_free vanilla_RF_bandclip vanilla_RF_structmask vanilla_RF_bandmask"
HIDDEN="128 128"
EPOCHS=50
W_CLIP_EDGES="0.0 0.2 0.35 0.5"
BAND_NEURON_ENDS="10,20 10,20"
TRACKED_PER_BAND=3
SEED=0
```

wrapper 내부에서는 아래처럼 배열을 만들어 Python 으로 넘긴다.

```bash
read -r -a MODELS_ARR <<< "${MODELS}"
read -r -a HIDDEN_ARR <<< "${HIDDEN}"
read -r -a W_CLIP_EDGES_ARR <<< "${W_CLIP_EDGES}"
read -r -a BAND_NEURON_ENDS_ARR <<< "${BAND_NEURON_ENDS}"
```

Python 호출 예시는 아래와 같다.

```bash
python -u run.py \
  --models "${MODELS_ARR[@]}" \
  --hidden "${HIDDEN_ARR[@]}" \
  --epochs "${EPOCHS}" \
  --w_clip_edges "${W_CLIP_EDGES_ARR[@]}" \
  --band_neuron_ends "${BAND_NEURON_ENDS_ARR[@]}" \
  --tracked_per_band "${TRACKED_PER_BAND}" \
  --seed "${SEED}"
```

---

## 6. 학습 규약

## 6.1 손실과 평가

분류 손실은 마지막 타임스텝 output membrane logits 에 대한 cross-entropy 를 사용한다. 즉, $\hat{\mathbf{p}} = \mathrm{softmax}(\nu_T^{(\mathrm{out})})$ 로 예측을 만들고 epoch 마다 train accuracy 와 test accuracy 를 모두 기록한다.

train accuracy 는 해당 epoch 전체 mini-batch 평균 정답률로 정의한다.

$$
\mathrm{Acc}_{\mathrm{train}}(e) = \frac{1}{N_{\mathrm{train}}} \sum_{n=1}^{N_{\mathrm{train}}} \mathbb{1}[\hat{y}_n = y_n]
$$

test accuracy 는 full test split 기준으로 계산한다.

$$
\mathrm{Acc}_{\mathrm{test}}(e) = \frac{1}{N_{\mathrm{test}}} \sum_{n=1}^{N_{\mathrm{test}}} \mathbb{1}[\hat{y}_n = y_n]
$$

validation split 이 별도로 존재하면 hyperparameter 선택은 validation 으로만 한다. validation split 이 없으면 고정 epoch budget 의 마지막 epoch test accuracy 를 primary scalar metric 으로 사용한다.

## 6.2 optimizer 와 고정 설정

권장 기본값은 아래와 같다.

- optimizer: AdamW
- base learning rate: $10^{-3}$
- gradient clipping: global norm 1.0
- reset: `none`
- threshold: 1.0 고정
- regularizer: 없음
- exact discretization: 사용
- hidden dense bias: 사용하지 않음
- output membrane bias: 허용 가능하나 모든 실험군에서 동일하게 유지

ordered band clipping 과 same-group dense mask 이외의 구조 차이는 금지한다.

## 6.3 초기화 규약

dense weight 와 output membrane weight 초기화는 네 실험군에서 동일한 distribution family 를 사용한다.

$damping$ 쪽은 공통 bounded parameterization 을 사용한다.

$$
\rho_{\ell,m} = \rho_{\min} + (\rho_{\max} - \rho_{\min}) \sigma\bigl(u^{(\rho)}_{\ell,m}\bigr)
$$

권장 범위는

$$
\rho \in [0.95, 0.999]
$$

이다.

frequency 초기화는 아래처럼 통제한다.

- 실험군 1, 3: 전체 support $[e_0, e_G]$ 안에서 log-uniform 또는 bounded-sigmoid 기반 초기화
- 실험군 2, 4: 각 뉴런이 배정된 band 내부에서 동일한 분포 family 로 초기화

즉, 실험군 2와 4는 band 제약 때문에 초기 분포 support 가 더 좁아지지만, 분포 family 자체는 동일하게 유지한다. 실험군 3은 구조 마스크만 적용하고 주파수 초기화 support 는 실험군 1과 동일하게 유지한다.

## 6.4 seed 반복

최종 비교는 단일 seed 결과만으로 결론내리지 않는다. 권장 반복 수는 5 seed 다.

$$
\mathcal{S} = \{0,1,2,3,4\}
$$

최종 표와 비교 플롯에는 mean 과 std 를 함께 넣는다.

---

## 7. vanilla RF 의 filter_property 분석 정의

이 절은 `paper/proposed/r_dh_snn_transfer_analysis.md` 의 vanilla RF 미정 부분을 이번 study 범위에서 구체화한다.

## 7.1 분석 대상 경로

vanilla RF 에서 분석 대상은 hidden neuron 의 scalar pre-activation $I_{\ell,m,t}$ 에서 spike comparator 입력인 $x_{\ell,m,t}$ 로 가는 선형 경로다. 즉, 공통 문서의 `soma_input` 은 vanilla RF 에서는 아래처럼 정의한다.

$$
\mathrm{soma\_input}_{\ell,m,t} := x_{\ell,m,t}
$$

spike, threshold, reset, output softmax readout 은 분석 대상에서 제외한다.

## 7.2 전달함수

뉴런 $m$ 의 상태공간은

$$
h_{m,t+1} = A_m h_{m,t} + B_m I_{m,t}
$$

$$
q_{m,t} = C h_{m,t}
$$

로 두고,

$$
C = \begin{bmatrix} 1 & 0 \end{bmatrix}, \qquad q_{m,t} = x_{m,t}
$$

로 정의한다. 그러면 exact discrete frequency response 는

$$
H_m\bigl(e^{j 2\pi f}\bigr) = C \bigl(e^{j 2\pi f} I - A_m\bigr)^{-1} B_m
$$

이다. magnitude 는

$$
A_m(f) = \left| H_m\bigl(e^{j 2\pi f}\bigr) \right|
$$

로 둔다.

exact grid 와 userbin grid 는 공통 분석 문서를 그대로 따른다. 권장 기본값은 `filter_F = 512` 이다.

## 7.3 공통 저장 통계

아래 통계는 모두 discrete normalized frequency $f \in [0,0.5]$ 축에서 계산한다. hidden layer 별로 아래 스칼라를 저장한다.

- `f_peak`
- `f_L`
- `f_H`
- `bw_3db`
- `r0`
- `rpi`
- `class`

분류 규칙 `lp`, `bp`, `hp`, `mixed` 와 $-3\,\mathrm{dB}$ passband 추출은 `r_dh_snn_transfer_analysis.md` 를 그대로 사용한다.

## 7.4 vanilla RF 전용 추가 통계

이번 study 는 $\omega$ band assignment 자체가 핵심이므로, 공통 통계에 더해 아래 값을 반드시 저장한다.

### 7.4.1 intrinsic discrete frequency

각 뉴런의 내부 공진주파수 proxy 를

$$
f^{(\omega)}_{\ell,m} = \frac{\phi_{\ell,m}}{2\pi}
$$

로 저장한다.

### 7.4.2 peak shift

수치적 전달함수 peak 와 내부 공진주파수 proxy 사이 차이를

$$
\Delta f_{\ell,m} = f_{\mathrm{peak},\ell,m} - f^{(\omega)}_{\ell,m}
$$

로 저장한다.

### 7.4.3 assigned band inclusion

실험군 2, 3, 4에서는 지정 band 를

$$
\mathcal{B}_{g_\ell(m)} = [e_{g_\ell(m)-1}, e_{g_\ell(m)})
$$

마지막 band 만

$$
\mathcal{B}_{G} = [e_{G-1}, e_G]
$$

로 두고, peak 포함 여부를

$$
\mathrm{peak\_inside}_{\ell,m} = \mathbb{1}\bigl[f_{\mathrm{peak},\ell,m} \in \mathcal{B}_{g_\ell(m)}\bigr]
$$

로 저장한다.

### 7.4.4 passband overlap ratio

principal passband 와 assigned band 의 overlap ratio 는

$$
\mathrm{overlap}_{\ell,m} = \frac{\left| [f_{L,\ell,m}, f_{H,\ell,m}] \cap \mathcal{B}_{g_\ell(m)} \right|}{\max\bigl(|\mathcal{B}_{g_\ell(m)}|, \varepsilon\bigr)}
$$

로 정의한다.

### 7.4.5 layer summary 에 추가할 값

layer `summary.json` 은 공통 키에 더해 아래를 포함해야 한다.

- `intrinsic_f_mean`, `intrinsic_f_std`, `intrinsic_f_median`
- `peak_shift_mean`, `peak_shift_abs_mean`, `peak_shift_std`
- `peak_inside_ratio` if band-assigned variant
- `overlap_mean`, `overlap_std` if band-assigned variant
- `bandwise_counts`
- `bandwise_f_peak_mean`
- `bandwise_bw_3db_mean`
- `bandwise_peak_inside_ratio` if band-assigned variant
- `variant_name`
- `mask_density` if masked variant

## 7.5 tracked neuron 선택 규약

tracked neuron 은 랜덤 샘플이 아니라 band group representative 로 선택한다.

- 각 hidden layer 각 band 에서 최대 `tracked_per_band` 개 선택
- 기본 우선순위는 첫 번째, 중앙, 마지막 인덱스
- 선택 결과는 `config.json` 의 `tracked_neurons` 에 band id 와 함께 기록

이 규칙은 band-assigned 실험에서 representative response 를 보기 쉽게 하기 위한 것이다.

---

## 8. 저장 산출물 규격

## 8.1 run 루트

run 루트는 아래 구조를 따른다.

```text
<result_root_abs>/<run_name>/
  config.json
  params.json
  epoch0001/
  epoch0002/
  ...
  epoch_trend/
  aggregate/              # optional, multi-seed 비교 후 생성
```

`config.json` 은 최소한 아래 키를 포함한다.

- `study_name`
- `dataset_name`
- `model_name`
- `variant_name`
- `model_spec_doc`
- `filter_property_doc`
- `study_spec_doc`
- `epochs`
- `hidden`
- `optimizer_hyperparameters`
- `reset_mode`
- `threshold`
- `w_clip_edges`
- `band_neuron_ends`
- `band_group_count`
- `tracked_per_band`
- `tracked_neurons`
- `filter_F`
- `userbin_edges`
- `mask_applied_layers`

기본적으로 `userbin_edges` 는 `w_clip_edges` 와 동일하게 둔다. 사용자가 별도 userbin 을 주지 않으면 자동으로 그렇게 설정한다.

## 8.2 epoch 단위 accuracy 산출물

각 epoch 폴더는 아래 파일을 저장한다.

```text
epochXXXX/
  accuracy/
    train_test_accuracy.png
    train_test_accuracy.json
```

`train_test_accuracy.json` 은 최소한 아래 키를 포함한다.

- `epoch`
- `train_accuracy`
- `test_accuracy`
- `variant_name`
- `seed`

## 8.3 epoch 단위 filter_property 산출물

각 hidden layer 는 공통 규격을 따르되, vanilla RF 전용 파일을 추가한다.

```text
epochXXXX/
  filter_property/
    <layer_name>/
      summary.json
      hist_f_peak.png
      hist_bw_3db.png
      hist_r0.png
      hist_rpi.png
      hist_intrinsic_f.png
      hist_peak_shift.png
      scatter_intrinsic_f_vs_f_peak.png
      band_assignment_summary.json      # band-assigned variant only
      tracked_neurons/
        neuron_<n>/
          branch_response_exact.png
          branch_response_userbin.png
          total_response_exact.png
          total_response_norm_exact.png
          total_response_userbin.png
          meta.json
```

vanilla RF 는 branch 가 1개뿐이므로 `branch_response_exact.png` 와 `branch_response_userbin.png` 는 single-line intrinsic response 를 저장한다. 즉, 공통 툴 호환성을 위해 파일명은 유지하되 의미상 total response 와 동일한 단일 branch 응답이다.

`meta.json` 은 공통 키에 더해 아래 값을 포함한다.

- `intrinsic_f`
- `omega`
- `b`
- `rho`
- `phi`
- `peak_shift`
- `assigned_band_id` if band-assigned variant
- `assigned_band_low` if band-assigned variant
- `assigned_band_high` if band-assigned variant
- `peak_inside_assigned_band` if band-assigned variant
- `overlap_ratio` if band-assigned variant

## 8.4 epoch trend 산출물

epoch 누적 추이는 아래를 최소 저장 단위로 한다.

```text
epoch_trend/
  accuracy/
    train_accuracy.png
    test_accuracy.png
  filter_property/
    <layer_name>_ratio.png
    <layer_name>_f_peak_mean.png
    <layer_name>_bw_3db_mean.png
    <layer_name>_r0_mean.png
    <layer_name>_rpi_mean.png
    <layer_name>_intrinsic_f_mean.png
    <layer_name>_peak_shift_abs_mean.png
    <layer_name>_peak_inside_ratio.png        # band-assigned variant only
    <layer_name>_overlap_mean.png             # band-assigned variant only
```

masked variant 는 구조 검증용으로 아래를 추가 저장하는 것을 권장한다.

```text
epoch_trend/
  structure/
    <src>_to_<dst>_offband_weight_l1.png
```

여기서 off-band leakage 는

$$
\epsilon_{\mathrm{off}}^{(\ell)} = \left\| W^{(\ell)} \odot \bigl(1 - M^{(\ell)}\bigr) \right\|_1
$$

로 정의한다. 규범 구현에서는 모든 epoch 에 대해 0 이어야 한다.

## 8.5 multi-seed aggregate 산출물

5 seed 실행이 끝난 뒤 study 루트 아래 `aggregate/` 를 생성하고 아래를 저장하는 것을 권장한다.

```text
aggregate/
  seed_summary.csv
  train_accuracy_mean_std.png
  test_accuracy_mean_std.png
  final_test_accuracy_bar.png
  peak_inside_ratio_bar.png
  bw_3db_mean_bar.png
```

`seed_summary.csv` 의 최소 컬럼은 아래와 같다.

- `dataset`
- `variant`
- `seed`
- `final_train_accuracy`
- `final_test_accuracy`
- `best_test_accuracy`
- `layer`
- `f_peak_mean_final`
- `bw_3db_mean_final`
- `intrinsic_f_mean_final`
- `peak_shift_abs_mean_final`
- `peak_inside_ratio_final` if band-assigned variant

---

## 9. 비교와 최종 보고 규약

최종 보고에는 아래 비교 항목을 반드시 포함한다.

1. 네 실험군의 train accuracy epoch curve
2. 네 실험군의 test accuracy epoch curve
3. hidden layer 별 `f_peak` histogram
4. hidden layer 별 `bw_3db` histogram
5. hidden layer 별 `intrinsic_f` 대 `f_peak` scatter
6. band-assigned variant 의 `peak_inside_ratio` 와 `overlap_mean`
7. `lp`, `bp`, `hp`, `mixed` 비율 추이
8. 파라미터 수와 활성 파라미터 수

최종 표는 아래 컬럼을 포함하는 것을 권장한다.

- `variant`
- `params_total`
- `params_trainable`
- `params_active`
- `final_test_accuracy_mean±std`
- `hidden1_f_peak_mean`
- `hidden1_bw_3db_mean`
- `hidden1_peak_inside_ratio` if band-assigned variant
- `hidden2_f_peak_mean`
- `hidden2_bw_3db_mean`
- `hidden2_peak_inside_ratio` if band-assigned variant

이 study 에서는 accuracy 하나만으로 결론내리지 않는다. 반드시 accuracy 와 band statistics 를 함께 해석해야 한다.

---

## 10. 구현 검증 체크리스트

## 10.1 공통 검증

- zero input 에서 모든 hidden state 가 발산하지 않아야 한다.
- `no-reset` 설정이 실제로 적용되어야 한다.
- exact ZOH 이산화가 사용되어야 한다.
- hidden-to-hidden dense weight shape 가 hidden width 와 일치해야 한다.

## 10.2 ordered band clipping 검증

실험군 2와 4에서는 모든 hidden neuron 에 대해 아래가 성립해야 한다.

$$
f^{(\omega)}_{\ell,m} \in \mathcal{B}_{g_\ell(m)}
$$

이를 epoch 마다 검사하고 위반 개수가 0 이어야 한다.

## 10.3 same-group dense mask 검증

실험군 3과 4에서는 모든 hidden-to-hidden layer 에 대해 아래가 성립해야 한다.

$$
W_{\mathrm{eff}}^{(\ell)} = W^{(\ell)} \odot M^{(\ell)}
$$

$$
W^{(\ell)} \odot \bigl(1 - M^{(\ell)}\bigr) = 0
$$

즉, off-band weight leakage 는 수치 오차 범위까지 0 이어야 한다.

## 10.4 unsupported 설정

아래 설정은 초기 구현에서 금지한다.

- hidden layer 개수가 1개인데 실험군 3 또는 4를 요청하는 경우
- `band_neuron_ends` 길이와 hidden layer 개수가 맞지 않는 경우
- `w_clip_edges` 가 strictly increasing 하지 않은 경우
- `w_clip_edges` 범위가 $[0, 0.5]$ 를 벗어나는 경우

이 경우 runner 는 조용히 보정하지 말고 즉시 에러를 내야 한다.

---

## 11. 권장 기본값

초기 motivation study 의 권장값은 아래와 같다.

- hidden layer 수: 2
- dataset: `SHD`
- `hidden = 256 256`, `epochs = 80`
- `w_clip_edges = 0.0 0.2 0.35 0.5`
- `band_neuron_ends = 10,20 10,20` 또는 layer width 에 맞는 동일 비율 배치
- `tracked_per_band = 3`
- `filter_F = 512`
- `reset_mode = none`
- `threshold = 1.0`
- `gradient_clip = 1.0`
- seed: `0 1 2 3 4`

`w_clip_edges` 는 첫 번째 motivation study 에서 userbin edge 와 동일하게 사용하는 것을 권장한다. 그러면 clip band 와 response summary band 가 같은 축에서 해석된다.

---

## 12. 최종 해석 가이드

본 study 의 핵심 해석은 아래 순서로 진행한다.

1. 실험군 1 대비 실험군 2의 accuracy 손실이 작은지 본다.
2. 실험군 2에서 `peak_inside_ratio` 와 `overlap_mean` 이 높게 유지되는지 본다.
3. 실험군 3에서 clipping 없이도 cross-band 연결 제거만으로 band purity 가 얼마나 증가하는지 본다.
4. 실험군 4가 실험군 2와 3 대비 추가 이득 또는 추가 손실을 만드는지 본다.
5. `intrinsic_f` 와 `f_peak` 의 shift 가 큰 layer 가 어디인지 본다.

즉, 이 실험은 단순히 성능 비교가 아니라, vanilla RF 의 공진주파수 해석축을 인위적으로 구조화했을 때 그 구조가 실제 필터 통계와 성능 양쪽에서 얼마나 유지되는지를 보는 4-way ablation study 다.
