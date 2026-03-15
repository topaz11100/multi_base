# 직렬 입력 XOR 벤치마크 명세서

파일명은 기존 호환성을 위해 `dh_snn_serial_xor_experiment_spec.md` 를 유지한다. 다만 본 문서의 실험 대상은 더 이상 DH-SNN 한 종류로 고정되지 않으며, `src/neurons/` 아래의 **임의 뉴런 모델** 을 공통 프로토콜로 비교하는 직렬 입력 XOR 벤치마크 명세를 정의한다.

본 벤치마크는 DH-SNN 논문의 delayed spiking XOR 와 multi-timescale spiking XOR 의 핵심 과제 구조를 유지하되, 입력은 S-MNIST 스타일의 **채널 1 직렬 데이터 스트림** 으로 통일하고, 출력 readout 은 **마지막 타임스텝의 출력층 막전위에 softmax를 적용하는 비스파이킹 2-way 분류기** 로 고정한다.

즉, 이 문서에서 말하는 **입력층(input layer)** 은 별도의 뉴런층이 아니라, 매 timestep 마다 모델로 들어가는 **데이터 스트림 자체** 를 뜻한다.

---

# 1. 데이터셋 생성 명세서

## 1.1 설계 원칙

1. 데이터셋은 학습 시점에 on-the-fly 로 생성하지 않는다.
2. 데이터셋은 사전에 오프라인 생성 후 고정한다.
3. train, validation, test split 은 생성 시점에 고정하고 이후 변경하지 않는다.
4. 모든 샘플은 채널 1 입력 스트림으로 저장한다.
5. 모든 입력은 이진 스파이크 스트림으로 저장한다.
6. 모든 메타데이터는 별도 manifest 파일에 기록한다.

## 1.2 공통 기호 및 기본 상수

공통 상수는 다음과 같이 정의한다.

- low-rate firing probability: $p_{low} = 0.2$
- high-rate firing probability: $p_{high} = 0.6$
- background noise probability: $p_{noise} = 0.01$
- 직렬화된 한 개 symbol window 길이: $L_s = 200$
- multi-timescale 에서 두 Signal 2 사이 gap 길이: $L_g = 100$
- multi-timescale 에서 query 횟수: $K = 6$
- delayed 기본 delay 길이: $L_d = 3600$

여기서 $L_s = 200$ 은 채널 1 직렬입력에서 low/high rate symbol 이 통계적으로 충분히 분리되도록 하기 위한 설계값이다. 즉 각 symbol 은 길이 200 의 이진 시퀀스로 표현되고, low/high class 는 발화확률만 다르다.

## 1.3 공통 샘플 생성 규칙

각 symbol class $c \in \{0,1\}$ 에 대해 대응 firing probability 를

$$
p(c) =
\begin{cases}
p_{low}, & c = 0 \\
p_{high}, & c = 1
\end{cases}
$$

로 둔다.

길이 $L$ 인 symbol spike window 는 다음과 같이 생성한다.

$$
z_t \sim \mathrm{Bernoulli}(p(c)), \quad t=1,\dots,L
$$

동일 구간의 background noise 는

$$
n_t \sim \mathrm{Bernoulli}(p_{noise}), \quad t=1,\dots,L
$$

로 생성하고, 최종 입력은 논리합으로 합성한다.

$$
x_t = \max(z_t, n_t)
$$

noise-only gap 는

$$
g_t \sim \mathrm{Bernoulli}(p_{noise})
$$

로 생성한다.

## 1.4 저장 포맷

모든 데이터셋은 프로젝트 내부 `datasets/` 가 아니라, 사용자 지정 절대경로 `data_root_abs` 아래에 저장한다.

```text
<data_root_abs>/
  delayed_xor_serial/
    meta.json
    train.npz
    val.npz
    test.npz
  multiscale_xor_serial/
    meta.json
    train.npz
    val.npz
    test.npz
```

각 파일의 필수 필드는 다음과 같다.

### delayed_xor_serial/*.npz

- `x`: shape `[N, T_d, 1]`, dtype `uint8`
- `y`: shape `[N]`, dtype `uint8`
- `eval_idx`: shape `[N]`, dtype `int64`
- `signal1_class`: shape `[N]`, dtype `uint8`
- `signal2_class`: shape `[N]`, dtype `uint8`
- `delay_len`: shape `[N]`, dtype `int64`

### multiscale_xor_serial/*.npz

- `x`: shape `[N, T_m, 1]`, dtype `uint8`
- `y_seq`: shape `[N, K]`, dtype `uint8`
- `query_eval_idx`: shape `[N, K]`, dtype `int64`
- `signal1_class`: shape `[N]`, dtype `uint8`
- `signal2_class_seq`: shape `[N, K]`, dtype `uint8`
- `num_queries`: shape `[N]`, dtype `int64`

### meta.json

다음 항목을 반드시 포함한다.

- 데이터셋 이름
- 생성 날짜
- random seed
- split 크기
- $p_{low}$, $p_{high}$, $p_{noise}$
- $L_s$, $L_g$, $L_d$, $K$
- 인덱스 기준이 0-index 인지 1-index 인지
- 파일 checksum
- 생성 스크립트 버전

## 1.5 delayed XOR 직렬입력 데이터셋

### 1.5.1 목적

이 데이터셋은 긴 delay 와 noise 를 통과한 뒤에도 첫 번째 symbol 정보를 보존하여 마지막 symbol 과 XOR 결정을 할 수 있는지를 평가한다. 이는 원 논문의 delayed spiking XOR 의 목적과 동일하게 장기기억 평가용이다. 

### 1.5.2 샘플 구조

각 샘플에서 두 개의 class 를 균등확률로 샘플링한다.

$$
a \sim \mathrm{Bernoulli}(0.5), \quad b \sim \mathrm{Bernoulli}(0.5)
$$

첫 번째 symbol $s(a)$ 와 마지막 symbol $s(b)$ 는 길이 $L_s$ 로 생성하고, 중간 delay 구간은 길이 $L_d$ 의 noise-only stream 으로 생성한다.

전체 입력은 다음과 같다.

$$
\mathbf{x} = [s(a), d, s(b)] \in \{0,1\}^{T_d \times 1}
$$

여기서

$$
T_d = 2L_s + L_d
$$

이다.

정답 label 은 XOR 로 정의한다.

$$
y = a \oplus b
$$

예측 시점은 전체 입력이 모두 끝난 마지막 시점이다.

$$
t_{eval}^{delay} = T_d
$$

파일에 저장할 `eval_idx` 는 0-index 기준으로 $T_d - 1$ 을 저장한다.

### 1.5.3 split 구성

권장 split 크기는 다음과 같다.

- train: 50,000
- val: 10,000
- test: 10,000

각 split 에서 $(a,b) \in \{(0,0),(0,1),(1,0),(1,1)\}$ 네 조합이 정확히 같은 비율이 되도록 층화 생성한다.

### 1.5.4 optional delay sweep

장기기억 길이 민감도 분석을 위해 아래의 고정 delay subset 을 추가로 생성할 수 있다.

$$
\mathcal{D} = \{400, 800, 1600, 2400, 3200, 3600\}
$$

이 경우 각 $L_d \in \mathcal{D}$ 마다 별도의 train, val, test 파일을 생성하고, 실험 시 delay 길이별 성능 곡선을 보고한다.

## 1.6 multi-timescale XOR 직렬입력 데이터셋

### 1.6.1 목적

이 데이터셋은 느리게 변하는 시작 symbol 을 기억하면서, 이후 반복적으로 등장하는 빠른 query symbol 들에 대해 매번 XOR 결정을 내릴 수 있는지를 평가한다. 이는 원 논문의 multi-timescale spiking XOR 와 같은 목적이며, Signal 1 은 느린 component, 반복되는 Signal 2 는 빠른 component 역할을 한다. 모델은 Signal 2 가 들어올 때마다 Signal 1 과 현재 Signal 2 의 XOR 를 예측해야 한다. 

### 1.6.2 샘플 구조

각 샘플에서 시작 symbol class $a$ 와 각 query class $b_k$ 를 균등확률로 샘플링한다.

$$
a \sim \mathrm{Bernoulli}(0.5)
$$

$$
b_k \sim \mathrm{Bernoulli}(0.5), \quad k=1,\dots,K
$$

Signal 1 은 길이 $L_s$ 의 한 개 symbol window 로 생성한다. 각 query 는 길이 $L_g$ 의 noise-only gap 뒤에 길이 $L_s$ 의 Signal 2 symbol 이 들어오는 구조로 만든다.

전체 입력은 다음과 같다.

$$
\mathbf{x} = [s(a), g_1, s(b_1), g_2, s(b_2), \dots, g_K, s(b_K)] \in \{0,1\}^{T_m \times 1}
$$

여기서

$$
T_m = L_s + K(L_g + L_s)
$$

이다.

각 query 의 정답은

$$
y_k = a \oplus b_k, \quad k=1,\dots,K
$$

로 정의한다.

각 query 에 대한 예측 시점은 해당 Signal 2 block 이 모두 입력된 마지막 시점이다.

$$
t_k^{multi} = L_s + kL_g + kL_s, \quad k=1,\dots,K
$$

파일에 저장할 `query_eval_idx` 는 0-index 기준으로 $t_k^{multi} - 1$ 을 저장한다.

### 1.6.3 split 구성

권장 split 크기는 다음과 같다.

- train: 50,000
- val: 10,000
- test: 10,000

각 split 에서 다음 조건을 만족하도록 생성한다.

1. `signal1_class` 는 0 과 1 이 정확히 50:50 이다.
2. 각 query 위치 $k$ 에 대해 `signal2_class_seq[:, k]` 는 0 과 1 이 정확히 50:50 이다.
3. 따라서 각 query 위치의 XOR label 도 0 과 1 이 거의 50:50 이 된다.

## 1.7 생성 후 고정 규칙

1. 데이터 생성이 끝나면 train, val, test 파일을 고정한다.
2. 실험 중 추가 샘플 생성, 재샘플링, data regeneration 을 금지한다.
3. 데이터 파일의 checksum 을 기록한다.
4. 모든 실험은 동일한 파일을 공유한다.

---

# 2. 실험 명세서

## 2.1 공통 실험 목적

- delayed XOR 직렬입력 과제에서 장기기억 성능 비교
- multi-timescale XOR 직렬입력 과제에서 느린 정보 기억과 빠른 정보 추적 및 결합 성능 비교
- 임의 뉴런 모델의 내부 동역학과 네트워크 구조가 시간적 feature integration 에 미치는 영향 비교

원 논문은 delayed XOR 를 장기기억, multi-timescale XOR 를 temporally heterogeneous information processing 검증용으로 사용했다. 본 명세서는 그 과제 의미를 유지하되, 모델 계열을 DH-SNN 에 한정하지 않고 `src/neurons/` 아래 임의 뉴런 구조로 일반화한다.

## 2.2 데이터 사용 규칙

1. 실험 시 데이터는 오프라인 생성된 고정 파일에서만 읽는다.
2. minibatch 구성 시에도 샘플 내용은 바뀌지 않는다.
3. train set 에서만 파라미터를 학습한다.
4. validation set 으로 model selection 을 수행한다.
5. test set 은 최종 1회 평가에만 사용한다.
6. 데이터 루트는 사용자 지정 절대경로 `data_root_abs` 로 받는다.
7. 결과 루트는 사용자 지정 절대경로 `result_root_abs` 로 받는다.
8. 프로젝트 상대경로를 기본 데이터 경로나 결과 경로로 사용하지 않는다.

## 2.3 입력 규격

모든 모델은 매 timestep 에 스칼라 입력 한 개만 받는다.

$$
x_t \in \{0,1\}, \quad \mathbf{x} \in \{0,1\}^{T \times 1}
$$

즉 입력 스트림은 S-MNIST 와 같은 채널 1 직렬입력 형식을 따른다. 이때 **입력층은 데이터 스트림 자체** 이며, 별도의 스파이킹/비스파이킹 입력 뉴런층을 뜻하지 않는다. 따라서 네트워크 표기는 항상 다음처럼 해석한다.

```text
input(data stream) -> hidden_1 -> ... -> hidden_L -> output(non-spiking, 2 neurons)
```

## 2.4 모델 입출력 인터페이스

### 2.4.1 hidden layer

hidden layer 는 bash/CLI 로 지정한 임의 뉴런 모델과 hidden 구조를 따른다.

- `--models <model_1 model_2 ...>` : 비교할 뉴런/모델 구조 목록
- `--hidden <h1 h2 ...>` : hidden layer 개수와 각 hidden layer 뉴런 수
  - 예: `--hidden 2 3` 이면 hidden layer 는 2개이고, hidden1 은 2개, hidden2 는 3개 뉴런을 갖는다.
  - 이 표기에서 input layer 는 데이터 스트림 자체이며, `--hidden` 에 포함되지 않는다.
  - `--hidden` 을 비우면 데이터 스트림이 곧바로 출력층으로 연결되는 것으로 해석한다.

### 2.4.2 출력층

출력층은 항상 **비스파이킹 readout 층** 으로 설계한다.

1. threshold 가 없다.
2. spike event 를 발생시키지 않는다.
3. reset 이 없다.
4. 출력층 상태는 실수값 막전위 벡터로 유지한다.
5. 이진 분류이므로 출력층 뉴런 수는 **항상 2** 이다.

출력층 막전위 벡터의 한 예시는 다음과 같다.

$$
\mathbf{u}^{out}_t = \rho \mathbf{u}^{out}_{t-1} + W_{out}\mathbf{h}_t + \mathbf{b}_{out}
$$

여기서 $\mathbf{h}_t$ 는 마지막 hidden layer 의 출력이고, $\rho \in [0,1]$ 는 누적 readout 이 필요할 때 사용하는 실수 leak 계수다. 중요한 점은 출력층이 **스파이크/리셋 없이 막전위만 유지** 한다는 것이다.

예측 확률은 항상 **마지막 타임스텝 출력층 막전위** 에 softmax 를 적용하여 계산한다.

$$
\hat{\mathbf{p}} = \mathrm{softmax}(\mathbf{u}^{out}_{T})
$$

## 2.5 delayed XOR 실험 프로토콜

### 2.5.1 추론 규칙

delayed XOR 에서는 전체 입력이 끝난 마지막 시점에서만 예측한다.

$$
\hat{\mathbf{p}}^{delay} = \mathrm{softmax}(\mathbf{u}^{out}_{T_d})
$$

최종 예측은

$$
\hat{y}^{delay} = \arg\max \hat{\mathbf{p}}^{delay}
$$

로 정의한다.

### 2.5.2 손실 함수

샘플 $i$ 의 손실은 마지막 시점에서의 cross-entropy 하나만 사용한다.

$$
\ell_i^{delay} = \mathrm{CE}(\hat{\mathbf{p}}^{delay}_i, y_i)
$$

미니배치 손실은

$$
\mathcal{L}^{delay} = \frac{1}{B}\sum_{i=1}^{B} \ell_i^{delay}
$$

로 정의한다.

### 2.5.3 평가 지표

- 최종시점 accuracy
- optional delay sweep 시 delay 길이별 accuracy curve
- 5 seed 평균과 표준편차

## 2.6 multi-timescale XOR 실험 프로토콜

### 2.6.1 추론 규칙

multi-timescale XOR 에서는 각 Signal 2 block 이 모두 입력될 때마다 한 번씩 예측한다.

$$
\hat{\mathbf{p}}_{i,k}^{multi} = \mathrm{softmax}(\mathbf{u}^{out}_{i,t_k^{multi}})
$$

각 query 의 예측 label 은

$$
\hat{y}_{i,k}^{multi} = \arg\max \hat{\mathbf{p}}_{i,k}^{multi}
$$

로 정의한다.

### 2.6.2 손실 함수

샘플 $i$ 의 손실은 모든 query 시점의 cross-entropy 를 누적한 값으로 정의한다.

$$
\ell_i^{multi} = \sum_{k=1}^{K} \mathrm{CE}(\hat{\mathbf{p}}_{i,k}^{multi}, y_{i,k})
$$

미니배치 손실은

$$
\mathcal{L}^{multi} = \frac{1}{B}\sum_{i=1}^{B} \ell_i^{multi}
$$

로 정의한다.

필요하면 보고용 보조지표로 query 수로 정규화한 평균 손실

$$
\bar{\mathcal{L}}^{multi} = \frac{1}{BK}\sum_{i=1}^{B}\sum_{k=1}^{K} \mathrm{CE}(\hat{\mathbf{p}}_{i,k}^{multi}, y_{i,k})
$$

를 함께 기록할 수 있다.

### 2.6.3 평가 지표

- query-wise accuracy

$$
\mathrm{Acc}_{query} = \frac{1}{NK}\sum_{i=1}^{N}\sum_{k=1}^{K} \mathbf{1}[\hat{y}_{i,k}^{multi} = y_{i,k}]
$$

- sequence exact-match accuracy

$$
\mathrm{Acc}_{seq} = \frac{1}{N}\sum_{i=1}^{N} \mathbf{1}[\forall k, \hat{y}_{i,k}^{multi} = y_{i,k}]
$$

- query 위치별 accuracy

$$
\mathrm{Acc}_{k} = \frac{1}{N}\sum_{i=1}^{N} \mathbf{1}[\hat{y}_{i,k}^{multi} = y_{i,k}]
$$

- 5 seed 평균과 표준편차

## 2.7 비교 모델 세트

비교 대상 모델은 문서에 하드코딩하지 않고 bash/CLI 인수 `--models` 로 받는다. 즉, 한 번의 benchmark run 은 다음을 비교 단위로 갖는다.

- 모델 축: `--models` 로 지정한 임의 뉴런 구조
- hidden 구조 축: `--hidden` 으로 지정한 hidden layer 개수와 각 층 뉴런 수
- 구조 하이퍼파라미터 축: 모델별로 필요한 `--S_min`, `--S_max`, `--th_len`, `--v_th`, `--v_pre` 등

다만 모든 모델은 동일한 데이터 split, 동일한 출력층 규칙(2-neuron non-spiking readout), 동일한 평가 프로토콜을 공유해야 한다.

## 2.8 공정 비교 규칙

1. 모든 비교 모델은 동일한 데이터 split 을 사용한다.
2. 동일한 optimizer 와 training budget 을 사용한다.
3. hidden 구조 비교 시 `--hidden` 으로 지정한 hidden layer 개수와 각 hidden layer 뉴런 수를 동일하게 맞춘다.
4. branch 수 또는 내부 구조 차이로 파라미터 수가 크게 달라질 수 있으므로, 최종 비교 표에는 실제 활성 파라미터 수를 함께 보고한다.
5. input layer 는 데이터 스트림 자체이므로, 공정 비교 시 hidden/output 계층만 모델 파라미터 비교 대상으로 본다.

## 2.9 학습 및 bash 전달 하이퍼파라미터

### 2.9.1 공통 학습 설정

권장 기본 학습 설정은 다음과 같다.

- optimizer: Adam 또는 AdamW
- initial learning rate: $10^{-3}$
- batch size: 128 또는 그 이상
- max epochs: 200 이내
- gradient clipping: global norm 1.0
- scheduler: validation plateau 기반 decay 또는 cosine decay
- early stopping patience: 20 epochs 내외
- 반복 실행 수: 5 seeds

최종 test 결과는 각 seed 에 대해 validation best checkpoint 또는 명세에서 정한 마지막 checkpoint 로 평가하고, 평균과 표준편차를 함께 보고한다.

### 2.9.2 bash/CLI 인수 설명

다음 인수들은 bash wrapper 가 python run script 로 전달해야 하는 핵심 하이퍼파라미터다.

#### 경로/실행

- `--out_root` : 결과 루트 경로
- `--exp_name` : run 이름 prefix
- `--timestamp` : 실행 시각 태그
- `--gpu` : 사용할 GPU 인덱스
- `--seed` : 난수 시드
- `--models` : 비교할 모델 목록
- `--hidden` : hidden layer 개수와 각 층 뉴런 수

#### 학습 예산 및 2단계 스케줄

- `--epochs` : 총 학습 epoch 수
- `--soft_mask_epochs` : Stage A(soft mask) 길이
- `--stabilize_epochs` : Stage B(stabilize) 길이
- `--ste_epochs` : Stage A 마지막 STE epoch 수
- `--steps_per_epoch` : epoch 당 학습 step 수
- `--batch_size` : batch size
- `--lr` : learning rate
- `--weight_decay` : layer connection weight 에 적용하는 AdamW weight decay
- `--weight_decay_dend_soma` : `my_R_DH_SNN` 의 `W_mix` 전용 weight decay

#### 평가/보고 cadence

- `--check_every` : 단위 에포크(unit epoch)로서 train/test accuracy 를 몇 epoch 마다 계산할지 지정한다.
  - 여기서 말하는 단위 에포크는 "매 epoch" 가 아니라 **bash CLI 인수로 받은 평가 간격** 이다.
  - 예를 들어 `--check_every 10` 이면 10, 20, 30, ... epoch 에서만 train/test accuracy 를 측정한다.
  - accuracy 플롯은 이 측정값들을 누적해 **학습 종료 후 한 번만** 저장한다.
- `--eval_batches` : 각 평가 시 사용할 배치 수

#### 모델 구조/뉴런 하이퍼파라미터

- `--S_min`, `--S_max` : 가변 branch 구조 범위
- `--th_len` : adaptive threshold kernel 길이
- `--v_th` : threshold 관련 초기값
- `--v_reset` : 사용하지 않는다. LIF 기준 모델은 막전위 감쇠계수를 학습하고 soft reset 으로 고정한다. 별도 LIF 변형 명세는 사용하지 않는다.
- `--v_pre` : pre-threshold 또는 readout 관련 기준값
- `--lambda_ortho` : orthogonality regularization 계수
- `--lambda_s` : 구조 변수 $s$ regularization 계수

#### delayed XOR 전용

- `--time_steps` : 전체 시퀀스 길이 또는 delay benchmark time steps
- `--channel_size` : 입력 직렬화에 사용하는 channel size 설정
- `--coding_time` : symbol coding 길이
- `--test_time` : delayed XOR 의 final query 길이
- `--noise_rate` : background noise rate
- `--rate_low`, `--rate_high` : low/high firing probability

#### multi-timescale XOR 전용

- `--time_steps` : 전체 시퀀스 길이
- `--channel_size` : 입력 직렬화에 사용하는 channel size 설정
- `--coding_time` : symbol coding 길이
- `--remain_time` : query 간 유지 구간 길이
- `--start_time` : 초기 Signal 1 이후 첫 query 시작 위치
- `--noise_rate` : background noise rate
- `--rate_low`, `--rate_high` : low/high firing probability

## 2.10 보고 항목과 저장 규칙

각 과제에서 최소한 아래 항목을 보고한다.

### 공통 산출물

- `hyperparameters.txt` 또는 동등한 텍스트 문서: bash/CLI 로 전달한 모든 하이퍼파라미터 값
- `active_params.txt` 또는 동등한 텍스트 문서: 모델 구조와 활성 파라미터 수 요약
- `train_test_accuracy.png` : train/test accuracy 단일 플롯
  - x축은 `--check_every` 로 측정한 epoch index
  - y축은 accuracy
  - 하나의 그림에 train/test 두 곡선을 함께 그린다.
  - **플롯 저장은 학습 완료 후 1회만** 수행한다.
  - per-epoch 또는 per-unit-epoch PNG 를 따로 저장하지 않는다.

### delayed XOR

- test accuracy
- optional delay sweep accuracy curve
- 파라미터 수
- 학습 안정성 지표인 5 seed 표준편차

### multi-timescale XOR

- query-wise accuracy
- sequence exact-match accuracy
- query 위치별 accuracy
- 파라미터 수
- 학습 안정성 지표인 5 seed 표준편차

## 2.11 금지 사항

1. 실험 도중 새 샘플 생성 금지
2. test set 기반 hyperparameter tuning 금지
3. 출력층 spike 발생 금지
4. 출력층 reset 사용 금지
5. delayed XOR 에서 중간 시점 예측을 학습 손실에 포함하는 것 금지
6. multi-timescale XOR 에서 query 종료 시점 이외의 timestep 에 대한 손실 추가 금지

---

# 3. 최종 요약

이 명세서는 원 논문의 delayed XOR 와 multi-timescale XOR 의 핵심 의미를 유지하면서, 다음 사항을 명시적으로 고정한다.

1. 입력층은 별도 뉴런층이 아니라 **채널 1 직렬 데이터 스트림 자체** 다.
2. 실험 대상 모델은 `src/neurons/` 아래 임의 뉴런 구조이며, hidden layer 개수와 각 층 뉴런 수는 `--hidden` 으로 받는다.
3. 출력층은 **항상 2개의 비스파이킹 뉴런** 으로 구성되며, 마지막 타임스텝 막전위에 softmax 를 적용해 이진 분류를 수행한다.
4. 단위 에포크는 `--check_every` 로 주는 **평가 간격** 을 뜻하며, 매 epoch 를 뜻하지 않는다.
5. train/test accuracy 플롯은 단위 에포크마다 새로 저장하지 않고, 누적 측정값을 사용해 **학습 종료 후 단일 PNG 한 장** 만 저장한다.

이렇게 하면 데이터셋 자체는 장기기억 및 다중시간스케일 처리 능력을 계속 검증하면서도, 실험 프로토콜은 임의 뉴런 모델 비교용 직렬 입력 SNN 벤치마크로 직접 사용할 수 있다.
