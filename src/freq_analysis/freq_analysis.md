# freq_analysis 실험 명세서

본 문서는 `freq_analysis` 실험에서 무엇을 측정하고, 어느 경로에 어떤 이름으로 저장하는지를 정의한다. 구체적인 학습 방법, optimizer 상세, 뉴런 동역학, 모델 정의는 각 모델 문서와 관련 문서로 이관한다.

실험 대상은 특정 제안 뉴런에 한정하지 않는다. `config.json` 에 `signal_manifest`, `timing_factor_keys`, `weight_group_keys` 를 제공하는 임의 뉴런 모델에 대해 동일한 저장 규칙을 적용한다. 필터적 성질 분석이 가능한 모델은 `paper/proposed/filter_analysis.md` 의 공통 규격을 따른다.

---

## 0. 범위

- 데이터셋: s-MNIST, s-CIFAR10, SHD, SSC
- 모델: `src/neurons/` 아래 임의 뉴런 모델
- 공통 전제: 입력층은 데이터 스트림 자체이고, hidden 구조는 `--hidden` 으로 지정한다.
- 출력 readout 은 runner 가 정의한 막전위 기반 readout 을 사용하며, 별도 `--readout` 인수는 두지 않는다.

---

## 1. 공통 실행 규칙

### 1.1 입력층 의미

이 문서에서 말하는 **입력층(input layer)** 은 별도 뉴런층이 아니라 **데이터 스트림 자체** 를 뜻한다. 따라서

```text
input -> hidden1 -> hidden2 -> ...
```

에서 `input` 은 데이터셋이 제공한 raw 시계열이며, `--hidden` 에 포함되지 않는다.

### 1.2 경로 규칙

- 데이터는 프로젝트 내부 상대경로가 아니라 사용자 지정 절대경로 `data_root_abs` 에서 읽는다.
- 결과는 프로젝트 내부 상대경로가 아니라 사용자 지정 절대경로 `result_root_abs` 아래에 저장한다.
- `data/`, `datasets/`, `result/` 같은 프로젝트 상대경로를 기본 경로로 가정하지 않는다.

### 1.3 bash/CLI 인수 설명

다음 인수들은 bash wrapper 가 python run script 로 넘겨주는 핵심 하이퍼파라미터다.

#### 공통 실행

- `--out_root` : 결과 저장 루트
- `--data_root` : 데이터 저장 루트
- `--exp_name` : 실험 이름
- `--timestamp` : 실행 시각 태그
- `--gpu` : 사용할 GPU 인덱스
- `--seed` : 랜덤 시드
- `--models <m1 m2 ...>` : 실행할 모델 목록
- `--hidden <h1 h2 ...>` : hidden layer 개수와 각 hidden layer 뉴런 수
  - 예: `--hidden 32 32 32` 이면 hidden layer 는 3개이며 각 층 뉴런 수는 32, 32, 32 다.
  - input layer 는 데이터 스트림 자체이므로 `--hidden` 에 포함되지 않는다.
- `--epochs` : 총 학습 epoch 수
- `--batch_size` : batch size
- `--lr` : learning rate
- `--num_workers` : DataLoader worker 수
- `--download` : 데이터 다운로드 허용 여부

#### 2단계 구조 학습

- `--soft_mask_epochs` : Stage A 길이
- `--stabilize_epochs` : Stage B 길이
- `--ste_epochs` : Stage A 마지막 STE epoch 수
- `--weight_decay` : layer connection weight 에 적용할 AdamW weight decay
- `--weight_decay_dend_soma` : `my_R_DH_SNN` 의 `W_mix` 전용 weight decay
- `--S_min`, `--S_max` : 가변 branch 구조 범위
- `--lambda_ortho` : orthogonality regularization 계수
- `--lambda_s` : 구조 변수 $s$ regularization 계수

#### 뉴런/임계 관련

- `--th_len` : adaptive threshold kernel 길이
- `--v_th` : threshold 초기값
- `--v_pre` : pre-threshold 기준값

#### 분석 cadence

- `--plot_every` : run-level accuracy 샘플링, timing/structure/weight distribution, filter-property snapshot 을 몇 epoch 마다 계산/저장할지 정하는 **단위 에포크** - `--analysis_every` : probe/label_set 입력, tracked neuron 신호, layer heatmap, layer-to-layer change 를 몇 epoch 마다 계산/저장할지 정하는 **단위 에포크** - `--convergence_every` : convergence 계열 보조 분석이 있을 경우 그 계산/저장 간격을 정하는 **단위 에포크**
- `--analysis_neurons <n1 n2 ...>` : hidden layer 별 tracked neuron 샘플 수
- `--fft_band_edges <e0 e1 ... eB>` : userbin 주파수 경계
- `--fft_band_reduce {mean,sum,l2,max}` : band 시각화 축약 방식

#### 데이터셋별 추가 인수

- s-CIFAR10: `--cifar_mode`
- SHD, SSC: `--T_event`

### 1.4 단위 에포크와 저장 규칙

본 문서에서 **단위 에포크(unit epoch)** 는 "매 epoch" 를 뜻하지 않는다. 각 산출물은 bash/CLI 인수로 지정한 cadence 를 만족하는 epoch 에서만 계산/저장한다.

- `plot_every = p > 0` 이면 `p, 2p, 3p, ...` epoch 에서만 해당 계열 산출물을 저장한다.
- `analysis_every = a > 0` 이면 `a, 2a, 3a, ...` epoch 에서만 해당 계열 산출물을 저장한다.
- `convergence_every = c > 0` 이면 convergence 계열 보조 분석도 `c, 2c, 3c, ...` epoch 에서만 계산한다.
- 어떤 산출물도 "모든 epoch" 에서 자동 저장하지 않는다.

선택된 epoch 의 폴더 이름은 항상 `epochXXXX/` 형식을 따른다. 예를 들어 `analysis_every = 10` 이면 `epoch0010/`, `epoch0020/`, ... 만 생성한다.

### 1.5 저장하지 않는 산출물

아래 항목은 본 명세의 필수 산출물이 아니다.

- checkpoint `.pt`
- loss 관련 플롯
- accuracy PNG 의 per-epoch 또는 per-unit-epoch 중복 저장

단, run-level accuracy 곡선 자체는 최종 단일 플롯으로 저장한다.

---

## 2. 주파수 표현 공통 규칙

### 2.1 exact 주파수 표현

모든 exact 주파수 계산의 기본 정의는 아래와 같다.

$$
X_{\mathrm{exact}}(x) = \mathrm{rFFT}(x)
$$

즉, exact 표현의 기본 객체는 **로그를 씌우지 않은 exact rFFT 결과** 다.

시각화와 band 집계를 위해 아래 파생량을 함께 정의한다.

$$
A_{\mathrm{exact},k}(x) = \left|X_{\mathrm{exact},k}(x)\right|
$$

$$
L_{\mathrm{exact},k}(x) = \log\left(1 + A_{\mathrm{exact},k}(x)\right)
$$

- exact 1D FFT line plot 은 기본적으로 $A_{\mathrm{exact}}$ 를 사용한다.
- **heatmap 시각화는 반드시 두 장을 함께 저장** 한다.
  - raw exact magnitude heatmap: $A_{\mathrm{exact}}$
  - log-compressed exact heatmap: $L_{\mathrm{exact}}$

### 2.2 userbin 주파수 표현

사용자 지정 bin 집합 $\mathcal{B}_b$ 에 대해 userbin magnitude 는 아래처럼 정의한다.

$$
A_{\mathrm{userbin},b}(x)
=
\frac{1}{|\mathcal{B}_b|}\sum_{k\in\mathcal{B}_b} A_{\mathrm{exact},k}(x)
$$

따라서 `config.json` 은 userbin 분석을 사용할 경우 항상 `userbin_edges` 를 포함해야 한다.

---

## 3. 입력집합 및 manifest 규칙

### 3.1 probe

probe 는 test split 에서 라벨별로 1개 샘플을 고정 선택한 집합이다.

- 선택 결과는 run 전체에서 고정하며 epoch 마다 같은 샘플을 사용한다.
- 선택된 샘플 정보는 `config.json` 의 `probe_indices` 에 기록한다.
- `epochXXXX/probe/label_<label>/` 는 항상 해당 라벨의 raw 데이터 스트림 1개를 뜻한다.

### 3.2 label_set

label_set 은 probe 와 다른 샘플로 구성한다.

- 같은 test split 에서 probe 에 사용되지 않은 샘플만 후보로 둔다.
- 라벨별로 1개 샘플을 고정 선택한다.
- 각 샘플은 서로 이어 붙이지 않고 독립 입력으로 모델에 넣는다.
- epoch 마다 라벨별 raw 결과를 저장한 뒤, 라벨 축에 대한 elementwise mean 과 median 을 계산해 추가 저장한다.
- 선택된 샘플 정보는 `config.json` 의 `label_set_indices` 에 기록한다.

### 3.3 tracked neurons

뉴런 단위 신호는 전체 뉴런이 아니라 tracked neuron 집합에 대해 저장한다.

- tracked neuron 인덱스는 레이어별로 고정한다.
- hidden layer 별 샘플 수는 `--analysis_neurons` 로 받는다.
- 선택 결과는 `config.json` 의 `tracked_neurons` 에 기록한다.

### 3.4 signal manifest

모델이 임의 구조를 가질 수 있으므로, 본 문서는 신호 의미를 하드코딩하지 않는다. 각 모델은 `config.json` 에 아래 manifest 를 제공해야 한다.

- `signal_manifest.neuron_input_keys`
- `signal_manifest.neuron_state_keys`
- `signal_manifest.neuron_spike_keys`
- `signal_manifest.layer_input_keys`
- `signal_manifest.layer_state_keys`
- `signal_manifest.layer_spike_keys`

각 `signal_key` 는 한 개의 의미적 신호 tensor 를 가리켜야 하며, 저장 직전에는 아래 규칙을 따른다.

- tracked neuron 단위: 시간축을 제외한 나머지 축은 multi-line 을 위한 subchannel 축으로 해석한다.
- layer 단위: 시간축을 제외한 나머지 축은 heatmap channel 축 하나로 flatten 한다.

### 3.5 tracked neuron state 변수 종류 명시

`state` 계열 저장은 모델이 실제로 갖는 내부 상태변수 자체값을 뜻한다. 각 뉴런 타입별 기본 저장 대상은 아래와 같다.

| 뉴런 타입              | tracked neuron 기본 state 저장값 | 보조 내부상태(선택 저장)      |
| ---------------------- | -------------------------------- | ----------------------------- |
| `LIF_neuron`         | `mem`                          | `spk`                       |
| `TC_LIF_neuron`      | `v1`, `v2`                   | `spk`                       |
| `TS_LIF_neuron`      | `vd`, `vs`                   | `sd_prev`, `ss_prev`      |
| `DH_SNN_neuron`      | `d_state`, `mem`             | `spk`                       |
| `my_DH_SNN_neuron`   | `d_state`, `mem`             | `spk`                       |
| `my_R_DH_SNN_neuron` | `d_state`, `mem`             | `spk`                       |
| `D_RF_neuron`        | `u`                            | `v`, `pre_hist`, `V_th` |
| `my_D_RF_neuron`     | `u`                            | `v`, `p_hist`, `V_th`   |

`LIF_neuron` 은 프로젝트 전역에서 공통으로 쓰는 단일 LIF 명세를 뜻한다. 즉, 막전위 감쇠계수(타이밍 팩터) $\alpha$ 가 학습 파라미터이며 reset 은 soft reset 으로 고정한다. 별도 LIF 변형 명세는 사용하지 않는다.

규칙은 다음과 같다.

1. 표의 "기본 state 저장값" 은 문서상 반드시 state 로 분류해 저장한다.
2. 표의 "보조 내부상태" 는 모델이 `signal_manifest.neuron_state_keys` 또는 별도 extended state key 로 노출한 경우에만 추가 저장한다.
3. 파라미터(`alpha`, `beta`, `tau`, `omega`, `v_th`, `a_k` 등)는 state 가 아니라 distribution/parameter 분석 대상으로 분리한다.

---

## 4. 신호별 저장 규칙

### 4.1 train/test accuracy

#### 저장할 신호 설명

run-level train accuracy 와 test accuracy 의 추이 플롯을 저장한다.

#### 어느 단위로 측정, 저장하는지 설명

- run 단위 신호다.
- accuracy 수치는 `plot_every` 단위 에포크마다 계산한다.
- 계산된 점들을 누적한 뒤 **학습 종료 후 단일 플롯 한 장만 저장** 한다.
- per-epoch 또는 per-unit-epoch accuracy PNG 는 만들지 않는다.

#### 어느 경로에 어떤 이름으로 저장되는지 설명

- run root: `train_test_accuracy.png`

### 4.2 timing factor 분포

#### 저장할 신호 설명

모델이 상태전이계수 또는 timing factor 로 선언한 모든 파라미터 분포를 저장한다.

#### 어느 단위로 측정, 저장하는지 설명

- layer 단위와 model 단위로 모두 저장한다.
- `plot_every` 단위 에포크에서만 저장한다.
- 저장 대상 key 는 `config.json` 의 `timing_factor_keys` 를 따른다.

#### 어느 경로에 어떤 이름으로 저장되는지 설명

- 레이어 단위: `epochXXXX/distribution/timing/layer_<layer_name>_<timing_key>.png`
- 모델 단위: `epochXXXX/distribution/timing/model_<timing_key>.png`

### 4.3 layer weight 분포

#### 저장할 신호 설명

레이어 연결과 모델 내부 mixing 을 포함한 모든 weight group 분포를 저장한다.

#### 어느 단위로 측정, 저장하는지 설명

- layer 단위와 model 단위로 모두 저장한다.
- `plot_every` 단위 에포크에서만 저장한다.
- 저장 대상 key 는 `config.json` 의 `weight_group_keys` 를 따른다.

#### 어느 경로에 어떤 이름으로 저장되는지 설명

- 레이어 단위: `epochXXXX/distribution/weights/layer_<layer_name>_<weight_group>.png`
- 모델 단위: `epochXXXX/distribution/weights/model_<weight_group>.png`

### 4.4 원본 입력 데이터 스트림(입력층)

#### 저장할 신호 설명

입력층은 데이터 스트림 자체이므로, 각 probe/label_set 샘플에 대해 원본 입력의 time heatmap 과 frequency heatmap 을 저장한다.

#### 어느 단위로 측정, 저장하는지 설명

- probe 와 label_set 둘 다 저장한다.
- label_set 은 라벨별 raw 결과와 aggregate mean, aggregate median 을 모두 저장한다.
- `analysis_every` 단위 에포크에서만 저장한다.
- frequency heatmap 은 반드시 아래 3종을 저장한다.
  - exact magnitude heatmap
  - exact log heatmap
  - userbin heatmap

#### 어느 경로에 어떤 이름으로 저장되는지 설명

- probe raw: `epochXXXX/probe/label_<label>/input_time.png`
- probe raw exact magnitude: `epochXXXX/probe/label_<label>/input_fft_exact_abs.png`
- probe raw exact log: `epochXXXX/probe/label_<label>/input_fft_exact_log.png`
- probe raw userbin: `epochXXXX/probe/label_<label>/input_fft_userbin.png`
- label_set raw: `epochXXXX/label_set/label_<label>/input_time.png`
- label_set aggregate mean: `epochXXXX/label_set/aggregate/mean/input_time.png`
- label_set aggregate median: `epochXXXX/label_set/aggregate/median/input_time.png`
- label_set raw/aggregate 의 frequency heatmap 은 같은 경로에서 파일명만 각각 `input_fft_exact_abs.png`, `input_fft_exact_log.png`, `input_fft_userbin.png` 로 바꿔 저장한다.

### 4.5 neuron input 신호

#### 저장할 신호 설명

각 tracked neuron 에 대해 모든 state 변수로 들어가는 입력 신호를 저장한다.

#### 어느 단위로 측정, 저장하는지 설명

- tracked neuron 단위로 저장한다.
- probe 와 label_set 둘 다 저장한다.
- label_set 은 라벨별 raw 결과와 aggregate mean, aggregate median 을 모두 저장한다.
- `analysis_every` 단위 에포크에서만 time, exact FFT line plot, userbin FFT line plot 을 저장한다.

#### 어느 경로에 어떤 이름으로 저장되는지 설명

- probe raw: `epochXXXX/probe/label_<label>/tracked_neurons/<layer_name>/neuron_<id>/input_<signal_key>_time.png`
- probe raw exact FFT: `epochXXXX/probe/label_<label>/tracked_neurons/<layer_name>/neuron_<id>/input_<signal_key>_fft_exact.png`
- probe raw userbin FFT: `epochXXXX/probe/label_<label>/tracked_neurons/<layer_name>/neuron_<id>/input_<signal_key>_fft_userbin.png`
- label_set raw: `epochXXXX/label_set/label_<label>/tracked_neurons/<layer_name>/neuron_<id>/input_<signal_key>_time.png`
- label_set aggregate mean: `epochXXXX/label_set/aggregate/mean/tracked_neurons/<layer_name>/neuron_<id>/input_<signal_key>_time.png`
- label_set aggregate median: `epochXXXX/label_set/aggregate/median/tracked_neurons/<layer_name>/neuron_<id>/input_<signal_key>_time.png`
- aggregate 의 FFT 는 같은 경로에서 파일명만 각각 `_fft_exact.png`, `_fft_userbin.png` 로 바꿔 저장한다.

### 4.6 neuron state 신호

#### 저장할 신호 설명

각 tracked neuron 에 대해 state 변수 자체값을 저장한다. state 변수 종류는 §3.5 표를 따른다.

#### 어느 단위로 측정, 저장하는지 설명

- tracked neuron 단위로 저장한다.
- probe 와 label_set 둘 다 저장한다.
- label_set 은 라벨별 raw 결과와 aggregate mean, aggregate median 을 모두 저장한다.
- `analysis_every` 단위 에포크에서만 time, exact FFT line plot, userbin FFT line plot 을 저장한다.

#### 어느 경로에 어떤 이름으로 저장되는지 설명

- probe raw: `epochXXXX/probe/label_<label>/tracked_neurons/<layer_name>/neuron_<id>/state_<signal_key>_time.png`
- probe raw exact FFT: `epochXXXX/probe/label_<label>/tracked_neurons/<layer_name>/neuron_<id>/state_<signal_key>_fft_exact.png`
- probe raw userbin FFT: `epochXXXX/probe/label_<label>/tracked_neurons/<layer_name>/neuron_<id>/state_<signal_key>_fft_userbin.png`
- label_set raw: `epochXXXX/label_set/label_<label>/tracked_neurons/<layer_name>/neuron_<id>/state_<signal_key>_time.png`
- label_set aggregate mean: `epochXXXX/label_set/aggregate/mean/tracked_neurons/<layer_name>/neuron_<id>/state_<signal_key>_time.png`
- label_set aggregate median: `epochXXXX/label_set/aggregate/median/tracked_neurons/<layer_name>/neuron_<id>/state_<signal_key>_time.png`
- aggregate 의 FFT 는 같은 경로에서 파일명만 각각 `_fft_exact.png`, `_fft_userbin.png` 로 바꿔 저장한다.

### 4.7 neuron spike 신호

#### 저장할 신호 설명

각 tracked neuron 의 spike 출력열을 저장한다. spike 가 없는 모델은 해당 key 를 비워 두고 파일을 만들지 않는다.

#### 어느 단위로 측정, 저장하는지 설명

- tracked neuron 단위로 저장한다.
- probe 와 label_set 둘 다 저장한다.
- label_set 은 라벨별 raw 결과와 aggregate mean, aggregate median 을 모두 저장한다.
- `analysis_every` 단위 에포크에서만 time, exact FFT line plot, userbin FFT line plot 을 저장한다.

#### 어느 경로에 어떤 이름으로 저장되는지 설명

- probe raw: `epochXXXX/probe/label_<label>/tracked_neurons/<layer_name>/neuron_<id>/spike_<signal_key>_time.png`
- probe raw exact FFT: `epochXXXX/probe/label_<label>/tracked_neurons/<layer_name>/neuron_<id>/spike_<signal_key>_fft_exact.png`
- probe raw userbin FFT: `epochXXXX/probe/label_<label>/tracked_neurons/<layer_name>/neuron_<id>/spike_<signal_key>_fft_userbin.png`
- label_set raw: `epochXXXX/label_set/label_<label>/tracked_neurons/<layer_name>/neuron_<id>/spike_<signal_key>_time.png`
- label_set aggregate mean: `epochXXXX/label_set/aggregate/mean/tracked_neurons/<layer_name>/neuron_<id>/spike_<signal_key>_time.png`
- label_set aggregate median: `epochXXXX/label_set/aggregate/median/tracked_neurons/<layer_name>/neuron_<id>/spike_<signal_key>_time.png`
- aggregate 의 FFT 는 같은 경로에서 파일명만 각각 `_fft_exact.png`, `_fft_userbin.png` 로 바꿔 저장한다.

### 4.8 layer input 신호 heatmap

#### 저장할 신호 설명

레이어의 모든 state 변수로 입력되는 값을 layer heatmap 으로 저장한다. branch 나 substate 축이 있는 경우 channel 축으로 flatten 한다.

#### 어느 단위로 측정, 저장하는지 설명

- layer 단위로 저장한다.
- probe 와 label_set 둘 다 저장한다.
- label_set 은 라벨별 raw 결과와 aggregate mean, aggregate median 을 모두 저장한다.
- `analysis_every` 단위 에포크에서만 time heatmap, exact magnitude heatmap, exact log heatmap, userbin heatmap 을 저장한다.

#### 어느 경로에 어떤 이름으로 저장되는지 설명

- probe raw: `epochXXXX/probe/label_<label>/layers/<layer_name>/input_<signal_key>_time.png`
- probe raw exact magnitude: `epochXXXX/probe/label_<label>/layers/<layer_name>/input_<signal_key>_fft_exact_abs.png`
- probe raw exact log: `epochXXXX/probe/label_<label>/layers/<layer_name>/input_<signal_key>_fft_exact_log.png`
- probe raw userbin: `epochXXXX/probe/label_<label>/layers/<layer_name>/input_<signal_key>_fft_userbin.png`
- label_set raw: `epochXXXX/label_set/label_<label>/layers/<layer_name>/input_<signal_key>_time.png`
- label_set aggregate mean: `epochXXXX/label_set/aggregate/mean/layers/<layer_name>/input_<signal_key>_time.png`
- label_set aggregate median: `epochXXXX/label_set/aggregate/median/layers/<layer_name>/input_<signal_key>_time.png`
- raw/aggregate 의 frequency heatmap 은 같은 경로에서 파일명만 각각 `_fft_exact_abs.png`, `_fft_exact_log.png`, `_fft_userbin.png` 로 바꿔 저장한다.

### 4.9 layer state 신호 heatmap

#### 저장할 신호 설명

레이어의 state 변수 자체값을 layer heatmap 으로 저장한다. branch 나 substate 축이 있는 경우 channel 축으로 flatten 한다.

#### 어느 단위로 측정, 저장하는지 설명

- layer 단위로 저장한다.
- probe 와 label_set 둘 다 저장한다.
- label_set 은 라벨별 raw 결과와 aggregate mean, aggregate median 을 모두 저장한다.
- `analysis_every` 단위 에포크에서만 time heatmap, exact magnitude heatmap, exact log heatmap, userbin heatmap 을 저장한다.

#### 어느 경로에 어떤 이름으로 저장되는지 설명

- probe raw: `epochXXXX/probe/label_<label>/layers/<layer_name>/state_<signal_key>_time.png`
- probe raw exact magnitude: `epochXXXX/probe/label_<label>/layers/<layer_name>/state_<signal_key>_fft_exact_abs.png`
- probe raw exact log: `epochXXXX/probe/label_<label>/layers/<layer_name>/state_<signal_key>_fft_exact_log.png`
- probe raw userbin: `epochXXXX/probe/label_<label>/layers/<layer_name>/state_<signal_key>_fft_userbin.png`
- label_set raw: `epochXXXX/label_set/label_<label>/layers/<layer_name>/state_<signal_key>_time.png`
- label_set aggregate mean: `epochXXXX/label_set/aggregate/mean/layers/<layer_name>/state_<signal_key>_time.png`
- label_set aggregate median: `epochXXXX/label_set/aggregate/median/layers/<layer_name>/state_<signal_key>_time.png`
- raw/aggregate 의 frequency heatmap 은 같은 경로에서 파일명만 각각 `_fft_exact_abs.png`, `_fft_exact_log.png`, `_fft_userbin.png` 로 바꿔 저장한다.

### 4.10 layer spike 신호 heatmap

#### 저장할 신호 설명

레이어의 spike 출력열을 layer heatmap 으로 저장한다. 기존 `layer_output` 저장 목적은 이 절로 통합한다.

#### 어느 단위로 측정, 저장하는지 설명

- layer 단위로 저장한다.
- probe 와 label_set 둘 다 저장한다.
- label_set 은 라벨별 raw 결과와 aggregate mean, aggregate median 을 모두 저장한다.
- `analysis_every` 단위 에포크에서만 time heatmap, exact magnitude heatmap, exact log heatmap, userbin heatmap 을 저장한다.

#### 어느 경로에 어떤 이름으로 저장되는지 설명

- probe raw: `epochXXXX/probe/label_<label>/layers/<layer_name>/spike_<signal_key>_time.png`
- probe raw exact magnitude: `epochXXXX/probe/label_<label>/layers/<layer_name>/spike_<signal_key>_fft_exact_abs.png`
- probe raw exact log: `epochXXXX/probe/label_<label>/layers/<layer_name>/spike_<signal_key>_fft_exact_log.png`
- probe raw userbin: `epochXXXX/probe/label_<label>/layers/<layer_name>/spike_<signal_key>_fft_userbin.png`
- label_set raw: `epochXXXX/label_set/label_<label>/layers/<layer_name>/spike_<signal_key>_time.png`
- label_set aggregate mean: `epochXXXX/label_set/aggregate/mean/layers/<layer_name>/spike_<signal_key>_time.png`
- label_set aggregate median: `epochXXXX/label_set/aggregate/median/layers/<layer_name>/spike_<signal_key>_time.png`
- raw/aggregate 의 frequency heatmap 은 같은 경로에서 파일명만 각각 `_fft_exact_abs.png`, `_fft_exact_log.png`, `_fft_userbin.png` 로 바꿔 저장한다.

### 4.11 레이어간 주파수 변화와 canonical delta

#### 분석 대상과 기본 정의

분석 대상은 레이어 spike 출력이다. 입력 스펙트럼을 계산할 수 있으면 `input -> hidden1` pair 를 포함하고, 그 뒤는 인접 레이어 pair 를 순서대로 사용한다. 여기서 `input` 은 데이터 스트림 자체다.

레이어 $\ell$ 의 spike 출력 $O^{(\ell)} \in \{0,1\}^{T \times N_\ell}$ 에 대해, 뉴런별 exact frequency bin power 를 먼저

$$
S_{k,n}^{(\ell)}
=
\left|\mathrm{rFFT}\left(O^{(\ell)}_{:,n}\right)[k]\right|^2
$$

로 정의한다.

그 다음 exact frequency bin 별 power 는 **평균 기준** 과 **중앙값 기준** 을 분리하여 아래처럼 따로 계산하고 저장한다.

$$
P_{k,\mathrm{mean}}^{(\ell)}
=
\frac{1}{N_\ell}\sum_{n=1}^{N_\ell} S_{k,n}^{(\ell)}
$$

$$
P_{k,\mathrm{median}}^{(\ell)}
=
\mathrm{median}_{n\in\{1,\ldots,N_\ell\}}\,S_{k,n}^{(\ell)}
$$

입력 pseudo-layer 0 에 대해서도 같은 방식으로, channel 별 exact rFFT power 를 먼저 계산한 뒤 channel 평균과 channel 중앙값을 각각 사용한다.

userbin power 도 두 기준을 분리하여 exact power 로부터 각각 계산한다.

$$
\bar{P}_{b,\mathrm{mean}}^{(\ell)}
=
\frac{1}{|\mathcal{B}_b|}
\sum_{k\in\mathcal{B}_b} P_{k,\mathrm{mean}}^{(\ell)}
$$

$$
\bar{P}_{b,\mathrm{median}}^{(\ell)}
=
\frac{1}{|\mathcal{B}_b|}
\sum_{k\in\mathcal{B}_b} P_{k,\mathrm{median}}^{(\ell)}
$$

#### exact frequency change 정의

변화량은 reducer $r \in \{\mathrm{mean}, \mathrm{median}\}$ 마다 **서로 독립적으로** 계산한다.

$$
A_{k,r}^{(\ell)} = \left|P_{k,r}^{(\ell)} - P_{k,r}^{(\ell-1)}\right|
$$

$$
D_{k,r}^{(\ell)} = P_{k,r}^{(\ell)} - P_{k,r}^{(\ell-1)}
$$

$$
R_{k,r}^{(\ell)} = \left|\log\frac{P_{k,r}^{(\ell)} + \varepsilon}{P_{k,r}^{(\ell-1)} + \varepsilon}\right|
$$

$$
L_{k,r}^{(\ell)} = \log\frac{P_{k,r}^{(\ell)} + \varepsilon}{P_{k,r}^{(\ell-1)} + \varepsilon}
$$

정규화 composition 변화도 reducer 마다 따로 계산한다.

$$
q_{k,r}^{(\ell)} = \frac{P_{k,r}^{(\ell)}}{\sum_j P_{j,r}^{(\ell)} + \varepsilon}
$$

$$
C_{k,r}^{(\ell)} = \left|q_{k,r}^{(\ell)} - q_{k,r}^{(\ell-1)}\right|
$$

$$
S_{k,r}^{(\ell)} = q_{k,r}^{(\ell)} - q_{k,r}^{(\ell-1)}
$$

각 기호의 의미는 아래와 같다.

- $A$ : raw absolute change. 절대 세기 차이의 크기
- $D$ : raw signed change. 절대 세기의 증가/감소 방향 포함 차이
- $R$ : absolute log-ratio change. 배율 변화 크기
- $L$ : signed log-ratio change. 배율 증가/감소 방향 포함 변화
- $C$ : absolute composition change. 전체 스펙트럼 budget 내 점유율 변화 크기
- $S$ : signed composition change. 점유율 증가/감소 방향 포함 변화

userbin 변화량은 위 식에서 $P_{k,r}$ 를 $\bar{P}_{b,r}$ 로 바꿔 같은 이름을 사용한다.

#### canonical delta

epoch trend 에 사용하는 canonical delta 는 power reducer 와 bin reducer 를 모두 분리하여 저장한다.

$$
\delta_{\mathrm{exact},r,\mathrm{mean}}^{(\ell)}(e)
=
\frac{1}{F}\sum_{k=0}^{F-1} A_{k,r}^{(\ell,e)}
$$

$$
\delta_{\mathrm{exact},r,\mathrm{median}}^{(\ell)}(e)
=
\mathrm{median}_{k\in\{0,\ldots,F-1\}} A_{k,r}^{(\ell,e)}
$$

$$
\delta_{\mathrm{userbin},r,\mathrm{mean}}^{(\ell)}(e)
=
\frac{1}{B}\sum_{b=0}^{B-1} A_{b,r}^{(\ell,e)}
$$

$$
\delta_{\mathrm{userbin},r,\mathrm{median}}^{(\ell)}(e)
=
\mathrm{median}_{b\in\{0,\ldots,B-1\}} A_{b,r}^{(\ell,e)}
$$

#### 어느 단위로 측정, 저장하는지 설명

- probe 와 label_set 둘 다 저장한다.
- label_set 은 라벨별 raw 결과와 aggregate mean, aggregate median 을 모두 저장한다.
- `analysis_every` 단위 에포크에서만 layer pair 별 변화 곡선과 전체 pair heatmap 을 저장한다.
- exact FFT 는 `power_mean` 과 `power_median` 두 하위 경로로 나누어 저장한다.
- userbin FFT 도 `power_mean` 과 `power_median` 두 하위 경로로 나누어 저장한다.
- epoch trend 는 선택된 단위 에포크에서 얻은 점들로 누적 저장한다.

#### 어느 경로에 어떤 이름으로 저장되는지 설명

probe raw 의 per-epoch 저장 경로는 아래와 같다.

- `epochXXXX/probe/label_<label>/inter_layer_change/exact_fft/power_mean/<src_layer>_to_<dst_layer>_raw_abs.png`
- `epochXXXX/probe/label_<label>/inter_layer_change/exact_fft/power_mean/<src_layer>_to_<dst_layer>_raw_signed.png`
- `epochXXXX/probe/label_<label>/inter_layer_change/exact_fft/power_mean/<src_layer>_to_<dst_layer>_abs_logratio.png`
- `epochXXXX/probe/label_<label>/inter_layer_change/exact_fft/power_mean/<src_layer>_to_<dst_layer>_signed_logratio.png`
- `epochXXXX/probe/label_<label>/inter_layer_change/exact_fft/power_mean/<src_layer>_to_<dst_layer>_abs_composition.png`
- `epochXXXX/probe/label_<label>/inter_layer_change/exact_fft/power_mean/<src_layer>_to_<dst_layer>_signed_composition.png`
- `epochXXXX/probe/label_<label>/inter_layer_change/exact_fft/power_mean/layer_pairs_raw_abs_heatmap.png`
- `epochXXXX/probe/label_<label>/inter_layer_change/exact_fft/power_mean/layer_pairs_raw_signed_heatmap.png`
- `epochXXXX/probe/label_<label>/inter_layer_change/exact_fft/power_mean/layer_pairs_abs_logratio_heatmap.png`
- `epochXXXX/probe/label_<label>/inter_layer_change/exact_fft/power_mean/layer_pairs_signed_logratio_heatmap.png`
- `epochXXXX/probe/label_<label>/inter_layer_change/exact_fft/power_mean/layer_pairs_abs_composition_heatmap.png`
- `epochXXXX/probe/label_<label>/inter_layer_change/exact_fft/power_mean/layer_pairs_signed_composition_heatmap.png`
- `epochXXXX/probe/label_<label>/inter_layer_change/exact_fft/power_median/<src_layer>_to_<dst_layer>_raw_abs.png`
- `epochXXXX/probe/label_<label>/inter_layer_change/exact_fft/power_median/<src_layer>_to_<dst_layer>_raw_signed.png`
- `epochXXXX/probe/label_<label>/inter_layer_change/exact_fft/power_median/<src_layer>_to_<dst_layer>_abs_logratio.png`
- `epochXXXX/probe/label_<label>/inter_layer_change/exact_fft/power_median/<src_layer>_to_<dst_layer>_signed_logratio.png`
- `epochXXXX/probe/label_<label>/inter_layer_change/exact_fft/power_median/<src_layer>_to_<dst_layer>_abs_composition.png`
- `epochXXXX/probe/label_<label>/inter_layer_change/exact_fft/power_median/<src_layer>_to_<dst_layer>_signed_composition.png`
- `epochXXXX/probe/label_<label>/inter_layer_change/exact_fft/power_median/layer_pairs_raw_abs_heatmap.png`
- `epochXXXX/probe/label_<label>/inter_layer_change/exact_fft/power_median/layer_pairs_raw_signed_heatmap.png`
- `epochXXXX/probe/label_<label>/inter_layer_change/exact_fft/power_median/layer_pairs_abs_logratio_heatmap.png`
- `epochXXXX/probe/label_<label>/inter_layer_change/exact_fft/power_median/layer_pairs_signed_logratio_heatmap.png`
- `epochXXXX/probe/label_<label>/inter_layer_change/exact_fft/power_median/layer_pairs_abs_composition_heatmap.png`
- `epochXXXX/probe/label_<label>/inter_layer_change/exact_fft/power_median/layer_pairs_signed_composition_heatmap.png`

userbin FFT 도 같은 경로에서 `exact_fft` 를 `userbin_fft` 로 바꿔 저장한다.

label_set raw 는 같은 파일명을 `epochXXXX/label_set/label_<label>/inter_layer_change/...` 아래에 저장한다.

label_set aggregate mean 은 같은 파일명을 `epochXXXX/label_set/aggregate/mean/inter_layer_change/...` 아래에 저장한다.

label_set aggregate median 은 같은 파일명을 `epochXXXX/label_set/aggregate/median/inter_layer_change/...` 아래에 저장한다.

epoch trend 플롯은 아래처럼 저장한다.

- probe raw exact FFT power_mean mean: `epoch_trend/probe/label_<label>/inter_layer_change/exact_fft/power_mean/raw_abs_mean_epoch.png`
- probe raw exact FFT power_mean median: `epoch_trend/probe/label_<label>/inter_layer_change/exact_fft/power_mean/raw_abs_median_epoch.png`
- probe raw exact FFT power_median mean: `epoch_trend/probe/label_<label>/inter_layer_change/exact_fft/power_median/raw_abs_mean_epoch.png`
- probe raw exact FFT power_median median: `epoch_trend/probe/label_<label>/inter_layer_change/exact_fft/power_median/raw_abs_median_epoch.png`
- probe raw userbin FFT power_mean mean: `epoch_trend/probe/label_<label>/inter_layer_change/userbin_fft/power_mean/raw_abs_mean_epoch.png`
- probe raw userbin FFT power_mean median: `epoch_trend/probe/label_<label>/inter_layer_change/userbin_fft/power_mean/raw_abs_median_epoch.png`
- probe raw userbin FFT power_median mean: `epoch_trend/probe/label_<label>/inter_layer_change/userbin_fft/power_median/raw_abs_mean_epoch.png`
- probe raw userbin FFT power_median median: `epoch_trend/probe/label_<label>/inter_layer_change/userbin_fft/power_median/raw_abs_median_epoch.png`
- label_set aggregate mean exact FFT power_mean mean: `epoch_trend/label_set/aggregate/mean/inter_layer_change/exact_fft/power_mean/raw_abs_mean_epoch.png`
- label_set aggregate mean exact FFT power_mean median: `epoch_trend/label_set/aggregate/mean/inter_layer_change/exact_fft/power_mean/raw_abs_median_epoch.png`
- label_set aggregate mean exact FFT power_median mean: `epoch_trend/label_set/aggregate/mean/inter_layer_change/exact_fft/power_median/raw_abs_mean_epoch.png`
- label_set aggregate mean exact FFT power_median median: `epoch_trend/label_set/aggregate/mean/inter_layer_change/exact_fft/power_median/raw_abs_median_epoch.png`
- label_set aggregate mean userbin FFT power_mean mean: `epoch_trend/label_set/aggregate/mean/inter_layer_change/userbin_fft/power_mean/raw_abs_mean_epoch.png`
- label_set aggregate mean userbin FFT power_mean median: `epoch_trend/label_set/aggregate/mean/inter_layer_change/userbin_fft/power_mean/raw_abs_median_epoch.png`
- label_set aggregate mean userbin FFT power_median mean: `epoch_trend/label_set/aggregate/mean/inter_layer_change/userbin_fft/power_median/raw_abs_mean_epoch.png`
- label_set aggregate mean userbin FFT power_median median: `epoch_trend/label_set/aggregate/mean/inter_layer_change/userbin_fft/power_median/raw_abs_median_epoch.png`
- label_set aggregate median exact FFT power_mean mean: `epoch_trend/label_set/aggregate/median/inter_layer_change/exact_fft/power_mean/raw_abs_mean_epoch.png`
- label_set aggregate median exact FFT power_mean median: `epoch_trend/label_set/aggregate/median/inter_layer_change/exact_fft/power_mean/raw_abs_median_epoch.png`
- label_set aggregate median exact FFT power_median mean: `epoch_trend/label_set/aggregate/median/inter_layer_change/exact_fft/power_median/raw_abs_mean_epoch.png`
- label_set aggregate median exact FFT power_median median: `epoch_trend/label_set/aggregate/median/inter_layer_change/exact_fft/power_median/raw_abs_median_epoch.png`
- label_set aggregate median userbin FFT power_mean mean: `epoch_trend/label_set/aggregate/median/inter_layer_change/userbin_fft/power_mean/raw_abs_mean_epoch.png`
- label_set aggregate median userbin FFT power_mean median: `epoch_trend/label_set/aggregate/median/inter_layer_change/userbin_fft/power_mean/raw_abs_median_epoch.png`
- label_set aggregate median userbin FFT power_median mean: `epoch_trend/label_set/aggregate/median/inter_layer_change/userbin_fft/power_median/raw_abs_mean_epoch.png`
- label_set aggregate median userbin FFT power_median median: `epoch_trend/label_set/aggregate/median/inter_layer_change/userbin_fft/power_median/raw_abs_median_epoch.png`

### 4.12 필터적 성질 신호

#### 저장할 신호 설명

파라미터 기반 선형 응답으로부터 필터적 성질을 저장한다. 현재는 `my_R_DH_SNN`, `my_D_RF`, 그리고 추후 추가될 RF 계열 모델이 대상이다.

#### 어느 단위로 측정, 저장하는지 설명

- layer 단위와 model epoch trend 단위로 저장한다.
- tracked neuron 상세 플롯을 함께 저장한다.
- `plot_every` 단위 에포크에서만 snapshot 을 저장한다.

#### 어느 경로에 어떤 이름으로 저장되는지 설명

- per-epoch: `epochXXXX/filter_property/<layer_name>/summary.json`
- per-epoch: `epochXXXX/filter_property/<layer_name>/hist_f_peak.png`
- per-epoch: `epochXXXX/filter_property/<layer_name>/hist_bw_3db.png`
- per-epoch: `epochXXXX/filter_property/<layer_name>/hist_r0.png`
- per-epoch: `epochXXXX/filter_property/<layer_name>/hist_rpi.png`
- tracked neuron: `epochXXXX/filter_property/<layer_name>/tracked_neurons/neuron_<id>/branch_response_exact.png`
- tracked neuron: `epochXXXX/filter_property/<layer_name>/tracked_neurons/neuron_<id>/branch_response_userbin.png`
- tracked neuron: `epochXXXX/filter_property/<layer_name>/tracked_neurons/neuron_<id>/total_response_exact.png`
- tracked neuron: `epochXXXX/filter_property/<layer_name>/tracked_neurons/neuron_<id>/total_response_norm_exact.png`
- tracked neuron: `epochXXXX/filter_property/<layer_name>/tracked_neurons/neuron_<id>/total_response_userbin.png`
- tracked neuron: `epochXXXX/filter_property/<layer_name>/tracked_neurons/neuron_<id>/meta.json`
- epoch trend: `epoch_trend/filter_property/<layer_name>_ratio.png`
- epoch trend: `epoch_trend/filter_property/<layer_name>_f_peak_mean.png`
- epoch trend: `epoch_trend/filter_property/<layer_name>_bw_3db_mean.png`
- epoch trend: `epoch_trend/filter_property/<layer_name>_r0_mean.png`
- epoch trend: `epoch_trend/filter_property/<layer_name>_rpi_mean.png`

상세 수식과 모델별 전달 정의는 `paper/proposed/filter_analysis.md` 를 따른다.

---

## 5. config.json 규격

run root 에는 반드시 `config.json` 하나를 둔다. 이 파일은 설정한 하이퍼파라미터와 저장 규칙의 단일 기준 문서다.

필수 키는 아래와 같다.

- `run_name`
- `data_root_abs`
- `result_root_abs`
- `dataset_name`
- `model_name`
- `model_spec_doc`
- `epochs`
- `input_layer_semantics` = `"data_stream"`
- `optimizer_hyperparameters`
- `model_hyperparameters`
- `analysis_cadence` = `{plot_every, analysis_every, convergence_every}`
- `userbin_edges`
- `probe_indices`
- `label_set_indices`
- `tracked_neurons`
- `signal_manifest`
- `timing_factor_keys`
- `weight_group_keys`
- `tracked_state_kinds`
- `filter_property_enabled`
- `filter_property_doc` if enabled

즉, 설정한 하이퍼파라미터와 저장 규칙은 `config.json` 하나로 통일한다.

---

## 6. 출력 구조 예시

```text
<result_root_abs>/<run_name>/
  config.json
  train_test_accuracy.png
  epoch0010/
    distribution/
      timing/
        layer_hidden1_alpha.png
        model_alpha.png
      weights/
        layer_hidden1_layer_connection_weight.png
        model_layer_connection_weight.png
    probe/
      label_0/
        input_time.png
        input_fft_exact_abs.png
        input_fft_exact_log.png
        input_fft_userbin.png
        tracked_neurons/
          hidden1/
            neuron_003/
              input_dendrite_input_time.png
              input_dendrite_input_fft_exact.png
              input_dendrite_input_fft_userbin.png
              state_mem_time.png
              state_mem_fft_exact.png
              state_mem_fft_userbin.png
        layers/
          hidden1/
            state_membrane_time.png
            state_membrane_fft_exact_abs.png
            state_membrane_fft_exact_log.png
            state_membrane_fft_userbin.png
        inter_layer_change/
          exact_fft/
            input_to_hidden1_raw_abs.png
            layer_pairs_raw_abs_heatmap.png
          userbin_fft/
            input_to_hidden1_raw_abs.png
            layer_pairs_raw_abs_heatmap.png
    label_set/
      label_0/
        input_time.png
        ...
      aggregate/
        mean/
          input_time.png
          ...
        median/
          input_time.png
          ...
    filter_property/
      hidden1/
        summary.json
        hist_f_peak.png
        tracked_neurons/
          neuron_003/
            total_response_exact.png
            meta.json
  epoch0020/
    ...
  epoch_trend/
    probe/
      label_0/
        inter_layer_change/
          exact_fft/
            raw_abs_mean_epoch.png
            raw_abs_median_epoch.png
    label_set/
      aggregate/
        mean/
          inter_layer_change/
            userbin_fft/
              raw_abs_mean_epoch.png
              raw_abs_median_epoch.png
    filter_property/
      hidden1_f_peak_mean.png
```

---

## 7. 최소 확인 사항

- `epochXXXX/` 폴더는 각 산출물 family 의 단위 에포크를 만족하는 경우에만 생성된다.
- 입력층은 데이터 스트림 자체라고 `config.json` 과 문서에 명시되어 있다.
- run root 에 `train_test_accuracy.png` 가 **1개만** 존재한다.
- `probe/label_<label>/` 아래에 원본 입력 `input_time.png`, `input_fft_exact_abs.png`, `input_fft_exact_log.png`, `input_fft_userbin.png` 가 존재한다.
- layer heatmap 계열은 exact magnitude 와 exact log 두 파일을 함께 저장한다.
- tracked neuron state 저장 대상은 §3.5 의 뉴런 타입별 state 변수 규칙을 따른다.
- 레이어간 canonical delta epoch trend 는 mean 과 median 두 플롯을 함께 저장한다.
- `config.json` 하나만으로 경로, 하이퍼파라미터, signal manifest, tracked neuron, probe, label_set 선택 결과를 재현할 수 있다.
