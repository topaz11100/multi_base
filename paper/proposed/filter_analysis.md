# Filter Property Analysis 명세서

본 문서의 정식 파일명은 `filter_analysis.md` 이다. 본 문서는 `my_R_DH_SNN`, `my_D_RF`, 그리고 추후 추가될 RF 계열 모델의 **필터적 성질 분석 공통 규격** 을 정의한다.

## 0. 목적과 적용 범위

필터적 성질 분석의 목적은 학습된 파라미터만으로 뉴런 내부의 선형 응답을 직접 계산하고, 그 응답이 어느 주파수 대역을 통과시키는지 epoch 단위로 기록하는 데 있다.

적용 범위는 다음과 같다.

- `my_R_DH_SNN` : 완전 지원
- `my_D_RF` : 완전 지원
- vanilla RF : 저장 구조와 통계 키는 본 문서를 따르되, 모델 문서가 추가되는 시점에 전달식을 확정한다.

본 분석은 데이터 입력에 대한 forward 출력이 아니라 파라미터 기반 선형 분석이다. 따라서 저장 경로는 probe 나 label_set 아래가 아니라 epoch 단위 `filter_property/` 아래에 둔다.

모든 출력 경로는 프로젝트 상대경로가 아니라 사용자 지정 절대경로 `result_root_abs` 아래에 둔다.

### 0.1 상위 bash/CLI 인수 설명

본 분석은 `freq_analysis` 실험에서 호출되며, 관련 인수는 다음과 같다.

- `--plot_every` : filter-property snapshot 을 몇 epoch 마다 계산/저장할지 정하는 **단위 에포크**
- `--analysis_neurons <n1 n2 ...>` : 레이어별 tracked neuron 상세 응답 플롯 개수
- `--fft_band_edges <e0 e1 ... eB>` : userbin 응답 계산에 쓸 주파수 경계

여기서 말하는 단위 에포크는 "매 epoch" 가 아니라, 상위 bash/CLI 인수로 정한 저장 간격이다.

---

## 1. 공통 분석 원칙

### 1.1 분석 대상 경로

모든 모델에서 분석 대상은 모델 문서가 정의한 입력 신호에서 `soma_input` 으로 가는 선형 경로다.

- 입력 신호 이름과 의미는 모델 문서가 정의한다.
- 상위 실험 명세에서 `input` 이라는 표현이 나오면, 이는 별도 뉴런층이 아니라 **데이터 스트림 자체** 를 뜻한다.
- 출력 신호는 항상 `soma_input` 으로 통일한다.
- spike, threshold, reset, readout 은 분석 대상에서 제외한다.

즉, 본 분석은 비선형 스파이크 발생기가 아니라 `input -> soma_input` 선형 필터를 다룬다.

### 1.2 주파수 표현

모든 모델은 exact 주파수 응답과 userbin 주파수 응답을 함께 저장한다.

exact 주파수 그리드는

$$
f_i = \frac{i}{2(F-1)}, \qquad i\in\{0,\ldots,F-1\}
$$

로 정의하고, 각 점의 각주파수는

$$
\omega_i = 2\pi f_i
$$

로 둔다.

exact 응답 크기는

$$
A_i = \left|H\!\left(e^{j\omega_i}\right)\right|
$$

이며, 정규화 응답은

$$
\tilde{A}_i = \frac{A_i}{\max_j A_j + \varepsilon}
$$

로 둔다.

userbin 응답은 사용자 지정 bin 집합 $\mathcal{B}_b$ 를 이용해

$$
\bar{A}_b = \frac{1}{|\mathcal{B}_b|}\sum_{i\in\mathcal{B}_b} A_i
$$

로 계산한다.

### 1.3 epoch 단위 저장

- 모든 snapshot 산출물은 `epochXXXX/filter_property/` 아래에 저장한다.
- epoch 누적 추이는 `epoch_trend/filter_property/` 아래에 저장한다.
- snapshot 은 상위 실험의 `plot_every` 단위 에포크에서만 저장한다. 즉, 모든 epoch 에서 저장하지 않는다.
- `.pt`, `metrics.csv`, loss 플롯은 본 분석의 산출물이 아니다.

---

## 2. 모델별 전달 정의

### 2.1 R-DH-SNN

R-DH-SNN 에서 분석 대상은 뉴런 $m$ 의 `dendrite_input` 에서 `soma_input` 으로 가는 경로다.

입력은

$$
x_m[t] = O_{\mathrm{soma},m}[t]
$$

이고, 출력은

$$
h_m[t]
=
\frac{1}{s_m}
\sum_{d=1}^{D}
\Bigl(G_{m,d}(s_m)\,w_{m,d}\,i_{m,d}[t]\Bigr)
$$

이다.

단일 가지 EMA 응답은

$$
H_{m,d}(z)
=
\frac{1-\alpha_{m,d}}{1-\alpha_{m,d}z^{-1}}
$$

이고, 전체 전달함수는

$$
H_{\mathrm{total},m}(z)
=
\frac{1}{s_m}
\sum_{d=1}^{D}
\Bigl(
G_{m,d}(s_m)\,w_{m,d}\,H_{m,d}(z)
\Bigr)
$$

로 계산한다.

R-DH 의 세포체는 IF + soft reset 으로 바뀌었지만, 본 분석은 `soma_input` 까지만 다루므로 세포체 leak 유무는 응답식에 포함하지 않는다.

### 2.2 D-RF

D-RF 에서 분석 대상은 모델 문서가 정의한 가지 입력에서 `soma_input` 으로 가는 경로다.

단일 가지의 복소 상태는

$$
z_{m,d}[t] = \rho_{m,d}\,z_{m,d}[t-1] + \kappa_{m,d}\,I_m[t]
$$

로 두고,

$$
\rho_{m,d} = \exp\!\left(\delta\left(-\frac{1}{\tau_{m,d}} + i\omega_{m,d}\right)\right)
$$

로 둔다.

D-RF 의 `soma_input` 은 가지 상태의 실수부를 사용하므로, 필터 분석은 실수 커널

$$
g_{m,d}[k] = \Re\!\left\{\kappa_{m,d}\rho_{m,d}^{k}\right\}, \qquad k=0,\ldots,T_{\mathrm{filter}}-1
$$

를 기준으로 수행한다.

뉴런 $m$ 의 전체 실수 커널은

$$
g_{\mathrm{total},m}[k]
=
\frac{1}{s_m}
\sum_{d=1}^{D}
M_{m,d}(s_m)\,g_{m,d}[k]
$$

로 정의한다.

exact 응답은 이 커널의 DTFT 또는 길이 $T_{\mathrm{filter}}$ FFT 로 계산한다.

$$
H_{\mathrm{total},m}\!\left(e^{j2\pi f_i}\right)
=
\sum_{k=0}^{T_{\mathrm{filter}}-1}
 g_{\mathrm{total},m}[k] e^{-j2\pi f_i k}
$$

이때 $T_{\mathrm{filter}}$ 는 분석에 사용하는 커널 절단 길이이며, 기본값은 실험의 시계열 길이 $T$ 와 동일하게 둔다.

### 2.3 vanilla RF

vanilla RF 는 추후 모델 문서가 추가되면 본 절에 전달식을 확정한다. 현재 단계에서는 아래 두 규칙만 고정한다.

- 저장 구조, 파일 이름, 통계 키는 본 문서를 그대로 따른다.
- 모델 문서가 없으면 분석을 생략하고 `config.json` 에 `filter_property_status = "planned"` 를 기록한다.

---

## 3. 공통 요약 통계와 분류 규칙

### 3.1 저장할 요약 통계

모든 모델은 뉴런별로 아래 스칼라를 저장한다.

- `f_peak` : exact 응답의 peak frequency
- `f_L` : principal $-3\,\mathrm{dB}$ passband lower edge
- `f_H` : principal $-3\,\mathrm{dB}$ passband upper edge
- `bw_3db` : $f_H - f_L$
- `r0` : 정규화 응답의 DC 비율
- `rpi` : 정규화 응답의 Nyquist 비율
- `class` : `lp`, `bp`, `hp`, `mixed`
- `active_branch_count`
- `s_value`

모델 특화 파라미터도 함께 저장한다.

- R-DH : `positive_mix_weight_count`, `negative_mix_weight_count`
- D-RF : `tau_mean`, `omega_mean`, `tau_std`, `omega_std`

### 3.2 $-3\,\mathrm{dB}$ passband 추출

half-power 임계값은

$$
\gamma = \frac{1}{\sqrt{2}}
$$

로 둔다.

정규화 응답에서 half-power 집합은

$$
\mathcal{P}_{3\mathrm{dB}} = \{ f_i \mid \tilde{A}_i \ge \gamma \}
$$

이고, global peak 를 포함하는 연결 성분의 좌우 경계를 각각 $f_L$, $f_H$ 로 정의한다.

### 3.3 분류 규칙

한 개의 principal component 에 대해

$$
\delta_f = \frac{0.5}{F-1}
$$

를 허용오차로 둘 때,

- LP: $f_L \le \delta_f$ 이고 $f_H < 0.5 - \delta_f$
- HP: $f_L > \delta_f$ 이고 $f_H \ge 0.5 - \delta_f$
- BP: $\delta_f < f_L < f_H < 0.5 - \delta_f$
- mixed: 위 세 경우에 안정적으로 속하지 않거나 peak component 가 다중 대역인 경우

분류는 exact 응답을 기준으로 수행하고, userbin 응답은 저장과 시각화에만 사용한다.

---

## 4. 저장 산출물

### 4.1 뉴런 단위 산출물

tracked neuron 마다 아래 파일을 저장한다.

```text
tracked_neurons/
  neuron_<n>/
    branch_response_exact.png
    branch_response_userbin.png
    total_response_exact.png
    total_response_norm_exact.png
    total_response_userbin.png
    meta.json
```

각 파일의 의미는 다음과 같다.

- `branch_response_exact.png` : 가지별 exact 응답 multi-line plot
- `branch_response_userbin.png` : 가지별 userbin 응답 multi-line plot
- `total_response_exact.png` : 전체 응답의 비정규화 exact plot
- `total_response_norm_exact.png` : 전체 응답의 정규화 exact plot
- `total_response_userbin.png` : 전체 응답의 userbin plot
- `meta.json` : 요약 통계와 모델 특화 파라미터

R-DH 처럼 branch cutoff 가 명확한 모델은 `meta.json` 안에 branch cutoff 배열을 추가할 수 있다. D-RF 처럼 branch peak 가 더 의미 있는 모델은 branch peak 배열을 저장한다.

### 4.2 레이어 단위 산출물

레이어마다 아래 파일을 저장한다.

```text
<layer_name>/
  summary.json
  hist_f_peak.png
  hist_bw_3db.png
  hist_r0.png
  hist_rpi.png
  tracked_neurons/
```

`summary.json` 은 최소한 아래 키를 포함한다.

- `num_neurons`
- `count_lp`, `count_bp`, `count_hp`, `count_mixed`
- `ratio_lp`, `ratio_bp`, `ratio_hp`, `ratio_mixed`
- `f_peak_mean`, `f_peak_std`, `f_peak_median`
- `bw_3db_mean`, `bw_3db_std`, `bw_3db_median`
- `r0_mean`, `r0_std`, `r0_median`
- `rpi_mean`, `rpi_std`, `rpi_median`

### 4.3 모델 전체 epoch 추이

epoch 누적 추이는 `epoch_trend/filter_property/` 아래에 저장한다.

```text
epoch_trend/
  filter_property/
    <layer_name>_ratio.png
    <layer_name>_f_peak_mean.png
    <layer_name>_bw_3db_mean.png
    <layer_name>_r0_mean.png
    <layer_name>_rpi_mean.png
```

각 플롯의 x축은 epoch 이고, y축은 해당 레이어 요약 통계다.

---

## 5. freq_analysis 실험과의 연동

`freq_analysis` 실험에서 필터적 성질 분석이 가능한 모델은 아래 경로를 사용한다.

```text
<result_root_abs>/<run_name>/
  epoch0001/
    filter_property/
      <layer_name>/
        summary.json
        hist_f_peak.png
        hist_bw_3db.png
        hist_r0.png
        hist_rpi.png
        tracked_neurons/
          neuron_<n>/
            branch_response_exact.png
            branch_response_userbin.png
            total_response_exact.png
            total_response_norm_exact.png
            total_response_userbin.png
            meta.json
  epoch_trend/
    filter_property/
      <layer_name>_ratio.png
      <layer_name>_f_peak_mean.png
      <layer_name>_bw_3db_mean.png
      <layer_name>_r0_mean.png
      <layer_name>_rpi_mean.png
```

모델이 필터적 성질 분석을 지원하지 않으면 `epochXXXX/filter_property/` 는 생성하지 않는다. 이 경우 지원 여부와 사유를 `config.json` 에 명시한다.
