# acc_benchmark 명세서

본 문서는 `acc_benchmark` 실험의 목적, 공정 비교 규칙, bash/CLI 인수 의미, 그리고 산출물 저장 규격을 정의한다. 실험 대상은 `src/neurons/` 아래의 임의 뉴런 모델이며, 목표는 **동일 hidden 구조와 동일 학습 예산에서 정확도 비교** 를 수행하는 것이다.

이 문서에서 말하는 **입력층(input layer)** 은 별도의 뉴런층이 아니라 **데이터셋의 직렬 데이터 스트림 자체** 를 뜻한다. 따라서 `--hidden` 은 input/output 을 제외한 hidden layer 만 지정한다.

---

## 0. 실험 목적

- 동일 데이터셋에서 여러 뉴런/모델 구조의 train/test accuracy 를 비교한다.
- hidden layer 개수와 각 hidden layer 뉴런 수는 공통으로 맞추고, 뉴런 내부 동역학만 바꿔 성능 차이를 본다.
- 가변 branch 모델의 경우 `S_min`, `S_max`, 정규화 항, 활성 파라미터 수를 함께 보고한다.

비교 대상 데이터셋은 s-MNIST, s-CIFAR10, SHD, SSC 를 기준으로 한다.

---

## 1. 모델 및 비교 단위

### 1.1 네트워크 표기

네트워크 표기는 항상 아래처럼 해석한다.

```text
input(data stream) -> hidden_1 -> ... -> hidden_L -> output(readout)
```

- `input(data stream)` : 데이터셋이 제공하는 시계열 입력 자체
- `hidden_l` : `--hidden` 으로 지정한 임의 뉴런 모델 층
- `output(readout)` : 데이터셋 class 수와 동일한 비스파이킹 readout 층

### 1.2 hidden 구조

- `--hidden <h1 h2 ...>` : hidden layer 개수와 각 hidden layer 뉴런 수를 지정한다.
- 예: `--hidden 128 64 32` 이면 hidden layer 는 3개이고, 각 층 뉴런 수는 128, 64, 32 다.
- input layer 는 데이터 스트림 자체이므로 `--hidden` 에 포함되지 않는다.

### 1.3 출력층

- 출력층 뉴런 수는 데이터셋 class 수로 고정한다.
- readout 은 막전위 기반 비스파이킹 출력층이다.
- 분류 점수는 출력층 막전위로부터 계산한다.
- output layer 에 spike/reset 을 두지 않는다.

### 1.4 LIF 기준 모델 통일

- `LIF` 는 막전위 감쇠계수(타이밍 팩터) $\alpha$ 를 학습하는 단일 구조다.
- reset 은 soft reset 으로 고정한다.

---

## 2. 정확도 측정 규칙

### 2.1 단위 에포크의 의미

`acc_benchmark` 에서 **단위 에포크(unit epoch)** 는 `--check_every` 로 전달하는 평가 간격을 뜻한다. 즉, 매 epoch 를 뜻하지 않는다.

- `--check_every 1` : 매 epoch 평가
- `--check_every 5` : 5, 10, 15, ... epoch 에서만 평가
- `--check_every 10` : 10, 20, 30, ... epoch 에서만 평가

### 2.2 accuracy 기록 방식

- train accuracy 와 test accuracy 는 **단위 에포크마다 수치만 누적** 한다.
- 누적된 accuracy 값들로 **학습 종료 후 단일 플롯 한 장** 을 그린다.
- 따라서 per-epoch PNG 나 per-unit-epoch PNG 를 여러 장 저장하지 않는다.

### 2.3 accuracy 플롯

- x축: `--check_every` 로 측정한 epoch index
- y축: accuracy
- 하나의 플롯에 train/test 두 곡선을 함께 그린다.
- 파일은 학습 종료 후 1회만 저장한다.

---

## 3. 공정 비교 규칙

1. 모든 모델은 동일한 데이터 split 을 사용한다.
2. 모든 모델은 동일한 hidden 구조 `--hidden` 을 사용한다.
3. 동일한 optimizer, epoch budget, batch size, 평가 cadence 를 사용한다.
4. 입력층은 데이터 스트림 자체이므로, 모델 비교에서 조절되는 구조는 hidden/output 부분만이다.
5. branch 기반 모델은 `S_min`, `S_max`, 활성 파라미터 수를 함께 보고한다.
6. 최종 비교 표에는 정확도뿐 아니라 실제 활성 파라미터 수를 함께 적는다.

---

## 4. bash/CLI 인수 설명

다음 인수들은 bash wrapper 가 python run script 로 넘겨주는 핵심 하이퍼파라미터다.

### 4.1 공통 실행 인수

- `--out_root` : 결과 저장 루트
- `--data_root` : 데이터 루트
- `--exp_name` : 실험 이름
- `--timestamp` : 실행 시각 태그
- `--gpu` : 사용할 GPU 인덱스
- `--seed` : 랜덤 시드
- `--models <m1 m2 ...>` : 비교할 모델 목록
- `--hidden <h1 h2 ...>` : hidden layer 개수 및 각 hidden layer 뉴런 수
- `--epochs` : 총 학습 epoch 수
- `--batch_size` : batch size
- `--lr` : learning rate
- `--num_workers` : DataLoader worker 수
- `--download` : 데이터 다운로드 허용 여부

### 4.2 2단계 구조 학습 인수

가변 branch 모델은 아래 인수를 추가로 사용한다.

- `--soft_mask_epochs` : Stage A 길이
- `--stabilize_epochs` : Stage B 길이
- `--ste_epochs` : Stage A 마지막 STE epoch 수
- `--S_min` : 연속 구조 변수 하한
- `--S_max <s1 s2 ...>` : branch 상한 또는 비교할 여러 S_max 값
- `--lambda_ortho` : orthogonality regularization 계수
- `--lambda_s` : 구조 변수 $s$ regularization 계수

### 4.3 뉴런/임계 관련 인수

- `--th_len` : adaptive threshold kernel 길이
- `--v_th` : threshold 초기값
- `--v_reset` : 사용하지 않는다. LIF 기준 모델은 soft reset 으로 고정하고 별도 reset 파라미터를 두지 않는다.
- `--v_pre` : pre-threshold 기준값
- `--weight_decay` : layer connection weight 에 적용할 AdamW weight decay
- `--weight_decay_dend_soma` : `my_R_DH_SNN` 의 `W_mix` 전용 weight decay

### 4.4 accuracy 측정 인수

- `--check_every` : train/test accuracy 를 몇 epoch 마다 계산할지 정하는 **단위 에포크**
- `--max_eval_batches` : 한 번의 평가에서 사용할 최대 배치 수. 0 이면 전체 평가셋 사용

### 4.5 데이터셋별 추가 인수

- s-CIFAR10: `--cifar_mode`
- SHD, SSC: `--T_event`

---

## 5. 산출물 구조

### 5.1 비교 항목 식별자

각 비교 항목의 식별자는 아래처럼 둔다.

```text
<neuron_structure>-<model_name>
```

여기서

- `neuron_structure` : `hidden`, `S_max`, 데이터셋별 구조 설정을 반영한 문자열
- `model_name` : 실제 모델 이름

예시는 다음과 같다.

```text
hidden_128-LIF
hidden_256_256-Smax8.0-my_R_DH_SNN
hidden_32_32_32-Smax4.0-my_D_RF
```

### 5.2 저장 파일

각 비교 항목마다 최소한 아래 파일을 저장한다.

```text
<result_root>/<run_name>/
  <neuron_structure>-<model_name>/
    train_test_accuracy.png
    hyperparameters.txt
    active_params.txt
```

- `train_test_accuracy.png` : 단위 에포크마다 측정한 train/test accuracy 를 학습 종료 후 단일 플롯으로 저장한 파일
- `hyperparameters.txt` : bash/CLI 로 전달한 하이퍼파라미터 값 요약
- `active_params.txt` : 활성 파라미터 수와 구조 요약

필요하면 timing factor 분포, 모델 전체 timing factor 분포, 활성 branch 분포 등을 추가로 저장할 수 있다. 다만 accuracy 비교의 핵심 산출물은 위 단일 accuracy 플롯이다.

---

## 6. 최소 확인 사항

- `--hidden` 이 hidden layer 개수와 각 층 뉴런 수를 정확히 지정한다.
- 입력층은 데이터 스트림 자체라고 문서와 결과 메타데이터에 명시되어 있다.
- `--check_every` 가 단위 에포크로 해석되고, 매 epoch 와 혼동되지 않는다.
- train/test accuracy 는 단위 에포크마다 측정하지만, PNG 는 학습 종료 후 **한 장만** 저장된다.
- 산출물 식별자는 `<neuron_structure>-<model_name>` 규칙을 따른다.
