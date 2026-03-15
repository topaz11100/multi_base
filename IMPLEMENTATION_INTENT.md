# 구현 의도 설명

이번 업데이트는 새 명세(모든 promft.txt, `paper/proposed/*`, `src/**/*.md`) 기준으로, 특히 **basic_long_term_mem 실험의 구버전(on-the-fly synthetic batch) 파이프라인을 고정 직렬 데이터셋 기반 구조로 전환**하는 데 초점을 맞췄다.

## 핵심 의도
- 실험 중 데이터 재생성 금지 원칙 반영: `data_root` 아래 `delayed_xor_serial`, `multiscale_xor_serial` 생성/재사용.
- 명세의 저장 규격 반영: split별 `train/val/test.npz`, `meta.json`, checksum 기록.
- XOR 실험 모델 구조를 단일 뉴런에서 hidden 다층 구조로 일반화: `--hidden` 인수 지원.
- SSH 단절 내성 실행 정책 유지: bash 래퍼에서 `nohup` 기반 백그라운드 실행 유지.
- 기존 코드 재활용 원칙 유지: 기존 뉴런/빌더/학습 로직을 최대한 활용하고 데이터/드라이버 경로만 교체.

## 설계 선택
- 데이터 생성/보존 로직은 공통 모듈(`src/common/long_term_mem_dataset.py`)로 분리하여 delayed/multiscale 모두 재사용.
- 학습은 고정 train split에서 미니배치 샘플링, 평가는 고정 test split 전체 배치 순회로 일관화.
- 출력은 기존과 동일하게 soma 기반 로짓을 사용하되, 출력층 스파이크는 비활성화하여 readout 일관성을 유지.

## 제외/유지 항목
- `rf_motivation`은 명세 부재 상태로 요청대로 미변경 유지.
- acc/freq 계열은 기존 구현이 이미 문서 명세에 가깝게 정리되어 있어 이번 변경의 중심을 XOR 파이프라인/스크립트 정합성에 두었다.
