# Lumbar Disc Bulging Classification

- 인하대학교 기계학습 프로젝트 (2026년 1학기)
- 요추 MRI(T2) 영상에서 추간판 팽윤(Disc Bulging)을 이진 분류

## 개요

요추 추간판(IVD, Intervertebral Disc)의 팽윤 여부를 T2 MRI 시상면(sagittal) 영상으로부터 판별한다.
분류(classification)의 기본 틀 안에서 전처리 전략, 모델 경량화, 평가 방법론을 단계적으로 비교한 실험 기록이다.

- **타겟**: `disc_bulging` (0/1) — 전체 추간판 중 약 49% 양성
- **입력**: 각 추간판(IVD 1~6)의 T2 MRI 중앙 시상면 슬라이스를 128×128로 크롭
- **모델**: 2-layer CNN + Global Average Pooling (19K params)
- **평가**: 3-Fold Stratified Cross-Validation (환자 레벨)

## 데이터셋

[SPIDER Dataset (Kaggle)](https://www.kaggle.com/datasets/zofiaknapiska/spider-lumbar-spine-segmentation-in-mr-images) — SPIne Degenerative disEase Records

| 항목           | 내용                                             |
| -------------- | ------------------------------------------------ |
| 환자 수        | 218명 (요통 환자, 4개 병원)                      |
| MRI 시리즈     | 447개 (T1 + T2)                                  |
| 세그멘테이션   | 척추체·추간판·척수강 마스크 (.mha)               |
| 방사선 소견    | 7가지 레이블 (Bulging, Herniation, Narrowing 등) |
| 유효 환자 (T2) | 209명, 추간판 샘플 1,254개                       |

데이터셋은 **MIT License**로 공개되어 있어 학술·상업적 용도로 자유롭게 사용 가능하다.

## 환경 설정

- [uv](https://docs.astral.sh/uv/) 기반, Python 3.12.

```bash
# uv 설치 (없는 경우)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 가상환경 생성 및 패키지 설치
uv sync

# Jupyter 커널 등록
uv run python -m ipykernel install --user --name lumbar-disc-bulging
```

> **CUDA 사용 시**: `pyproject.toml` 하단의 `[tool.uv.sources]` 주석을 해제하고 `uv sync` 재실행.

## 실행

```bash
# 실험 노트북 (EDA + 보고서 포함 실험만 정리된 버전)
uv run jupyter notebook lumbar_disc_bulging_clean.ipynb
```

셀을 위에서부터 순서대로 실행하면 된다.
학습 결과는 `results/` 하위 디렉토리에 자동 저장된다 (loss_curves.pdf, confusion_matrices.pdf, metrics.csv).

> **전체 탐색 과정**이 궁금하다면 `lumber_disc_bulging_prediction.ipynb`(Section 1~17)을 참고한다.

## 프로젝트 구조

```
├── data/                                 # 데이터셋 (별도 다운로드 필요)
├── models/                               # 저장된 모델 가중치 (.pth)
├── lumbar_disc_bulging_clean.ipynb       # 실행용 노트북 (EDA + 주요 실험)
├── lumber_disc_bulging_prediction.ipynb  # 전체 탐색 노트북 (Section 1~17, 시행착오 포함 전체 실험)
├── pyproject.toml                        # uv 환경 설정
└── README.md
```
