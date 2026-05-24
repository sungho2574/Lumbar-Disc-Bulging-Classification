# 실험 결과 요약

## 데이터셋 개요

- **출처**: SPIDER (Lumbar Spine Segmentation in MR Images, Kaggle)
- **이미지**: T2 MRI `.mha` 파일 (3D 볼륨)
- **레이블**: `radiological_gradings.csv` — IVD 1~6번 추간판별 진단 등급
- **전처리 공통**: 최단 축의 중앙 슬라이스 추출 (시상면 기준)

---

## Section 4 — 전척추 이미지 멀티레이블 베이스라인

### 설정

| 항목 | 내용 |
|------|------|
| **타겟** | Disc Herniation (IVD 1~6) |
| **샘플 단위** | 환자 1명 = 샘플 1개 |
| **데이터** | 217명, Train 151 / Val 33 / Test 33 |
| **입력** | 전척추 중앙 슬라이스 → 224×224 강제 리사이즈 (정규화 없음) |
| **레이블** | 6-dim 벡터 [IVD1..IVD6 herniation], 양성 비율 5.5% |
| **모델** | SpiderCNN — Conv(1→32→64→128) + BN + Pool × 3 → FC(256) → FC(6) |
| **손실함수** | BCEWithLogitsLoss |
| **옵티마이저** | Adam lr=1e-4, Early Stopping patience=10 |

### 학습 로그

```
Epoch   1: train=10.4500  val=0.1722
Epoch   5: train=0.2003   val=0.1730
Epoch  10: train=0.1214   val=0.1669
Epoch  15: train=0.0670   val=0.2163
Early stopping at epoch 19
```

### 결과

| Split | Accuracy | Precision | Recall | F1 |
|-------|----------|-----------|--------|----|
| **Val** | 0.9583 | 0.0000 | 0.0000 | 0.0000 |
| **Test** | 0.9570 | 0.0000 | 0.0000 | 0.0000 |

*디스크별 결과 (Val):*

| Disc | Accuracy | Precision | Recall | F1 |
|------|----------|-----------|--------|----|
| IVD 1 | 0.8750 | 0.0 | 0.0 | 0.0 |
| IVD 2 | 0.8750 | 0.0 | 0.0 | 0.0 |
| IVD 3 | 1.0000 | 0.0 | 0.0 | 0.0 |
| IVD 4 | 1.0000 | 0.0 | 0.0 | 0.0 |
| IVD 5 | 1.0000 | 0.0 | 0.0 | 0.0 |
| IVD 6 | 1.0000 | 0.0 | 0.0 | 0.0 |

### 분석

- 양성(herniation) 비율이 **5.5%**로 극단적으로 불균형
- 모델이 모든 샘플을 **무조건 Negative로 예측**하는 trivial solution에 수렴
- Accuracy만 높고 Precision/Recall/F1 전부 0 → 사실상 학습 실패
- 전척추 이미지를 그대로 입력하여 모델이 어느 디스크를 봐야 하는지 알 수 없음

---

## Section 5 — 디스크 크롭 + Focal Loss (Herniation)

### 설정

| 항목 | 내용 |
|------|------|
| **타겟** | Disc Herniation (이진분류) |
| **샘플 단위** | 디스크 1개 = 샘플 1개 |
| **데이터** | 209명, 1,254개 디스크 샘플, Train 876 / Val 186 / Test 192 |
| **입력** | 마스크(201~206)로 개별 추간판 크롭 + PixelSpacing 기반 40mm 여백 → 128×128 (정규화 없음) |
| **레이블** | 이진 (herniation 0/1), 양성 비율 5.7% |
| **모델** | SpiderDiscCNN — Conv(1→32→64→128) + BN + Pool × 3 → FC(256) → FC(1) |
| **손실함수** | Focal Loss (alpha=0.25, gamma=2.0) |
| **옵티마이저** | Adam lr=1e-4, Early Stopping patience=10 |

### 학습 로그

```
Epoch   1: train=0.0685  val=0.0142
Epoch   5: train=0.0098  val=0.0177
Epoch  10: train=0.0072  val=0.0165
Epoch  15: train=0.0058  val=0.0145
Early stopping at epoch 17
```

### 결과

| Split | Accuracy | Precision | Recall | F1 |
|-------|----------|-----------|--------|----|
| **Val** | 0.9462 | 0.0000 | 0.0000 | 0.0000 |
| **Test** | 0.9323 | 0.0000 | 0.0000 | 0.0000 |

### 분석

- 마스크 크롭으로 입력 품질은 개선됐으나 **양성 비율 5.7%로 여전히 극단적 불균형**
- Focal Loss를 적용해도 Trivial solution 수렴 탈피 실패
- Herniation 자체가 너무 희귀 이벤트 → 현재 데이터 규모(~70개 양성 샘플)로 학습 불가

---

## Section 6 — 디스크 크롭 베이스라인 (Bulging)

### 설정

| 항목 | 내용 |
|------|------|
| **타겟** | Disc Bulging (이진분류) — 타겟 교체 |
| **샘플 단위** | 디스크 1개 = 샘플 1개 |
| **데이터** | 209명, 1,254개 디스크 샘플, Train 876 / Val 186 / Test 192 |
| **입력** | Section 5와 동일 (마스크 크롭 128×128, 정규화 없음) |
| **레이블** | 이진 (bulging 0/1), 양성 비율 55.3% |
| **모델** | SpiderDiscCNN (Section 5와 동일 아키텍처) |
| **손실함수** | BCEWithLogitsLoss (불균형 처리 불필요) |
| **옵티마이저** | Adam lr=1e-4, Early Stopping patience=10 |

### 학습 로그

```
Epoch   1: train=0.6973  val=0.5928
Epoch   5: train=0.4103  val=0.4520
Epoch  10: train=0.2691  val=0.5469
Early stopping at epoch 15
```

### 결과

| Split | Accuracy | Precision | Recall | F1 |
|-------|----------|-----------|--------|----|
| **Val** | 0.8065 | 0.8182 | 0.8491 | 0.8333 |
| **Test** | 0.7708 | 0.7193 | 0.8723 | 0.7885 |

### 분석

- 타겟을 Bulging(55.3% 균형)으로 바꾸자 **처음으로 실제 학습 성공**
- Val F1 0.83으로 준수한 성능
- 정규화 없이 원시 픽셀값 학습 → train loss는 내려가나 val loss는 에폭 5 이후 상승 → overfitting 경향

---

## Section 7 — Z-score 정규화 + Data Augmentation (Bulging)

### 설정

| 항목 | 내용 |
|------|------|
| **타겟** | Disc Bulging (Section 6과 동일) |
| **샘플 단위** | 디스크 1개 = 샘플 1개 |
| **데이터** | Section 6과 동일 (876 / 186 / 192) |
| **입력** | 마스크 크롭 128×128 + **Z-score 정규화** (crop 단위 mean/std) |
| **증강** | Train only — RandomHorizontalFlip(p=0.5) + RandomRotation(±15°) |
| **레이블** | Section 6과 동일 |
| **모델** | SpiderDiscCNN (동일) |
| **손실함수** | BCEWithLogitsLoss |
| **옵티마이저** | Adam lr=1e-4, Early Stopping patience=10 |

### 학습 로그

```
Epoch   1: train=0.8612  val=0.6068
Epoch   5: train=0.5065  val=0.4502
Epoch  10: train=0.4478  val=0.4137
Epoch  20: train=0.4006  val=0.3967
Epoch  30: train=0.3797  val=0.3903
Early stopping at epoch 32
```

### 결과

| Split | Accuracy | Precision | Recall | F1 |
|-------|----------|-----------|--------|----|
| **Val** | 0.8118 | 0.8034 | 0.8868 | **0.8430** |
| **Test** | 0.7969 | 0.7273 | 0.9362 | **0.8186** |

### Section 6 대비 변화량

| 지표 | Val Δ | Test Δ |
|------|-------|--------|
| Accuracy | +0.0053 | +0.0261 |
| Precision | −0.0148 | +0.0080 |
| Recall | +0.0377 | +0.0639 ↑ |
| F1 | +0.0097 | +0.0301 ↑ |

### 분석

- Early stopping이 epoch 32까지 연장 → 정규화 덕분에 학습이 더 안정적
- **Recall이 크게 향상** (Val +3.8%, Test +6.4%) → bulging을 더 잘 잡아냄
- Precision은 소폭 하락 → 탐지를 더 공격적으로 하면서 False Positive 소폭 증가
- 전반적으로 F1 기준 Section 6 대비 개선

---

## Section 8 — Physical-Scale Preprocessing (등방 리샘플 + 중심 크롭)

### 문제 인식 (Section 6/7 한계)

Section 6/7의 전처리 흐름:
```
PixelSpacing 기반 픽셀 margin 계산 → bbox 크롭 → T.Resize(128×128) 강제 리사이즈
```
- `sh ≠ sw`이면 종횡비(aspect ratio) 왜곡 상태로 학습
- 환자마다 1픽셀이 나타내는 실제 물리 거리가 다름 → 모델이 스케일 불변성 없이 학습

### 설정

| 항목 | 내용 |
|------|------|
| **타겟** | Disc Bulging (Section 6/7과 동일) |
| **샘플 단위** | 디스크 1개 = 샘플 1개 |
| **데이터** | Section 6/7과 동일 split (Train 876 / Val 186 / Test 192) |
| **입력 파이프라인** | ① 중앙 시상면 슬라이스 추출 → ② `scipy.ndimage.zoom`으로 **TARGET_SP = 0.75 mm/px** 등방 리샘플 → ③ 디스크 중심 기준 **128×128 중심 크롭 + 제로패딩(검정)** |
| **물리적 시야** | 128px × 0.75mm = **96mm × 96mm** 고정 (디스크 ~15mm + 여백 40mm × 2 ≈ 95mm) |
| **레이블** | Section 6/7과 동일 |
| **모델** | SpiderDiscCNN (동일 아키텍처) |
| **손실함수** | BCEWithLogitsLoss |
| **옵티마이저** | Adam lr=1e-4, Early Stopping patience=10 |
| **정규화·증강** | 없음 (전처리 효과만 단독 측정) |

### Section 6 vs Section 8 전처리 비교

| 항목 | Section 6 | Section 8 |
|------|-----------|-----------|
| 스케일 통일 | ✗ (환자마다 다름) | ✅ 0.75mm/px 등방 |
| 종횡비 | 왜곡 (T.Resize) | ✅ 정확 (중심 크롭) |
| 패딩 | 없음 | 제로패딩(검정) |
| 출력 크기 | 128×128 | 128×128 |

### 학습 로그

```
(학습 후 업데이트)
```

### 결과

| Split | Accuracy | Precision | Recall | F1 |
|-------|----------|-----------|--------|----|
| **Val** | — | — | — | — |
| **Test** | — | — | — | — |

### 분석

(학습 후 업데이트)

---

## 종합 비교

| Section | 타겟 | 입력 | 정규화 | 증강 | 손실함수 | Test F1 |
|---------|------|------|--------|------|----------|---------|
| **4** | Herniation (6-label) | 전척추 224×224 | ✗ | ✗ | BCE | 0.0000 ❌ |
| **5** | Herniation (binary) | 디스크 크롭 128×128 | ✗ | ✗ | Focal Loss | 0.0000 ❌ |
| **6** | **Bulging** (binary) | bbox 크롭 → Resize 128×128 | ✗ | ✗ | BCE | 0.7885 ✅ |
| **7** | **Bulging** (binary) | bbox 크롭 → Resize 128×128 | Z-score | Flip+Rot | BCE | **0.8186** ✅ |
| **8** | **Bulging** (binary) | 등방 리샘플 → 중심 크롭 128×128 | ✗ | ✗ | BCE | — (학습 후) |

**핵심 관찰**
1. Herniation(~5%)은 현재 규모에서 학습 자체가 불가 — 클래스 불균형이 결정적
2. 디스크 크롭(Section 5→6)은 타겟이 맞으면 효과적인 전처리
3. Z-score + 증강(Section 7)은 특히 Recall을 개선하며 일반화 성능을 높임
4. Section 8은 물리적 스케일 통일이 성능에 미치는 영향을 측정 — 정규화·증강 없이 전처리만으로 비교
