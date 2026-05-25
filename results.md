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
Epoch   1: train=0.6861  val=0.5409
Epoch   5: train=0.3662  val=0.4599
Epoch  10: train=0.2329  val=0.4629
Epoch  15: train=0.1953  val=0.4604
Early stopping at epoch 18
```

### 결과

| Split | Accuracy | Precision | Recall | F1 |
|-------|----------|-----------|--------|----|
| **Val** | 0.7903 | 0.8018 | 0.8396 | 0.8203 |
| **Test** | 0.7656 | 0.7207 | 0.8511 | **0.7805** |

### Section 6 대비 변화량

| 지표 | Val Δ | Test Δ |
|------|-------|--------|
| Accuracy | −0.0162 | −0.0052 |
| Precision | −0.0164 | +0.0014 |
| Recall | −0.0095 | −0.0212 |
| F1 | −0.0130 | −0.0080 |

### 분석

- Early stopping이 epoch 18로 Section 6(15)보다 소폭 연장 — 물리적 스케일 통일로 학습 신호가 다소 안정화
- Val F1 0.8203으로 Section 6(0.8333) 대비 소폭 하락, Section 7(0.8430)보다도 낮음
- Test F1 0.7805로 Section 6(0.7885) 대비 소폭 하락
- 등방 리샘플만으로는 성능이 뚜렷하게 오르지 않음 — 정규화·증강 없이 전처리만 바꾼 효과는 제한적
- Recall은 Val/Test 모두 0.83~0.85 수준으로 비교적 양호하게 유지
- 참고: PixelSpacing 분포 — sh 중앙값 0.624mm, sw 중앙값 0.625mm로 대부분 등방에 가까우며 비등방 환자는 9명(P58 ratio=0.105 등). 데이터셋 자체가 심각하게 비등방인 케이스가 적어 리샘플 효과가 제한적이었을 가능성 있음

---

## Section 9 — Multi-Slice 3-Channel Input (좌·중·우 시상면)

### 문제 인식 (Section 6~8 한계)

Section 6~8은 모두 중앙 시상면 슬라이스 **1장**만 사용.
디스크 팽윤·탈출은 3D 현상으로 좌측/우측으로 편측 발생하는 경우가 흔해,
중앙 슬라이스 하나로는 편측 병변을 놓칠 수 있다.

### 설정

| 항목 | 내용 |
|------|------|
| **타겟** | Disc Bulging (동일) |
| **샘플 단위** | 디스크 1개 = 샘플 1개 |
| **데이터** | 동일 split (Train 876 / Val 186 / Test 192) |
| **슬라이스 선택** | 물리적 ±4mm 오프셋 — `offset = max(1, round(4.0 / through_sp))` → 대부분 ±1슬라이스 (through_sp 중앙값 3.3mm) |
| **입력 파이프라인** | 좌·중·우 3장 각각 Section 8 전처리 (TARGET_SP=0.75mm/px, 128×128 중심 크롭) → **(3, 128, 128)** 스택 |
| **디스크 중심** | 중앙 슬라이스 마스크 기준으로 계산, 3장 동일 좌표 크롭 |
| **모델** | **SpiderDiscCNN3** — 첫 Conv2d만 1→3채널 변경, 나머지 동일 (파라미터 8,482,817) |
| **손실함수** | BCEWithLogitsLoss |
| **옵티마이저** | Adam lr=1e-4, Early Stopping patience=10 |
| **정규화·증강** | 없음 |

### 학습 로그

```
Epoch   1: train=0.7395  val=0.5438
Epoch   5: train=0.4321  val=0.5204
Epoch  10: train=0.2608  val=0.4599
Epoch  15: train=0.1431  val=0.4849
Early stopping at epoch 19
```

### 결과

| Split | Accuracy | Precision | Recall | F1 |
|-------|----------|-----------|--------|----|
| **Val** | 0.7903 | 0.7769 | 0.8868 | 0.8282 |
| **Test** | 0.7865 | 0.7431 | 0.8617 | **0.7980** |

### Section 8 대비 변화량

| 지표 | Val Δ | Test Δ |
|------|-------|--------|
| Accuracy | +0.0000 | +0.0209 ↑ |
| Precision | −0.0249 | +0.0224 ↑ |
| Recall | +0.0472 ↑ | +0.0106 ↑ |
| F1 | +0.0079 | +0.0175 ↑ |

### 분석

- Early stopping epoch 19로 Section 8(18)과 유사 — 학습 안정성 비슷
- **Test F1 0.7980** — Section 8(0.7805) 대비 +0.0175 개선, Section 6 베이스라인(0.7885)도 상회
- **Recall이 특히 향상** (Val +4.7%, Test +1.1%) — 3채널 입력이 편측 병변 탐지에 도움을 준 것으로 해석
- Precision은 Val에서 소폭 하락 — 탐지를 더 공격적으로 하며 False Positive 소폭 증가
- 단, 최고 성능 Section 7(Test F1 0.8186)에는 미치지 못함 — 정규화·증강 없는 전처리 개선의 한계
- 참고: 데이터셋 through-plane spacing 중앙값 3.3mm로, 대부분 환자는 실질적으로 ±1슬라이스(±3.3mm ≈ ±4mm) 추출

---

## Section 10 — Multi-Slice + Z-score + Augmentation (Sec9 + Sec7 조합)

### 배경

Sec 7(Z-score + 증강)과 Sec 9(3채널 멀티슬라이스)가 각각 다른 축에서 효과를 보였으므로
두 개선을 합쳐 시너지 여부를 측정한다.

### 설정

| 항목 | 내용 |
|------|------|
| **타겟** | Disc Bulging (동일) |
| **데이터** | 동일 split (Train 876 / Val 186 / Test 192) |
| **입력** | 좌·중·우 3슬라이스 (물리적 ±4mm) → 각각 등방 리샘플(0.75mm/px) + 중심 크롭 128×128 |
| **정규화** | 채널별 독립 Z-score |
| **증강** | Train only — RandomHorizontalFlip(p=0.5) + RandomRotation(±15°) **(3채널 동시 적용)** |
| **모델** | SpiderDiscCNN3 (동일, 파라미터 8,482,817) |
| **손실함수** | BCEWithLogitsLoss |
| **옵티마이저** | Adam lr=1e-4, Early Stopping patience=10 |

### 학습 로그

```
Epoch   1: train=0.8102  val=0.6060
Epoch   5: train=0.5069  val=0.4570
Epoch  10: train=0.4436  val=0.4280
Epoch  15: train=0.3759  val=0.4353
Early stopping at epoch 18
```

### 결과

| Split | Accuracy | Precision | Recall | F1 |
|-------|----------|-----------|--------|----|
| **Val** | 0.7957 | 0.7881 | 0.8774 | 0.8304 |
| **Test** | 0.7708 | 0.7156 | 0.8571 | **0.7800** |

### 주요 섹션 대비 변화량

| 지표 | vs Sec 7 (Val) | vs Sec 7 (Test) | vs Sec 9 (Val) | vs Sec 9 (Test) |
|------|---------------|----------------|---------------|----------------|
| Accuracy | −0.0161 | −0.0261 | +0.0054 | −0.0157 |
| Precision | −0.0153 | −0.0275 | +0.0112 | −0.0275 |
| Recall | −0.0094 | −0.0791 | −0.0094 | −0.0046 |
| **F1** | **−0.0126** | **−0.0386** | **+0.0022** | **−0.0180** |

### 분석

- Val F1 0.8304 — Sec 9(0.8282) 대비 소폭 개선, Sec 7(0.8430)보다는 낮음
- **Test F1 0.7800** — Sec 7(0.8186), Sec 9(0.7980) 모두 하회. 조합의 시너지 없음
- 3채널 입력에 Z-score + 증강을 추가해도 Test 성능이 오르지 않은 원인 분석:
  - 데이터 876개로 3채널 입력은 이미 입력 복잡도가 높은데, 증강까지 더해 학습 난이도 상승
  - Early stopping(18 epoch)이 학습 조기 종료 — Z-score + 증강 조합은 수렴에 더 많은 에폭이 필요할 수 있음 (Sec 7은 32 epoch)
  - Val-Test 격차(0.8304 → 0.7800)가 크게 벌어짐 → val set 과적합 가능성

---

## Section 11 — CNN Feature Extraction + Tabular ML 분류

### 배경

같은 방사선 판독 레이블(Pfirrmann, Narrowing, Herniation 등)은 데이터 누수로 제외.
Sec 7 CNN을 피처 추출기로 활용하고, 판독 외 정형 정보(환자/스캐너)와 결합해
end-to-end 학습 없이 ML 분류기로 예측한다.

### 설정

| 항목 | 내용 |
|------|------|
| **CNN 피처** | Sec 7 `model_v2` 펜얼티미트 레이어 → **256차원** |
| **정형 피처** | IVD 위치 one-hot(6) + sex/num_vertebrae/num_discs/MagneticFieldStrength/SliceThickness/SpacingBetweenSlices(6) + Manufacturer one-hot(3) = **15차원** |
| **결합 피처** | CNN(256) + tabular(15) = **271차원** |
| **분류기** | Logistic Regression / Random Forest(n=300) / Gradient Boosting(n=200, lr=0.05) |
| **데이터** | 동일 split (Train 876 / Val 186 / Test 192) |

### Val 결과 비교

| 분류기 | Val Acc | Val Prec | Val Rec | Val F1 |
|--------|---------|----------|---------|--------|
| Logistic Regression | 0.6882 | 0.7182 | 0.7453 | 0.7315 |
| **Random Forest** | 0.7796 | 0.7826 | 0.8491 | **0.8145** |
| Gradient Boosting | 0.7419 | 0.7589 | 0.8019 | 0.7798 |
| *[Sec 7 CNN 단독]* | *0.8118* | *0.8034* | *0.8868* | *0.8430* |

### 결과 (최고 Val F1: Random Forest)

| Split | Accuracy | Precision | Recall | F1 |
|-------|----------|-----------|--------|----|
| **Val** | 0.7796 | 0.7826 | 0.8491 | 0.8145 |
| **Test** | 0.6823 | 0.6271 | 0.8132 | **0.7081** |

| 분류기 | Test Acc | Test Prec | Test Rec | Test F1 |
|--------|----------|-----------|----------|---------|
| Logistic Regression | 0.6458 | 0.6036 | 0.7363 | 0.6634 |
| Random Forest | 0.6823 | 0.6271 | 0.8132 | 0.7081 |
| Gradient Boosting | 0.6667 | 0.6195 | 0.7692 | 0.6863 |
| *[Sec 7 CNN 단독]* | *0.7969* | *0.7273* | *0.9362* | *0.8186* |

### 분석

- **모든 ML 분류기가 Sec 7 CNN 단독(0.8186)보다 낮음** — 정형 피처 결합의 효과 없음
- Val-Test 갭이 매우 큼 (RF: 0.8145 → 0.7081, Δ0.107) — CNN 단독(0.8430 → 0.8186, Δ0.024)보다 훨씬 불안정
- 원인 분석:
  - CNN 256-dim 피처는 BCE 손실로 학습된 표현 — RF/GB에 최적화된 피처가 아님
  - 876개 샘플에 271차원 입력 → 트리 계열 분류기의 과적합 가능성
  - 정형 피처 15차원(IVD 위치, 스캐너 정보)이 bulging 예측에 실질적 정보 추가 없음
- Feature importance 확인: CNN 피처(256개)가 상위 대부분 차지, 정형 피처의 기여 미미

---

## 종합 비교

| Section | 타겟 | 입력 | 정규화 | 증강 | 손실함수 | Test F1 |
|---------|------|------|--------|------|----------|---------|
| **4** | Herniation (6-label) | 전척추 224×224 | ✗ | ✗ | BCE | 0.0000 ❌ |
| **5** | Herniation (binary) | 디스크 크롭 128×128 | ✗ | ✗ | Focal Loss | 0.0000 ❌ |
| **6** | **Bulging** (binary) | bbox 크롭 → Resize 128×128 | ✗ | ✗ | BCE | 0.7885 ✅ |
| **7** | **Bulging** (binary) | bbox 크롭 → Resize 128×128 | Z-score | Flip+Rot | BCE | **0.8186** ✅ |
| **8** | **Bulging** (binary) | 등방 리샘플 → 중심 크롭 128×128 | ✗ | ✗ | BCE | 0.7805 |
| **9** | **Bulging** (binary) | 등방 리샘플 → 좌·중·우 3채널 128×128 | ✗ | ✗ | BCE | 0.7980 |
| **10** | **Bulging** (binary) | 등방 리샘플 → 좌·중·우 3채널 128×128 | Z-score | Flip+Rot | BCE | 0.7800 |
| **11** | **Bulging** (binary) | CNN(256) + tabular(15) = 271차원 | — | — | RF/GB/LR | 0.7081 |

**핵심 관찰**
1. Herniation(~5%)은 현재 규모에서 학습 자체가 불가 — 클래스 불균형이 결정적
2. 디스크 크롭(Section 5→6)은 타겟이 맞으면 효과적인 전처리
3. Z-score + 증강(Section 7)이 현재까지 최고 성능(Test F1 0.8186)
4. Section 8 등방 리샘플은 정규화·증강 없이는 Section 6 대비 성능 개선 없음 — 데이터셋 대부분이 이미 등방(sh≈sw≈0.625mm)이어서 효과 제한적
5. Section 9 멀티슬라이스는 Section 8 대비 Test F1 +0.018 향상 — 3D 공간 정보 활용의 효과, 특히 Recall 개선
6. Section 10 조합(Sec9+Sec7)은 기대와 달리 Test F1 0.7800으로 오히려 하락 — 3채널 입력 복잡도 + 증강의 조합이 현재 데이터 규모(876개)에서 학습 난이도를 높인 것으로 보임
7. Section 11 CNN+tabular ML 융합은 Test F1 0.7081로 최저 — CNN 피처를 ML 분류기에 넘기는 방식은 end-to-end 학습보다 열세; 환자/스캐너 정형 피처가 bulging 예측에 추가 정보 제공 못함
