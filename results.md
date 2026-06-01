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

## Section 5-1 — SpiderDiscCNN_GAP + Focal Loss + Stratified Split (Herniation)

### 배경

Sec 5(SpiderDiscCNN 8.4M + Focal Loss)의 실패 원인이 모델 과용량에도 있다고 판단,
이후 bulging 실험에서 사용할 경량 모델(SpiderDiscCNN_GAP)과 동일한 구조로 herniation을 재시도.
클래스 비율을 보존하는 stratified split과 최대 200 epoch 학습을 추가 적용.

### 설정

| 항목 | 내용 |
|------|------|
| **타겟** | Disc Herniation (이진분류) |
| **샘플 단위** | 디스크 1개 = 샘플 1개 |
| **데이터** | 209명, Train 876 / Val 186 / Test 192 (stratified split, seed=0) |
| **양성 비율** | Train 5.6% / Val 5.4% / Test 6.2% |
| **입력** | bbox crop + T.Resize(128×128), 정규화 없음 |
| **모델** | **SpiderDiscCNN_GAP** — Conv(1→32→64) + GAP + Dropout(0.3) + FC(64→1), **18,881 params** |
| **손실함수** | Focal Loss (alpha=0.25, gamma=2.0) |
| **옵티마이저** | Adam lr=1e-4, Early Stopping patience=15 |
| **최대 에폭** | 200 |

### 학습 로그

```
 Epoch | Train Loss |  Val Loss |  Val F1 | Wait
----------------------------------------------------
     1 |     2.6045 |    1.1877 |  0.0000 |    0
    10 |     0.3148 |    0.1437 |  0.0000 |    0
    20 |     0.0603 |    0.0408 |  0.0000 |    1
    30 |     0.0319 |    0.0184 |  0.0000 |    0
    50 |     0.0165 |    0.0165 |  0.0000 |    2
    70 |     0.0164 |    0.0159 |  0.0000 |    2
   100 |     0.0144 |    0.0157 |  0.0000 |    8
   120 |     0.0134 |    0.0154 |  0.0000 |    0
   130 |     0.0129 |    0.0158 |  0.0000 |   10
Early stop @ epoch 135
```

### 결과

| Split | Accuracy | Precision | Recall | F1 |
|-------|----------|-----------|--------|----|
| **Val** | 0.9462 | 0.0000 | 0.0000 | 0.0000 |
| **Test** | 0.9375 | 0.0000 | 0.0000 | 0.0000 |

### 분석

- 경량 모델 + Focal Loss + Stratified Split + 135 epoch 학습에도 Val F1 = 0.0000 유지
- Val loss는 꾸준히 감소(1.19 → 0.015)했으나, 이는 loss 함수가 양성 샘플을 완전히 무시하고
  음성 샘플의 loss만 줄이는 방향으로 수렴한 것 (trivial solution의 다른 형태)
- **결론**: Herniation 양성 샘플 ~70개(5.7%)로는 현재 데이터 규모에서 학습이 근본적으로 불가능

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

## Section 12 — 경량 모델: 2 Conv + Global Average Pooling (Sec 8 전처리)

### 배경

Section 6~11의 SpiderDiscCNN(8.4M 파라미터)은 876개 학습 데이터 대비 파라미터가 과도하게 많아
(샘플당 파라미터 1:9,680) epoch 5 이내에 train/val loss가 역전되는 조기 과적합 발생.
모델을 데이터 규모에 맞게 대폭 축소하고, Flatten+FC 대신 GAP으로 교체.

### 설정

| 항목 | 내용 |
|------|------|
| **타겟** | Disc Bulging (동일) |
| **데이터** | Sec 8과 동일 split + DataLoader (Train 876 / Val 186 / Test 192) |
| **전처리** | Sec 8과 동일 (등방 리샘플 TARGET_SP=0.75mm/px + 중심 크롭 128×128) |
| **모델** | **SpiderDiscCNN_GAP** — Conv(1→32→64) × 2 + GAP + Dropout(0.3) + FC(64→1) |
| **파라미터** | **19,073개** (Sec 6~11의 8.4M 대비 444배 감소, 샘플당 1:22) |
| **손실함수** | BCEWithLogitsLoss |
| **옵티마이저** | Adam lr=1e-4, Early Stopping patience=10 |
| **정규화·증강** | 없음 |

### Sec 6~11 vs Sec 12 모델 구조 비교

| | SpiderDiscCNN (Sec 6~11) | **SpiderDiscCNN_GAP (Sec 12)** |
|---|---|---|
| Conv 블록 | 3개 (1→32→64→128) | **2개 (1→32→64)** |
| 피처맵 처리 | Flatten 32,768 → FC(256) | **GAP → 64** |
| Dropout | 0.5 | **0.3** |
| 총 파라미터 | 8,482,241 | **19,073** |
| 샘플당 파라미터 | 1 : 9,680 | **1 : 22** |

### 학습 로그

```
Epoch   1: train=0.8305  val=0.7939
Epoch   5: train=0.7095  val=0.6906
Epoch  10: train=0.6757  val=0.6578
Epoch  20: train=0.6511  val=0.6165
Epoch  30: train=0.6328  val=0.5962
Epoch  40: train=0.6208  val=0.5880
Epoch  50: train=0.6216  val=0.5767
(Early stopping 미발동 — 50 epoch 내내 val loss 감소)
```

### 결과

| Split | Accuracy | Precision | Recall | F1 |
|-------|----------|-----------|--------|----|
| **Val** | 0.8172 | 0.8214 | 0.8679 | **0.8440** |
| **Test** | 0.6823 | 0.6336 | 0.8646 | 0.7313 |

### 분석

- **Val F1 0.8440 — 전 섹션 최고**, 50 epoch 내내 train > val 유지 (과적합 없음)
- **Test F1 0.7313 — Val 대비 갭 0.1127** (다른 섹션들 갭 0.02~0.04의 3~5배)
- 학습 곡선이 정상적(val < train 유지)임에도 Test 성능이 낮은 것은 모델 과적합이 아닌 **Val-Test set 분포 불일치** 가능성을 강하게 시사
- Val(31명)과 Test(32명)의 소규모 환자 집합 간 자연적 분포 편차 + **동일 Val set으로 12개 섹션 선택을 반복한 메타 수준의 val 과적합** 누적 효과로 해석
- → K-fold 교차검증으로 성능 추정치의 신뢰도 검증 필요

---

## Section 13 — 3-Fold Cross-Validation Baseline

### 배경

Section 12까지의 모든 실험은 동일한 단일 Val set(31명)으로 모델 선택을 반복해왔다.
12개 섹션에 걸친 반복 선택이 메타 수준의 Val 과적합을 누적시킨 것으로 판단되며,
이를 해소하기 위해 환자 레벨 Stratified 3-Fold CV를 도입한다.

CV 설정은 보고서 베이스라인인 **Sec 6-1** 구성과 동일하게 유지한다.

> **Sec 6-1 참고 (단일 hold-out, seed=0)**
> - 모델: SpiderDiscCNN_GAP (18,881 params), 전처리: bbox crop → Resize(128×128), BCE, patience=10
> - Val F1: 0.8085 / Test F1: 0.6979 / Val-Test 갭: 0.111

### 설정

| 항목 | 내용 |
|------|------|
| **타겟** | Disc Bulging (이진분류) |
| **샘플 단위** | 디스크 1개 = 샘플 1개 |
| **전체 데이터** | 환자 209명, 디스크 샘플 1,254개 |
| **입력** | bbox crop + T.Resize(128×128), 정규화 없음 (Sec 6-1과 동일) |
| **모델** | SpiderDiscCNN_GAP — Conv(1→32→64) + GAP + Dropout(0.3) + FC(64→1), **18,881 params** |
| **손실함수** | BCEWithLogitsLoss |
| **옵티마이저** | Adam lr=1e-4, Early Stopping patience=10 |
| **CV** | StratifiedKFold(n_splits=3, shuffle=True, seed=0), 환자 레벨 stratify |
| **Stratify 기준** | 해당 환자에 ≥1 bulging disc이면 양성 |
| **결과 저장** | `results/13_cv_baseline/` — `metrics.csv`, `loss_curves.pdf`, `confusion_matrices.pdf` |

### 학습 로그 (요약)

```
Fold 1 | Train 139 pids (834 smp) | Val 70 pids (420 smp)
  Ep 1  : train=39.8701  val=4.2890  F1=0.6821
  Ep 10 : train= 3.2312  val=0.9530  F1=0.6681
  Ep 30 : train= 0.6487  val=0.5870  F1=0.7077
  Ep 50 : train= 0.6104  val=0.5569  F1=0.7329  → early stop

Fold 2 | Train 139 pids (834 smp) | Val 70 pids (420 smp)
  Ep 1  : train=19.0608  val=4.9329  F1=0.0000
  Ep 10 : train= 1.3977  val=0.6526  F1=0.7586
  Ep 40 : train= 0.5896  val=0.5599  F1=0.7927
  Ep 50 : train= 0.5755  val=0.5508  F1=0.7746  → early stop

Fold 3 | Train 140 pids (840 smp) | Val 69 pids (414 smp)
  Ep 1  : train=21.0673  val=9.2877  F1=0.7222
  Ep 10 : train= 1.9116  val=0.7700  F1=0.6591
  Ep 40 : train= 0.5968  val=0.5644  F1=0.7683
  Ep 50 : train= 0.5852  val=0.5640  F1=0.7769  → early stop
```

### 결과

| Fold | Acc | Precision | Recall | F1 |
|------|-----|-----------|--------|----|
| 1 | 0.7119 | 0.7477 | 0.7186 | 0.7329 |
| 2 | 0.7548 | 0.7729 | 0.7763 | 0.7746 |
| 3 | 0.7150 | 0.6973 | 0.8761 | 0.7765 |
| **Mean** | **0.7272** | **0.7393** | **0.7903** | **0.7613** |
| **Std** | 0.0195 | 0.0315 | 0.0650 | 0.0201 |

### 분석

- 3-Fold CV Val F1: **0.7613 ± 0.0201**
- Fold 2가 F1 0.7746으로 가장 높고, Fold 1이 0.7329로 가장 낮음 — fold 간 편차 존재
- 단일 hold-out(Sec 6-1) Val F1 0.8085 대비 낮지만, 이는 CV가 전체 데이터를 val로 순환하므로
  더 보수적인 추정치임 (메타 val 과적합 없음)
- Recall 편차(std=0.065)가 Precision(0.031)보다 큰 것은 fold별 양성 분포 차이에 기인
- **Sec 6-1 단일 split Val F1(0.8085) vs CV Mean F1(0.7613)의 0.047 차이**가
  메타 수준 val 과적합의 크기를 보여줌

---

## Section 14 — 3-Fold CV + Isotropic Resampling (Sec 8 방식)

### 배경

Sec 13(bbox crop + Resize 베이스라인)과 동일 모델·CV 설정에
**Sec 8의 등방 리샘플링 + 중심 크롭 전처리**를 적용하여
물리적 스케일 통일이 CV 기반 성능에 미치는 영향을 측정한다.

- 입력 파이프라인: `scipy.ndimage.zoom` → TARGET_SP=0.75mm/px 등방 리샘플 → 디스크 중심 기준 128×128 중심 크롭 + 제로패딩
- Sec 13과 달리 종횡비 왜곡 없음, 환자 간 물리적 FOV(96mm×96mm) 일정
- 정규화·증강 없음 — 전처리 효과만 단독 격리

### 설정

| 항목 | 내용 |
|------|------|
| **타겟** | Disc Bulging (이진분류) |
| **전체 데이터** | 환자 209명, 디스크 샘플 1,254개 |
| **입력** | `scipy.ndimage.zoom` → 0.75mm/px 등방 리샘플 → 디스크 중심 128×128 중심 크롭 + 제로패딩 |
| **정규화·증강** | 없음 |
| **모델** | SpiderDiscCNN_GAP (18,881 params) |
| **손실함수** | BCEWithLogitsLoss |
| **옵티마이저** | Adam lr=1e-4, Early Stopping patience=10 |
| **CV** | StratifiedKFold(n_splits=3, shuffle=True, seed=0) |
| **결과 저장** | `results/14_cv_iso/` |

### 결과

| Fold | Acc | Precision | Recall | F1 |
|------|-----|-----------|--------|----|
| 1 | 0.7500 | 0.7739 | 0.7706 | 0.7722 |
| 2 | 0.7286 | 0.7545 | 0.7412 | 0.7478 |
| 3 | 0.7053 | 0.7393 | 0.7393 | 0.7393 |
| **Mean** | **0.7280** | **0.7559** | **0.7504** | **0.7531** |
| **Std** | 0.0182 | 0.0142 | 0.0143 | 0.0140 |

### Sec 13 대비 변화량

| 지표 | Sec 13 Mean | Sec 14 Mean | Δ |
|------|-------------|-------------|---|
| Accuracy  | 0.7272 | 0.7280 | +0.0008 |
| Precision | 0.7393 | 0.7559 | **+0.0166** |
| Recall    | 0.7903 | 0.7504 | −0.0399 |
| F1        | 0.7613 | 0.7531 | −0.0082 |
| F1 Std    | 0.0201 | 0.0140 | −0.0061 (더 안정) |

### 분석

- CV Mean F1 **0.7531 ± 0.0140** — Sec 13(0.7613 ± 0.020) 대비 소폭 하락(−0.0082)
- Precision은 +0.017 향상됐으나 Recall이 −0.040 하락 → 모델이 더 보수적으로 예측
- F1 표준편차는 0.014로 Sec 13(0.020)보다 작아 fold 간 편차는 감소
- 등방 리샘플링이 CV 기반에서도 성능을 개선하지 못함 — Sec 8 단일 hold-out 결과(Sec 6 대비 −0.008)와 일치
- 원인: 데이터셋의 in-plane spacing이 이미 대부분 등방(sh≈sw≈0.625mm)에 가까워 0.75mm/px 리샘플의 실질 효과 제한적
- bbox crop → T.Resize 방식이 중심 크롭 + 제로패딩보다 오히려 성능이 좋은 이유는
  bbox 기반 크롭이 디스크 주변 조직만 타이트하게 포함하여 모델에 더 유의미한 지역 특징을 제공하기 때문으로 추정

---

## Section 15 — 3-Fold CV + N4 Bias Field Correction + Foreground Normalization + Augmentation

### 배경

Section 14(등방 리샘플, CV F1=0.7531)에서 물리적 스케일 통일만으로는 성능 향상이 없었다.
Sec 15는 척추 MRI 특화 전처리 2단계를 추가한다:
1. **N4ITK Bias Field Correction** — 코일 근접도 차이에서 오는 posterior↔anterior 신호 불균일 보정. 2D mid-sagittal 슬라이스 단위로 미리 계산 후 `data/n4_2d/{pid}.npz`에 캐시.
2. **Foreground Normalization** — crop 내 ≠0 픽셀의 mean/std로 전체 crop 정규화.

학습 조건은 등방 리샘플(0.75mm/px) + 중심 크롭(128×128 + 제로패딩)을 유지하고,
Train-only MRI-safe 증강을 추가:
- `RandomAffine(degrees=10, translate=(0.08, 0.08))`
- `ElasticTransform(alpha=25.0, sigma=4.0)`
- Gaussian Noise (std=0.02)

### 설정

| 항목 | 내용 |
|------|------|
| **타겟** | Disc Bulging (이진분류) |
| **전체 데이터** | 환자 209명, 디스크 샘플 1,254개 |
| **입력** | N4 보정 2D 슬라이스 → 0.75mm/px 등방 리샘플 → 중심 크롭 128×128 + 제로패딩 |
| **정규화** | Foreground(≠0) 픽셀 mean/std → 전체 crop |
| **증강 (Train only)** | RandomAffine(±10°, ±8%) + ElasticTransform + Gaussian Noise(std=0.02) |
| **모델** | SpiderDiscCNN_GAP (18,881 params) |
| **손실함수** | BCEWithLogitsLoss |
| **옵티마이저** | Adam lr=1e-4, Early Stopping patience=10 |
| **CV** | StratifiedKFold(n_splits=3, shuffle=True, seed=0) |
| **결과 저장** | `results/15_cv_n4_norm/` |

### 결과

| Fold | Acc | Precision | Recall | F1 |
|------|-----|-----------|--------|----|
| 1 | 0.5500 | 0.5505 | 0.9913 | 0.7079 |
| 2 | 0.6071 | 0.5845 | 0.9561 | 0.7255 |
| 3 | 0.5700 | 0.5680 | 1.0000 | 0.7245 |
| **Mean** | **0.5757** | **0.5676** | **0.9825** | **0.7193** |
| **Std** | 0.0237 | 0.0139 | 0.0190 | 0.0081 |

### Section 14 대비 변화량

| 지표 | Sec 14 Mean | Sec 15 Mean | Δ |
|------|-------------|-------------|---|
| Accuracy  | 0.7280 | 0.5757 | **−0.1523** |
| Precision | 0.7559 | 0.5676 | −0.1883 |
| Recall    | 0.7504 | 0.9825 | +0.2321 |
| F1        | 0.7531 | 0.7193 | −0.0338 |

### 분석

- Recall이 모든 fold에서 0.99~1.00에 도달 → 모델이 거의 모든 샘플을 **양성(Bulging)으로 예측**하는 Trivial Positive Solution에 수렴
- Precision 0.57 ≈ 전체 양성 비율(55.3%)과 유사 → 기준 확률 수준의 정밀도
- Accuracy 0.576은 Section 14(0.728)는 물론 무조건 양성 예측(55.3%)보다도 낮음 → 학습 실패
- 원인 분석:
  1. **ElasticTransform + Gaussian Noise 조합**이 N4 보정·정규화로 이미 약해진 신호를 추가 왜곡해 학습 신호 파괴
  2. **Foreground 정규화** 후 zero-padding 영역이 큰 음의 값(≈ -mean/std)을 가지게 되어 GAP이 이를 의미 있는 특징으로 잘못 학습했을 가능성
  3. 증강이 너무 공격적 → 실제 disc 형태 특징이 훼손된 상태로 학습
- 전처리·증강을 한꺼번에 모두 추가한 것이 문제 → 각 요소를 분리해 효과를 검증하는 Ablation이 필요

---

## Section 16 — 3-Fold CV + Multi-Slice 3-Channel Input

### 설정

| 항목 | 내용 |
|------|------|
| **모델** | SpiderDiscCNN_GAP_3ch (19,457 params, in_channels=3) |
| **입력** | 등방 리샘플(0.75mm/px) → 중심 크롭 128×128, L/C/R 슬라이스 (물리적 ±4mm) |
| **정규화** | 없음 |
| **증강** | 없음 |
| **손실함수** | BCEWithLogitsLoss |
| **옵티마이저** | Adam (lr=1e-4), patience=10 |
| **최대 에폭** | 200 |
| **평가** | StratifiedKFold(n=3, seed=0), 환자 레벨 stratification |
| **결과 저장** | `results/16_cv_3ch/` |

### 결과

| Fold | Accuracy | Precision | Recall | F1 |
|------|----------|-----------|--------|----|
| 1 | 0.7524 | 0.7433 | 0.8398 | 0.7886 |
| 2 | 0.7405 | 0.7196 | 0.8553 | 0.7816 |
| 3 | 0.6932 | 0.7277 | 0.7308 | 0.7292 |
| **Mean** | **0.7287** | **0.7302** | **0.8086** | **0.7665** |
| **Std** | 0.0255 | 0.0099 | 0.0554 | 0.0265 |

### Section 14 대비 비교

| 지표 | Sec 14 (1ch) | Sec 16 (3ch) | Δ |
|------|--------------|--------------|---|
| Accuracy  | 0.7280 | 0.7287 | +0.0007 |
| Precision | 0.7559 | 0.7302 | −0.0257 |
| Recall    | 0.7504 | 0.8086 | +0.0582 |
| F1        | 0.7531 | 0.7665 | **+0.0134** |

### 분석

- F1=0.767로 Sec 14(0.753) 대비 +0.013 향상 → L/C/R 3채널로 3D 공간 정보 활용이 소폭 기여
- Fold 1·2 F1(0.789, 0.782)은 안정적, Fold 3 F1(0.729)이 유독 낮아 Recall std=0.055로 큰 편
  - Fold 3 검증 집합의 환자 구성(스캐너 종류, 극단적 픽셀 스페이싱 등)에 따른 편차로 추정
- Precision std=0.010으로 매우 안정적 → 양성 예측의 정밀도는 fold 간 일관성 높음
- 200 에폭까지 손실이 계속 하강, 학습률 스케줄러 추가 시 추가 개선 가능성

---

## Section 17 — 3-Fold CV + N4 Bias Field Correction + Foreground Normalization (증강 제거)

### 목적

Section 15 (N4 + Norm + 증강)에서 trivial positive solution이 발생한 원인을 분리하기 위한 ablation.
증강만 제거하고 나머지 설정은 Sec 15와 동일하게 유지.

### 설정

| 항목 | 내용 |
|------|------|
| **모델** | SpiderDiscCNN_GAP_b0 (18,881 params) |
| **입력** | N4 보정 2D 슬라이스 → 등방 리샘플(0.75mm/px) → 중심 크롭 128×128 + 제로패딩 |
| **정규화** | Foreground (≠0) 픽셀 mean/std |
| **증강** | 없음 |
| **손실함수** | BCEWithLogitsLoss |
| **옵티마이저** | Adam (lr=1e-4), patience=10 |
| **최대 에폭** | 200 |
| **평가** | StratifiedKFold(n=3, seed=0), 환자 레벨 stratification |
| **결과 저장** | `results/17_cv_n4_noaug/` |

### 결과

| Fold | Accuracy | Precision | Recall | F1 |
|------|----------|-----------|--------|----|
| 1 | 0.6857 | 0.6725 | 0.8355 | 0.7452 |
| 2 | 0.6976 | 0.6906 | 0.8026 | 0.7424 |
| 3 | 0.7464 | 0.7247 | 0.8889 | 0.7985 |
| **Mean** | **0.7099** | **0.6959** | **0.8423** | **0.7620** |
| **Std** | 0.0262 | 0.0217 | 0.0355 | 0.0258 |

### Section 14 / 15 대비 비교

| 실험 | N4 | Norm | Aug | Mean F1 | Std |
|------|----|------|-----|---------|-----|
| Sec 14 (iso resample) | ✗ | ✗ | ✗ | 0.7531 | 0.014 |
| Sec 15 (N4+Norm+Aug) | ✓ | ✓ | ✓ | 0.7193 ⚠️ | 0.008 |
| **Sec 17 (N4+Norm, no aug)** | ✓ | ✓ | ✗ | **0.7620** | 0.026 |

### 분석

- 증강 제거 후 trivial positive 문제 즉시 해소 → **증강이 Sec 15 실패의 직접 원인임을 확인**
- F1=0.762은 Sec 14(0.753) 대비 +0.009 향상 → N4 보정·foreground 정규화 자체는 유효한 전처리
- Recall=0.842로 안정화, Precision=0.696으로 Sec 15(0.569) 대비 크게 회복
- Fold 3의 F1(0.799)이 Fold 1·2(0.745, 0.742)보다 유독 높아 fold간 편차(std=0.026)가 이전 실험보다 큰 편 — 환자 구성 변동에 따른 분산

---

## 종합 비교

| Section | 모델 | 입력 | 정규화 | 증강 | 평가 방식 | CV/Test F1 |
|---------|------|------|--------|------|-----------|---------|
| **4** | SpiderCNN (8.4M) | 전척추 224×224 | ✗ | ✗ | Hold-out | 0.0000 ❌ |
| **5** | SpiderDiscCNN (8.4M) | 디스크 크롭 128×128 | ✗ | ✗ | Hold-out | 0.0000 ❌ |
| **6** | SpiderDiscCNN (8.4M) | bbox 크롭 → Resize 128×128 | ✗ | ✗ | Hold-out | 0.7885 |
| **7** | SpiderDiscCNN (8.4M) | bbox 크롭 → Resize 128×128 | Z-score | Flip+Rot | Hold-out | **0.8186** |
| **8** | SpiderDiscCNN (8.4M) | 등방 리샘플 → 중심 크롭 128×128 | ✗ | ✗ | Hold-out | 0.7805 |
| **9** | SpiderDiscCNN3 (8.4M) | 등방 리샘플 → 좌·중·우 3채널 128×128 | ✗ | ✗ | Hold-out | 0.7980 |
| **10** | SpiderDiscCNN3 (8.4M) | 등방 리샘플 → 좌·중·우 3채널 128×128 | Z-score | Flip+Rot | Hold-out | 0.7800 |
| **11** | RF/GB/LR | CNN(256) + tabular(15) | — | — | Hold-out | 0.7081 |
| **12** | SpiderDiscCNN_GAP (19K) | 등방 리샘플 → 중심 크롭 128×128 | ✗ | ✗ | Hold-out | 0.7313 ⚠️ |
| **13** | SpiderDiscCNN_GAP (19K) | bbox 크롭 → Resize 128×128 | ✗ | ✗ | **3-Fold CV** | 0.7613 ± 0.020 |
| **14** | SpiderDiscCNN_GAP (19K) | 등방 리샘플(0.75mm) → 중심 크롭 128×128 | ✗ | ✗ | **3-Fold CV** | 0.7531 ± 0.014 |
| **15** | SpiderDiscCNN_GAP (19K) | N4 → 등방 리샘플 → 중심 크롭 128×128 | Foreground | Affine+Elastic+Noise | **3-Fold CV** | 0.7193 ± 0.008 ⚠️ |
| **16** | SpiderDiscCNN_GAP_3ch (19K) | 등방 리샘플 → 중심 크롭 128×128, L/C/R 3채널 | ✗ | ✗ | **3-Fold CV** | 0.7665 ± 0.027 |
| **17** | SpiderDiscCNN_GAP (19K) | N4 → 등방 리샘플 → 중심 크롭 128×128 | Foreground | 없음 | **3-Fold CV** | **0.7620 ± 0.026** |

> Sec 6-1 (보고서 베이스라인, 단일 hold-out): Val F1=0.8085 / Test F1=0.6979

**핵심 관찰**
1. Herniation(~5%)은 현재 규모에서 학습 자체가 불가 — 클래스 불균형이 결정적
2. 디스크 크롭(Section 5→6)은 타겟이 맞으면 효과적인 전처리
3. Z-score + 증강(Section 7)이 단일 hold-out 기준 최고 성능(Test F1 0.8186)
4. Section 8 등방 리샘플은 정규화·증강 없이는 Section 6 대비 성능 개선 없음 — 데이터셋 대부분이 이미 등방(sh≈sw≈0.625mm)이어서 효과 제한적
5. Section 9 멀티슬라이스는 Section 8 대비 Test F1 +0.018 향상 — 3D 공간 정보 활용 효과
6. Section 10 조합(Sec9+Sec7)은 Test F1 0.7800으로 오히려 하락 — 입력 복잡도 증가가 현재 데이터 규모에서 학습 난이도를 높임
7. Section 11 CNN+tabular 융합은 Test F1 0.7081로 최저 — 정형 피처의 bulging 예측 기여 없음
8. Section 12 GAP 경량 모델: Val F1 0.8440(최고)이나 Val-Test 갭 0.113 — **단일 split 기반 평가 신뢰도 한계 노출**
9. Section 13은 3-Fold CV로 단일 split 분산 문제를 해소, 이후 모든 실험의 기본 평가 방식으로 사용
10. Section 14 등방 리샘플(CV 기반)도 Sec 13 대비 F1 −0.008로 개선 없음 — 단일 hold-out(Sec 8 vs Sec 6)과 동일한 패턴. 데이터셋 자체가 거의 등방이라 리샘플 효과 제한적
11. Section 15(N4+정규화+증강 복합)는 Recall≈1.0, Precision≈0.57의 Trivial Positive Solution으로 학습 실패 — ElasticTransform+Noise가 학습 신호를 파괴
12. Section 16(3채널 멀티슬라이스)은 F1=0.767로 Sec 14 대비 +0.013 향상 — L/C/R 공간 정보 활용의 소폭 기여 확인. 단 Fold 3 Recall이 낮아 Recall std=0.055로 fold 간 편차가 큼
13. Section 17(증강 제거 ablation)에서 trivial positive 즉시 해소. F1=0.762(Sec 14 대비 +0.009)로 N4+정규화의 순수 효과 확인 — 증강 설계의 중요성 재확인
14. Sec 16·17 모두 Sec 13 베이스라인(0.7613)을 소폭 상회하나 개선폭이 제한적 — 데이터 규모(209명)가 더 복잡한 입력·전처리의 효과를 흡수하기에 부족한 것으로 추정
