---
title: "LCD Mura 불량 분석과 Demura 알고리즘 — 광학 엔지니어 실무 노트"
date: 2026-03-28
tags: ["LCD", "Mura", "Demura", "광학설계", "디스플레이", "화질"]
categories: ["LCD 광학설계"]
description: "LCD 패널에서 Mura(표시 불균일)가 왜 발생하고, Demura 알고리즘으로 어떻게 보정하는지 광학설계 엔지니어 관점에서 설명합니다."
showToc: true
tocopen: true
---

## Mura란 무엇인가

LCD를 균일한 회색을 표시하도록 설정했을 때, 화면 일부가 밝거나 어둡게 보이는 현상을 **Mura**(무라)라고 합니다. 일본어로 "얼룩"을 뜻합니다.

Mura는 완성품 수율과 직결되기 때문에 디스플레이 제조에서 가장 중요한 품질 지표 중 하나입니다. 눈에 띄지 않는 수준으로 줄이거나, Demura(디무라) 보정으로 소프트웨어적으로 제거합니다.

---

## Mura의 발생 원인

Mura는 단일 원인이 아닌 **공정·재료·설계의 복합 결과**입니다.

### 1. BLU (Back Light Unit) 기원 Mura

```
LED 어레이 → 도광판 → 확산판 → 프리즘 시트 → LCD 패널
     ↑
  핫스팟 / 광량 불균일
```

- **LED 광량 편차**: LED 간 휘도 편차가 ±5% 이상이면 화면에서 보임
- **도광판 스크래치**: 물류·조립 중 발생하는 미세 스크래치
- **에어갭 불균일**: 프리즘 시트와 확산판 사이의 간격 변동

### 2. 액정 셀 기원 Mura

- **셀갭(Cell Gap) 불균일**: 두 유리 기판 사이 간격이 고르지 않으면 위상 지연이 달라져 투과율 차이 발생
- **러빙 불균일**: TN 모드에서 배향막 러빙 방향이 불균일하면 프리-틸트 각도 편차
- **스페이서 자국**: 셀갭 유지용 스페이서 주변 액정 배향 교란

### 3. 전극/구동 기원 Mura

- **ITO 저항 불균일**: 투명 전극의 면저항 분포가 고르지 않으면 전압 강하 차이
- **데이터 라인 커플링**: 인접 픽셀 데이터 라인 간 전기적 크로스토크

---

## Mura 분류 체계

| 분류 | 특징 | 주요 원인 |
|------|------|---------|
| **H-Mura** | 수평 띠 형태 | 게이트 라인 저항 불균일 |
| **V-Mura** | 수직 띠 형태 | 데이터 라인 저항 불균일 |
| **Spot Mura** | 원형/타원형 얼룩 | 스페이서, 이물질 |
| **Corner Dark** | 모서리 어두움 | BLU 프레임 광 누설 |
| **Edge Dark** | 가장자리 어두움 | 도광판 설계 |
| **Sparkle/Grain** | 미세 반짝임 | 확산판·AR 코팅 입자 산란 |

---

## 광학 측정 및 계측

Mura를 정량화하려면 계측 시스템이 필요합니다.

### JEITA Mura 측정 기준

JEITA ED-2522는 Mura를 정량화하는 업계 표준입니다.

```python
import numpy as np
from scipy import ndimage


def calculate_mura_index(luminance_map: np.ndarray, patch_size: int = 64) -> dict:
    """
    JEITA 기반 Mura 지수 계산
    
    Args:
        luminance_map: 휘도 측정값 (2D array, cd/m²)
        patch_size: 로컬 분석 패치 크기
    
    Returns:
        mura_metrics: 각종 Mura 지수
    """
    # 1. 전역 균일도 (Global Uniformity)
    L_mean = luminance_map.mean()
    L_max = luminance_map.max()
    L_min = luminance_map.min()

    global_uniformity = (L_max - L_min) / (L_max + L_min)  # 0이 완벽

    # 2. 로컬 대비 (Local Contrast) — 패치별 평균 대비 편차
    local_means = []
    rows, cols = luminance_map.shape
    for r in range(0, rows - patch_size, patch_size // 2):
        for c in range(0, cols - patch_size, patch_size // 2):
            patch = luminance_map[r:r+patch_size, c:c+patch_size]
            local_means.append(patch.mean())

    local_means = np.array(local_means)
    local_contrast = local_means.std() / local_means.mean()  # CV (변동계수)

    # 3. SDCM (Standard Deviation of Color Matching) 방식 Mura
    # 인간 시각 감도를 고려한 공간 필터 적용
    gaussian_blur = ndimage.gaussian_filter(luminance_map, sigma=10)
    residual = luminance_map - gaussian_blur
    sdcm_mura = residual.std() / L_mean

    return {
        'global_uniformity': global_uniformity,
        'local_contrast_cv': local_contrast,
        'sdcm_mura': sdcm_mura,
        'L_mean': L_mean,
        'L_max': L_max,
        'L_min': L_min,
    }
```

---

## Demura 알고리즘

Demura는 측정된 Mura를 역으로 보정하는 알고리즘입니다. 각 픽셀에 보정 값을 계산해 패널 내부 메모리(LUT)에 저장합니다.

### 개념

```
측정 휘도 L_meas(x,y) = L_target × G(x,y)
                              ↑
                       게인 불균일 (Mura 원인)

보정 목표: G_corr(x,y) = 1 / G(x,y)
→ 각 픽셀 입력 코드를 G_corr으로 스케일링
```

### 구현

```python
import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy.ndimage import gaussian_filter


class DemuraEngine:
    def __init__(self, target_luminance: float, max_correction_db: float = 1.0):
        self.L_target = target_luminance
        # 최대 보정량 제한 (너무 많이 보정하면 색도 변화)
        self.max_gain = 10 ** (max_correction_db / 20)
        self.min_gain = 10 ** (-max_correction_db / 20)

    def compute_correction_lut(
        self,
        measured_map: np.ndarray,    # 측정 휘도 맵
        smooth_sigma: float = 5.0,   # 보정 맵 스무딩 (측정 노이즈 제거)
    ) -> np.ndarray:
        """
        Demura 보정 LUT 계산
        
        Returns:
            gain_map: 각 픽셀 보정 게인 (1.0이 보정 없음)
        """
        # 1. 측정 노이즈 제거
        smoothed = gaussian_filter(measured_map, sigma=smooth_sigma)

        # 2. 정규화된 게인 맵 계산
        gain_map = self.L_target / smoothed

        # 3. 보정량 클리핑 (과보정 방지)
        gain_map = np.clip(gain_map, self.min_gain, self.max_gain)

        # 4. 에지 처리 — 패널 가장자리는 보정 약화 (액정 배향 교란 영역)
        edge_mask = self._create_edge_mask(gain_map.shape, edge_width=20)
        gain_map = 1.0 + (gain_map - 1.0) * edge_mask

        return gain_map

    def _create_edge_mask(self, shape: tuple, edge_width: int) -> np.ndarray:
        """가장자리로 갈수록 보정 강도를 줄이는 마스크"""
        rows, cols = shape
        mask = np.ones((rows, cols))

        # 코사인 창 함수로 부드러운 전환
        for i in range(edge_width):
            weight = 0.5 * (1 - np.cos(np.pi * i / edge_width))
            mask[i, :] = weight
            mask[-(i+1), :] = weight
            mask[:, i] = np.minimum(mask[:, i], weight)
            mask[:, -(i+1)] = np.minimum(mask[:, -(i+1)], weight)

        return mask

    def apply_correction(
        self,
        input_frame: np.ndarray,
        gain_map: np.ndarray,
    ) -> np.ndarray:
        """
        입력 프레임에 Demura 보정 적용
        
        Note: 실제 패널에서는 LUT가 하드웨어에 내장되어 픽셀 단위로 적용됨
        """
        corrected = input_frame.astype(np.float32) * gain_map
        return np.clip(corrected, 0, 255).astype(np.uint8)


# 사용 예시
demura = DemuraEngine(target_luminance=500.0)  # 500 cd/m² 목표

# 측정된 휘도 맵 (실제로는 계측 장비에서 취득)
measured = np.load('luminance_map_gray128.npy')  # shape: (1080, 1920)

gain_map = demura.compute_correction_lut(measured)
print(f"보정 범위: {gain_map.min():.3f} ~ {gain_map.max():.3f}")
print(f"보정 후 예상 균일도: {(gain_map * measured).std() / (gain_map * measured).mean() * 100:.2f}%")
```

---

## 폐루프 보정 시스템

단 한 번 보정으로 끝나지 않습니다. 패널은 에이징(aging)으로 특성이 변하고, 온도에 따라 휘도가 달라집니다.

```
측정 → Demura 계산 → LUT 업로드 → 재측정 → 수렴 판정
  ↑                                              ↓
  └──────────── 수렴 안 됨 ──────────────────────┘
```

**수렴 조건 예시:**
- 전역 균일도 < 5%
- 로컬 대비 CV < 2%
- 최대 반복 횟수: 5회

---

## 양산 현장에서 Mura 관리

### 인라인 검사 vs. 오프라인 검사

| 방식 | 장점 | 단점 |
|------|------|------|
| **인라인 검사** | 실시간 피드백, 100% 검사 | 속도 제약, 조명 환경 통제 어려움 |
| **오프라인 샘플링** | 정밀 측정, 다양한 조건 | 지연 피드백, 전수 불가 |

양산에서는 두 방식을 병행합니다. 인라인으로 큰 불량을 잡고, 오프라인 샘플링으로 공정 트렌드를 모니터링합니다.

### Mura 원인 분석 시트 (실무 템플릿)

```
패널 ID: ___________  측정일: ___________  측정자: ___________

1. Mura 유형: [ ] H-Mura  [ ] V-Mura  [ ] Spot  [ ] Corner  [ ] Edge
2. 발생 위치: x=_____ y=_____ (패널 좌상단 기준 mm)
3. 휘도 편차: _____% (정상 대비)
4. 검출 그레이 레벨: [ ] 전체  [ ] 저계조(0~64)  [ ] 중계조  [ ] 고계조

의심 공정:
[ ] BLU 조립  [ ] 셀 공정  [ ] 편광판 부착  [ ] 모듈 조립

물성 확인:
- 셀갭 측정값: _____ μm (목표: _____ μm)
- 편광판 부착 압력: _____ kPa
- 큐어링 온도/시간: _____°C / _____ min
```

---

## 핵심 정리

1. **Mura는 복합 원인** — BLU, 셀갭, 전극 중 어디서 시작됐는지 데이터 없이 추정 금지
2. **계측 정확도가 Demura 품질을 결정** — 노이즈 많은 측정값으로 보정하면 오히려 악화
3. **보정량에 상한 필요** — 과보정은 색도 편차를 유발
4. **에이징 고려 필수** — 납품 시 완벽해도 6개월 후 다시 Mura가 나타날 수 있음

다음 글에서는 LCD 광학 스택 설계와 BLU 필름 구성 최적화를 다루겠습니다.
