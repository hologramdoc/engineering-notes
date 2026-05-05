---
title: "LCD Mura 불량 분석과 Demura 알고리즘 — 광학 이론부터 폐루프 보정까지"
date: 2026-03-28
tags: ["LCD", "Mura", "Demura", "광학설계", "디스플레이", "화질", "BLU", "셀갭", "균일도보정"]
categories: ["LCD 광학설계"]
description: "LCD Mura의 물리적 발생 원인, 계측 이론, Demura 알고리즘 구현, 폐루프 보정 시스템 설계까지. LCD 광학설계 엔지니어 실무 노트."
showToc: true
tocopen: true
---

## 광학설계 엔지니어의 핵심 철학

> **"광학 문제는 대부분 단일 원인이 아니라 중첩 원인이다."**
>
> "설계는 계산만으로 끝나지 않고 측정, 해석, 보정, 공정 피드백으로 닫혀야 한다."

Mura 하나를 보더라도 빠르게 가설화할 수 있어야 합니다.

```
Mura 발생
   ├── Cell gap 불균일?
   ├── 편광판 라미네이션 응력?
   ├── BLU 조도 불균일?
   ├── 알고리즘 오검?
   └── 시야각 의존 현상?
```

원인군을 분리할 수 있는 **최소 실험**을 설계하는 것이 핵심입니다.

---

## LCD 광학 스택 구조

빛이 LED에서 눈까지 오는 경로를 이해해야 Mura의 원인을 찾을 수 있습니다.

```
눈
 ↑
 │  편광판 (Polarizer, 상판)
 │
 │  ← 액정 셀 (LC Cell) →  [전압 인가 → 액정 배향 변화 → 위상 지연 변화]
 │      ┌──────────────────────────────────────────┐
 │      │  컬러 필터 (CF)    R G B R G B R G B    │ ← 위판 유리
 │      │  ITO 공통 전극 (Vcom)                    │
 │      │  액정 레이어 (Cell Gap: 3~5 μm)          │ ← 셀갭 불균일 → Mura
 │      │  ITO 픽셀 전극                           │
 │      │  TFT 어레이                              │ ← TFT 특성 편차 → Mura
 │      └──────────────────────────────────────────┘ ← 하판 유리
 │  편광판 (Polarizer, 하판)
 │
 │  BLU (Back Light Unit)
 │      확산판 (Diffuser)
 │      프리즘 시트 × 2 (BEF)
 │      확산판 (Diffuser)
 │      도광판 (LGP, Light Guide Plate)      ← 도광판 불균일 → Mura
 │      반사 시트
 │      LED 어레이 (엣지 타입 또는 직하 타입) ← LED 편차 → Hot-spot Mura
```

---

## 이론: Mura 발생 메커니즘

### 1. 셀갭(Cell Gap) 불균일과 위상 지연

LCD의 투과율은 두 편광판 사이를 통과하는 빛의 **위상 지연(retardation)**에 의해 결정됩니다.

Malus 법칙과 Jones Matrix 표현:

```
투과율 T = sin²(Γ/2)

여기서 Γ = 위상 지연 = (2π/λ) × Δn × d

Δn: 액정 복굴절률 (ordinary - extraordinary index)
d: 셀갭 (Cell Gap)
λ: 파장
```

셀갭 `d`가 위치에 따라 달라지면 `Γ`가 달라지고 → **투과율 차이 = Mura**가 됩니다.

예를 들어 FFS 모드에서 목표 셀갭 3.5μm, Δn=0.085일 때:

```
정상: T = sin²(π × 0.085 × 3.5 / 0.550) ≈ sin²(1.693) ≈ 0.89
+0.1μm 편차: T = sin²(π × 0.085 × 3.6 / 0.550) ≈ 0.91 → ΔT = 0.02 (2%)
```

2% 투과율 차이는 육안으로 **확실히 보입니다.** 셀갭 공차가 왜 ±0.05μm 이내여야 하는지의 이유입니다.

### 2. 편광판 응력 복굴절 (Stress Birefringence)

편광판이나 위상지연판에 기계적 응력이 가해지면 재료의 굴절률이 방향에 따라 달라집니다. 라미네이션 압력, 베젤 체결 응력, 열팽창 차이가 주원인입니다.

```python
# 응력 복굴절에 의한 추가 위상 지연 계산
def stress_birefringence_retardation(stress_mpa: float, thickness_um: float,
                                      stress_optic_coeff: float = 2.5e-12) -> float:
    """
    C: 광탄성 계수 (Pa^-1) — 유리 약 2.5e-12, 폴리카보네이트 약 80e-12
    σ: 응력 (Pa)
    d: 두께 (m)
    R = C × σ × d  (nm 단위)
    """
    stress_pa = stress_mpa * 1e6
    thickness_m = thickness_um * 1e-6
    retardation_m = stress_optic_coeff * stress_pa * thickness_m
    return retardation_m * 1e9  # nm 단위

# 예시: 1 MPa 응력, 500μm 유리
r = stress_birefringence_retardation(1.0, 500)
print(f"응력 복굴절 위상 지연: {r:.2f} nm")
```

### 3. BLU 광량 불균일

LED 어레이의 휘도 편차가 있으면 도광판을 통과한 후에도 공간적 불균일이 남습니다.

```
LED 휘도 편차 분포:  μ ± σ (제조 공차)
도광판 확산 효과:    Gaussian spreading σ_LGP
필름 스택 균일화:    추가 확산

최종 불균일 ≈ σ_LED / (√(2π) × σ_LGP) × f(필름조합)
```

직하형(Direct) BLU는 확산 거리(OD, Optical Distance)를 충분히 확보해야 LED 핫스팟이 보이지 않습니다.

---

## Mura 분류와 원인 분리 실험

| 유형 | 형태 | 주요 원인 | 분리 실험 |
|------|------|---------|---------|
| **H-Mura** | 수평 줄무늬 | 게이트 라인 저항 편차 | 게이트 방향 회전 후 재측정 |
| **V-Mura** | 수직 줄무늬 | 데이터 라인 저항 편차 | 단색 패턴으로 위치 확인 |
| **Spot Mura** | 원형/타원 | 스페이서, 이물질, 기포 | Cross-polar 촬영 |
| **Corner Dark** | 모서리 어둠 | BLU 광 누설, 베젤 차광 | BLU 단독 점등 |
| **Edge Dark** | 가장자리 어둠 | 도광판 설계, 입사각 | 입사면 방향별 비교 |
| **Sparkle** | 미세 반짝임 | 확산판 입자 산란, AR 코팅 | 시야각별 관찰 |
| **G-Mura** | 그라데이션 | 셀갭 경사 (기판 처짐) | 간섭계 측정 |

### Mura 원인 분리 결정 트리

```
Mura 발견
    │
    ├─[BLU off 후 사라짐?] → YES → BLU 기원 (LED, LGP, 필름)
    │
    ├─[다른 패턴에서도 같은 위치?] → YES → 셀 기원 (셀갭, 전극)
    │
    ├─[압력 인가 시 변화?] → YES → 응력 복굴절 (편광판, 기구)
    │
    ├─[온도 변화 시 이동?] → YES → 열팽창 관련
    │
    └─[시야각 의존?] → YES → 액정 배향 또는 위상지연 문제
```

---

## 광학 계측 시스템

### 계측 장비 구성

```
테스트 패턴 생성기 (Gray 0~255)
        │
        ▼
   LCD 패널 (DUT)
        │
        ▼ 빛
   카메라 기반 계측 시스템
   ┌─────────────────────────┐
   │  텔레센트릭 렌즈         │ ← 시야각 오차 제거
   │  CCD/CMOS 센서 (16bit)  │ ← 다이나믹 레인지 확보
   │  휘도 교정               │ ← NIST traceable 기준
   └─────────────────────────┘
        │
        ▼
   Konica Minolta CA-410 / 
   Radiant ProMetric 등 2D 색채계
```

### JEITA ED-2522 기반 Mura 정량화

```python
import numpy as np
from scipy import ndimage
from scipy.signal import convolve2d


def calculate_mura_metrics(luminance_map: np.ndarray) -> dict:
    """
    JEITA ED-2522 준거 Mura 지수 계산

    Args:
        luminance_map: 2D 휘도 측정값 (cd/m²)

    Returns:
        dict: 전역 균일도, 로컬 균일도, SDCM Mura 지수
    """
    L = luminance_map.astype(np.float64)
    L_mean = L.mean()
    L_max, L_min = L.max(), L.min()

    # 1. 전역 균일도 (IEC 62341-6-4 기준)
    global_uniformity_pct = (L_max - L_min) / (L_max + L_min) * 100

    # 2. 로컬 균일도 — 9점(3×3 그리드) 측정
    rows, cols = L.shape
    grid_r = np.linspace(rows * 0.1, rows * 0.9, 3).astype(int)
    grid_c = np.linspace(cols * 0.1, cols * 0.9, 3).astype(int)
    nine_points = [L[r, c] for r in grid_r for c in grid_c]
    nine_pt_uniformity = (max(nine_points) - min(nine_points)) / np.mean(nine_points) * 100

    # 3. SDCM (Standard Deviation Contrast Matching) Mura
    # 인간 시각 공간 주파수 특성 반영: 저주파 성분이 더 잘 보임
    # Gaussian 필터로 저주파 배경 추출
    sigma_px = min(rows, cols) * 0.02   # 화면 크기의 2%
    background = ndimage.gaussian_filter(L, sigma=sigma_px)
    residual = L - background           # 배경 대비 잔차 = 로컬 Mura
    sdcm_mura_pct = residual.std() / L_mean * 100

    # 4. 웨버 대비 (Weber Contrast) — 심리물리학 기준
    weber_contrast = (L - L_mean) / L_mean

    return {
        'L_mean_cdm2': L_mean,
        'L_max_cdm2': L_max,
        'L_min_cdm2': L_min,
        'global_uniformity_pct': global_uniformity_pct,
        'nine_pt_uniformity_pct': nine_pt_uniformity,
        'sdcm_mura_pct': sdcm_mura_pct,
        'weber_contrast_max': weber_contrast.max(),
        'weber_contrast_min': weber_contrast.min(),
    }


def mura_visibility_threshold(luminance_cdm2: float) -> float:
    """
    인간 시각의 Mura 시인성 임계값 추정
    Weber 법칙: ΔL/L = k (k ≈ 0.01~0.03 for sustained viewing)
    """
    k = 0.015   # 지속 관찰 조건
    return luminance_cdm2 * k
```

---

## Demura 알고리즘

### 개념과 수식

```
목표: 위치 (x,y)에서 측정 휘도 = 목표 휘도
      L_meas(x,y) = L_target

현실: L_meas(x,y) = L_target × G(x,y)
      여기서 G(x,y)는 위치별 게인 불균일 (Mura 원인)

보정: 입력 코드 I(x,y)를 I_corr(x,y) = I / G(x,y)로 스케일링
      → L_corr(x,y) = L_target × G(x,y) × (1/G(x,y)) = L_target ✓
```

### Demura 엔진 구현

```python
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.interpolate import RectBivariateSpline


class DemuraEngine:
    """
    LCD 패널 Demura 보정 LUT 생성기

    실제 패널에서는 생성된 LUT를 DDI(Display Driver IC)의
    감마/오프셋 레지스터에 기록하거나 별도 보정 메모리에 저장합니다.
    """

    def __init__(self,
                 target_luminance: float,
                 max_correction_db: float = 1.0,
                 smooth_sigma: float = 5.0):
        self.L_target = target_luminance
        # 최대 보정량 ±1dB = 약 ±12% — 과보정 시 색도 변화 유발
        self.max_gain = 10 ** (max_correction_db / 20)
        self.min_gain = 10 ** (-max_correction_db / 20)
        self.smooth_sigma = smooth_sigma

    def compute_gain_map(self, measured_map: np.ndarray) -> np.ndarray:
        """
        보정 게인 맵 계산

        Args:
            measured_map: 균일 입력 코드에서 측정한 휘도 맵 (cd/m²)

        Returns:
            gain_map: 픽셀별 보정 게인 (1.0 = 보정 없음)
        """
        # 1. 측정 노이즈 제거 (Gaussian smoothing)
        smoothed = gaussian_filter(measured_map.astype(np.float64), sigma=self.smooth_sigma)

        # 2. 게인 계산: 목표 / 실측
        gain_map = self.L_target / np.clip(smoothed, smoothed.mean() * 0.1, None)

        # 3. 보정량 클리핑 (과보정 방지)
        gain_map = np.clip(gain_map, self.min_gain, self.max_gain)

        # 4. 가장자리 페이드아웃
        edge_mask = self._create_cosine_edge_mask(gain_map.shape, edge_ratio=0.05)
        gain_map = 1.0 + (gain_map - 1.0) * edge_mask

        return gain_map

    def _create_cosine_edge_mask(self, shape: tuple, edge_ratio: float = 0.05) -> np.ndarray:
        """
        가장자리로 갈수록 보정 강도를 0으로 줄이는 코사인 마스크.
        액정 배향 교란이 심한 패널 가장자리에서 과보정을 방지합니다.
        """
        rows, cols = shape
        edge_r = int(rows * edge_ratio)
        edge_c = int(cols * edge_ratio)

        mask = np.ones((rows, cols))

        # 상하단 코사인 페이드
        fade_r = 0.5 * (1 - np.cos(np.pi * np.arange(edge_r) / edge_r))
        mask[:edge_r, :]  = fade_r[:, np.newaxis]
        mask[-edge_r:, :] = fade_r[::-1, np.newaxis]

        # 좌우 코사인 페이드
        fade_c = 0.5 * (1 - np.cos(np.pi * np.arange(edge_c) / edge_c))
        mask[:, :edge_c]  = np.minimum(mask[:, :edge_c],  fade_c[np.newaxis, :])
        mask[:, -edge_c:] = np.minimum(mask[:, -edge_c:], fade_c[::-1][np.newaxis, :])

        return mask

    def generate_lut_per_gray(self, gray_measurements: dict) -> dict:
        """
        그레이 레벨별 보정 LUT 생성

        Args:
            gray_measurements: {gray_level: luminance_map} — 예: {64: map64, 128: map128, 255: map255}

        Returns:
            lut: {gray_level: gain_map}
        """
        lut = {}
        for gray, meas_map in gray_measurements.items():
            # 그레이별 목표 휘도 (감마 2.2 가정)
            target_L = self.L_target * (gray / 255.0) ** 2.2
            engine = DemuraEngine(target_L, self.max_gain, self.smooth_sigma)
            lut[gray] = engine.compute_gain_map(meas_map)
        return lut

    def evaluate_correction(self, original_map: np.ndarray, gain_map: np.ndarray) -> dict:
        """보정 전후 균일도 지표 비교"""
        corrected_map = original_map * gain_map

        def uniformity(m):
            return (m.max() - m.min()) / (m.max() + m.min()) * 100

        def cv(m):
            return m.std() / m.mean() * 100

        return {
            'before_global_uniformity': uniformity(original_map),
            'after_global_uniformity':  uniformity(corrected_map),
            'before_cv_pct':  cv(original_map),
            'after_cv_pct':   cv(corrected_map),
            'improvement_pct': (1 - cv(corrected_map) / cv(original_map)) * 100,
        }
```

---

## 폐루프 보정 시스템

단 한 번의 보정으로 끝나지 않습니다. 재측정 → 재보정 → 수렴 판정을 반복합니다.

```
┌─────────────┐
│ 초기 측정    │  Gray 32/64/128/192/255 패턴별 2D 휘도 맵 취득
└──────┬──────┘
       ▼
┌─────────────┐
│ Demura 계산  │  그레이별 게인 맵 생성 → LUT 생성
└──────┬──────┘
       ▼
┌─────────────┐
│ LUT 업로드   │  DDI 레지스터 또는 보정 메모리에 기록
└──────┬──────┘
       ▼
┌─────────────┐
│ 재측정       │  보정 적용 후 동일 패턴 재측정
└──────┬──────┘
       ▼
┌─────────────┐     수렴 기준:
│ 수렴 판정    │  → 전역 균일도 < 5%
└──────┬──────┘  → SDCM Mura < 2%
       │         → 최대 반복: 5회
  수렴? NO → 루프 반복
       │ YES
       ▼
   완료 (LUT 저장)
```

```python
def run_demura_loop(measure_fn, upload_lut_fn,
                    target_uniformity_pct: float = 5.0,
                    max_iter: int = 5) -> bool:
    """
    measure_fn: () → luminance_map (계측 장비 연동)
    upload_lut_fn: (gain_map) → None (DDI 연동)
    """
    engine = DemuraEngine(target_luminance=500.0)

    for iteration in range(max_iter):
        meas_map = measure_fn()
        metrics = calculate_mura_metrics(meas_map)

        print(f"[Iter {iteration+1}] 전역 균일도: {metrics['global_uniformity_pct']:.2f}%")

        if metrics['global_uniformity_pct'] < target_uniformity_pct:
            print(f"수렴 완료 (iter={iteration+1})")
            return True

        gain_map = engine.compute_gain_map(meas_map)
        upload_lut_fn(gain_map)

    print(f"경고: {max_iter}회 반복 후 미수렴. 하드웨어 원인 검토 필요.")
    return False
```

---

## 양산 현장 관리 체계

### 인라인 vs 오프라인 검사

```
라인 흐름:  셀 공정 → BLU 조립 → 모듈 조립 → 에이징 → 검사 → 출하

인라인 AOI (Automatic Optical Inspection):
    └── 100% 검사, 실시간 피드백, 대형 Mura/이물/스크래치 검출
    └── 속도 제약: 1장/3~5초

오프라인 2D 색채계 측정 (샘플링):
    └── 전수 불가, 정밀 계측, Demura 데이터 생성
    └── 그레이 레벨별 다각도 측정, 기준기 교정 포함
```

### SPC (Statistical Process Control) 적용

```python
import numpy as np


class MuraSPC:
    """
    Mura 지수에 대한 통계적 공정 관리 (Shewhart 관리도)
    UCL = μ + 3σ (상한 관리 한계)
    LCL = μ - 3σ (하한 관리 한계) — 음수면 0으로 처리
    """
    def __init__(self, warmup_samples: int = 25):
        self.history = []
        self.warmup = warmup_samples
        self.UCL = None
        self.mean = None
        self.sigma = None

    def update(self, mura_pct: float) -> str:
        self.history.append(mura_pct)

        if len(self.history) == self.warmup:
            self._set_control_limits()
            return "INIT"

        if self.UCL is None:
            return "WARMUP"

        if mura_pct > self.UCL:
            return "OOC"    # Out of Control — 공정 이상
        if self._nelson_rule_2(mura_pct):
            return "TREND"  # Nelson Rule 2: 연속 9점이 중심선 한쪽
        return "OK"

    def _set_control_limits(self):
        arr = np.array(self.history)
        self.mean = arr.mean()
        self.sigma = arr.std()
        self.UCL = self.mean + 3 * self.sigma

    def _nelson_rule_2(self, current: float) -> bool:
        """Nelson Rule 2: 최근 9점이 모두 평균 초과 or 이하"""
        if len(self.history) < 9:
            return False
        recent = self.history[-9:]
        return all(x > self.mean for x in recent) or all(x < self.mean for x in recent)
```

---

## 핵심 정리

| 주제 | 핵심 포인트 |
|------|-----------|
| **Mura 원인** | 단일 원인 아님 — BLU, 셀갭, 전극, 응력 복합 작용 |
| **계측 우선** | 노이즈 많은 측정값으로 Demura하면 오히려 악화 |
| **보정량 한계** | ±1dB 초과 보정은 색도 변화 유발 |
| **에이징 고려** | 납품 시 완벽해도 6개월 후 재발 가능 |
| **폐루프 필수** | 1회 보정으로 끝내지 말고 수렴까지 반복 |

---

## 참고 자료

- JEITA ED-2522: *Measuring Methods for Mura of FPD* — Japan Electronics and Information Technology Industries Association
- IEC 62341-6-4: *OLED displays — Part 6-4: Measurements of optical and electro-optical parameters*
- Konoshima, S. et al. (2004). *"Mura (Nonuniformity) and Its Evaluation in Flat Panel Displays"*, SID Symposium Digest
- Weber, E.H. (1834). *De Pulsu, Resorptione, Auditu et Tactu* — Weber 법칙 원전
- Born, M. & Wolf, E. (1999). *Principles of Optics*, 7th Edition, Cambridge University Press — 편광과 위상 지연 이론
- Yariv, A. & Yeh, P. (2007). *Photonics: Optical Electronics in Modern Communications*, Oxford University Press — Jones Matrix 이론
- SID (Society for Information Display): [sid.org](https://www.sid.org) — 디스플레이 학술 자료
- ICDM (International Committee for Display Metrology): [icdm-sid.org](http://www.icdm-sid.org) — 계측 표준
