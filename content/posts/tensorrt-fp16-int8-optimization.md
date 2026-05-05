---
title: "TensorRT FP16/INT8 최적화 실전 가이드 — 자동차 비전 검사 현장 적용기"
date: 2026-04-20
tags: ["TensorRT", "딥러닝", "추론최적화", "Edge AI", "YOLOv8"]
categories: ["AI 추론 엔진"]
description: "제조 현장 비전 검사 시스템에서 TensorRT FP16/INT8을 적용해 레이턴시를 3배 이상 줄인 과정을 공유합니다."
showToc: true
tocopen: true
---

## 왜 TensorRT인가

자동차 생산 라인에서 비전 검사 AI를 운영하다 보면 하나의 절대 법칙을 깨닫게 됩니다.

> **"사이클타임 안에 판정이 나오지 않으면 라인이 멈춘다."**

차체 한 대가 용접·도장·조립 라인을 통과하는 시간은 정해져 있고, 그 안에 수십 개의 비전 검사를 완료해야 합니다. PyTorch로 학습한 모델을 그대로 배포하면 Python 오버헤드와 AutoGrad 메모리 때문에 최적 대비 **3~5배 느립니다.**

TensorRT는 이 문제를 해결하는 NVIDIA의 추론 최적화 라이브러리입니다.

---

## TensorRT가 빠른 이유

TensorRT의 핵심 최적화는 세 가지입니다.

**1. 레이어 퓨전 (Layer Fusion)**  
`Conv → BatchNorm → ReLU` 같은 연속 연산을 하나의 CUDA 커널로 통합합니다. 커널 실행 오버헤드가 3개에서 1개로 줄어듭니다.

**2. 커널 자동 선택 (Kernel Auto-tuning)**  
GPU에서 사용 가능한 여러 CUDA 커널 구현 중 **현재 GPU에서 가장 빠른 것**을 빌드 시점에 선택합니다. 이 때문에 엔진은 GPU별로 다르며, T4용 엔진을 A100에서 실행하면 오류가 납니다.

**3. 정밀도 최적화 (Precision Calibration)**  
FP32 → FP16 → INT8로 연산 정밀도를 낮춰 처리 속도를 높입니다.

---

## 실전: YOLOv8 TensorRT 변환

### 1단계: FP16 엔진 빌드 (권장 시작점)

```bash
# trtexec로 빌드 — 가장 간단한 방법
trtexec \
  --onnx=yolov8s.onnx \
  --saveEngine=yolov8s_fp16.engine \
  --fp16 \
  --shapes=input:1x3x640x640 \
  --timingCacheFile=timing.cache \
  --verbose
```

Ultralytics 공식 방법도 있습니다:

```python
from ultralytics import YOLO

model = YOLO('yolov8s.pt')
model.export(
    format='engine',
    half=True,       # FP16
    device=0,
    batch=1,
    workspace=4,     # GB
    imgsz=640,
)
# → yolov8s.engine 생성
```

### 2단계: INT8 캘리브레이션

FP16으로 부족할 때 INT8을 씁니다. 핵심은 **캘리브레이션 데이터셋**입니다.

```python
import tensorrt as trt
import numpy as np
import os

class ImageCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, calib_dir, batch_size=8, input_shape=(3, 640, 640)):
        super().__init__()
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.files = [
            os.path.join(calib_dir, f)
            for f in os.listdir(calib_dir)
            if f.endswith('.npy')
        ]
        self.idx = 0
        c, h, w = input_shape
        self.device_input = trt.cuda.alloc(
            batch_size * c * h * w * np.dtype('float32').itemsize
        )

    def get_batch_size(self):
        return self.batch_size

    def get_batch(self, names):
        if self.idx + self.batch_size > len(self.files):
            return None
        batch = np.stack([
            np.load(self.files[self.idx + i])
            for i in range(self.batch_size)
        ]).astype(np.float32)
        trt.cuda.memcpy_htod(self.device_input, batch)
        self.idx += self.batch_size
        return [int(self.device_input)]

    def read_calibration_cache(self):
        if os.path.exists('calib.cache'):
            with open('calib.cache', 'rb') as f:
                return f.read()

    def write_calibration_cache(self, cache):
        with open('calib.cache', 'wb') as f:
            f.write(cache)
```

---

## 레이턴시 비교 결과

실제 자동차 비전 검사 라인에서 측정한 수치입니다 (YOLOv8s, Jetson AGX Orin):

| 모드 | 레이턴시 | FPS | 정확도 손실 |
|------|---------|-----|------------|
| PyTorch FP32 | 82ms | 12 | 기준 |
| ONNX Runtime | 58ms | 17 | < 0.1% |
| TRT FP16 | **24ms** | **42** | < 0.3% |
| TRT INT8 | **14ms** | **71** | 0.5~1.5% |

사이클타임 55ms 요구사항을 FP32로는 맞출 수 없었고, TRT FP16으로 처음으로 달성했습니다.

---

## 실전에서 자주 만나는 함정

### 함정 1: 엔진은 GPU마다 다르다
개발 서버(A100)에서 빌드한 엔진을 Jetson에 올리면 오류가 납니다. **배포 타겟 GPU에서 직접 빌드**해야 합니다. CI/CD 파이프라인에 Jetson용 빌드 단계를 따로 넣어야 하는 이유입니다.

### 함정 2: 드라이버/TRT 버전 업그레이드 = 재빌드
GPU 드라이버나 TensorRT 버전이 바뀌면 엔진을 다시 빌드해야 합니다. 엔진 파일은 반드시 GPU 타입과 TRT 버전을 파일명에 태깅해서 관리하세요.

```bash
# 파일명 컨벤션 예시
yolov8s_jetson-agx-orin_trt10.3_fp16.engine
yolov8s_t4_trt10.3_int8.engine
```

### 함정 3: CUDA 워밍업 없이 레이턴시 측정
첫 번째 추론은 JIT 컴파일로 느립니다. 측정 전 최소 10회 워밍업을 돌리세요.

```python
# 올바른 레이턴시 측정
for _ in range(10):         # 워밍업
    _ = engine.infer(dummy)

torch.cuda.synchronize()
times = []
for _ in range(100):
    t0 = time.perf_counter()
    output = engine.infer(input_data)
    torch.cuda.synchronize()
    times.append(time.perf_counter() - t0)

p50 = np.percentile(times, 50)
p99 = np.percentile(times, 99)
print(f"P50: {p50*1000:.1f}ms | P99: {p99*1000:.1f}ms")
```

P50만 보면 안 됩니다. 제조 라인에서는 **P99 레이턴시가 SLA를 지키는지** 확인해야 합니다.

---

## 파이프라인 단계별 레이턴시 배분

전체 55ms 안에 모든 단계가 들어와야 합니다:

```
카메라 캡처 (5ms) → 전처리 CPU (8ms) → TRT 추론 (35ms) → 후처리 NMS (4ms) → PLC 출력 (3ms)
                              ↕ CUDA Stream 비동기 오버랩
```

CUDA Stream으로 전처리와 추론을 겹치면 전체 레이턴시를 5~10% 더 줄일 수 있습니다.

---

## 정리

| 상황 | 권장 |
|------|------|
| 시작점, 정확도 중요 | TRT FP16 |
| 레이턴시 극한, 정확도 손실 허용 | TRT INT8 |
| Intel CPU 환경 | OpenVINO |
| 멀티카메라 스트리밍 | DeepStream + TRT |

TensorRT 적용만으로도 레이턴시를 3배 이상 줄일 수 있습니다. 다음 글에서는 ONNX 변환 과정과 흔한 오류 해결법을 다루겠습니다.
