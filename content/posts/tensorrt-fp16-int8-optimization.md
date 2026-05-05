---
title: "TensorRT FP16/INT8 최적화 실전 가이드 — 자동차 비전 검사 현장 적용기"
date: 2026-04-20
tags: ["TensorRT", "딥러닝", "추론최적화", "Edge AI", "YOLOv8", "FP16", "INT8"]
categories: ["AI 추론 엔진"]
description: "제조 현장 비전 검사 시스템에서 TensorRT FP16/INT8을 적용해 레이턴시를 3배 이상 줄인 과정. 부동소수점 이론부터 캘리브레이션 구현까지."
showToc: true
tocopen: true
---

## 왜 TensorRT인가

자동차 생산 라인에서 비전 검사 AI를 운영하다 보면 하나의 절대 법칙을 깨닫게 됩니다.

> **"사이클타임 안에 판정이 나오지 않으면 라인이 멈춘다."**

반도체·디스플레이 공장과 달리, 자동차 생산 라인에는 '**사이클타임(cycle time)**'이라는 절대적 시간 제약이 있습니다. 차체 한 대가 용접·도장·조립 라인을 통과하는 시간 안에 수십 개의 비전 검사를 완료해야 합니다. PyTorch로 학습한 모델을 그대로 배포하면 Python 오버헤드와 AutoGrad 메모리 때문에 최적 대비 **3~5배 느립니다.**

TensorRT는 NVIDIA가 제공하는 고성능 딥러닝 추론 최적화 라이브러리로, 이 문제를 해결합니다.

---

## GPU 소프트웨어 스택 이해

TensorRT를 이해하려면 먼저 GPU 소프트웨어 스택 전체를 봐야 합니다.

```
┌─────────────────────────────────────────────────────┐
│   애플리케이션 (Python 추론 서버 코드)               │
├─────────────────────────────────────────────────────┤
│   TensorRT / ONNX Runtime / PyTorch                 │
├─────────────────────────────────────────────────────┤
│   CUDA Runtime  (libcuda.so)                        │
├─────────────────────────────────────────────────────┤
│   NVIDIA 드라이버  (호스트 설치)                    │
├─────────────────────────────────────────────────────┤
│   GPU 하드웨어  (Tensor Core / CUDA Core)           │
└─────────────────────────────────────────────────────┘
```

PyTorch는 CUDA Runtime 위에서 동작하지만 AutoGrad 그래프, Python GIL, 동적 메모리 할당 등의 오버헤드가 있습니다. TensorRT는 이 중간 레이어들을 제거하고 **특정 GPU에 최적화된 커널**을 직접 실행합니다.

---

## 이론: 부동소수점 정밀도

### FP32 / FP16 / INT8의 차이

딥러닝 추론에서 사용하는 숫자 형식의 비트 구조:

```
FP32 (32비트):  [1 부호][8 지수][23 가수]
FP16 (16비트):  [1 부호][5 지수][10 가수]   ← 최대값: 65,504
BF16 (16비트):  [1 부호][8 지수][7 가수]    ← 지수 범위 = FP32와 동일
INT8 (8비트):   [-128 ~ 127 정수]
```

| 형식 | 비트 | 메모리 | 속도 | 정확도 손실 | 비전검사 권장 |
|------|------|--------|------|------------|--------------|
| FP32 | 32 | 기준 | 1× | 없음 | 개발·검증용 |
| TF32 | 19 | 기준 | 1.5× | 미미 | A100 자동 적용 |
| **FP16** | **16** | **50%** | **2~3×** | **0.1~0.3%** | **★ 현장 권장** |
| BF16 | 16 | 50% | 2~3× | 0.1~0.3% | 학습에 유리 |
| INT8 | 8 | 25% | 3~5× | 0.5~2% | 검증 후 적용 |
| INT4 | 4 | 12.5% | 4~8× | 2~5% | 극한 엣지 |

### FP16의 오버플로우 위험

FP16의 최대값은 **65,504**입니다. Softmax나 Sigmoid 내부의 지수 연산 결과가 이를 초과하면 `inf`(오버플로우) 또는 `0`(언더플로우)이 됩니다. TensorRT는 이를 자동 처리하지만, 일부 레이어에서 FP32를 유지해야 하는 경우가 있습니다.

```python
# 특정 레이어를 FP32로 고정 (TRT Python API)
for layer in network:
    if layer.name in HIGH_PRECISION_LAYERS:
        layer.precision = trt.DataType.FLOAT
        layer.set_output_type(0, trt.DataType.FLOAT)
```

---

## TensorRT가 빠른 이유: 3가지 핵심 최적화

### 1. 레이어 퓨전 (Layer Fusion)

```
Before fusion:
  Conv2D kernel → (GPU memory write)
  BatchNorm kernel → (GPU memory write)
  ReLU kernel → (GPU memory write)

After fusion:
  Conv2D + BN + ReLU  (단일 CUDA 커널, 중간 결과 레지스터에 유지)
```

커널 실행 오버헤드와 메모리 왕복이 3→1로 줄어듭니다. ResNet-50 기준으로 약 200개의 오퍼레이션이 퓨전 후 약 80개로 줄어드는 효과가 있습니다.

### 2. 커널 자동 선택 (Kernel Auto-tuning)

같은 Conv2D라도 입력 크기, 배치 크기, GPU 아키텍처에 따라 가장 빠른 CUDA 커널 구현이 다릅니다. TensorRT 빌드 시간이 긴 이유가 바로 이 **수천 개의 커널 조합을 실측 비교**하기 때문입니다.

```bash
# timingCacheFile로 커널 선택 결과를 재사용 (빌드 시간 단축)
trtexec --onnx=model.onnx \
        --saveEngine=model.engine \
        --fp16 \
        --timingCacheFile=timing.cache  # 두 번째부터 빌드 속도 대폭 향상
```

### 3. 정밀도 최적화

Tensor Core는 FP16 연산에서 FP32 대비 이론적으로 8×, INT8에서 16× 처리량을 가집니다 (NVIDIA A100 기준 312 TFLOPS FP16 vs 19.5 TFLOPS FP32).

---

## 실전: YOLOv8 TensorRT 변환

### FP16 엔진 빌드 — 두 가지 방법

**방법 1: Ultralytics 공식 API (권장)**

```python
from ultralytics import YOLO

model = YOLO('yolov8s.pt')
model.export(
    format='engine',
    half=True,       # FP16
    device=0,
    batch=1,
    workspace=4,     # GPU 워크스페이스 GB
    imgsz=640,
    simplify=True,   # onnxsim 적용
)
# → yolov8s.engine 생성
```

**방법 2: TensorRT Python API (세밀한 제어)**

```python
import tensorrt as trt

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def build_engine(onnx_path: str, engine_path: str,
                 fp16: bool = True, workspace_gb: int = 4,
                 input_shape: tuple = (1, 3, 640, 640)) -> None:
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, TRT_LOGGER)

    with open(onnx_path, 'rb') as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            raise RuntimeError('ONNX 파싱 실패')

    config = builder.create_builder_config()
    config.set_memory_pool_limit(
        trt.MemoryPoolType.WORKSPACE, workspace_gb * (1 << 30)
    )

    if fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    # 동적 배치 프로파일 설정
    profile = builder.create_optimization_profile()
    profile.set_shape(
        'input',
        (1,) + input_shape[1:],   # min
        input_shape,               # opt
        (8,) + input_shape[1:],   # max
    )
    config.add_optimization_profile(profile)

    engine_bytes = builder.build_serialized_network(network, config)
    with open(engine_path, 'wb') as f:
        f.write(engine_bytes)
    print(f'엔진 저장: {engine_path}')
```

### TRT 추론 클래스

```python
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np


class TRTInference:
    def __init__(self, engine_path: str):
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(TRT_LOGGER)
        with open(engine_path, 'rb') as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()

        self.inputs, self.outputs, self.bindings = [], [], []
        self.stream = cuda.Stream()

        for binding in self.engine:
            size = trt.volume(self.engine.get_binding_shape(binding))
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            self.bindings.append(int(device_mem))
            if self.engine.binding_is_input(binding):
                self.inputs.append({'host': host_mem, 'device': device_mem})
            else:
                self.outputs.append({'host': host_mem, 'device': device_mem})

    def infer(self, img: np.ndarray) -> np.ndarray:
        np.copyto(self.inputs[0]['host'], img.ravel())
        cuda.memcpy_htod_async(
            self.inputs[0]['device'], self.inputs[0]['host'], self.stream
        )
        self.context.execute_async_v2(
            bindings=self.bindings, stream_handle=self.stream.handle
        )
        cuda.memcpy_dtoh_async(
            self.outputs[0]['host'], self.outputs[0]['device'], self.stream
        )
        self.stream.synchronize()
        return self.outputs[0]['host']
```

---

## INT8 캘리브레이션

### 캘리브레이터 구현

INT8은 각 레이어의 활성화값 범위를 사전에 측정해 **스케일 인자(scale factor)**를 결정해야 합니다. 이를 **캘리브레이션**이라고 합니다.

```
FP32 활성화값 분포 측정 → 최적 클리핑 범위 결정 → INT8 스케일 인자 계산
                         (KL divergence 최소화)
```

```python
class ImageCalibrator(trt.IInt8EntropyCalibrator2):
    """
    IInt8EntropyCalibrator2: KL divergence 기반 — 비전 검사에 권장
    IInt8MinMaxCalibrator:   min/max 기반 — 이상값에 민감
    IInt8LegacyCalibrator:   구형 방식
    """
    def __init__(self, calib_dir: str, batch_size: int = 8,
                 input_shape: tuple = (3, 640, 640), cache_file: str = 'calib.cache'):
        super().__init__()
        self.batch_size = batch_size
        self.cache_file = cache_file
        c, h, w = input_shape

        self.files = sorted([
            os.path.join(calib_dir, f)
            for f in os.listdir(calib_dir)
            if f.lower().endswith(('.npy', '.png', '.jpg'))
        ])
        self.idx = 0

        # 페이지 잠금 메모리 (DMA 전송 최적화)
        self.device_input = cuda.mem_alloc(
            batch_size * c * h * w * np.dtype('float32').itemsize
        )
        self.host_buffer = cuda.pagelocked_empty(
            (batch_size, c, h, w), dtype=np.float32
        )

    def get_batch_size(self) -> int:
        return self.batch_size

    def get_batch(self, names):
        if self.idx + self.batch_size > len(self.files):
            return None

        for i in range(self.batch_size):
            img = cv2.imread(self.files[self.idx + i])
            img = cv2.resize(img, (640, 640))
            img = img[:, :, ::-1].astype(np.float32) / 255.0  # BGR→RGB, normalize
            self.host_buffer[i] = np.transpose(img, (2, 0, 1))

        cuda.memcpy_htod(self.device_input, self.host_buffer)
        self.idx += self.batch_size
        return [int(self.device_input)]

    def read_calibration_cache(self):
        if os.path.exists(self.cache_file):
            with open(self.cache_file, 'rb') as f:
                return f.read()

    def write_calibration_cache(self, cache):
        with open(self.cache_file, 'wb') as f:
            f.write(cache)
```

**캘리브레이션 데이터 선택 기준:**
- 최소 500장, 권장 1,000~5,000장
- **실제 라인 환경 이미지**로 구성 (ImageNet 사용 금지)
- 정상·불량 샘플을 모두 포함
- 조명 변동, 시간대별 조건 다양성 확보

---

## FP16 정확도 검증 스크립트

배포 전 반드시 FP32 대비 정확도를 비교하세요.

```python
import numpy as np
from tqdm import tqdm


def compare_precision(fp32_engine, fp16_engine, val_images, val_labels, threshold=0.5):
    results = {
        'fp32': {'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0},
        'fp16': {'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0},
    }

    for img, label in tqdm(zip(val_images, val_labels)):
        for name, engine in [('fp32', fp32_engine), ('fp16', fp16_engine)]:
            pred = engine.infer(img)
            is_defect_pred = pred['confidence'] > threshold
            is_defect_true = (label == 1)

            if is_defect_true and is_defect_pred:
                results[name]['tp'] += 1
            elif not is_defect_true and is_defect_pred:
                results[name]['fp'] += 1
            elif is_defect_true and not is_defect_pred:
                results[name]['fn'] += 1
            else:
                results[name]['tn'] += 1

    for name, r in results.items():
        precision = r['tp'] / (r['tp'] + r['fp'] + 1e-9)
        recall = r['tp'] / (r['tp'] + r['fn'] + 1e-9)
        f1 = 2 * precision * recall / (precision + recall + 1e-9)
        fpr = r['fp'] / (r['fp'] + r['tn'] + 1e-9)  # 오검출률
        fnr = r['fn'] / (r['fn'] + r['tp'] + 1e-9)  # 미검출률 — 비전검사 핵심 지표
        print(f"{name}: Precision={precision:.4f} Recall={recall:.4f} F1={f1:.4f} "
              f"FPR={fpr:.4f} FNR={fnr:.4f}")
```

> **비전 검사에서 FNR(미검출률)이 FPR(오검출률)보다 훨씬 중요합니다.** 불량을 놓치면 리콜 리스크지만, 정상을 불량으로 판정하면 이젝터만 더 작동합니다.

---

## 레이턴시 측정 — 올바른 방법

```python
import time
import numpy as np

def measure_latency(engine, input_data, warmup=10, iterations=200):
    # CUDA JIT 컴파일 워밍업 — 반드시 필요
    for _ in range(warmup):
        _ = engine.infer(input_data)

    import torch
    torch.cuda.synchronize()

    times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        _ = engine.infer(input_data)
        torch.cuda.synchronize()   # GPU 작업 완료까지 대기
        times.append((time.perf_counter() - t0) * 1000)

    times = np.array(times)
    print(f"  P50: {np.percentile(times, 50):.2f}ms")
    print(f"  P95: {np.percentile(times, 95):.2f}ms")
    print(f"  P99: {np.percentile(times, 99):.2f}ms")  # ← SLA 기준
    print(f"  Max: {times.max():.2f}ms")
    return times
```

P50만 보면 안 됩니다. **P99가 SLA를 만족하는지** 확인해야 합니다. 일반적으로 P99는 P50의 1.5~3배입니다.

---

## 실측 결과 (YOLOv8s, Jetson AGX Orin)

| 모드 | P50 레이턴시 | P99 레이턴시 | FPS | 메모리 | mAP 손실 |
|------|------------|------------|-----|--------|---------|
| PyTorch FP32 | 68ms | 82ms | 12 | 4.2GB | 기준 |
| ONNX Runtime | 46ms | 58ms | 17 | 2.1GB | <0.1% |
| **TRT FP16** | **21ms** | **24ms** | **42** | **1.1GB** | **<0.3%** |
| TRT INT8 | 12ms | 14ms | 71 | 0.6GB | 0.8% |

사이클타임 55ms 요구사항: FP32로는 불가능, **TRT FP16으로 처음 달성**.

---

## 파이프라인 단계별 레이턴시 배분

전체 55ms 안에 모든 단계가 들어와야 합니다.

```
단계                     목표     P99 실측
────────────────────────────────────────
카메라 캡처               5ms     3.2ms
이미지 전송 (GigE Vision)  3ms     2.1ms
GPU 메모리 복사           2ms     1.8ms
전처리 (GPU resize/norm)   5ms     4.3ms
TRT FP16 추론            30ms    24.1ms  ✓
후처리 + NMS              4ms     3.1ms
PLC 신호 전송             3ms     2.2ms
────────────────────────────────────────
합계                     52ms    40.8ms  ✓ (여유 14.2ms)
```

CUDA Stream으로 전처리와 추론을 비동기 오버랩하면 추가로 5~8ms를 더 줄일 수 있습니다.

---

## 엔진 버전 관리 컨벤션

```bash
# 파일명에 GPU 타입 + TRT 버전 + 정밀도를 명시
yolov8s_jetson-agx-orin_trt10.3_fp16_b1.engine
yolov8s_t4_trt10.3_int8_b1.engine
yolov8s_a100_trt10.3_fp16_b8.engine
```

엔진은 GPU 드라이버/TRT 버전이 바뀌면 재빌드가 필요합니다. 자동화하지 않으면 배포 시 실수가 발생합니다.

---

## 정리

| 상황 | 권장 전략 |
|------|---------|
| 시작점 / 정확도 중요 | TRT FP16 |
| 레이턴시 극한 / 정확도 손실 허용 가능 | TRT INT8 + 충분한 캘리브레이션 |
| Intel CPU 환경 | OpenVINO |
| 멀티카메라 스트리밍 | DeepStream + TRT |
| 개발·검증 단계 | ONNX Runtime (플랫폼 독립) |

---

## 참고 자료

- NVIDIA TensorRT Documentation: [docs.nvidia.com/deeplearning/tensorrt](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html)
- Ultralytics YOLOv8 Export: [docs.ultralytics.com/modes/export](https://docs.ultralytics.com/modes/export/)
- NVIDIA Jetson AGX Orin 스펙: [developer.nvidia.com/embedded/jetson-agx-orin](https://developer.nvidia.com/embedded/jetson-agx-orin)
- NVIDIA Tensor Core 성능 (A100): [Ampere Architecture Whitepaper](https://images.nvidia.com/aem-dam/en-zz/Solutions/data-center/nvidia-ampere-architecture-whitepaper.pdf)
- INT8 Inference with TensorRT: [NVIDIA Developer Blog](https://developer.nvidia.com/blog/int8-inference-autonomous-vehicles-tensorrt/)
- GigE Vision Standard: [AIA EMVA Standard](https://www.emva.org/standards-technology/genicam/genicam-downloads/)
