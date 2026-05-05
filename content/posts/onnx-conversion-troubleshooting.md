---
title: "PyTorch → ONNX → TensorRT 변환 완벽 가이드 — 이론과 실패 없이 넘어가는 법"
date: 2026-04-15
tags: ["ONNX", "TensorRT", "PyTorch", "모델변환", "추론최적화", "그래프최적화"]
categories: ["AI 추론 엔진"]
description: "ONNX 그래프 구조 이론부터 PyTorch 변환, 그래프 최적화, TensorRT 배포까지. 현장에서 자주 만나는 오류 TOP 5와 해결법 포함."
showToc: true
tocopen: true
cover:
  image: "https://images.unsplash.com/photo-1555949963-ff9fe0c870eb?w=1260&h=630&fit=crop&auto=format"
  alt: "Python 코드 — ONNX 모델 변환"
  relative: false
---

## ONNX란 무엇인가

ONNX(Open Neural Network Exchange)는 Microsoft와 Facebook이 2017년 공동 제안한 딥러닝 모델 교환 표준입니다. 서로 다른 프레임워크로 학습된 모델을 하나의 표준 포맷으로 직렬화하고, 다양한 런타임에서 실행할 수 있게 합니다.

```
┌─────────────────────────────────────────────────────────────────┐
│  학습 프레임워크                                                 │
│  PyTorch · TensorFlow · MXNet · PaddlePaddle                   │
└─────────────────┬───────────────────────────────────────────────┘
                  │  torch.onnx.export() / tf2onnx
                  ▼
┌─────────────────────────────────────────────────────────────────┐
│  ONNX  (.onnx — Protocol Buffer 직렬화 형식)                    │
│  노드(Operator) + 엣지(Tensor) + 초기화값(Weight)               │
└─────────────────┬───────────────────────────────────────────────┘
                  │  변환
          ┌───────┼───────────┬───────────────┐
          ▼       ▼           ▼               ▼
       TensorRT  OpenVINO   CoreML        ONNX Runtime
      (NVIDIA)  (Intel)    (Apple)   (검증·프로토타이핑)
```

### ONNX 파일 구조

ONNX 파일은 Protocol Buffer 형식으로 저장된 계산 그래프입니다.

```
ONNX Model
├── ir_version         — ONNX IR 버전
├── opset_imports      — 사용된 opset 버전 목록
│   └── version: 17    ← TRT 8.x/9.x 호환 권장
├── graph
│   ├── node[]         — 연산자 노드 (Conv, BN, ReLU, ...)
│   ├── input[]        — 그래프 입력 텐서
│   ├── output[]       — 그래프 출력 텐서
│   └── initializer[]  — 가중치 (학습된 파라미터)
└── metadata_props     — 모델 메타데이터
```

---

## 이론: ONNX 변환 두 가지 방식

PyTorch는 모델을 ONNX로 변환할 때 두 가지 방식을 지원합니다.

### 방식 1: Trace (기본값)

```python
# torch.onnx.export 내부적으로 torch.jit.trace 사용
dummy_input = torch.randn(1, 3, 640, 640)
torch.onnx.export(model, dummy_input, 'model.onnx')
```

dummy_input으로 한 번 실행하면서 실행된 경로(trace)를 ONNX 그래프로 기록합니다.

**장점:** 간단, 대부분의 모델에 적용 가능  
**단점:** `if/for` 분기가 고정됨 — dummy_input 경로만 기록

### 방식 2: Script

```python
# torch.jit.script로 전체 제어 흐름 분석
scripted = torch.jit.script(model)
torch.onnx.export(scripted, dummy_input, 'model.onnx')
```

**장점:** 동적 제어 흐름 처리 가능  
**단점:** 일부 Python 문법 지원 안 됨

**선택 기준:**
- 일반 CNN/Transformer → Trace
- `if label == 'train': ...` 같은 분기 → Script 또는 분기 제거

---

## 변환 파이프라인

```
PyTorch (.pt)
     ↓  torch.onnx.export (opset=17)
ONNX Raw (.onnx)
     ↓  onnxsim (그래프 단순화)
ONNX Simplified (.onnx)
     ↓  onnx.checker.check_model
ONNX Verified
     ↓  onnxruntime (수치 검증)
ONNX Validated
     ↓  trtexec / TRT Python API
TensorRT Engine (.engine)
```

---

## 1단계: PyTorch → ONNX

```python
import torch
import torch.nn as nn


def export_to_onnx(
    model: nn.Module,
    save_path: str,
    input_shape: tuple = (1, 3, 640, 640),
    opset: int = 17,          # TRT 8.x/9.x 호환. TRT 10.x는 19 사용
    dynamic: bool = False,
) -> None:
    model.eval()
    dummy_input = torch.randn(*input_shape).cuda()

    dynamic_axes = None
    if dynamic:
        dynamic_axes = {
            'input':  {0: 'batch', 2: 'height', 3: 'width'},
            'output': {0: 'batch'},
        }

    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy_input,
            save_path,
            opset_version=opset,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes=dynamic_axes,
            do_constant_folding=True,  # 상수 폴딩: 고정값 연산 사전 계산
            export_params=True,        # 가중치 포함
            verbose=False,
        )

    # 즉시 유효성 검사
    import onnx
    model_onnx = onnx.load(save_path)
    onnx.checker.check_model(model_onnx)
    print(f"ONNX 저장 완료: {save_path}")
    print(f"  opset: {model_onnx.opset_import[0].version}")
    print(f"  노드 수: {len(model_onnx.graph.node)}")
```

### opset 버전 선택 기준

| opset | TRT 호환 | ONNX Runtime 호환 | 비고 |
|-------|---------|-----------------|------|
| ≤ 12 | TRT 7.x | 구형 | 레거시 |
| 13 | TRT 8.x | ○ | INT8 양자화 지원 시작 |
| **17** | **TRT 8.x/9.x** | **○** | **현장 권장** |
| 19 | TRT 10.x | ○ | LayerNorm native 지원 |
| 21 | 일부 미지원 | ○ | 최신 기능 |

---

## 2단계: 그래프 단순화 (onnxsim)

변환 직후의 ONNX 그래프에는 불필요한 노드가 많습니다. onnxsim으로 단순화하면 TRT 변환 성공률이 크게 오릅니다.

```python
import onnx
from onnxsim import simplify

model = onnx.load('model_raw.onnx')
print(f"단순화 전 노드 수: {len(model.graph.node)}")

simplified, ok = simplify(model)
assert ok, "단순화 실패 — 원본 모델로 진행"

onnx.save(simplified, 'model_simplified.onnx')
print(f"단순화 후 노드 수: {len(simplified.graph.node)}")
# 전형적으로 20~40% 노드 감소
```

**onnxsim이 수행하는 최적화:**
- `Identity` 노드 제거
- 연속 `Transpose` 통합
- 상수 전파 (constant propagation)
- 불필요한 `Cast` 제거
- `Reshape → Reshape` 병합

[Netron](https://netron.app)으로 단순화 전후를 시각적으로 비교하면 그 차이를 직관적으로 확인할 수 있습니다.

---

## 3단계: 수치 검증 (ONNX Runtime)

```python
import onnxruntime as ort
import numpy as np


def verify_onnx(onnx_path: str, pytorch_model: nn.Module,
                test_input: torch.Tensor, atol: float = 1e-4) -> None:
    """PyTorch 출력과 ONNX Runtime 출력을 수치 비교"""

    # PyTorch 기준값
    pytorch_model.eval()
    with torch.no_grad():
        pt_out = pytorch_model(test_input).cpu().numpy()

    # ONNX Runtime 출력
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    session = ort.InferenceSession(
        onnx_path,
        sess_options=sess_options,
        providers=['CUDAExecutionProvider', 'CPUExecutionProvider'],
    )
    ort_out = session.run(None, {'input': test_input.cpu().numpy()})[0]

    max_diff = np.abs(pt_out - ort_out).max()
    mean_diff = np.abs(pt_out - ort_out).mean()

    print(f"최대 오차: {max_diff:.6f}")
    print(f"평균 오차: {mean_diff:.6f}")

    if max_diff > atol:
        raise ValueError(f"수치 검증 실패: 최대 오차 {max_diff:.6f} > 허용 {atol}")
    print("검증 통과!")
```

**허용 오차 기준:**
- FP32 모델: `atol=1e-5`
- FP16 모델: `atol=1e-3`
- INT8 모델: `atol=1e-2`

---

## 자주 만나는 오류 TOP 5

### 오류 1: `Unsupported op: aten::xxx`

```
RuntimeError: Unsupported: ONNX export of operator aten::upsample_bilinear2d
```

**원인:** PyTorch op이 ONNX opset에 없거나 버전 불일치

**해결:**
```python
# 방법 1: opset 버전 올리기
torch.onnx.export(..., opset_version=17)

# 방법 2: 커스텀 심볼릭 등록
from torch.onnx import register_custom_op_symbolic

def upsample_bilinear_symbolic(g, input, output_size, align_corners, scales):
    return g.op("Resize", input,
                g.op("Constant", value_t=torch.tensor([], dtype=torch.float32)),
                g.op("Constant", value_t=torch.tensor(scales, dtype=torch.float32)),
                coordinate_transformation_mode_s="align_corners" if align_corners else "half_pixel",
                mode_s="linear")

register_custom_op_symbolic('aten::upsample_bilinear2d', upsample_bilinear_symbolic, 17)
```

---

### 오류 2: Dynamic control flow

```
torch.jit.frontend.UnsupportedNodeError: control flow is not supported
```

**원인:** `if/for` 분기가 Trace 방식에서 처리 불가

**해결:**
```python
# 방법 1: torch.jit.script
@torch.jit.script
def forward_impl(x: torch.Tensor, training: bool) -> torch.Tensor:
    if training:
        return dropout(x)
    return x

# 방법 2: 추론 전용 래퍼로 분기 제거
class InferenceWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.model.eval()      # BatchNorm, Dropout 추론 모드 고정
        self.model.training = False

    def forward(self, x):
        return self.model(x)   # 항상 추론 경로만 실행
```

---

### 오류 3: opset version mismatch

```
Error: Node 'LayerNorm' requires opset 17, but graph uses opset 12
```

**해결:** `opset_version=17` 명시. 또는 모델을 업그레이드:

```python
import onnx
from onnx import version_converter

model = onnx.load('model_opset12.onnx')
converted = version_converter.convert_version(model, 17)
onnx.save(converted, 'model_opset17.onnx')
```

---

### 오류 4: Shape inference failed

```
onnx.onnx_pb.TensorShapeProto: shape is not determined
```

**원인:** 동적 형상 추적 실패

**해결:**
```python
# dummy_input 형상을 실제 배포 입력과 일치시키기
dummy_input = torch.randn(1, 3, 640, 640).cuda()  # 배치 고정

# 또는 dynamic_axes를 명시적으로 지정
dynamic_axes = {
    'input': {0: 'batch'},        # 배치만 동적
    'output': {0: 'batch'},
}
# height/width까지 동적으로 하면 TRT 빌드 시간 증가 + 성능 저하
```

---

### 오류 5: dict/list 반환 모델

```
RuntimeError: Only tensors or tuples of tensors can be output from traced functions
```

**원인:** `forward()`가 딕셔너리나 리스트를 반환

**해결:**
```python
class OnnxExportWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        out = self.model(x)       # {'boxes': ..., 'scores': ..., 'labels': ...}
        # ONNX는 텐서 튜플만 지원
        return out['boxes'], out['scores'], out['labels']

# 변환 시
wrapper = OnnxExportWrapper(model).cuda().eval()
torch.onnx.export(wrapper, dummy_input, 'model.onnx',
                  output_names=['boxes', 'scores', 'labels'])
```

---

## 그래프 검사 및 수정

```python
import onnx
from onnx import numpy_helper

model = onnx.load('model.onnx')

# 모든 노드 출력
for node in model.graph.node:
    print(f"op={node.op_type:20s} name={node.name}")

# 중간 레이어 출력 활성화 (디버깅)
for node in model.graph.node:
    for output in node.output:
        if output:
            model.graph.output.append(
                onnx.ValueInfoProto(name=output)
            )
onnx.save(model, 'model_debug.onnx')

# ONNX Runtime으로 중간값 확인
session = ort.InferenceSession('model_debug.onnx')
outputs = session.run(None, {'input': test_input.numpy()})
print(f"중간 레이어 출력 개수: {len(outputs)}")
```

---

## 변환 완료 체크리스트

```
□ model.eval() 호출 완료
□ dummy_input 형상 = 실제 배포 입력 형상
□ opset_version=17 명시 (또는 타겟 TRT 버전에 맞는 opset)
□ do_constant_folding=True
□ onnxsim으로 그래프 단순화 완료
□ onnx.checker.check_model 통과
□ ONNX Runtime 수치 검증 완료 (max_diff < atol)
□ Netron으로 그래프 시각적 확인
□ trtexec --onnx=model.onnx 변환 테스트 완료
□ TRT 엔진 레이턴시 측정 완료 (P50/P99)
```

---

## 참고 자료

- ONNX Official Documentation: [onnx.ai/onnx](https://onnx.ai/onnx/)
- ONNX Operator Specifications: [onnx.ai/onnx/operators](https://onnx.ai/onnx/operators/)
- Netron Graph Visualizer: [netron.app](https://netron.app)
- onnxsim GitHub: [github.com/daquexian/onnx-simplifier](https://github.com/daquexian/onnx-simplifier)
- ONNX Runtime API: [onnxruntime.ai](https://onnxruntime.ai/)
- torch.onnx 공식 문서: [pytorch.org/docs/stable/onnx.html](https://pytorch.org/docs/stable/onnx.html)
- ONNX opset 변경 이력: [github.com/onnx/onnx/blob/main/docs/Changelog.md](https://github.com/onnx/onnx/blob/main/docs/Changelog.md)
