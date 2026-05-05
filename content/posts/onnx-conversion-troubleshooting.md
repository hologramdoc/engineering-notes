---
title: "PyTorch → ONNX → TensorRT 변환 완벽 가이드 — 실패 없이 넘어가는 법"
date: 2026-04-15
tags: ["ONNX", "TensorRT", "PyTorch", "모델변환", "추론최적화"]
categories: ["AI 추론 엔진"]
description: "PyTorch 모델을 ONNX로 변환하고 TensorRT 엔진까지 만드는 전 과정. 현장에서 자주 만나는 오류와 해결법을 담았습니다."
showToc: true
tocopen: true
---

## ONNX란 무엇인가

ONNX(Open Neural Network Exchange)는 Microsoft와 Facebook이 2017년 공동 제안한 딥러닝 모델 교환 표준입니다. PyTorch, TensorFlow 등 다양한 프레임워크로 학습된 모델을 **하나의 표준 포맷**으로 저장하고, TensorRT·OpenVINO·CoreML 등 다양한 런타임에서 실행할 수 있게 합니다.

변환 체인은 이렇습니다:

```
PyTorch (.pt) → ONNX (.onnx) → TensorRT (.engine)
                     ↓
              ONNX Runtime (검증용)
```

ONNX Runtime으로 먼저 검증하고, TensorRT로 최종 배포하는 것이 안전한 패턴입니다.

---

## 1단계: PyTorch → ONNX 변환

```python
import torch
import torch.nn as nn


def export_to_onnx(
    model: nn.Module,
    save_path: str,
    input_shape: tuple = (1, 3, 640, 640),
    opset: int = 17,
    dynamic: bool = False,
) -> None:
    model.eval()
    dummy_input = torch.randn(*input_shape).cuda()

    dynamic_axes = None
    if dynamic:
        dynamic_axes = {
            'input': {0: 'batch', 2: 'height', 3: 'width'},
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
            do_constant_folding=True,  # 상수 폴딩 최적화
            export_params=True,
        )
    print(f"ONNX 저장: {save_path}")
```

**opset 버전 선택 기준:**

| opset | TRT 호환 | 비고 |
|-------|---------|------|
| 12 이하 | TRT 7.x | 구형 |
| 17 | TRT 8.x/9.x | **현장 권장** |
| 19 | TRT 10.x | 최신 기능 사용 시 |
| 21 | 최신 | TRT 미지원 op 있음 |

> opset을 명시하지 않으면 기본값이 사용되는데, 이 기본값이 TRT와 맞지 않아 변환 실패하는 경우가 많습니다. 항상 명시적으로 지정하세요.

---

## 2단계: ONNX 모델 검증

변환 후 반드시 PyTorch 출력과 수치를 비교해야 합니다.

```python
import onnxruntime as ort
import numpy as np


def verify_onnx(onnx_path: str, pytorch_model: nn.Module, test_input: torch.Tensor) -> None:
    # PyTorch 출력
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
    print(f"최대 오차: {max_diff:.6f}")
    assert max_diff < 1e-4, f"오차 초과: {max_diff}"
    print("검증 통과!")
```

---

## 3단계: 그래프 단순화 (필수)

`onnxsim`으로 그래프를 단순화하면 TensorRT 변환 성공률이 크게 올라갑니다.

```python
from onnxsim import simplify
import onnx

model = onnx.load("model.onnx")
simplified, ok = simplify(model)
assert ok, "단순화 실패"
onnx.save(simplified, "model_simplified.onnx")
```

Netron(`netron.app`)으로 단순화 전후 노드 수를 비교해 보면 체감됩니다. 노드 수가 30~40% 줄어드는 경우도 있습니다.

---

## 자주 만나는 오류 TOP 5

### 오류 1: `Unsupported op: aten::xxx`

```
RuntimeError: Unsupported: ONNX export of operator aten::upsample_bilinear2d
```

**원인:** PyTorch op이 ONNX에 없거나, 해당 opset에서 지원하지 않음

**해결:**
```python
# 방법 1: opset 올리기
torch.onnx.export(..., opset_version=17)

# 방법 2: 해당 op를 지원하는 커스텀 등록
from torch.onnx import register_custom_op_symbolic

@register_custom_op_symbolic('aten::upsample_bilinear2d', opset_version=17)
def upsample_bilinear2d_symbolic(g, input, output_size, ...):
    ...
```

---

### 오류 2: `Dynamic control flow` 오류

```
torch.jit.frontend.UnsupportedNodeError: control flow is not supported
```

**원인:** `if/for` 분기가 ONNX 추적(trace) 방식에서 처리 불가

**해결:**
```python
# 방법 1: torch.jit.script 사용
@torch.jit.script
def my_forward(x):
    ...

# 방법 2: 분기 제거 — 추론 시에는 고정 경로만 필요한 경우 많음
# training 분기를 model.eval()로 비활성화 후 export
```

---

### 오류 3: `opset version mismatch`

```
Error: Node 'LayerNorm' requires opset 17, but graph uses opset 12
```

**해결:** `opset_version=17`을 명시적으로 지정.

---

### 오류 4: `Shape inference failed`

**원인:** 동적 형상 추적 실패

**해결:**
```python
# dummy_input 형상을 실제 배포 입력과 동일하게 고정
dummy_input = torch.randn(1, 3, 640, 640).cuda()  # 배치 고정

# 또는 dynamic_axes 조정
dynamic_axes = {'input': {0: 'batch'}}  # 배치만 동적으로
```

---

### 오류 5: `TRT 변환 시 CUDA error`

**원인:** GPU 메모리 부족 (워크스페이스 크기 문제)

**해결:**
```bash
trtexec --onnx=model.onnx \
        --saveEngine=model.engine \
        --fp16 \
        --workspace=2048  # MB 단위, 줄여보기
```

---

## ONNX 관련 자주 받는 질문

**Q: Custom Loss Function 사용 모델도 변환되나요?**  
A: 네. 손실 함수는 학습에만 쓰이고 추론 그래프에 포함되지 않습니다. `forward()`만 변환됩니다.

**Q: `forward()`가 딕셔너리를 반환하면 변환이 안 됩니다.**  
A: ONNX는 텐서 출력만 지원합니다. 래퍼 모델로 딕셔너리를 튜플로 언패킹하세요.

```python
class OnnxWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        out = self.model(x)  # dict 반환
        return out['boxes'], out['scores'], out['labels']  # tuple로
```

**Q: INT8 양자화된 모델도 ONNX로 변환되나요?**  
A: 네. `QLinearConv` 등 양자화 연산자는 ONNX opset 13+에서 지원됩니다.

---

## 변환 체크리스트

```
□ model.eval() 호출 확인
□ dummy_input을 실제 입력과 동일한 형상으로 설정
□ opset_version=17 명시
□ do_constant_folding=True 설정
□ onnxsim으로 그래프 단순화
□ ONNX Runtime으로 수치 검증 (max_diff < 1e-4)
□ Netron으로 그래프 시각적 확인
□ trtexec로 TRT 변환 테스트
```

다음 글에서는 Prometheus + Grafana로 제조 현장 AI 추론 모니터링 시스템을 구축하는 방법을 다루겠습니다.
