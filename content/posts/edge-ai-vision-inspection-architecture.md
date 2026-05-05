---
title: "자동차 비전 검사 AI 시스템 아키텍처 — Edge에서 MES까지"
date: 2026-04-05
tags: ["Edge AI", "비전검사", "시스템설계", "Jetson", "PLC", "제조AI"]
categories: ["시스템 설계"]
description: "자동차 생산 라인 비전 검사 AI 시스템의 4계층 아키텍처를 설계하고 운영한 경험. 센서 계층부터 MES 연동까지 실무 관점에서 정리합니다."
showToc: true
tocopen: true
---

## 자동차 비전 검사의 특수성

반도체·디스플레이 공장과 달리 자동차 생산 라인에는 '**사이클타임**'이라는 절대적 시간 제약이 있습니다. 차체 한 대가 용접·도장·조립 라인을 통과하는 시간 안에 수십 개의 비전 검사를 완료해야 합니다.

이 제약이 시스템 설계 전체를 지배합니다:
- 모델 정확도 못지않게 **결정적 레이턴시(deterministic latency)** 가 중요
- 추론 지연 = 라인 정지 = 직접적 비용 손실
- 불량 미검출 = 리콜 리스크

---

## 4계층 아키텍처

```
┌────────────────────────────────────────────────────┐
│          관리 계층 (MES / SCADA / ERP)              │
│  불량 이력 기록 · 라인 제어 명령 · KPI 리포팅       │
└──────────────────────┬─────────────────────────────┘
                       │ REST API / OPC-UA
┌──────────────────────┴─────────────────────────────┐
│         서버 계층 (x86 GPU 서버)                    │
│  복잡 모델 추론 · 다중 카메라 집계 · 재학습 파이프라인│
│  Dell/HP + A100/H100 · 10GbE                       │
└──────────────────────┬─────────────────────────────┘
                       │ GigE Vision / 10GbE
┌──────────────────────┴─────────────────────────────┐
│         엣지 계층 (Jetson / 산업용 PC)              │
│  실시간 TRT 추론 · 전처리 · PLC 통신                │
│  NVIDIA Jetson AGX Orin · RTX/T4 내장 PC           │
└──────────────────────┬─────────────────────────────┘
                       │ GigE Vision · Modbus TCP
┌──────────────────────┴─────────────────────────────┐
│         센서 계층                                   │
│  산업용 카메라 · 조명 컨트롤러 · 인코더 · PLC        │
│  Basler · Cognex · Keyence                          │
└────────────────────────────────────────────────────┘
```

---

## 계층별 상세 설계

### 1. 센서 계층

산업용 카메라는 소비자용 카메라와 완전히 다른 특성을 요구합니다:

| 요구사항 | 이유 |
|---------|------|
| 트리거 동기화 (<1ms 지터) | 차체 이동 중 정확한 타이밍 |
| 글로벌 셔터 | 고속 이동 체에서 모션 블러 방지 |
| GigE Vision 프로토콜 | 표준화된 카메라 제어 |
| IP67 방진방수 | 도장 공정 환경 |

**조명 설계가 모델보다 중요한 경우가 많습니다.** 조명이 나쁘면 아무리 좋은 모델을 써도 미검출이 생깁니다.

```python
# 카메라 트리거 설정 예시 (pypylon - Basler 카메라)
from pypylon import pylon

camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
camera.Open()

# 하드웨어 트리거 설정
camera.TriggerMode.Value = "On"
camera.TriggerSource.Value = "Line1"    # PLC 신호 수신 핀
camera.TriggerActivation.Value = "RisingEdge"
camera.ExposureTime.Value = 500.0       # 마이크로초

camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
```

---

### 2. 엣지 계층

NVIDIA Jetson AGX Orin이 현재 Edge 배포의 기준 하드웨어입니다.

**선택 기준:**

| 장치 | TOPS | 전력 | 적합 용도 |
|------|------|------|---------|
| Jetson Orin Nano | 40 TOPS | 10W | 단일 카메라, 경량 모델 |
| Jetson AGX Orin | 275 TOPS | 60W | 멀티 카메라, 복잡 모델 |
| 산업용 PC + T4 | ~260 TOPS | 150W | 고성능, 유지보수 편의 |

**엣지에서 추론 파이프라인 설계:**

```python
import tensorrt as trt
import numpy as np
import threading
import queue
from dataclasses import dataclass

@dataclass
class InspectionResult:
    camera_id: str
    timestamp: float
    defects: list
    confidence: float
    latency_ms: float


class EdgeInferenceEngine:
    def __init__(self, engine_path: str, camera_id: str):
        self.camera_id = camera_id
        self.input_queue = queue.Queue(maxsize=4)
        self.result_queue = queue.Queue(maxsize=4)
        self._load_engine(engine_path)
        self._start_workers()

    def _load_engine(self, engine_path: str):
        logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, 'rb') as f:
            self.engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        resized = cv2.resize(frame, (640, 640))
        normalized = resized.astype(np.float32) / 255.0
        return np.transpose(normalized, (2, 0, 1))[np.newaxis, :]

    def _infer_worker(self):
        """추론 워커 — 별도 스레드에서 실행"""
        import time
        import pycuda.driver as cuda
        import pycuda.autoinit

        while True:
            frame, trigger_time = self.input_queue.get()
            if frame is None:
                break

            t0 = time.perf_counter()
            input_data = self._preprocess(frame)

            # GPU 메모리에 복사 및 추론
            # (실제 TRT 추론 코드 생략 — 엔진 구조에 따라 다름)
            output = self._run_trt(input_data)

            latency_ms = (time.perf_counter() - t0) * 1000

            result = InspectionResult(
                camera_id=self.camera_id,
                timestamp=trigger_time,
                defects=self._parse_detections(output),
                confidence=output['scores'].max() if len(output['scores']) > 0 else 0.0,
                latency_ms=latency_ms,
            )
            self.result_queue.put(result)

    def _start_workers(self):
        self._worker = threading.Thread(target=self._infer_worker, daemon=True)
        self._worker.start()
```

---

### 3. PLC 통신 — 가장 중요한 인터페이스

비전 검사 시스템에서 PLC 통신은 핵심입니다. 카메라 트리거, 불량품 이젝터 제어, 라인 정지 신호가 모두 PLC를 통합니다.

```python
from pymodbus.client import ModbusTcpClient
import time

class PLCInterface:
    """Modbus TCP로 PLC와 통신"""

    # Modbus 코일 주소 (PLC 설계에 따라 다름)
    COIL_EJECT_DEFECT = 0x0001   # 불량품 이젝터
    COIL_LINE_STOP = 0x0002      # 라인 정지 (연속 불량 시)
    COIL_ALARM = 0x0003          # 알람 신호

    # 레지스터 주소
    REG_DEFECT_COUNT = 0x0100    # 누적 불량 수
    REG_CYCLE_COUNT = 0x0101    # 누적 사이클 수

    def __init__(self, plc_ip: str, port: int = 502):
        self.client = ModbusTcpClient(plc_ip, port=port)
        self.client.connect()

    def send_defect_signal(self, defect_type: int, position: float):
        """불량 판정 결과를 PLC로 전송"""
        # 이젝터 활성화
        self.client.write_coil(self.COIL_EJECT_DEFECT, True)
        time.sleep(0.1)  # 이젝터 동작 시간
        self.client.write_coil(self.COIL_EJECT_DEFECT, False)

        # 불량 카운터 증가
        current_count = self.client.read_holding_registers(
            self.REG_DEFECT_COUNT, count=1
        ).registers[0]
        self.client.write_register(self.REG_DEFECT_COUNT, current_count + 1)

    def emergency_stop(self, reason: str):
        """연속 불량 발생 시 라인 비상 정지"""
        self.client.write_coil(self.COIL_LINE_STOP, True)
        self.client.write_coil(self.COIL_ALARM, True)
        # 알람 해제는 수동 확인 후 진행
```

**주의사항:** PLC 통신 지연이 결정적 레이턴시에 포함됩니다. Modbus TCP는 보통 1~3ms, PROFINET은 <1ms입니다.

---

### 4. MES 연동

검사 결과는 MES(Manufacturing Execution System)에 기록되어야 추적성(traceability)이 확보됩니다.

```python
import requests
from datetime import datetime

class MESConnector:
    def __init__(self, mes_endpoint: str, api_key: str):
        self.endpoint = mes_endpoint
        self.headers = {'X-API-Key': api_key, 'Content-Type': 'application/json'}

    def log_inspection_result(self, result: InspectionResult, vehicle_id: str):
        payload = {
            'vehicle_id': vehicle_id,
            'inspection_time': datetime.fromtimestamp(result.timestamp).isoformat(),
            'camera_id': result.camera_id,
            'result': 'DEFECT' if result.defects else 'PASS',
            'defects': [
                {
                    'type': d['class'],
                    'confidence': d['confidence'],
                    'bbox': d['bbox'],
                }
                for d in result.defects
            ],
            'model_version': 'yolov8s-v2.3.1',
            'latency_ms': result.latency_ms,
        }
        # 비동기 전송 — MES 응답이 느려도 검사 파이프라인 블로킹 안 됨
        requests.post(f"{self.endpoint}/inspections", json=payload,
                     headers=self.headers, timeout=2.0)
```

---

## 레이턴시 예산 관리

55ms 안에 모든 단계가 완료되어야 합니다:

```
단계                  목표      실측 (P99)
─────────────────────────────────────────
카메라 캡처           5ms       3.2ms
이미지 전송 (GigE)    3ms       2.1ms
GPU 메모리 복사       2ms       1.8ms
전처리 (GPU)          5ms       4.3ms
TRT 추론 (FP16)      30ms      27.4ms ✓
후처리 + NMS          4ms       3.1ms
PLC 신호 전송         3ms       2.2ms
─────────────────────────────────────────
합계                 52ms      44.1ms ✓ (SLA 55ms 대비 여유 10.9ms)
```

---

## 운영하면서 배운 것들

1. **조명 변동이 가장 많은 거짓 경보를 만든다** — 교대 근무 시 조명 설정 변경이 모델 성능에 직격
2. **카메라 렌즈 오염은 서서히 악화된다** — Prometheus로 모델 신뢰도 트렌드를 보면 조기 감지 가능
3. **PLC 통신은 절대 동기로 처리하지 않는다** — 타임아웃 1번에 라인이 멈춤
4. **엔진은 항상 버전 태그와 함께 저장한다** — GPU 드라이버 업데이트 후 재빌드 필요성을 잊기 쉬움

다음 글에서는 LCD Mura 불량 분석과 Demura 알고리즘을 다루겠습니다.
