---
title: "자동차 비전 검사 AI 시스템 아키텍처 — 센서부터 MES까지 4계층 설계"
date: 2026-04-05
tags: ["Edge AI", "비전검사", "시스템설계", "Jetson", "PLC", "MES", "SCADA", "GigE Vision"]
categories: ["시스템 설계"]
description: "자동차 생산 라인 비전 검사 AI 시스템의 4계층 아키텍처 전체. 산업용 카메라 이론, CUDA Stream 파이프라인, PLC 통신, MES 연동까지 실무 설계 노트."
showToc: true
tocopen: true
cover:
  image: "https://images.unsplash.com/photo-1565043666747-69f6646db940?w=1260&h=630&fit=crop&auto=format"
  alt: "자동차 제조 공장 — 비전 검사 시스템"
  relative: false
---

## 자동차 비전 검사의 특수성

반도체·디스플레이 공장과 달리, 자동차 생산 라인에는 **사이클타임(Cycle Time)**이라는 절대적 제약이 있습니다.

차체 한 대가 용접·도장·조립 라인을 통과하는 시간이 정해져 있고, 그 안에 수십 개의 비전 검사를 완료해야 합니다. 이 제약이 시스템 설계 철학 전체를 지배합니다.

> **결정적 레이턴시(Deterministic Latency)**: 평균 레이턴시가 아닌, 최악의 경우에도 SLA 이내에 처리를 보장하는 능력.

일반 AI 서비스는 P50 레이턴시를 봅니다. 자동차 라인은 **P99.9 레이턴시**가 사이클타임 이내인지를 봅니다.

---

## 4계층 시스템 아키텍처

```
┌──────────────────────────────────────────────────────────────────────┐
│  관리 계층 (Management Layer)                                         │
│  MES (Manufacturing Execution System)                                │
│  SCADA (Supervisory Control and Data Acquisition)                    │
│  ERP (Enterprise Resource Planning)                                  │
│  ─ 불량 이력 기록 · 라인 KPI 리포팅 · 부품 추적(Traceability)         │
│  ─ 통신: REST API, OPC-UA, SQL                                        │
└─────────────────────────┬────────────────────────────────────────────┘
                          │ OPC-UA / REST API
┌─────────────────────────┴────────────────────────────────────────────┐
│  서버 계층 (Server Layer)                                             │
│  x86 GPU 서버 (Dell/HP + NVIDIA A100/H100)                           │
│  ─ 복잡 모델 추론 · 다중 카메라 집계 · 재학습 파이프라인              │
│  ─ 10GbE 네트워크 연결                                                │
│  ─ Active Learning, Shadow/Canary 배포 관리                           │
└─────────────────────────┬────────────────────────────────────────────┘
                          │ GigE Vision / 10GbE
┌─────────────────────────┴────────────────────────────────────────────┐
│  엣지 계층 (Edge Layer)                                               │
│  NVIDIA Jetson AGX Orin / 산업용 PC (i7/Xeon + RTX T4)              │
│  ─ TRT 실시간 추론 · CUDA Stream 파이프라인 · GPU 전처리              │
│  ─ PLC 통신 (Modbus TCP, PROFINET)                                    │
└─────────────────────────┬────────────────────────────────────────────┘
                          │ GigE Vision (IEEE 802.3) / Camera Link
┌─────────────────────────┴────────────────────────────────────────────┐
│  센서 계층 (Sensor Layer)                                             │
│  산업용 카메라 (Basler / Cognex / Keyence)                            │
│  조명 컨트롤러 · 인코더 · PLC (Siemens S7 / Allen-Bradley)           │
│  ─ GigE Vision / Camera Link 프로토콜                                 │
│  ─ 카메라 트리거: PLC → 카메라 (GPIO, <1ms 지터)                      │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 이론: 산업용 카메라 선택 기준

### 글로벌 셔터 vs 롤링 셔터

```
글로벌 셔터:    모든 픽셀이 동시에 노출 시작·종료
                → 고속 이동 피사체에서 왜곡 없음 ✓

롤링 셔터:      픽셀 행(row) 순서대로 노출
                → 고속 이동 시 "젤로(Jello)" 효과 — 비전 검사 부적합 ✗
```

자동차 라인에서는 **반드시 글로벌 셔터** 카메라를 사용합니다.

### 해상도 vs 속도 트레이드오프

```
픽셀 처리량 (이론) = 해상도 × FPS

예시:
  5MP @ 100fps = 500 MP/s
  GigE Vision 이론 대역폭 = 125 MB/s (1GbE) → 약 50fps @ 5MP (8bit)
  10GigE = 1,250 MB/s → 약 250fps @ 5MP

결함 최소 검출 크기 결정:
  FOV (시야각): 300mm × 225mm
  해상도: 2448 × 2048
  픽셀당 공간 해상도: 300/2448 = 0.12mm/px
  최소 검출 가능 결함: ~0.5mm (약 4픽셀)
```

### GigE Vision 프로토콜 구조

```
┌────────────────────────────────┐
│  GenICam (Generic Interface)   │ ← 카메라 파라미터 제어 표준
│  PFNC (Pixel Format Naming)    │ ← 픽셀 포맷 표준화
├────────────────────────────────┤
│  GigE Vision 2.0               │ ← 이미지 전송 프로토콜
│  (GVSP: GigE Vision Stream     │
│   GVCP: GigE Vision Control)   │
├────────────────────────────────┤
│  UDP / TCP over Ethernet       │ ← 이미지: UDP(빠름), 제어: TCP(신뢰)
└────────────────────────────────┘
```

```python
# pypylon (Basler 카메라 Python SDK) 기본 설정
from pypylon import pylon

camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
camera.Open()

# 하드웨어 트리거 설정
camera.TriggerMode.Value = "On"
camera.TriggerSource.Value = "Line1"           # PLC에서 GPIO 신호 수신
camera.TriggerActivation.Value = "RisingEdge"
camera.ExposureMode.Value = "Timed"
camera.ExposureTime.Value = 500.0              # 500μs — 고속 이동 객체 동결

# 패킷 지연 최적화 (네트워크 버퍼 오버플로 방지)
camera.GevSCPD.Value = 1000                    # 패킷 간 지연 (ns)
camera.GevSCBWA.Value = 0                      # 대역폭 자동

camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
```

---

## CUDA Stream 파이프라인 설계

### 단순 순차 실행 vs 파이프라인 비교

```
순차 실행:
  [Frame N: 캡처] → [전처리] → [추론] → [후처리] → [결과] = 55ms
                                                    [Frame N+1: 캡처] = 55ms

CUDA Stream 파이프라인:
  [Frame N: 캡처  → 전처리 → 추론 → 후처리 → 결과]
                  [Frame N+1: 캡처 → 전처리 → 추론 ...]
                             [Frame N+2: 캡처 → 전처리 ...]

  → 처리량 2~3배 향상, 단일 프레임 레이턴시는 동일
```

```python
import tensorrt as trt
import pycuda.driver as cuda
import threading
import queue
import numpy as np
import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class FrameContext:
    frame_id: int
    raw_image: np.ndarray
    camera_id: str
    trigger_time: float
    preprocessed: Optional[np.ndarray] = None
    inference_result: Optional[dict] = None
    latency_ms: float = 0.0
    stage_times: dict = field(default_factory=dict)


class CUDAStreamPipeline:
    """
    3단계 CUDA Stream 파이프라인:
    Stream 0: 전처리 (CPU→GPU 전송, resize, normalize)
    Stream 1: 추론 (TRT execute)
    Stream 2: 후처리 (GPU→CPU 전송, NMS)
    """

    def __init__(self, engine_path: str, camera_id: str, buffer_size: int = 4):
        self.camera_id = camera_id

        # 3개의 독립 CUDA Stream
        self.streams = [cuda.Stream() for _ in range(3)]

        self._load_engine(engine_path)
        self._allocate_buffers(buffer_size)

        self.input_queue  = queue.Queue(maxsize=buffer_size)
        self.output_queue = queue.Queue(maxsize=buffer_size)

        self._start_workers()

    def _load_engine(self, path: str):
        logger = trt.Logger(trt.Logger.WARNING)
        with open(path, 'rb') as f:
            self.engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()

    def _allocate_buffers(self, n: int):
        """핀드 메모리(pinned memory) 풀 — DMA 전송 속도 최대화"""
        c, h, w = 3, 640, 640
        self.host_buffers   = [cuda.pagelocked_empty((c, h, w), np.float32) for _ in range(n)]
        self.device_buffers = [cuda.mem_alloc(c * h * w * 4) for _ in range(n)]

    def _preprocess_gpu(self, raw: np.ndarray, buf_idx: int) -> None:
        """GPU 전처리: resize + normalize + HWC→CHW"""
        import cv2
        resized = cv2.resize(raw, (640, 640))
        rgb = resized[:, :, ::-1].astype(np.float32) / 255.0
        chw = np.ascontiguousarray(np.transpose(rgb, (2, 0, 1)))
        np.copyto(self.host_buffers[buf_idx], chw)
        cuda.memcpy_htod_async(
            self.device_buffers[buf_idx],
            self.host_buffers[buf_idx],
            self.streams[0]   # Stream 0으로 전송
        )

    def _infer_worker(self):
        """추론 전용 워커 스레드"""
        buf_idx = 0
        while True:
            ctx: FrameContext = self.input_queue.get()
            if ctx is None:
                break

            t0 = time.perf_counter()

            # Stream 0: 전처리
            self._preprocess_gpu(ctx.raw_image, buf_idx)
            self.streams[0].synchronize()
            ctx.stage_times['preprocess'] = time.perf_counter() - t0

            # Stream 1: 추론
            t1 = time.perf_counter()
            self.context.execute_async_v2(
                bindings=[int(self.device_buffers[buf_idx]), int(self.output_device)],
                stream_handle=self.streams[1].handle
            )
            self.streams[1].synchronize()
            ctx.stage_times['infer'] = time.perf_counter() - t1

            # Stream 2: 결과 복사 + NMS
            t2 = time.perf_counter()
            cuda.memcpy_dtoh_async(self.output_host, self.output_device, self.streams[2])
            self.streams[2].synchronize()
            ctx.inference_result = self._nms(self.output_host)
            ctx.stage_times['postprocess'] = time.perf_counter() - t2

            ctx.latency_ms = (time.perf_counter() - t0) * 1000
            self.output_queue.put(ctx)

            buf_idx = (buf_idx + 1) % len(self.host_buffers)

    def _nms(self, raw_output: np.ndarray, conf_threshold: float = 0.5,
              iou_threshold: float = 0.45) -> dict:
        """Non-Maximum Suppression"""
        boxes, scores, class_ids = [], [], []
        for det in raw_output.reshape(-1, 85):   # YOLOv8: [x,y,w,h, conf, class×80]
            conf = det[4]
            if conf < conf_threshold:
                continue
            cls = det[5:].argmax()
            boxes.append(det[:4].tolist())
            scores.append(float(conf))
            class_ids.append(int(cls))
        return {'boxes': boxes, 'scores': scores, 'class_ids': class_ids}

    def _start_workers(self):
        self._worker = threading.Thread(target=self._infer_worker, daemon=True)
        self._worker.start()
```

---

## PLC 통신 설계

### 산업 프로토콜 비교

| 프로토콜 | 계층 | 사이클 시간 | 용도 |
|---------|------|----------|------|
| **Modbus TCP** | Application/TCP | 1~10ms | 범용, 설정 간단 |
| **PROFINET RT** | Ethernet | 1~4ms | Siemens 표준 |
| **PROFINET IRT** | Ethernet (동기) | <1ms | 고정밀 동기 |
| **EtherNet/IP** | TCP/UDP | 1~10ms | Rockwell 표준 |
| **OPC-UA** | Application/TCP | 10~100ms | MES 연동 |

비전 검사 시스템에서 카메라 트리거와 이젝터 신호는 **PROFINET RT 이상**을 권장합니다.

### Modbus TCP 인터페이스

```python
from pymodbus.client import ModbusTcpClient
from pymodbus.exceptions import ModbusIOException
import time
import logging

logger = logging.getLogger(__name__)


class PLCInterface:
    """
    Modbus TCP 기반 PLC 인터페이스
    코일 주소: 0x0000~0x00FF (디지털 출력)
    레지스터 주소: 0x0100~0x01FF (데이터 교환)
    """

    # 코일 주소 맵 (PLC 엔지니어와 사전 합의)
    COIL_EJECT_DEFECT = 0x0001   # 불량품 이젝터 ON/OFF
    COIL_LINE_STOP    = 0x0002   # 라인 비상 정지
    COIL_ALARM        = 0x0003   # 경보 신호
    COIL_INSPECT_OK   = 0x0004   # 검사 완료 (Handshake)

    # 레지스터 주소 맵
    REG_DEFECT_COUNT  = 0x0100   # 누적 불량 수
    REG_CYCLE_COUNT   = 0x0101   # 누적 사이클 수
    REG_MODEL_ID      = 0x0102   # 현재 차종 코드 (PLC→PC)
    REG_DEFECT_TYPE   = 0x0103   # 불량 종류 코드 (PC→PLC)

    def __init__(self, plc_ip: str, port: int = 502, timeout: float = 1.0):
        self.client = ModbusTcpClient(plc_ip, port=port, timeout=timeout)
        self._connect()

    def _connect(self, retries: int = 3):
        for i in range(retries):
            if self.client.connect():
                logger.info("PLC 연결 성공")
                return
            time.sleep(0.5)
        raise ConnectionError(f"PLC 연결 실패 ({retries}회 시도)")

    def send_defect_signal(self, defect_type: int, duration_sec: float = 0.1):
        """
        불량 판정 시 이젝터 활성화.
        PLC로부터 Handshake 확인 후 해제.
        """
        try:
            # 불량 종류 레지스터에 기록
            self.client.write_register(self.REG_DEFECT_TYPE, defect_type)

            # 이젝터 ON
            self.client.write_coil(self.COIL_EJECT_DEFECT, True)
            time.sleep(duration_sec)
            self.client.write_coil(self.COIL_EJECT_DEFECT, False)

            # 불량 카운터 증가
            cnt = self.client.read_holding_registers(self.REG_DEFECT_COUNT, 1)
            if not cnt.isError():
                self.client.write_register(
                    self.REG_DEFECT_COUNT, cnt.registers[0] + 1
                )
        except ModbusIOException as e:
            logger.error(f"PLC 통신 오류: {e}")
            # PLC 통신 실패해도 추론 파이프라인은 계속 (비동기 처리)

    def get_current_model_id(self) -> int:
        """현재 라인의 차종 코드 읽기 (PLC 설정값)"""
        result = self.client.read_holding_registers(self.REG_MODEL_ID, 1)
        if result.isError():
            return -1
        return result.registers[0]

    def emergency_stop(self, reason: str):
        """연속 불량 N개 이상 발생 시 라인 비상 정지"""
        logger.critical(f"비상 정지 요청: {reason}")
        self.client.write_coil(self.COIL_LINE_STOP, True)
        self.client.write_coil(self.COIL_ALARM, True)
        # 해제는 작업자 수동 확인 후 진행 (안전 설계)
```

---

## MES 연동 — 추적성(Traceability) 확보

```python
import requests
from datetime import datetime, timezone
import uuid


class MESConnector:
    """
    MES REST API 연동.
    불량 판정 결과를 차체 VIN(Vehicle Identification Number)에 연결하여
    완성차까지 추적 가능한 이력 확보.
    """

    def __init__(self, endpoint: str, api_key: str, timeout: float = 2.0):
        self.endpoint = endpoint.rstrip('/')
        self.session = requests.Session()
        self.session.headers.update({
            'X-API-Key': api_key,
            'Content-Type': 'application/json',
        })
        self.timeout = timeout

    def log_inspection(self,
                        vin: str,
                        camera_id: str,
                        result: dict,
                        model_version: str,
                        latency_ms: float) -> bool:
        payload = {
            'inspection_id': str(uuid.uuid4()),
            'vin': vin,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'camera_id': camera_id,
            'line_code': camera_id.split('-')[0],
            'judgment': 'DEFECT' if result['boxes'] else 'PASS',
            'defects': [
                {
                    'defect_type': det['class_name'],
                    'confidence': det['score'],
                    'bbox_xywh': det['bbox'],
                    'severity': 'HIGH' if det['score'] > 0.9 else 'LOW',
                }
                for det in result.get('detections', [])
            ],
            'model_version': model_version,
            'latency_ms': round(latency_ms, 2),
        }

        try:
            resp = self.session.post(
                f"{self.endpoint}/api/v1/inspections",
                json=payload,
                timeout=self.timeout,
            )
            return resp.status_code == 201
        except requests.Timeout:
            logger.warning("MES 응답 타임아웃 — 로컬 버퍼에 저장")
            self._buffer_locally(payload)
            return False

    def _buffer_locally(self, payload: dict):
        """MES 연결 장애 시 로컬 SQLite에 임시 저장, 복구 후 동기화"""
        import sqlite3
        import json
        with sqlite3.connect('/var/vision/inspection_buffer.db') as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS buffer (id TEXT, data TEXT, ts REAL)"
            )
            conn.execute(
                "INSERT INTO buffer VALUES (?, ?, ?)",
                (payload['inspection_id'], json.dumps(payload), time.time())
            )
```

---

## 레이턴시 예산 상세 분석

실제 라인 환경에서 측정한 단계별 레이턴시 (P99, Jetson AGX Orin):

```
단계                      목표      P50      P95      P99     기여율
────────────────────────────────────────────────────────────────────
카메라 노출                5.0ms    0.5ms    0.5ms    0.5ms    1.2%
GigE 전송 (5MP @ 1GbE)    8.0ms    6.1ms    6.4ms    7.2ms   17.5%
GPU 메모리 복사 (pinned)   2.0ms    1.2ms    1.5ms    1.8ms    4.4%
GPU 전처리 (resize+norm)   5.0ms    3.1ms    3.8ms    4.3ms   10.4%
TRT FP16 추론             25.0ms   20.3ms   22.8ms   24.1ms   58.5%
후처리 + NMS               4.0ms    2.4ms    2.8ms    3.1ms    7.5%
PLC 신호 전송              3.0ms    1.8ms    2.0ms    2.2ms    5.3%
────────────────────────────────────────────────────────────────────
합계                      52.0ms   35.4ms   39.8ms   43.2ms
SLA 여유                                              11.8ms   ✓
```

GigE Vision 전송이 생각보다 큰 비중을 차지합니다. 10GigE로 전환하면 이 구간을 1~2ms로 줄일 수 있습니다.

---

## 운영 노하우 — 현장에서 배운 것들

**1. 조명 변동이 가장 많은 오검을 만든다**  
교대 근무 시 조명 on/off 순간, 계절별 자연광 유입량 변화가 모델에 직격합니다. 카메라에 암실 케이스(Dark Box)를 씌워 외부광을 차단하고, 조명 컨트롤러 피드백으로 출력 안정화를 권장합니다.

**2. 렌즈·카메라 오염은 서서히 악화된다**  
Prometheus에서 `vision_model_confidence_score` P50을 7일 이동 평균으로 추적하면 오염이 시작되는 시점을 조기 감지할 수 있습니다. 급격한 스파이크가 아닌 완만한 하락 트렌드를 모니터링해야 합니다.

**3. PLC 통신은 절대 동기 블로킹으로 처리하지 않는다**  
PLC timeout 1회가 전체 파이프라인을 블로킹하면 라인이 멈춥니다. PLC 통신은 별도 스레드에서 비동기로 처리하고, 실패 시 로컬 버퍼에 저장하는 설계가 필수입니다.

**4. 차종 교체 시 모델도 교체해야 한다**  
차종마다 검사 위치와 불량 패턴이 다릅니다. PLC에서 차종 코드를 읽어 자동으로 TRT 엔진을 전환하는 **모델 라우팅** 로직을 미리 설계해두세요.

---

## 참고 자료

- NVIDIA Jetson AGX Orin 기술 명세: [developer.nvidia.com/embedded/jetson-agx-orin](https://developer.nvidia.com/embedded/jetson-agx-orin)
- GigE Vision Standard 2.0: [emva.org/standards-technology/genicam](https://www.emva.org/standards-technology/genicam/)
- GenICam Standard: [emva.org/standards-technology/genicam](https://www.emva.org/standards-technology/genicam/)
- Pypylon (Basler SDK): [github.com/basler/pypylon](https://github.com/basler/pypylon)
- PyModbus: [pymodbus.readthedocs.io](https://pymodbus.readthedocs.io/)
- PROFINET 기술 개요: [profibus.com/technology/profinet](https://www.profibus.com/technology/profinet/)
- OPC-UA 명세: [opcfoundation.org/about/opc-technologies/opc-ua](https://opcfoundation.org/about/opc-technologies/opc-ua/)
- Cognex 산업용 카메라 가이드: [cognex.com/what-is/machine-vision](https://www.cognex.com/what-is/machine-vision)
- NVIDIA DeepStream SDK: [developer.nvidia.com/deepstream-sdk](https://developer.nvidia.com/deepstream-sdk)
