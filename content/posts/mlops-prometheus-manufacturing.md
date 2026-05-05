---
title: "제조 현장 MLOps — Prometheus로 AI 추론 시스템 모니터링하기"
date: 2026-04-10
tags: ["MLOps", "Prometheus", "Grafana", "모니터링", "SLA", "제조AI"]
categories: ["MLOps & 운영"]
description: "자동차 생산 라인 AI 추론 시스템의 운영 모니터링 체계를 Prometheus/Grafana로 구축한 경험을 공유합니다. SLA/SLO 설계부터 장애 대응까지."
showToc: true
tocopen: true
---

## 제조 현장 MLOps는 왜 다른가

일반적인 웹 서비스 MLOps와 제조 현장 MLOps는 근본적으로 다릅니다.

웹 서비스에서 추론이 200ms 걸리면 "좀 느리네" 수준이지만, 자동차 생산 라인에서 55ms SLA를 초과하면 **라인이 멈추고 수천만 원의 손실**이 발생합니다. 불량 판정 오류는 리콜로 이어질 수 있습니다.

이 글은 실제 비전 검사 AI 시스템에서 구축한 모니터링 체계를 공유합니다.

---

## 모니터링 스택 구성

```
[추론 엔진] → [Prometheus Exporter] → [Prometheus] → [Grafana]
                                            ↓
                                       [AlertManager]
                                            ↓
                                    [PagerDuty / Slack]
```

**핵심 컴포넌트:**
- **Prometheus**: 메트릭 수집 및 저장
- **Grafana**: 시각화 대시보드
- **AlertManager**: 알림 라우팅
- **Custom Exporter**: 비전 검사 특화 메트릭 노출

---

## 비전 검사 특화 메트릭 설계

일반 ML 메트릭(정확도, 손실)과 달리, 제조 현장은 **비즈니스 임팩트**와 직결된 메트릭이 필요합니다.

```python
from prometheus_client import (
    Counter, Histogram, Gauge, start_http_server
)
import time

# === 처리량 메트릭 ===
INSPECTIONS_TOTAL = Counter(
    'vision_inspections_total',
    '검사 완료 총 건수',
    ['camera_id', 'result']  # result: pass / defect / error
)

# === 레이턴시 메트릭 ===
INFERENCE_LATENCY = Histogram(
    'vision_inference_latency_seconds',
    '추론 레이턴시 분포',
    ['camera_id', 'model_version'],
    buckets=[0.010, 0.020, 0.030, 0.040, 0.050, 0.060, 0.080, 0.100]
    #        10ms   20ms   30ms   40ms   50ms   60ms   80ms   100ms
)

# === 품질 메트릭 ===
DEFECT_RATE = Gauge(
    'vision_defect_rate_1h',
    '최근 1시간 불량률 (슬라이딩 윈도우)',
    ['line_id', 'defect_type']
)

MODEL_CONFIDENCE = Histogram(
    'vision_model_confidence',
    '모델 신뢰도 점수 분포',
    ['camera_id'],
    buckets=[0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
)

# === GPU 리소스 메트릭 ===
GPU_UTILIZATION = Gauge('vision_gpu_utilization_percent', 'GPU 사용률', ['device'])
GPU_MEMORY_USED = Gauge('vision_gpu_memory_used_bytes', 'GPU 메모리 사용량', ['device'])
GPU_TEMPERATURE = Gauge('vision_gpu_temperature_celsius', 'GPU 온도', ['device'])


class VisionInspectionExporter:
    def __init__(self, camera_id: str, model_version: str):
        self.camera_id = camera_id
        self.model_version = model_version

    def record_inspection(self, latency_sec: float, result: str, confidence: float):
        INSPECTIONS_TOTAL.labels(
            camera_id=self.camera_id,
            result=result
        ).inc()

        INFERENCE_LATENCY.labels(
            camera_id=self.camera_id,
            model_version=self.model_version
        ).observe(latency_sec)

        MODEL_CONFIDENCE.labels(
            camera_id=self.camera_id
        ).observe(confidence)
```

---

## SLA / SLO / SLI 정의

제조 현장 AI 시스템의 SLA를 설계할 때 세 가지를 구분해야 합니다.

| 용어 | 정의 | 예시 |
|------|------|------|
| **SLI** (지표) | 측정 가능한 서비스 수준 지표 | P99 추론 레이턴시 |
| **SLO** (목표) | 내부 달성 목표 | P99 < 50ms, 가용성 99.9% |
| **SLA** (약속) | 외부 계약 수준 | P99 < 55ms, 가용성 99.5% |

**핵심 SLI 목록:**

```yaml
# 레이턴시 SLI
- name: inference_p99_latency
  query: histogram_quantile(0.99, vision_inference_latency_seconds_bucket)
  slo: 0.050  # 50ms 이내

# 가용성 SLI
- name: availability
  query: |
    rate(vision_inspections_total{result!="error"}[5m])
    /
    rate(vision_inspections_total[5m])
  slo: 0.999  # 99.9%

# 검출 정확도 SLI (주기적 샘플링)
- name: defect_detection_accuracy
  slo: 0.995  # 99.5% 이상
```

---

## Prometheus 알림 규칙

```yaml
# alert_rules.yml
groups:
  - name: vision_ai_alerts
    interval: 15s
    rules:
      # P99 레이턴시 SLO 위반
      - alert: InferenceLatencyHigh
        expr: |
          histogram_quantile(0.99,
            rate(vision_inference_latency_seconds_bucket[5m])
          ) > 0.050
        for: 2m
        labels:
          severity: critical
          team: ai-ops
        annotations:
          summary: "P99 추론 레이턴시 SLO 위반"
          description: |
            카메라 {{ $labels.camera_id }}의 P99 레이턴시가
            {{ $value | humanizeDuration }} 로 SLO(50ms)를 초과했습니다.
            즉시 확인 필요.

      # GPU 온도 과열
      - alert: GPUTemperatureHigh
        expr: vision_gpu_temperature_celsius > 85
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "GPU 온도 경고 — {{ $labels.device }}"
          description: "현재 온도: {{ $value }}°C. 냉각 시스템 점검 필요"

      # 불량률 급증
      - alert: DefectRateSpike
        expr: |
          vision_defect_rate_1h > 0.05
          AND
          vision_defect_rate_1h > 2 * vision_defect_rate_1h offset 1h
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "불량률 급증 감지"
          description: |
            라인 {{ $labels.line_id }}에서 불량률이
            전 시간 대비 2배 이상 증가했습니다. (현재: {{ $value | humanizePercentage }})
```

---

## 장애 대응 — RCA 프로세스

현장에서 장애가 발생하면 다음 순서로 대응합니다.

### 즉각 대응 (0~5분)

```bash
# 1. 현재 상태 확인
curl http://localhost:9090/api/v1/query?query=up{job="vision-ai"}

# 2. 최근 에러 로그 확인
journalctl -u vision-inference -n 100 --since "10 min ago"

# 3. GPU 상태 확인
nvidia-smi --query-gpu=name,temperature.gpu,utilization.gpu,memory.used \
  --format=csv,noheader

# 4. 레이턴시 분포 확인
# Grafana → Vision AI 대시보드 → Latency Heatmap
```

### 근본 원인 분석 (RCA) 템플릿

```markdown
## 인시던트 요약
- **발생 시각:** 2026-04-10 14:32 KST
- **감지 방법:** Prometheus Alert (InferenceLatencyHigh)
- **영향 범위:** 카메라 CAM-03, CAM-04 (용접 라인 2구역)
- **영향 기간:** 14:32 ~ 15:01 (29분)

## 타임라인
- 14:32: Alert 발생 (P99 > 50ms)
- 14:35: 온-콜 엔지니어 응답
- 14:41: GPU 메모리 누수 확인 (85% → 97% 증가 추세)
- 14:55: 추론 서비스 재시작
- 15:01: 레이턴시 정상화 확인

## 근본 원인
TRT 엔진의 메모리 풀 미해제 버그.
장시간 운영 시 GPU 메모리가 누적되어 추론 속도 저하 유발.

## 재발 방지
- [ ] 메모리 풀 명시적 해제 코드 추가
- [ ] GPU 메모리 사용률 > 90% 시 사전 알림 추가
- [ ] 주 1회 계획 재시작 스케줄 설정
```

---

## Grafana 대시보드 핵심 패널

실제 운영에 쓰는 대시보드 패널들입니다:

**1. 레이턴시 히트맵** — 시간대별 분포 변화를 한눈에
```
histogram_quantile(0.50, sum by (le) (rate(vision_inference_latency_seconds_bucket[1m])))
histogram_quantile(0.99, sum by (le) (rate(vision_inference_latency_seconds_bucket[1m])))
```

**2. 에러율 게이지** — SLO 달성 여부를 색상으로
```
1 - (
  rate(vision_inspections_total{result="error"}[5m])
  /
  rate(vision_inspections_total[5m])
)
```

**3. GPU 리소스 멀티라인** — 온도·사용률·메모리를 하나의 그래프에

---

## 정리

제조 현장 AI 모니터링의 핵심은:

1. **레이턴시를 P99로 본다** — 평균은 의미 없다
2. **불량률 이상은 즉시 알린다** — 라인 품질 문제일 수 있다
3. **GPU 리소스를 선제 모니터링한다** — 과열·메모리 누수는 서서히 온다
4. **RCA는 템플릿으로 표준화한다** — 야간 장애 후 머리가 안 돌아갈 때를 대비

다음 글에서는 Shadow/Canary 배포로 운영 중인 모델을 무중단 교체하는 방법을 다루겠습니다.
