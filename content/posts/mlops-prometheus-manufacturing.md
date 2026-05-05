---
title: "제조 현장 MLOps — Prometheus/Grafana로 AI 추론 시스템 완전 모니터링"
date: 2026-04-10
tags: ["MLOps", "Prometheus", "Grafana", "모니터링", "SLA", "SLO", "RCA", "제조AI", "Observability"]
categories: ["MLOps & 운영"]
description: "자동차 생산 라인 AI 추론 시스템의 운영 모니터링 체계 전체. 관측가능성(Observability) 이론, SLA/SLO 설계, Prometheus 메트릭 구현, 장애 대응 RCA까지."
showToc: true
tocopen: true
---

## 제조 MLOps는 왜 다른가

일반적인 웹 서비스 MLOps와 제조 현장 MLOps는 근본적으로 다릅니다.

| 항목 | 웹 서비스 | 제조 현장 AI |
|------|---------|------------|
| 레이턴시 허용 | 수백ms | **55ms 이하 (사이클타임)** |
| 장애 영향 | 사용자 불편 | **라인 정지 → 직접 손실** |
| 오류 방향성 | FP/FN 균형 | **FN(미검출) 최소화 절대 원칙** |
| 데이터 드리프트 원인 | 사용자 행동 변화 | **조명 변동, 렌즈 오염, 부품 교체** |
| 모델 업데이트 | 롤백 쉬움 | **라인 정지 없이 교체 필요** |

---

## 이론: 관측가능성(Observability) 3대 축

Google SRE 방법론에서 정의하는 관측가능성의 3대 축:

```
┌─────────────────────────────────────────────────────────────┐
│                    Observability                            │
├──────────────┬──────────────────────┬───────────────────────┤
│   Metrics    │       Logs           │      Traces           │
│  (무슨 일)   │   (왜 일어났나)      │   (어디서 일어났나)   │
│              │                      │                       │
│  Prometheus  │  Loki / ELK Stack    │  Jaeger / Tempo       │
│  Grafana     │  journalctl          │  OpenTelemetry        │
│  AlertManager│                      │                       │
└──────────────┴──────────────────────┴───────────────────────┘
```

비전 검사 시스템에서는 **Metrics가 핵심**입니다. 레이턴시·처리량·불량률이 SLA에 직결되기 때문입니다.

---

## 모니터링 스택 아키텍처

```
[TRT 추론 엔진]
      │
      │ HTTP /metrics (Prometheus 형식)
      ▼
[Custom Exporter]  ←── GPU 메트릭 (nvidia-smi)
      │
      ▼
[Prometheus Server]  ←── scrape_interval: 15s
      │
      ├──▶ [Grafana]        — 대시보드 시각화
      │
      └──▶ [AlertManager]   — 알림 라우팅
                │
                ├──▶ PagerDuty  (Critical — P1)
                ├──▶ Slack      (Warning  — P2)
                └──▶ Email      (Info     — P3)
```

---

## SLA / SLO / SLI 설계

### 세 개념의 계층 구조

```
SLA (Service Level Agreement) — 고객·운영팀과의 외부 약속
  └── SLO (Service Level Objective) — 내부 달성 목표 (SLA보다 엄격)
        └── SLI (Service Level Indicator) — 실제 측정 지표
```

### 비전 검사 시스템 SLI/SLO 설계

| SLI | 측정 방법 | SLO (내부 목표) | SLA (외부 약속) |
|-----|---------|--------------|--------------|
| 추론 P99 레이턴시 | Histogram 분위수 | < 45ms | < 55ms |
| 가용성 | 비오류 요청 비율 | 99.95% | 99.5% |
| 불량 검출 정확도 | 주기적 골든셋 검증 | ≥ 99.7% | ≥ 99.5% |
| GPU 메모리 사용률 | nvidia-smi | < 85% | < 90% |

> SLO는 SLA보다 항상 엄격해야 합니다. SLO가 깨졌을 때 대응하면 SLA는 지킬 수 있습니다.

### 에러 버짓 (Error Budget)

```
가용성 SLO = 99.95%
30일 에러 버짓 = 30일 × 24h × 60min × (1 - 0.9995) = 21.6분

→ 한 달에 21.6분 이상 장애가 나면 SLO 위반
→ 이 예산을 아끼면 배포 속도를 높이거나 실험적 기능을 더 배포 가능
```

---

## 비전 검사 특화 메트릭 구현

```python
from prometheus_client import (
    Counter, Histogram, Gauge, Summary,
    start_http_server, CollectorRegistry
)
import time
import threading


REGISTRY = CollectorRegistry()

# === 처리량 ===
INSPECTIONS_TOTAL = Counter(
    'vision_inspections_total',
    '검사 완료 건수',
    ['camera_id', 'result'],        # result: pass / defect / error / timeout
    registry=REGISTRY,
)

# === 레이턴시 ===
INFERENCE_LATENCY = Histogram(
    'vision_inference_latency_seconds',
    '추론 레이턴시 분포',
    ['camera_id', 'model_version'],
    # 사이클타임 55ms 기준 세밀한 버킷 설계
    buckets=[0.010, 0.015, 0.020, 0.025, 0.030, 0.035, 0.040,
             0.045, 0.050, 0.055, 0.060, 0.080, 0.100, 0.200],
    registry=REGISTRY,
)

# === 모델 품질 ===
MODEL_CONFIDENCE = Histogram(
    'vision_model_confidence_score',
    '모델 신뢰도 점수 분포',
    ['camera_id', 'result'],
    buckets=[0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.98, 0.99, 1.0],
    registry=REGISTRY,
)

DEFECT_RATE = Gauge(
    'vision_defect_rate_sliding',
    '슬라이딩 윈도우 불량률',
    ['line_id', 'window_minutes'],
    registry=REGISTRY,
)

# === GPU 리소스 ===
GPU_UTILIZATION  = Gauge('vision_gpu_utilization_pct',  'GPU 사용률 (%)', ['device'], registry=REGISTRY)
GPU_MEMORY_USED  = Gauge('vision_gpu_memory_used_mb',   'GPU 메모리 사용 (MB)', ['device'], registry=REGISTRY)
GPU_TEMPERATURE  = Gauge('vision_gpu_temperature_c',    'GPU 온도 (°C)', ['device'], registry=REGISTRY)
GPU_POWER_DRAW   = Gauge('vision_gpu_power_draw_w',     'GPU 전력 소비 (W)', ['device'], registry=REGISTRY)

# === 파이프라인 단계별 레이턴시 (병목 분석용) ===
STAGE_LATENCY = Histogram(
    'vision_stage_latency_seconds',
    '파이프라인 단계별 레이턴시',
    ['camera_id', 'stage'],  # stage: capture / preprocess / infer / postprocess / plc
    buckets=[0.001, 0.002, 0.005, 0.010, 0.020, 0.030, 0.050],
    registry=REGISTRY,
)


class VisionMetricsExporter:
    def __init__(self, camera_id: str, model_version: str, port: int = 8000):
        self.camera_id = camera_id
        self.model_version = model_version
        start_http_server(port, registry=REGISTRY)
        self._start_gpu_collector()

    def record_inspection(self, latency_sec: float, result: str,
                          confidence: float, stage_times: dict):
        INSPECTIONS_TOTAL.labels(
            camera_id=self.camera_id, result=result
        ).inc()

        INFERENCE_LATENCY.labels(
            camera_id=self.camera_id, model_version=self.model_version
        ).observe(latency_sec)

        MODEL_CONFIDENCE.labels(
            camera_id=self.camera_id, result=result
        ).observe(confidence)

        # 단계별 레이턴시 기록
        for stage, t in stage_times.items():
            STAGE_LATENCY.labels(
                camera_id=self.camera_id, stage=stage
            ).observe(t)

    def _collect_gpu_metrics(self):
        import subprocess, re
        while True:
            try:
                out = subprocess.check_output([
                    'nvidia-smi',
                    '--query-gpu=index,utilization.gpu,memory.used,temperature.gpu,power.draw',
                    '--format=csv,noheader,nounits'
                ]).decode()
                for line in out.strip().split('\n'):
                    idx, util, mem, temp, pwr = [x.strip() for x in line.split(',')]
                    device = f'gpu{idx}'
                    GPU_UTILIZATION.labels(device=device).set(float(util))
                    GPU_MEMORY_USED.labels(device=device).set(float(mem))
                    GPU_TEMPERATURE.labels(device=device).set(float(temp))
                    GPU_POWER_DRAW.labels(device=device).set(float(pwr))
            except Exception:
                pass
            time.sleep(5)

    def _start_gpu_collector(self):
        t = threading.Thread(target=self._collect_gpu_metrics, daemon=True)
        t.start()
```

---

## Prometheus 알림 규칙 (alert_rules.yml)

```yaml
groups:
  - name: vision_ai_slo
    interval: 15s
    rules:

      # P99 레이턴시 SLO 위반 — Critical
      - alert: InferenceLatencyP99Critical
        expr: |
          histogram_quantile(0.99,
            sum by (le, camera_id) (
              rate(vision_inference_latency_seconds_bucket[5m])
            )
          ) > 0.045
        for: 2m
        labels:
          severity: critical
          team: ai-ops
        annotations:
          summary: "P99 레이턴시 SLO 위반 — {{ $labels.camera_id }}"
          description: |
            현재 P99: {{ $value | humanizeDuration }}
            SLO 기준: 45ms (SLA 기준: 55ms)
            즉시 확인 필요

      # GPU 온도 경고
      - alert: GPUTemperatureHigh
        expr: vision_gpu_temperature_c > 82
        for: 3m
        labels:
          severity: warning
        annotations:
          summary: "GPU 온도 경고 — {{ $labels.device }}"
          description: "현재 {{ $value }}°C. 냉각 확인 필요"

      # GPU 온도 위험 (즉시 조치)
      - alert: GPUTemperatureCritical
        expr: vision_gpu_temperature_c > 90
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "GPU 온도 위험 — {{ $labels.device }}"
          description: "현재 {{ $value }}°C. 즉시 냉각 조치 필요. 서비스 중단 위험."

      # 불량률 급증
      - alert: DefectRateSpike
        expr: |
          vision_defect_rate_sliding{window_minutes="60"}
          > 2 * (vision_defect_rate_sliding{window_minutes="60"} offset 1h)
          AND
          vision_defect_rate_sliding{window_minutes="60"} > 0.02
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "불량률 급증 — 라인 {{ $labels.line_id }}"
          description: |
            현재 불량률: {{ $value | humanizePercentage }}
            전 시간 대비 2배 이상 증가. 공정 확인 필요.

      # 모델 신뢰도 하락 (드리프트 조기 감지)
      - alert: ModelConfidenceDrop
        expr: |
          histogram_quantile(0.5,
            rate(vision_model_confidence_score_bucket{result="pass"}[30m])
          ) < 0.80
        for: 15m
        labels:
          severity: warning
        annotations:
          summary: "모델 신뢰도 하락 — {{ $labels.camera_id }}"
          description: |
            Pass 판정의 중위 신뢰도가 0.80 미만.
            조명 변동 또는 모델 드리프트 가능성 확인 필요.

      # 에러 버짓 소진율
      - alert: ErrorBudgetBurnRateFast
        expr: |
          (
            rate(vision_inspections_total{result="error"}[1h])
            / rate(vision_inspections_total[1h])
          ) > 14.4 * (1 - 0.9995)
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "에러 버짓 빠른 소진"
          description: "현재 속도로는 {{ $value | humanizeDuration }} 내 월간 에러 버짓 소진"
```

---

## Grafana 핵심 쿼리

**대시보드 패널 1: 레이턴시 분위수 시계열**
```promql
# P50
histogram_quantile(0.50,
  sum by (le) (rate(vision_inference_latency_seconds_bucket[1m]))
) * 1000   # ms 단위

# P99
histogram_quantile(0.99,
  sum by (le) (rate(vision_inference_latency_seconds_bucket[1m]))
) * 1000
```

**대시보드 패널 2: 실시간 가용성 (SLO 달성률)**
```promql
100 * (
  1 - (
    rate(vision_inspections_total{result="error"}[5m])
    / rate(vision_inspections_total[5m])
  )
)
```

**대시보드 패널 3: 파이프라인 단계별 레이턴시 히트맵**
```promql
sum by (le, stage) (
  rate(vision_stage_latency_seconds_bucket[5m])
)
```

**대시보드 패널 4: 카메라별 FPS (처리량)**
```promql
sum by (camera_id) (
  rate(vision_inspections_total[1m])
) * 60
```

---

## 장애 대응 — RCA 프로세스

### 5단계 대응 체계

```
1단계 (0~5분):  즉각 감지 · 영향 범위 파악 · 서비스 복구
2단계 (5~30분): 근본 원인 가설 수립 · 데이터 수집
3단계 (30분~):  RCA 문서 작성
4단계 (1~3일):  재발 방지 조치 실행
5단계 (1주):   효과 검증 · 지식 공유
```

### 즉각 대응 명령어

```bash
# 현재 상태 확인
curl -s http://localhost:9090/api/v1/query \
  --data-urlencode 'query=up{job="vision-ai"}' | python3 -m json.tool

# P99 레이턴시 현재값
curl -s http://localhost:9090/api/v1/query \
  --data-urlencode 'query=histogram_quantile(0.99, rate(vision_inference_latency_seconds_bucket[5m]))' \
  | python3 -c "import sys,json; d=json.load(sys.stdin); print(float(d['data']['result'][0]['value'][1])*1000, 'ms')"

# GPU 상태
nvidia-smi --query-gpu=name,temperature.gpu,utilization.gpu,memory.used,memory.total \
  --format=csv,noheader

# 최근 에러 로그
journalctl -u vision-inference -n 50 --since "15 min ago" --no-pager

# 추론 서비스 재시작 (마지막 수단)
sudo systemctl restart vision-inference
```

### RCA 문서 템플릿

```markdown
## 인시던트 개요
- **ID:** INC-2026-042
- **발생:** 2026-04-10 14:32 KST
- **감지:** Prometheus Alert (InferenceLatencyP99Critical)
- **영향:** CAM-03, CAM-04 (용접 라인 2구역) — P99 82ms (SLA 55ms 위반)
- **기간:** 14:32 ~ 15:01 (29분)
- **심각도:** P1 (SLA 위반, 라인 속도 저하)

## 타임라인
| 시각 | 이벤트 |
|------|--------|
| 14:32 | Alert 발생 (P99 > 45ms) |
| 14:35 | 온-콜 응답 |
| 14:38 | GPU 메모리 97% 확인 (정상 65%) |
| 14:41 | 메모리 누수 패턴 확인 (7일 전부터 누적) |
| 14:55 | 추론 서비스 재시작 |
| 15:01 | P99 23ms 복구 확인 |

## 5-Why 근본 원인 분석
1. **Why P99 레이턴시가 높았나?** GPU 메모리 97% → 페이지 폴트 증가
2. **Why GPU 메모리가 97%?** TRT 컨텍스트 소멸 시 메모리 풀 미해제
3. **Why 메모리 풀이 미해제?** `context.__del__` 미구현, 명시적 해제 코드 없음
4. **Why 오래 지속됐나?** GPU 메모리 알림이 90% 기준 — 빠른 감지 불가
5. **Why 알림 기준이 90%?** 초기 설계 시 드리프트 패턴 미고려

## 재발 방지 조치
- [x] `context.destroy()` 명시적 호출 코드 추가
- [x] GPU 메모리 알림 기준 90% → 80%로 강화
- [x] 주간 계획 재시작 스케줄 추가 (일요일 03:00)
- [ ] 메모리 추세 기반 사전 알림 추가 (7일 선형 추세 > +2%/일)

## 교훈
GPU 메모리 누수는 서서히 진행됩니다. P99 히트맵을 주간 단위로 검토하면
조기 감지가 가능합니다. 급격한 스파이크만 보지 말고 **완만한 성능 저하 트렌드**를
주시해야 합니다.
```

---

## Shadow / Canary 배포 — 무중단 모델 교체

```python
import random

class ShadowDeployment:
    """
    Shadow 배포: 새 모델이 운영 모델과 함께 실행되지만 결과는 적용 안 됨
    → 실제 운영 데이터로 새 모델 검증
    """
    def __init__(self, production_engine, shadow_engine, shadow_ratio=1.0):
        self.production = production_engine
        self.shadow = shadow_engine
        self.shadow_ratio = shadow_ratio     # 1.0 = 100% 요청에 shadow 실행

    def infer(self, image):
        prod_result = self.production.infer(image)   # 항상 실행, 결과 적용

        if random.random() < self.shadow_ratio:
            shadow_result = self.shadow.infer(image) # 비교용, 결과 미적용
            self._log_comparison(prod_result, shadow_result)

        return prod_result   # 항상 production 결과 반환


class CanaryDeployment:
    """
    Canary 배포: 일부 요청만 새 모델로 처리
    → 점진적으로 비율을 높여 위험 최소화
    """
    def __init__(self, production_engine, canary_engine, canary_ratio=0.05):
        self.production = production_engine
        self.canary = canary_engine
        self.canary_ratio = canary_ratio    # 5% → 10% → 25% → 50% → 100%

    def infer(self, image):
        if random.random() < self.canary_ratio:
            result = self.canary.infer(image)
            INSPECTIONS_TOTAL.labels(camera_id='*', result='canary').inc()
        else:
            result = self.production.infer(image)
        return result

    def increase_canary(self, new_ratio: float):
        """P99 레이턴시와 불량률 검증 후 비율 증가"""
        assert 0 < new_ratio <= 1.0
        self.canary_ratio = new_ratio
        print(f"Canary 비율 변경: {new_ratio:.0%}")
```

---

## 데이터 드리프트 감지

```python
from scipy import stats
import numpy as np
from collections import deque


class ConfidenceDistributionMonitor:
    """
    모델 신뢰도 분포 변화로 드리프트를 조기 감지
    KS 검정(Kolmogorov-Smirnov)으로 분포 비교
    """
    def __init__(self, window_size: int = 1000, alpha: float = 0.05):
        self.reference = deque(maxlen=window_size)  # 기준 분포 (배포 초기)
        self.current   = deque(maxlen=window_size)  # 현재 분포
        self.alpha = alpha                           # 유의 수준
        self.is_reference_set = False

    def update(self, confidence: float):
        if not self.is_reference_set:
            self.reference.append(confidence)
            if len(self.reference) == self.reference.maxlen:
                self.is_reference_set = True
        else:
            self.current.append(confidence)
            if len(self.current) >= 100:   # 충분한 샘플 모이면 검정
                self._run_ks_test()

    def _run_ks_test(self):
        stat, p_value = stats.ks_2samp(
            list(self.reference), list(self.current)
        )
        if p_value < self.alpha:
            print(f"[DRIFT DETECTED] KS stat={stat:.4f}, p={p_value:.4f}")
            print("→ 조명 변동, 렌즈 오염, 또는 제품 변경 가능성 확인 필요")
        return stat, p_value
```

---

## 참고 자료

- Google SRE Book: [sre.google/sre-book](https://sre.google/sre-book/table-of-contents/)
- Prometheus 공식 문서: [prometheus.io/docs](https://prometheus.io/docs/introduction/overview/)
- Grafana 대시보드 예시: [grafana.com/grafana/dashboards](https://grafana.com/grafana/dashboards/)
- AlertManager 설정: [prometheus.io/docs/alerting/latest/alertmanager](https://prometheus.io/docs/alerting/latest/alertmanager/)
- MLOps 성숙도 모델 (Google): [cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning](https://cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning)
- KS 검정 (드리프트 감지): [scipy.stats.ks_2samp](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ks_2samp.html)
- NVIDIA GPU 모니터링: [docs.nvidia.com/deploy/nvml-api](https://docs.nvidia.com/deploy/nvml-api/index.html)
