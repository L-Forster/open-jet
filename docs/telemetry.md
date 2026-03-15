# Telemetry

`open-jet` emits real OpenTelemetry logs, traces, and metrics. The app does not write its own telemetry files anymore; it exports OTLP/HTTP to an OpenTelemetry Collector when telemetry is enabled.

## What is sent

Telemetry is intentionally narrow:

- anonymous install id
- session id and turn id
- timestamps
- runtime name, device profile, context window, and safe model name
- TTFT and turn/tool durations
- tool names, approval decisions, and high-level shell command classification
- error type, redacted error summary, and error hash
- CPU, memory, load average, and process RSS metrics

It does not send:

- prompt text
- tool stdout/stderr
- file paths
- model filesystem paths
- tool arguments that could leak private data

## Client config

`config.yaml` supports:

```yaml
logging:
  directory: session_logs
  enabled: true
  retention_days: 30
  max_sessions: 100

telemetry:
  install_id_path: .openjet/state/telemetry_identity.json
  broadcast:
    enabled: false
    endpoint: https://your-collector.example.com
    timeout_seconds: 3.0
    export_logs: true
    export_metrics: true
    export_traces: true
    headers:
      x-api-key: your-token
```

`telemetry.broadcast.endpoint` is the collector base URL. `open-jet` appends `/v1/logs`, `/v1/metrics`, and `/v1/traces`.

`logging.directory` now stores session manifests only. Telemetry signal storage belongs to the collector and its configured exporters.

## Recommended server path

Use an OpenTelemetry Collector as the ingress service.

Recommended flow:

1. `open-jet` sends OTLP/HTTP to the collector.
2. The collector batches, filters, retries, and redacts as needed.
3. The collector forwards logs, metrics, and traces to your actual backend or local file exporters.

That is the intended OpenTelemetry architecture. The app owns instrumentation; the collector owns transport and storage.

For a minimal collector receiver config, see [`docs/examples/otel-collector.yaml`](examples/otel-collector.yaml).

## Practical backend choices

Good starting points:

- Grafana stack: Loki for logs, Tempo for traces, Mimir or Prometheus remote write for metrics.
- ClickHouse-based analytics: good if you want custom product/reliability queries.
- Managed OTLP backend: lowest ops burden if self-hosting is not important.

If you only need a first pass, start with the collector and a debug exporter to verify payloads, then wire in a real backend.
