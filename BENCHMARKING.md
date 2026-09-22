# Benchmarking with ModelLab

ModelLab runs versioned, reproducible workloads against an OpenAI-compatible
deployment. The worker streams every response so it measures user-visible
latency rather than merely timing a completed HTTP request.

## Workloads

Built-in JSON definitions live in `src/modellab/workloads/data`:

- `smoke-test`: quick end-to-end validation
- `short-chat`: varied everyday instructions
- `shared-prefix`: repeated context for prefix-cache comparisons
- `long-context`: prompt-processing stress
- `quality`: deterministic exact, numeric, containment, and JSON checks

`GET /v1/workloads` lists them. `GET /v1/workloads/{name}/{version}` returns the
full immutable definition. Each evaluation stores the workload version, SHA-256
content hash, and generation settings, proving later comparisons used identical
prompts.

## Run an evaluation

```bash
curl -sS http://127.0.0.1:8000/v1/evaluation-runs \
  -H 'Content-Type: application/json' \
  -d '{
    "model_deployment_id": "YOUR_DEPLOYMENT_ID",
    "workload_name": "short-chat",
    "workload_version": "1.0.0",
    "warmup_request_count": 5,
    "request_count": 100,
    "concurrency": 4
  }'
```

Warm-up requests use the same workload but complete before the measurement clock
starts. Their results are intentionally excluded.

## Metrics

- **TTFT**: request start to first nonempty generated content or reasoning delta.
- **End-to-end latency**: request start to the final `[DONE]` event.
- **Output throughput**: successful completion tokens divided by measured wall time.
- **Request throughput**: successful requests divided by measured wall time.
- **Error rate**: failed measured requests divided by scheduled measured requests.
- **Quality score**: mean deterministic case score; unscored cases are excluded.

P50, P95, and P99 use nearest rank. Every measured request retains its case ID,
success/error, token counts, generated text, timings, output rate, and quality
score. A run succeeds when at least one measured request succeeds; individual
failures remain visible. It fails if warm-up cannot complete or every measured
request fails. Ephemeral deployments are cleaned up after either outcome.

## Compare two runs

Run the same workload, request count, concurrency, and warm-up count against two
model profiles, then call:

```bash
curl -sS http://127.0.0.1:8000/v1/evaluation-comparisons \
  -H 'Content-Type: application/json' \
  -d '{
    "baseline_run_id": "BASELINE_RUN_ID",
    "candidate_run_id": "CANDIDATE_RUN_ID"
  }'
```

The API rejects mismatched experiments and returns absolute and percentage
changes for available latency, throughput, reliability, and quality metrics.
A negative latency percentage is faster; positive throughput or quality is better.

## Recommended experiment matrix

Change one profile setting at a time and repeat each run at least three times:

1. Prefix caching off vs. on with `shared-prefix`.
2. Concurrency 1, 2, 4, and 8 with `short-chat`.
3. Default precision vs. supported quantization using identical workloads.
4. Short vs. long context using the same deployment.
5. Every performance candidate against `quality` to catch regressions.

Chunk intervals are not labeled inter-token latency because an SSE chunk can
contain more than one token. GPU and KV-cache utilization still require separate
runtime telemetry and are not inferred from response timing.
