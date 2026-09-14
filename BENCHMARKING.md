# Benchmark measurement notes

ModelLab's benchmark worker sends OpenAI-compatible streaming chat-completion
requests to a ready deployment. It records timing with a monotonic clock so wall
clock adjustments cannot distort a run.

## Request behavior

Each request currently uses one of three deterministic smoke-test prompts,
`temperature: 0`, `max_tokens: 128`, and these streaming options:

```json
{
  "stream": true,
  "stream_options": {"include_usage": true}
}
```

The configured evaluation concurrency bounds the number of simultaneous HTTP
streams. The smoke workload validates the measurement path; it is not yet a
representative production dataset or a quality evaluation.

## Metrics

- **TTFT** is measured from immediately before sending a request until the first
  nonempty generated `content` or `reasoning_content` delta arrives.
- **End-to-end latency** is measured from immediately before sending a request
  until its `[DONE]` event arrives.
- **Aggregate output tokens per second** is the sum of completion tokens reported
  by all final usage events divided by the wall-clock duration of the evaluation.
- **Per-request output tokens per second** is that request's completion tokens
  divided by its end-to-end latency.
- P50, P95, and P99 use the nearest-rank method across successful requests.

ModelLab persists every request's index, TTFT, end-to-end latency, completion-token
count, and output rate, plus aggregate latency percentiles and throughput.

## Failure rules

The run fails if any scheduled request returns an HTTP error, invalid JSON, no
generated content, no final `[DONE]` marker, or no positive completion-token usage.
An ephemeral deployment is cleaned up after either success or failure.

## Current limitations

- Prompt and output token counts depend on the server's final usage event.
- Chunk arrival intervals are not labeled as inter-token latency because one SSE
  chunk is not guaranteed to contain exactly one token.
- GPU utilization, KV-cache utilization, quality scoring, request-rate pacing,
  warm-up exclusion, and representative versioned datasets are not implemented.
- Small sample percentiles are mathematically valid but not statistically useful;
  serious comparisons require larger repeated runs under controlled conditions.
