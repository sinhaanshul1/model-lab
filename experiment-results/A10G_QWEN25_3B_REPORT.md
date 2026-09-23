# A10G Qwen2.5-3B configuration experiment

Date: 2026-09-22/23 UTC  
Hardware: AWS EC2 `g5.xlarge`, NVIDIA A10G 23,028 MiB  
Model: `Qwen/Qwen2.5-3B-Instruct` served by vLLM  
Workloads: ModelLab version `2.0.0`  
Measured runs: 29/29 succeeded  
Measured requests: 1,499/1,499 succeeded  
Warm-up requests: 5 per run, excluded from metrics  
Suite wall time: 2,624.8 seconds (43 minutes 44.8 seconds)

Raw evidence:

- `a10g-qwen25-3b-20260923T024129Z.csv`
- `a10g-qwen25-3b-20260923T024129Z.json`

All performance conditions were repeated three times. The tables report the
arithmetic mean of those independent runs. Each comparison held the model,
hardware, workload version, prompts, generation settings, warm-up count, and
measured request count constant unless the named variable was the factor under
test.

## Concurrency scaling

| Concurrency | P50 TTFT (ms) | P95 TTFT (ms) | P95 E2E (ms) | Output tok/s | Requests/s |
|---:|---:|---:|---:|---:|---:|
| 1 | 38.24 | 39.54 | 2,306.93 | 65.58 | 1.81 |
| 2 | 45.97 | 61.04 | 2,455.09 | 120.97 | 3.36 |
| 4 | 47.99 | 63.98 | 1,450.60 | 207.67 | 6.37 |
| 8 | 50.81 | 81.77 | 2,052.50 | 337.39 | 10.10 |

Moving from concurrency 1 to 8 increased output throughput by **414.5%** and
request throughput by **458.4%**. The tradeoff was a **32.9%** increase in P50
TTFT and a **106.8%** increase in P95 TTFT. P95 end-to-end latency was still
**11.0% lower**. For throughput-oriented traffic on this workload, concurrency
8 used the A10G much more efficiently. Latency-sensitive applications may prefer
concurrency 2 or 4.

## Prefix caching

Shared-prefix workload, concurrency 4:

| Setting | P50 TTFT (ms) | P95 TTFT (ms) | P95 E2E (ms) | Output tok/s | Requests/s |
|---|---:|---:|---:|---:|---:|
| Disabled | 64.36 | 111.02 | 467.29 | 179.38 | 10.41 |
| Enabled | 47.15 | 57.01 | 386.99 | 217.08 | 12.37 |

Enabling prefix caching reduced P50 TTFT by **26.7%**, reduced P95 TTFT by
**48.6%**, reduced P95 end-to-end latency by **17.2%**, and increased output
throughput by **21.0%**. Request throughput increased by **18.8%**. This is the
strongest configuration improvement observed without a latency tradeoff.

## Scheduler capacity (`max_num_seqs`)

Short-chat workload, concurrency 8:

| Max sequences | P50 TTFT (ms) | P95 TTFT (ms) | P95 E2E (ms) | Output tok/s | Requests/s |
|---:|---:|---:|---:|---:|---:|
| 1 | 3,303.21 | 5,353.77 | 5,804.58 | 66.39 | 1.83 |
| 8 (control) | 50.81 | 81.77 | 2,052.50 | 337.39 | 10.10 |
| 16 | 51.30 | 89.86 | 2,056.29 | 345.66 | 10.35 |

Restricting the scheduler to one sequence caused queueing: P50 TTFT grew by
approximately **65×**, while output throughput fell **80.3%** relative to the
control. Raising the limit from 8 to 16 produced only a **2.5%** throughput gain,
with P50 TTFT essentially unchanged and P95 TTFT **9.9% worse**. For concurrency
8, `max_num_seqs=8` is the balanced setting; 16 provides no compelling benefit.

## Long-context workload

With the control profile at concurrency 4, the longer shared context averaged:

- P50 TTFT: **75.43 ms**
- P95 TTFT: **185.19 ms**
- P95 end-to-end latency: **2,196.63 ms**
- Output throughput: **223.08 tokens/s**

This demonstrates why configuration advice must account for workload shape:
first-token latency was materially higher than on short-chat prompts even though
the model and server settings were unchanged.

## Quality safeguard

Both the control and prefix-caching profiles scored **51/52 (98.08%)** on the
deterministic quality workload. Both missed the same factual case,
`fact-gpu-company`, answering `AMD` instead of `NVIDIA`. Prefix caching therefore
preserved measured quality while improving shared-prefix performance.

## Product-level conclusion

The experiment demonstrates the product's intended value: the best configuration
depends on workload and service objective, and seemingly harmless limits can have
large quantitative effects. On this hardware/model pair, ModelLab identified:

- prefix caching as a clear shared-prefix win;
- concurrency 8 as a strong throughput configuration with a tail-TTFT cost;
- `max_num_seqs=1` as a severe bottleneck under concurrent load; and
- `max_num_seqs=16` as unnecessary for the tested concurrency.

These results should be described as measurements for this specific model,
hardware, vLLM version, and workload—not universal guarantees. Additional models,
quantization formats, GPUs, and repeated quality trials would broaden the claim.
