# ModelLab

**A reproducible benchmarking control plane for finding efficient LLM-serving configurations.**

ModelLab dynamically launches an open-weight model behind vLLM, runs a versioned
workload against its OpenAI-compatible streaming endpoint, measures latency,
throughput, reliability, and deterministic quality, then removes the model
container while retaining its weights and experimental results.

The project answers a practical systems question:

> For this model, GPU, and traffic pattern, which serving configuration gives the
> best latency/throughput tradeoff without reducing output quality?

ModelLab currently supports local mock evaluations and real single-GPU vLLM
evaluations through Docker. It has been exercised on an AWS EC2 `g5.xlarge`
instance with an NVIDIA A10G.

## Measured results

The included experiment evaluated `Qwen/Qwen2.5-3B-Instruct` with vLLM on one
NVIDIA A10G. It completed **29/29 evaluation runs** and **1,499/1,499 measured
requests**, with three independent repetitions for every performance condition.

### Prefix caching

On the shared-prefix workload at concurrency 4:

| Metric | Disabled | Enabled | Change |
|---|---:|---:|---:|
| P50 TTFT | 64.36 ms | 47.15 ms | **26.7% lower** |
| P95 TTFT | 111.02 ms | 57.01 ms | **48.6% lower** |
| P95 end-to-end latency | 467.29 ms | 386.99 ms | **17.2% lower** |
| Output throughput | 179.38 tok/s | 217.08 tok/s | **21.0% higher** |
| Request throughput | 10.41 req/s | 12.37 req/s | **18.8% higher** |

Both configurations scored **51/52 (98.08%)** on the deterministic quality
workload. Prefix caching improved performance without changing measured quality.

### Concurrency scaling

Using the control profile on the short-chat workload:

| Concurrency | P50 TTFT | P95 TTFT | Output tok/s | Requests/s |
|---:|---:|---:|---:|---:|
| 1 | 38.24 ms | 39.54 ms | 65.58 | 1.81 |
| 2 | 45.97 ms | 61.04 ms | 120.97 | 3.36 |
| 4 | 47.99 ms | 63.98 ms | 207.67 | 6.37 |
| 8 | 50.81 ms | 81.77 ms | 337.39 | 10.10 |

Concurrency 8 produced **5.15× the output-token throughput** of concurrency 1,
with a 32.9% increase in median TTFT and a 106.8% increase in P95 TTFT. This is
the central tradeoff ModelLab exposes: the most efficient setting depends on
whether the application prioritizes interactive latency or aggregate throughput.

### Scheduler capacity

At request concurrency 8:

| `max_num_seqs` | P50 TTFT | P95 TTFT | Output tok/s |
|---:|---:|---:|---:|
| 1 | 3,303.21 ms | 5,353.77 ms | 66.39 |
| 8 | 50.81 ms | 81.77 ms | 337.39 |
| 16 | 51.30 ms | 89.86 ms | 345.66 |

Restricting vLLM to one sequence caused severe queueing and reduced throughput
by 80.3%. Increasing the limit from 8 to 16 added only 2.5% throughput while
worsening P95 TTFT by 9.9%, so 8 was the balanced choice for the tested load.

The full methodology, limitations, raw measurements, and per-request outputs are
available in:

- [`experiment-results/A10G_QWEN25_3B_REPORT.md`](experiment-results/A10G_QWEN25_3B_REPORT.md)
- [`experiment-results/a10g-qwen25-3b-20260923T024129Z.csv`](experiment-results/a10g-qwen25-3b-20260923T024129Z.csv)
- [`experiment-results/a10g-qwen25-3b-20260923T024129Z.json`](experiment-results/a10g-qwen25-3b-20260923T024129Z.json)

These are measurements for one model, vLLM version, GPU, and workload suite—not
universal performance claims.

## How it works

```text
                 create profile / deployment / evaluation
Developer  ───────────────────────────────────────────────► FastAPI control plane
                                                                  │
                                                                  ├──► PostgreSQL
                                                                  │    profiles
                                                                  │    deployments
                                                                  │    runs + metrics
                                                                  │
                                               claim queued run   │
Benchmark worker ◄────────────────────────────────────────────────┘
       │
       │ streaming POST /v1/chat/completions
       ▼
Ephemeral vLLM container ───────► NVIDIA GPU
       │
       └── model files cached in a persistent Docker volume on EBS
```

1. A **model profile** records the model and serving parameters to test:
   engine, quantization, prefix caching, maximum concurrent sequences, and
   maximum context length.
2. A **deployment** asks the configured provider to turn that profile into a
   reachable inference endpoint. The Docker provider starts an isolated,
   GPU-enabled vLLM container and waits for its health endpoint.
3. An **evaluation run** references the deployment and an immutable workload
   version. Its workload SHA-256 hash and generation settings are stored with
   the run.
4. The **benchmark worker** atomically claims queued runs, performs excluded
   warm-ups, sends concurrent streaming requests, and records aggregate and
   per-request results.
5. For an ephemeral deployment, the worker stops and removes the vLLM container
   after success or failure. PostgreSQL results and the persistent model cache
   remain available.
6. The comparison API accepts two compatible successful runs and reports
   absolute and percentage metric changes. It rejects mismatched workloads,
   request counts, concurrency, or warm-up counts.

## What is measured

- P50, P95, and P99 **time to first token (TTFT)**
- P50, P95, and P99 **end-to-end latency**
- Aggregate and per-request **output tokens per second**
- **Requests per second**
- Input/output token totals
- Successful and failed requests, including **error rate**
- Generated text and errors for every measured request
- Deterministic quality score, scored count, and passed count

TTFT is measured from request start until the first nonempty streamed `content`
or `reasoning_content` delta. End-to-end latency ends at the stream's `[DONE]`
event. Warm-up traffic is completed before the measurement clock begins.

## Versioned workloads

Definitions live in [`src/modellab/workloads/data`](src/modellab/workloads/data).
Version `2.0.0` contains **112 unique cases**:

| Workload | Cases | Purpose |
|---|---:|---|
| `smoke-test` | 5 | Validate deployment and measurement end to end |
| `short-chat` | 25 | Summarization, rewriting, extraction, and instruction following |
| `shared-prefix` | 15 | Measure the effect of prefix-cache reuse |
| `long-context` | 15 | Exercise prompt processing with a longer shared context |
| `quality` | 52 | Deterministically scored arithmetic, factual, multiple-choice, containment, exact-match, and JSON tasks |

Quality scoring supports exact match, normalized exact match, multiple choice,
required-term containment, numeric tolerance, valid JSON, and a deliberately
small JSON Schema subset. Version `1.0.0` remains available so historical runs
remain reproducible.

## Stack

- Python 3.13
- FastAPI and Pydantic
- PostgreSQL 16 and SQLAlchemy
- Docker Compose and Docker Engine API
- vLLM's OpenAI-compatible server
- HTTPX streaming client
- Pytest
- AWS EC2 GPU infrastructure for the recorded experiment

## Quick start: local mock path

The default Compose stack does not need a GPU or access to the Docker socket.
It runs PostgreSQL, the API, the worker, and a deterministic OpenAI-compatible
mock endpoint.

```bash
git clone https://github.com/sinhaanshul1/model-lab.git
cd model-lab
docker compose up --build -d
curl http://127.0.0.1:8000/health
```

Interactive API documentation is available at <http://127.0.0.1:8000/docs>.

Create a mock profile:

```bash
PROFILE_JSON=$(curl -sS http://127.0.0.1:8000/v1/model-profiles \
  -H 'Content-Type: application/json' \
  -d '{
    "name": "local-mock-control",
    "model": "modellab/mock-1",
    "engine": "mock",
    "prefix_caching": false,
    "max_concurrent_sequences": 8,
    "max_context_tokens": 4096
  }')
echo "$PROFILE_JSON"
```

Copy the returned `id`, then deploy it:

```bash
curl -sS http://127.0.0.1:8000/v1/model-profiles/PROFILE_ID/deployments \
  -H 'Content-Type: application/json' \
  -d '{"provider":"mock","lifecycle_policy":"ephemeral"}'
```

Copy the deployment `id` and queue an evaluation:

```bash
curl -sS http://127.0.0.1:8000/v1/evaluation-runs \
  -H 'Content-Type: application/json' \
  -d '{
    "model_deployment_id": "DEPLOYMENT_ID",
    "workload_name": "smoke-test",
    "workload_version": "2.0.0",
    "warmup_request_count": 2,
    "request_count": 5,
    "concurrency": 1
  }'
```

Poll `GET /v1/evaluation-runs/RUN_ID` to retrieve the final metrics.

Stop the local stack without deleting its PostgreSQL volume:

```bash
docker compose down
```

## Real GPU path

The GPU overlay deliberately requires an explicit opt-in because mounting
`/var/run/docker.sock` gives the API and worker container-management privileges.
Use it only on a trusted development host.

Prerequisites:

- Linux host with an NVIDIA GPU, working driver, Docker, Compose, and NVIDIA
  Container Toolkit
- Enough GPU memory and disk capacity for the selected model
- A pinned vLLM image
- Network access to download a public Hugging Face model

```bash
sudo env MODELLAB_VLLM_IMAGE=vllm/vllm-openai:v0.21.0 \
  docker compose \
  -f docker-compose.yaml \
  -f docker-compose.docker-provider.yaml \
  up --build -d
```

Create a vLLM model profile:

```bash
curl -sS http://127.0.0.1:8000/v1/model-profiles \
  -H 'Content-Type: application/json' \
  -d '{
    "name": "qwen-2.5-3b-control",
    "model": "Qwen/Qwen2.5-3B-Instruct",
    "engine": "vllm",
    "quantization": null,
    "prefix_caching": false,
    "max_concurrent_sequences": 8,
    "max_context_tokens": 4096
  }'
```

Deploy it with provider `docker`, then submit evaluations exactly as in the mock
workflow. The first deployment downloads model files; later deployments reuse
the named `modellab-model-cache` volume. See
[`EC2_DEVELOPMENT.md`](EC2_DEVELOPMENT.md) for the AWS runbook and cost-safety
checklist.

## API overview

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/health` | API health check |
| `GET` | `/v1/workloads` | List immutable workload versions and hashes |
| `GET` | `/v1/workloads/{name}/{version}` | Read a workload definition |
| `POST` | `/v1/model-profiles` | Create a serving configuration |
| `GET` | `/v1/model-profiles` | List profiles |
| `GET` | `/v1/model-profiles/{id}` | Read one profile |
| `POST` | `/v1/model-profiles/{id}/deployments` | Launch a mock or Docker/vLLM deployment |
| `GET` | `/v1/model-deployments/{id}` | Read deployment status and endpoint metadata |
| `POST` | `/v1/model-deployments/{id}/stop` | Stop a ready deployment |
| `POST` | `/v1/evaluation-runs` | Queue a benchmark |
| `GET` | `/v1/evaluation-runs` | List evaluations |
| `GET` | `/v1/evaluation-runs/{id}` | Read status, aggregates, and per-request results |
| `POST` | `/v1/evaluation-comparisons` | Compare compatible completed runs |

## Testing

The suite covers workload validation and hashing, scoring, streaming TTFT,
partial request failures, deployment state transitions, Docker/vLLM command
construction, comparison fairness, PostgreSQL persistence, and the complete
API-to-worker-to-cleanup lifecycle.

With a Python virtual environment and dedicated `modellab_test` PostgreSQL
database configured:

```bash
python -m pip install -r requirements.txt
export PYTHONPATH=src
export MODELLAB_TEST_DATABASE_URL=postgresql+psycopg://localhost/modellab_test
pytest -q
```

The current suite contains **28 passing tests**.

## Persistence and cleanup

- `postgres_data` stores profiles, deployments, evaluation runs, and metrics.
- `modellab-model-cache` stores downloaded Hugging Face artifacts.
- Stopping or removing an ephemeral vLLM container unloads GPU memory but does
  not delete either volume.
- `docker compose down` preserves named volumes unless `--volumes` is supplied.
- Stopping EC2 ends compute charges, but EBS storage continues to be billed.

Do not expose port 8000 publicly on EC2. Bind it locally through an SSH tunnel,
and restrict inbound SSH to your current IP.

## Current scope and limitations

Implemented today:

- Mock and Docker/vLLM deployment providers
- One dynamically created model container per deployment
- Single-GPU benchmarking
- Streaming performance metrics and per-request evidence
- Versioned workloads and deterministic quality scoring
- PostgreSQL persistence and fair comparison validation
- Ephemeral container cleanup and persistent weight caching

Not yet implemented:

- Automatic EC2 provisioning and shutdown through the API
- Multi-GPU or multi-replica orchestration
- SGLang runtime provider (the enum exists, but the Docker provider is vLLM-only)
- GPU utilization, GPU-memory, power, and KV-cache telemetry
- Cost-per-token calculation from cloud billing data
- Authentication, multi-tenancy, production queueing, or a web dashboard
- General-purpose JSON Schema validation or model-as-judge scoring
- A versioned migration framework; development schema upgrades are currently additive

## Repository guide

```text
src/modellab/api/           FastAPI endpoints and request/response models
src/modellab/deployments/   Deployment manager and mock/Docker providers
src/modellab/storage/       SQLAlchemy records, repositories, and schema setup
src/modellab/worker/        Queue polling, streaming requests, and metrics
src/modellab/workloads/     Versioned definitions, registry, and scorers
tests/                      Unit and PostgreSQL-backed integration tests
experiment-results/         Raw A10G evidence and human-readable analysis
```

For deeper methodology, see [`BENCHMARKING.md`](BENCHMARKING.md).
