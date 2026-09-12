# EC2 GPU development runbook

This runbook starts the complete ModelLab development stack on one GPU-backed
EC2 instance. The API, worker, and PostgreSQL use CPU resources; only vLLM model
containers request the GPU.

## Before connecting

- Allow inbound SSH (TCP 22) only from your current IP address.
- Do not expose port 8000 publicly. Use the SSH tunnel below.
- Set the instance-initiated shutdown behavior to **Stop**.

## 1. Connect and set a cost-safety timer

Replace the key path and public address with this instance's values:

```bash
ssh -i /path/to/modellab.pem ubuntu@EC2_PUBLIC_IP
```

Immediately schedule a shutdown. This example allows three hours:

```bash
sudo shutdown -h +180
```

Cancel the timer only when intentionally replacing it:

```bash
sudo shutdown -c
```

## 2. Verify GPU container access

```bash
nvidia-smi
docker --version
sudo systemctl enable --now docker
sudo docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu24.04 nvidia-smi
```

Both `nvidia-smi` commands must show the NVIDIA GPU before continuing.

## 3. Clone ModelLab

```bash
git clone https://github.com/sinhaanshul1/model-lab.git
cd model-lab
```

For a private repository, use a GitHub deploy key or a fine-grained token with
read-only repository access. Do not put a token in the clone URL or commit it.

## 4. Start the GPU-enabled stack

The named `modellab-model-cache` volume lives under Docker's data root on the
persistent EBS disk. Removing an ephemeral deployment container does not remove
its downloaded model weights.

```bash
export MODELLAB_VLLM_IMAGE=vllm/vllm-openai:v0.21.0
sudo -E docker compose \
  -f docker-compose.yaml \
  -f docker-compose.docker-provider.yaml \
  up --build -d
```

Verify the control plane:

```bash
sudo -E docker compose \
  -f docker-compose.yaml \
  -f docker-compose.docker-provider.yaml \
  ps
curl http://127.0.0.1:8000/health
```

## 5. Reach the API from the development computer

Open a second local terminal and keep this tunnel open:

```bash
ssh -i /path/to/modellab.pem \
  -N -L 8000:127.0.0.1:8000 \
  ubuntu@EC2_PUBLIC_IP
```

The API is now available locally at `http://127.0.0.1:8000`, with interactive
documentation at `http://127.0.0.1:8000/docs`.

## 6. Run a small real-model smoke test

Create a profile through the local SSH tunnel:

```bash
curl -sS http://127.0.0.1:8000/v1/model-profiles \
  -H 'Content-Type: application/json' \
  -d '{
    "name": "qwen-2.5-3b-smoke",
    "model": "Qwen/Qwen2.5-3B-Instruct",
    "engine": "vllm",
    "quantization": null,
    "prefix_caching": false,
    "max_concurrent_sequences": 8,
    "max_context_tokens": 4096
  }'
```

Copy the returned profile `id`, then create a deployment. The first request may
take several minutes while Docker and Hugging Face download artifacts:

```bash
curl -sS http://127.0.0.1:8000/v1/model-profiles/PROFILE_ID/deployments \
  -H 'Content-Type: application/json' \
  -d '{"provider":"docker","lifecycle_policy":"ephemeral"}'
```

Copy the returned deployment `id`, then queue an evaluation:

```bash
curl -sS http://127.0.0.1:8000/v1/evaluation-runs \
  -H 'Content-Type: application/json' \
  -d '{
    "model_deployment_id": "DEPLOYMENT_ID",
    "workload_name": "gpu-smoke-test",
    "request_count": 3,
    "concurrency": 1
  }'
```

Copy the returned run `id` and check it until it succeeds:

```bash
curl -sS http://127.0.0.1:8000/v1/evaluation-runs/RUN_ID
```

The worker removes the ephemeral vLLM container after the benchmark. Its weights
remain in `modellab-model-cache` for future deployments.

## 7. Stop work and verify the EC2 state

```bash
sudo -E docker compose \
  -f docker-compose.yaml \
  -f docker-compose.docker-provider.yaml \
  down
sudo shutdown -h now
```

Do not add `--volumes` to `docker compose down`; that would remove PostgreSQL's
development data. Verify in the EC2 console that the instance reaches
**Stopped**.
