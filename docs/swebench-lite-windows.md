# SWE-bench Lite Windows Runbook

This runbook covers the remaining prerequisites and the exact commands needed to run VTM's SWE-bench Lite flow on a Windows machine with an RTX 4090.

Use Ubuntu under WSL2 for the benchmark run. Do not run the workflow from native PowerShell or `cmd.exe`, and keep the repository checkout inside the WSL filesystem, for example `~/src/vtm`, not `/mnt/c/...`.

## What still needs to be done before the run

Repo-side, no extra code changes are required for a normal SWE-bench Lite run.

Machine-side, make sure the Windows host has:

- Windows 11 or a current Windows 10 build with WSL2 enabled
- Docker Desktop configured to use the WSL2 backend
- an NVIDIA driver that supports WSL2 GPU passthrough
- enough free disk for the official harness and Docker images
- Git installed inside Ubuntu
- `uv` installed inside Ubuntu
- Python 3.12 available to `uv`
- a local OpenAI-compatible model endpoint if VTM will generate patches through `scripts/vtm_local_patcher.py`

Recommended capacity for the official SWE-bench harness:

- at least 120 GB of free disk
- at least 16 GB of RAM
- at least 8 CPU cores available to Docker

## Prerequisites

### 1. Install and update WSL2

Run these in an elevated PowerShell window:

```powershell
wsl --install -d Ubuntu-24.04
wsl --update
```

Reboot if Windows asks for it, then open the Ubuntu shell.

### 2. Install Docker Desktop

Install Docker Desktop for Windows and make sure:

- the WSL2 backend is enabled
- your Ubuntu distro has Docker Desktop WSL integration enabled
- Docker Desktop has enough virtual disk allocated for SWE-bench images and caches

Optional GPU validation from Ubuntu:

```bash
docker run --rm -it --gpus=all nvcr.io/nvidia/k8s/cuda-sample:nbody nbody -gpu -benchmark
```

GPU acceleration is useful for your local model server, but the official SWE-bench harness itself is primarily a Docker-based evaluation step.

### 3. Install repo tooling inside Ubuntu

Run these in Ubuntu:

```bash
sudo apt update
sudo apt install -y git curl build-essential
curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"
uv python install 3.12
```

Confirm the toolchain:

```bash
git --version
uv --version
uv python list
```

## Clone and prepare the repository

Clone the repository into the WSL filesystem and install the benchmark dependencies:

```bash
git clone <your-vtm-repo-url> ~/src/vtm
cd ~/src/vtm
uv sync --dev --extra bench
```

Run the targeted benchmark smoke tests before the real run:

```bash
uv run pytest -q tests/test_swebench.py tests/test_benchmark_cli.py
```

## Validate the official SWE-bench harness

Before using VTM, verify that the harness itself can run on the machine:

```bash
uv run python -m swebench.harness.run_evaluation \
  --dataset_name princeton-nlp/SWE-bench_Lite \
  --predictions_path gold \
  --max_workers 1 \
  --instance_ids astropy__astropy-14182 \
  --run_id validate-gold
```

If this fails, fix the Docker or harness environment first. VTM depends on that command path for final evaluation.

## Prepare a targeted SWE-bench Lite manifest

Start with one instance instead of the full Lite set:

```bash
mkdir -p .benchmarks/generated .benchmarks/swebench-lite

uv run python -m vtm.benchmarks.prepare_swebench_lite \
  --output-manifest .benchmarks/generated/swebench-lite.json \
  --cache-root .benchmarks/swebench-lite \
  --repo astropy__astropy \
  --instance astropy__astropy-14182
```

This command downloads the dataset metadata if needed, prepares local repo caches, and creates per-instance base and gold refs inside the cache repo clones.

## Start the local patch generator endpoint

If you want VTM to generate patches through the included local patcher, start an OpenAI-compatible HTTP endpoint on the Windows 4090 machine and then export the environment variables from Ubuntu:

```bash
export VTM_LOCAL_LLM_BASE_URL=http://127.0.0.1:8000
export VTM_LOCAL_LLM_MODEL=<your-model-name>
export VTM_LOCAL_LLM_API_KEY=vtm-local
export PATCHER_SCRIPT="$PWD/scripts/vtm_local_patcher.py"
```

`PATCHER_SCRIPT` should be absolute. The benchmark executor runs from each per-task workspace clone, so a relative `scripts/vtm_local_patcher.py` path is not reliable there.

## Run one targeted VTM SWE-bench Lite job

Use one instance and one harness worker first:

```bash
uv run python -m vtm.benchmarks.run \
  --manifest .benchmarks/generated/swebench-lite.json \
  --suite coding \
  --mode lexical \
  --output .benchmarks/swebench-lite-astropy-14182 \
  --repo astropy__astropy \
  --pair astropy__astropy-14182 \
  --executor-command "python $PATCHER_SCRIPT --task-file {task_file} --workspace {workspace}" \
  --swebench-dataset-name princeton-nlp/SWE-bench_Lite \
  --swebench-harness-workers 1
```

This writes:

- task packs under `.benchmarks/.../task-packs`
- writable workspaces under `.benchmarks/.../workspaces`
- executor logs and patches under `.benchmarks/.../executor-artifacts`
- harness outputs such as `predictions.jsonl`, `swebench_harness_results.json`, `harness.stdout`, `harness.stderr`, and `logs/`

## Scale to the full Lite set

After the targeted run succeeds, prepare the full manifest and run the full evaluation:

```bash
uv run python -m vtm.benchmarks.prepare_swebench_lite \
  --output-manifest .benchmarks/generated/swebench-lite.json \
  --cache-root .benchmarks/swebench-lite

uv run python -m vtm.benchmarks.run \
  --manifest .benchmarks/generated/swebench-lite.json \
  --suite coding \
  --mode lexical \
  --output .benchmarks/swebench-lite-full \
  --executor-command "python $PATCHER_SCRIPT --task-file {task_file} --workspace {workspace}" \
  --swebench-dataset-name princeton-nlp/SWE-bench_Lite \
  --swebench-harness-workers 4
```

Increase `--swebench-harness-workers` only after confirming Docker, disk, and CPU capacity on that machine.

## Troubleshooting

If the VTM run fails, check these files first:

- `.benchmarks/.../harness.stderr`
- `.benchmarks/.../harness.stdout`
- `.benchmarks/.../logs/`
- `.benchmarks/.../predictions.jsonl`
- `.benchmarks/.../swebench_harness_results.json`
- `.benchmarks/.../executor-artifacts/<case-id>/command.stderr`
- `.benchmarks/.../executor-artifacts/<case-id>/produced.patch`
- `.benchmarks/.../workspaces/<mode>/<case-id>/.vtm-local-patcher/response.txt`

Common causes:

- Docker Desktop is installed but WSL integration is not enabled for Ubuntu
- Docker Desktop does not have enough free virtual disk
- the local model server is not listening on `127.0.0.1:8000`
- `VTM_LOCAL_LLM_MODEL` does not match the model name the server expects
- the patcher script path was passed as a relative path instead of an absolute path
- the harness works for `gold` predictions but fails on generated predictions because the produced patch is empty or does not apply cleanly

## References

- Official SWE-bench repository: <https://github.com/SWE-bench/SWE-bench>
- Docker Desktop GPU support for Windows: <https://docs.docker.com/desktop/features/gpu/>
