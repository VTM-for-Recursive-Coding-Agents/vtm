# Launchers Layout

This directory stores generated execution bundles, not benchmark outputs.

Structure:

- `launchers/local/` for generated local launcher bundles
- `launchers/chpc/` for generated CHPC submission bundles

Each bundle can contain:

- per-provider launch scripts
- a `run_all.sh` or `submit_all.sh` convenience entrypoint
- a `manifest.json` metadata file

For LiveCodeBench method bundles, use `scripts/local/queue_livecodebench_methods.sh` for local runs and `scripts/chpc/queue_livecodebench_methods.sh` for CHPC submission bundles.

For CHPC local-model runs with explicit Slurm resources, prefer `scripts/chpc/submit_livecodebench_local_model.sh`.

Example smoke bundle:

```bash
scripts/local/queue_livecodebench_methods.sh \
	--model qwen3.5-35b-a3b \
	--lm-studio-model-id qwen/qwen3.5-35b-a3b \
	--queue-tag implsmoke \
	--n 1 \
	--evaluate false \
	--max-instances 1 \
	--max-tokens 512 \
	--rlm-max-iterations 2
```

Generated LiveCodeBench bundles can contain four provider launchers:

- `baseline`
- `rag`
- `rlm`
- `rlm_rag`

Example full bundle:

```bash
scripts/local/queue_livecodebench_methods.sh \
	--model qwen3.5-35b-a3b \
	--lm-studio-model-id qwen/qwen3.5-35b-a3b \
	--queue-tag 20260405 \
	--n 10 \
	--evaluate true \
	--max-tokens 512
```

Benchmark outputs remain under `results/raw/`.