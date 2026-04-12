# CHPC Launchers

This directory is reserved for CHPC-specific submission wrappers.

Rules:

- Keep benchmark logic shared. Do not copy LiveCodeBench or SWE-bench runner internals here.
- Put scheduler-specific files here only.
- Generated CHPC submission bundles belong under `launchers/chpc/`.
- Actual benchmark outputs still belong under `results/raw/`.

Current entrypoint:

- `scripts/chpc/setup_chpc.sh`
- `scripts/chpc/run_chpc.sh`
- `scripts/chpc/queue_livecodebench_methods.sh`
- `scripts/chpc/submit_livecodebench_local_model.sh`

## Storage policy

CHPC enforces a 50 GB HOME quota. Do not let benchmark caches, temporary files,
or virtualenvs grow under HOME.

Recommended policy:

- Put CHPC runtime state on scratch-backed VAST storage such as `/scratch/general/vast/$USER/vtm`.
- Keep launcher bundles under `launchers/chpc/` and benchmark outputs under `results/raw/`.
- Treat cache, tmp, pip, Hugging Face, Torch, Triton, and matplotlib state as disposable scratch data.

Helpful CHPC commands:

- `mychpc storage`
- `ncdu $HOME`

The setup and submission scripts now export scratch-backed values for:

- `XDG_CACHE_HOME`
- `HF_HOME`
- `HUGGINGFACE_HUB_CACHE`
- `HF_DATASETS_CACHE`
- `TRANSFORMERS_CACHE`
- `TORCH_HOME`
- `TRITON_CACHE_DIR`
- `MPLCONFIGDIR`
- `PIP_CACHE_DIR`
- `UV_CACHE_DIR`
- `TMPDIR`

Typical flow:

1. Prepare a scratch-backed environment with `scripts/chpc/setup_chpc.sh`.
2. Optionally validate the environment on an interactive GPU node with `scripts/chpc/run_chpc.sh`.
3. Generate and submit a CHPC launcher bundle with `scripts/chpc/submit_livecodebench_local_model.sh`.

For the local-model path, prefer the submission helper. It generates the bundle,
builds the provider commands, and submits all four jobs with explicit Slurm flags.
It now emits absolute repo paths in method commands so remote cluster submissions
do not resolve drivers from the Slurm spool directory, and it routes Slurm logs
to `logs/slurm/` inside the repo.

Setup example:

```bash
scripts/chpc/setup_chpc.sh \
	--storage-root /scratch/general/vast/$USER/vtm \
	--python-module python
```

Fast setup note:

- If `benchmarks/LiveCodeBench/.venv-granite`, `.venv-chpc`, or `.venv` already exists and can import `lcb_runner`, `scripts/chpc/setup_chpc.sh` may reuse it instead of reinstalling dependencies.
- HOME-backed virtualenvs are not linked into the scratch target anymore, because that pushes large runtime packages back under the CHPC HOME quota.
- Use `--force-install` only when you explicitly want a fresh target virtualenv build.
- The `--force-install` path now prefers Python 3.11-compatible interpreters, removes any HOME-backed target link, and installs runtime dependencies with `pip --prefer-binary --only-binary=:all:` plus `--no-build-isolation` for the editable package, so it stays on scratch and fails fast instead of hanging in long source builds.

Interactive smoke example:

```bash
scripts/chpc/run_chpc.sh \
	--model Qwen/Qwen2.5-Coder-32B-Instruct \
	--provider baseline \
	--max-instances 1 \
	--storage-root /scratch/general/vast/$USER/vtm
```

Smoke example:

```bash
scripts/chpc/submit_livecodebench_local_model.sh \
	--model Qwen/Qwen2.5-Coder-32B-Instruct \
	--storage-root /scratch/general/vast/$USER/vtm \
	--cluster granite \
	--account cs6966 \
	--partition soc-gpu-gh200-class-grn \
	--qos soc-gpu-class-grn \
	--gres gpu:1 \
	--time 02:00:00 \
	--cpus-per-task 8 \
	--mem 64G \
	--queue-tag qwen25coder32b_smoke \
	--n 1 \
	--evaluate false \
	--max-instances 1 \
	--rlm-max-iterations 2
```

For longer production runs, you can keep a shared `--time` or override individual
providers with `--time-baseline`, `--time-rag`, `--time-rlm`, and `--time-rlm-rag`.
That is useful when `rlm` and `rlm_rag` need substantially more walltime than
`baseline` and `rag`.

If you target a Granite class partition ending in `-grn`, the helper will default
to `--cluster granite` automatically. You can also pass `--cluster granite`
explicitly for clarity.

The generated LiveCodeBench bundles now include four providers:

- `baseline`
- `rag`
- `rlm`
- `rlm_rag`