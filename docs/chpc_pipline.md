# Running a Benchmark Job on CHPC

This guide explains how to run a full LiveCodeBench benchmark job on the CHPC cluster using the provided scripts and environment.

## 1. Environment Setup

- Make sure you are on a CHPC login node
- Load any required modules and activate your Python environment if needed.
- (Optional) Source the CHPC environment script for scratch-backed cache setup:

  ```bash
  source scripts/chpc/chpc_env.sh
  vtm_chpc_setup_environment /scratch/general/vast/<YOUR-USERNAME>/vtm
  ```

## 2. Generate and Submit Benchmark Jobs

Use the submission helper script to generate and submit all four provider jobs (baseline, rag, rlm, rlm_rag) in one step. Example command:

```bash
scripts/chpc/submit_livecodebench_local_model.sh \
  --storage-root /scratch/general/vast/<YOUR-USERNAME>/vtm \
  --python-bin /scratch/general/vast/<YOUR-USERNAME>/vtm/venvs/livecodebench/bin/python \
  --model Qwen/Qwen2.5-Coder-32B-Instruct \
  --account soc-gpu-np \
  --partition soc-gpu-np \
  --time 12:00:00 \
  --queue-tag chpc_full_benchmark_<DATE>
```
### Example:

```bash
scripts/chpc/submit_livecodebench_local_model.sh --storage-root /scratch/general/vast/u1406806/vtm --python-bin /scratch/general/vast/u1406806/vtm/venvs/livecodebench/bin/python --model Qwen/Qwen2.5-Coder-32B-Instruct --account soc-gpu-np --partition soc-gpu-np --time 6:00:00 --queue-tag chpc_full_benchmark_20260407a
```

- Adjust `--model`, `--time`, and other arguments as needed.
- The script will create a launcher bundle in `launchers/chpc/` and submit all jobs to Slurm.

## 3. Monitor Job Status

Check your job status with:

```bash
squeue -u <YOUR-USERNAME>
```

Or for more details:

```bash
sacct -j <JOBID> --format=JobID,State,Elapsed,ExitCode,NodeList
```

## 4. Check Logs and Results

- Slurm output and error logs are written to `logs/slurm/`.
- Benchmark results are saved in `results/raw/livecodebench/`.

## 5. Cleanup (Optional)

To remove old launcher bundles and logs:

```bash
rm -rf launchers/chpc/<OLD-BUNDLE-NAME>
rm logs/slurm/*smoke* logs/slurm/*dryrun* logs/slurm/*validate* logs/slurm/*repair*
```

---

For more details, see the README files in `scripts/chpc/` and `launchers/`.
