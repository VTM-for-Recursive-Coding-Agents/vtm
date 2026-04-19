# Pass/Fail Dashboard Summary

## Method Metrics

| method | pass | fail | total | pass_rate |
|---|---:|---:|---:|---:|
| baseline | 439 | 616 | 1055 | 41.61% |
| rag | 451 | 604 | 1055 | 42.75% |
| rlm-partial | 30 | 84 | 114 | 26.32% |
| rlm | 1 | 847 | 848 | 0.12% |
| rlm_rag-latest50 | 3 | 47 | 50 | 6.00% |
| rlm_rag-partial | 16 | 62 | 78 | 20.51% |
| rlm_rag | 19 | 101 | 120 | 15.83% |

## RLMFix Chunk Health

| kind | chunk | nonempty | total | nonempty_rate |
|---|---|---:|---:|---:|
| rlm | c0 | 20 | 20 | 100.00% |
| rlm | c106 | 12 | 12 | 100.00% |
| rlm | c212 | 9 | 9 | 100.00% |
| rlm | c318 | 17 | 17 | 100.00% |
| rlm | c424 | 13 | 13 | 100.00% |
| rlm | c530 | 8 | 8 | 100.00% |
| rlm | c636 | 9 | 9 | 100.00% |
| rlm | c742 | 7 | 8 | 87.50% |
| rlm | c848 | 6 | 6 | 100.00% |
| rlm | c954 | 22 | 23 | 95.65% |
| rlm_rag | c0 | 8 | 8 | 100.00% |
| rlm_rag | c106 | 8 | 8 | 100.00% |
| rlm_rag | c212 | 19 | 19 | 100.00% |
| rlm_rag | c318 | 5 | 5 | 100.00% |
| rlm_rag | c424 | 5 | 5 | 100.00% |
| rlm_rag | c530 | 1 | 1 | 100.00% |
| rlm_rag | c636 | 1 | 1 | 100.00% |
| rlm_rag | c742 | 8 | 8 | 100.00% |
| rlm_rag | c848 | 10 | 10 | 100.00% |
| rlm_rag | c954 | 13 | 13 | 100.00% |

## Intersection-Only Comparison

Model prefix: `Qwen2.5-Coder-Ins-32B-`

Shared question count across baseline/rag/rlm/rlm_rag: **62**

| method | source_file | total_labeled | pass_on_intersection | fail_on_intersection | pass@1_on_intersection |
|---|---|---:|---:|---:|---:|
| baseline | Qwen2.5-Coder-Ins-32B-baseline_passfail.tsv | 1055 | 30 | 32 | 48.39% |
| rag | Qwen2.5-Coder-Ins-32B-rag_passfail.tsv | 1055 | 32 | 30 | 51.61% |
| rlm | Qwen2.5-Coder-Ins-32B-rlm-partial_passfail.tsv | 114 | 17 | 45 | 27.42% |
| rlm+rag | Qwen2.5-Coder-Ins-32B-rlm_rag-partial_passfail.tsv | 78 | 13 | 49 | 20.97% |
