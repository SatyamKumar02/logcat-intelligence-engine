# Eval Results — Baseline (Zero-Shot)

## Summary

| Metric | Value |
|---|---|
| Total tasks | 10 |
| Category accuracy | 90.0% |
| Avg keyword hit rate | 0.87 |
| Avg confidence | 0.90 |
| Avg trajectory score | 1.00 |
| Avg judge score | 8.4 / 10 |

## Per-Task Results

| Task | Expected | Predicted | Correct | Keyword Hit Rate | Confidence | Trajectory | Judge |
|---|---|---|---|---|---|---|---|
| eval_001 | anr | anr | yes | 1.00 | 0.85 | 1.00 | 8.0 |
| eval_002 | crash | crash | yes | 0.67 | 0.90 | 1.00 | 8.0 |
| eval_003 | oom | oom | yes | 1.00 | 0.85 | 1.00 | 8.0 |
| eval_004 | gpu_fault | gpu_fault | yes | 1.00 | 0.95 | 1.00 | 9.0 |
| eval_005 | oom_kill | oom_kill | yes | 0.67 | 0.95 | 1.00 | 9.0 |
| eval_006 | thermal | thermal | yes | 1.00 | 0.90 | 1.00 | 9.0 |
| eval_007 | camera_crash | camera_crash | yes | 1.00 | 0.90 | 1.00 | 8.0 |
| eval_008 | kernel_panic | kernel_panic | yes | 1.00 | 0.95 | 1.00 | 9.0 |
| eval_009 | binder_failure | binder_failure | yes | 0.67 | 0.90 | 1.00 | 8.0 |
| eval_010 | memory_leak | oom | no | 0.67 | 0.80 | 1.00 | 8.0 |
