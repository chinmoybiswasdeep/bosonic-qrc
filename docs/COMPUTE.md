# Compute planning

Calibration measured 2,728 Piquasso backend calls and completed in about 29 s
on the recorded local platform. Production declares 50 reservoir/data runs,
about 256,500 backend steps, 198 targets, and 4,999 null refits per target.
Null refitting dominates; a conservative serial estimate is 12–15 hours.
Benchmark the first shard on the target machine before scheduling the full job.

CI runs smoke only. Production is never launched automatically.

