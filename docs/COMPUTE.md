# Compute planning

Calibration used 2,320 Perceval backend calls and completed in about 13 s on
the recorded platform. Production declares 100 model/reservoir/data runs,
roughly 240,000 backend evaluations, 12 targets, and 4,999 null refits per
target. A conservative serial estimate is 2–4 hours; benchmark a shard on the
target machine first. CI runs smoke only and never launches production.

