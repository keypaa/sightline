# DECISION-0018: K2.6 parity tolerance

Date: 2026-11-15
Status: accepted
Work package: WP6

## Context
WP6 requires forward outputs to match a reference on at least 20 images at several resolutions within a frozen tolerance.

## Calibration
Modal T4, 20 images at 448, 672 and 896, fp32, eager attention:
- Reference: `immanuelpeter/MoonViT-K2.6` @ `84ec70ac59184bfa74fca77af9e1cdeccb32b5d6`.
- Ours: vendored official K2.6 tower @ `7eb5002f6aadc958aed6a9177b7ed26bb94011bb` with the extracted weights.
- 60 pairs; tokens 256 / 576 / 1024.

## Results
- max_abs 0.0 across all 60.
- min cosine 0.99999982.
- allclose(atol=1e-5, rtol=0) true for all 60.
- Elapsed 180.8 s, peak RSS 8.16 GB, CPU 177.3 s, max GPU 5.68 GB. [V]

## Decision
Freeze the K2.6 parity pass criteria at:
- max_abs <= 1e-5
- cosine >= 0.9999
- allclose(atol=1e-5, rtol=0) for 20 images at 448, 672 and 896.

## Consequences
The K2.6 tower is parity-verified. The same criteria apply to any future reference. The observed max_abs is 0.0, so the criteria are conservative.

## Open items
none