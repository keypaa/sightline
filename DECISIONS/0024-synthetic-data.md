# DECISION-0024: Minimal synthetic data

Date: 2026-11-15
Status: accepted
Work package: WP9

## Context
WP9 provides just enough data for the WP10 32-example overfit: 32 rendered scenes with exact answers, deterministic by seed, with a hashed manifest. The full generator is M1.

## Decision
- 8 colors x 4 shapes = 32 scenes, 448x448 PNG, white background.
- One question template: "What color is the shape?"; the answer is the color. The 32 answers are balanced (4 per color), so the text alone scores at chance (1/8).
- Deterministic: a seeded shuffle of the color/shape pairs, PIL rendering, and a per-image SHA-256 of the PNG bytes.
- Manifest: generator version, pillow version, seed, n, and per-example image sha256, question, answer, color, shape, count, template id, split and seed. The manifest sha256 is the SHA-256 of the canonical JSON.
- Regenerating from the same seed reproduces the manifest hash.

## Evidence
- 8 WP9 tests pass: determinism, seed sensitivity, balanced answers, question not identifying the answer, manifest roundtrip, unique images, image size/labels, constant question with varying answer. [V]
- Manifest: `experiments/wp9_synthetic/manifest.json`, manifest sha256 `2fe44a9db59edcc6d215ced49d5a19e58d65a9ba242154275baae70e422ce1b7`. [V]
- Full suite: 75 passed, 74.3 s, peak 13.1 GB, 276.8 CPU-s. [V]

## Consequences
The overfit set is ready for WP10. The full M1 generator will add count, relative position, words and held-out compositions.

## Open items
- [U] Full M1 generator (color, shape, count, relative position, words, held-out compositions). Owner: implementation agent. Closing step: M1.