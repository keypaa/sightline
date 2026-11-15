# DECISION-0002: One image marker, bare patches by default

Date: 2026-11-15
Status: accepted
Work package: WP0

## Context
The prompt needs one literal image marker that expands to N placeholder positions. Native models wrap patches in begin/end markers; Baseten's paper does not say whether it used them.

## Options considered
1. One marker, bare patches (no begin or end markers).
2. One marker plus begin/end structure tokens.
3. Several marker types or learned structure tokens.

## Decision
Use one literal `<image>` marker that expands to N placeholder positions. The default is bare patches: nothing surrounds the projected vectors. Marker use becomes an M1 ablation.

## Evidence
- M0 breakdown, 2b decision 2: one literal marker, bare patches by default; markers are an M1 ablation. [V]
- Main document section 7.6: WebBrain inserts bare patches; Baseten's paper says nothing about markers. [V]
- Qwen3's vision marker tokens exist in the vocabulary, but a text-only Qwen3 has probably never trained on them. [I]

## Consequences
No begin or end tokens in M0. The token contract tests cover placeholder expansion, count matching and embedding replacement. The marker ablation is M1 work, not M0.

## Open items
- [U] Whether markers change vision quality. Owner: implementation agent. Closing step: M1 marker ablation with the frozen thresholds in DECISIONS/.