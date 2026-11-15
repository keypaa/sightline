# DECISIONS

One short dated file per decision, named `NNNN-short-title.md` and based on `_template.md`. Entries here override the documents in docs/. Never edit an accepted entry; add a new one that supersedes it.

## First records to write (WP0 and WP1)

| No. | Title | Source |
| --- | --- | --- |
| 0001 | Canvas 448x448, token count derived from config | M0 breakdown, 2b, decision 1 |
| 0002 | Image marker: one literal marker, bare patches by default | 2b, decision 2 |
| 0003 | Placeholder id for Qwen3 (confirm in the pinned tokenizer) | 2b, decision 3 |
| 0004 | Merge location and token order | 2b, decision 4 |
| 0005 | Chat template, thinking off, loss on answer tokens only | 2b, decision 5 |
| 0006 | Memorization controls | 2b, decision 6 |
| 0007 | Precision policy | 2b, decision 7 |
| 0008 | Seeds 0, 1, 2; never 42 | 2b, decision 8 |
| 0009 | Attention backend policy | 2b, decision 9 |
| 0010 | Library versions (WP0 rule) | WP0 |
| 0011 | Reader and tokenizer revision pins | WP1 |
| 0012 | K3 revision pin f831ab66814297da540d832a5235f8e904f29d06 | WP5, extraction result |
| 0013 | Padding side: right padding | WP2 |
| 0014 | Golden-splice and padding tolerances | WP2 |
| 0015 | Vendor the K3 MoonViT-V2 tower | WP5 |
| 0016 | K3 tower attention has no SDPA path | WP5 |
| 0017 | K2.6 MoonViT extraction and vendoring | WP6 |
| 0018 | K2.6 parity tolerance | WP6 |
| 0019 | Modal is a test harness, not production | WP6 |
| 0020 | Generation path | WP3 |
| 0021 | Freezing and the parameter test | WP4 |
| 0022 | Projector shape and norm ratio | WP7 |
| 0023 | Training loop, checkpoint and resume | WP8 |
| 0024 | Minimal synthetic data | WP9 |
| 0025 | 32-example overfit pass rule and control thresholds | WP10 |
| 0026 | WP10 overfit runs on A10G | WP10 |
| 0027 | G0 sign-off | WP11 |
| 0028 | Chunked-prefill tolerance | WP11 |
