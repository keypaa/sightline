# DECISION-0026: WP10 overfit runs on A10G

Date: 2026-11-15
Status: accepted
Work package: WP10

## Context
WP10 needs the real Qwen3-1.7B reader, the frozen K3 MoonViT-V2 tower and the WP7 projector on Modal. The plan was a T4 (16 GB).

## Decision
Run WP10 on **A10G (24 GB)** with micro-batch 8. The T4 OOMs at micro-batch 8 (13.75 GiB allocated) and still OOMs at micro-batch 4 (13.24 GiB). The A10G fits the original micro-batch 8.

## Evidence
- T4 OOM at micro-batch 8: 13.75 GiB allocated, 103 MiB free. [V]
- T4 OOM at micro-batch 4: 13.24 GiB allocated, 492 MiB free. [V]
- A10G 24 GB: the function proceeds past training. [V]

## Consequences
WP10 uses A10G; the T4 remains the default for WP6 parity. The cost is higher, but it is a single GPU, not a multi-GPU node. If the A10G also OOMs, reduce micro-batch to 4 or add gradient checkpointing to the frozen reader.

## Open items
- [U] Gradient checkpointing on the frozen reader (if the A10G OOMs or to reduce memory). Owner: implementation agent. Closing step: WP10.