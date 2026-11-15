# DECISION-0009: SDPA by default; FlashAttention 2 only after parity

Date: 2026-11-15
Status: accepted
Work package: WP0

## Context
The local card is compute capability 12.0 (sm_120). FlashAttention 3 and 4 do not run there; FlashAttention 2 needs a compatible build. The backend must be recorded in every run.

## Options considered
1. PyTorch SDPA everywhere.
2. FlashAttention 2 everywhere.
3. SDPA by default, FlashAttention 2 only after a parity test.

## Decision
Use PyTorch SDPA by default for the Qwen3 readers and the K3 tower. Switch to FlashAttention 2 only after a parity test, and record the backend in every run record.

## Evidence
- M0 breakdown, 2b decision 9. [V]
- Qwen3-1.7B config at revision 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e: 16 attention heads, 8 KV heads, head_dim 128. [V]
- Qwen3-4B, 8B, 14B and 32B configs: 8 KV heads and head_dim 128 at every size. [V]
- AGENTS.md environment notes: FlashAttention 3 and 4 do not run on sm_120; FlashAttention 2 needs a build or community wheel. [S, via the document]

## Consequences
The default path needs no extra kernel. The run record's `attention_backend` is `sdpa`. WP5 adds the SDPA-versus-FlashAttention-2 parity test for the tower.

## Open items
- [U] Availability of an sm_120 FlashAttention 2 wheel. Owner: implementation agent. Closing step: WP5 parity test; if unavailable, stay on SDPA and record the attempt.