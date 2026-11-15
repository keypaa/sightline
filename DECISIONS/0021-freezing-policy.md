# DECISION-0021: Freezing and the parameter test

Date: 2026-11-15
Status: accepted
Work package: WP4

## Context
Only the projector trains. The reader and encoder must be frozen (eval mode, `requires_grad=False`) and their weights unchanged.

## Decision
- `freeze(module)` sets eval mode and `requires_grad=False` on every parameter.
- `parameter_hashes` is the SHA-256 of the parameter bytes; `assert_frozen` compares a before/after mapping.
- The 20-step parameter test uses a tiny Qwen3 model and a dummy `nn.Linear` projector, because the real projector is WP7; the freezing logic is independent of the projector architecture.
- The real Qwen3-1.7B + K3 tower conformance test freezes both, runs one forward/backward with a dummy projector, and checks that only the projector has gradients and every frozen hash is unchanged.

## Evidence
- Tiny model: 20 steps; only the projector has non-zero grads; the projector weight changes; the frozen hashes are unchanged. [V]
- Real Qwen3-1.7B + K3 tower: projector grad not None; model and tower grads all None; frozen hashes unchanged; peak 12.64 GB, CPU 126.9 s. [V]

## Consequences
The freezing policy is implemented and tested. WP7 supplies the real projector; the freezing logic is unchanged. The real 20-step run is optional and can run on the GPU or Modal.

## Open items
- [U] Real 20-step parameter run on the RTX PRO 6000 (optional; the tiny 20-step test and the real one-step hash test cover the logic). Owner: implementation agent. Closing step: run 20 steps on the GPU and record.