# DECISION-0010: Library versions

Date: 2026-11-15
Status: accepted (GPU runtime check pending)
Work package: WP0

## Context
WP0 chooses versions by this rule: the newest stable torch build whose `get_arch_list()` includes the local GPU's architecture, with a transformers version that loads the pinned Qwen3 checkpoint and the vendored K3 module.

## Options considered
1. Keep the versions already installed in the environment.
2. Resolve the newest versions from the index without checking the local architecture.
3. Pin the installed versions and lock them.

## Decision
Pin the versions already installed and verified in this environment, and lock them with `uv.lock`.

| Package | Version |
| --- | --- |
| Python | 3.14.8 |
| torch | 2.13.0+cu130 |
| transformers | 5.12.1 |
| tokenizers | 0.22.2 |
| safetensors | 0.8.0 |
| numpy | 2.5.1 |
| huggingface-hub | 1.25.1 |
| pytest | 9.1.1 |
| ruff | 0.15.1 |

## Evidence
- `torch.__version__ == "2.13.0+cu130"`; `torch.cuda.get_arch_list() == ["sm_75", "sm_80", "sm_86", "sm_90", "sm_100", "sm_120"]`; `transformers.__version__ == "5.12.1"`. [V]
- `uv lock` resolved 63 packages and wrote `uv.lock`; torch resolves to `2.13.0+cu130` from `https://download.pytorch.org/whl/cu130`. [V]
- The current shell reports `torch.cuda.is_available() == False` and `device_count == 0`, with no `/dev/nvidia*` and no `nvidia-smi`; only an AMD Renoir iGPU is present. [V]
- torchvision is installed as `0.28.0+cpu`, which does not match the cu130 torch build. [V]

## Consequences
`uv sync` reproduces the locked environment. torchvision is deliberately not a WP0 dependency; it is added in WP5 with its own DECISIONS entry and a matching CUDA build. The GPU runtime check is still open.

## Open items
- [U] GPU runtime on the RTX PRO 6000 (sm_120): `torch.cuda.is_available()` and a small matmul. Owner: implementation agent. Closing step: run on the GPU workstation and record the result in the WP10 run record.
- [U] torchvision build matching torch 2.13.0+cu130. Owner: implementation agent. Closing step: WP5 dependency decision.