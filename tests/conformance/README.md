# Conformance suite

Every reader adapter must pass the same tests. The Qwen3-1.7B M0 suite:

| Test | File | Pass |
| --- | --- | --- |
| Golden splice (fp32) | `test_qwen3_conformance_fp32.py` | exact |
| Golden splice (bf16) | `test_qwen3_conformance_bf16.py` | max abs <= 1e-3 |
| Logit change | `test_qwen3_conformance_fp32.py` | later logits change; unperturbed identical |
| Placeholder count | `test_qwen3_conformance_fp32.py` | raises |
| Padding invariance | `test_qwen3_conformance_fp32.py` | max abs <= 1e-4 |
| Placeholder not used | `test_qwen3_conformance_fp32.py` | exact |
| Generation parity | `test_qwen3_generation.py` | identical tokens |
| Chunked prefill | `test_qwen3_generation.py` | max abs <= 1e-4 |
| Parameter test | `test_freezing_qwen3.py` | frozen hashes unchanged |
| Resume | `tests/unit/test_train.py` | max abs <= 1e-6 |
| Determinism | `tests/unit/test_train.py` | identical losses |

The K3/K2.6 tower parity tests (`test_vision_k3_tower.py`, `test_vision_k26_tower.py`) are heavy and run on Modal:

```bash
python3 modal/conformance.py
```

Run the Qwen3 suite locally:

```bash
pytest -m slow tests/conformance/test_qwen3_conformance_fp32.py \
  tests/conformance/test_qwen3_conformance_bf16.py \
  tests/conformance/test_qwen3_generation.py \
  tests/conformance/test_freezing_qwen3.py
```

All tower and reader tests are `slow`-marked and skipped by the default `pytest`; use `-m slow` to run them.