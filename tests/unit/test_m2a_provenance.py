def test_no_kilo_rows_remain_in_the_20k_tree():
    from pathlib import Path
    import json
    import pytest

    root = Path("experiments/m2a_iter20k")
    if not root.exists():
        pytest.skip("20k tree is not present in this checkout")
    assert (root / "fetched.jsonl").exists(), "fetched index must survive the rebuild"

    rows = []
    for path in root.glob("*conversations*.jsonl"):
        for line in path.read_text().splitlines():
            if line.strip():
                rows.append(json.loads(line))
    assert all("kilo" not in r.get("model", "") for r in rows)