#!/usr/bin/env python3
"""Extract the MoonViT-V2 tower and projector from moonshotai/Kimi-K3 without
downloading the checkpoint. Reads the safetensors index, then fetches only the
byte ranges of tensors whose names start with vision_tower. or mm_projector.
Raw bytes are copied (no dtype conversion). Writes one safetensors file plus a
manifest with per-tensor sha256.

  python extract_k3_vision.py --revision <commit> --out k3_vision.safetensors
  python extract_k3_vision.py --compare old_extraction.safetensors --out k3_vision.safetensors
Test without network: --local-dir DIR (index + shards on disk).
"""
from __future__ import annotations
import argparse, hashlib, json, math, os, struct, sys
from pathlib import Path

PREFIXES = ("vision_tower.", "mm_projector.")
DTYPE_BYTES = {"BF16": 2, "F16": 2, "F32": 4, "F64": 8, "I64": 8, "I32": 4, "I8": 1, "U8": 1, "BOOL": 1}


class HttpSource:
    def __init__(self, repo: str, revision: str):
        import requests
        self.requests, self.repo, self.rev = requests, repo, revision
        tok = os.environ.get("HF_TOKEN")
        self.headers = {"Authorization": f"Bearer {tok}"} if tok else {}

    def _url(self, name): return f"https://huggingface.co/{self.repo}/resolve/{self.rev}/{name}"

    def whole(self, name: str) -> bytes:
        r = self.requests.get(self._url(name), headers=self.headers, timeout=120); r.raise_for_status(); return r.content

    def read(self, name: str, a: int, b: int) -> bytes:  # inclusive range
        r = self.requests.get(self._url(name), headers={**self.headers, "Range": f"bytes={a}-{b}"}, timeout=300)
        r.raise_for_status()
        if r.status_code != 206 or len(r.content) != b - a + 1:
            raise RuntimeError(f"range {a}-{b} of {name}: status {r.status_code}, {len(r.content)} bytes")
        return r.content


class LocalSource:
    def __init__(self, root: str): self.root = Path(root)
    def whole(self, name): return (self.root / name).read_bytes()
    def read(self, name, a, b):
        with open(self.root / name, "rb") as f: f.seek(a); return f.read(b - a + 1)


def read_header(src, shard: str):
    n = struct.unpack("<Q", src.read(shard, 0, 7))[0]
    return n, json.loads(src.read(shard, 8, 7 + n))


def write_safetensors(path: Path, tensors: dict, metadata: dict):
    names = sorted(tensors); header, off = {"__metadata__": metadata}, 0
    for k in names:
        t = tensors[k]; header[k] = {"dtype": t["dtype"], "shape": t["shape"], "data_offsets": [off, off + len(t["raw"])]}; off += len(t["raw"])
    hb = json.dumps(header, separators=(",", ":")).encode(); hb += b" " * (-len(hb) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hb))); f.write(hb)
        for k in names: f.write(tensors[k]["raw"])


def load_safetensors_raw(path: Path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]; hdr = json.loads(f.read(n)); base = 8 + n; out = {}
        for k, v in hdr.items():
            if k == "__metadata__": continue
            a, b = v["data_offsets"]; f.seek(base + a); out[k] = {"dtype": v["dtype"], "shape": v["shape"], "raw": f.read(b - a)}
    return out


def extract(src, expect_tower, expect_proj, allow_mismatch):
    index = json.loads(src.whole("model.safetensors.index.json")); wm = index["weight_map"]
    wanted = {k: s for k, s in wm.items() if k.startswith(PREFIXES)}
    n_t = sum(k.startswith(PREFIXES[0]) for k in wanted); n_p = sum(k.startswith(PREFIXES[1]) for k in wanted)
    print(f"index: {len(wm)} tensors, {len(set(wm.values()))} shards; wanted {n_t} tower + {n_p} projector tensors")
    if (n_t, n_p) != (expect_tower, expect_proj) and not allow_mismatch:
        sys.exit(f"expected {expect_tower}+{expect_proj}, found {n_t}+{n_p}; pass --allow-mismatch to continue")
    tensors = {}
    for shard in sorted(set(wanted.values())):
        n, hdr = read_header(src, shard); base = 8 + n
        for k in sorted(x for x, s in wanted.items() if s == shard):
            info = hdr[k]; a, b = info["data_offsets"]
            raw = src.read(shard, base + a, base + b - 1)
            need = math.prod(info["shape"]) * DTYPE_BYTES[info["dtype"]]
            if len(raw) != need: sys.exit(f"{k}: {len(raw)} bytes, expected {need}")
            tensors[k] = {"dtype": info["dtype"], "shape": info["shape"], "raw": raw, "shard": shard}
        print(f"  {shard}: {sum(1 for t in tensors.values() if t['shard']==shard)} tensors read")
    return tensors


def compare(new, old_path, strip):
    old = load_safetensors_raw(Path(old_path))
    if strip: old = {(strip + k if not k.startswith(PREFIXES) else k): v for k, v in old.items()}
    same = [k for k in new if k in old and new[k]["raw"] == old[k]["raw"] and new[k]["shape"] == old[k]["shape"] and new[k]["dtype"] == old[k]["dtype"]]
    diff = [k for k in new if k in old and k not in same]
    print(f"compare: {len(same)} identical, {len(diff)} different, {len([k for k in new if k not in old])} missing from old, {len([k for k in old if k not in new])} extra in old")
    for k in diff[:10]: print("  differs:", k)
    return {"identical": len(same), "different": diff, "missing_from_old": [k for k in new if k not in old], "extra_in_old": [k for k in old if k not in new]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo", default="moonshotai/Kimi-K3"); p.add_argument("--revision", default="main")
    p.add_argument("--out", default="k3_vision.safetensors"); p.add_argument("--local-dir")
    p.add_argument("--expect-tower", type=int, default=165); p.add_argument("--expect-projector", type=int, default=3)
    p.add_argument("--allow-mismatch", action="store_true"); p.add_argument("--compare"); p.add_argument("--strip-prefix-for-compare", default="")
    a = p.parse_args()
    src = LocalSource(a.local_dir) if a.local_dir else HttpSource(a.repo, a.revision)
    if a.revision == "main" and not a.local_dir: print("WARNING: pin a commit hash with --revision for reproducibility", file=sys.stderr)
    tensors = extract(src, a.expect_tower, a.expect_projector, a.allow_mismatch)
    nbytes = sum(len(t["raw"]) for t in tensors.values())
    params = {pre: sum(math.prod(t["shape"]) for k, t in tensors.items() if k.startswith(pre)) for pre in PREFIXES}
    print(f"dtypes: {sorted({t['dtype'] for t in tensors.values()})}; bytes {nbytes:,}; parameters {params}")
    write_safetensors(Path(a.out), tensors, {"format": "pt", "source_repo": a.repo, "source_revision": a.revision})
    manifest = {"repo": a.repo, "revision": a.revision, "bytes": nbytes, "parameters": params,
                "file_sha256": hashlib.sha256(Path(a.out).read_bytes()).hexdigest(),
                "tensors": {k: {"shape": t["shape"], "dtype": t["dtype"], "shard": t["shard"], "sha256": hashlib.sha256(t["raw"]).hexdigest()} for k, t in sorted(tensors.items())}}
    if a.compare: manifest["compare"] = compare(tensors, a.compare, a.strip_prefix_for_compare)
    Path(a.out + ".manifest.json").write_text(json.dumps(manifest, indent=1))
    print("wrote", a.out, "and manifest")


if __name__ == "__main__":
    main()
