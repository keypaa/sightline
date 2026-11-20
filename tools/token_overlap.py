#!/usr/bin/env python3
"""Count shared token byte strings between the K3 tokenizer and a reader's.

Section 16 item 6 makes this an M2 entry criterion: the embedding-alignment init
is only available if the two tokenizers share enough vocabulary, and it is never
a default. Baseten's Kimi-to-Qwen3 pair shared 95,369 identical strings [S].

Comparison is at the **byte** level, which is the only fair basis here: K3 ships
a tiktoken-style ``tiktoken.model`` of base64 byte strings, Qwen ships a BPE
``tokenizer.json`` of text tokens. Encoding both to UTF-8 bytes makes them
comparable without executing K3's ``tokenization_kimi.py``.

    PYTHONPATH=src python tools/token_overlap.py \
        --out experiments/m2a_entry/token_overlap_k3_qwen3.json
"""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

QWEN3_REVISION = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
K3_REVISION = "f831ab66814297da540d832a5235f8e904f29d06"


def _is_utf8(token: bytes) -> bool:
    try:
        token.decode("utf-8")
    except UnicodeDecodeError:
        return False
    return True


def reader_vocab(name: str, revision: str) -> set[bytes]:
    """Token strings from tokenizer.json, as UTF-8 bytes."""

    from huggingface_hub import hf_hub_download

    path = hf_hub_download(name, "tokenizer.json", revision=revision)
    data = json.loads(Path(path).read_text())
    vocab = data.get("model", {}).get("vocab")
    tokens: set[bytes] = set()
    if isinstance(vocab, dict):
        tokens.update(token.encode("utf-8") for token in vocab)
    elif isinstance(vocab, list):
        tokens.update(item["content"].encode("utf-8") for item in vocab)
    for added in data.get("added_tokens", []):
        tokens.add(added["content"].encode("utf-8"))
    return tokens


def tiktoken_vocab(name: str, revision: str) -> set[bytes]:
    """Token byte strings from a tiktoken ``.model`` file."""

    from huggingface_hub import hf_hub_download

    path = hf_hub_download(name, "tiktoken.model", revision=revision)
    tokens: set[bytes] = set()
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            piece, _, _rank = line.partition(" ")
            tokens.add(base64.b64decode(piece))
    return tokens


def compare(
    reader: set[bytes], encoder: set[bytes], reader_text: set[str], encoder_text: set[str]
) -> dict:
    shared = reader & encoder
    shared_text = reader_text & encoder_text
    return {
        "reader_tokens": len(reader),
        "encoder_tokens": len(encoder),
        "shared_bytes": len(shared),
        "shared_bytes_fraction_of_reader": round(len(shared) / len(reader), 4) if reader else None,
        "shared_strings": len(shared_text),
        "shared_strings_fraction_of_reader": round(len(shared_text) / len(reader_text), 4)
        if reader_text
        else None,
        "encoder_tokens_not_valid_utf8": len(encoder) - len(encoder_text),
        "reader_only": len(reader) - len(shared),
        "encoder_only": len(encoder) - len(shared),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reader", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--reader-revision", default=QWEN3_REVISION)
    parser.add_argument("--encoder", default="moonshotai/Kimi-K3")
    parser.add_argument("--encoder-revision", default=K3_REVISION)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    reader = reader_vocab(args.reader, args.reader_revision)
    encoder = tiktoken_vocab(args.encoder, args.encoder_revision)
    reader_text = {token.decode("utf-8") for token in reader}
    encoder_text = {token.decode("utf-8") for token in encoder if _is_utf8(token)}
    result = {
        "method": "utf-8 byte strings; K3 tiktoken.model vs reader tokenizer.json",
        "reader": args.reader,
        "reader_revision": args.reader_revision,
        "encoder": args.encoder,
        "encoder_revision": args.encoder_revision,
        **compare(reader, encoder, reader_text, encoder_text),
    }
    text = json.dumps(result, indent=2)
    print(text)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
