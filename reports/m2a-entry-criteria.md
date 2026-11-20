# M2 entry criteria

Section 16 of the project document lists what must be settled before M2a. Status
as of 2026-11-21.

| # | item | status | evidence |
| --- | --- | --- | --- |
| 6 | Embedding-alignment init | **closed, negative** | `tools/token_overlap.py` |
| 7 | Data sources and licenses | **drafted** | this file; per-source table complete, per-image checks open |
| 8 | Annotator choice, terms and cost | **closed** | `DECISIONS/0040`, `reports/m2a-annotator-pilot.md` |

## 6. Token overlap between the K3 encoder tokenizer and Qwen3

Method: compare token vocabularies at the byte level. K3 ships a tiktoken-style
`tiktoken.model` of base64 byte strings and custom code (`tokenization_kimi.py`);
Qwen ships a BPE `tokenizer.json` of text tokens. Encoding both to UTF-8 bytes
makes them comparable **without executing K3's repository code**. Both the byte
and the decoded-string comparisons give the same answer, so the result is not an
artefact of the encoding choice.

| | value |
| --- | --- |
| reader tokens (Qwen3) | 151,669 |
| encoder tokens (K3) | 163,584 |
| **shared** | **34,995** |
| shared, fraction of reader | **23.1%** |
| K3 tokens that are not valid UTF-8 | 1,172 |
| reader-only | 116,674 |
| encoder-only | 128,589 |

Revisions: reader `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`, encoder
`f831ab66814297da540d832a5235f8e904f29d06`. Record:
`experiments/m2a_entry/token_overlap_k3_qwen3.json`.

**Qwen3-4B has a byte-identical `tokenizer.json` to Qwen3-1.7B** (sha256
`aeb13307a71acd8f…`), so this applies to the M2a reader.

### Reading it

Only **23%** of the reader's vocabulary appears in the encoder's. Baseten's
Kimi-to-Qwen3 pair shared 95,369 strings [S], which is 2.7× more than we
measure; the difference may be a different Kimi version, a different Qwen size,
or a different counting method. Recorded as a conflict rather than resolved. [U]

**Decision consequence:** the embedding-alignment init would have an embedding to
copy for under a quarter of the reader's tokens. It stays an **ablation, never a
default**, which is what `DECISIONS/0022` already says. This is now measured
rather than assumed.

## 8. Annotator

Closed in `DECISIONS/0040`: `kilo/nvidia/nemotron-3.5-super-vl:free`, 3,179
tokens per conversation, 127M for 40k, inside the 200M/day cap. Pricing still
`[U]`.

## 7. Data sources and licenses

Verified 2026-11-21 from each source's own terms page. The headline licence on a
dataset card usually covers the **annotations**, not the pixels — COCO is the
clear case, and the same trap applies to most image datasets.

| source | annotations | images | verdict for M2a |
| --- | --- | --- | --- |
| **COCO** | CC BY 4.0 | **not owned by the consortium**; each photo keeps its Flickr licence, recorded per image in the annotation file | usable, but **requires per-image licence review** |
| **Flickr30k** | captions free to use | Flickr images, provided for **non-commercial research and education only** | research use only; **not** for anything commercial |
| **PixMo** (Ai2) | **ODC-BY-1.0** | distributed as **URLs**, fetched from third-party hosts; rights unknown per image | annotations usable with attribution; **image provenance is the risk** |
| **The Cauldron** (HF M4) | **CC BY 4.0** for the collection and prompts | redistributed inside, but **50 sub-datasets each keep their own licence** | usable with **per-subset review**; terms require disclosing use if a model is trained |

Sources: `cocodataset.org/#termsofuse`; the Flickr30k providers' own statement;
`huggingface.co/datasets/allenai/pixmo-points`; `huggingface.co/datasets/HuggingFaceM4/the_cauldron`. [V]

### Consequences for M2a

- **"CC BY 4.0" on a card is not permission for the photographs.** COCO's own
  terms say the consortium does not own image copyright. Any pipeline that
  assumes otherwise is wrong.
- **COCO is the best-behaved source** because it ships a licence id per image, so
  the review can be automated rather than manual. That is a concrete reason to
  prefer it.
- **Flickr30k is out** for any commercial use, and its third-party CC0 re-uploads
  do not change the original providers' non-commercial statement.
- **PixMo's and Cauldron's images need provenance work before download**, not
  after. The Cauldron's 50 sub-licences make it the highest-effort option.
- Every source that enters gets a row in the manifest with its licence id and the
  per-image or per-subset check recorded. No source enters without one.

### Open items

- [U] The Cauldron's per-subset licences (50 of them). Owner: implementation
  agent. Closing step: enumerate the subsets M2a would actually use and record
  each licence.
- [U] PixMo image provenance at scale. Closing step: sample 100 image URLs and
  record host and licence status.