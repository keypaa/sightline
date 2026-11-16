# Measurement pitfalls

Durable notes from measurement mistakes. Each one cost a run to discover. Add to
this file rather than relearning.

## 1. The M1 splits are ordered by answer

`generate_m1_dataset` emits examples grouped by answer: all the reds, then the
greens, then the blues, then shapes, then counts. Consequences:

- `split[:N]` is **not** a sample. The first 20 test examples are all
  `expected='red'`, which flatters any model with a colour prior.
- `--eval-limit` and `--debug-generations` therefore use `sample_evenly`, a
  stride, in `tools/run_m1.py`. Verified: 20 examples go from 1 distinct answer
  to 16.
- Anything that subsets a split must sample evenly, or shuffle with a recorded
  seed first.

## 2. Exact match scores zero on prose

G1 is frozen to exact match (`DECISIONS/0032`). A reader that answers
`'the shape is red'` scores zero even when the colour is right, because the
comparison is `answer == 'red'` after stripping whitespace and punctuation.

Consequence: **1.4% exact match does not mean the model cannot see.** Before
reading any accuracy number, print 20 expected-vs-generated pairs
(`--debug-generations`). If the outputs are prose, the number measures format
compliance, not vision.

Changing the metric is allowed but must go through a DECISIONS entry that says
why, and the seed re-run under both metrics. Do not loosen a frozen criterion
after seeing a failing result.

## 3. Accuracy below chance is a bug, not a weak model

Chance for colour is 1/8, shape 1/4, count 1/5. If the correct condition scores
**below** chance while the mismatched control sits **at** chance, the image is
not being ignored — it is being used wrongly. An ignored image would put both at
chance.

Observed in the seed-0 run: color 1.4% correct against 12.5% for the mismatched
control.

## 4. Evaluate the train split before trusting the test split

The train split is the only way to separate three failures that look identical on
test:

| train accuracy | meaning |
| --- | --- |
| high | memorisation; test is a generalisation gap |
| ~0 | the evaluation path is broken |
| below chance | the mapping is wrong, or training never fit |

Seed 0 scored 0/7 colour and 1/7 shape on **train**, which ruled out memorisation
and pointed at optimisation. Cost: one 2-minute debug run.

## 5. Probe before projecting

The A10G micro-batch 8 probe predicted 32361.2 MB for the PRO 6000 run; the run
recorded exactly 32361.2 MB. The probe is cheap and it is the only way to size a
micro-batch on a card you do not have.

## 6. Sample memory scales with batch, so a trim can unlock a bigger batch

The vocabulary projection is `[batch, seq, vocab]`, not `[batch, hidden]`. At
micro-batch 16 and seq 271 that is 10.5 GB of bf16 logits, most of it for
positions that carry no label. Trimming it with `logits_to_keep` saved 11.5% of
step time and, on the PRO 6000, about 10 GB — which is what makes a larger
micro-batch plausible. The remaining step time is the frozen reader's forward and
backward, so micro-batch size, not `torch.compile`, is the lever.