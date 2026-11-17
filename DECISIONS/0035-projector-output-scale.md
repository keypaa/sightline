# DECISION-0035: scale the projector output to the reader's embedding RMS

Date: 2026-11-17
Status: accepted (pending the A10G confirmation probe)
Work package: M1b

## Context

Seed 0 ran 2048 steps and failed G1: colour 1.4%, shape 0.0%, count 0.8%
against chance rates of 1/8, 1/4 and 1/5. A debug run on the train split
answered 0/7 colour and 1/7 shape, so the model never fit the training data.

The A10G gradient-norm check (50 steps, micro-batch 4) gave:

- mean projector gradient norm over the last 25 steps: 829.4, maximum 1264.4
- mean loss over the last 25 steps: 20.56
- `ln(151936) = 11.93` for a uniform guess over the reader's vocabulary

A loss of 20.6 is far above chance entropy. The weak-gradient hypothesis is
falsified: the projector receives a large signal, and the run sits in an extreme
region of the frozen reader's input space rather than a flat one.

## Decision

- The projector output is multiplied by a scalar `output_scale`, registered as a
  buffer so it is saved and restored with the checkpoint.
- `calibrate_output_scale` sets it once, before training, so the RMS of the
  projector output matches the RMS of the reader's token embedding weight. Both
  values are measured from the model and data at runtime; nothing is hard-coded.
- The calibration uses the first micro-batch's features and
  `model.get_input_embeddings().weight`.
- `--no-output-scale` restores the previous behaviour for A/B comparison.

## Rationale

The rule in this project is to derive quantities from the checkpoint rather than
recall or invent them. A constant such as `1/sqrt(hidden)` would be a guess about
the reader's embedding distribution; measuring the RMS of the embedding weight is
the same claim with evidence.

## Acceptance

The 50-step A10G probe is the test. It passes if the initial loss falls to at or
below ~11.9 and the gradient norm drops into single digits. If the loss stays
high, scaling is not the cause and the next hypothesis is the recipe itself.

## Evidence

- A10G 50-step probe: mean gradient norm 829.4, mean loss 20.56, step 9.69 s. [V]
- `ln(151936) = 11.93`, arithmetic on the reader's vocabulary size. [V]
- Seed-0 train debug: 0/7 colour, 1/7 shape, 5/6 count. [V]

## Consequences

The seed-0 checkpoint predates the buffer and is not directly comparable; the
eval path warns and loads with `output_scale = 1.0` if the key is absent.

## Open items

- [U] Whether output scaling alone fixes training. Owner: implementation agent.
  Closing step: the 50-step A10G probe, then a 2048-step run.