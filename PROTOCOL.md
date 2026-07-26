# Preregistered recovery protocol

This clean recovery attempt uses only Kubernetes runs launched after
2026-07-26T21:36:24.092Z. It tests the released ISO-Merger rather than the
unreleased ISO-Optimizer.

## Bounded substitutions

- Shared base: `google/bert_uncased_L-2_H-128_A-2` with one identically
  initialized binary classification head.
- Complementary public tasks: GLUE SST-2 sentiment and GLUE QNLI entailment.
- Specialists: matched full-parameter supervised updates from the exact shared
  checkpoint, 4,096 examples and 160 optimizer steps per task.
- Four independent shuffle/training seeds; each Kubernetes job allocates four
  NVIDIA RTX PRO 6000 Blackwell GPUs and trains the two specialists concurrently
  on two GPUs each.
- All two-dimensional parameters, including embeddings and the task head, are
  merged and checked. SVD, tangent projection, coefficient solving, and polar
  retraction use float64; checkpoints use float32.

This substitutes small supervised classifiers for the paper's 1.5B/7B
generative RLVR specialists. It tests the selected merger mechanism and
small-scale functional behavior, not the paper's absolute benchmark numbers or
online-RL optimizer claim.

## Methods and fixed decision rules

- ISO-Merger: released unit-retention construction, base spectrum restoration,
  and 0.9 leading-mode keep ratio.
- No restoration: identical ISO frames and mask, reconstructed with the mean
  expert spectrum.
- No mask: identical ISO construction with keep ratio 1.0.
- Task Arithmetic: shared base plus the sum of both task vectors
  (`lambda = 1.0`).
- Uniform average: arithmetic mean of the two specialist checkpoints
  (equivalently Task Arithmetic with `lambda = 0.5`).

Primary functional metric is the balanced mean of SST-2 and QNLI validation
accuracy. Per-task gain retention is `(merged - base) / (specialist - base)`.
The functional claim is aligned in this bounded setup if ISO's mean retention
is at least both baselines' mean retention across the aggregate of successful
seeds. Mechanism ablations are interpreted directionally, not required to be
significant.

The mechanistic claim passes when every two-dimensional ISO matrix has maximum
singular-value error at most `1e-10 * sigma_max` in float64 reconstruction and
at most `1e-5 * sigma_max` after float32 checkpoint casting.
