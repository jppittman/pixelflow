# Guide cold-start training report (strict-v1 labels)

> Budget denominator: REGISTERED_PRIMARY_BUDGET_APPLICATIONS = 100 (docs/plans/2026-09-01-phase3-registration.md §4, classical-band primary tier), imported by both gen_strict_labels and saturate_guided_until_applications so budget_fraction means the same thing at mint time and at deploy time. The first mint/train pass of this round (2026-09-01) used a 195-application placeholder (this round's measured median application count, before B was registered) — a train/deploy denominator skew, caught and fixed before this checkpoint (see git history for the placeholder-era numbers).

TRAIN: 660404 samples, 223 families, positive rate 0.5220%.

DEV (held out, never trained on): 155231 samples, 54 families, positive rate 0.5901%.

**Loss weighting**: pos_weight = 190.588 (inverse class frequency (negatives/positives = 656957/3447 measured on this TRAIN split) — the simplest defensible cold-start choice: unweighted BCE lets a trainer collapse to predicting the majority class and still score >99% raw accuracy (flagged by gen_strict_labels's own report), and there is no prior Guide run to tune a fancier (focal-loss-style) weighting against yet).

**Held-out (DEV-family) ranking quality**: AUC-ROC = 0.9902, PR-AUC (average precision) = 0.4595.

**Sanity check** — Spearman correlation between the learned per-rule mean predicted probability and the DEV-measured per-rule strict-positive rate: ρ = 0.7860. A model that learned nothing beyond noise would show ρ near 0; a model that only reproduced each rule's overall base rate (ignoring candidate-local structure) would still show ρ close to 1 here, since both quantities are monotonic in the same underlying per-rule tendency — this check confirms the model tracks the label semantics, not that it beats a per-rule lookup table (a saturation-quality evaluation, out of scope for this report, would be needed for that).

## Training curve

| epoch | lr | train weighted loss | DEV AUC | DEV PR-AUC |
|---:|---:|---:|---:|---:|
| 0 | 0.01000 | 0.955141 | 0.9869 | 0.3675 |
| 1 | 0.00952 | 0.938590 |  |  |
| 2 | 0.00909 | 0.894296 |  |  |
| 3 | 0.00870 | 0.875612 |  |  |
| 4 | 0.00833 | 0.850254 |  |  |
| 5 | 0.00800 | 0.831786 | 0.9886 | 0.3475 |
| 6 | 0.00769 | 0.795984 |  |  |
| 7 | 0.00741 | 0.770605 |  |  |
| 8 | 0.00714 | 0.752872 |  |  |
| 9 | 0.00690 | 0.750724 |  |  |
| 10 | 0.00667 | 0.732111 | 0.9888 | 0.3987 |
| 11 | 0.00645 | 0.717396 |  |  |
| 12 | 0.00625 | 0.717468 |  |  |
| 13 | 0.00606 | 0.728438 |  |  |
| 14 | 0.00588 | 0.709710 |  |  |
| 15 | 0.00571 | 0.667734 | 0.9901 | 0.4236 |
| 16 | 0.00556 | 0.676415 |  |  |
| 17 | 0.00541 | 0.640559 |  |  |
| 18 | 0.00526 | 0.647318 |  |  |
| 19 | 0.00513 | 0.633347 |  |  |
| 20 | 0.00500 | 0.624242 | 0.9889 | 0.3842 |
| 21 | 0.00488 | 0.622862 |  |  |
| 22 | 0.00476 | 0.613833 |  |  |
| 23 | 0.00465 | 0.604408 |  |  |
| 24 | 0.00455 | 0.585172 |  |  |
| 25 | 0.00444 | 0.595480 | 0.9903 | 0.4449 |
| 26 | 0.00435 | 0.594052 |  |  |
| 27 | 0.00426 | 0.580495 |  |  |
| 28 | 0.00417 | 0.569017 |  |  |
| 29 | 0.00408 | 0.583554 | 0.9902 | 0.4595 |

## Calibration (population-quantile buckets, dense toward the top, DEV)

| quantile range | n | predicted range | mean predicted | actual positive rate |
|---|---:|---|---:|---:|
| [0.000, 0.500) | 77616 | [0.000000, 0.000000] | 0.000000 | 0.000000 |
| [0.500, 0.750) | 38807 | [0.000000, 0.000000] | 0.000000 | 0.000052 |
| [0.750, 0.900) | 23285 | [0.000000, 0.000405] | 0.000045 | 0.000429 |
| [0.900, 0.950) | 7761 | [0.000405, 0.156610] | 0.020698 | 0.002964 |
| [0.950, 0.980) | 4657 | [0.156723, 0.950615] | 0.647374 | 0.023835 |
| [0.980, 0.990) | 1553 | [0.950619, 0.985138] | 0.970017 | 0.110753 |
| [0.990, 0.995) | 776 | [0.985144, 0.993831] | 0.990047 | 0.256443 |
| [0.995, 0.998) | 466 | [0.993838, 0.997650] | 0.995726 | 0.429185 |
| [0.998, 0.999) | 155 | [0.997659, 0.998751] | 0.998311 | 0.574194 |
| [0.999, 1.000) | 155 | [0.998752, 0.999865] | 0.999166 | 0.709677 |

## Per-rule: learned priority (DEV mean predicted) vs measured strict-bound rate

| rule | idx | train fired | train rate | DEV fired | DEV measured rate | DEV mean predicted |
|---|---:|---:|---:|---:|---:|---:|
| power-rsqrt | 53 | 707 | 0.18105 | 187 | 0.19251 | 0.52255 |
| power-recip | 52 | 2556 | 0.16080 | 799 | 0.14393 | 0.50052 |
| recip-sqrt | 60 | 741 | 0.16464 | 213 | 0.15962 | 0.47223 |
| power-sqrt | 51 | 3632 | 0.13546 | 994 | 0.12575 | 0.43855 |
| even-negation | 34 | 2440 | 0.05533 | 641 | 0.06240 | 0.39871 |
| even-negation | 35 | 17851 | 0.05557 | 5591 | 0.04686 | 0.37562 |
| power-combine | 47 | 1122 | 0.04278 | 319 | 0.05016 | 0.25011 |
| constant-fold | 8 | 25434 | 0.01860 | 5463 | 0.02105 | 0.21410 |
| canonicalize | 0 | 489 | 0.01636 | 175 | 0.01143 | 0.17325 |
| odd-negation | 30 | 568 | 0.01408 | 182 | 0.00549 | 0.17238 |
| cos-angle-addition | 37 | 340 | 0.00882 | 103 | 0.00000 | 0.12214 |
| reverse-angle-addition | 38 | 563 | 0.01066 | 134 | 0.00746 | 0.10965 |
| log2-exp2-cancel | 44 | 62 | 0.00000 | 14 | 0.00000 | 0.10766 |
| ln-exp-cancel | 42 | 43 | 0.00000 | 6 | 0.00000 | 0.09946 |
| doubling | 20 | 6616 | 0.00499 | 1222 | 0.00655 | 0.06161 |
| exp2-log2-cancel | 43 | 63 | 0.00000 | 14 | 0.00000 | 0.05310 |
| idempotent | 16 | 73 | 0.00000 | 19 | 0.00000 | 0.04985 |
| canonicalize | 4 | 179 | 0.00000 | 72 | 0.00000 | 0.04287 |
| fma-fusion | 59 | 109823 | 0.00407 | 22242 | 0.00486 | 0.04217 |
| odd-negation | 31 | 309 | 0.00324 | 97 | 0.01031 | 0.04106 |
| idempotent | 17 | 70 | 0.00000 | 24 | 0.00000 | 0.03580 |
| sin-angle-addition | 36 | 381 | 0.00262 | 109 | 0.00000 | 0.03387 |
| factor | 19 | 25516 | 0.00231 | 6538 | 0.00352 | 0.03159 |
| associative | 25 | 528 | 0.00379 | 496 | 0.00000 | 0.02436 |
| associative | 24 | 422 | 0.00000 | 45 | 0.00000 | 0.01818 |
| reverse-associative | 28 | 439 | 0.00000 | 47 | 0.00000 | 0.01713 |
| exp-ln-cancel | 41 | 43 | 0.00000 | 6 | 0.00000 | 0.01501 |
| reverse-associative | 27 | 33458 | 0.00078 | 9646 | 0.00187 | 0.01256 |
| exp-homomorphism | 45 | 93 | 0.00000 | 52 | 0.00000 | 0.00785 |
| associative | 23 | 30859 | 0.00055 | 8962 | 0.00045 | 0.00727 |
| associative | 22 | 51592 | 0.00033 | 9634 | 0.00021 | 0.00339 |
| inverse-annihilation | 3 | 474 | 0.01477 | 309 | 0.00000 | 0.00275 |
| power-identity | 49 | 257 | 0.00000 | 74 | 0.00000 | 0.00269 |
| reverse-associative | 29 | 547 | 0.00000 | 537 | 0.00372 | 0.00156 |
| reverse-associative | 26 | 56485 | 0.00014 | 10496 | 0.00029 | 0.00133 |
| halving | 21 | 10635 | 0.00009 | 1933 | 0.00000 | 0.00125 |
| commutative | 11 | 2925 | 0.00000 | 906 | 0.00000 | 0.00045 |
| commutative | 12 | 3166 | 0.00000 | 1258 | 0.00000 | 0.00026 |
| commutative | 10 | 82389 | 0.00000 | 22455 | 0.00000 | 0.00020 |
| commutative | 9 | 131272 | 0.00000 | 28259 | 0.00000 | 0.00015 |
| involution | 1 | 16507 | 0.00000 | 5347 | 0.00000 | 0.00006 |
| distribute | 18 | 22535 | 0.00000 | 5058 | 0.00000 | 0.00002 |
| identity | 13 | 10515 | 0.00000 | 3299 | 0.00000 | 0.00000 |
| annihilator | 15 | 3392 | 0.00000 | 1120 | 0.00000 | 0.00000 |
| half-angle-product | 39 | 221 | 0.00452 | 110 | 0.00000 | 0.00000 |
| cancellation | 2 | 9 | 0.00000 | 9 | 0.00000 | 0.00000 |
| cancellation | 6 | 48 | 0.00000 | 15 | 0.00000 | 0.00000 |
| inverse-annihilation | 7 | 56 | 0.00000 | 0 | 0.00000 | 0.00000 |
| identity | 14 | 1929 | 0.00000 | 0 | 0.00000 | 0.00000 |
| power-zero | 48 | 8 | 0.12500 | 0 | 0.00000 | 0.00000 |
| log2-power | 56 | 11 | 0.00000 | 0 | 0.00000 | 0.00000 |
| diff-of-squares | 58 | 11 | 0.00000 | 0 | 0.00000 | 0.00000 |

Checkpoint written to `pixelflow-pipeline/data/guide_checkpoint_strict_v1.json`.
