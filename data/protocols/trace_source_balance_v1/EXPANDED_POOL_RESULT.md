# TRACE expanded-pool and source-balance result

Status: **A2 and A3 screens complete; neither advanced**. Development evidence only; zero protected-test images were opened.

## Frozen expanded pool

The expanded manifest contains 2,512 label-hidden training photographs: 1,917 RFMiD training records and 595 exact-deduplicated official FIVES training records. FIVES test remains unopened and excluded. The manifest SHA-256 is:

`d03badaddbf591b177180c2b9c1c24ea8b06670b2cc68611c0c40d0029f5e272`

## A2: expanded pool, natural-frequency sampling

A2 used the same initialization, objective, augmentations, seed and 9,580 optimizer updates as matched A1 seed 20261111. It changed only the training pool. The 38,320 draws were 29,250 RFMiD (76.33%) and 9,070 FIVES (23.67%), matching the manifest proportions.

Training completed in 1,451.72 seconds and opened zero protected-test images. Checkpoint SHA-256:

`5ee7a2064665d36946d47990c1d32406bd552e0d65feea9e459aa23e60adae54`

| Arm | K | Macro AUPRC | Macro AUROC |
|---|---:|---:|---:|
| A1 RFMiD-only | 3 | 0.2313 | 0.8106 |
| A2 RFMiD+FIVES natural | 3 | 0.2353 | 0.8123 |
| A1 RFMiD-only | 5 | 0.2725 | 0.8479 |
| A2 RFMiD+FIVES natural | 5 | 0.2777 | 0.8504 |

Matched A2 minus A1 macro-AUPRC differences:

- K=3: **+0.0040**, within-seed 95% interval **[-0.0031, +0.0106]**.
- K=5: **+0.0052**, within-seed 95% interval **[-0.0017, +0.0119]**.

A2 failed the predeclared +0.02 gate at both K values. The harm gates passed, but the small point gains are compatible with no material benefit. The evidence therefore does not support the claim that simply adding FIVES improved RFMiD few-shot ranking.

## A3: expanded pool, source-balanced sampling

A3 used the same expanded pool, initialization, objective, augmentations, seed and 9,580 optimizer updates as A2. The only change was the sampler. Its 38,320 draws were exactly balanced: 19,160 RFMiD and 19,160 FIVES.

Training completed in 1,455.41 seconds and opened zero protected-test images. Checkpoint SHA-256:

`bf5d9e5773ba794937d5fa9a5dcdcebbbaf47627534edabb0d642c6b03ebacfb`

| Arm | K | Macro AUPRC | Macro AUROC |
|---|---:|---:|---:|
| A2 natural sampling | 3 | 0.2353 | 0.8123 |
| A3 source-balanced | 3 | 0.2377 | 0.8145 |
| A2 natural sampling | 5 | 0.2777 | 0.8504 |
| A3 source-balanced | 5 | 0.2794 | 0.8517 |

Matched A3 minus A2 macro-AUPRC differences:

- K=3: **+0.0024**, within-seed 95% interval **[-0.0033, +0.0085]**.
- K=5: **+0.0017**, within-seed 95% interval **[-0.0057, +0.0095]**.

The clean-AUROC and per-label harm gates passed; the worst per-label AUPRC difference was -0.0209. However, A3 failed the predeclared +0.02 improvement gate at both K values, and both uncertainty intervals include zero.

## Decision

Neither adding FIVES under natural sampling nor balancing RFMiD and FIVES produced a material improvement on this RFMiD development endpoint. A2 and A3 will not be replicated. The three-seed A1 arm remains the strongest justified adapted model in this experiment. This does **not** show that FIVES is useless or that source balancing never works; it shows only that these changes did not clear the fixed gate under this dataset, objective and evaluation.

The protected test remains unopened. These results do not support claims of camera generalization, clinical safety or external-dataset superiority.
