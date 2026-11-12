# TRACE source-balance development protocol v1

Status: **core arm ready; expanded-pool arms blocked pending record-level admission**. Frozen 12 November 2026 before any A2/A3 training. This is a development study and does not modify `trace_final_v1`.

## Question

At a fixed DINOv2-S initialization, optimizer, augmentation recipe, image resolution and update budget:

1. Does adding an audited multi-source label-hidden fundus pool improve low-support ranking over the RFMiD-only SSL control?
2. Given that same expanded pool, does source-balanced sampling improve ranking over natural-frequency sampling?

The objective remains the implemented TRACE SSL recipe: rank-8 LoRA in the final four DINOv2-S attention blocks, projection head, EMA teacher and MoCo-style queue. DINO/iBOT, lesion-guided views, full-field-versus-crop retesting, learned QC, feature fusion and graph reranking are outside this protocol.

## Arms and estimands

| Arm | Definition | Estimand |
|---|---|---|
| A0 | Frozen DINOv2-S | Operational reference |
| A1 | Existing SSL recipe on the 1,917-image RFMiD-v1 training pool | Overall core-pool adaptation, A1 minus A0 |
| A2 | Same recipe and updates on an admitted expanded pool with natural-frequency sampling | Added-pool effect, A2 minus A1 |
| A3 | Same admitted expanded pool and updates with equal expected source contribution per draw | Balancing effect, A3 minus A2 |

A2 and A3 cannot start until every record has an allowed role, source ID, hash and related-group exclusion decision. A source folder or aggregate image count is not admission.

## Fixed model and training controls

- Initial checkpoint: verified official DINOv2-S/14, SHA-256 `b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9`.
- Input: cached frozen TRACE geometry at 448 pixels; photometric correction disabled.
- View recipe: the existing `plain_crop_matched` recipe. The prior 20-epoch full-field-minus-crop difference was +0.0017 with an interval crossing zero, so view selection is closed for this study.
- Training recipe: rank 8, alpha 8, last four attention blocks, projection 384-384-128, queue 4,096, temperature 0.2, EMA momentum 0.996, AdamW, learning rate 0.0001 and weight decay 0.01.
- Screen seed: 20261111. Replication seeds, if a candidate passes screening: 20261111, 20261112 and 20261113.
- Screen budget: 9,580 optimizer updates for every trained arm. Epoch counts may differ because dataset sizes differ. The sampler must record every source draw.
- Effective batch size, accumulation, precision and optimizer-step definition are frozen after the separate 50-step benchmark. If the recipe cannot complete within the sprint budget, revise this protocol before A2/A3 scores exist and record the revision.

Existing epoch-20 A1 artifacts may support development context, but a mechanism claim involving A2/A3 requires matched implementations, update accounting and seeds. Do not insert A2/A3 into the locked old final comparison.

## Development evaluation

- Queries: the existing 639 RFMiD-v1 validation records, in their frozen order. These queries are excluded from every SSL pool and support set.
- Support: the existing `trace_fewshot_v1` RFMiD-v1 training schedules and negatives, identically reused across encoders.
- Primary endpoint: macro AUPRC at K=3 over count-eligible fixed low-support labels.
- Secondary endpoints: macro AUPRC at K=5; per-label AUPRC and AUROC; macro AUROC; AP divided by prevalence; sensitivity at a separately fitted fixed-specificity rule where a distinct permitted tuning partition exists.
- K=1 is descriptive. K=10/20 are outside this version. K=0 does not apply to these image-only encoders.
- Pairing: identical support draws, queries, labels, head and query ordering for all arms.
- Uncertainty: paired resampling over labels, support draws and shared query/image indices when per-query scores are available. The same sampled image indices must apply to all labels and compared models where their query sets align. Patient clustering is not modeled where patient IDs are absent.

## Screening and claim rules

An arm advances only when the K=3 and K=5 macro-AUPRC point differences for its intended contrast are positive, clean macro AUROC decreases by no more than 0.01, and no eligible label loses more than 0.10 mean AUPRC. These are engineering screening rules, not proof of superiority.

After three matched seeds, report the distribution across seeds and paired uncertainty. A superiority statement requires the predeclared paired interval for the primary contrast to exclude zero. Sign agreement alone is insufficient. If the evidence fails, retain A0 or the strongest simpler control and report the result.

The old locked RFMiD-v1 test comparison cannot select this model. Without a separately frozen fresh cohort or organizer-hidden evaluation, new A1-A3 findings remain development-level.

## Shift, calibration and explanation scope

Paraguay, PALM, EDID, JSIEC and RFMiD 2.0 remain exploratory/development sources under their existing limits. If target-site recalibration is evaluated, its calibration records and evaluation records must be disjoint and grouped by known patient/eye/group identifiers when available. Report ranking, threshold transfer, calibration and risk-coverage separately.

The TRACE graph is explanation-only in this sprint. It preserves input scores and ordering and cannot create image-level findings or change predictions.

## Leakage and admission rules

- Never admit a test image, current/future query image, known patient/eye/group mate or confirmed/likely near duplicate of one.
- Keep dataset source split and original identifiers. Never infer patient or camera identity from filename or image size without recording the mapping basis.
- Record licence/access basis, archive/member hash, preprocessing hash and likely foundation-model exposure.
- DR-heavy sources may be admitted only with an explicit source-balance analysis; image count alone is not a reason for inclusion.
- Any cohort used to tune data, sampling, model, head, threshold, calibration or graph content becomes development data.

## Current gate result

Only A0 and the existing core A1 are ready. EyePACS/APTOS images are absent locally; FDR26 is incomplete; EyeQ contains labels without photographs. Local PALM, Paraguay, EDID and JSIEC retain evaluation roles. RFMiD 2.0 test is blocked. IDRiD and FIVES are not admitted to SSL v2 until their overlaps, intended future evaluation roles and record-level exclusions are resolved.
