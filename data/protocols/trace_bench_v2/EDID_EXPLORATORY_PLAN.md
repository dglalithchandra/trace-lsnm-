# EDID external label evaluation plan

Status: **dataset acquired, duplicate screened, exploratory query manifest frozen; model evaluation pending**.

## Scope

Use the original-image EDID archive as an exploratory external cohort for RFMiD-trained few-shot support. It can help test whether low-support finding representations transfer to a different dataset. It is not a per-camera evaluation. Patient IDs and per-image camera IDs are unavailable.

## Query construction

Read `data/manifests/edid_exploratory_manifest_v2.csv`. Use only rows with `analysis_eligible=True`. Every reviewed exact or near-copy group present in multiple class folders is excluded from all queries. Within-class reviewed copies contribute one query. Exclude the anterior-segment Pterygium class.

Primary target folders: `Macular Scar` (326 eligible images) and `Retinitis Pigmentosa` (124). Use the 777 eligible `Healthy` images as controls for each target. Do not label images from other disease folders as target-negative just because their folder has a different name. Secondary sanity target: `Disc Edema` (106 images) versus the same healthy controls. `Central Serous Chorioretinopathy` is withheld pending source-definition mapping to RFMiD `CSR`. `Retinal Detachment` must not be equated with RFMiD retinal traction (`RT`).

Support comes only from RFMiD v1 training. For macular scar, reuse the frozen RFMiD v1 1/3/5-shot episode plan. For retinitis pigmentosa, RFMiD has only six training positives; create a deterministic 1/3-shot support schedule before examining EDID scores. The existing five-shot schedule would have only six possible positive subsets, so do not present 50 nominal replicates as 50 independent draws. Use the existing base-label-selected head settings and the same query records for every frozen encoder. Disc edema is a base-label transfer check, not a novel-label result.

Report per-target AUROC, AUPRC, AUPRC divided by target prevalence, positive and control counts, uncertainty intervals, and each encoder's results. Treat image-level intervals as optimistic when the same patient may have contributed multiple photos. Compare target-positive versus healthy controls only; this is an easier task than distinguishing the target from other eye diseases. No result should be framed as clinical screening accuracy or camera invariance.

## Duplicate-screen gate completed before scoring

1. The eligible EDID pool was screened using coarse hashes followed by local retinal-detail geometry. All 66 passing pairs were reviewed side by side. Ten further passing pairs against previously excluded exact cross-class groups were also reviewed. The 76 decisions are logged in `data/manifests/edid_near_duplicate_visual_review.csv`.
2. A priority cross-dataset screen checked 75 EDID-to-RFMiD-v1-training pairs with local retinal-detail geometry. None passed. This does not prove that all cross-dataset overlap is absent.
3. The revised query manifest is frozen at `data/manifests/edid_exploratory_manifest_v2.csv`. Freeze the deterministic RP support schedule before loading EDID model scores. Hash screens are not exhaustive; patient independence and clinical label validity are unverified.

The official archive, audit counts, and source limitations are documented in `data/EDID_SOURCE.md`. The restricted BRSET and Birdshot-WIDE datasets are not prerequisites for this exploratory run.
