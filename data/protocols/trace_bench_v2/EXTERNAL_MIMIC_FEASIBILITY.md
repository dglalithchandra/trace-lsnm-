# External disease-control feasibility after EDID

11 November 2026. This is an **outcome-blind source and count audit** for potential follow-up cohorts. No JSIEC encoder scores were opened.

## JSIEC-1000, already acquired

The official JSIEC-1000 ZIP in `data/external/JSIEC/` contains 1,000 image files: 997 JPEGs and three TIFFs. Its exact-byte audit found four duplicate groups, all within a category, leaving 996 distinct eligible files. The candidate categories contain **22 retinitis pigmentosa (RP), eight Bietti crystalline dystrophy, and 37 normal** images after exact deduplication. None of these 67 files is byte-identical to an image in the EDID archive or RFMiD-v1 training pool. This does not rule out re-encoded/cropped copies or related eyes. The manifest and counts are in `data/manifests/jsiec_exploratory_manifest.csv` and `data/manifests/jsiec_exploratory_summary.json`.

The [original JSIEC study](https://pmc.ncbi.nlm.nih.gov/articles/PMC8355164/) groups RP and Bietti as related subclasses with similar fundus signs. Its larger research collection used ophthalmologist and senior-retina-specialist labeling. We have **not** verified patient identifiers, complete multilabel annotations, per-image camera IDs, or a patient-level split for this 1,000-image public subset. The local source record also has unclear redistribution terms (`data/external/JSIEC/SOURCE.md`).

**Decision:** RP versus Bietti could be a descriptive hard-control stress test after near-copy screening, but eight Bietti images are too few to establish a stable ranking difference between encoders. Do not promote this subset to a confirmatory differential-diagnosis benchmark. The public JSIEC subset has no direct macular-scar category, so it cannot independently resolve the EDID macular-scar anomaly.

## Other candidate sources

- [RIPS](https://www.icar.cnr.it/en/sites-rips-datasetrips/) has 120 retinal images from only four patients and is designed for pigment-sign segmentation. It requires a signed agreement and author-provided download instructions. Its many images are not 120 independent cases. It is unsuitable for a robust independent-patient RP comparison.
- The [Inherited Retinal Diseases Image Bank](https://github.com/riovs/ird_dataset) describes 391 individuals, including 158 RP, 62 Stargardt and 171 healthy. Its images require an access request, so it is a promising future patient-level source but not currently available locally. Its label definitions and data terms would need review before use.
- [BRSET](https://physionet.org/content/brazilian-ophthalmological/1.0.0/) provides patient and camera metadata plus multiple disease/finding fields, but the image files require credentialed PhysioNet access, which the team does not have. Do not represent it as an available validation set.

## Project consequence

There is no presently available external cohort here that can turn the EDID RP or macular-scar result into a strong patient-independent differential-diagnosis claim. Keep EDID and any JSIEC RP/Bietti analysis explicitly exploratory. Continue the challenge's central encoder experiment with a frozen evaluation protocol and accessible training data. Do not tune the encoder using EDID or JSIEC and then call either a confirmatory test.
