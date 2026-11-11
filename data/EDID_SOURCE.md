# Eye Disease Image Dataset source and role

Source: [Mendeley Data, version 1](https://data.mendeley.com/datasets/s9bfhswzjb/1), DOI `10.17632/s9bfhswzjb.1`. The linked [data paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC11528549/) describes images collected at two hospitals in Bangladesh. The Mendeley record lists CC BY 4.0.

The official Mendeley file API lists two separate archives. TRACE acquires **Original Dataset.zip** only. The augmented archive is excluded from evaluation because its transformed copies would inflate counts and could cross splits.

| Official file | Bytes | Official SHA-256 |
|---|---:|---|
| Original Dataset.zip | 1,655,647,957 | `4cc006a9a769cb5172d552b69395d65fa1f1794af27baf32574a027051ea188d` |

The source reports 5,335 original images over ten classes. One class, pterygium, contains anterior-segment images and must be excluded from a retinal fundus benchmark. Candidate cross-dataset comparisons include macular scar and optic disc edema. Central serous chorioretinopathy versus RFMiD's central serous retinopathy requires definition review. Do not use a class-folder label as evidence that every other disease is absent from the image.

The official archive was downloaded and matched the published SHA-256. All 5,335 JPEG members read successfully. However, 470 byte-identical groups contain 484 extra copies; **464 groups span different class folders**, including 201 groups involving the `Healthy` folder. Exact copies may indicate multi-condition images stored under several labels or inconsistent folder assignments. The archive cannot be treated as an ordinary single-label benchmark. The conservative exploratory manifest excludes every cross-folder group, retains one copy of within-folder duplicates, and excludes pterygium. It leaves 4,370 distinct fundus image files, including 352 macular-scar, 128 retinitis-pigmentosa, 108 disc-edema, and 823 healthy-folder images. These are folder counts after exact-image filtering, not verified independent-patient counts.

This source does not provide verified per-image camera IDs or patient identifiers in the public description or image EXIF. It can strengthen exploratory external label evaluation after further near-duplicate and cross-dataset image screening. It cannot by itself establish camera-specific transfer or patient-independent performance.

Download and audit scripts: `scripts/download_edid_original.py`, `scripts/audit_edid_original.py`, and `scripts/build_edid_exploratory_manifest.py`. The local archive is excluded from git. The audit outputs are under `data/manifests/`.
