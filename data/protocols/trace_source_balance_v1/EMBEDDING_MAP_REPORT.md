# TRACE A1 embedding-map report

Status: **reproducible development visualization implemented; local output is PCA fallback, not UMAP**.

`scripts/generate_trace_a1_embedding_map.py` loads only the selected A1 train and validation feature files: 1,917 RFMiD training records and 639 RFMiD validation records. It never requests a test split. The output has two panels: low-support label groups and reported camera domains.

The Windows run on 13 November 2026 could not import Numba's `_dynfunc` extension because Application Control blocked the DLL. The script therefore produced an explicitly labelled two-component PCA fallback. It did **not** relabel PCA as UMAP. The same script will automatically use deterministic UMAP (`n_neighbors=15`, `min_dist=0.1`, cosine metric, seed 20261111) on a machine where the approved Numba runtime loads, such as the Mac environment after dependency verification.

Local generated artifacts are under `outputs/trace_source_balance_v1/a1_embedding_map/` and are intentionally Git-ignored. The generated summary recorded 2,556 development records and zero protected-test accesses. The reported camera domains were 1,988 Topcon 3D OCT-2000, 203 Topcon TRC-NW300 and 365 unresolved 4288×2848 records; these assignments are partly inferred from dimensions.

The plot is descriptive. Visual separation or overlap in PCA/UMAP does not prove disease discrimination, biological meaning, domain invariance or camera generalization.
