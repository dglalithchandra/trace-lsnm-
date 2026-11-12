# TRACE SSL v1 starting point

Status: **training input, view preflight and five-epoch matched contrastive pilot complete; no adapted encoder selected**. See `FIVE_EPOCH_PILOT.md`.

`unlabeled_train_manifest.csv` contains 1,917 RFMiD-v1 **training** photographs and only image IDs, ZIP member paths and image hashes. The SSL cache and trainer consume this manifest, not the disease-label CSV or broad analysis manifest. EDID, JSIEC, RFMiD validation and the protected RFMiD test remain outside the SSL training image pool. `protocol.json` records the input hash and intended matched comparison.

The view preflight sampled 60 training images with fixed seed 20261111. A standard random crop retained an average 95.8% of the retinal mask (minimum 85.3%); a mild rotation/translation full-field view retained an average 99.999% (minimum 99.984%). The eight-row contact sheet at `tmp/trace_ssl_view_preflight.jpg` was inspected locally. These are **geometric mask-retention measurements**, not proof of lesion preservation, encoder improvement or clinical safety. The full-field view was tested against the crop baseline at matched images, backbone, optimizer, update count and evaluation episodes.

The training script is `scripts/train_trace_ssl_v1.py`; the matched pilot and limits are in `FIVE_EPOCH_PILOT.md`. Retain the frozen operational baseline until a predeclared transfer gate is met. The full-field view itself is an adaptation design, not established novelty.
