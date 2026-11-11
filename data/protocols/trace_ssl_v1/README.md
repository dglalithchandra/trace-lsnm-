# TRACE SSL v1 starting point

Status: **training input and view preflight frozen; no contrastive checkpoint trained yet**.

`unlabeled_train_manifest.csv` contains 1,917 RFMiD-v1 **training** photographs and only image IDs, ZIP member paths and image hashes. The loader for the forthcoming SSL experiment must consume this file, not the disease-label CSV or broad analysis manifest. EDID, JSIEC, RFMiD validation and the protected RFMiD test remain outside the SSL training image pool. `protocol.json` records the input hash and intended matched comparison.

The view preflight sampled 60 training images with fixed seed 20261111. A standard random crop retained an average 95.8% of the retinal mask (minimum 85.6%); a mild rotation/translation full-field view retained an average 99.999% (minimum 99.989%). The eight-row contact sheet at `tmp/trace_ssl_view_preflight.jpg` was inspected locally. These are **geometric mask-retention measurements**, not proof of lesion preservation, encoder improvement or clinical safety. The full-field view is a candidate to test against the crop baseline at matched images, backbone, optimizer, update count and evaluation episodes.

Next implementation: a label-hidden MoCo/InfoNCE training script starting from the verified DINOv2-S/14 weights, with a no-gradient momentum teacher updated by exponential moving average, bounded view transformations, checkpoint/resume and collapse checks. Train the plain-crop baseline and full-field candidate with matched compute before selecting any added neighbor weighting or patch loss. Evaluate only on existing development episodes; retain the best frozen baseline unless the predeclared transfer gate is met. The full-field view itself is an adaptation design, not established novelty.
