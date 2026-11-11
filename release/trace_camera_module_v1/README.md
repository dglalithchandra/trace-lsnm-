# TRACE Camera Module v1 — Research Release

This is the completed, evidence-locked camera/preprocessing component for
TRACE. It is a research prototype, not a medical device.

## Active pipeline

1. Validate a colour-fundus input without modifying the source file.
2. Detect the retinal field.
3. Crop, centre, resize to 448×448 and apply a common circular mask.
4. Record blur, noise, glare, compression and geometric coverage diagnostics.
5. Route invalid images to rejection and expose provisional warnings.
6. Run the frozen DINOv2-S/RFMiD disease scorer as a research guardrail.
7. Return standardized image, mask and complete JSON audit.

Photometric harmonisation, the shift adapter and learned artifact-QC models are
disabled because they failed their evidence gates.

## Run

```powershell
.\.venv\Scripts\python.exe scripts\run_trace_camera_module.py `
  --image "C:\path\to\fundus.png" `
  --output-dir "C:\path\to\trace-output" `
  --top-k 10 `
  --device cuda
```

The command refuses to overwrite existing outputs unless `--overwrite` is
explicitly supplied.

## Outputs

- `trace_standardized.png`: geometry-normalized retinal image;
- `trace_retinal_mask.png`: common circular mask;
- `trace_result.json`: preprocessing/QC audit, limitations and disease scores.

Scores are not clinical certainty or reliably externally calibrated
probabilities. Positive/negative thresholds are provisional RFMiD development
thresholds and failed to transfer reliably to Paraguay.

## Supported claim

TRACE provides camera-ID-agnostic geometric preprocessing and development
quality diagnostics for supported colour-fundus photographs.

It does **not** establish universal-camera generalisation, clinical safety,
camera-invariant representations or reliable diagnosis.

