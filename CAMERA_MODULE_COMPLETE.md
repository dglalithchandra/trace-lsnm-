# TRACE camera module completion record

**Completion date:** 2026-11-11  
**Release:** `trace-camera-module-v1.0-research`

The camera/preprocessing work is complete at the strongest evidence-supported
boundary available from the current data.

## Final active component

- input validation;
- fundus detection;
- crop, centring, 448×448 resize and common mask;
- technical-quality measurements and provisional warnings;
- frozen DINOv2-S/RFMiD research scoring;
- standardized image, mask and machine-readable audit output.

## Rejected components

- six-characteristic photometric harmonisation;
- post-encoder shift-consistency adapter;
- learned glare detector;
- learned resolution-loss detector;
- learned field-of-view detector;
- universal confidence/conformal safety claims.

These were rejected because they failed clean-performance, external-transfer,
coverage or real-image proxy validation gates. Their artifacts remain available
for audit and must not be silently enabled.

## Final command

```powershell
.\.venv\Scripts\python.exe scripts\run_trace_camera_module.py `
  --image "C:\path\to\fundus.png" `
  --output-dir "C:\path\to\trace-output" `
  --top-k 10 `
  --device cuda
```

## Remaining external dependency

Future clinical or universal-camera claims require new expert-labelled quality
data, verified camera metadata and untouched external evaluation. Those are
evidence dependencies, not unfinished implementation in this release.

