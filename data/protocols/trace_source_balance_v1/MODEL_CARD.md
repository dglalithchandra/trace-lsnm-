# TRACE A1 encoder model card

Status: **development-selected research encoder; not clinically validated**.

## Intended use

TRACE A1 converts one colour-fundus photograph into a 384-value embedding after the frozen TRACE geometry/mask preprocessing. The embedding may be used for research retrieval or as input to a few-shot head fitted on permitted labelled support examples. It is not itself a disease probability, diagnostic confidence or clinical decision.

## Selected artifact

The callable default is the predeclared screen seed 20261111, rather than whichever replication seed scored highest. Its checkpoint SHA-256 is `fead3b6163e1ee29a3f5091a899270e265c24c42c52dc679fc11a941318f7aec`. Checkpoints are excluded from Git; an authorized user must place the local artifact at `models/trace_source_balance_v1/a1_seed20261111/last.pt`.

The encoder starts from official DINOv2-S/14 and adds rank-8 LoRA to the final four attention blocks. It was continued for 9,580 optimizer updates on 1,917 label-hidden RFMiD-v1 training photographs. Photometric harmonisation is disabled.

## Development evidence

Across three training seeds, A1 improved fixed RFMiD-validation macro AUPRC over its own frozen DINOv2-S initialization by a mean +0.0341 at K=3 and +0.0474 at K=5. All seeds cleared the +0.02 screening gate. Effects were heterogeneous: EDN worsened in all seeds, while CSR and CRVO improved strongly.

Adding 595 FIVES training images produced only +0.0040 and +0.0052 at K=3/5. Source balancing added only +0.0024 and +0.0017 beyond natural sampling. Those arms failed the advancement gate and were not selected.

## Limitations and prohibited claims

- The evidence reuses RFMiD development queries and is not fresh external confirmation.
- Patient identifiers are unavailable, so patient-independent performance is not established.
- The result does not establish camera generalization, cross-site calibration, clinical safety or universal rare-disease benefit.
- Quality warnings are informative only; they are not validated failure detectors.
- Do not interpret embedding similarity as a calibrated probability.
- Do not use this component for autonomous diagnosis or patient care.

## Reproduction

```powershell
.\.venv\Scripts\python.exe scripts\embed_trace_a1_image.py --image C:\path\to\fundus.jpg --output embedding.npy
```

The loader verifies the base DINOv2 checkpoint, the A1 configuration, completion step and the selected checkpoint SHA-256 before inference.
