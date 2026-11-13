# TRACE A1 deployment-artifact report

Status: **LoRA-only inference artifact exported and verified**.

The selected seed-20261111 training checkpoint is 183,049,039 bytes because it contains the full student and teacher, optimizer, scaler, queue and resume state. `scripts/export_trace_a1_adapter.py` extracts only the 16 LoRA tensors required for A1 inference: 73,728 learned parameters.

The resulting local `adapter.pt` is 300,979 bytes with SHA-256 `8857f811b723cc0b5559b5a02d32ae116b6754c84fcc9c3603e835ba967e529c`. This is a **608× reduction** in the TRACE-specific artifact size (99.84% smaller). The official DINOv2-S base checkpoint remains a separate verified dependency.

The adapter records and validates:

- original A1 training-checkpoint SHA-256 `fead3b6163e1ee29a3f5091a899270e265c24c42c52dc679fc11a941318f7aec`;
- official DINOv2-S base SHA-256 `b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9`;
- the frozen A1 configuration, seed, completed update count, tensor count and parameter count.

On a real cached validation photograph, the compact adapter and full training checkpoint produced exactly equal float32 embeddings (`max_abs_difference = 0.0`). Model checkpoints and adapters remain Git-ignored; the tracked exporter and strict loader reproduce and validate the artifact.
