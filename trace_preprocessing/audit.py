"""Write reproducible, image-free preprocessing audit records."""

from __future__ import annotations

import json
from pathlib import Path

from .types import PreprocessingResult


def write_audit_record(result: PreprocessingResult, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(result.audit_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
