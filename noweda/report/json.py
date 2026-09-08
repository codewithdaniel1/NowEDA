"""Strict JSON export without changing the in-memory analysis report."""
import json
import math
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd


def _json_safe(value):
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, np.floating):
        return _json_safe(float(value))
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Mapping):
        result = {}
        for key, item in value.items():
            # JSON object keys are strings, including numeric/tuple column labels.
            label = str(key)
            if label in result:
                raise ValueError("Column labels collide after conversion to JSON strings: " + label)
            result[label] = _json_safe(item)
        return result
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def generate_json_report(report, output_path):
    """Write UTF-8 JSON; undefined/nonfinite statistics become null.

    Column labels become strings. Colliding labels or unsupported custom plugin
    values raise before the destination is opened, preserving existing files.
    """
    content = json.dumps(_json_safe(report), indent=2, ensure_ascii=False, allow_nan=False)
    Path(output_path).write_text(content + "\n", encoding="utf-8")
