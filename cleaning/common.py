"""Paths and helpers shared by every cleaning step."""
import hashlib
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW_PATH = ROOT / "US_Accidents_March23_sampled_500k.csv"
CLEAN_PATH = ROOT / "US_Accidents_cleaned.csv"
LOG_PATH = ROOT / "reports" / "cleaning_log.md"


def md5(path):
    """Checksum of a file. Used to prove the original CSV is never modified."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


class CleaningLog:
    """Records the table shape before and after every step, plus any extra numbers a step wants to keep."""

    def __init__(self):
        self.steps = []
        self.notes = {}

    def add(self, step, detail, before_shape, after_shape):
        self.steps.append({"step": step, "detail": detail,
                           "rows_before": before_shape[0], "rows_after": after_shape[0],
                           "cols_before": before_shape[1], "cols_after": after_shape[1]})
        print(f"{step:<34} {before_shape[0]:>7,} x {before_shape[1]:<2} -> "
              f"{after_shape[0]:>7,} x {after_shape[1]:<2}  {detail}")

    def to_frame(self):
        return pd.DataFrame(self.steps)
