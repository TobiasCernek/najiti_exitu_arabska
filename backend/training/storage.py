"""Shared storage paths for training data.

Extracted frames can become huge, so keep them out of the project tree by
default. Override with EXITNAV_TRAINING_DIR when you want a custom dataset path.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path


def get_training_dir() -> Path:
    custom = os.getenv("EXITNAV_TRAINING_DIR")
    if custom:
        return Path(custom).expanduser()

    base = os.getenv("LOCALAPPDATA") or os.getenv("APPDATA")
    if base:
        return Path(base) / "ExitNav" / "training"

    return Path(tempfile.gettempdir()) / "ExitNav" / "training"


TRAINING_DIR = get_training_dir()
