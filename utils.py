"""Utility helpers: naming, date generation, and tensor conversions."""

from __future__ import annotations

import datetime
from typing import Optional

import numpy as np


def make_channel_names(prefix: str, n: int) -> list[str]:
    return [f"{prefix}{i}" for i in range(n)]


def make_geo_names(n: int, prefix: str = "Geo") -> list[str]:
    return [f"{prefix}{i}" for i in range(n)]


def make_time_labels(n_times: int, start_date: str = "2021-01-25") -> list[str]:
    start = datetime.date.fromisoformat(start_date)
    return [
        (start + datetime.timedelta(weeks=w)).strftime("%Y-%m-%d")
        for w in range(n_times)
    ]


def add_suffix(name: str, suffix: str) -> str:
    return f"{name}_{suffix}"


def remove_suffix(name: str, suffix: str) -> str:
    return name.replace(f"_{suffix}", "")


def to_numpy(x) -> np.ndarray:
    """Convert a TensorFlow tensor or numpy array to numpy."""
    if hasattr(x, "numpy"):
        return x.numpy()
    return np.asarray(x)
