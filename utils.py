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


# ---------------------------------------------------------------------------
# Dtype safety at the Meridian boundary
# ---------------------------------------------------------------------------
#
# The simulator builds all of its own tensors in float32. Meridian, however,
# does not guarantee the dtype of what it hands back:
#
#   * Meridian >= 2.0 defaults to a JAX backend whose float width is
#     ``np.float64 if jax.config.jax_enable_x64 else np.float32``.
#   * Arrays produced under that backend (e.g. ``knots.get_knot_info().weights``,
#     which is built with ``dtype=backend.np_float_dtype``) can therefore be
#     float64, or a JAX array rather than a TF tensor.
#
# Mixing those into a float32 graph raises errors such as::
#
#     InvalidArgumentError: cannot compute Einsum as input #1(zero-based) was
#     expected to be a float tensor but is a double tensor [Op:Einsum]
#
# ``as_float`` normalises anything Meridian returns to the simulator's working
# dtype, so the simulator is insulated from Meridian's backend and precision
# settings. Always route Meridian-produced arrays through it before combining
# them with simulator tensors.

FLOAT_DTYPE = "float32"


def as_float(x, dtype=None):
    """Coerce any array-like to a TensorFlow tensor of the working dtype.

    Handles numpy arrays (any precision), TF tensors, JAX arrays, and Python
    scalars/sequences. This is the single conversion point for values that
    originate inside Meridian.

    Args:
        x: Array-like value (numpy, TF, JAX, list, scalar).
        dtype: Target dtype; defaults to the simulator's ``FLOAT_DTYPE``
            (float32).

    Returns:
        A ``tf.Tensor`` of the requested dtype.
    """
    import tensorflow as tf  # local import: keeps TF out of config import path

    target = tf.as_dtype(dtype or FLOAT_DTYPE)
    if isinstance(x, tf.Tensor):
        return x if x.dtype == target else tf.cast(x, target)
    # np.asarray also converts JAX arrays and other __array__ providers.
    return tf.convert_to_tensor(np.asarray(x), dtype=target)


def check_meridian_compat(warn: bool = True) -> dict:
    """Inspect the installed Meridian and report compatibility.

    The simulator is a TensorFlow package: it builds tensors with ``tf`` and
    combines them with values produced inside Meridian. Meridian 2.0 introduced
    a backend abstraction that defaults to **JAX**, whose float width is
    ``float64`` when ``jax_enable_x64`` is set. ``as_float`` insulates the
    simulator from that, but the pairing is not yet validated end to end, so
    this reports what is installed and how to pin it down.

    Set ``MERIDIAN_BACKEND=tensorflow`` *before* importing Meridian to force the
    TensorFlow backend, which fixes Meridian's float width at float32.

    Args:
        warn: Emit a ``UserWarning`` when a not-yet-validated combination is
            detected.

    Returns:
        Dict with ``version``, ``major``, ``backend``, and ``validated``.
    """
    import warnings as _warnings

    info = {"version": None, "major": None, "backend": None, "validated": False}
    try:
        import importlib.metadata as _md

        info["version"] = _md.version("google-meridian")
        info["major"] = int(str(info["version"]).split(".")[0])
    except Exception:  # noqa: BLE001 - best effort only
        return info

    # Which backend is actually live? `get_backend()` is the authority: it
    # reflects the backend Meridian initialised at import time (which is why
    # MERIDIAN_BACKEND must be set *before* importing meridian). Fall back to
    # the environment variable, then the declared default, then "unknown" --
    # never report a backend we did not actually observe.
    import os as _os

    backend_val = None
    try:
        from meridian.backend import config as _bcfg  # type: ignore

        getter = getattr(_bcfg, "get_backend", None)
        if callable(getter):
            backend_val = getattr(getter(), "value", None) or str(getter())
        if backend_val is None:
            default = getattr(_bcfg, "_DEFAULT_BACKEND", None)
            if default is not None:
                backend_val = getattr(default, "value", None) or str(default)
    except Exception:  # noqa: BLE001 - reporting only
        pass
    if backend_val is None:
        backend_val = _os.environ.get("MERIDIAN_BACKEND")
    if backend_val is None and info["major"] == 1:
        backend_val = "tensorflow"  # 1.x predates the JAX default
    info["backend"] = (backend_val or "unknown").lower()

    # Verified combinations (see CHANGELOG 2.2.1):
    #   1.x  + TensorFlow           -> works
    #   2.x  + TensorFlow backend   -> works, bit-identical output
    #   2.x  + JAX backend          -> BROKEN: Meridian's JAX ops reject the
    #                                  TF tensors this simulator builds.
    backend = (info["backend"] or "").lower()
    info["validated"] = info["major"] == 1 or backend == "tensorflow"

    if info["major"] is not None and info["major"] >= 2 and backend == "jax":
        raise RuntimeError(
            f"meridian_simulator is a TensorFlow package, but google-meridian "
            f"{info['version']} is running on its JAX backend, which cannot "
            f"consume TensorFlow tensors (you would hit errors such as "
            f"\"Error interpreting argument ... as an abstract array ... "
            f"EagerTensor\").\n\n"
            f"Fix by either:\n"
            f"  (a) selecting Meridian's TensorFlow backend BEFORE importing "
            f"meridian -- this is verified to work and reproduces 1.x output "
            f"exactly:\n"
            f"        import os\n"
            f"        os.environ['MERIDIAN_BACKEND'] = 'tensorflow'\n"
            f"        import meridian   # must come after the line above\n"
            f"  (b) pinning Meridian 1.x:  pip install 'google-meridian<2'\n\n"
            f"If Meridian was already imported, the environment variable has "
            f"no effect -- restart the kernel and set it in the first cell."
        )

    if warn and not info["validated"]:
        _warnings.warn(
            f"meridian_simulator is validated against google-meridian 1.x and "
            f"against 2.x on its TensorFlow backend; found {info['version']} "
            f"(backend={info['backend']}). If you hit dtype or backend errors, "
            f"set MERIDIAN_BACKEND=tensorflow before importing meridian, or "
            f"pin `pip install \"google-meridian<2\"`.",
            UserWarning,
            stacklevel=2,
        )
    return info
