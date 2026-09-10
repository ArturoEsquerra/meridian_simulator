# Meridian 2.x compatibility

Google released **Meridian 2.0.0 on 2026-09-03**. It changed the default compute
backend from TensorFlow to **JAX**. Because `meridian_simulator` builds
TensorFlow tensors, that default is incompatible — and because a plain
`pip install google-meridian` now resolves to 2.x, previously working notebooks
started failing with no change on your side.

## What you would see

Depending on which boundary you hit first:

```
InvalidArgumentError: cannot compute Einsum as input #1(zero-based) was
expected to be a float tensor but is a double tensor [Op:Einsum]
```

```
TypeError: Error interpreting argument to <function true_divide> as an abstract
array. The problematic value is of type
<class 'tensorflow.python.framework.ops.EagerTensor'>
```

Both trace to the same cause. Under the JAX backend Meridian's float width is
`np.float64 if jax.config.jax_enable_x64 else np.float32`, and its array ops are
JAX functions that cannot consume TF tensors. Confirmed on a real 2.0.0 install:

```
DEFAULT backend: Backend.JAX
np_float_dtype : <class 'numpy.float64'>
knot weights   : float64
```

Since `meridian_simulator` 2.2.2, the JAX combination raises a `RuntimeError`
naming the remedy instead of failing later with the messages above.

## Verified compatibility matrix

| Meridian | Backend | Status |
|---|---|---|
| 1.x | TensorFlow (only backend) | ✅ Works |
| 2.x | TensorFlow (`MERIDIAN_BACKEND=tensorflow`) | ✅ Works — output **bit-identical** to 1.x |
| 2.x | JAX (2.x default) | ❌ Not supported |

Every row was tested against a real install, not inferred.

---

## Option A — stay on Meridian 1.x (recommended)

This is the default and needs nothing from you: the package pins
`google-meridian>=1.6,<2`.

```bash
pip install -e /path/to/meridian_simulator      # brings Meridian 1.x
```

In Colab, pin explicitly, because a bare install now resolves to 2.x:

```python
%pip install -q "google-meridian<2"
%pip install -q git+https://github.com/<you>/meridian_simulator.git
```

Verify:

```python
import importlib.metadata as md
print(md.version("google-meridian"))     # expect 1.x
```

---

## Option B — run Meridian 2.x on its TensorFlow backend

Use this if you need a 2.x feature. Output is bit-identical to 1.x, so existing
datasets and recorded ground truth still reproduce.

### Step 1 — use an isolated environment

Meridian 2.x pulls `tensorflow>=2.21`, plus `jax`/`jaxlib` as **core**
dependencies. Do not install it over a working 1.x environment.

```bash
python3 -m venv ~/venvs/meridian2          # Python >= 3.10
source ~/venvs/meridian2/bin/activate
python -m pip install --upgrade pip
```

> Create the venv **without** `--system-site-packages`. Inheriting an older
> system `tensorflow-probability` shadows the pinned `tfp-nightly` and breaks
> the JAX substrate import.

### Step 2 — install Meridian 2.x

```bash
pip install "google-meridian==2.0.0"
```

Expect this warning — it is harmless:

```
WARNING: tfp-nightly 0.26.0.dev20260130 does not provide the extra 'substrates-jax'
```

Meridian 2.0.0 requests `tfp-nightly[substrates-jax]`, but that pinned nightly
only declares the extras `jax`, `tf`, and `tfds`. pip warns and installs the
base package; since `jax` and `jaxlib` are core dependencies of Meridian 2.x
they arrive anyway, so nothing is actually missing.

### Step 3 — install the simulator without disturbing the pin

The simulator pins Meridian to `<2`, so install it with `--no-deps` to stop pip
downgrading Meridian back to 1.x:

```bash
pip install --no-deps -e /path/to/meridian_simulator
```

### Step 4 — select the TensorFlow backend *first*

Meridian fixes its backend **at import time**. The environment variable has no
effect if `meridian` (or anything importing it) has already been imported.

```python
import os
os.environ["MERIDIAN_BACKEND"] = "tensorflow"   # MUST be before any meridian import

import meridian_simulator                       # safe now
```

In a notebook, put this in the **first cell**. If you set it later, restart the
kernel — re-running the cell will not help.

Or set it outside Python:

```bash
export MERIDIAN_BACKEND=tensorflow
python your_script.py
```

### Step 5 — verify before you trust it

```python
import os
os.environ["MERIDIAN_BACKEND"] = "tensorflow"

import importlib.metadata as md
from meridian.backend import config as bcfg
from meridian import backend
from meridian_simulator.utils import check_meridian_compat

print("meridian      :", md.version("google-meridian"))   # 2.0.0
print("active backend:", bcfg.get_backend())              # Backend.TENSORFLOW
print("float dtype   :", backend.np_float_dtype)          # numpy.float32
print("compat        :", check_meridian_compat(warn=False))
# {'version': '2.0.0', 'major': 2, 'backend': 'tensorflow', 'validated': True}
```

If `active backend` says `Backend.JAX`, the variable was set too late.

### Step 6 — confirm reproducibility

This exact config produces the same ROIs on 1.7.0 and on 2.0.0 + TF backend:

```python
from meridian_simulator import (
    MeridianSimulator, SimulationConfig, MediaChannelConfig,
)

r = MeridianSimulator(SimulationConfig(
    n_times=52, n_geos=3, seed=42,
    media_channels=[
        MediaChannelConfig(name="search", target_roi=3.0),
        MediaChannelConfig(name="tv", target_roi=0.8),
    ],
)).run()

print(dict(zip(r.channel_names, r.ground_truth["roi_m"].round(3))))
# {'search': 2.947, 'tv': 0.805}   <- identical on 1.x and 2.x+TF
```

If you see those two numbers, your environment is sound.

---

## Colab recipe for Option B

```python
# Cell 1 — set the backend before anything imports meridian
import os
os.environ["MERIDIAN_BACKEND"] = "tensorflow"

# Cell 2 — install (restart the runtime afterwards if prompted)
%pip install -q "google-meridian==2.0.0"
%pip install -q --no-deps git+https://github.com/<you>/meridian_simulator.git

# Cell 3 — if the runtime restarted, re-run Cell 1 FIRST, then verify
from meridian.backend import config as bcfg
print(bcfg.get_backend())        # must print Backend.TENSORFLOW
```

A runtime restart clears the environment variable, so Cell 1 must run again
before any Meridian import.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `RuntimeError: ... running on its JAX backend` | Variable set late, or not at all | Set it in the first cell, restart the kernel |
| `cannot compute Einsum ... double tensor` | simulator < 2.2.1 with Meridian 2.x | Upgrade the simulator |
| `AttributeError: module 'jax.interpreters.xla' has no attribute ...` | Old system `tensorflow-probability` shadowing the pinned nightly | Rebuild the venv without `--system-site-packages` |
| pip downgrades Meridian to 1.x | The simulator's `<2` pin | Install the simulator with `--no-deps` |
| `does not provide the extra 'substrates-jax'` | Meridian 2.0.0 packaging quirk | Harmless — ignore |

## Why the pin stays at `<2`

Meridian 2.x works only on a non-default backend that must be selected before
import — a fragile requirement to impose on every user. Keeping
`google-meridian>=1.6,<2` means a plain `pip install` always yields a working
environment; 2.x remains available opt-in via the steps above.
