# Getting Started

This walkthrough takes you from install to a fitted Meridian model whose estimates you can compare against the simulator's ground truth.

## 1. Install

```bash
pip install -e /path/to/meridian_simulator
```

Verify:

```python
import meridian_simulator
print(meridian_simulator.__all__)
```

## 2. Your first simulation

Start minimal: two paid channels, one control, a seasonal baseline.

```python
from meridian_simulator import (
    MeridianSimulator, SimulationConfig, MediaChannelConfig,
    ContextVariableConfig, BaselineConfig, SeasonalityComponent,
)

cfg = SimulationConfig(
    n_times=104,          # two years of weekly data
    n_geos=6,
    seed=42,              # full reproducibility
    media_channels=[
        MediaChannelConfig(name="search", target_roi=3.0),
        MediaChannelConfig(name="tv",     target_roi=0.9),
    ],
    context_variables=[
        ContextVariableConfig(name="gdp_index", ar1_coef=0.8, series_mean=100.0),
    ],
    baseline=BaselineConfig(
        n_knots=26,
        trend_slope=2.0,
        seasonality=[SeasonalityComponent(amplitude=2.0, period_weeks=52)],
    ),
)

result = MeridianSimulator(cfg).run()
result.summary()
```

`summary()` prints the shape of the simulation and — the point of the exercise — the **true ROI per channel**.

## 3. What you get back

`result` is a `SimulationResult`:

| Attribute | What it is |
|---|---|
| `geo_df` | Long DataFrame, one row per geo × week, Meridian-conventional column names |
| `national_df` | Same data aggregated nationally |
| `ground_truth` | Dict of every true parameter and realized effect ([full reference](ground-truth-reference.md)) |
| `kpi_gt`, `unit_value_gt`, `population_g` | Raw arrays |
| `xr_dict` | xarray DataArrays if you prefer building `InputData` manually |
| `config` | The config that produced it (round-trippable) |

Persist everything with `result.save("output/")` → `geo_data.csv`, `national_data.csv`, `ground_truth.pkl`.

## 4. Fit Meridian on the simulated data

The `geo_df` column conventions line up with Meridian's `DataFrameInputDataBuilder`:

```python
from meridian.data.data_frame_input_data_builder import DataFrameInputDataBuilder
from meridian.model import model, spec

df = result.geo_df

builder = (
    DataFrameInputDataBuilder(kpi_type="non_revenue")
    .with_kpi(df, kpi_col="conversions")
    .with_revenue_per_kpi(df, revenue_per_kpi_col="revenue_per_conversion")
    .with_population(df, population_col="population")
    .with_media(
        df,
        media_cols=["search_impression", "tv_impression"],
        media_spend_cols=["search_spend", "tv_spend"],
        media_channels=["search", "tv"],
    )
    .with_controls(df, control_cols=["gdp_index_control"])
)

mmm = model.Meridian(input_data=builder.build())
mmm.sample_prior(500)
mmm.sample_posterior(n_chains=4, n_adapt=500, n_burnin=500, n_keep=1000, seed=1)
```

## 5. Check recovery

```python
import numpy as np
from meridian.analysis import analyzer

roi = np.asarray(analyzer.Analyzer(mmm).roi())
roi_flat = roi.reshape(-1, roi.shape[-1])

for i, ch in enumerate(result.channel_names):
    true = result.ground_truth["roi_m"][i]
    lo, hi = np.percentile(roi_flat[:, i], [5, 95])
    hit = "within" if lo <= true <= hi else "OUTSIDE"
    print(f"{ch}: true ROI {true:.2f} is {hit} the 90% CI [{lo:.2f}, {hi:.2f}]")
```

For the full automated version of this loop — 13 rubric checks, scored — use [`meridian_grader`](../../meridian_grader/README.md):

```python
from meridian_grader import MeridianGrader
MeridianGrader(mmm, ground_truth=result.ground_truth,
               gt_channel_names=result.channel_names).run_all().summary()
```

## 6. Where to go next

- Turn up the realism: traps, promos, noise, flighting — [Advanced Features](advanced-features.md)
- Tune every knob — [Configuration Reference](configuration-reference.md)
- Runnable end-to-end scripts — [examples/](../examples/)

## Reproducibility contract

The same `SimulationConfig` (including `seed`) on the same package version produces identical outputs — DataFrames included. Randomness flows through exactly two controlled sources: TensorFlow's global seed and a `numpy.random.Generator`, both set from `config.seed`.
