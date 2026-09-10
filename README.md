# meridian_simulator

**Simulate realistic, Meridian-compatible MMM datasets with known ground truth.**

`meridian_simulator` generates complete geo-level Marketing Mix Modeling datasets for [Google Meridian](https://github.com/google/meridian) — media impressions and spend, reach & frequency channels, organic media, controls, and a KPI assembled from a fully specified causal data-generating process. Because *you* wrote the ground truth, you can verify whether any MMM — Meridian or otherwise — actually recovers it.

Built for three audiences:

- **MMM practitioners** validating modeling choices (priors, adstock, knots) against data where the right answer is known
- **Educators & competition designers** creating challenge datasets with deliberate, realistic traps
- **Researchers** benchmarking MMM methods on data with configurable difficulty

Its companion package, [`meridian_grader`](../meridian_grader), automatically grades fitted Meridian models against the ground truth this simulator records.

---

## Why simulate?

An MMM fitted on real data can never be validated — the true ROI is unobservable. Simulated data inverts that: every ROI, adstock decay, saturation curve, and baseline component is chosen up front and recorded, so "did the model recover the truth?" becomes a measurable question. The catch is that naive simulated data is *too easy*: i.i.d. media, clean controls, no measurement noise. This simulator closes that gap with **native realism features** — multicollinearity traps, endogenous variables, promotional shocks, demand-synchronized flighting, and controllable observation noise — so your synthetic data interrogates a model the way real data would.

## Installation

```bash
git clone https://github.com/<you>/meridian_simulator
pip install -e meridian_simulator
```

Requires Python ≥ 3.10 and `google-meridian` (the simulator reuses Meridian's own transformers, priors, and adstock/Hill code so simulated data is consistent with what Meridian expects).

### Meridian version compatibility

**Default: `google-meridian` 1.x**, pinned automatically (`>=1.6,<2`) so a plain install always works.

Meridian **2.0.0** (2026-09-03) changed its default compute backend from TensorFlow to **JAX**. Because this simulator builds TensorFlow tensors, the JAX backend is incompatible — Meridian's JAX ops reject TF tensors. Verified matrix:

| Meridian | Backend | Status |
|---|---|---|
| 1.x | TensorFlow (only) | ✅ Works |
| 2.x | TensorFlow | ✅ Works — output **bit-identical** to 1.x |
| 2.x | JAX (2.x default) | ❌ Not supported |

Running 2.x on JAX raises a `RuntimeError` naming the fix, rather than failing later with a confusing tensor-type error.

**On Colab**, pin explicitly — a bare `pip install google-meridian` now resolves to 2.x:

```python
%pip install -q "google-meridian<2"
%pip install -q git+https://github.com/<you>/meridian_simulator.git
```

**To use Meridian 2.x**, select its TensorFlow backend in the very first cell, before anything imports `meridian`:

```python
import os
os.environ["MERIDIAN_BACKEND"] = "tensorflow"   # must precede `import meridian`
```

Full instructions: [docs/meridian-2-compatibility.md](docs/meridian-2-compatibility.md).

## Quickstart

```python
from meridian_simulator import (
    MeridianSimulator, SimulationConfig, MediaChannelConfig,
    ContextVariableConfig, BaselineConfig, SeasonalityComponent,
)

cfg = SimulationConfig(
    n_times=104, n_geos=6, seed=42,
    media_channels=[
        MediaChannelConfig(name="search", target_roi=3.0),
        MediaChannelConfig(name="tv",     target_roi=0.9),
    ],
    context_variables=[
        ContextVariableConfig(name="gdp_index", ar1_coef=0.8, series_mean=100.0),
    ],
    baseline=BaselineConfig(
        n_knots=26, trend_slope=2.0,
        seasonality=[SeasonalityComponent(amplitude=2.0, period_weeks=52)],
    ),
)

result = MeridianSimulator(cfg).run()
result.summary()                      # printed overview incl. true ROIs
result.geo_df                         # long geo × week DataFrame, Meridian-ready
result.ground_truth["roi_m"]          # the answers
result.save("output/")                # CSVs + ground_truth.pkl
```

`result.geo_df` plugs straight into Meridian's `DataFrameInputDataBuilder` — column naming follows its conventions (`{channel}_impression`, `{channel}_spend`, `{var}_control`, …).

## The full arsenal

| Capability | Config | One-liner |
|---|---|---|
| Paid media (impressions) | `MediaChannelConfig` | `target_roi` back-solves the true ROI; pin `alpha`/`ec`/`slope` or sample from Meridian priors |
| Paid media (reach & frequency) | `RFChannelConfig` | Poisson-arrival reach/frequency derivation |
| Organic media | `OrganicMediaChannelConfig` / `OrganicRFChannelConfig` | Unpaid channels with adstock+Hill but no spend |
| Non-media treatments | `NonMediaChannelConfig` | Price, distribution, promo intensity series |
| Controls | `ContextVariableConfig` | AR(1) persistence, trends, geo heterogeneity |
| Baseline | `BaselineConfig` | Geo intercepts, spline trend (knots), stacked seasonal waves |
| **Collinear traps** | `CollinearVariableConfig` | Distractors derived from population/controls; zero causal effect |
| **Endogenous traps** | `EndogenousVariableConfig` | Driven by lagged media spend or the KPI itself (reverse causality) |
| **Promo events & shocks** | `PromoEventConfig` | Multiplicative KPI lifts; hideable flags; negative shocks; `baseline_only` mode |
| **Demand-synced flighting** | `seasonal_flighting` on channels | Media buys INTO high season → media–seasonality confounding |
| **KPI observation noise** | `kpi_noise_pct` | One knob: coefficient of variation of sales measurement error |

Everything is recorded in `result.ground_truth` — including the *realized* correlations, multipliers, and noise, not just the configured targets.

## A competition-grade dataset in one config

```python
from meridian_simulator import (
    MeridianSimulator, SimulationConfig, MediaChannelConfig,
    OrganicMediaChannelConfig, ContextVariableConfig,
    CollinearVariableConfig, EndogenousVariableConfig, PromoEventConfig,
    BaselineConfig, SeasonalityComponent,
)

cfg = SimulationConfig(
    n_times=104, n_geos=6, seed=2024,
    media_channels=[
        MediaChannelConfig(name="google", target_roi=3.0, alpha=0.25, max_lag=4),
        MediaChannelConfig(name="youtube", target_roi=2.5, alpha=0.55,
                           seasonal_flighting=0.5),      # buys into high season
        MediaChannelConfig(name="tv", target_roi=0.8, alpha=0.70, max_lag=12),
    ],
    organic_media_channels=[OrganicMediaChannelConfig(name="organic_search")],
    context_variables=[
        ContextVariableConfig(name="gdp_index", ar1_coef=0.85, series_mean=100.0),
        ContextVariableConfig(name="competitor_spend", ar1_coef=0.6, series_mean=5.0),
    ],
    collinear_variables=[            # multicollinearity traps
        CollinearVariableConfig(name="store_count", source="population",
                                coefficient=0.03, noise_std=5.0, round_decimals=0),
        CollinearVariableConfig(name="consumer_confidence", source="gdp_index",
                                coefficient=0.6, intercept=40.0, noise_std=3.0),
    ],
    endogenous_variables=[           # endogeneity trap (like Google query volume)
        EndogenousVariableConfig(name="gqv_index", driver="youtube", lag=1, weight=0.65),
    ],
    promo_events=[                   # structural lifts with HIDDEN flags
        PromoEventConfig(name="hot_sale", weeks=[20, 21], lift_pct=0.30,
                         include_flag_in_output=False),
        PromoEventConfig(name="stockout", weeks=[60], lift_pct=-0.40,
                         allow_negative_lift=True),      # negative shock
    ],
    kpi_noise_pct=0.03,              # 3% sales measurement noise
    baseline=BaselineConfig(
        n_knots=26, trend_slope=4.0,
        seasonality=[SeasonalityComponent(amplitude=3.0, period_weeks=52)],
    ),
)

result = MeridianSimulator(cfg).run()
```

Contestants receive `result.geo_df` (traps included, promo flags absent). You keep `result.ground_truth` (every recipe, multiplier, and true parameter). The [`meridian_grader`](../meridian_grader) package closes the loop by scoring fitted models against it automatically.

## Documentation

| Document | Contents |
|---|---|
| [Getting Started](docs/getting-started.md) | Install, first simulation, fitting Meridian on the output, verifying recovery |
| [Configuration Reference](docs/configuration-reference.md) | Every dataclass and field, with defaults and guidance |
| [Advanced Features](docs/advanced-features.md) | Traps, promo modes (`baseline_only`, negative shocks), flighting, noise — with the math |
| [Ground Truth Reference](docs/ground-truth-reference.md) | Every key in `result.ground_truth` |
| [examples/](examples/) | Runnable scripts: quickstart, competition dataset, traps & realism study |
| [CHANGELOG](CHANGELOG.md) | Version history and compatibility notes |

## Backward compatibility

All realism features are opt-in with inert defaults: a v1-era `SimulationConfig` produces the same outputs on v2 (plus additional ground-truth keys). See the [CHANGELOG](CHANGELOG.md) for the precise compatibility contract.

## Testing

```bash
pytest tests/          # 34 tests: shapes, ground truth, ROI back-solve, all realism features
```

## License & citation

Internal analytical tooling. If this simulator informs published MMM research, cite Google's Meridian alongside it — the data-generating process deliberately mirrors Meridian's model family (adstock → Hill, hierarchical geo coefficients, knot-based baselines).

## API Reference

- [**Walkthrough notebook**](Meridian_Simulator_Walkthrough.ipynb) — guided tour of every feature, simulate → fit → validate in ~2 min
- [API Reference](docs/api-reference.md) — every class, field, and function, Meridian-style
