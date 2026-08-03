"""Example 1 — Quickstart: minimal simulation and ground-truth inspection.

Run:  python examples/01_quickstart.py
"""

import numpy as np

from meridian_simulator import (
    BaselineConfig,
    ContextVariableConfig,
    MediaChannelConfig,
    MeridianSimulator,
    SeasonalityComponent,
    SimulationConfig,
)

cfg = SimulationConfig(
    n_times=104,
    n_geos=6,
    seed=42,
    media_channels=[
        MediaChannelConfig(name="search", target_roi=3.0),
        MediaChannelConfig(name="tv", target_roi=0.9, alpha=0.7, max_lag=12),
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

print("\nOutput DataFrame:")
print(result.geo_df.head())

print("\nGround-truth ROIs (the answers a good MMM should recover):")
for name, roi in zip(result.channel_names, result.ground_truth["roi_m"]):
    print(f"  {name}: {roi:.2f}x")

print("\nGround-truth adstock decays:")
for name, a in zip(result.channel_names, result.ground_truth["alpha_m"]):
    print(f"  {name}: alpha = {a:.3f}")

# Reproducibility: same config, same seed -> identical data
result2 = MeridianSimulator(cfg).run()
assert np.allclose(result.kpi_gt, result2.kpi_gt)
print("\nReproducibility verified: identical config -> identical KPI.")
