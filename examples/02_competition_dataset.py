"""Example 2 — A competition-grade dataset with every trap, in one config.

Reproduces the structure of the Mexico MMM Challenge dataset natively:
multicollinearity traps, an endogenous GQV-style variable, hidden promo
events, a negative structural shock, demand-synchronized TV flighting, and
measurement noise on the KPI.

Run:  python examples/02_competition_dataset.py
"""

import numpy as np

from meridian_simulator import (
    BaselineConfig,
    CollinearVariableConfig,
    ContextVariableConfig,
    EndogenousVariableConfig,
    MediaChannelConfig,
    MeridianSimulator,
    OrganicMediaChannelConfig,
    PromoEventConfig,
    SeasonalityComponent,
    SimulationConfig,
)

cfg = SimulationConfig(
    n_times=104, n_geos=6, seed=2024, start_date="2024-01-06",
    media_channels=[
        MediaChannelConfig(name="google_performance", target_roi=3.0,
                           alpha=0.25, ec=0.65, slope=1.5, max_lag=4),
        MediaChannelConfig(name="youtube", target_roi=2.5,
                           alpha=0.55, ec=0.70, slope=1.2, max_lag=8),
        MediaChannelConfig(name="meta_performance", target_roi=1.5,
                           alpha=0.20, ec=0.55, slope=1.5, max_lag=4),
        MediaChannelConfig(name="tv", target_roi=0.8,
                           alpha=0.70, ec=0.55, slope=0.9, max_lag=12,
                           seasonal_flighting=0.6),   # TV buys into high season
        MediaChannelConfig(name="ooh", target_roi=1.2,
                           alpha=0.60, ec=0.58, slope=1.0, max_lag=10),
    ],
    organic_media_channels=[OrganicMediaChannelConfig(name="organic_search")],
    context_variables=[
        ContextVariableConfig(name="gdp_index", ar1_coef=0.85,
                              series_mean=100.0, series_std=3.5, trend=8.0),
        ContextVariableConfig(name="competitor_spend", ar1_coef=0.6,
                              series_mean=5.0, series_std=1.5, gamma_c_mean=-1.8),
    ],
    collinear_variables=[
        CollinearVariableConfig(name="store_count", source="population",
                                coefficient=0.03, noise_std=5.0, round_decimals=0),
        CollinearVariableConfig(name="consumer_confidence", source="gdp_index",
                                coefficient=0.6, intercept=40.0, noise_std=3.0,
                                round_decimals=1),
        CollinearVariableConfig(name="category_spend_index",
                                source="competitor_spend",
                                coefficient=1.2, intercept=1.5, noise_std=0.6,
                                round_decimals=2),
    ],
    endogenous_variables=[
        EndogenousVariableConfig(name="gqv_index", driver="youtube",
                                 lag=1, weight=0.65, base=50.0, scale=10.0),
    ],
    promo_events=[
        # Hidden positive events — contestants must engineer the flags
        PromoEventConfig(name="hot_sale_2024", weeks=[19, 20], lift_pct=0.30,
                         lift_geo_std=0.05, include_flag_in_output=False),
        PromoEventConfig(name="buen_fin_2024", weeks=[45], lift_pct=0.45,
                         lift_geo_std=0.05, include_flag_in_output=False),
        # A negative structural shock (supply disruption)
        PromoEventConfig(name="supply_disruption", weeks=[60], lift_pct=-0.30,
                         allow_negative_lift=True, include_flag_in_output=False),
    ],
    kpi_noise_pct=0.03,
    baseline=BaselineConfig(
        tau_mean=18.0, tau_std=1.5, n_knots=26, trend_slope=4.0,
        seasonality=[
            SeasonalityComponent(amplitude=3.5, period_weeks=52.0),
            SeasonalityComponent(amplitude=1.2, period_weeks=26.0, phase_weeks=4.0),
        ],
        unit_value_low=150.0, unit_value_high=250.0,
        population_low=300, population_high=600,
    ),
)

result = MeridianSimulator(cfg).run()
result.summary()

df = result.geo_df
gt = result.ground_truth

print("\n--- What contestants see ---")
print(f"Columns: {sorted(df.columns)}")
leaked = [c for c in df.columns if "hot_sale" in c or "buen_fin" in c
          or "disruption" in c]
print(f"Promo flag columns: {leaked if leaked else 'NONE (hidden)'}")

print("\n--- What the instructor keeps ---")
for s in gt["collinear_variables"]:
    print(f"  trap {s['name']:22s} ~ {s['source']:18s} r={s['realized_correlation']:.3f}")
for s in gt["endogenous_variables"]:
    print(f"  trap {s['name']:22s} ~ lag{s['lag']}({s['driver']} spend)   "
          f"r={s['realized_correlation_with_lagged_driver']:.3f}")
for s in gt["promo_events"]:
    kind = "SHOCK" if s["lift_pct"] < 0 else "promo"
    print(f"  {kind} {s['name']:20s} weeks={s['weeks']} lift={s['lift_pct']:+.0%}")
print(f"  KPI noise realized CV: {gt['kpi_noise_realized_cv']:.4f}")

# TV flighting: national TV volume should track the seasonal wave
tv_weekly = df.groupby("time")["tv_impression"].sum().reindex(result.time_names)
corr = np.corrcoef(tv_weekly.values, gt["seasonality_t"])[0, 1]
print(f"  corr(national TV volume, baseline seasonality) = {corr:.3f} "
      f"(media-seasonality confounding)")

result.save("competition_output/")
