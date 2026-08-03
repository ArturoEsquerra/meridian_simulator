# Configuration Reference

Every simulation is fully described by one `SimulationConfig`. This page documents every dataclass and field. All fields have defaults; you only specify what you want to control.

## SimulationConfig

Top-level container.

| Field | Type / default | Meaning |
|---|---|---|
| `n_times` | `int = 156` | Number of weekly periods (≥ 2) |
| `n_geos` | `int = 20` | Number of geos; `1` triggers national-model mode |
| `start_date` | `str = "2021-01-25"` | First period, `YYYY-MM-DD` |
| `seed` | `int = 1320` | Seeds TF and NumPy; identical configs → identical outputs |
| `media_channels` | `list[MediaChannelConfig]` | Paid impression-based channels |
| `rf_channels` | `list[RFChannelConfig]` | Paid reach & frequency channels |
| `organic_media_channels` | `list[OrganicMediaChannelConfig]` | Unpaid impression channels |
| `organic_rf_channels` | `list[OrganicRFChannelConfig]` | Unpaid R&F channels |
| `non_media_channels` | `list[NonMediaChannelConfig]` | Non-media drivers (price, distribution) |
| `context_variables` | `list[ContextVariableConfig]` | True confounders / controls |
| `collinear_variables` | `list[CollinearVariableConfig]` | Distractors collinear with real variables (zero effect) |
| `endogenous_variables` | `list[EndogenousVariableConfig]` | Distractors driven by media or KPI (zero effect) |
| `promo_events` | `list[PromoEventConfig]` | Structural KPI lifts/shocks |
| `kpi_noise_pct` | `float = 0.0` | CV of multiplicative KPI observation noise |
| `baseline` | `BaselineConfig` | Intercepts, trend, seasonality, additive noise |

Validation at construction: geo/time bounds, promo weeks in range, negative lifts require `allow_negative_lift`, collinear `source` must name `"population"` or a configured context variable, endogenous `driver` must name `"kpi"` or a configured paid channel, `seasonal_flighting` ∈ [0, 1].

Convenience properties: `is_national`, `n_media_channels`, `n_rf_channels`, `n_organic_media_channels`, `n_organic_rf_channels`, `n_non_media_channels`, `n_context_variables`, `n_collinear_variables`, `n_endogenous_variables`, `n_promo_events`, `n_total_paid_channels`.

---

## MediaChannelConfig

One paid impression-based channel. The KPI contribution follows Meridian's structure: population-scaled impressions → Adstock(α, max_lag) → Hill(ec, slope) → × hierarchical geo coefficient.

| Field | Default | Meaning |
|---|---|---|
| `name` | `"channel"` | Column prefix in outputs (`{name}_impression`, `{name}_spend`) |
| `target_roi` | `None` | If set, the effect size is **back-solved** so true ROI ≈ this value. The recommended way to control channel strength. |
| `beta_m_mean` / `beta_m_std` | `0.9` / `0.1` | Hierarchical log-normal effect prior; `beta_m_mean` used only when `target_roi is None`; `beta_m_std` controls geo heterogeneity either way |
| `alpha` | `None` | Adstock decay ∈ [0,1]. `None` → sampled from Meridian's prior. Fix it for controlled recovery experiments. Upper-funnel ≈ 0.5–0.8, lower-funnel ≈ 0.1–0.3. |
| `ec` / `slope` | `None` / `None` | Hill half-saturation and slope; `None` → sampled |
| `cpm_low` / `cpm_high` | `0.011` / `0.012` | Uniform CPM range; with impression scale, determines spend share |
| `max_lag` | `8` | Adstock carryover window (periods) |
| `impression_mean_channel` | `1.0` | Channel-level mean of impressions-per-capita (log-ish scale of activity volume) |
| `impression_mean_time` | `0.8` | Weekly activity mean |
| `impression_std` | `0.5` | Idiosyncratic weekly/geo noise in activity |
| `seasonal_flighting` | `0.0` | ∈ [0,1]. Modulates weekly activity with the baseline seasonal wave — media buys into high season. See [Advanced Features](advanced-features.md#demand-synchronized-flighting). |

**Sizing a channel:** expected weekly impressions/capita ≈ `impression_mean_channel + impression_mean_time`; spend ≈ impressions × CPM. Use small `impression_mean_channel` plus wide CPM for "small channel" rubric scenarios.

## RFChannelConfig

Same fields as `MediaChannelConfig` (with `beta_rf_mean`/`beta_rf_std`), but the channel is delivered as **reach and frequency**: impressions are converted through a Poisson arrival model, Hill applies to frequency, adstock to reach × adjusted frequency — mirroring Meridian's RF treatment. Also supports `seasonal_flighting`.

## OrganicMediaChannelConfig / OrganicRFChannelConfig

Unpaid channels: same adstock/Hill machinery, no spend and no CPM. Fields: `beta_om_mean`/`beta_om_std` (or `beta_orf_*`), `alpha`, `ec`, `slope`, `max_lag`, `impression_mean_channel`, `impression_mean_time`, `impression_std`.

## NonMediaChannelConfig

Geo-time series that causally affect the KPI without being media (price index, distribution, promo intensity). Standardized internally; enters the KPI with hierarchical coefficient `gamma_n ± xi_n` deviation per geo.

| Field | Default |
|---|---|
| `gamma_n_mean` / `gamma_n_std` | `1.0` / `0.5` |
| `xi_n_std` | `0.3` |
| `series_mean` / `series_std` | `0.0` / `1.0` |

## ContextVariableConfig

True confounders the analyst *should* include as controls.

| Field | Default | Meaning |
|---|---|---|
| `gamma_c_mean` / `gamma_c_std` | `3.5` / `0.5` | Coefficient prior (KPI-per-capita units per SD) |
| `xi_c_std` | `0.3` | Geo-level coefficient deviation |
| `series_mean` / `series_std` | `0.0` / `3.0` | Level and scale of the raw series |
| `ar1_coef` | `0.0` | AR(1) persistence; 0.8–0.9 for macro-style series |
| `trend` | `0.0` | Linear drift over the horizon |

Output column: `{name}_control`.

## CollinearVariableConfig

Distractor: `value = coefficient × source + intercept + N(0, noise_std)`; **zero causal effect**.

| Field | Default | Meaning |
|---|---|---|
| `source` | `"population"` | `"population"` or a context variable's `name` |
| `coefficient` / `intercept` | `1.0` / `0.0` | Linear map |
| `noise_std` | `1.0` | Lower → tighter collinearity (higher VIF) |
| `round_decimals` | `None` | `0` for integer-like variables (store counts) |

Ground truth records the recipe and the **realized correlation**.

## EndogenousVariableConfig

Distractor: `value = base + scale × (weight × standardize(lag(driver)) + (1−weight) × N(0,1))`; **zero causal effect**. Modeling it as a control soaks up media credit — the endogeneity/mediator trap.

| Field | Default | Meaning |
|---|---|---|
| `driver` | `"kpi"` | A paid channel name (drives via its **spend**) or `"kpi"` for reverse causality |
| `lag` | `1` | Periods of lag on the driver |
| `weight` | `0.65` | Variance share explained by the driver, ∈ [0,1] |
| `base` / `scale` | `50.0` / `10.0` | Output level and spread (index-style defaults) |

## PromoEventConfig

Multiplicative structural lift (or shock) on specific weeks. See [Advanced Features](advanced-features.md#promotional-events--structural-shocks) for the exact math of the modes.

| Field | Default | Meaning |
|---|---|---|
| `weeks` | `[]` | 0-based period indices the event covers |
| `lift_pct` | `0.25` | Fractional lift; negative allowed with `allow_negative_lift` |
| `lift_geo_std` | `0.0` | Per-geo lift jitter (heterogeneous response) |
| `include_flag_in_output` | `True` | `False` hides the binary flag from output data (contestants must engineer it) |
| `baseline_only` | `False` | `True`: lift scales only non-media-driven KPI → recorded ROIs stay exactly true. `False`: lift scales total KPI → media works harder during promos. |
| `allow_negative_lift` | `False` | Enables `lift_pct < 0` (stockouts, disruptions); realized lift floored at −0.95 |

## BaselineConfig

The non-media demand structure: `baseline_gt = tau_g + mu_t + trend_t + seasonality_t + eps_gt` (per-capita).

| Field | Default | Meaning |
|---|---|---|
| `tau_mean` / `tau_std` | `15.0` / `1.2` | Geo intercept distribution — keep `tau_mean` comfortably positive |
| `n_knots` | `None` (=`n_times`) | Knots for the time-varying `mu_t`; fewer → smoother |
| `knot_std` | `2.0` | Knot value prior std |
| `trend_slope` | `0.0` | Linear drift across the horizon (per-capita units) |
| `seasonality` | `[]` | List of `SeasonalityComponent(amplitude, period_weeks, phase_weeks)`, stacked additively |
| `noise_std` | `0.5` | Additive per-capita baseline noise (see also `SimulationConfig.kpi_noise_pct`) |
| `unit_value_low/high` | `0.0345` / `0.0355` | Revenue-per-KPI uniform range |
| `population_low/high` | `1e5` / `1e6` | Geo population uniform range |

### Two noise knobs, two purposes

- `BaselineConfig.noise_std` — *structural* noise inside the baseline (demand volatility). Additive, per-capita, part of the data-generating process the model should absorb into its baseline.
- `SimulationConfig.kpi_noise_pct` — *observation* noise on the final KPI (measurement error). Multiplicative, expressed as a CV, applied after everything else. This is the "how messy are the sales data" knob; realized CV is recorded in the ground truth.
