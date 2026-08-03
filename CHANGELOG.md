# Changelog

## 2.1.0 — 2026-07-31

### Added
- **Negative structural shocks**: `PromoEventConfig.allow_negative_lift` enables `lift_pct < 0` to model stockouts, disruptions, and demand collapses (realized lift floored at −0.95). Negative `lift_pct` without the flag raises `ValueError` at config time.
- **Demand-synchronized flighting**: `seasonal_flighting ∈ [0, 1]` on `MediaChannelConfig` and `RFChannelConfig` modulates weekly media activity with the baseline seasonal wave, producing realistic media–seasonality confounding.
- `promo_events` ground-truth specs now record `allow_negative_lift`.

### Compatibility
- Fully backward compatible: both features default off (`allow_negative_lift=False`, `seasonal_flighting=0.0`) and reproduce 2.0.0 behavior exactly, including random draws.

## 2.0.0 — 2026-07-31

### Added
- **Native dataset traps** — previously post-processing, now first-class config:
  - `CollinearVariableConfig`: distractor variables linearly derived from population or a context variable; zero causal effect; recipe + realized correlation recorded in ground truth.
  - `EndogenousVariableConfig`: distractors driven by lagged channel spend or the KPI itself (reverse causality); zero causal effect.
  - `PromoEventConfig`: multiplicative KPI lifts on configured weeks, per-geo jitter, optional hiding of the binary flags from output data (`include_flag_in_output=False`), and a `baseline_only` mode that lifts only the non-media-driven KPI so recorded ROIs stay exactly true.
- **KPI observation noise**: `SimulationConfig.kpi_noise_pct` — multiplicative noise on the final KPI expressed as a coefficient of variation; realized multipliers and CV recorded.
- New ground-truth keys: `baseline_kpi_gt`, `promo_events`, `promo_multiplier_gt`, `promo_baseline_multiplier_gt`, `promo_flags_tc`, `kpi_noise_pct`, `kpi_noise_multiplier_gt`, `kpi_noise_realized_cv`, `collinear_variables`, `endogenous_variables`.
- New `SimulationResult` fields: `collinear_variable_names`, `endogenous_variable_names`, `promo_event_names` (default to empty lists).
- Config validation with actionable messages: unknown collinear sources, unknown endogenous drivers, out-of-range promo weeks.

### Fixed
- `media.py`: `tf.tensor_scatter_nd_update` was called with an empty index list when no channel pinned `alpha`, which current TensorFlow rejects (`InvalidArgumentError`). Now guarded.

### Compatibility
- Fully backward compatible with 1.x: all new config fields default to empty/off; a 1.x-style `SimulationConfig` produces the same `geo_df` / `national_df` / ground-truth values as before, with only *additional* ground-truth keys. New `SimulationResult` fields use `default_factory`, so positional construction and all 1.x attribute access are unaffected.

## 1.0.0

Initial release: paid impression and R&F channels with `target_roi` back-solving, organic media (impression and R&F), non-media channels, AR(1) context variables, knot-based baseline with trend and stacked seasonality, Meridian-conventional DataFrame/xarray outputs, ground-truth recording.
