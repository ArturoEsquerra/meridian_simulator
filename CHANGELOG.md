# Changelog

## 2.2.2 — 2026-09-09

### Added
- **Verified Meridian 2.x compatibility matrix.** Meridian 2.0.0 was installed and tested directly (previously the dependency would not resolve locally, so 2.x behaviour was inferred from source):

  | Meridian | Backend | Status |
  |---|---|---|
  | 1.x | TensorFlow (only backend) | Works |
  | 2.x | TensorFlow (`MERIDIAN_BACKEND=tensorflow`) | **Works** — output bit-identical to 1.x |
  | 2.x | JAX (2.x default) | **Not supported** — Meridian's JAX ops cannot consume TensorFlow tensors |

- `check_meridian_compat()` now **raises a `RuntimeError` with the exact remedy** when it detects Meridian 2.x running on the JAX backend, instead of letting the run fail later with `TypeError: Error interpreting argument ... as an abstract array ... EagerTensor`. Backend detection prefers Meridian's live `get_backend()`, falling back to `MERIDIAN_BACKEND` and then the declared default; it reports `"unknown"` rather than guessing.
- `docs/meridian-2-compatibility.md` — step-by-step for installing and validating Meridian 2.x alongside the simulator.

### Notes
- The `google-meridian>=1.6,<2` pin stays: it guarantees a working environment from a plain `pip install`. Meridian 2.x is supported on an opt-in basis by overriding the pin and selecting the TensorFlow backend (see the doc above).
- Confirmed on a real 2.0.0 install: default backend is JAX, `np_float_dtype` is `float64`, and `knots.get_knot_info().weights` is float64 — the precise origin of the 2.2.1 einsum failure.

## 2.2.1 — 2026-09-09

### Fixed
- **`InvalidArgumentError: cannot compute Einsum ... expected to be a float tensor but is a double tensor`.**
  Google released **Meridian 2.0.0 on 2026-09-03**, which changed the default compute backend from TensorFlow to **JAX** (`backend/config.py: _DEFAULT_BACKEND = Backend.JAX`). Under the JAX backend Meridian's float width is `np.float64 if jax.config.jax_enable_x64 else np.float32`, so arrays it builds with `dtype=backend.np_float_dtype` — notably `knots.get_knot_info().weights` — can come back as **float64**. The simulator fed those straight into `tf.einsum` against its own float32 tensors, producing the error above (input #1 is the knot weights). A fresh `pip install google-meridian` in Colab now resolves to 2.0.0, which is why this appeared without any change to the simulator.

### Added
- `utils.as_float()` — the single conversion point for values originating inside Meridian. Normalises numpy (any precision), TF tensors, JAX arrays, and Python sequences to the simulator's float32. Applied at every Meridian boundary: knot weights, prior samples (`alpha/ec/slope` for media, R&F, and organic), `MediaTransformer` / `CenteringAndScalingTransformer` outputs, and `HillTransformer` / `AdstockTransformer` outputs.
- `utils.check_meridian_compat()` — reports the installed Meridian version and backend, and warns on majors the simulator has not been validated against. Called automatically at the start of `MeridianSimulator.run()`.

### Changed
- **Dependency pinned to `google-meridian>=1.6,<2`.** The simulator is TensorFlow-based and is validated against Meridian 1.x only; 2.0's JAX backend is not yet validated end to end. Pinning stops fresh installs from silently adopting an untested major version.

### Compatibility
- Under Meridian 1.x the casts are identity operations: output is **bit-identical** to 2.2.0 (verified — seed 42 reproduces `search=2.947`, `tv=0.805` exactly). Existing datasets and recorded ground truth still reproduce.
- If you need Meridian 2.x, set `MERIDIAN_BACKEND=tensorflow` **before** importing `meridian` to select its TensorFlow backend, which fixes Meridian's float width at float32. This path is untested here — verify before relying on it.

## 2.2.0 — 2026-08-05

### Added
- **Incrementality experiments** (`ExperimentConfig` + `experiments.py`): simulate randomized lift studies against the ground truth. Each experiment measures a paid channel's TRUE ROI over its window (full duration or `start_week`/`end_week`), then reports a noisy point estimate + standard error controlled by `se_pct`, with optional systematic `bias_pct` (e.g. short-horizon studies missing adstocked effects — a calibration trap).
- **Meridian-ready calibration priors**: experiment results are moment-matched to LogNormal `roi_m` prior parameters (`lognormal_from_point_and_se`), and `ground_truth["experiment_calibration"]` packages per-channel `roi_mu`/`roi_sigma` (defaults preserved for uncalibrated channels), a `calibrated` flag vector, and — for windowed experiments — the `ModelSpec(roi_calibration_period=...)` boolean mask. Multiple experiments per channel resolve to the most precise.
- Config validation: unknown channels, half-specified windows, out-of-range weeks, and non-positive `se_pct` fail at construction.

### Compatibility
- Fully backward compatible: `experiments` defaults to empty; without experiments, no random draws are consumed and ground truth gains only `experiments: []` / `experiment_calibration: None`.

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
