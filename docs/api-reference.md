# meridian_simulator — API Reference

*Version 2.2.2. Generated from the source by
`docs/generate_api_reference.py` — rerun it after any code change.
Style follows the [Meridian API reference](https://developers.google.com/meridian/reference/api/meridian):
every public class, field, and function, with parameters and defaults.*

## Package overview

| Module | Purpose |
|---|---|
| `meridian_simulator.config` | Declarative configuration dataclasses (the public API surface) |
| `meridian_simulator.simulator` | `MeridianSimulator` orchestrator and `SimulationResult` container |
| `meridian_simulator.experiments` | Simulated incrementality experiments and calibration priors |
| `meridian_simulator.augment` | Distractor variables, promotional events, KPI observation noise |
| `meridian_simulator.baseline` | Baseline components: intercepts, trend, seasonality, noise |
| `meridian_simulator.media` | Paid media: impressions, R&F, spend, Hill-Adstock, ROI back-solve |
| `meridian_simulator.organic` | Organic (unpaid) media channels |
| `meridian_simulator.context` | Context (control) variables and non-media channels |
| `meridian_simulator.output` | DataFrame / xarray output assembly |

All public names import from the package root:
`from meridian_simulator import SimulationConfig, MeridianSimulator, ...`

---

## meridian_simulator.config

### `SimulationConfig`

Top-level configuration for the Meridian dataset simulator.

| Field | Type | Default | Description |
|---|---|---|---|
| `n_times` | `int` | `156` | Number of time periods (weeks). |
| `n_geos` | `int` | `20` | Number of geographic units.  Set to 1 for a national model. |
| `start_date` | `str` | `'2021-01-25'` | First time period as a `YYYY-MM-DD` string. |
| `seed` | `int` | `1320` | Random seed for reproducibility. |
| `media_channels` | `list[MediaChannelConfig]` | `[]` | List of impression-based paid channel configs. |
| `rf_channels` | `list[RFChannelConfig]` | `[]` | List of reach-and-frequency channel configs. |
| `organic_media_channels` | `list[OrganicMediaChannelConfig]` | `[]` | List of organic impression channel configs. |
| `organic_rf_channels` | `list[OrganicRFChannelConfig]` | `[]` | List of organic RF channel configs. |
| `non_media_channels` | `list[NonMediaChannelConfig]` | `[]` | List of non-media channel configs. |
| `context_variables` | `list[ContextVariableConfig]` | `[]` | List of context/control variable configs. |
| `collinear_variables` | `list[CollinearVariableConfig]` | `[]` | List of collinear distractor variable configs. These appear in the output data but have no causal effect on KPI. |
| `endogenous_variables` | `list[EndogenousVariableConfig]` | `[]` | List of endogenous distractor variable configs. Driven by lagged media activity or the KPI itself; no causal effect on KPI. |
| `promo_events` | `list[PromoEventConfig]` | `[]` | List of promotional event configs.  Each applies a multiplicative structural lift to the KPI on its event weeks. |
| `experiments` | `list[ExperimentConfig]` | `[]` | List of incrementality experiment configs.  Each simulates a lift study measuring a paid channel's true window ROI, reported with configurable noise/bias and converted to Meridian-ready calibration priors in the ground truth. |
| `kpi_noise_pct` | `float` | `0.0` | Coefficient of variation of multiplicative observation noise applied to the FINAL KPI: `kpi *= (1 + N(0, kpi_noise_pct))`. 0.0 disables it.  This is the main knob for how noisy sales data are relative to signal; `baseline.noise_std` remains available as additive per-capita noise inside the baseline.  Typical values: 0.01 (very clean) to 0.10 (noisy real-world data). |
| `baseline` | `BaselineConfig` | `BaselineConfig(tau_mean=15.0, tau_std=1.2, n_knots=None, knot_std=2.0, trend_slope=0.0, seasonality=[], noise_std=0.5, unit_value_low=0.0345, unit_value_high=0.0355, population_low=100000.0, population_high=1000000.0)` | Baseline (intercept + trend + seasonality) config. |

### `MediaChannelConfig`

Configuration for a single paid impression-based media channel.

| Field | Type | Default | Description |
|---|---|---|---|
| `name` | `str` | `'channel'` | Human-readable channel label (e.g. `"paid_search"`). |
| `target_roi` | `Optional[float]` | `None` | Desired ground-truth ROI (incremental revenue / spend). When set, `beta_m` is back-solved so that E[ROI] ≈ target_roi. Mutually exclusive with `beta_m_mean`. |
| `beta_m_mean` | `float` | `0.9` | Mean of the hierarchical log-normal prior for geo-level media coefficients.  Only used when `target_roi` is None. |
| `beta_m_std` | `float` | `0.1` | Std of the hierarchical distribution (eta_m). |
| `alpha` | `Optional[float]` | `None` | Adstock decay parameter in [0, 1].  If None, sampled from Meridian's default prior. |
| `ec` | `Optional[float]` | `None` | Hill saturation half-saturation point (ec > 0).  If None, sampled. |
| `slope` | `Optional[float]` | `None` | Hill slope parameter (> 0).  If None, sampled. |
| `cpm_low` | `float` | `0.011` | Lower bound for cost-per-thousand impressions uniform draw. |
| `cpm_high` | `float` | `0.012` | Upper bound for cost-per-thousand impressions uniform draw. |
| `max_lag` | `int` | `8` | Maximum lag for adstock carry-over effect (in time periods). |
| `impression_mean_channel` | `float` | `1.0` | Mean channel effect used to draw impressions. |
| `impression_mean_time` | `float` | `0.8` | Mean time effect used to draw impressions. |
| `impression_std` | `float` | `0.5` | Std of the geo-time idiosyncratic impression noise. |
| `seasonal_flighting` | `float` | `0.0` | Strength in [0, 1] of demand-synchronized media flighting.  0.0 (default) draws media activity independently of the baseline seasonality — the historical behavior.  Positive values modulate the channel's weekly activity with the baseline seasonal wave (media planners buying INTO high season), creating realistic media–seasonality confounding that tests whether an analyst's knot/control choices can separate media from demand. |

### `RFChannelConfig`

Configuration for a single reach-and-frequency media channel.

All adstock/Hill parameters apply to the frequency dimension.

| Field | Type | Default | Description |
|---|---|---|---|
| `name` | `str` | `'rf_channel'` | Human-readable channel label (e.g. `"youtube"`). |
| `target_roi` | `Optional[float]` | `None` | Desired ground-truth ROI.  When set, beta_rf is back-solved. |
| `beta_rf_mean` | `float` | `0.9` | Hierarchical mean for log-normal beta prior. |
| `beta_rf_std` | `float` | `0.1` | Hierarchical std (eta_rf). |
| `alpha` | `Optional[float]` | `None` | Adstock decay in [0, 1].  If None, sampled. |
| `ec` | `Optional[float]` | `None` | Half-saturation.  If None, sampled. |
| `slope` | `Optional[float]` | `None` | Hill slope.  If None, sampled. |
| `cpm_low` | `float` | `0.011` | Lower bound for CPM uniform draw. |
| `cpm_high` | `float` | `0.012` | Upper bound for CPM uniform draw. |
| `max_lag` | `int` | `8` | Maximum adstock lag. |
| `impression_mean_channel` | `float` | `1.0` | Channel-level impression mean. |
| `impression_mean_time` | `float` | `0.8` | Time-level impression mean. |
| `impression_std` | `float` | `0.5` | Geo-time idiosyncratic impression std. |
| `seasonal_flighting` | `float` | `0.0` |  |

### `OrganicMediaChannelConfig`

Configuration for an organic (unpaid) impression-based media channel.

Organic channels affect KPI but have no associated spend/CPM.

| Field | Type | Default | Description |
|---|---|---|---|
| `name` | `str` | `'organic_channel'` | Channel label. |
| `beta_om_mean` | `float` | `0.5` | Hierarchical mean of organic media coefficient. |
| `beta_om_std` | `float` | `0.1` | Hierarchical std (eta_om). |
| `alpha` | `Optional[float]` | `None` | Adstock decay.  If None, sampled from prior. |
| `ec` | `Optional[float]` | `None` | Half-saturation.  If None, sampled. |
| `slope` | `Optional[float]` | `None` | Hill slope.  If None, sampled. |
| `max_lag` | `int` | `8` | Maximum adstock lag. |
| `impression_mean_channel` | `float` | `0.8` | Channel impression mean. |
| `impression_mean_time` | `float` | `0.6` | Time impression mean. |
| `impression_std` | `float` | `0.4` | Geo-time noise std. |

### `OrganicRFChannelConfig`

Configuration for an organic reach-and-frequency channel.

| Field | Type | Default | Description |
|---|---|---|---|
| `name` | `str` | `'organic_rf_channel'` |  |
| `beta_orf_mean` | `float` | `0.5` |  |
| `beta_orf_std` | `float` | `0.1` |  |
| `alpha` | `Optional[float]` | `None` |  |
| `ec` | `Optional[float]` | `None` |  |
| `slope` | `Optional[float]` | `None` |  |
| `max_lag` | `int` | `8` |  |
| `impression_mean_channel` | `float` | `0.8` |  |
| `impression_mean_time` | `float` | `0.6` |  |
| `impression_std` | `float` | `0.4` |  |

### `NonMediaChannelConfig`

Configuration for a non-media time-series channel.

These are exogenous variables that affect KPI but are not media
(e.g. price index, distribution score, promotion flags).

| Field | Type | Default | Description |
|---|---|---|---|
| `name` | `str` | `'non_media_channel'` | Channel label. |
| `gamma_n_mean` | `float` | `1.0` | Mean of the normal prior on the coefficient. |
| `gamma_n_std` | `float` | `0.5` | Std of the normal prior on the coefficient. |
| `xi_n_std` | `float` | `0.3` | Std of the geo-level deviation (HalfNormal). |
| `series_mean` | `float` | `0.0` | Mean of the raw time series. |
| `series_std` | `float` | `1.0` | Std of the raw time series. |

### `ContextVariableConfig`

Configuration for a context/control variable.

Context variables (e.g. GDP index, exchange rate, competitor spend) are
geo-time varying variables that control for confounders.

| Field | Type | Default | Description |
|---|---|---|---|
| `name` | `str` | `'context_var'` | Variable label (e.g. `"gdp_index"`). |
| `gamma_c_mean` | `float` | `3.5` | Mean of normal prior on the coefficient. |
| `gamma_c_std` | `float` | `0.5` | Std of normal prior on the coefficient. |
| `xi_c_std` | `float` | `0.3` | Std of geo-level deviation (HalfNormal). |
| `series_mean` | `float` | `0.0` | Mean of the simulated raw series. |
| `series_std` | `float` | `3.0` | Std of the simulated raw series. |
| `ar1_coef` | `float` | `0.0` | Auto-regressive AR(1) coefficient for temporal correlation. Set to 0.0 for iid draws, up to ~0.9 for persistent series. |
| `trend` | `float` | `0.0` | Optional linear drift added to the series over time. |

### `CollinearVariableConfig`

Configuration for a collinear distractor variable.

Collinear variables are derived from an existing series (population or a
context variable) via a linear map plus noise::

    value = coefficient * source + intercept + N(0, noise_std)

They appear in the output dataset but have ZERO causal effect on the KPI —
their purpose is to test whether an analyst detects and handles
multicollinearity (e.g. via VIF) instead of throwing every column into the
model.

| Field | Type | Default | Description |
|---|---|---|---|
| `name` | `str` | `'collinear_var'` | Output column label (e.g. `"store_count"`). |
| `source` | `str` | `'population'` | What the variable is collinear with.  Either the literal string `"population"` or the `name` of a configured `ContextVariableConfig`. |
| `coefficient` | `float` | `1.0` | Linear multiplier applied to the source series. |
| `intercept` | `float` | `0.0` | Constant offset added after the multiplication. |
| `noise_std` | `float` | `1.0` | Std of the additive Gaussian noise.  Relative to the scale of `coefficient * source`; small values give near-perfect collinearity (VIF → ∞), larger values weaken it. |
| `round_decimals` | `Optional[int]` | `None` | If not None, round the final series to this many decimals (use 0 for integer-like variables such as store counts). |

### `EndogenousVariableConfig`

Configuration for an endogenous distractor variable.

Endogenous variables are *symptoms* of the system, not causes: they are
driven by a lagged media channel's activity (e.g. Google query volume
rising after YouTube bursts) or by the KPI itself (reverse causality).
Including them as controls in an MMM soaks up media credit and biases ROI
downward — the classic mediator/endogeneity trap.

The series is built as::

    z = weight * normalize(lagged driver) + (1 - weight) * N(0, 1)
    value = base + scale * z

They have ZERO direct causal effect on the KPI.

| Field | Type | Default | Description |
|---|---|---|---|
| `name` | `str` | `'endogenous_var'` | Output column label (e.g. `"gqv_index"`). |
| `driver` | `str` | `'kpi'` | Name of the paid media / R&F channel whose SPEND drives the series, or the literal string `"kpi"` for reverse causality. |
| `lag` | `int` | `1` | Number of time periods the driver is lagged by (>= 0). |
| `weight` | `float` | `0.65` | Share of variance explained by the driver, in [0, 1]. 0.65 gives a clearly detectable but not degenerate correlation. |
| `base` | `float` | `50.0` | Additive level of the output series. |
| `scale` | `float` | `10.0` | Multiplier applied to the standardized mixed series. |

### `PromoEventConfig`

Configuration for a promotional event that structurally lifts the KPI.

Promo events model commercial actions (deep discounts, special payment
terms, in-store events) that lift sales independently of media — the
textbook use case for Meridian's ``non_media_treatments``.  The lift is
applied MULTIPLICATIVELY to the final KPI on the configured weeks::

    kpi[:, week] *= (1 + lift_pct + N(0, lift_geo_std))   per geo

| Field | Type | Default | Description |
|---|---|---|---|
| `name` | `str` | `'promo_event'` | Event label (e.g. `"hot_sale_2024"`).  When `include_flag_in_output` is True, a binary `{name}` column is added to the output DataFrames. |
| `weeks` | `list[int]` | `[]` | Time-period indices (0-based) on which the event is active. |
| `lift_pct` | `float` | `0.25` | Fractional KPI lift on event weeks (0.25 = +25%). |
| `lift_geo_std` | `float` | `0.0` | Std of geo-level jitter added to the lift, so regions respond heterogeneously.  0.0 applies a uniform lift. |
| `include_flag_in_output` | `bool` | `True` | Whether to include the binary event flag as a column in the output DataFrames.  Set False to force analysts to engineer the flags from domain knowledge. |
| `baseline_only` | `bool` | `False` | Controls what the multiplicative lift applies to. False (default): the lift scales the TOTAL KPI on event weeks — media contributions are lifted too, so true channel ROI on those weeks runs slightly above the recorded `roi_m` (media works harder during promos; realistic synergy). |
| `allow_negative_lift` | `bool` | `False` | Permit `lift_pct < 0` to model NEGATIVE structural shocks — stockouts, supply disruptions, store closures, demand collapses.  When True, per-geo realized lifts are floored at -0.95 (KPI can drop up to 95% but never below 0). When False (default, historical behavior) negative realized lifts are clamped to 0. |

### `ExperimentConfig`

Configuration for a simulated incrementality experiment.

Simulates a randomized lift study (geo holdout, conversion lift) run on a
paid channel. The experiment measures the channel's TRUE ROI over its
window from the recorded ground-truth contributions, then reports a noisy
— and optionally biased — point estimate with a standard error, exactly
the two numbers a real experiment vendor delivers. Results are converted
to Meridian-ready LogNormal ``roi_m`` prior parameters by moment matching,
and windowed experiments additionally produce the
``ModelSpec(roi_calibration_period=...)`` mask.

| Field | Type | Default | Description |
|---|---|---|---|
| `name` | `str` | `'experiment'` | Experiment label (e.g. `"youtube_geo_holdout_q3"`). |
| `channel` | `str` | `''` | Name of the paid channel under test. Must match a configured media or R&F channel. |
| `start_week` | `Optional[int]` | `None` | First week (0-based) of the experiment window, or None for a full-duration experiment. |
| `end_week` | `Optional[int]` | `None` | Last week (inclusive) of the window, or None for full-duration. `start_week`/`end_week` must be set together. |
| `se_pct` | `float` | `0.15` | Relative precision of the study — the standard error as a fraction of the true ROI. 0.10 is a tight, well-powered geo experiment; 0.30 is a small or noisy study. |
| `bias_pct` | `float` | `0.0` | Systematic bias of the study as a fraction of true ROI. 0.0 = unbiased. Negative values model the common real-world case of short-horizon experiments missing long-term/adstocked effects (e.g. -0.20 = the study captures only 80% of true incrementality). A deliberately biased experiment is a calibration TRAP: analysts who calibrate hard on it pull the posterior toward the wrong ROI. |

### `BaselineConfig`

Configuration for the baseline (intercept + trend + seasonality) component.

| Field | Type | Default | Description |
|---|---|---|---|
| `tau_mean` | `float` | `15.0` | Mean of the geo-level intercept N(tau_mean, tau_std). |
| `tau_std` | `float` | `1.2` | Std of the geo-level intercept. |
| `n_knots` | `Optional[int]` | `None` | Number of knots for the time-varying intercept mu_t. Set to n_times for full-flexibility, or a smaller integer for smoother trends.  `None` defaults to n_times (full knots). |
| `knot_std` | `float` | `2.0` | Std of the Normal prior on knot values. |
| `trend_slope` | `float` | `0.0` | Additive linear trend per time period (in KPI-per-capita units).  Positive values give upward trend. |
| `seasonality` | `list[SeasonalityComponent]` | `[]` | List of SeasonalityComponent instances stacked additively. |
| `noise_std` | `float` | `0.5` | Std of the iid residual error term epsilon_gt. |
| `unit_value_low` | `float` | `0.0345` | Lower bound for revenue-per-KPI unit. |
| `unit_value_high` | `float` | `0.0355` | Upper bound for revenue-per-KPI unit. |
| `population_low` | `float` | `100000.0` | Lower bound for geo population (drawn Uniform). |
| `population_high` | `float` | `1000000.0` | Upper bound for geo population (drawn Uniform). |

### `SeasonalityComponent`

A single sinusoidal seasonality component.

| Field | Type | Default | Description |
|---|---|---|---|
| `amplitude` | `float` | `2.0` | Peak-to-mean amplitude of the wave. |
| `period_weeks` | `float` | `52.0` | Period of the cycle in time periods (weeks if weekly data). |
| `phase_weeks` | `float` | `0.0` | Phase offset in time periods. |


---

## meridian_simulator.simulator

### `MeridianSimulator`

Simulate a complete Meridian-compatible dataset from a config.

Usage::

    from meridian_simulator.config import (
        SimulationConfig, MediaChannelConfig, RFChannelConfig,
        ContextVariableConfig, SeasonalityComponent, BaselineConfig,
    )
    from meridian_simulator.simulator import MeridianSimulator

    cfg = SimulationConfig(
        n_times=104,
        n_geos=10,
        seed=42,
        media_channels=[
            MediaChannelConfig(name="paid_search", target_roi=3.5),
            MediaChannelConfig(name="display",     target_roi=1.8),
        ],
        rf_channels=[
            RFChannelConfig(name="youtube", target_roi=2.2),
        ],
        context_variables=[
            ContextVariableConfig(name="gdp_index",  ar1_coef=0.8),
            ContextVariableConfig(name="competitor_spend", ar1_coef=0.5),
        ],
        baseline=BaselineConfig(
            trend_slope=5.0,
            seasonality=[SeasonalityComponent(amplitude=3.0, period_weeks=52)],
            n_knots=26,
        ),
    )
    result = MeridianSimulator(cfg).run()
    result.summary()
    result.save("output/")

#### `MeridianSimulator.run(self) -> 'SimulationResult'`

Execute the full simulation pipeline.

Returns:
    A :class:`SimulationResult` containing all outputs.

### `SimulationResult`

Holds every output produced by the simulator.

Attributes:
    config: The SimulationConfig used to produce this result.
    ground_truth: Dict of ground-truth parameter values (on raw KPI scale
        and on Meridian's transformed scale where applicable).
    geo_df: Wide geo × time Pandas DataFrame, Meridian-compatible.
    national_df: Nationally aggregated DataFrame.
    xr_dict: Dict of xr.DataArrays (one per variable type).
    kpi_gt: KPI tensor, shape (n_geos, n_times).
    unit_value_gt: Revenue-per-KPI tensor, shape (n_geos, n_times).
    population_g: Population tensor, shape (n_geos,).
    geo_names: List of geo labels.
    time_names: List of time-period labels.
    channel_names: List of paid impression channel labels.
    rf_channel_names: List of R&F channel labels.
    organic_channel_names: List of organic impression channel labels.
    organic_rf_channel_names: List of organic R&F channel labels.
    non_media_channel_names: List of non-media channel labels.
    context_variable_names: List of context variable labels.
    collinear_variable_names: List of collinear distractor variable labels.
    endogenous_variable_names: List of endogenous distractor variable labels.
    promo_event_names: List of promo event labels.

#### `SimulationResult.save(self, output_dir: 'str | Path' = '.') -> 'None'`

Save DataFrames to CSV and ground_truth dict to pickle.

Args:
    output_dir: Directory where files will be written.

#### `SimulationResult.summary(self) -> 'None'`

Print a concise summary of the simulation.

#### `SimulationResult.plot_kpi(self, figsize=(12, 4)) -> 'None'`

Plot mean KPI across geos over time.

#### `SimulationResult.plot_media_spend(self, figsize=(12, 4)) -> 'None'`

Plot total spend per channel over time (stacked bar).


---

## meridian_simulator.experiments

### `simulate_experiments(cfgs: 'list[ExperimentConfig]', contribution_gtm: 'np.ndarray', cost_gtm: 'np.ndarray', unit_value_gt: 'np.ndarray', paid_channel_names: 'list[str]', n_times: 'int', rng: 'np.random.Generator') -> 'dict'`

Simulate incrementality experiments against the ground truth.

For each experiment, the true window ROI is computed from the recorded
media contributions, then observed with configurable noise and bias::

    roi_true = sum_{g, t in W} contribution(g,t,m) * v(g,t) / spend_W
    roi_hat  = roi_true * (1 + bias_pct) + N(0, (se_pct * roi_true)^2)
    se_hat   = se_pct * roi_true

Args:
    cfgs: Experiment configs.
    contribution_gtm: Media-driven KPI contribution per geo-time-channel,
        shape (n_geos, n_times, n_paid_channels), in KPI units.
    cost_gtm: Spend, same shape.
    unit_value_gt: Revenue per KPI unit, shape (n_geos, n_times).
    paid_channel_names: Names aligned with the channel axis.
    n_times: Number of time periods.
    rng: Random generator (seeded by SimulationConfig.seed).

Returns:
    Dict with:
      `results` – list of per-experiment dicts (channel, window, true
                    window ROI, point estimate, standard error, bias).
      `calibration` – prior-building payload (see
                    :func:`build_calibration_priors`).

### `build_calibration_priors(results: 'list[dict]', paid_channel_names: 'list[str]', n_times: 'int', default_mu: 'float' = 0.2, default_sigma: 'float' = 0.9) -> 'dict'`

Assemble Meridian-ready calibration inputs from experiment results.

Channels without an experiment keep Meridian's default ROI prior
(LogNormal(0.2, 0.9)). When any experiment is date-restricted, a
`roi_calibration_period` mask is built: True on the experiment window
for calibrated channels, True everywhere for channels calibrated over the
full duration or not calibrated at all (Meridian's convention: the prior
applies to the masked period).

If a channel has multiple experiments, the most precise (smallest
standard error) wins.

Returns:
    Dict with:
      `roi_mu`     – (n_channels,) LogNormal mu per channel.
      `roi_sigma`  – (n_channels,) LogNormal sigma per channel.
      `calibrated` – (n_channels,) bool, experiment-informed or default.
      `roi_calibration_period` – (n_times, n_channels) bool mask, or
                        None when every experiment is full-duration.
      `channel_names` – channel order for all arrays.

Usage with Meridian::

    import tensorflow_probability as tfp
    from meridian.model import prior_distribution, spec

    cal = result.ground_truth["experiment_calibration"]
    prior = prior_distribution.PriorDistribution(
        roi_m=tfp.distributions.LogNormal(
            cal["roi_mu"].astype("float32"),
            cal["roi_sigma"].astype("float32"),
        )
    )
    model_spec = spec.ModelSpec(
        prior=prior,
        roi_calibration_period=cal["roi_calibration_period"],
    )

### `lognormal_from_point_and_se(point_estimate: 'float', standard_error: 'float') -> 'tuple[float, float]'`

Moment-match a LogNormal(mu, sigma) to an experiment estimate.

Chooses parameters so the LogNormal's mean equals the point estimate and
its standard deviation equals the standard error::

    sigma^2 = ln(1 + se^2 / pe^2)
    mu      = ln(pe) - sigma^2 / 2

This is the standard conversion for turning a lift study's (estimate, SE)
into a Meridian `roi_m` prior.

Args:
    point_estimate: Experiment ROI point estimate (> 0).
    standard_error: Standard error of the estimate (> 0).

Returns:
    (mu, sigma) for `tfp.distributions.LogNormal(mu, sigma)`.


---

## meridian_simulator.augment

### `simulate_collinear_variables(cfgs: 'list[CollinearVariableConfig]', context_gtc: 'np.ndarray', context_variable_names: 'list[str]', population_g: 'np.ndarray', n_times: 'int', rng: 'np.random.Generator') -> 'dict'`

Simulate collinear distractor variables.

Args:
    cfgs: Collinear variable configs.
    context_gtc: Raw context series, shape (n_geos, n_times, n_ctx).
    context_variable_names: Names aligned with the last axis of
        `context_gtc`.
    population_g: Population per geo, shape (n_geos,).
    n_times: Number of time periods.
    rng: Random generator.

Returns:
    Dict with:
      `collinear_gtc` – shape (n_geos, n_times, n_collinear)
      `spec`          – list of dicts describing each variable's recipe
                          (for the ground-truth record).

### `simulate_endogenous_variables(cfgs: 'list[EndogenousVariableConfig]', cost_gtm: 'np.ndarray', paid_channel_names: 'list[str]', kpi_gt: 'np.ndarray', rng: 'np.random.Generator') -> 'dict'`

Simulate endogenous distractor variables.

Each variable mixes a lagged, standardized driver series with independent
noise::

    z     = weight * standardize(lag(driver)) + (1 - weight) * N(0, 1)
    value = base + scale * z

Args:
    cfgs: Endogenous variable configs.
    cost_gtm: Paid media spend, shape (n_geos, n_times, n_paid_total)
        covering impression channels then R&F channels.
    paid_channel_names: Names aligned with the last axis of `cost_gtm`.
    kpi_gt: Final KPI, shape (n_geos, n_times) — used when driver='kpi'.
    rng: Random generator.

Returns:
    Dict with:
      `endogenous_gtc` – shape (n_geos, n_times, n_endog)
      `spec`           – list of recipe dicts for the ground truth.

### `apply_promo_events(cfgs: 'list[PromoEventConfig]', kpi_gt: 'np.ndarray', rng: 'np.random.Generator', baseline_kpi_gt: 'np.ndarray | None' = None) -> 'dict'`

Apply multiplicative promo lifts to the KPI.

Two lift modes are supported per event via `PromoEventConfig.baseline_only`:

  - `baseline_only=False` (default): the lift scales the TOTAL KPI,
    media contributions included.  True channel ROI on event weeks runs
    slightly above the recorded `roi_m` (media works harder in promos).
  - `baseline_only=True`: the lift scales only the non-media-driven
    portion of the KPI (`baseline_kpi_gt`).  Media contributions are
    untouched, so recorded ground-truth ROIs remain exactly true.

Combined semantics when both modes are active on the same week::

    kpi_new = baseline * M_base * M_full + (kpi - baseline) * M_full

where `M_base` collects baseline-only multipliers and `M_full`
collects total-KPI multipliers.

Args:
    cfgs: Promo event configs.
    kpi_gt: KPI before promo effects, shape (n_geos, n_times).
    rng: Random generator.
    baseline_kpi_gt: Non-media-driven KPI portion (baseline + context +
        non-media contributions), shape (n_geos, n_times).  Required if
        any event has `baseline_only=True`.

Returns:
    Dict with:
      `kpi_gt`                 – KPI after lifts, same shape.
      `multiplier_gt`          – realized total-KPI multiplier (M_full).
      `baseline_multiplier_gt` – realized baseline-only multiplier (M_base).
      `flags_tc`               – binary flags, shape (n_times, n_events).
      `spec`                   – list of recipe dicts for the ground truth.

### `apply_kpi_noise(kpi_gt: 'np.ndarray', kpi_noise_pct: 'float', rng: 'np.random.Generator') -> 'dict'`

Apply multiplicative observation noise to the final KPI.

`kpi *= (1 + N(0, kpi_noise_pct))`, floored at 0 to keep KPI valid.

Args:
    kpi_gt: KPI before noise, shape (n_geos, n_times).
    kpi_noise_pct: Coefficient of variation of the noise (0 disables).
    rng: Random generator.

Returns:
    Dict with:
      `kpi_gt`        – noisy KPI, same shape.
      `multiplier_gt` – realized noise multipliers (ground truth).
      `realized_cv`   – realized coefficient of variation of the noise.


---

## meridian_simulator.baseline

### `simulate_baseline(cfg: 'BaselineConfig', n_times: 'int', n_geos: 'int') -> 'dict'`

Simulate all baseline components.

Args:
    cfg: BaselineConfig instance.
    n_times: Number of time periods.
    n_geos: Number of geos.

Returns:
    Dict with keys:
      `tau_g`          – shape (n_geos,)
      `mu_t`           – shape (n_times,)
      `trend_t`        – shape (n_times,)
      `seasonality_t`  – shape (n_times,)
      `eps_gt`         – shape (n_geos, n_times)
      `baseline_gt`    – shape (n_geos, n_times)  [sum of all components]


---

## meridian_simulator.media

### `simulate_paid_media(media_cfgs: 'list[MediaChannelConfig]', rf_cfgs: 'list[RFChannelConfig]', p_g: 'tf.Tensor', unit_value: 'tf.Tensor', n_times: 'int', prior, seasonality_t: 'Optional[tf.Tensor]' = None) -> 'dict'`

Simulate all paid media channels end-to-end.

Steps: impressions → R&F derivation → transform → HillAdstock →
       beta back-solve or draw → spend.

Returns a rich dict with raw and transformed tensors, coefficients, spend,
adstock/Hill parameters, and ROI diagnostics.

### `simulate_impressions(channel_cfgs: 'list[MediaChannelConfig] | list[RFChannelConfig]', n_geos: 'int', n_times: 'int', p_g: 'tf.Tensor', seasonality_t: 'Optional[tf.Tensor]' = None) -> 'dict'`

Simulate raw impressions for all channels (paid or organic).

Args:
    seasonality_t: Optional baseline seasonality wave, shape (n_times,).
        Channels with `seasonal_flighting > 0` have their weekly
        activity mean modulated by this wave (normalized to unit peak),
        simulating demand-synchronized media buying.  `None` or a
        zero flighting value reproduces the historical iid behavior.

Returns dict with:
  `ipc_gtm`       – impressions per capita  (n_geos, n_times, n_ch)
  `impression_gtm`– total impressions       (n_geos, n_times, n_ch)

### `simulate_reach_frequency(ipc_gtm: 'tf.Tensor', impression_gtm: 'tf.Tensor', p_g: 'tf.Tensor') -> 'dict'`

Derive reach and frequency from impressions via Poisson arrival model.

Returns dict with keys `reach_gtm` and `freq_gtm`.


---

## meridian_simulator.organic

### `simulate_organic_media(media_cfgs: 'list[OrganicMediaChannelConfig]', rf_cfgs: 'list[OrganicRFChannelConfig]', p_g: 'tf.Tensor', n_times: 'int', prior) -> 'dict'`

Simulate organic impression-based and R&F channels.

Returns dict with:
  `organic_impression_gtm`   – raw impressions, (n_geos, n_times, n_om)
  `organic_reach_gtm`        – reach for R&F channels, (n_geos, n_times, n_orf)
  `organic_freq_gtm`         – frequency, (n_geos, n_times, n_orf)
  `organic_media_transformed`– HillAdstock output, (n_geos, n_times, n_om)
  `organic_rf_transformed`   – HillAdstock output, (n_geos, n_times, n_orf)
  `beta_om`                  – hierarchical means, (n_om,)
  `eta_om`                   – hierarchical stds, (n_om,)
  `beta_gom`                 – geo-level coefficients, (n_geos, n_om)
  `beta_orf`                 – hierarchical means for R&F, (n_orf,)
  `eta_orf`                  – hierarchical stds, (n_orf,)
  `beta_gorf`                – geo-level coefficients, (n_geos, n_orf)
  `alpha_om`, `ec_om`, `slope_om`  – impression adstock/Hill params
  `alpha_orf`, `ec_orf`, `slope_orf`– R&F adstock/Hill params


---

## meridian_simulator.context

### `simulate_context_variables(cfgs: 'list[ContextVariableConfig]', n_geos: 'int', n_times: 'int', p_g: 'tf.Tensor', rng: 'np.random.Generator') -> 'dict'`

Simulate context (control) variables.

Returns dict with:
  `context_gtc`             – raw series, shape (n_geos, n_times, n_c)
  `transformed_context_gtc` – standardised series (as Meridian expects)
  `gamma_c`                 – national-level control coefficients, (n_c,)
  `xi_c`                    – geo-level deviation std, (n_c,)
  `gamma_gc`                – geo-level coefficients, (n_geos, n_c)

### `simulate_non_media_channels(cfgs: 'list[NonMediaChannelConfig]', n_geos: 'int', n_times: 'int', rng: 'np.random.Generator') -> 'dict'`

Simulate non-media time-series channels.

Non-media channels are standardised geo-time series (e.g., promotions,
price reductions) that enter the KPI equation with their own coefficients.

Returns dict with:
  `non_media_gtc`             – raw series, (n_geos, n_times, n_n)
  `transformed_non_media_gtc` – mean-centred unit-std series
  `gamma_n`                   – national-level coefficients, (n_n,)
  `xi_n`                      – geo-level deviation std, (n_n,)
  `gamma_gn`                  – geo-level coefficients, (n_geos, n_n)


---

## meridian_simulator.output

### `build_xarray_dataset(*, geo_names: 'list[str]', time_names: 'list[str]', media_channel_names: 'list[str]', rf_channel_names: 'list[str]', organic_media_channel_names: 'list[str]', organic_rf_channel_names: 'list[str]', non_media_channel_names: 'list[str]', context_variable_names: 'list[str]', kpi_gt: 'np.ndarray', unit_value_gt: 'np.ndarray', population_g: 'np.ndarray', impression_gtm: 'np.ndarray', cost_gtm: 'np.ndarray', reach_gtm: 'np.ndarray', freq_gtm: 'np.ndarray', organic_impression_gtm: 'np.ndarray', organic_reach_gtm: 'np.ndarray', organic_freq_gtm: 'np.ndarray', non_media_gtc: 'np.ndarray', context_gtc: 'np.ndarray') -> 'dict[str, xr.DataArray]'`

Build a dict of xr.DataArrays ready for Meridian's InputData builder.

### `build_geo_dataframe(xr_dict: 'dict[str, xr.DataArray]') -> 'pd.DataFrame'`

Flatten the xarray dict into a single wide Pandas DataFrame (geo-level).

Column naming convention (matches DataFrameInputDataBuilder expectations):
  - Media impressions : `{channel}_impression`
  - Media spend       : `{channel}_spend`
  - R&F reach         : `{channel}_reach`
  - R&F frequency     : `{channel}_frequency`
  - Organic impr.     : `{channel}_organic_impression`
  - Organic reach     : `{channel}_organic_reach`
  - Organic freq.     : `{channel}_organic_frequency`
  - Controls          : `{var}_control`  (already suffixed in xr coords)
  - Non-media         : kept as-is (channel name)

### `build_national_dataframe(geo_df: 'pd.DataFrame') -> 'pd.DataFrame'`

Aggregate the geo-level DataFrame to a national (single-geo) level.
