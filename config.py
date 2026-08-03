"""Configuration dataclasses for the Meridian simulator.

All parameters that govern the simulation are declared here, with sensible
defaults that mirror or extend the logic from Meridian's official demo notebook
(demo/RF_Data_Simulation_for_Meridian.ipynb).
"""

from __future__ import annotations

import dataclasses
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Channel-level configs
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class MediaChannelConfig:
    """Configuration for a single paid impression-based media channel.

    Attributes:
        name: Human-readable channel label (e.g. ``"paid_search"``).
        target_roi: Desired ground-truth ROI (incremental revenue / spend).
            When set, ``beta_m`` is back-solved so that E[ROI] ≈ target_roi.
            Mutually exclusive with ``beta_m_mean``.
        beta_m_mean: Mean of the hierarchical log-normal prior for geo-level
            media coefficients.  Only used when ``target_roi`` is None.
        beta_m_std: Std of the hierarchical distribution (eta_m).
        alpha: Adstock decay parameter in [0, 1].  If None, sampled from
            Meridian's default prior.
        ec: Hill saturation half-saturation point (ec > 0).  If None, sampled.
        slope: Hill slope parameter (> 0).  If None, sampled.
        cpm_low: Lower bound for cost-per-thousand impressions uniform draw.
        cpm_high: Upper bound for cost-per-thousand impressions uniform draw.
        max_lag: Maximum lag for adstock carry-over effect (in time periods).
        impression_mean_channel: Mean channel effect used to draw impressions.
        impression_mean_time: Mean time effect used to draw impressions.
        impression_std: Std of the geo-time idiosyncratic impression noise.
        seasonal_flighting: Strength in [0, 1] of demand-synchronized media
            flighting.  0.0 (default) draws media activity independently of
            the baseline seasonality — the historical behavior.  Positive
            values modulate the channel's weekly activity with the baseline
            seasonal wave (media planners buying INTO high season), creating
            realistic media–seasonality confounding that tests whether an
            analyst's knot/control choices can separate media from demand.
    """

    name: str = "channel"
    target_roi: Optional[float] = None
    beta_m_mean: float = 0.9
    beta_m_std: float = 0.1
    alpha: Optional[float] = None          # adstock decay
    ec: Optional[float] = None             # half-saturation
    slope: Optional[float] = None          # Hill slope
    cpm_low: float = 0.011
    cpm_high: float = 0.012
    max_lag: int = 8
    impression_mean_channel: float = 1.0
    impression_mean_time: float = 0.8
    impression_std: float = 0.5
    seasonal_flighting: float = 0.0


@dataclasses.dataclass
class RFChannelConfig:
    """Configuration for a single reach-and-frequency media channel.

    All adstock/Hill parameters apply to the frequency dimension.

    Attributes:
        name: Human-readable channel label (e.g. ``"youtube"``).
        target_roi: Desired ground-truth ROI.  When set, beta_rf is back-solved.
        beta_rf_mean: Hierarchical mean for log-normal beta prior.
        beta_rf_std: Hierarchical std (eta_rf).
        alpha: Adstock decay in [0, 1].  If None, sampled.
        ec: Half-saturation.  If None, sampled.
        slope: Hill slope.  If None, sampled.
        cpm_low: Lower bound for CPM uniform draw.
        cpm_high: Upper bound for CPM uniform draw.
        max_lag: Maximum adstock lag.
        impression_mean_channel: Channel-level impression mean.
        impression_mean_time: Time-level impression mean.
        impression_std: Geo-time idiosyncratic impression std.
    """

    name: str = "rf_channel"
    target_roi: Optional[float] = None
    beta_rf_mean: float = 0.9
    beta_rf_std: float = 0.1
    alpha: Optional[float] = None
    ec: Optional[float] = None
    slope: Optional[float] = None
    cpm_low: float = 0.011
    cpm_high: float = 0.012
    max_lag: int = 8
    impression_mean_channel: float = 1.0
    impression_mean_time: float = 0.8
    impression_std: float = 0.5
    seasonal_flighting: float = 0.0


@dataclasses.dataclass
class OrganicMediaChannelConfig:
    """Configuration for an organic (unpaid) impression-based media channel.

    Organic channels affect KPI but have no associated spend/CPM.

    Attributes:
        name: Channel label.
        beta_om_mean: Hierarchical mean of organic media coefficient.
        beta_om_std: Hierarchical std (eta_om).
        alpha: Adstock decay.  If None, sampled from prior.
        ec: Half-saturation.  If None, sampled.
        slope: Hill slope.  If None, sampled.
        max_lag: Maximum adstock lag.
        impression_mean_channel: Channel impression mean.
        impression_mean_time: Time impression mean.
        impression_std: Geo-time noise std.
    """

    name: str = "organic_channel"
    beta_om_mean: float = 0.5
    beta_om_std: float = 0.1
    alpha: Optional[float] = None
    ec: Optional[float] = None
    slope: Optional[float] = None
    max_lag: int = 8
    impression_mean_channel: float = 0.8
    impression_mean_time: float = 0.6
    impression_std: float = 0.4


@dataclasses.dataclass
class OrganicRFChannelConfig:
    """Configuration for an organic reach-and-frequency channel."""

    name: str = "organic_rf_channel"
    beta_orf_mean: float = 0.5
    beta_orf_std: float = 0.1
    alpha: Optional[float] = None
    ec: Optional[float] = None
    slope: Optional[float] = None
    max_lag: int = 8
    impression_mean_channel: float = 0.8
    impression_mean_time: float = 0.6
    impression_std: float = 0.4


@dataclasses.dataclass
class NonMediaChannelConfig:
    """Configuration for a non-media time-series channel.

    These are exogenous variables that affect KPI but are not media
    (e.g. price index, distribution score, promotion flags).

    Attributes:
        name: Channel label.
        gamma_n_mean: Mean of the normal prior on the coefficient.
        gamma_n_std: Std of the normal prior on the coefficient.
        xi_n_std: Std of the geo-level deviation (HalfNormal).
        series_mean: Mean of the raw time series.
        series_std: Std of the raw time series.
    """

    name: str = "non_media_channel"
    gamma_n_mean: float = 1.0
    gamma_n_std: float = 0.5
    xi_n_std: float = 0.3
    series_mean: float = 0.0
    series_std: float = 1.0


@dataclasses.dataclass
class ContextVariableConfig:
    """Configuration for a context/control variable.

    Context variables (e.g. GDP index, exchange rate, competitor spend) are
    geo-time varying variables that control for confounders.

    Attributes:
        name: Variable label (e.g. ``"gdp_index"``).
        gamma_c_mean: Mean of normal prior on the coefficient.
        gamma_c_std: Std of normal prior on the coefficient.
        xi_c_std: Std of geo-level deviation (HalfNormal).
        series_mean: Mean of the simulated raw series.
        series_std: Std of the simulated raw series.
        ar1_coef: Auto-regressive AR(1) coefficient for temporal correlation.
            Set to 0.0 for iid draws, up to ~0.9 for persistent series.
        trend: Optional linear drift added to the series over time.
    """

    name: str = "context_var"
    gamma_c_mean: float = 3.5
    gamma_c_std: float = 0.5
    xi_c_std: float = 0.3
    series_mean: float = 0.0
    series_std: float = 3.0
    ar1_coef: float = 0.0
    trend: float = 0.0


@dataclasses.dataclass
class CollinearVariableConfig:
    """Configuration for a collinear distractor variable.

    Collinear variables are derived from an existing series (population or a
    context variable) via a linear map plus noise::

        value = coefficient * source + intercept + N(0, noise_std)

    They appear in the output dataset but have ZERO causal effect on the KPI —
    their purpose is to test whether an analyst detects and handles
    multicollinearity (e.g. via VIF) instead of throwing every column into the
    model.

    Attributes:
        name: Output column label (e.g. ``"store_count"``).
        source: What the variable is collinear with.  Either the literal
            string ``"population"`` or the ``name`` of a configured
            ``ContextVariableConfig``.
        coefficient: Linear multiplier applied to the source series.
        intercept: Constant offset added after the multiplication.
        noise_std: Std of the additive Gaussian noise.  Relative to the scale
            of ``coefficient * source``; small values give near-perfect
            collinearity (VIF → ∞), larger values weaken it.
        round_decimals: If not None, round the final series to this many
            decimals (use 0 for integer-like variables such as store counts).
    """

    name: str = "collinear_var"
    source: str = "population"
    coefficient: float = 1.0
    intercept: float = 0.0
    noise_std: float = 1.0
    round_decimals: Optional[int] = None


@dataclasses.dataclass
class EndogenousVariableConfig:
    """Configuration for an endogenous distractor variable.

    Endogenous variables are *symptoms* of the system, not causes: they are
    driven by a lagged media channel's activity (e.g. Google query volume
    rising after YouTube bursts) or by the KPI itself (reverse causality).
    Including them as controls in an MMM soaks up media credit and biases ROI
    downward — the classic mediator/endogeneity trap.

    The series is built as::

        z = weight * normalize(lagged driver) + (1 - weight) * N(0, 1)
        value = base + scale * z

    They have ZERO direct causal effect on the KPI.

    Attributes:
        name: Output column label (e.g. ``"gqv_index"``).
        driver: Name of the paid media / R&F channel whose SPEND drives the
            series, or the literal string ``"kpi"`` for reverse causality.
        lag: Number of time periods the driver is lagged by (>= 0).
        weight: Share of variance explained by the driver, in [0, 1].
            0.65 gives a clearly detectable but not degenerate correlation.
        base: Additive level of the output series.
        scale: Multiplier applied to the standardized mixed series.
    """

    name: str = "endogenous_var"
    driver: str = "kpi"
    lag: int = 1
    weight: float = 0.65
    base: float = 50.0
    scale: float = 10.0


@dataclasses.dataclass
class PromoEventConfig:
    """Configuration for a promotional event that structurally lifts the KPI.

    Promo events model commercial actions (deep discounts, special payment
    terms, in-store events) that lift sales independently of media — the
    textbook use case for Meridian's ``non_media_treatments``.  The lift is
    applied MULTIPLICATIVELY to the final KPI on the configured weeks::

        kpi[:, week] *= (1 + lift_pct + N(0, lift_geo_std))   per geo

    Attributes:
        name: Event label (e.g. ``"hot_sale_2024"``).  When
            ``include_flag_in_output`` is True, a binary ``{name}`` column is
            added to the output DataFrames.
        weeks: Time-period indices (0-based) on which the event is active.
        lift_pct: Fractional KPI lift on event weeks (0.25 = +25%).
        lift_geo_std: Std of geo-level jitter added to the lift, so regions
            respond heterogeneously.  0.0 applies a uniform lift.
        include_flag_in_output: Whether to include the binary event flag as a
            column in the output DataFrames.  Set False to force analysts to
            engineer the flags from domain knowledge.
        baseline_only: Controls what the multiplicative lift applies to.
            False (default): the lift scales the TOTAL KPI on event weeks —
            media contributions are lifted too, so true channel ROI on those
            weeks runs slightly above the recorded ``roi_m`` (media works
            harder during promos; realistic synergy).
            True: the lift scales only the non-media-driven (baseline +
            context + non-media) portion of the KPI.  Media contributions are
            untouched, so the recorded ground-truth ROIs remain exactly true.
        allow_negative_lift: Permit ``lift_pct < 0`` to model NEGATIVE
            structural shocks — stockouts, supply disruptions, store
            closures, demand collapses.  When True, per-geo realized lifts
            are floored at -0.95 (KPI can drop up to 95% but never below 0).
            When False (default, historical behavior) negative realized
            lifts are clamped to 0.
    """

    name: str = "promo_event"
    weeks: list[int] = dataclasses.field(default_factory=list)
    lift_pct: float = 0.25
    lift_geo_std: float = 0.0
    include_flag_in_output: bool = True
    baseline_only: bool = False
    allow_negative_lift: bool = False


# ---------------------------------------------------------------------------
# Baseline / time-series structure
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class SeasonalityComponent:
    """A single sinusoidal seasonality component.

    Attributes:
        amplitude: Peak-to-mean amplitude of the wave.
        period_weeks: Period of the cycle in time periods (weeks if weekly data).
        phase_weeks: Phase offset in time periods.
    """

    amplitude: float = 2.0
    period_weeks: float = 52.0
    phase_weeks: float = 0.0


@dataclasses.dataclass
class BaselineConfig:
    """Configuration for the baseline (intercept + trend + seasonality) component.

    Attributes:
        tau_mean: Mean of the geo-level intercept N(tau_mean, tau_std).
        tau_std: Std of the geo-level intercept.
        n_knots: Number of knots for the time-varying intercept mu_t.
            Set to n_times for full-flexibility, or a smaller integer for
            smoother trends.  ``None`` defaults to n_times (full knots).
        knot_std: Std of the Normal prior on knot values.
        trend_slope: Additive linear trend per time period (in KPI-per-capita
            units).  Positive values give upward trend.
        seasonality: List of SeasonalityComponent instances stacked additively.
        noise_std: Std of the iid residual error term epsilon_gt.
        unit_value_low: Lower bound for revenue-per-KPI unit.
        unit_value_high: Upper bound for revenue-per-KPI unit.
        population_low: Lower bound for geo population (drawn Uniform).
        population_high: Upper bound for geo population (drawn Uniform).
    """

    tau_mean: float = 15.0
    tau_std: float = 1.2
    n_knots: Optional[int] = None          # defaults to n_times
    knot_std: float = 2.0
    trend_slope: float = 0.0
    seasonality: list[SeasonalityComponent] = dataclasses.field(
        default_factory=list
    )
    noise_std: float = 0.5
    unit_value_low: float = 0.0345
    unit_value_high: float = 0.0355
    population_low: float = 1e5
    population_high: float = 1e6


# ---------------------------------------------------------------------------
# Top-level simulation config
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class SimulationConfig:
    """Top-level configuration for the Meridian dataset simulator.

    Attributes:
        n_times: Number of time periods (weeks).
        n_geos: Number of geographic units.  Set to 1 for a national model.
        start_date: First time period as a ``YYYY-MM-DD`` string.
        seed: Random seed for reproducibility.
        media_channels: List of impression-based paid channel configs.
        rf_channels: List of reach-and-frequency channel configs.
        organic_media_channels: List of organic impression channel configs.
        organic_rf_channels: List of organic RF channel configs.
        non_media_channels: List of non-media channel configs.
        context_variables: List of context/control variable configs.
        collinear_variables: List of collinear distractor variable configs.
            These appear in the output data but have no causal effect on KPI.
        endogenous_variables: List of endogenous distractor variable configs.
            Driven by lagged media activity or the KPI itself; no causal
            effect on KPI.
        promo_events: List of promotional event configs.  Each applies a
            multiplicative structural lift to the KPI on its event weeks.
        kpi_noise_pct: Coefficient of variation of multiplicative observation
            noise applied to the FINAL KPI: ``kpi *= (1 + N(0, kpi_noise_pct))``.
            0.0 disables it.  This is the main knob for how noisy sales data
            are relative to signal; ``baseline.noise_std`` remains available
            as additive per-capita noise inside the baseline.  Typical values:
            0.01 (very clean) to 0.10 (noisy real-world data).
        baseline: Baseline (intercept + trend + seasonality) config.
    """

    n_times: int = 156
    n_geos: int = 20
    start_date: str = "2021-01-25"
    seed: int = 1320
    media_channels: list[MediaChannelConfig] = dataclasses.field(
        default_factory=list
    )
    rf_channels: list[RFChannelConfig] = dataclasses.field(
        default_factory=list
    )
    organic_media_channels: list[OrganicMediaChannelConfig] = dataclasses.field(
        default_factory=list
    )
    organic_rf_channels: list[OrganicRFChannelConfig] = dataclasses.field(
        default_factory=list
    )
    non_media_channels: list[NonMediaChannelConfig] = dataclasses.field(
        default_factory=list
    )
    context_variables: list[ContextVariableConfig] = dataclasses.field(
        default_factory=list
    )
    collinear_variables: list[CollinearVariableConfig] = dataclasses.field(
        default_factory=list
    )
    endogenous_variables: list[EndogenousVariableConfig] = dataclasses.field(
        default_factory=list
    )
    promo_events: list[PromoEventConfig] = dataclasses.field(
        default_factory=list
    )
    kpi_noise_pct: float = 0.0
    baseline: BaselineConfig = dataclasses.field(
        default_factory=BaselineConfig
    )

    def __post_init__(self):
        if self.n_geos < 1:
            raise ValueError("n_geos must be >= 1.")
        if self.n_times < 2:
            raise ValueError("n_times must be >= 2.")
        if self.kpi_noise_pct < 0:
            raise ValueError("kpi_noise_pct must be >= 0.")

        valid_ctx = {c.name for c in self.context_variables}
        for cv in self.collinear_variables:
            if cv.source != "population" and cv.source not in valid_ctx:
                raise ValueError(
                    f"CollinearVariableConfig '{cv.name}' has source "
                    f"'{cv.source}', which is neither 'population' nor a "
                    f"configured context variable name ({sorted(valid_ctx)})."
                )

        valid_drivers = (
            {c.name for c in self.media_channels}
            | {c.name for c in self.rf_channels}
            | {"kpi"}
        )
        for ev in self.endogenous_variables:
            if ev.driver not in valid_drivers:
                raise ValueError(
                    f"EndogenousVariableConfig '{ev.name}' has driver "
                    f"'{ev.driver}', which is neither 'kpi' nor a configured "
                    f"paid channel name ({sorted(valid_drivers - {'kpi'})})."
                )
            if ev.lag < 0:
                raise ValueError(f"EndogenousVariableConfig '{ev.name}': lag must be >= 0.")
            if not 0.0 <= ev.weight <= 1.0:
                raise ValueError(f"EndogenousVariableConfig '{ev.name}': weight must be in [0, 1].")

        for pe in self.promo_events:
            bad = [w for w in pe.weeks if not 0 <= w < self.n_times]
            if bad:
                raise ValueError(
                    f"PromoEventConfig '{pe.name}' has week indices {bad} "
                    f"outside [0, {self.n_times - 1}]."
                )
            if pe.lift_pct < 0 and not pe.allow_negative_lift:
                raise ValueError(
                    f"PromoEventConfig '{pe.name}' has lift_pct={pe.lift_pct} "
                    f"< 0. Set allow_negative_lift=True to model negative "
                    f"structural shocks (stockouts, disruptions)."
                )

        for mc in list(self.media_channels) + list(self.rf_channels):
            if not 0.0 <= mc.seasonal_flighting <= 1.0:
                raise ValueError(
                    f"Channel '{mc.name}': seasonal_flighting must be in "
                    f"[0, 1], got {mc.seasonal_flighting}."
                )

    @property
    def is_national(self) -> bool:
        return self.n_geos == 1

    @property
    def n_media_channels(self) -> int:
        return len(self.media_channels)

    @property
    def n_rf_channels(self) -> int:
        return len(self.rf_channels)

    @property
    def n_organic_media_channels(self) -> int:
        return len(self.organic_media_channels)

    @property
    def n_organic_rf_channels(self) -> int:
        return len(self.organic_rf_channels)

    @property
    def n_non_media_channels(self) -> int:
        return len(self.non_media_channels)

    @property
    def n_context_variables(self) -> int:
        return len(self.context_variables)

    @property
    def n_collinear_variables(self) -> int:
        return len(self.collinear_variables)

    @property
    def n_endogenous_variables(self) -> int:
        return len(self.endogenous_variables)

    @property
    def n_promo_events(self) -> int:
        return len(self.promo_events)

    @property
    def n_total_paid_channels(self) -> int:
        return self.n_media_channels + self.n_rf_channels
