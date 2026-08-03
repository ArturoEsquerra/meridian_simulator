"""Native simulation of dataset 'traps' and KPI observation noise.

This module generates the components that make a simulated dataset behave like
real-world data an analyst must interrogate rather than trust:

  - **Collinear variables**: linear functions of population or a context
    variable plus noise.  Present in the data, absent from the KPI equation.
  - **Endogenous variables**: driven by lagged media activity or by the KPI
    itself (reverse causality).  Present in the data, absent from the KPI
    equation — including them as controls biases media ROI.
  - **Promo events**: multiplicative structural lifts to the KPI on specific
    weeks, the canonical use case for Meridian's ``non_media_treatments``.
  - **KPI observation noise**: multiplicative noise on the final KPI whose
    magnitude is a single interpretable knob (coefficient of variation).

All functions take an explicit ``np.random.Generator`` so results are fully
reproducible from ``SimulationConfig.seed``.
"""

from __future__ import annotations

import numpy as np

from meridian_simulator.config import (
    CollinearVariableConfig,
    EndogenousVariableConfig,
    PromoEventConfig,
)


# ---------------------------------------------------------------------------
# Collinear distractor variables
# ---------------------------------------------------------------------------


def simulate_collinear_variables(
    cfgs: list[CollinearVariableConfig],
    context_gtc: np.ndarray,
    context_variable_names: list[str],
    population_g: np.ndarray,
    n_times: int,
    rng: np.random.Generator,
) -> dict:
    """Simulate collinear distractor variables.

    Args:
        cfgs: Collinear variable configs.
        context_gtc: Raw context series, shape (n_geos, n_times, n_ctx).
        context_variable_names: Names aligned with the last axis of
            ``context_gtc``.
        population_g: Population per geo, shape (n_geos,).
        n_times: Number of time periods.
        rng: Random generator.

    Returns:
        Dict with:
          ``collinear_gtc`` – shape (n_geos, n_times, n_collinear)
          ``spec``          – list of dicts describing each variable's recipe
                              (for the ground-truth record).
    """
    n_geos = len(population_g)
    if not cfgs:
        return {
            "collinear_gtc": np.zeros((n_geos, n_times, 0), dtype=np.float32),
            "spec": [],
        }

    parts = []
    spec = []
    for c in cfgs:
        if c.source == "population":
            # Population is constant over time; broadcast to (n_geos, n_times)
            source = np.repeat(
                population_g[:, np.newaxis], n_times, axis=1
            ).astype(np.float32)
        else:
            idx = context_variable_names.index(c.source)
            source = np.asarray(context_gtc[:, :, idx], dtype=np.float32)

        noise = rng.normal(0.0, c.noise_std, size=source.shape).astype(np.float32)
        series = c.coefficient * source + c.intercept + noise
        if c.round_decimals is not None:
            series = np.round(series, c.round_decimals)

        parts.append(series)
        spec.append(
            {
                "name": c.name,
                "source": c.source,
                "coefficient": c.coefficient,
                "intercept": c.intercept,
                "noise_std": c.noise_std,
                "realized_correlation": float(
                    np.corrcoef(series.ravel(), source.ravel())[0, 1]
                ),
            }
        )

    return {
        "collinear_gtc": np.stack(parts, axis=-1),
        "spec": spec,
    }


# ---------------------------------------------------------------------------
# Endogenous distractor variables
# ---------------------------------------------------------------------------


def simulate_endogenous_variables(
    cfgs: list[EndogenousVariableConfig],
    cost_gtm: np.ndarray,
    paid_channel_names: list[str],
    kpi_gt: np.ndarray,
    rng: np.random.Generator,
) -> dict:
    """Simulate endogenous distractor variables.

    Each variable mixes a lagged, standardized driver series with independent
    noise::

        z     = weight * standardize(lag(driver)) + (1 - weight) * N(0, 1)
        value = base + scale * z

    Args:
        cfgs: Endogenous variable configs.
        cost_gtm: Paid media spend, shape (n_geos, n_times, n_paid_total)
            covering impression channels then R&F channels.
        paid_channel_names: Names aligned with the last axis of ``cost_gtm``.
        kpi_gt: Final KPI, shape (n_geos, n_times) — used when driver='kpi'.
        rng: Random generator.

    Returns:
        Dict with:
          ``endogenous_gtc`` – shape (n_geos, n_times, n_endog)
          ``spec``           – list of recipe dicts for the ground truth.
    """
    n_geos, n_times = kpi_gt.shape
    if not cfgs:
        return {
            "endogenous_gtc": np.zeros((n_geos, n_times, 0), dtype=np.float32),
            "spec": [],
        }

    parts = []
    spec = []
    for c in cfgs:
        if c.driver == "kpi":
            driver = np.asarray(kpi_gt, dtype=np.float32)
        else:
            idx = paid_channel_names.index(c.driver)
            driver = np.asarray(cost_gtm[:, :, idx], dtype=np.float32)

        # Lag the driver along time; back-fill the first `lag` periods
        lagged = np.empty_like(driver)
        if c.lag == 0:
            lagged[:] = driver
        else:
            lagged[:, c.lag:] = driver[:, : n_times - c.lag]
            lagged[:, : c.lag] = driver[:, [0]]

        std = lagged.std()
        driver_z = (lagged - lagged.mean()) / (std if std > 0 else 1.0)

        noise = rng.normal(0.0, 1.0, size=driver.shape).astype(np.float32)
        z = c.weight * driver_z + (1.0 - c.weight) * noise
        series = (c.base + c.scale * z).astype(np.float32)

        parts.append(series)
        spec.append(
            {
                "name": c.name,
                "driver": c.driver,
                "lag": c.lag,
                "weight": c.weight,
                "realized_correlation_with_lagged_driver": float(
                    np.corrcoef(series.ravel(), driver_z.ravel())[0, 1]
                ),
            }
        )

    return {
        "endogenous_gtc": np.stack(parts, axis=-1),
        "spec": spec,
    }


# ---------------------------------------------------------------------------
# Promotional events
# ---------------------------------------------------------------------------


def apply_promo_events(
    cfgs: list[PromoEventConfig],
    kpi_gt: np.ndarray,
    rng: np.random.Generator,
    baseline_kpi_gt: np.ndarray | None = None,
) -> dict:
    """Apply multiplicative promo lifts to the KPI.

    Two lift modes are supported per event via ``PromoEventConfig.baseline_only``:

      - ``baseline_only=False`` (default): the lift scales the TOTAL KPI,
        media contributions included.  True channel ROI on event weeks runs
        slightly above the recorded ``roi_m`` (media works harder in promos).
      - ``baseline_only=True``: the lift scales only the non-media-driven
        portion of the KPI (``baseline_kpi_gt``).  Media contributions are
        untouched, so recorded ground-truth ROIs remain exactly true.

    Combined semantics when both modes are active on the same week::

        kpi_new = baseline * M_base * M_full + (kpi - baseline) * M_full

    where ``M_base`` collects baseline-only multipliers and ``M_full``
    collects total-KPI multipliers.

    Args:
        cfgs: Promo event configs.
        kpi_gt: KPI before promo effects, shape (n_geos, n_times).
        rng: Random generator.
        baseline_kpi_gt: Non-media-driven KPI portion (baseline + context +
            non-media contributions), shape (n_geos, n_times).  Required if
            any event has ``baseline_only=True``.

    Returns:
        Dict with:
          ``kpi_gt``                 – KPI after lifts, same shape.
          ``multiplier_gt``          – realized total-KPI multiplier (M_full).
          ``baseline_multiplier_gt`` – realized baseline-only multiplier (M_base).
          ``flags_tc``               – binary flags, shape (n_times, n_events).
          ``spec``                   – list of recipe dicts for the ground truth.
    """
    kpi = np.asarray(kpi_gt, dtype=np.float32)
    n_geos, n_times = kpi.shape
    mult_full = np.ones((n_geos, n_times), dtype=np.float32)
    mult_base = np.ones((n_geos, n_times), dtype=np.float32)
    flags = np.zeros((n_times, len(cfgs)), dtype=np.float32)
    spec = []

    if any(c.baseline_only for c in cfgs) and baseline_kpi_gt is None:
        raise ValueError(
            "baseline_kpi_gt is required when any PromoEventConfig has "
            "baseline_only=True."
        )

    for j, c in enumerate(cfgs):
        target = mult_base if c.baseline_only else mult_full
        allow_neg = getattr(c, "allow_negative_lift", False)
        lift_floor = -0.95 if allow_neg else 0.0
        for w in c.weeks:
            geo_lift = c.lift_pct + rng.normal(0.0, c.lift_geo_std, size=n_geos)
            geo_lift = np.maximum(geo_lift, lift_floor).astype(np.float32)
            target[:, w] *= 1.0 + geo_lift
            flags[w, j] = 1.0
        spec.append(
            {
                "name": c.name,
                "weeks": list(c.weeks),
                "lift_pct": c.lift_pct,
                "lift_geo_std": c.lift_geo_std,
                "include_flag_in_output": c.include_flag_in_output,
                "baseline_only": c.baseline_only,
                "allow_negative_lift": allow_neg,
            }
        )

    if baseline_kpi_gt is not None:
        baseline = np.asarray(baseline_kpi_gt, dtype=np.float32)
        media_part = kpi - baseline
        kpi_new = baseline * mult_base * mult_full + media_part * mult_full
    else:
        kpi_new = kpi * mult_full

    return {
        "kpi_gt": kpi_new,
        "multiplier_gt": mult_full,
        "baseline_multiplier_gt": mult_base,
        "flags_tc": flags,
        "spec": spec,
    }


# ---------------------------------------------------------------------------
# KPI observation noise
# ---------------------------------------------------------------------------


def apply_kpi_noise(
    kpi_gt: np.ndarray,
    kpi_noise_pct: float,
    rng: np.random.Generator,
) -> dict:
    """Apply multiplicative observation noise to the final KPI.

    ``kpi *= (1 + N(0, kpi_noise_pct))``, floored at 0 to keep KPI valid.

    Args:
        kpi_gt: KPI before noise, shape (n_geos, n_times).
        kpi_noise_pct: Coefficient of variation of the noise (0 disables).
        rng: Random generator.

    Returns:
        Dict with:
          ``kpi_gt``        – noisy KPI, same shape.
          ``multiplier_gt`` – realized noise multipliers (ground truth).
          ``realized_cv``   – realized coefficient of variation of the noise.
    """
    kpi = np.asarray(kpi_gt, dtype=np.float32)
    if kpi_noise_pct <= 0:
        return {
            "kpi_gt": kpi,
            "multiplier_gt": np.ones_like(kpi),
            "realized_cv": 0.0,
        }

    multiplier = 1.0 + rng.normal(0.0, kpi_noise_pct, size=kpi.shape).astype(
        np.float32
    )
    multiplier = np.maximum(multiplier, 0.0)
    return {
        "kpi_gt": kpi * multiplier,
        "multiplier_gt": multiplier,
        "realized_cv": float(multiplier.std()),
    }
