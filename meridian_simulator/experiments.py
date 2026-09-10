"""Simulate incrementality experiments and derive Meridian calibration priors.

Meridian's key differentiating feature is incrementality calibration: informing
the ROI priors of a Bayesian MMM with the results of randomized lift
experiments (geo holdouts, conversion-lift studies). Meridian supports two
calibration geometries:

  1. **Whole-duration calibration** — a per-channel informative LogNormal
     prior on ``roi_m``, treating the experiment as a measurement of the
     channel's ROI over the full modeling window.
  2. **Date-restricted calibration** — the same informative prior combined
     with ``ModelSpec(roi_calibration_period=...)``, a boolean
     ``(n_media_times, n_media_channels)`` mask declaring that the prior's
     ROI applies only to media executed during the experiment window.

This module simulates the experiments themselves. Given the simulator's
ground truth, each configured experiment computes the channel's TRUE ROI over
its window, then produces a noisy — and optionally biased — point estimate
with a standard error, exactly as a real lift study would deliver. The
results are converted to LogNormal prior parameters by moment matching and
packaged, together with the calibration-period mask, into everything a
Meridian ``ModelSpec`` needs.

The window-ROI numerator counts media-driven outcome occurring *within* the
window. Meridian's own definition attributes the adstocked carryover of
window media to the window; for windows much longer than the adstock
half-life (and always for full-duration experiments) the two coincide, and
the discrepancy is recorded nowhere as truth — the experiment estimate is
noisy anyway. See the whitepaper for discussion.
"""

from __future__ import annotations

import numpy as np

from meridian_simulator.config import ExperimentConfig


def lognormal_from_point_and_se(
    point_estimate: float, standard_error: float
) -> tuple[float, float]:
    """Moment-match a LogNormal(mu, sigma) to an experiment estimate.

    Chooses parameters so the LogNormal's mean equals the point estimate and
    its standard deviation equals the standard error::

        sigma^2 = ln(1 + se^2 / pe^2)
        mu      = ln(pe) - sigma^2 / 2

    This is the standard conversion for turning a lift study's (estimate, SE)
    into a Meridian ``roi_m`` prior.

    Args:
        point_estimate: Experiment ROI point estimate (> 0).
        standard_error: Standard error of the estimate (> 0).

    Returns:
        (mu, sigma) for ``tfp.distributions.LogNormal(mu, sigma)``.
    """
    if point_estimate <= 0:
        raise ValueError(
            f"point_estimate must be > 0 for a LogNormal prior, got "
            f"{point_estimate}. Truncate or floor experiment estimates before "
            "prior conversion."
        )
    sigma2 = float(np.log(1.0 + (standard_error / point_estimate) ** 2))
    mu = float(np.log(point_estimate) - sigma2 / 2.0)
    return mu, float(np.sqrt(sigma2))


def simulate_experiments(
    cfgs: list[ExperimentConfig],
    contribution_gtm: np.ndarray,
    cost_gtm: np.ndarray,
    unit_value_gt: np.ndarray,
    paid_channel_names: list[str],
    n_times: int,
    rng: np.random.Generator,
) -> dict:
    """Simulate incrementality experiments against the ground truth.

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
          ``results`` – list of per-experiment dicts (channel, window, true
                        window ROI, point estimate, standard error, bias).
          ``calibration`` – prior-building payload (see
                        :func:`build_calibration_priors`).
    """
    if not cfgs:
        return {"results": [], "calibration": None}

    results = []
    for c in cfgs:
        m = paid_channel_names.index(c.channel)
        w0 = 0 if c.start_week is None else c.start_week
        w1 = n_times if c.end_week is None else c.end_week + 1

        revenue_contrib = (
            contribution_gtm[:, w0:w1, m] * unit_value_gt[:, w0:w1]
        ).sum()
        spend = cost_gtm[:, w0:w1, m].sum()
        roi_true = float(revenue_contrib / max(spend, 1e-9))

        se = c.se_pct * roi_true
        point = roi_true * (1.0 + c.bias_pct) + float(rng.normal(0.0, se))
        point = max(point, 0.05)  # a lift study never reports ROI <= 0 as a
        # LogNormal-convertible estimate; floor mirrors practical truncation.

        mu, sigma = lognormal_from_point_and_se(point, se)

        results.append(
            {
                "name": c.name,
                "channel": c.channel,
                "channel_index": m,
                "start_week": w0,
                "end_week": w1 - 1,
                "full_duration": c.start_week is None and c.end_week is None,
                "roi_true_window": roi_true,
                "point_estimate": float(point),
                "standard_error": float(se),
                "bias_pct": c.bias_pct,
                "se_pct": c.se_pct,
                "prior_mu": mu,
                "prior_sigma": sigma,
            }
        )

    calibration = build_calibration_priors(results, paid_channel_names, n_times)
    return {"results": results, "calibration": calibration}


def build_calibration_priors(
    results: list[dict],
    paid_channel_names: list[str],
    n_times: int,
    default_mu: float = 0.2,
    default_sigma: float = 0.9,
) -> dict:
    """Assemble Meridian-ready calibration inputs from experiment results.

    Channels without an experiment keep Meridian's default ROI prior
    (LogNormal(0.2, 0.9)). When any experiment is date-restricted, a
    ``roi_calibration_period`` mask is built: True on the experiment window
    for calibrated channels, True everywhere for channels calibrated over the
    full duration or not calibrated at all (Meridian's convention: the prior
    applies to the masked period).

    If a channel has multiple experiments, the most precise (smallest
    standard error) wins.

    Returns:
        Dict with:
          ``roi_mu``     – (n_channels,) LogNormal mu per channel.
          ``roi_sigma``  – (n_channels,) LogNormal sigma per channel.
          ``calibrated`` – (n_channels,) bool, experiment-informed or default.
          ``roi_calibration_period`` – (n_times, n_channels) bool mask, or
                            None when every experiment is full-duration.
          ``channel_names`` – channel order for all arrays.

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
    """
    n_ch = len(paid_channel_names)
    roi_mu = np.full(n_ch, default_mu, dtype=np.float64)
    roi_sigma = np.full(n_ch, default_sigma, dtype=np.float64)
    calibrated = np.zeros(n_ch, dtype=bool)

    best_se = np.full(n_ch, np.inf)
    windowed = False
    mask = np.ones((n_times, n_ch), dtype=bool)

    for r in results:
        m = r["channel_index"]
        if r["standard_error"] < best_se[m]:
            best_se[m] = r["standard_error"]
            roi_mu[m] = r["prior_mu"]
            roi_sigma[m] = r["prior_sigma"]
            calibrated[m] = True
            if not r["full_duration"]:
                windowed = True
                mask[:, m] = False
                mask[r["start_week"]: r["end_week"] + 1, m] = True
            else:
                mask[:, m] = True

    return {
        "roi_mu": roi_mu,
        "roi_sigma": roi_sigma,
        "calibrated": calibrated,
        "roi_calibration_period": mask if windowed else None,
        "channel_names": list(paid_channel_names),
    }
