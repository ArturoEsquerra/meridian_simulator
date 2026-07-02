"""Simulate baseline KPI components: geo intercepts, trend, seasonality, noise.

The baseline per-capita KPI at geo g and time t is:

    baseline_gt = tau_g + mu_t + trend_t + seasonality_t + epsilon_gt

where:
  - tau_g    ~ N(tau_mean, tau_std)              [geo-level intercept]
  - mu_t                                          [smooth time-varying baseline
                                                   parameterized via B-spline
                                                   knots; n_knots controls
                                                   smoothness]
  - trend_t  = trend_slope * t / n_times         [linear drift]
  - seasonality_t = Σ_k amp_k * sin(2π*t/T_k + φ_k)
  - epsilon_gt ~ N(0, noise_std)                 [iid observation noise]
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from meridian_simulator.config import BaselineConfig, SeasonalityComponent


def _build_seasonality(
    n_times: int,
    components: list[SeasonalityComponent],
) -> np.ndarray:
    """Return a 1-D array of shape (n_times,) with stacked seasonal waves."""
    t = np.arange(n_times, dtype=np.float32)
    seasonal = np.zeros(n_times, dtype=np.float32)
    for c in components:
        phase_rad = 2.0 * np.pi * c.phase_weeks / c.period_weeks
        seasonal += c.amplitude * np.sin(
            2.0 * np.pi * t / c.period_weeks + phase_rad
        )
    return seasonal


def _build_mu_t(
    n_times: int,
    n_knots: int,
    knot_std: float,
) -> tf.Tensor:
    """Return time-varying baseline mu_t via Meridian's knot interpolation.

    When n_knots == n_times this reduces to sampling mu_t directly from
    N(0, knot_std).  Fewer knots produce a smoother curve.

    Args:
        n_times: Number of time periods.
        n_knots: Number of B-spline knots.
        knot_std: Standard deviation of the Normal prior on knot values.

    Returns:
        Tensor of shape (n_times,).
    """
    # Import here to avoid import-time GPU requirement failures if Meridian
    # is not installed; callers handle ImportError gracefully.
    from meridian.model import knots as meridian_knots  # type: ignore

    knots_k = tfp.distributions.Normal(0.0, knot_std).sample(n_knots)
    knot_info = meridian_knots.get_knot_info(n_times, n_knots, False)
    weights = tf.convert_to_tensor(knot_info.weights)  # (n_knots, n_times)
    mu_t = tf.einsum("k,kt->t", knots_k, weights)
    return mu_t


def simulate_baseline(
    cfg: BaselineConfig,
    n_times: int,
    n_geos: int,
) -> dict:
    """Simulate all baseline components.

    Args:
        cfg: BaselineConfig instance.
        n_times: Number of time periods.
        n_geos: Number of geos.

    Returns:
        Dict with keys:
          ``tau_g``          – shape (n_geos,)
          ``mu_t``           – shape (n_times,)
          ``trend_t``        – shape (n_times,)
          ``seasonality_t``  – shape (n_times,)
          ``eps_gt``         – shape (n_geos, n_times)
          ``baseline_gt``    – shape (n_geos, n_times)  [sum of all components]
    """
    n_knots = cfg.n_knots if cfg.n_knots is not None else n_times

    tau_g = tfp.distributions.Normal(cfg.tau_mean, cfg.tau_std).sample(n_geos)

    mu_t = _build_mu_t(n_times, n_knots, cfg.knot_std)

    t = tf.cast(tf.range(n_times), tf.float32)
    trend_t = cfg.trend_slope * t / tf.cast(n_times, tf.float32)

    seasonality_t = tf.constant(
        _build_seasonality(n_times, cfg.seasonality), dtype=tf.float32
    )

    sigma = cfg.noise_std
    eps_gt = tfp.distributions.Normal(0.0, sigma).sample([n_geos, n_times])

    baseline_gt = (
        tau_g[:, tf.newaxis]
        + mu_t[tf.newaxis, :]
        + trend_t[tf.newaxis, :]
        + seasonality_t[tf.newaxis, :]
        + eps_gt
    )

    return {
        "tau_g": tau_g,
        "mu_t": mu_t,
        "trend_t": trend_t,
        "seasonality_t": seasonality_t,
        "eps_gt": eps_gt,
        "baseline_gt": baseline_gt,
    }
