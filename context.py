"""Simulate context (control) variables and non-media channels.

Context variables represent external factors that affect KPI but are not media
(e.g., GDP index, exchange rate, competitor spend, seasonality proxies).
Each variable can optionally follow an AR(1) process to simulate temporal
persistence, and can include a linear trend.

Non-media channels (e.g., promotions, price index, distribution score) are
treated analogously but fed into Meridian's ``non_media_channel`` slots.
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from meridian_simulator.config import ContextVariableConfig, NonMediaChannelConfig


# ---------------------------------------------------------------------------
# AR(1) helper
# ---------------------------------------------------------------------------


def _simulate_ar1_series(
    n_times: int,
    n_geos: int,
    mean: float,
    std: float,
    ar1_coef: float,
    trend: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Simulate an AR(1) geo × time series of shape (n_geos, n_times).

    Each geo shares the same AR(1) structure but draws independent realisations.

    Args:
        rng: A ``np.random.Generator`` (from ``np.random.default_rng``).
             Using the new-style Generator API ensures NumPy 2.x compatibility
             and makes seeding explicit rather than relying on global state.
    """
    rho = np.clip(ar1_coef, -0.99, 0.99)
    # Stationary variance of AR(1): Var(x_t) = std² / (1 - rho²)
    innov_std = std * np.sqrt(max(1.0 - rho ** 2, 1e-6))

    series = np.zeros((n_geos, n_times), dtype=np.float32)
    series[:, 0] = rng.normal(mean, std, size=n_geos).astype(np.float32)

    for t in range(1, n_times):
        series[:, t] = (
            rho * series[:, t - 1]
            + (1.0 - rho) * mean
            + rng.normal(0.0, innov_std, size=n_geos).astype(np.float32)
            + trend * t / n_times
        )

    return series


# ---------------------------------------------------------------------------
# Context variables
# ---------------------------------------------------------------------------


def simulate_context_variables(
    cfgs: list[ContextVariableConfig],
    n_geos: int,
    n_times: int,
    p_g: tf.Tensor,
    rng: np.random.Generator,
) -> dict:
    """Simulate context (control) variables.

    Returns dict with:
      ``context_gtc``             – raw series, shape (n_geos, n_times, n_c)
      ``transformed_context_gtc`` – standardised series (as Meridian expects)
      ``gamma_c``                 – national-level control coefficients, (n_c,)
      ``xi_c``                    – geo-level deviation std, (n_c,)
      ``gamma_gc``                – geo-level coefficients, (n_geos, n_c)
    """
    n_c = len(cfgs)

    if n_c == 0:
        zeros = tf.zeros((n_geos, n_times, 0), dtype=tf.float32)
        return {
            "context_gtc": zeros,
            "transformed_context_gtc": zeros,
            "gamma_c": tf.zeros((0,)),
            "xi_c": tf.zeros((0,)),
            "gamma_gc": tf.zeros((n_geos, 0)),
        }

    # --- Raw series ---------------------------------------------------------
    raw_parts = []
    for c in cfgs:
        raw_parts.append(
            _simulate_ar1_series(
                n_times, n_geos, c.series_mean, c.series_std,
                c.ar1_coef, c.trend, rng,
            )
        )
    context_gtc = tf.constant(
        np.stack(raw_parts, axis=-1), dtype=tf.float32
    )  # (n_geos, n_times, n_c)

    # --- Standardise (Meridian's CenteringAndScalingTransformer) ------------
    from meridian.model import transformers as meridian_tr  # type: ignore

    ctx_transformer = meridian_tr.CenteringAndScalingTransformer(
        tensor=context_gtc, population=p_g, population_scaling_id=None
    )
    transformed_context_gtc = ctx_transformer.forward(context_gtc)

    # --- Coefficients --------------------------------------------------------
    gamma_c_vals = []
    xi_c_vals = []
    gamma_gc_parts = []

    for c in cfgs:
        gamma_ci = float(
            tfp.distributions.Normal(c.gamma_c_mean, c.gamma_c_std).sample().numpy()
        )
        xi_ci = float(
            tfp.distributions.HalfNormal(c.xi_c_std).sample().numpy()
        )
        dev = tfp.distributions.Normal(0.0, 1.0).sample(n_geos)
        gamma_gci = gamma_ci + xi_ci * dev   # (n_geos,)

        gamma_c_vals.append(gamma_ci)
        xi_c_vals.append(xi_ci)
        gamma_gc_parts.append(gamma_gci)

    gamma_c = tf.constant(gamma_c_vals, dtype=tf.float32)
    xi_c = tf.constant(xi_c_vals, dtype=tf.float32)
    gamma_gc = tf.stack(gamma_gc_parts, axis=-1)  # (n_geos, n_c)

    return {
        "context_gtc": context_gtc,
        "transformed_context_gtc": transformed_context_gtc,
        "gamma_c": gamma_c,
        "xi_c": xi_c,
        "gamma_gc": gamma_gc,
    }


# ---------------------------------------------------------------------------
# Non-media channels
# ---------------------------------------------------------------------------


def simulate_non_media_channels(
    cfgs: list[NonMediaChannelConfig],
    n_geos: int,
    n_times: int,
    rng: np.random.Generator,
) -> dict:
    """Simulate non-media time-series channels.

    Non-media channels are standardised geo-time series (e.g., promotions,
    price reductions) that enter the KPI equation with their own coefficients.

    Returns dict with:
      ``non_media_gtc``             – raw series, (n_geos, n_times, n_n)
      ``transformed_non_media_gtc`` – mean-centred unit-std series
      ``gamma_n``                   – national-level coefficients, (n_n,)
      ``xi_n``                      – geo-level deviation std, (n_n,)
      ``gamma_gn``                  – geo-level coefficients, (n_geos, n_n)
    """
    n_n = len(cfgs)

    if n_n == 0:
        zeros = tf.zeros((n_geos, n_times, 0), dtype=tf.float32)
        return {
            "non_media_gtc": zeros,
            "transformed_non_media_gtc": zeros,
            "gamma_n": tf.zeros((0,)),
            "xi_n": tf.zeros((0,)),
            "gamma_gn": tf.zeros((n_geos, 0)),
        }

    raw_parts = []
    for c in cfgs:
        raw = rng.normal(
            c.series_mean, c.series_std, size=(n_geos, n_times)
        ).astype(np.float32)
        raw_parts.append(raw)

    non_media_gtc = tf.constant(
        np.stack(raw_parts, axis=-1), dtype=tf.float32
    )

    # Standardise channel-by-channel (zero mean, unit std across geos × times)
    mean_n = tf.reduce_mean(non_media_gtc, axis=(0, 1), keepdims=True)
    std_n = tf.math.reduce_std(non_media_gtc, axis=(0, 1), keepdims=True)
    transformed_non_media_gtc = (non_media_gtc - mean_n) / tf.maximum(std_n, 1e-9)

    gamma_n_vals = []
    xi_n_vals = []
    gamma_gn_parts = []

    for c in cfgs:
        gamma_ni = float(
            tfp.distributions.Normal(c.gamma_n_mean, c.gamma_n_std).sample().numpy()
        )
        xi_ni = float(
            tfp.distributions.HalfNormal(c.xi_n_std).sample().numpy()
        )
        dev = tfp.distributions.Normal(0.0, 1.0).sample(n_geos)
        gamma_gni = gamma_ni + xi_ni * dev

        gamma_n_vals.append(gamma_ni)
        xi_n_vals.append(xi_ni)
        gamma_gn_parts.append(gamma_gni)

    gamma_n = tf.constant(gamma_n_vals, dtype=tf.float32)
    xi_n = tf.constant(xi_n_vals, dtype=tf.float32)
    gamma_gn = tf.stack(gamma_gn_parts, axis=-1)  # (n_geos, n_n)

    return {
        "non_media_gtc": non_media_gtc,
        "transformed_non_media_gtc": transformed_non_media_gtc,
        "gamma_n": gamma_n,
        "xi_n": xi_n,
        "gamma_gn": gamma_gn,
    }
