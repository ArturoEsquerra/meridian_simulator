"""Simulate paid media channels (impression-based and reach-and-frequency).

For each channel the simulation proceeds in these steps:

  1. Draw raw impressions using a three-level random-effects model identical
     to Meridian's demo notebook:
         ipc_gtm = max(u_m + u_tm + u_gtm, 0)   [impressions per capita]
         impression_gtm = round(ipc_gtm * p_gtm)

  2. For R&F channels, derive reach and frequency from impressions via the
     Poisson arrival model from the demo notebook.

  3. Transform impressions / reach through Meridian's MediaTransformer
     (population-scaling and standardisation).

  4. Apply the HillAdstock transformation using either user-specified or
     prior-sampled alpha / ec / slope parameters.

  5. Simulate media coefficients (beta_gm / beta_grf).  When a target_roi is
     provided the hierarchical mean beta_m is back-solved so that E[ROI] equals
     the target.  Otherwise beta_m is drawn from a Normal(beta_m_mean, ...).

  6. Simulate spend from impressions and CPM.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from meridian_simulator.config import MediaChannelConfig, RFChannelConfig
from meridian_simulator.utils import as_float


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _sample_adstock_hill_params(prior, n_channels: int):
    """Sample alpha, ec, slope from Meridian's default prior or return None."""
    # Meridian's prior samples follow its backend's float width; coerce so the
    # simulator's float32 graph stays consistent (see utils.as_float).
    alpha = as_float(prior.alpha_m.sample()) if n_channels > 0 else None
    ec = as_float(prior.ec_m.sample()) if n_channels > 0 else None
    slope = as_float(prior.slope_m.sample()) if n_channels > 0 else None
    return alpha, ec, slope


def _apply_adstock_hill(
    transformed_media,    # (n_geos, n_times, n_channels)
    alpha,                # (n_channels,)
    ec,                   # (n_channels,)
    slope,                # (n_channels,)
    max_lag: int,
    n_times: int,
    is_rf: bool = False,
    freq_gtm=None,        # only for RF
):
    """Apply HillAdstock to transformed media tensor.

    For impression channels: Adstock first, then Hill.
    For RF channels: Hill on frequency first, then Adstock on reach.
    """
    from meridian.model import adstock_hill  # type: ignore

    hill_tr = adstock_hill.HillTransformer(ec=ec, slope=slope)
    adstock_tr = adstock_hill.AdstockTransformer(
        alpha=alpha, max_lag=max_lag, n_times_output=n_times
    )

    if not is_rf:
        media_out = hill_tr.forward(adstock_tr.forward(as_float(transformed_media)))
    else:
        # RF: hill applied to frequency, adstock to reach × adjusted_frequency
        adj_freq = hill_tr.forward(as_float(freq_gtm))
        media_out = adstock_tr.forward(as_float(transformed_media) * as_float(adj_freq))

    # Meridian's transformers return tensors in its backend's float width.
    return as_float(media_out)


def _solve_beta_m_for_roi(
    target_roi: float,
    p_g: tf.Tensor,           # (n_geos,)
    unit_value: tf.Tensor,    # (n_geos, n_times)
    media_transformed: tf.Tensor,  # (n_geos, n_times)  – single channel
    cost_m: tf.Tensor,        # scalar – total spend for channel
    eta_m: float,             # geo-level std for LogNormal
) -> float:
    """Back-solve the hierarchical mean beta_m so that E[ROI] == target_roi.

    ROI_m = Σ_{g,t}(p_g * unit_value_gt * media_tf_gtm * beta_gm_g) / C_m

    Since beta_gm_g ~ LogNormal(beta_m, eta_m):
        E[beta_gm_g] = exp(beta_m + eta_m² / 2)

    So:
        E[ROI_m] = exp(beta_m + eta_m²/2) * eff_m / C_m

    where eff_m = Σ_{g,t}(p_g * unit_value_gt * media_tf_gtm).

    Solving for beta_m:
        beta_m = log(target_roi * C_m / eff_m) - eta_m² / 2
    """
    eff_m = tf.reduce_sum(
        p_g[:, tf.newaxis] * unit_value * media_transformed
    ).numpy()
    c_m = float(cost_m.numpy())

    if eff_m <= 0 or c_m <= 0:
        raise ValueError(
            "Cannot back-solve beta_m: media efficiency or spend is zero."
        )

    expected_beta_gm = target_roi * c_m / eff_m
    if expected_beta_gm <= 0:
        raise ValueError(
            f"Target ROI {target_roi} implies a non-positive beta. "
            "Check that impressions, unit_value, and spend are consistent."
        )

    beta_m = np.log(expected_beta_gm) - 0.5 * eta_m ** 2
    return float(beta_m)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def simulate_impressions(
    channel_cfgs: list[MediaChannelConfig] | list[RFChannelConfig],
    n_geos: int,
    n_times: int,
    p_g: tf.Tensor,  # (n_geos,)
    seasonality_t: Optional[tf.Tensor] = None,  # (n_times,), optional
) -> dict:
    """Simulate raw impressions for all channels (paid or organic).

    Args:
        seasonality_t: Optional baseline seasonality wave, shape (n_times,).
            Channels with ``seasonal_flighting > 0`` have their weekly
            activity mean modulated by this wave (normalized to unit peak),
            simulating demand-synchronized media buying.  ``None`` or a
            zero flighting value reproduces the historical iid behavior.

    Returns dict with:
      ``ipc_gtm``       – impressions per capita  (n_geos, n_times, n_ch)
      ``impression_gtm``– total impressions       (n_geos, n_times, n_ch)
    """
    n_ch = len(channel_cfgs)
    if n_ch == 0:
        shape = (n_geos, n_times, 0)
        return {
            "ipc_gtm": tf.zeros(shape),
            "impression_gtm": tf.zeros(shape),
        }

    p_gtm = tf.reshape(p_g, (-1, 1, 1))

    # Broadcast channel-level means from per-channel config
    u_m_vals = [c.impression_mean_channel for c in channel_cfgs]
    u_tm_mean = [c.impression_mean_time for c in channel_cfgs]
    imp_std = [c.impression_std for c in channel_cfgs]

    u_m = tf.constant(u_m_vals, dtype=tf.float32)   # (n_ch,)

    # Normalized seasonality wave for demand-synchronized flighting
    s_norm = None
    if seasonality_t is not None:
        s = tf.cast(seasonality_t, tf.float32)
        peak = tf.reduce_max(tf.abs(s))
        if float(peak.numpy()) > 0:
            s_norm = s / peak  # (n_times,) in [-1, 1]

    # Time random effects — shape (n_times, n_ch)
    u_tm_parts = []
    for i, c in enumerate(channel_cfgs):
        u_tm_i = tfp.distributions.Normal(
            c.impression_mean_time, c.impression_std
        ).sample([n_times])
        flight = getattr(c, "seasonal_flighting", 0.0)
        if s_norm is not None and flight > 0.0:
            # Modulate the weekly activity mean with the demand wave:
            # planners buy INTO high season (media-seasonality confounding).
            u_tm_i = u_tm_i + c.impression_mean_time * flight * s_norm
        u_tm_parts.append(u_tm_i)
    u_tm = tf.stack(u_tm_parts, axis=-1)  # (n_times, n_ch)

    # Geo-time random effects — shape (n_geos, n_times, n_ch)
    u_gtm_parts = []
    for c in channel_cfgs:
        u_gtm_parts.append(
            tfp.distributions.Normal(0.0, c.impression_std)
            .sample([n_geos, n_times])
        )
    u_gtm = tf.stack(u_gtm_parts, axis=-1)  # (n_geos, n_times, n_ch)

    ipc_gtm = tf.maximum(u_m + u_tm[tf.newaxis, :, :] + u_gtm, 0.0)
    impression_gtm = tf.round(ipc_gtm * p_gtm)

    return {"ipc_gtm": ipc_gtm, "impression_gtm": impression_gtm}


def simulate_reach_frequency(
    ipc_gtm: tf.Tensor,         # (n_geos, n_times, n_rf_ch)
    impression_gtm: tf.Tensor,  # (n_geos, n_times, n_rf_ch)
    p_g: tf.Tensor,             # (n_geos,)
) -> dict:
    """Derive reach and frequency from impressions via Poisson arrival model.

    Returns dict with keys ``reach_gtm`` and ``freq_gtm``.
    """
    p_gtm = tf.reshape(p_g, (-1, 1, 1))

    reach_mean = p_gtm * (1.0 - tf.math.exp(-ipc_gtm))
    reach_scale = tf.math.sqrt(
        p_gtm * (tf.math.exp(-ipc_gtm) - tf.math.exp(-2.0 * ipc_gtm))
    )
    reach_gtm = tf.round(
        tfp.distributions.Normal(reach_mean, reach_scale).sample()
    )
    reach_gtm = tf.maximum(reach_gtm, 0.0)

    freq_gtm = tf.where(
        impression_gtm == 0,
        tf.zeros_like(impression_gtm),
        impression_gtm / tf.maximum(reach_gtm, 1.0),
    )

    return {"reach_gtm": reach_gtm, "freq_gtm": freq_gtm}


def transform_media(
    impression_gtm: tf.Tensor,
    p_g: tf.Tensor,
    reach_gtm: Optional[tf.Tensor] = None,
) -> dict:
    """Apply Meridian's MediaTransformer (population-scale + standardise).

    Returns dict with ``transformed_ipc_gtm`` (and optionally ``transformed_rpc_gtm``).
    """
    from meridian.model import transformers as meridian_tr  # type: ignore

    result = {}

    if impression_gtm.shape[-1] > 0:
        imp_tr = meridian_tr.MediaTransformer(media=impression_gtm, population=p_g)
        result["transformed_ipc_gtm"] = as_float(imp_tr.forward(impression_gtm))

    if reach_gtm is not None and reach_gtm.shape[-1] > 0:
        reach_tr = meridian_tr.MediaTransformer(media=reach_gtm, population=p_g)
        result["transformed_rpc_gtm"] = as_float(reach_tr.forward(reach_gtm))

    return result


def simulate_paid_media(
    media_cfgs: list[MediaChannelConfig],
    rf_cfgs: list[RFChannelConfig],
    p_g: tf.Tensor,
    unit_value: tf.Tensor,
    n_times: int,
    prior,  # PriorDistribution broadcast object
    seasonality_t: Optional[tf.Tensor] = None,  # (n_times,), optional
) -> dict:
    """Simulate all paid media channels end-to-end.

    Steps: impressions → R&F derivation → transform → HillAdstock →
           beta back-solve or draw → spend.

    Returns a rich dict with raw and transformed tensors, coefficients, spend,
    adstock/Hill parameters, and ROI diagnostics.
    """
    n_geos = int(p_g.shape[0])
    n_m = len(media_cfgs)
    n_rf = len(rf_cfgs)

    # ---- Impression-based channels ----------------------------------------
    imp_sim = simulate_impressions(
        media_cfgs, n_geos, n_times, p_g, seasonality_t=seasonality_t
    )
    imp_gtm = imp_sim["impression_gtm"]       # (n_geos, n_times, n_m)
    ipc_gtm = imp_sim["ipc_gtm"]

    # ---- RF channels ----------------------------------------------------------
    rf_sim = simulate_impressions(
        rf_cfgs, n_geos, n_times, p_g, seasonality_t=seasonality_t
    )
    rf_imp_gtm = rf_sim["impression_gtm"]     # (n_geos, n_times, n_rf)
    rf_ipc_gtm = rf_sim["ipc_gtm"]

    rf_derived = simulate_reach_frequency(rf_ipc_gtm, rf_imp_gtm, p_g)
    reach_gtm = rf_derived["reach_gtm"]       # (n_geos, n_times, n_rf)
    freq_gtm = rf_derived["freq_gtm"]

    # ---- Transform ----------------------------------------------------------
    tr = transform_media(imp_gtm, p_g, reach_gtm if n_rf > 0 else None)
    trans_ipc = tr.get(
        "transformed_ipc_gtm",
        tf.zeros((n_geos, n_times, 0), dtype=tf.float32),
    )
    trans_rpc = tr.get(
        "transformed_rpc_gtm",
        tf.zeros((n_geos, n_times, 0), dtype=tf.float32),
    )

    # ---- Adstock/Hill params ------------------------------------------------
    # Impression channels
    if n_m > 0:
        alpha_m = tf.constant(
            [c.alpha for c in media_cfgs], dtype=tf.float32
        ) if all(c.alpha is not None for c in media_cfgs) else as_float(prior.alpha_m.sample())

        ec_m = tf.constant(
            [c.ec for c in media_cfgs], dtype=tf.float32
        ) if all(c.ec is not None for c in media_cfgs) else as_float(prior.ec_m.sample())

        slope_m = tf.constant(
            [c.slope for c in media_cfgs], dtype=tf.float32
        ) if all(c.slope is not None for c in media_cfgs) else as_float(prior.slope_m.sample())

        # Fall back to prior for channels where only some params are set
        if any(c.alpha is None for c in media_cfgs):
            alpha_m_sampled = as_float(prior.alpha_m.sample())
            fixed_idx = [
                [i] for i, c in enumerate(media_cfgs) if c.alpha is not None
            ]
            if fixed_idx:
                alpha_m = tf.tensor_scatter_nd_update(
                    alpha_m_sampled,
                    fixed_idx,
                    [c.alpha for c in media_cfgs if c.alpha is not None],
                )
            else:
                alpha_m = alpha_m_sampled

        max_lag_m = max((c.max_lag for c in media_cfgs), default=8)
        media_transformed = _apply_adstock_hill(
            trans_ipc, alpha_m, ec_m, slope_m, max_lag_m, n_times
        )
    else:
        alpha_m = ec_m = slope_m = tf.zeros((0,))
        media_transformed = tf.zeros((n_geos, n_times, 0), dtype=tf.float32)

    # RF channels
    if n_rf > 0:
        alpha_rf = tf.constant(
            [c.alpha for c in rf_cfgs], dtype=tf.float32
        ) if all(c.alpha is not None for c in rf_cfgs) else as_float(prior.alpha_rf.sample())

        ec_rf = tf.constant(
            [c.ec for c in rf_cfgs], dtype=tf.float32
        ) if all(c.ec is not None for c in rf_cfgs) else as_float(prior.ec_rf.sample())

        slope_rf = tf.constant(
            [c.slope for c in rf_cfgs], dtype=tf.float32
        ) if all(c.slope is not None for c in rf_cfgs) else as_float(prior.slope_rf.sample())

        max_lag_rf = max((c.max_lag for c in rf_cfgs), default=8)
        rf_transformed = _apply_adstock_hill(
            trans_rpc, alpha_rf, ec_rf, slope_rf, max_lag_rf, n_times,
            is_rf=True, freq_gtm=freq_gtm,
        )
    else:
        alpha_rf = ec_rf = slope_rf = tf.zeros((0,))
        rf_transformed = tf.zeros((n_geos, n_times, 0), dtype=tf.float32)

    all_media_transformed = tf.concat(
        [media_transformed, rf_transformed], axis=-1
    )  # (n_geos, n_times, n_m + n_rf)

    # ---- Spend --------------------------------------------------------------
    all_cfgs = list(media_cfgs) + list(rf_cfgs)
    all_imp = tf.concat([imp_gtm, rf_imp_gtm], axis=-1)

    cpm_vals = tf.constant(
        [
            tfp.distributions.Uniform(c.cpm_low, c.cpm_high).sample().numpy()
            for c in all_cfgs
        ],
        dtype=tf.float32,
    )  # (n_total,)
    cost_gtm = all_imp * cpm_vals[tf.newaxis, tf.newaxis, :]

    # ---- Beta coefficients --------------------------------------------------
    all_media_cfgs_merged = list(media_cfgs) + list(rf_cfgs)
    n_total = n_m + n_rf

    beta_m_vals = []    # hierarchical means
    eta_m_vals = []     # geo-level stds

    for i, c in enumerate(all_media_cfgs_merged):
        eta_i = float(
            tfp.distributions.HalfNormal(
                getattr(c, "beta_m_std", getattr(c, "beta_rf_std", 0.1))
            ).sample().numpy()
        )
        eta_m_vals.append(eta_i)

        if c.target_roi is not None:
            # Back-solve beta_m from target ROI
            total_cost_ch = tf.reduce_sum(cost_gtm[..., i])
            beta_m_i = _solve_beta_m_for_roi(
                target_roi=c.target_roi,
                p_g=p_g,
                unit_value=unit_value,
                media_transformed=all_media_transformed[..., i],
                cost_m=total_cost_ch,
                eta_m=eta_i,
            )
        else:
            mean_attr = getattr(c, "beta_m_mean", getattr(c, "beta_rf_mean", 0.9))
            beta_m_i = float(
                tfp.distributions.Normal(mean_attr, 0.1).sample().numpy()
            )
        beta_m_vals.append(beta_m_i)

    beta_m = tf.constant(beta_m_vals, dtype=tf.float32)   # (n_total,)
    eta_m = tf.constant(eta_m_vals, dtype=tf.float32)     # (n_total,)

    beta_gm_dev = tfp.distributions.Normal(0.0, 1.0).sample(
        [n_geos, n_total]
    )
    beta_gm = tf.exp(beta_m + eta_m * beta_gm_dev)  # (n_geos, n_total)

    # ---- Ground-truth ROI (on raw KPI per-capita scale) --------------------
    # roi_m = Σ_{g,t}(p_g * unit_value_gt * media_transformed_gtm * beta_gm_g)
    #         / Σ_{g,t}(cost_gtm)
    if n_total > 0:
        incr_rev_m = tf.einsum(
            "g,gt,gtm,gm->m", p_g, unit_value, all_media_transformed, beta_gm
        )
        total_cost_m = tf.reduce_sum(cost_gtm, axis=(0, 1))
        roi_m = incr_rev_m / tf.maximum(total_cost_m, 1e-9)
    else:
        roi_m = tf.zeros((0,))

    return {
        # Raw media
        "impression_gtm": imp_gtm,
        "ipc_gtm": ipc_gtm,
        "rf_impression_gtm": rf_imp_gtm,
        "reach_gtm": reach_gtm,
        "freq_gtm": freq_gtm,
        # Transformed
        "transformed_ipc_gtm": trans_ipc,
        "transformed_rpc_gtm": trans_rpc,
        "all_media_transformed": all_media_transformed,
        # Adstock/Hill
        "alpha_m": alpha_m,
        "ec_m": ec_m,
        "slope_m": slope_m,
        "alpha_rf": alpha_rf,
        "ec_rf": ec_rf,
        "slope_rf": slope_rf,
        # Coefficients
        "beta_m": beta_m[:n_m],
        "beta_rf": beta_m[n_m:],
        "eta_m": eta_m[:n_m],
        "eta_rf": eta_m[n_m:],
        "beta_gm": beta_gm[:, :n_m],
        "beta_grf": beta_gm[:, n_m:],
        # Spend
        "cost_gtm": cost_gtm,
        "cpm_m": cpm_vals,
        # ROI diagnostics
        "roi_m": roi_m[:n_m],
        "roi_rf": roi_m[n_m:],
    }
