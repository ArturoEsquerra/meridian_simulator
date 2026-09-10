"""Simulate organic media channels (impression-based and reach-and-frequency).

Organic channels influence KPI but have no associated spend.  They are
structurally identical to paid media channels, minus cost simulation.
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from meridian_simulator.config import OrganicMediaChannelConfig, OrganicRFChannelConfig
from meridian_simulator.utils import as_float
from meridian_simulator.media import (
    simulate_impressions,
    simulate_reach_frequency,
    transform_media,
    _apply_adstock_hill,
)


def simulate_organic_media(
    media_cfgs: list[OrganicMediaChannelConfig],
    rf_cfgs: list[OrganicRFChannelConfig],
    p_g: tf.Tensor,
    n_times: int,
    prior,  # PriorDistribution broadcast object (for sampling missing params)
) -> dict:
    """Simulate organic impression-based and R&F channels.

    Returns dict with:
      ``organic_impression_gtm``   – raw impressions, (n_geos, n_times, n_om)
      ``organic_reach_gtm``        – reach for R&F channels, (n_geos, n_times, n_orf)
      ``organic_freq_gtm``         – frequency, (n_geos, n_times, n_orf)
      ``organic_media_transformed``– HillAdstock output, (n_geos, n_times, n_om)
      ``organic_rf_transformed``   – HillAdstock output, (n_geos, n_times, n_orf)
      ``beta_om``                  – hierarchical means, (n_om,)
      ``eta_om``                   – hierarchical stds, (n_om,)
      ``beta_gom``                 – geo-level coefficients, (n_geos, n_om)
      ``beta_orf``                 – hierarchical means for R&F, (n_orf,)
      ``eta_orf``                  – hierarchical stds, (n_orf,)
      ``beta_gorf``                – geo-level coefficients, (n_geos, n_orf)
      ``alpha_om``, ``ec_om``, ``slope_om``  – impression adstock/Hill params
      ``alpha_orf``, ``ec_orf``, ``slope_orf``– R&F adstock/Hill params
    """
    n_geos = int(p_g.shape[0])
    n_om = len(media_cfgs)
    n_orf = len(rf_cfgs)

    # ---- Impression-based organic channels ----------------------------------
    if n_om > 0:
        imp_sim = simulate_impressions(media_cfgs, n_geos, n_times, p_g)
        org_imp_gtm = imp_sim["impression_gtm"]

        tr = transform_media(org_imp_gtm, p_g)
        trans_oipc = tr.get(
            "transformed_ipc_gtm",
            tf.zeros((n_geos, n_times, 0), dtype=tf.float32),
        )

        alpha_om = (
            tf.constant([c.alpha for c in media_cfgs], dtype=tf.float32)
            if all(c.alpha is not None for c in media_cfgs)
            else as_float(prior.alpha_om.sample())
        )
        ec_om = (
            tf.constant([c.ec for c in media_cfgs], dtype=tf.float32)
            if all(c.ec is not None for c in media_cfgs)
            else as_float(prior.ec_om.sample())
        )
        slope_om = (
            tf.constant([c.slope for c in media_cfgs], dtype=tf.float32)
            if all(c.slope is not None for c in media_cfgs)
            else as_float(prior.slope_om.sample())
        )
        max_lag_om = max((c.max_lag for c in media_cfgs), default=8)
        org_media_transformed = _apply_adstock_hill(
            trans_oipc, alpha_om, ec_om, slope_om, max_lag_om, n_times
        )

        beta_om_vals = [
            float(
                tfp.distributions.Normal(c.beta_om_mean, 0.1).sample().numpy()
            )
            for c in media_cfgs
        ]
        eta_om_vals = [
            float(
                tfp.distributions.HalfNormal(c.beta_om_std).sample().numpy()
            )
            for c in media_cfgs
        ]
        beta_om = tf.constant(beta_om_vals, dtype=tf.float32)
        eta_om = tf.constant(eta_om_vals, dtype=tf.float32)
        dev_om = tfp.distributions.Normal(0.0, 1.0).sample([n_geos, n_om])
        beta_gom = tf.exp(beta_om + eta_om * dev_om)
    else:
        org_imp_gtm = tf.zeros((n_geos, n_times, 0), dtype=tf.float32)
        trans_oipc = org_imp_gtm
        org_media_transformed = org_imp_gtm
        alpha_om = ec_om = slope_om = tf.zeros((0,))
        beta_om = eta_om = tf.zeros((0,))
        beta_gom = tf.zeros((n_geos, 0))

    # ---- Organic R&F channels -----------------------------------------------
    if n_orf > 0:
        orf_sim = simulate_impressions(rf_cfgs, n_geos, n_times, p_g)
        orf_imp_gtm = orf_sim["impression_gtm"]
        orf_ipc_gtm = orf_sim["ipc_gtm"]

        orf_derived = simulate_reach_frequency(orf_ipc_gtm, orf_imp_gtm, p_g)
        org_reach_gtm = orf_derived["reach_gtm"]
        org_freq_gtm = orf_derived["freq_gtm"]

        tr_orf = transform_media(tf.zeros((n_geos, n_times, 0)), p_g, org_reach_gtm)
        trans_orpc = tr_orf.get(
            "transformed_rpc_gtm",
            tf.zeros((n_geos, n_times, 0), dtype=tf.float32),
        )

        alpha_orf = (
            tf.constant([c.alpha for c in rf_cfgs], dtype=tf.float32)
            if all(c.alpha is not None for c in rf_cfgs)
            else as_float(prior.alpha_orf.sample())
        )
        ec_orf = (
            tf.constant([c.ec for c in rf_cfgs], dtype=tf.float32)
            if all(c.ec is not None for c in rf_cfgs)
            else as_float(prior.ec_orf.sample())
        )
        slope_orf = (
            tf.constant([c.slope for c in rf_cfgs], dtype=tf.float32)
            if all(c.slope is not None for c in rf_cfgs)
            else as_float(prior.slope_orf.sample())
        )
        max_lag_orf = max((c.max_lag for c in rf_cfgs), default=8)
        org_rf_transformed = _apply_adstock_hill(
            trans_orpc, alpha_orf, ec_orf, slope_orf, max_lag_orf, n_times,
            is_rf=True, freq_gtm=org_freq_gtm,
        )

        beta_orf_vals = [
            float(
                tfp.distributions.Normal(c.beta_orf_mean, 0.1).sample().numpy()
            )
            for c in rf_cfgs
        ]
        eta_orf_vals = [
            float(
                tfp.distributions.HalfNormal(c.beta_orf_std).sample().numpy()
            )
            for c in rf_cfgs
        ]
        beta_orf_t = tf.constant(beta_orf_vals, dtype=tf.float32)
        eta_orf_t = tf.constant(eta_orf_vals, dtype=tf.float32)
        dev_orf = tfp.distributions.Normal(0.0, 1.0).sample([n_geos, n_orf])
        beta_gorf = tf.exp(beta_orf_t + eta_orf_t * dev_orf)
    else:
        org_reach_gtm = tf.zeros((n_geos, n_times, 0), dtype=tf.float32)
        org_freq_gtm = org_reach_gtm
        org_rf_transformed = org_reach_gtm
        alpha_orf = ec_orf = slope_orf = tf.zeros((0,))
        beta_orf_t = eta_orf_t = tf.zeros((0,))
        beta_gorf = tf.zeros((n_geos, 0))

    return {
        "organic_impression_gtm": org_imp_gtm,
        "organic_reach_gtm": org_reach_gtm,
        "organic_freq_gtm": org_freq_gtm,
        "organic_media_transformed": org_media_transformed,
        "organic_rf_transformed": org_rf_transformed,
        "alpha_om": alpha_om,
        "ec_om": ec_om,
        "slope_om": slope_om,
        "alpha_orf": alpha_orf,
        "ec_orf": ec_orf,
        "slope_orf": slope_orf,
        "beta_om": beta_om,
        "eta_om": eta_om,
        "beta_gom": beta_gom,
        "beta_orf": beta_orf_t,
        "eta_orf": eta_orf_t,
        "beta_gorf": beta_gorf,
    }
