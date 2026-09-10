"""Main MeridianSimulator class.

Orchestrates all simulation sub-modules and exposes the final outputs as:
  - ``SimulationResult`` – a dataclass holding every tensor/DataFrame
  - ``ground_truth``      – a dict of ground-truth parameter values
  - Helper methods for saving to CSV/pickle and for plotting
"""

from __future__ import annotations

import dataclasses
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp

from meridian_simulator.augment import (
    apply_kpi_noise,
    apply_promo_events,
    simulate_collinear_variables,
    simulate_endogenous_variables,
)
from meridian_simulator.baseline import simulate_baseline
from meridian_simulator.experiments import simulate_experiments
from meridian_simulator.config import SimulationConfig
from meridian_simulator.context import (
    simulate_context_variables,
    simulate_non_media_channels,
)
from meridian_simulator.media import simulate_paid_media
from meridian_simulator.organic import simulate_organic_media
from meridian_simulator.output import (
    build_geo_dataframe,
    build_national_dataframe,
    build_xarray_dataset,
)
from meridian_simulator.utils import (
    check_meridian_compat,
    make_channel_names,
    make_geo_names,
    make_time_labels,
    to_numpy,
)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class SimulationResult:
    """Holds every output produced by the simulator.

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
    """

    config: SimulationConfig
    ground_truth: dict
    geo_df: pd.DataFrame
    national_df: pd.DataFrame
    xr_dict: dict
    kpi_gt: np.ndarray
    unit_value_gt: np.ndarray
    population_g: np.ndarray
    geo_names: list[str]
    time_names: list[str]
    channel_names: list[str]
    rf_channel_names: list[str]
    organic_channel_names: list[str]
    organic_rf_channel_names: list[str]
    non_media_channel_names: list[str]
    context_variable_names: list[str]
    collinear_variable_names: list[str] = dataclasses.field(default_factory=list)
    endogenous_variable_names: list[str] = dataclasses.field(default_factory=list)
    promo_event_names: list[str] = dataclasses.field(default_factory=list)

    def save(self, output_dir: str | Path = ".") -> None:
        """Save DataFrames to CSV and ground_truth dict to pickle.

        Args:
            output_dir: Directory where files will be written.
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        self.geo_df.to_csv(out / "geo_data.csv", index=False)
        self.national_df.to_csv(out / "national_data.csv", index=False)
        with open(out / "ground_truth.pkl", "wb") as f:
            pickle.dump(self.ground_truth, f)
        print(
            f"Saved geo_data.csv, national_data.csv, and ground_truth.pkl "
            f"to '{out}'."
        )

    def summary(self) -> None:
        """Print a concise summary of the simulation."""
        gt = self.ground_truth
        cfg = self.config
        lines = [
            "=" * 60,
            "Meridian Simulation Summary",
            "=" * 60,
            f"  Time periods  : {cfg.n_times}",
            f"  Geos          : {cfg.n_geos}",
            f"  Start date    : {cfg.start_date}",
            f"  Seed          : {cfg.seed}",
            "",
            f"  Paid imp. channels : {cfg.n_media_channels}",
            f"  Paid R&F channels  : {cfg.n_rf_channels}",
            f"  Organic imp. ch.   : {cfg.n_organic_media_channels}",
            f"  Organic R&F ch.    : {cfg.n_organic_rf_channels}",
            f"  Non-media channels : {cfg.n_non_media_channels}",
            f"  Context variables  : {cfg.n_context_variables}",
            f"  Collinear vars     : {cfg.n_collinear_variables}",
            f"  Endogenous vars    : {cfg.n_endogenous_variables}",
            f"  Promo events       : {cfg.n_promo_events}",
            f"  KPI noise (CV)     : {cfg.kpi_noise_pct:.3f}",
            "",
            "Ground-truth ROI (impression channels):",
        ]
        if "roi_m" in gt and len(gt["roi_m"]) > 0:
            for name, roi in zip(self.channel_names, gt["roi_m"]):
                lines.append(f"    {name:30s}: {roi:.4f}")
        else:
            lines.append("    (none)")

        lines.append("Ground-truth ROI (R&F channels):")
        if "roi_rf" in gt and len(gt["roi_rf"]) > 0:
            for name, roi in zip(self.rf_channel_names, gt["roi_rf"]):
                lines.append(f"    {name:30s}: {roi:.4f}")
        else:
            lines.append("    (none)")

        total_rev = float(np.sum(self.kpi_gt * self.unit_value_gt))
        lines.append("")
        lines.append(f"  Total revenue   : {total_rev / 1e6:.2f}M")
        if "total_spend" in gt:
            lines.append(f"  Total spend     : {gt['total_spend'] / 1e6:.2f}M")
        lines.append("=" * 60)
        print("\n".join(lines))

    def plot_kpi(self, figsize=(12, 4)) -> None:
        """Plot mean KPI across geos over time."""
        import matplotlib.pyplot as plt  # type: ignore

        kpi_mean = np.mean(self.kpi_gt, axis=0)
        plt.figure(figsize=figsize)
        plt.plot(self.time_names, kpi_mean, label="Mean KPI")
        plt.xticks(rotation=45, ha="right")
        plt.xlabel("Time")
        plt.ylabel("KPI (mean across geos)")
        plt.title("Simulated KPI")
        plt.tight_layout()
        plt.show()

    def plot_media_spend(self, figsize=(12, 4)) -> None:
        """Plot total spend per channel over time (stacked bar)."""
        import matplotlib.pyplot as plt  # type: ignore

        gt = self.ground_truth
        if "cost_gtm" not in gt:
            print("No spend data available.")
            return

        cost = gt["cost_gtm"]             # (n_geos, n_times, n_total)
        total_per_time = cost.sum(axis=0) # (n_times, n_total)
        all_names = self.channel_names + self.rf_channel_names

        fig, ax = plt.subplots(figsize=figsize)
        bottom = np.zeros(len(self.time_names))
        for i, name in enumerate(all_names):
            ax.bar(
                self.time_names,
                total_per_time[:, i],
                bottom=bottom,
                label=name,
                alpha=0.8,
            )
            bottom += total_per_time[:, i]

        ax.set_xticks([])
        ax.set_xlabel("Time")
        ax.set_ylabel("Total spend (all geos)")
        ax.set_title("Simulated Media Spend")
        ax.legend(loc="upper right")
        plt.tight_layout()
        plt.show()


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------


class MeridianSimulator:
    """Simulate a complete Meridian-compatible dataset from a config.

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
    """

    def __init__(self, config: SimulationConfig) -> None:
        self.cfg = config

    def _build_prior(self):
        """Build Meridian's PriorDistribution broadcast for prior sampling."""
        from meridian.model import prior_distribution  # type: ignore

        n_geos = self.cfg.n_geos
        n_m = self.cfg.n_media_channels
        n_rf = self.cfg.n_rf_channels
        n_om = self.cfg.n_organic_media_channels
        n_orf = self.cfg.n_organic_rf_channels
        n_non = self.cfg.n_non_media_channels
        n_ctx = self.cfg.n_context_variables

        # n_knots must be >= 1 for prior broadcast
        n_knots = max(
            self.cfg.baseline.n_knots if self.cfg.baseline.n_knots else self.cfg.n_times,
            1,
        )

        prior = prior_distribution.PriorDistribution.broadcast(
            prior_distribution.PriorDistribution(),
            n_geos=n_geos,
            n_media_channels=max(n_m, 1),   # broadcast needs >= 1
            n_controls=max(n_ctx, 1),
            n_rf_channels=max(n_rf, 1),
            unique_sigma_for_each_geo=False,
            n_knots=n_knots,
            is_national=self.cfg.is_national,
            n_organic_media_channels=max(n_om, 1),
            n_organic_rf_channels=max(n_orf, 1),
            n_non_media_channels=max(n_non, 1),
            set_total_media_contribution_prior=False,
            kpi=0,
            total_spend=0,
        )
        return prior

    def run(self) -> SimulationResult:
        """Execute the full simulation pipeline.

        Returns:
            A :class:`SimulationResult` containing all outputs.
        """
        cfg = self.cfg
        check_meridian_compat()   # warn early on untested Meridian majors
        tf.random.set_seed(cfg.seed)
        # Use the new-style Generator API (NumPy 2.x compatible).
        # np.random.default_rng is reproducible and does not mutate global state.
        rng = np.random.default_rng(cfg.seed)

        prior = self._build_prior()

        # ---- Names ----------------------------------------------------------
        geo_names = make_geo_names(cfg.n_geos)
        time_names = make_time_labels(cfg.n_times, cfg.start_date)
        channel_names = [c.name for c in cfg.media_channels] or make_channel_names(
            "channel", cfg.n_media_channels
        )
        rf_channel_names = [c.name for c in cfg.rf_channels] or make_channel_names(
            "rf_channel", cfg.n_rf_channels
        )
        organic_channel_names = [c.name for c in cfg.organic_media_channels]
        organic_rf_channel_names = [c.name for c in cfg.organic_rf_channels]
        non_media_channel_names = [c.name for c in cfg.non_media_channels]
        context_variable_names = [c.name for c in cfg.context_variables]

        # ---- Population & unit value ----------------------------------------
        population_g = tfp.distributions.Uniform(
            cfg.baseline.population_low, cfg.baseline.population_high
        ).sample(cfg.n_geos)

        unit_value_gt = tfp.distributions.Uniform(
            cfg.baseline.unit_value_low, cfg.baseline.unit_value_high
        ).sample((cfg.n_geos, cfg.n_times))

        # ---- Baseline -------------------------------------------------------
        base = simulate_baseline(cfg.baseline, cfg.n_times, cfg.n_geos)

        # ---- Context variables ----------------------------------------------
        ctx = simulate_context_variables(
            cfg.context_variables, cfg.n_geos, cfg.n_times, population_g, rng
        )

        # ---- Non-media channels ---------------------------------------------
        non_media = simulate_non_media_channels(
            cfg.non_media_channels, cfg.n_geos, cfg.n_times, rng
        )

        # ---- Paid media -------------------------------------------------------
        media = simulate_paid_media(
            cfg.media_channels,
            cfg.rf_channels,
            population_g,
            unit_value_gt,
            cfg.n_times,
            prior,
            seasonality_t=base["seasonality_t"],
        )

        # ---- Organic media --------------------------------------------------
        organic = simulate_organic_media(
            cfg.organic_media_channels,
            cfg.organic_rf_channels,
            population_g,
            cfg.n_times,
            prior,
        )

        # ---- KPI assembly ---------------------------------------------------
        # KPI per capita (raw scale) =
        #   baseline_gt
        #   + Σ_c γ_{g,c} * z_{g,t,c}          [context]
        #   + Σ_n γ_{g,n} * z_{g,t,n}          [non-media]
        #   + Σ_m β_{g,m} * hs_m(x_{g,t,m})    [paid media]
        #   + Σ_om β_{g,om} * hs_om(x_{g,t,om})[organic media]

        kpi_per_cap = tf.maximum(base["baseline_gt"], 0.0)

        if cfg.n_context_variables > 0:
            kpi_per_cap = kpi_per_cap + tf.einsum(
                "gtc,gc->gt",
                ctx["transformed_context_gtc"],
                ctx["gamma_gc"],
            )

        if cfg.n_non_media_channels > 0:
            kpi_per_cap = kpi_per_cap + tf.einsum(
                "gtc,gc->gt",
                non_media["transformed_non_media_gtc"],
                non_media["gamma_gn"],
            )

        # Snapshot the non-media-driven KPI portion (baseline + context +
        # non-media) BEFORE media contributions are added.  Used by promo
        # events configured with baseline_only=True.
        kpi_per_cap_non_media = kpi_per_cap

        n_total_paid = cfg.n_media_channels + cfg.n_rf_channels
        if n_total_paid > 0:
            beta_all = tf.concat(
                [media["beta_gm"], media["beta_grf"]], axis=-1
            )
            kpi_per_cap = kpi_per_cap + tf.einsum(
                "gtm,gm->gt",
                media["all_media_transformed"],
                beta_all,
            )

        n_total_organic = cfg.n_organic_media_channels + cfg.n_organic_rf_channels
        if n_total_organic > 0:
            beta_org = tf.concat(
                [organic["beta_gom"], organic["beta_gorf"]], axis=-1
            )
            org_transformed = tf.concat(
                [organic["organic_media_transformed"],
                 organic["organic_rf_transformed"]], axis=-1
            )
            kpi_per_cap = kpi_per_cap + tf.einsum(
                "gtm,gm->gt", org_transformed, beta_org
            )

        kpi_gt = kpi_per_cap * population_g[:, tf.newaxis]

        # ---- Promo events (structural KPI lifts) -----------------------------
        baseline_kpi_gt = to_numpy(
            kpi_per_cap_non_media * population_g[:, tf.newaxis]
        )
        promo = apply_promo_events(
            cfg.promo_events, to_numpy(kpi_gt), rng,
            baseline_kpi_gt=baseline_kpi_gt,
        )

        # ---- KPI observation noise -------------------------------------------
        noise = apply_kpi_noise(promo["kpi_gt"], cfg.kpi_noise_pct, rng)
        kpi_gt = tf.constant(noise["kpi_gt"], dtype=tf.float32)

        # ---- Endogenous distractor variables ---------------------------------
        endog = simulate_endogenous_variables(
            cfg.endogenous_variables,
            to_numpy(media["cost_gtm"]),
            channel_names + rf_channel_names,
            noise["kpi_gt"],
            rng,
        )

        # ---- Collinear distractor variables -----------------------------------
        collinear = simulate_collinear_variables(
            cfg.collinear_variables,
            to_numpy(ctx["context_gtc"]),
            context_variable_names,
            to_numpy(population_g),
            cfg.n_times,
            rng,
        )

        # ---- Incrementality experiments --------------------------------------
        # True per-(geo, time, channel) media contribution in KPI units,
        # computed on the pre-event KPI scale so experiment truths are
        # consistent with the recorded roi_m ground truth.
        if cfg.experiments:
            beta_all_np = to_numpy(
                tf.concat([media["beta_gm"], media["beta_grf"]], axis=-1)
            )  # (n_geos, n_paid)
            contribution_gtm = (
                to_numpy(media["all_media_transformed"])
                * beta_all_np[:, np.newaxis, :]
                * to_numpy(population_g)[:, np.newaxis, np.newaxis]
            )  # (n_geos, n_times, n_paid) in KPI units
            exp_sim = simulate_experiments(
                cfg.experiments,
                contribution_gtm,
                to_numpy(media["cost_gtm"]),
                to_numpy(unit_value_gt),
                channel_names + rf_channel_names,
                cfg.n_times,
                rng,
            )
        else:
            exp_sim = {"results": [], "calibration": None}

        collinear_variable_names = [c.name for c in cfg.collinear_variables]
        endogenous_variable_names = [c.name for c in cfg.endogenous_variables]
        promo_event_names = [c.name for c in cfg.promo_events]

        # ---- Ground-truth dict ----------------------------------------------
        # Scale parameters to match Meridian's transformed KPI scale
        from meridian.model import transformers as meridian_tr  # type: ignore

        kpi_transformer = meridian_tr.KpiTransformer(
            kpi=kpi_gt, population=population_g
        )
        kpi_mean = float(kpi_transformer.population_scaled_mean.numpy())
        kpi_std = float(kpi_transformer.population_scaled_stdev.numpy())

        gt: dict = {
            # Baseline
            "tau_g": to_numpy(base["tau_g"]),
            "mu_t": to_numpy(base["mu_t"]),
            "trend_t": to_numpy(base["trend_t"]),
            "seasonality_t": to_numpy(base["seasonality_t"]),
            "kpi_mean": kpi_mean,
            "kpi_std": kpi_std,
            # Intercept (geo+time combined, centred on transformed scale)
            "intercept_gt": (
                to_numpy(base["tau_g"][:, np.newaxis]
                         + base["mu_t"][np.newaxis, :]) - kpi_mean
            ) / kpi_std,
            # Context
            "gamma_c": to_numpy(ctx["gamma_c"]),
            "xi_c": to_numpy(ctx["xi_c"]),
            "gamma_gc": to_numpy(ctx["gamma_gc"]) / kpi_std,
            # Non-media
            "gamma_n": to_numpy(non_media["gamma_n"]),
            "xi_n": to_numpy(non_media["xi_n"]),
            "gamma_gn": to_numpy(non_media["gamma_gn"]) / kpi_std,
            # Paid media (transformed scale)
            "beta_m": to_numpy(media["beta_m"]) - np.log(kpi_std),
            "beta_rf": to_numpy(media["beta_rf"]) - np.log(kpi_std),
            "eta_m": to_numpy(media["eta_m"]),
            "eta_rf": to_numpy(media["eta_rf"]),
            "beta_gm": to_numpy(media["beta_gm"]) / kpi_std,
            "beta_grf": to_numpy(media["beta_grf"]) / kpi_std,
            # Adstock / Hill
            "alpha_m": to_numpy(media["alpha_m"]),
            "ec_m": to_numpy(media["ec_m"]),
            "slope_m": to_numpy(media["slope_m"]),
            "alpha_rf": to_numpy(media["alpha_rf"]),
            "ec_rf": to_numpy(media["ec_rf"]),
            "slope_rf": to_numpy(media["slope_rf"]),
            # ROI (raw scale, ground truth)
            "roi_m": to_numpy(media["roi_m"]),
            "roi_rf": to_numpy(media["roi_rf"]),
            # Organic media
            "beta_om": to_numpy(organic["beta_om"]) - np.log(kpi_std) if cfg.n_organic_media_channels > 0 else np.array([]),
            "eta_om": to_numpy(organic["eta_om"]),
            "beta_gom": to_numpy(organic["beta_gom"]) / kpi_std,
            "alpha_om": to_numpy(organic["alpha_om"]),
            "ec_om": to_numpy(organic["ec_om"]),
            "slope_om": to_numpy(organic["slope_om"]),
            "beta_orf": to_numpy(organic["beta_orf"]) - np.log(kpi_std) if cfg.n_organic_rf_channels > 0 else np.array([]),
            "eta_orf": to_numpy(organic["eta_orf"]),
            "beta_gorf": to_numpy(organic["beta_gorf"]) / kpi_std,
            # Spend (raw tensors for diagnostics)
            "cost_gtm": to_numpy(media["cost_gtm"]),
            "total_spend": float(tf.reduce_sum(media["cost_gtm"]).numpy()),
            # Promo events (structural lifts)
            "promo_events": promo["spec"],
            "promo_multiplier_gt": promo["multiplier_gt"],
            "promo_baseline_multiplier_gt": promo["baseline_multiplier_gt"],
            "promo_flags_tc": promo["flags_tc"],
            "baseline_kpi_gt": baseline_kpi_gt,
            # KPI observation noise
            "kpi_noise_pct": cfg.kpi_noise_pct,
            "kpi_noise_multiplier_gt": noise["multiplier_gt"],
            "kpi_noise_realized_cv": noise["realized_cv"],
            # Distractor variables (zero causal effect on KPI)
            "collinear_variables": collinear["spec"],
            "endogenous_variables": endog["spec"],
            # Incrementality experiments (simulated lift studies)
            "experiments": exp_sim["results"],
            "experiment_calibration": exp_sim["calibration"],
        }

        # ---- Build output DataFrames / xarrays ------------------------------
        n_m = cfg.n_media_channels
        n_rf = cfg.n_rf_channels

        xr_dict = build_xarray_dataset(
            geo_names=geo_names,
            time_names=time_names,
            media_channel_names=channel_names,
            rf_channel_names=rf_channel_names,
            organic_media_channel_names=organic_channel_names,
            organic_rf_channel_names=organic_rf_channel_names,
            non_media_channel_names=non_media_channel_names,
            context_variable_names=context_variable_names,
            kpi_gt=to_numpy(kpi_gt),
            unit_value_gt=to_numpy(unit_value_gt),
            population_g=to_numpy(population_g),
            impression_gtm=to_numpy(media["impression_gtm"]),
            cost_gtm=to_numpy(media["cost_gtm"]),
            reach_gtm=to_numpy(media["reach_gtm"]),
            freq_gtm=to_numpy(media["freq_gtm"]),
            organic_impression_gtm=to_numpy(organic["organic_impression_gtm"]),
            organic_reach_gtm=to_numpy(organic["organic_reach_gtm"]),
            organic_freq_gtm=to_numpy(organic["organic_freq_gtm"]),
            non_media_gtc=to_numpy(non_media["non_media_gtc"]),
            context_gtc=to_numpy(ctx["context_gtc"]),
        )

        geo_df = build_geo_dataframe(xr_dict)

        # ---- Append distractor variables and promo flags to geo_df -----------
        # Build an aligned (geo, time)-keyed frame and merge, so we never rely
        # on geo_df's row order.
        extra_cols: dict[str, np.ndarray] = {}
        for j, name in enumerate(collinear_variable_names):
            extra_cols[name] = collinear["collinear_gtc"][:, :, j]
        for j, name in enumerate(endogenous_variable_names):
            extra_cols[name] = endog["endogenous_gtc"][:, :, j]

        if extra_cols or any(
            p.include_flag_in_output for p in cfg.promo_events
        ):
            key = pd.MultiIndex.from_product(
                [geo_names, time_names], names=["geo", "time"]
            )
            extra_df = pd.DataFrame(index=key).reset_index()
            for name, arr in extra_cols.items():
                extra_df[name] = np.asarray(arr).reshape(-1)
            for j, p in enumerate(cfg.promo_events):
                if p.include_flag_in_output:
                    flag_t = promo["flags_tc"][:, j]  # (n_times,)
                    extra_df[p.name] = np.tile(flag_t, len(geo_names))
            geo_df = geo_df.merge(extra_df, on=["geo", "time"], how="left")

        national_df = build_national_dataframe(geo_df)

        return SimulationResult(
            config=cfg,
            ground_truth=gt,
            geo_df=geo_df,
            national_df=national_df,
            xr_dict=xr_dict,
            kpi_gt=to_numpy(kpi_gt),
            unit_value_gt=to_numpy(unit_value_gt),
            population_g=to_numpy(population_g),
            geo_names=geo_names,
            time_names=time_names,
            channel_names=channel_names,
            rf_channel_names=rf_channel_names,
            organic_channel_names=organic_channel_names,
            organic_rf_channel_names=organic_rf_channel_names,
            non_media_channel_names=non_media_channel_names,
            context_variable_names=context_variable_names,
            collinear_variable_names=collinear_variable_names,
            endogenous_variable_names=endogenous_variable_names,
            promo_event_names=promo_event_names,
        )
