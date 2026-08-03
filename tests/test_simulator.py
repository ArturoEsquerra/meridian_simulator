"""Tests for MeridianSimulator.

Run with:  pytest tests/
"""

import numpy as np
import pytest

from meridian_simulator import (
    MeridianSimulator,
    SimulationConfig,
    MediaChannelConfig,
    RFChannelConfig,
    OrganicMediaChannelConfig,
    NonMediaChannelConfig,
    ContextVariableConfig,
    BaselineConfig,
    SeasonalityComponent,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def minimal_result():
    """Minimal simulation: 2 geos, 20 weeks, 1 paid channel, 1 RF channel."""
    cfg = SimulationConfig(
        n_times=20,
        n_geos=2,
        seed=0,
        media_channels=[
            MediaChannelConfig(name="search", target_roi=3.0)
        ],
        rf_channels=[
            RFChannelConfig(name="youtube", target_roi=2.0)
        ],
        context_variables=[
            ContextVariableConfig(name="gdp", ar1_coef=0.5)
        ],
        baseline=BaselineConfig(n_knots=5, trend_slope=1.0),
    )
    return MeridianSimulator(cfg).run()


@pytest.fixture(scope="module")
def full_result():
    """Full simulation with all channel types and seasonality."""
    cfg = SimulationConfig(
        n_times=52,
        n_geos=5,
        seed=42,
        media_channels=[
            MediaChannelConfig(name="search", target_roi=3.5),
            MediaChannelConfig(name="display", target_roi=1.5, alpha=0.5, ec=0.5, slope=2.0),
        ],
        rf_channels=[
            RFChannelConfig(name="youtube", target_roi=2.2),
        ],
        organic_media_channels=[
            OrganicMediaChannelConfig(name="organic_search"),
        ],
        non_media_channels=[
            NonMediaChannelConfig(name="promotions"),
        ],
        context_variables=[
            ContextVariableConfig(name="gdp_index", ar1_coef=0.8, trend=0.5),
            ContextVariableConfig(name="competitor_spend"),
        ],
        baseline=BaselineConfig(
            trend_slope=3.0,
            n_knots=13,
            seasonality=[SeasonalityComponent(amplitude=2.0, period_weeks=52.0)],
        ),
    )
    return MeridianSimulator(cfg).run()


# ---------------------------------------------------------------------------
# Shape tests
# ---------------------------------------------------------------------------


class TestShapes:
    def test_kpi_shape(self, minimal_result):
        assert minimal_result.kpi_gt.shape == (2, 20)

    def test_unit_value_shape(self, minimal_result):
        assert minimal_result.unit_value_gt.shape == (2, 20)

    def test_population_shape(self, minimal_result):
        assert minimal_result.population_g.shape == (2,)

    def test_geo_df_rows(self, minimal_result):
        # Should have n_geos × n_times rows
        assert len(minimal_result.geo_df) == 2 * 20

    def test_national_df_rows(self, minimal_result):
        assert len(minimal_result.national_df) == 20

    def test_full_geo_df_has_kpi(self, full_result):
        assert "conversions" in full_result.geo_df.columns

    def test_channel_names(self, full_result):
        assert full_result.channel_names == ["search", "display"]
        assert full_result.rf_channel_names == ["youtube"]


# ---------------------------------------------------------------------------
# Ground-truth content tests
# ---------------------------------------------------------------------------


class TestGroundTruth:
    def test_roi_keys_exist(self, minimal_result):
        gt = minimal_result.ground_truth
        assert "roi_m" in gt
        assert "roi_rf" in gt

    def test_roi_positive(self, minimal_result):
        gt = minimal_result.ground_truth
        assert np.all(gt["roi_m"] > 0)
        assert np.all(gt["roi_rf"] > 0)

    def test_roi_target_approximately_met(self, minimal_result):
        """ROI back-solve should bring ground-truth ROI close to target."""
        gt = minimal_result.ground_truth
        # Tolerance is loose because geo-level beta variance introduces noise
        assert abs(gt["roi_m"][0] - 3.0) / 3.0 < 0.5, (
            f"ROI {gt['roi_m'][0]:.2f} is too far from target 3.0"
        )

    def test_kpi_positive(self, minimal_result):
        assert np.all(minimal_result.kpi_gt >= 0)

    def test_ground_truth_has_baseline_components(self, full_result):
        gt = full_result.ground_truth
        for key in ("tau_g", "mu_t", "trend_t", "seasonality_t"):
            assert key in gt, f"Missing key: {key}"

    def test_trend_not_zero(self, full_result):
        gt = full_result.ground_truth
        # trend_t should increase monotonically with slope 3.0
        assert gt["trend_t"][-1] > gt["trend_t"][0]

    def test_seasonality_nonzero(self, full_result):
        gt = full_result.ground_truth
        assert not np.allclose(gt["seasonality_t"], 0.0)


# ---------------------------------------------------------------------------
# Config edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_national_model(self):
        cfg = SimulationConfig(
            n_times=12,
            n_geos=1,
            seed=1,
            media_channels=[MediaChannelConfig(name="tv", target_roi=2.0)],
        )
        result = MeridianSimulator(cfg).run()
        assert result.kpi_gt.shape == (1, 12)
        assert result.config.is_national is True

    def test_no_media_channels(self):
        cfg = SimulationConfig(
            n_times=10,
            n_geos=3,
            seed=2,
            context_variables=[ContextVariableConfig(name="gdp")],
        )
        result = MeridianSimulator(cfg).run()
        assert result.kpi_gt.shape == (3, 10)
        assert len(result.channel_names) == 0

    def test_explicit_adstock_params(self):
        cfg = SimulationConfig(
            n_times=15,
            n_geos=2,
            seed=3,
            media_channels=[
                MediaChannelConfig(
                    name="tv",
                    alpha=0.7,
                    ec=0.5,
                    slope=2.0,
                    target_roi=2.5,
                )
            ],
        )
        result = MeridianSimulator(cfg).run()
        # Verify that the stored alpha matches what was specified
        gt = result.ground_truth
        np.testing.assert_allclose(gt["alpha_m"][0], 0.7, atol=1e-5)

    def test_context_ar1(self):
        cfg = SimulationConfig(
            n_times=30,
            n_geos=3,
            seed=4,
            context_variables=[
                ContextVariableConfig(name="gdp", ar1_coef=0.9, series_mean=100.0)
            ],
        )
        result = MeridianSimulator(cfg).run()
        # GDP column should exist in the geo dataframe
        assert any("gdp" in col for col in result.geo_df.columns)

    def test_many_knots(self):
        cfg = SimulationConfig(
            n_times=20,
            n_geos=2,
            seed=5,
            baseline=BaselineConfig(n_knots=20),  # full knots
        )
        result = MeridianSimulator(cfg).run()
        assert result.ground_truth["mu_t"].shape == (20,)

    def test_few_knots_smooth(self):
        cfg = SimulationConfig(
            n_times=20,
            n_geos=2,
            seed=5,
            baseline=BaselineConfig(n_knots=3),  # very smooth
        )
        result = MeridianSimulator(cfg).run()
        assert result.ground_truth["mu_t"].shape == (20,)


class TestAdvancedFeatures:
    """Native collinear/endogenous variables, promo events, and KPI noise."""

    def _base_cfg(self, **kwargs):
        from meridian_simulator.config import (
            CollinearVariableConfig,
            EndogenousVariableConfig,
            PromoEventConfig,
        )
        defaults = dict(
            n_times=30,
            n_geos=3,
            seed=7,
            media_channels=[MediaChannelConfig(name="search", target_roi=2.0)],
            context_variables=[
                ContextVariableConfig(name="gdp", ar1_coef=0.8, series_mean=100.0)
            ],
        )
        defaults.update(kwargs)
        return SimulationConfig(**defaults)

    def test_collinear_variable(self):
        from meridian_simulator.config import CollinearVariableConfig

        cfg = self._base_cfg(
            collinear_variables=[
                CollinearVariableConfig(
                    name="consumer_confidence",
                    source="gdp",
                    coefficient=0.6,
                    intercept=40.0,
                    noise_std=0.5,
                ),
                CollinearVariableConfig(
                    name="store_count",
                    source="population",
                    coefficient=0.03,
                    noise_std=10.0,
                    round_decimals=0,
                ),
            ]
        )
        result = MeridianSimulator(cfg).run()

        assert "consumer_confidence" in result.geo_df.columns
        assert "store_count" in result.geo_df.columns
        # High realized correlation with source is the whole point
        spec = result.ground_truth["collinear_variables"]
        by_name = {s["name"]: s for s in spec}
        assert by_name["consumer_confidence"]["realized_correlation"] > 0.9
        # store_count is rounded to integers
        assert (result.geo_df["store_count"] % 1 == 0).all()

    def test_collinear_bad_source_raises(self):
        from meridian_simulator.config import CollinearVariableConfig
        import pytest

        with pytest.raises(ValueError, match="neither 'population'"):
            self._base_cfg(
                collinear_variables=[
                    CollinearVariableConfig(name="x", source="nonexistent")
                ]
            )

    def test_endogenous_variable(self):
        from meridian_simulator.config import EndogenousVariableConfig

        cfg = self._base_cfg(
            endogenous_variables=[
                EndogenousVariableConfig(
                    name="gqv_index", driver="search", lag=1, weight=0.65
                )
            ]
        )
        result = MeridianSimulator(cfg).run()

        assert "gqv_index" in result.geo_df.columns
        spec = result.ground_truth["endogenous_variables"][0]
        assert spec["driver"] == "search"
        assert spec["realized_correlation_with_lagged_driver"] > 0.4

    def test_endogenous_bad_driver_raises(self):
        from meridian_simulator.config import EndogenousVariableConfig
        import pytest

        with pytest.raises(ValueError, match="neither 'kpi'"):
            self._base_cfg(
                endogenous_variables=[
                    EndogenousVariableConfig(name="x", driver="nonexistent")
                ]
            )

    def test_promo_events_lift_kpi(self):
        from meridian_simulator.config import PromoEventConfig

        promo_weeks = [10, 11]
        cfg_no = self._base_cfg(seed=11)
        cfg_yes = self._base_cfg(
            seed=11,
            promo_events=[
                PromoEventConfig(
                    name="hot_sale", weeks=promo_weeks, lift_pct=0.5
                )
            ],
        )
        r_no = MeridianSimulator(cfg_no).run()
        r_yes = MeridianSimulator(cfg_yes).run()

        # Flag column present and correct
        assert "hot_sale" in r_yes.geo_df.columns
        flagged = r_yes.geo_df[r_yes.geo_df["hot_sale"] == 1.0]
        assert set(flagged["time"]) == {r_yes.time_names[w] for w in promo_weeks}

        # KPI on promo weeks lifted by ~1.5x vs the identical-seed baseline
        ratio = r_yes.kpi_gt[:, promo_weeks] / r_no.kpi_gt[:, promo_weeks]
        np.testing.assert_allclose(ratio, 1.5, rtol=1e-4)
        # Non-promo weeks unchanged
        other = [w for w in range(cfg_no.n_times) if w not in promo_weeks]
        np.testing.assert_allclose(
            r_yes.kpi_gt[:, other], r_no.kpi_gt[:, other], rtol=1e-5
        )

    def test_promo_flag_excluded_from_output(self):
        from meridian_simulator.config import PromoEventConfig

        cfg = self._base_cfg(
            promo_events=[
                PromoEventConfig(
                    name="buen_fin",
                    weeks=[5],
                    lift_pct=0.3,
                    include_flag_in_output=False,
                )
            ]
        )
        result = MeridianSimulator(cfg).run()
        # Flag hidden from participants but recorded in ground truth
        assert "buen_fin" not in result.geo_df.columns
        assert result.ground_truth["promo_events"][0]["name"] == "buen_fin"
        assert result.ground_truth["promo_flags_tc"][5, 0] == 1.0

    def test_promo_week_out_of_range_raises(self):
        from meridian_simulator.config import PromoEventConfig
        import pytest

        with pytest.raises(ValueError, match="outside"):
            self._base_cfg(
                promo_events=[PromoEventConfig(name="x", weeks=[999])]
            )

    def test_kpi_noise_control(self):
        cfg_clean = self._base_cfg(seed=13, kpi_noise_pct=0.0)
        cfg_noisy = self._base_cfg(seed=13, kpi_noise_pct=0.08)
        r_clean = MeridianSimulator(cfg_clean).run()
        r_noisy = MeridianSimulator(cfg_noisy).run()

        assert r_clean.ground_truth["kpi_noise_realized_cv"] == 0.0
        realized = r_noisy.ground_truth["kpi_noise_realized_cv"]
        assert 0.05 < realized < 0.11

        # Noise perturbs KPI but keeps it non-negative
        assert (r_noisy.kpi_gt >= 0).all()
        assert not np.allclose(r_clean.kpi_gt, r_noisy.kpi_gt)

    def test_reproducibility_with_advanced_features(self):
        from meridian_simulator.config import (
            CollinearVariableConfig,
            EndogenousVariableConfig,
            PromoEventConfig,
        )

        def make_cfg():
            return self._base_cfg(
                seed=21,
                collinear_variables=[
                    CollinearVariableConfig(name="cc", source="gdp", coefficient=0.6)
                ],
                endogenous_variables=[
                    EndogenousVariableConfig(name="gqv", driver="search")
                ],
                promo_events=[
                    PromoEventConfig(name="promo", weeks=[3], lift_pct=0.2)
                ],
                kpi_noise_pct=0.05,
            )

        import pandas as pd

        r1 = MeridianSimulator(make_cfg()).run()
        r2 = MeridianSimulator(make_cfg()).run()
        np.testing.assert_allclose(r1.kpi_gt, r2.kpi_gt)
        pd.testing.assert_frame_equal(r1.geo_df, r2.geo_df)

    def test_promo_baseline_only_preserves_media_contribution(self):
        from meridian_simulator.config import PromoEventConfig

        promo_weeks = [10, 11]
        cfg_no = self._base_cfg(seed=17)
        cfg_full = self._base_cfg(
            seed=17,
            promo_events=[PromoEventConfig(
                name="promo", weeks=promo_weeks, lift_pct=0.5,
                baseline_only=False)],
        )
        cfg_base = self._base_cfg(
            seed=17,
            promo_events=[PromoEventConfig(
                name="promo", weeks=promo_weeks, lift_pct=0.5,
                baseline_only=True)],
        )
        r_no = MeridianSimulator(cfg_no).run()
        r_full = MeridianSimulator(cfg_full).run()
        r_base = MeridianSimulator(cfg_base).run()

        gt = r_base.ground_truth
        baseline = gt["baseline_kpi_gt"]
        media_part_no = r_no.kpi_gt - baseline

        # baseline_only: kpi = baseline * 1.5 + media_part (media untouched)
        expected = baseline[:, promo_weeks] * 1.5 + media_part_no[:, promo_weeks]
        np.testing.assert_allclose(
            r_base.kpi_gt[:, promo_weeks], expected, rtol=1e-4
        )

        # full mode lifts strictly more than baseline_only (media lifted too)
        assert (
            r_full.kpi_gt[:, promo_weeks] > r_base.kpi_gt[:, promo_weeks]
        ).all()

        # both modes leave non-promo weeks identical to the no-promo run
        other = [w for w in range(cfg_no.n_times) if w not in promo_weeks]
        np.testing.assert_allclose(
            r_base.kpi_gt[:, other], r_no.kpi_gt[:, other], rtol=1e-5
        )

        # spec records the mode
        assert r_base.ground_truth["promo_events"][0]["baseline_only"] is True
        assert r_full.ground_truth["promo_events"][0]["baseline_only"] is False

    def test_negative_shock(self):
        from meridian_simulator.config import PromoEventConfig

        shock_weeks = [8, 9]
        cfg_no = self._base_cfg(seed=23)
        cfg_shock = self._base_cfg(
            seed=23,
            promo_events=[PromoEventConfig(
                name="stockout", weeks=shock_weeks, lift_pct=-0.5,
                allow_negative_lift=True)],
        )
        r_no = MeridianSimulator(cfg_no).run()
        r_shock = MeridianSimulator(cfg_shock).run()

        ratio = r_shock.kpi_gt[:, shock_weeks] / r_no.kpi_gt[:, shock_weeks]
        np.testing.assert_allclose(ratio, 0.5, rtol=1e-4)
        assert (r_shock.kpi_gt >= 0).all()

    def test_negative_lift_requires_flag(self):
        from meridian_simulator.config import PromoEventConfig
        import pytest

        with pytest.raises(ValueError, match="allow_negative_lift"):
            self._base_cfg(
                promo_events=[PromoEventConfig(name="x", weeks=[1],
                                               lift_pct=-0.3)]
            )

    def test_seasonal_flighting_creates_confounding(self):
        cfg_kwargs = dict(
            n_times=104, n_geos=3, seed=31,
            context_variables=[],
            baseline=BaselineConfig(
                n_knots=13,
                seasonality=[SeasonalityComponent(amplitude=3.0,
                                                  period_weeks=52.0)],
            ),
        )
        cfg_iid = SimulationConfig(
            media_channels=[MediaChannelConfig(
                name="tv", target_roi=2.0, seasonal_flighting=0.0)],
            **cfg_kwargs,
        )
        cfg_sync = SimulationConfig(
            media_channels=[MediaChannelConfig(
                name="tv", target_roi=2.0, seasonal_flighting=0.9)],
            **cfg_kwargs,
        )
        r_iid = MeridianSimulator(cfg_iid).run()
        r_sync = MeridianSimulator(cfg_sync).run()

        def season_corr(result):
            s = result.ground_truth["seasonality_t"]
            imp = (
                result.geo_df.groupby("time")["tv_impression"].sum()
                .reindex(result.time_names).values
            )
            return np.corrcoef(imp, s)[0, 1]

        c_iid = season_corr(r_iid)
        c_sync = season_corr(r_sync)
        assert c_sync > 0.5, f"synced flighting corr too low: {c_sync:.3f}"
        assert c_sync > abs(c_iid) + 0.2, (
            f"flighting should raise seasonality correlation "
            f"(iid={c_iid:.3f}, sync={c_sync:.3f})"
        )

    def test_flighting_out_of_range_raises(self):
        import pytest

        with pytest.raises(ValueError, match="seasonal_flighting"):
            self._base_cfg(
                media_channels=[MediaChannelConfig(
                    name="x", seasonal_flighting=1.5)]
            )
