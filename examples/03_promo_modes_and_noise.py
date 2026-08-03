"""Example 3 — Promo lift modes and the KPI noise knob, side by side.

Demonstrates:
  1. baseline_only=False vs True: how the two promo modes treat media
     contributions (and therefore whether recorded ROIs stay exactly true).
  2. kpi_noise_pct: how observation noise degrades apparent fit quality.

Run:  python examples/03_promo_modes_and_noise.py
"""

import numpy as np

from meridian_simulator import (
    BaselineConfig,
    MediaChannelConfig,
    MeridianSimulator,
    PromoEventConfig,
    SimulationConfig,
)

PROMO_WEEKS = [20, 21]


def make_cfg(promo=None, noise=0.0, seed=7):
    return SimulationConfig(
        n_times=52, n_geos=4, seed=seed,
        media_channels=[
            MediaChannelConfig(name="search", target_roi=3.0,
                               alpha=0.3, ec=0.6, slope=1.3),
        ],
        promo_events=promo or [],
        kpi_noise_pct=noise,
        baseline=BaselineConfig(n_knots=13, trend_slope=1.0),
    )


# ── 1. Promo modes ──────────────────────────────────────────────────────────
r_none = MeridianSimulator(make_cfg()).run()
r_full = MeridianSimulator(make_cfg(promo=[
    PromoEventConfig(name="promo", weeks=PROMO_WEEKS, lift_pct=0.5,
                     baseline_only=False)])).run()
r_base = MeridianSimulator(make_cfg(promo=[
    PromoEventConfig(name="promo", weeks=PROMO_WEEKS, lift_pct=0.5,
                     baseline_only=True)])).run()

B = r_base.ground_truth["baseline_kpi_gt"]     # non-media-driven KPI portion
media_part = r_none.kpi_gt - B                 # media-driven portion

print("=== Promo lift modes (lift = +50% on weeks 20-21) ===")
w = PROMO_WEEKS
print(f"KPI without promo, promo weeks:      {r_none.kpi_gt[:, w].sum():,.0f}")
print(f"KPI with FULL lift (default):        {r_full.kpi_gt[:, w].sum():,.0f}"
      f"   <- media amplified too")
print(f"KPI with BASELINE-ONLY lift:         {r_base.kpi_gt[:, w].sum():,.0f}"
      f"   <- media contributions untouched")

# In baseline-only mode, the media-driven KPI portion is bit-for-bit unchanged:
np.testing.assert_allclose(
    r_base.kpi_gt[:, w] - 1.5 * B[:, w], media_part[:, w], rtol=1e-4
)
print("\nVerified: baseline_only=True leaves media contributions exactly "
      "unchanged,\nso ground_truth['roi_m'] stays exactly true.")

# ── 2. KPI observation noise ────────────────────────────────────────────────
print("\n=== KPI observation noise knob ===")
for noise in [0.0, 0.02, 0.05, 0.10]:
    r = MeridianSimulator(make_cfg(noise=noise, seed=11)).run()
    cv = r.ground_truth["kpi_noise_realized_cv"]
    # Apparent week-over-week volatility of national KPI
    nat = r.kpi_gt.sum(axis=0)
    wow = np.abs(np.diff(nat) / nat[:-1]).mean() * 100
    print(f"  kpi_noise_pct={noise:.2f} -> realized CV={cv:.4f}, "
          f"mean |WoW change| = {wow:.1f}%")

print("\nHigher noise -> noisier KPI -> lower apparent R-squared for ANY "
      "model.\nUse 0.01-0.03 for clean data, 0.05-0.10 for messy reality.")
