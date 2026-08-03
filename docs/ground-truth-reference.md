# Ground Truth Reference

`result.ground_truth` is a plain dict recording everything needed to verify recovery. Array shapes use `G` = geos, `T` = times, `M` = paid impression channels, `RF` = R&F channels, `OM`/`ORF` = organic counterparts, `C` = context variables, `N` = non-media channels, `E` = promo events.

## Baseline

| Key | Shape | Meaning |
|---|---|---|
| `tau_g` | `(G,)` | Geo intercepts (per-capita KPI units) |
| `mu_t` | `(T,)` | Knot-interpolated time-varying baseline |
| `trend_t` | `(T,)` | Linear trend component |
| `seasonality_t` | `(T,)` | Stacked seasonal waves |
| `kpi_mean`, `kpi_std` | scalar | Meridian `KpiTransformer` scaling constants |
| `intercept_gt` | `(G, T)` | Combined intercept on Meridian's transformed scale |
| `baseline_kpi_gt` | `(G, T)` | Non-media-driven KPI portion (baseline + context + non-media), pre-promo/noise — the `B` in promo mode math |

## Media (paid)

| Key | Shape | Meaning |
|---|---|---|
| `roi_m`, `roi_rf` | `(M,)`, `(RF,)` | **True ROI per channel** (raw scale) |
| `beta_m`, `beta_rf` | `(M,)`, `(RF,)` | Hierarchical effect means (Meridian's transformed scale) |
| `eta_m`, `eta_rf` | `(M,)`, `(RF,)` | Hierarchical stds |
| `beta_gm`, `beta_grf` | `(G, M)`, `(G, RF)` | Geo-level coefficients (transformed scale) |
| `alpha_m`, `alpha_rf` | `(M,)`, `(RF,)` | **True adstock decays** |
| `ec_m`, `slope_m`, `ec_rf`, `slope_rf` | `(M,)`, … | True Hill parameters |
| `cost_gtm` | `(G, T, M+RF)` | Spend tensor |
| `total_spend` | scalar | Total media spend |

## Organic media

`beta_om`, `eta_om`, `beta_gom`, `alpha_om`, `ec_om`, `slope_om` and the `_orf` equivalents — same semantics as paid, no spend.

## Controls & non-media

| Key | Shape | Meaning |
|---|---|---|
| `gamma_c`, `xi_c` | `(C,)` | Context coefficient means and geo-deviation stds |
| `gamma_gc` | `(G, C)` | Geo-level context coefficients (transformed scale) |
| `gamma_n`, `xi_n`, `gamma_gn` | `(N,)`, `(N,)`, `(G, N)` | Non-media equivalents |

## Promo events

| Key | Shape | Meaning |
|---|---|---|
| `promo_events` | list of dicts | Per-event recipe: `name`, `weeks`, `lift_pct`, `lift_geo_std`, `include_flag_in_output`, `baseline_only`, `allow_negative_lift` |
| `promo_multiplier_gt` | `(G, T)` | Realized total-KPI multipliers (`M_full`) |
| `promo_baseline_multiplier_gt` | `(G, T)` | Realized baseline-only multipliers (`M_base`) |
| `promo_flags_tc` | `(T, E)` | Binary event flags (even when hidden from outputs) |

## Observation noise

| Key | Meaning |
|---|---|
| `kpi_noise_pct` | Configured CV |
| `kpi_noise_multiplier_gt` | `(G, T)` realized noise multipliers |
| `kpi_noise_realized_cv` | Realized CV of the multipliers |

## Distractor variables

| Key | Meaning |
|---|---|
| `collinear_variables` | List of recipes: `name`, `source`, `coefficient`, `intercept`, `noise_std`, `realized_correlation` |
| `endogenous_variables` | List of recipes: `name`, `driver`, `lag`, `weight`, `realized_correlation_with_lagged_driver` |

Both lists exist so graders can verify trap handling **without hand-maintaining trap registries** — [`meridian_grader`](../../meridian_grader/README.md)'s `A2_trap_exclusion` check consumes them directly.

## Scale notes

- `roi_m` / `roi_rf` are on the raw revenue scale — directly comparable to Meridian's `Analyzer.roi()` posteriors.
- `beta_*` / `gamma_*` keys are stored on **Meridian's internally transformed scale** (divided by `kpi_std`, log-shifted for medias), matching what Meridian's posteriors estimate. Compare like with like.
- `alpha_*`, `ec_*`, `slope_*` are scale-free and directly comparable to posteriors.
