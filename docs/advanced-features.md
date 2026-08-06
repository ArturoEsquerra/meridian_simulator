# Advanced Features

The features that make simulated data behave like data you'd meet in the wild. Each is opt-in, fully recorded in the ground truth, and backward compatible (defaults reproduce the pre-feature behavior exactly).

---

## Multicollinearity traps (`CollinearVariableConfig`)

**The real-world problem.** Datasets arrive with redundant variables — store counts that track population, confidence indices that track GDP. Analysts who throw every column into the model inflate variance and destabilize coefficients; VIF analysis exists precisely to catch this.

**What the simulator does.** Generates variables as a linear map of a real series plus noise:

```
value = coefficient × source + intercept + N(0, noise_std)
```

with `source` either `"population"` or any configured context variable. The variables appear in `geo_df` looking perfectly legitimate — but the KPI equation never touches them: **their true causal effect is zero**. A correct analysis detects the collinearity (the recipe's realized correlation is recorded in `ground_truth["collinear_variables"]`) and excludes them.

**Tuning difficulty.** `noise_std` sets how obvious the trap is: small values give r ≈ 0.99 (blatant VIF explosion), larger values give r ≈ 0.7–0.8 (only a careful correlation analysis catches it).

---

## Endogeneity traps (`EndogenousVariableConfig`)

**The real-world problem.** Some series are *symptoms* of marketing, not causes of sales: branded query volume rises after YouTube bursts; site traffic tracks campaign flights. Adding them as controls violates the backdoor criterion — they're mediators or colliders — and soaks up media credit, biasing ROI toward zero.

**What the simulator does.**

```
z     = weight × standardize(lag(driver)) + (1 − weight) × N(0, 1)
value = base + scale × z
```

`driver` is a paid channel (the variable follows its lagged **spend**) or `"kpi"` (reverse causality: the variable follows sales themselves). Zero direct effect on the KPI. `weight=0.65` with `lag=1` gives a correlation that lagged-correlation analysis finds easily but which doesn't scream from a plain correlation matrix.

**The canonical example** — Google query volume driven by YouTube:

```python
EndogenousVariableConfig(name="gqv_index", driver="youtube", lag=1, weight=0.65)
```

---

## Promotional events & structural shocks (`PromoEventConfig`)

**The real-world problem.** Retailers run Hot Sale, Black Friday, El Buen Fin — commercial actions (discounts, financing, exclusives) that move sales independently of media. Omitting them biases everything the model touches (omitted-variable bias); modeling them as generic *controls* rather than *non-media treatments* misstates their causal role. And demand doesn't only jump up — stockouts and disruptions knock it down.

**What the simulator does.** Applies multiplicative lifts to the KPI on the configured weeks, with optional per-geo jitter:

```
multiplier_g = 1 + max(lift_pct + N(0, lift_geo_std), floor)
```

where `floor` is 0 normally and −0.95 when `allow_negative_lift=True`.

### The two lift modes (`baseline_only`)

Let `B` be the non-media-driven KPI portion (baseline + context + non-media contributions) and `M` the media-driven portion. With baseline-only multiplier `M_base` and total multiplier `M_full`:

```
kpi_new = B · M_base · M_full  +  M · M_full
```

| Mode | Semantics | When to use |
|---|---|---|
| `baseline_only=False` (default) | Lift scales **everything** — media contributions are amplified during promos. True channel ROI on event weeks runs slightly above the recorded `roi_m`. | Realism: media genuinely converts better during high-traffic events. |
| `baseline_only=True` | Lift scales only `B`. Media contributions untouched → recorded ground-truth ROIs remain **exactly** true. | Clean recovery experiments; grading where exact ROI truth matters. |

The realized multipliers are stored per geo-week in `ground_truth["promo_multiplier_gt"]` (total) and `["promo_baseline_multiplier_gt"]` (baseline-only), with `B` itself in `["baseline_kpi_gt"]`.

### Hiding the flags

`include_flag_in_output=False` keeps the KPI spikes but removes the binary columns from `geo_df` — the competition trap where contestants must engineer event flags from domain knowledge. The flags always remain in `ground_truth["promo_flags_tc"]`.

### Negative shocks

```python
PromoEventConfig(name="stockout_q3", weeks=[60, 61], lift_pct=-0.40,
                 allow_negative_lift=True)
```

Models structural breaks *downward*: stockouts, store closures, supply disruptions. A −40% lift multiplies the KPI by 0.6 on those weeks. Setting a negative `lift_pct` without the flag raises a `ValueError` at config time — the flag is deliberate friction so old configs can't change meaning silently.

---

## Demand-synchronized flighting (`seasonal_flighting`)

**The real-world problem.** The best-documented confounder in the MMM literature: media planners buy *into* high season. TV is heavy before Christmas because demand is high before Christmas — so media and seasonal demand are correlated, and a model with a too-flexible baseline hands seasonal sales to the baseline (media under-credited) while a too-rigid one hands them to media (over-credited). Knot selection is exactly the lever that decides this.

**What the simulator does.** Each paid channel accepts `seasonal_flighting ∈ [0, 1]`. The channel's weekly activity mean is modulated by the baseline's own seasonal wave (normalized to unit peak):

```
u_tm ← u_tm + impression_mean_time × seasonal_flighting × s_norm(t)
```

At `0.0` (default) media activity is drawn independently of season — the historical behavior. At `0.5–0.9` national media volume visibly tracks the seasonal wave, and separating media from demand becomes the genuine identification problem it is in practice.

```python
MediaChannelConfig(name="tv", target_roi=0.9, alpha=0.7,
                   seasonal_flighting=0.7)   # heavy seasonal buyer
```

Requires the baseline to actually have seasonality (`BaselineConfig.seasonality` non-empty); with a flat baseline the knob is a no-op.

---

## KPI observation noise (`kpi_noise_pct`)

**The real-world problem.** Sales data carry measurement error — reporting lags, restatements, attribution glitches. Simulated data without observation noise makes every fit metric unrealistically flattering (R² ≈ 0.99) and hides how uncertainty propagates into ROI credible intervals.

**What the simulator does.** One interpretable knob — the coefficient of variation of multiplicative noise on the *final* KPI:

```
kpi ← kpi × max(1 + N(0, kpi_noise_pct), 0)
```

| `kpi_noise_pct` | Regime |
|---|---|
| `0.0` | Noise-free (default; historical behavior) |
| `0.01–0.03` | Clean enterprise data |
| `0.05–0.10` | Typical messy reality |
| `> 0.10` | Stress-testing |

Applied after promo lifts and before the KPI transformer, so Meridian sees exactly the recorded truth. The realized multipliers and CV land in `ground_truth["kpi_noise_multiplier_gt"]` and `["kpi_noise_realized_cv"]`. This is *observation* noise; for *structural* demand volatility use `BaselineConfig.noise_std` (additive, inside the baseline).

---

## Composition order

For reasoning about combined effects, the pipeline applies, in order:

1. Baseline (intercepts + knots + trend + seasonality + additive noise)
2. Context and non-media contributions
3. *(snapshot: `baseline_kpi_gt` = everything so far, per capita × population)*
4. Paid + organic media contributions (with optional seasonal flighting upstream in the activity draws)
5. Promo events (mode-aware multiplicative lifts)
6. KPI observation noise
7. KPI transformer scaling, ground-truth recording, output assembly

Distractor variables (collinear, endogenous) are generated after step 6 — they observe the finished system but never feed back into it.


## Incrementality experiments & prior calibration (v2.2)

Meridian's differentiating feature is calibrating ROI priors with incrementality experiments. The simulator closes the loop by simulating the experiments themselves:

```python
from meridian_simulator import ExperimentConfig

cfg = SimulationConfig(
    ...,
    experiments=[
        # Full-duration conversion lift on Google
        ExperimentConfig(name="google_lift", channel="google_performance", se_pct=0.10),
        # 8-week geo holdout on TV, weeks 40-47
        ExperimentConfig(name="tv_holdout", channel="tv",
                         start_week=40, end_week=47, se_pct=0.20),
        # TRAP: a short-horizon study that misses 30% of true incrementality
        ExperimentConfig(name="yt_short", channel="youtube",
                         se_pct=0.08, bias_pct=-0.30),
    ],
)
result = MeridianSimulator(cfg).run()
```

Each experiment measures the channel's **true window ROI** from the recorded contributions, then reports `point_estimate` and `standard_error` (`ground_truth["experiments"]`). `ground_truth["experiment_calibration"]` is Meridian-ready:

```python
import tensorflow_probability as tfp
from meridian.model import prior_distribution, spec

cal = result.ground_truth["experiment_calibration"]
prior = prior_distribution.PriorDistribution(
    roi_m=tfp.distributions.LogNormal(
        cal["roi_mu"].astype("float32"), cal["roi_sigma"].astype("float32"))
)
model_spec = spec.ModelSpec(
    prior=prior,
    roi_calibration_period=cal["roi_calibration_period"],  # None if all full-duration
)
```

Uncalibrated channels keep Meridian's default `LogNormal(0.2, 0.9)`. The `bias_pct` knob turns calibration itself into a judgment test: an analyst who calibrates hard on a biased study pulls the posterior toward the wrong ROI — and the ground truth records exactly how wrong.
