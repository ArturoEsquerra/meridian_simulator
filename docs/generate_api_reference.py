"""Regenerate docs/api-reference.md from the live package.

Run after any change to public classes, fields, or docstrings::

    python docs/generate_api_reference.py

The reference is derived by introspection, so it cannot drift from the code:
if a field is added without a docstring entry, its Description cell renders
empty and the omission is visible in review.
"""

from __future__ import annotations

import dataclasses
import inspect
import re
from pathlib import Path

import meridian_simulator
from meridian_simulator import augment as ms_augment
from meridian_simulator import baseline as ms_baseline
from meridian_simulator import config as ms_config
from meridian_simulator import context as ms_context
from meridian_simulator import experiments as ms_experiments
from meridian_simulator import media as ms_media
from meridian_simulator import organic as ms_organic
from meridian_simulator import output as ms_output
from meridian_simulator import simulator as ms_simulator

OUT = Path(__file__).resolve().parent / "api-reference.md"


# ---------------------------------------------------------------------------
# Introspection helpers
# ---------------------------------------------------------------------------


def field_default(f: dataclasses.Field) -> str:
    if f.default is not dataclasses.MISSING:
        return repr(f.default)
    if f.default_factory is not dataclasses.MISSING:  # type: ignore[misc]
        try:
            return repr(f.default_factory())
        except Exception:  # noqa: BLE001
            return f.default_factory.__name__ + "()"
    return "(required)"


def type_name(t) -> str:
    s = str(t)
    for junk in ("typing.", "<class '", "'>"):
        s = s.replace(junk, "")
    return s


def summary(obj) -> str:
    return (inspect.getdoc(obj) or "").split("\n\n")[0].replace("\n", " ")


def attr_descriptions(obj) -> dict[str, str]:
    """Parse the Attributes:/Args: block of a Google-style docstring."""
    doc = inspect.getdoc(obj) or ""
    out: dict[str, str] = {}
    current, buf, in_section = None, [], False
    for line in doc.split("\n"):
        stripped = line.strip()
        if stripped in ("Attributes:", "Args:"):
            in_section = True
            continue
        if not in_section:
            continue
        if stripped and not line.startswith(" "):
            break
        if stripped.startswith(("Returns:", "Raises:", "Usage", "Example")):
            break
        m = re.match(r"^(\w+):\s*(.*)", stripped)
        indent = len(line) - len(line.lstrip())
        if m and indent <= 8:
            if current:
                out[current] = " ".join(buf).strip()
            current, buf = m.group(1), [m.group(2)]
        elif current and stripped:
            buf.append(stripped)
    if current:
        out[current] = " ".join(buf).strip()
    return out


def render_dataclass(cls, lines: list[str]) -> None:
    lines.append(f"### `{cls.__name__}`\n")
    lines.append(summary(cls) + "\n")
    body = (inspect.getdoc(cls) or "").split("Attributes:")[0].strip()
    rest = body[len(summary(cls)):].strip()
    if rest:
        lines.append(rest + "\n")
    descs = attr_descriptions(cls)
    lines.append("| Field | Type | Default | Description |")
    lines.append("|---|---|---|---|")
    for f in dataclasses.fields(cls):
        d = descs.get(f.name, "").replace("|", "\\|").replace("``", "`")
        lines.append(
            f"| `{f.name}` | `{type_name(f.type)}` | `{field_default(f)}` | {d} |"
        )
    lines.append("")


def render_function(fn, lines: list[str], name: str | None = None,
                    heading: str = "###") -> None:
    name = name or fn.__name__
    try:
        sig = str(inspect.signature(fn))
    except (ValueError, TypeError):
        sig = "(...)"
    lines.append(f"{heading} `{name}{sig}`\n")
    lines.append((inspect.getdoc(fn) or "*No docstring.*").replace("``", "`") + "\n")


def render_class(cls, lines: list[str], methods: list[str]) -> None:
    lines.append(f"### `{cls.__name__}`\n")
    lines.append((inspect.getdoc(cls) or "").replace("``", "`") + "\n")
    for m in methods:
        fn = getattr(cls, m, None)
        if fn is None:
            continue
        static = inspect.getattr_static(cls, m)
        if isinstance(static, property):
            lines.append(f"#### `{cls.__name__}.{m}` *(property)*\n")
            lines.append(((inspect.getdoc(static.fget) if static.fget else "") or "") + "\n")
        else:
            render_function(fn, lines, name=f"{cls.__name__}.{m}", heading="####")


# ---------------------------------------------------------------------------
# Document
# ---------------------------------------------------------------------------


def build() -> str:
    L: list[str] = []
    L.append(
        f"""# meridian_simulator — API Reference

*Version {meridian_simulator.__version__}. Generated from the source by
`docs/generate_api_reference.py` — rerun it after any code change.
Style follows the [Meridian API reference](https://developers.google.com/meridian/reference/api/meridian):
every public class, field, and function, with parameters and defaults.*

## Package overview

| Module | Purpose |
|---|---|
| `meridian_simulator.config` | Declarative configuration dataclasses (the public API surface) |
| `meridian_simulator.simulator` | `MeridianSimulator` orchestrator and `SimulationResult` container |
| `meridian_simulator.experiments` | Simulated incrementality experiments and calibration priors |
| `meridian_simulator.augment` | Distractor variables, promotional events, KPI observation noise |
| `meridian_simulator.baseline` | Baseline components: intercepts, trend, seasonality, noise |
| `meridian_simulator.media` | Paid media: impressions, R&F, spend, Hill-Adstock, ROI back-solve |
| `meridian_simulator.organic` | Organic (unpaid) media channels |
| `meridian_simulator.context` | Context (control) variables and non-media channels |
| `meridian_simulator.output` | DataFrame / xarray output assembly |

All public names import from the package root:
`from meridian_simulator import SimulationConfig, MeridianSimulator, ...`

---

## meridian_simulator.config
"""
    )
    for cls in [
        ms_config.SimulationConfig, ms_config.MediaChannelConfig,
        ms_config.RFChannelConfig, ms_config.OrganicMediaChannelConfig,
        ms_config.OrganicRFChannelConfig, ms_config.NonMediaChannelConfig,
        ms_config.ContextVariableConfig, ms_config.CollinearVariableConfig,
        ms_config.EndogenousVariableConfig, ms_config.PromoEventConfig,
        ms_config.ExperimentConfig, ms_config.BaselineConfig,
        ms_config.SeasonalityComponent,
    ]:
        render_dataclass(cls, L)

    L.append("\n---\n\n## meridian_simulator.simulator\n")
    render_class(ms_simulator.MeridianSimulator, L, ["run"])
    L.append("### `SimulationResult`\n")
    L.append((inspect.getdoc(ms_simulator.SimulationResult) or "").replace("``", "`") + "\n")
    for m in ["save", "summary", "plot_kpi", "plot_media_spend"]:
        render_function(getattr(ms_simulator.SimulationResult, m), L,
                        name=f"SimulationResult.{m}", heading="####")

    for title, mod, fns in [
        ("experiments", ms_experiments,
         [ms_experiments.simulate_experiments,
          ms_experiments.build_calibration_priors,
          ms_experiments.lognormal_from_point_and_se]),
        ("augment", ms_augment,
         [ms_augment.simulate_collinear_variables,
          ms_augment.simulate_endogenous_variables,
          ms_augment.apply_promo_events, ms_augment.apply_kpi_noise]),
        ("baseline", ms_baseline, [ms_baseline.simulate_baseline]),
        ("media", ms_media,
         [ms_media.simulate_paid_media, ms_media.simulate_impressions,
          ms_media.simulate_reach_frequency]),
        ("organic", ms_organic, [ms_organic.simulate_organic_media]),
        ("context", ms_context,
         [ms_context.simulate_context_variables,
          ms_context.simulate_non_media_channels]),
        ("output", ms_output,
         [ms_output.build_xarray_dataset, ms_output.build_geo_dataframe,
          ms_output.build_national_dataframe]),
    ]:
        L.append(f"\n---\n\n## meridian_simulator.{title}\n")
        for fn in fns:
            render_function(fn, L)

    return "\n".join(L)


if __name__ == "__main__":
    OUT.write_text(build())
    print(f"wrote {OUT} ({len(build().splitlines())} lines)")
