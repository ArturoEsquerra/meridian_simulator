"""Convert simulation tensors into xarray DataArrays and Pandas DataFrames.

The output formats are designed to be directly consumable by Meridian's
``InputData`` class.  Both geo-level and national-aggregated datasets are
produced.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import tensorflow as tf
import xarray as xr

from meridian_simulator.utils import (
    add_suffix,
    remove_suffix,
    to_numpy,
)

# Meridian canonical dimension names
GEO_DIM = "geo"
TIME_DIM = "time"
CHANNEL_DIM = "channel"
RF_CHANNEL_DIM = "rf_channel"
ORGANIC_CHANNEL_DIM = "organic_channel"
ORGANIC_RF_CHANNEL_DIM = "organic_rf_channel"
NON_MEDIA_DIM = "non_media_channel"
CONTROL_DIM = "control"

KPI_COL = "conversions"
POPULATION_COL = "population"
UNIT_VALUE_COL = "revenue_per_conversion"
IMPRESSIONS_SUFFIX = "impression"
SPEND_SUFFIX = "spend"
REACH_SUFFIX = "reach"
FREQ_SUFFIX = "frequency"
CONTROL_SUFFIX = "control"


def build_xarray_dataset(
    *,
    geo_names: list[str],
    time_names: list[str],
    media_channel_names: list[str],
    rf_channel_names: list[str],
    organic_media_channel_names: list[str],
    organic_rf_channel_names: list[str],
    non_media_channel_names: list[str],
    context_variable_names: list[str],
    kpi_gt: np.ndarray,
    unit_value_gt: np.ndarray,
    population_g: np.ndarray,
    impression_gtm: np.ndarray,
    cost_gtm: np.ndarray,
    reach_gtm: np.ndarray,
    freq_gtm: np.ndarray,
    organic_impression_gtm: np.ndarray,
    organic_reach_gtm: np.ndarray,
    organic_freq_gtm: np.ndarray,
    non_media_gtc: np.ndarray,
    context_gtc: np.ndarray,
) -> dict[str, xr.DataArray]:
    """Build a dict of xr.DataArrays ready for Meridian's InputData builder."""

    n_geos = len(geo_names)
    n_times = len(time_names)
    n_m = len(media_channel_names)
    n_rf = len(rf_channel_names)
    n_om = len(organic_media_channel_names)
    n_orf = len(organic_rf_channel_names)
    n_non = len(non_media_channel_names)
    n_ctx = len(context_variable_names)

    out = {}

    out["kpi"] = xr.DataArray(
        kpi_gt,
        dims=[GEO_DIM, TIME_DIM],
        coords={GEO_DIM: geo_names, TIME_DIM: time_names},
        name=KPI_COL,
    )

    out["unit_value"] = xr.DataArray(
        unit_value_gt,
        dims=[GEO_DIM, TIME_DIM],
        coords={GEO_DIM: geo_names, TIME_DIM: time_names},
        name=UNIT_VALUE_COL,
    )

    out["population"] = xr.DataArray(
        population_g,
        dims=[GEO_DIM],
        coords={GEO_DIM: geo_names},
        name=POPULATION_COL,
    )

    if n_m > 0:
        out["media"] = xr.DataArray(
            impression_gtm,
            dims=[GEO_DIM, TIME_DIM, CHANNEL_DIM],
            coords={
                GEO_DIM: geo_names,
                TIME_DIM: time_names,
                CHANNEL_DIM: media_channel_names,
            },
            name=IMPRESSIONS_SUFFIX,
        )
        out["spend"] = xr.DataArray(
            cost_gtm[..., :n_m],
            dims=[GEO_DIM, TIME_DIM, CHANNEL_DIM],
            coords={
                GEO_DIM: geo_names,
                TIME_DIM: time_names,
                CHANNEL_DIM: media_channel_names,
            },
            name=SPEND_SUFFIX,
        )

    if n_rf > 0:
        out["reach"] = xr.DataArray(
            reach_gtm,
            dims=[GEO_DIM, TIME_DIM, RF_CHANNEL_DIM],
            coords={
                GEO_DIM: geo_names,
                TIME_DIM: time_names,
                RF_CHANNEL_DIM: rf_channel_names,
            },
            name=REACH_SUFFIX,
        )
        out["frequency"] = xr.DataArray(
            freq_gtm,
            dims=[GEO_DIM, TIME_DIM, RF_CHANNEL_DIM],
            coords={
                GEO_DIM: geo_names,
                TIME_DIM: time_names,
                RF_CHANNEL_DIM: rf_channel_names,
            },
            name=FREQ_SUFFIX,
        )
        out["rf_spend"] = xr.DataArray(
            cost_gtm[..., n_m:],
            dims=[GEO_DIM, TIME_DIM, RF_CHANNEL_DIM],
            coords={
                GEO_DIM: geo_names,
                TIME_DIM: time_names,
                RF_CHANNEL_DIM: rf_channel_names,
            },
            name=add_suffix(SPEND_SUFFIX, "rf"),
        )

    if n_om > 0:
        out["organic_media"] = xr.DataArray(
            organic_impression_gtm,
            dims=[GEO_DIM, TIME_DIM, ORGANIC_CHANNEL_DIM],
            coords={
                GEO_DIM: geo_names,
                TIME_DIM: time_names,
                ORGANIC_CHANNEL_DIM: organic_media_channel_names,
            },
            name="organic_impression",
        )

    if n_orf > 0:
        out["organic_reach"] = xr.DataArray(
            organic_reach_gtm,
            dims=[GEO_DIM, TIME_DIM, ORGANIC_RF_CHANNEL_DIM],
            coords={
                GEO_DIM: geo_names,
                TIME_DIM: time_names,
                ORGANIC_RF_CHANNEL_DIM: organic_rf_channel_names,
            },
            name="organic_reach",
        )
        out["organic_frequency"] = xr.DataArray(
            organic_freq_gtm,
            dims=[GEO_DIM, TIME_DIM, ORGANIC_RF_CHANNEL_DIM],
            coords={
                GEO_DIM: geo_names,
                TIME_DIM: time_names,
                ORGANIC_RF_CHANNEL_DIM: organic_rf_channel_names,
            },
            name="organic_frequency",
        )

    if n_ctx > 0:
        ctx_names_col = [add_suffix(n, CONTROL_SUFFIX) for n in context_variable_names]
        out["controls"] = xr.DataArray(
            context_gtc,
            dims=[GEO_DIM, TIME_DIM, CONTROL_DIM],
            coords={
                GEO_DIM: geo_names,
                TIME_DIM: time_names,
                CONTROL_DIM: ctx_names_col,
            },
            name="control_value",
        )

    if n_non > 0:
        out["non_media"] = xr.DataArray(
            non_media_gtc,
            dims=[GEO_DIM, TIME_DIM, NON_MEDIA_DIM],
            coords={
                GEO_DIM: geo_names,
                TIME_DIM: time_names,
                NON_MEDIA_DIM: non_media_channel_names,
            },
            name="non_media_value",
        )

    return out


def build_geo_dataframe(xr_dict: dict[str, xr.DataArray]) -> pd.DataFrame:
    """Flatten the xarray dict into a single wide Pandas DataFrame (geo-level).

    Column naming convention (matches DataFrameInputDataBuilder expectations):
      - Media impressions : ``{channel}_impression``
      - Media spend       : ``{channel}_spend``
      - R&F reach         : ``{channel}_reach``
      - R&F frequency     : ``{channel}_frequency``
      - Organic impr.     : ``{channel}_organic_impression``
      - Organic reach     : ``{channel}_organic_reach``
      - Organic freq.     : ``{channel}_organic_frequency``
      - Controls          : ``{var}_control``  (already suffixed in xr coords)
      - Non-media         : kept as-is (channel name)
    """
    frames = []

    def pivot_channel(
        da: xr.DataArray,
        value_name: str,
        channel_dim: str,
        col_suffix: str,
    ) -> pd.DataFrame:
        """Pivot a (geo, time, channel) DataArray; rename cols to {ch}_{suffix}."""
        df = da.to_dataframe(name=value_name).reset_index()
        df = df.pivot(
            index=[GEO_DIM, TIME_DIM],
            columns=channel_dim,
            values=value_name,
        ).reset_index()
        df.columns.name = None
        # Rename channel columns to {channel}_{col_suffix}
        ch_cols = [c for c in df.columns if c not in (GEO_DIM, TIME_DIM)]
        df = df.rename(columns={c: f"{c}_{col_suffix}" for c in ch_cols})
        return df

    def flat(da: xr.DataArray, value_name: str, channel_dim: str) -> pd.DataFrame:
        """For non-channel DataArrays (already have good column names)."""
        df = da.to_dataframe(name=value_name).reset_index()
        if channel_dim and channel_dim in df.columns:
            df = df.pivot(
                index=[GEO_DIM, TIME_DIM],
                columns=channel_dim,
                values=value_name,
            ).reset_index()
            df.columns.name = None
        return df

    if "kpi" in xr_dict:
        frames.append(xr_dict["kpi"].to_dataframe(name=KPI_COL).reset_index())
    if "unit_value" in xr_dict:
        frames.append(xr_dict["unit_value"].to_dataframe(name=UNIT_VALUE_COL).reset_index())
    if "population" in xr_dict:
        frames.append(xr_dict["population"].to_dataframe(name=POPULATION_COL).reset_index())

    # Media channels — produce {channel}_impression and {channel}_spend cols
    if "media" in xr_dict:
        frames.append(pivot_channel(xr_dict["media"], IMPRESSIONS_SUFFIX, CHANNEL_DIM, IMPRESSIONS_SUFFIX))
    if "spend" in xr_dict:
        frames.append(pivot_channel(xr_dict["spend"], SPEND_SUFFIX, CHANNEL_DIM, SPEND_SUFFIX))

    # R&F channels
    if "reach" in xr_dict:
        frames.append(pivot_channel(xr_dict["reach"], REACH_SUFFIX, RF_CHANNEL_DIM, REACH_SUFFIX))
    if "frequency" in xr_dict:
        frames.append(pivot_channel(xr_dict["frequency"], FREQ_SUFFIX, RF_CHANNEL_DIM, FREQ_SUFFIX))
    if "rf_spend" in xr_dict:
        frames.append(pivot_channel(xr_dict["rf_spend"], SPEND_SUFFIX, RF_CHANNEL_DIM, SPEND_SUFFIX))

    # Organic channels
    if "organic_media" in xr_dict:
        frames.append(pivot_channel(xr_dict["organic_media"], "organic_impression", ORGANIC_CHANNEL_DIM, "impression"))
    if "organic_reach" in xr_dict:
        frames.append(pivot_channel(xr_dict["organic_reach"], "organic_reach", ORGANIC_RF_CHANNEL_DIM, "reach"))
    if "organic_frequency" in xr_dict:
        frames.append(pivot_channel(xr_dict["organic_frequency"], "organic_frequency", ORGANIC_RF_CHANNEL_DIM, "frequency"))

    # Controls — coords are already named {var}_control
    if "controls" in xr_dict:
        frames.append(flat(xr_dict["controls"], "control_value", CONTROL_DIM))

    # Non-media — kept as channel name
    if "non_media" in xr_dict:
        frames.append(flat(xr_dict["non_media"], "non_media_value", NON_MEDIA_DIM))

    result = frames[0]
    for f in frames[1:]:
        merge_on = [c for c in [GEO_DIM, TIME_DIM, POPULATION_COL] if c in result.columns and c in f.columns]
        result = result.merge(f, on=merge_on, how="left")

    return result


def build_national_dataframe(geo_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate the geo-level DataFrame to a national (single-geo) level."""
    agg: dict[str, str] = {}

    for col in geo_df.columns:
        if col in (GEO_DIM, TIME_DIM, POPULATION_COL):
            continue
        elif col == KPI_COL:
            agg[col] = "sum"
        elif col == UNIT_VALUE_COL:
            agg[col] = "mean"
        elif col.endswith(f"_{IMPRESSIONS_SUFFIX}") or col.endswith(f"_{SPEND_SUFFIX}"):
            agg[col] = "sum"
        elif col.endswith(f"_{REACH_SUFFIX}"):
            agg[col] = "sum"
        elif col.endswith(f"_{CONTROL_SUFFIX}") or col == "non_media_value":
            agg[col] = "mean"
        else:
            agg[col] = "mean"

    freq_cols = [c for c in geo_df.columns if c.endswith(f"_{FREQ_SUFFIX}")]
    for fc in freq_cols:
        if fc in agg:
            del agg[fc]

    national = geo_df.groupby(TIME_DIM).agg(agg).reset_index()

    for fc in freq_cols:
        ch = remove_suffix(fc, FREQ_SUFFIX)
        rc = add_suffix(ch, REACH_SUFFIX)
        if rc in geo_df.columns:
            national[fc] = (
                geo_df.groupby(TIME_DIM)
                .apply(
                    lambda g, _fc=fc, _rc=rc: (
                        np.average(g[_fc], weights=g[_rc])
                        if g[_rc].sum() > 0
                        else 0.0
                    ),
                    include_groups=False,
                )
                .values
            )

    return national
