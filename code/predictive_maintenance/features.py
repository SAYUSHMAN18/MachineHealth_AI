from __future__ import annotations

from datetime import timedelta
import re
from typing import Any

import numpy as np
import pandas as pd


RAW_MEASUREMENT_COLUMNS = [
    "iron_ppm",
    "copper_ppm",
    "aluminium_ppm",
    "chromium_ppm",
    "lead_ppm",
    "silicon_ppm",
    "water_pct",
    "fuel_dilution_pct",
    "soot_pct",
    "viscosity_cst",
    "oxidation",
    "tbn",
]

TEXT_SIGNAL_PATTERNS = {
    "lab_text_wear_signal": r"\bwear\b|\biron\b|\bcopper\b|\bchromium\b|\blead\b|\bmetal",
    "lab_text_contamination_signal": r"\bcontamina|\bdirt\b|\bsilicon\b|\bingress\b|\bdebris\b",
    "lab_text_coolant_signal": r"\bwater\b|\bcoolant\b|\bglycol\b|\bantifreeze\b",
    "lab_text_fuel_signal": r"\bfuel\b|\bdilution\b",
    "lab_text_degradation_signal": r"\bviscosity\b|\boxidation\b|\bacidity\b|\btbn\b|\bsoot\b",
    "lab_text_urgent_signal": r"\bimmediate|\burgent|\bcritical|\bshutdown|\binspect now|\baction required",
    "lab_text_resample_signal": r"\bresample|\bsample again|\bhistory needed|\bmore sample history",
}


def add_sos_trends(sos: pd.DataFrame) -> pd.DataFrame:
    out = sos.sort_values(["asset_id", "component", "sample_date"]).copy()
    groups = out.groupby(["asset_id", "component"], dropna=False)
    out["previous_sample_date"] = groups["sample_date"].shift(1)
    out["days_since_previous_sample"] = (
        out["sample_date"] - out["previous_sample_date"]
    ).dt.days
    out["previous_fluid_hours"] = groups["fluid_hours"].shift(1)
    out["fluid_hours_delta"] = out["fluid_hours"] - out["previous_fluid_hours"]

    for column in RAW_MEASUREMENT_COLUMNS:
        previous = groups[column].shift(1)
        out[f"{column}_delta"] = out[column] - previous
        out[f"{column}_rate_100h"] = (
            100
            * out[f"{column}_delta"]
            / out["fluid_hours_delta"].where(out["fluid_hours_delta"].gt(0))
        )
        # A short rolling average reduces sensitivity to one noisy laboratory
        # result while remaining understandable to maintenance engineers.
        out[f"{column}_mean_3"] = groups[column].transform(
            lambda values: values.rolling(3, min_periods=1).mean()
        )
    return out


def telemetry_asset_summary(telemetry: pd.DataFrame) -> pd.DataFrame:
    if telemetry.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for asset_id, group in telemetry.groupby("asset_id", dropna=False):
        group = group.sort_values("event_time")
        valid_op = group["operating_hours"].dropna()
        valid_odo = group["odometer"].dropna()
        rows.append(
            {
                "asset_id": asset_id,
                "serial_number": group["serial_number"].dropna().iloc[-1]
                if group["serial_number"].notna().any()
                else pd.NA,
                "machine_model": group["machine_model"].dropna().iloc[-1]
                if group["machine_model"].notna().any()
                else pd.NA,
                "snapshots": len(group),
                "start": group["event_time"].min(),
                "end": group["event_time"].max(),
                "operating_hours_gain": float(valid_op.iloc[-1] - valid_op.iloc[0])
                if len(valid_op) >= 2
                else np.nan,
                "odometer_gain": float(valid_odo.iloc[-1] - valid_odo.iloc[0])
                if len(valid_odo) >= 2
                else np.nan,
                "median_reporting_gap_hours": group["gap_hours"].median(),
                "max_reporting_gap_hours": group["gap_hours"].max(),
                "negative_operating_hour_deltas": int(group["operating_hours_reset"].sum()),
                "negative_odometer_deltas": int(group["odometer_reset"].sum()),
            }
        )
    return pd.DataFrame(rows)


def _telemetry_window_features(
    telemetry: pd.DataFrame, asset_id: str, sample_date: pd.Timestamp
) -> dict[str, float]:
    missing = {
        "telemetry_available": 0.0,
        "operating_hours_7d": np.nan,
        "operating_hours_30d": np.nan,
        "operating_hours_90d": np.nan,
        "distance_30d": np.nan,
        "mean_utilization_30d": np.nan,
        "telemetry_age_days": np.nan,
    }
    if telemetry is None or telemetry.empty:
        return missing

    asset = telemetry[
        (telemetry["asset_id"].astype(str) == str(asset_id))
        & (telemetry["event_time"] <= sample_date)
    ].sort_values("event_time")
    if asset.empty:
        return missing

    result: dict[str, float] = {"telemetry_available": 1.0}
    latest = asset.iloc[-1]
    result["telemetry_age_days"] = float((sample_date - latest["event_time"]).total_seconds() / 86400)
    for days in [7, 30, 90]:
        window = asset[asset["event_time"] >= sample_date - timedelta(days=days)]
        result[f"operating_hours_{days}d"] = float(
            window["operating_hours_delta"].clip(lower=0).sum(min_count=1)
        )
        if days == 30:
            result["distance_30d"] = float(window["odometer_delta"].clip(lower=0).sum(min_count=1))
            result["mean_utilization_30d"] = float(window["utilization_rate"].median())
    return result


def build_training_table(
    sos: pd.DataFrame,
    telemetry: pd.DataFrame,
    work_orders: pd.DataFrame | None = None,
    horizon_days: int = 30,
) -> pd.DataFrame:
    """Create sample-level features labelled only by future confirmed corrective WOs.

    No label is inferred from S.O.S severity or laboratory measurements. Doing so
    would train the model to reproduce its own inputs rather than predict an
    independently observed maintenance outcome.
    """
    if work_orders is None or work_orders.empty:
        return pd.DataFrame()

    label_source = work_orders.get(
        "failure_label_source", pd.Series("text_inferred", index=work_orders.index)
    )
    wo_df = work_orders[label_source.eq("explicit")].copy()
    if wo_df.empty:
        return pd.DataFrame()

    # A missing work order can only mean "no event" when the asset is actually
    # covered by the outcome extract. Assets with no WO coverage remain useful
    # at scoring time, but must not silently become negative training examples.
    covered_assets = set(wo_df["asset_id"].dropna().astype(str))
    scoring = build_scoring_table(sos, telemetry, wo_df)
    scoring = scoring[scoring["asset_id"].astype(str).isin(covered_assets)].copy()
    if scoring.empty:
        return pd.DataFrame()

    stated_observation_end = wo_df.get(
        "observation_end_date", pd.Series(pd.NaT, index=wo_df.index)
    )
    stated_by_asset = (
        wo_df.assign(_observation_end=stated_observation_end)
        .groupby(wo_df["asset_id"].astype(str))["_observation_end"]
        .max()
    )
    stated_available = stated_observation_end.dropna()
    extract_observation_end = (
        stated_available.max()
        if not stated_available.empty
        else wo_df["opened_date"].max()
    )

    rows: list[dict[str, Any]] = []

    for _, sample in scoring.iterrows():
        sample_date = sample["sample_date"]
        asset_id = sample["asset_id"]
        component = sample["component"]
        if pd.isna(sample_date) or pd.isna(asset_id):
            continue
        sample_date = pd.Timestamp(sample_date)
        asset_observation_end = stated_by_asset.get(str(asset_id), pd.NaT)
        observation_end = (
            asset_observation_end
            if pd.notna(asset_observation_end)
            else extract_observation_end
        )
        # Do not label a recent sample as negative when the full future horizon
        # is not observable for this asset in the work-order extract.
        if pd.isna(observation_end) or sample_date + timedelta(days=int(horizon_days)) > observation_end:
            continue

        future = wo_df[
            (wo_df["asset_id"].astype(str) == str(asset_id))
            & (wo_df["opened_date"] > sample_date)
            & (wo_df["opened_date"] <= sample_date + timedelta(days=int(horizon_days)))
            & wo_df["is_corrective"]
            & wo_df["confirmed_failure"]
        ]
        if pd.notna(component) and wo_df["component"].notna().any():
            future = future[future["component"].astype(str) == str(component)]

        row = sample.to_dict()
        row["corrective_wo_within_horizon"] = int(not future.empty)
        row["horizon_days"] = int(horizon_days)
        rows.append(row)

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("sample_date").reset_index(drop=True)


def build_scoring_table(
    sos: pd.DataFrame,
    telemetry: pd.DataFrame,
    work_orders: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build past-only features for every scoreable S.O.S observation.

    Telemetry and work-order history are optional feature sources. Missing
    modalities are represented explicitly instead of blocking a prediction.
    Future work orders are never consulted here.
    """
    samples = add_sos_trends(sos)
    if samples.empty:
        return pd.DataFrame()

    samples["lab_history_samples"] = (
        samples.groupby(["asset_id", "component"], dropna=False).cumcount() + 1
    )
    telemetry_groups = {
        str(asset_id): group.sort_values("event_time")
        for asset_id, group in telemetry.groupby("asset_id", dropna=False)
    } if telemetry is not None and not telemetry.empty else {}

    wo_df = work_orders if work_orders is not None else pd.DataFrame()
    wo_assets = (
        set(wo_df["asset_id"].dropna().astype(str))
        if not wo_df.empty and "asset_id" in wo_df.columns else set()
    )

    rows: list[dict[str, Any]] = []
    for _, sample in samples.iterrows():
        sample_date = sample["sample_date"]
        asset_id = sample["asset_id"]
        component = sample["component"]
        if pd.isna(sample_date) or pd.isna(asset_id):
            continue
        sample_date = pd.Timestamp(sample_date)
        asset_key = str(asset_id)

        prior_count: float = np.nan
        if asset_key in wo_assets:
            prior = wo_df[
                (wo_df["asset_id"].astype(str) == asset_key)
                & (wo_df["opened_date"] < sample_date)
                & wo_df["is_corrective"]
                & wo_df["confirmed_failure"]
            ]
            if pd.notna(component) and wo_df["component"].notna().any():
                prior = prior[prior["component"].astype(str) == str(component)]
            prior_count = float(len(prior))

        measurements = [sample.get(column) for column in RAW_MEASUREMENT_COLUMNS]
        measurement_count = int(sum(pd.notna(value) for value in measurements))
        interpretation_text = str(sample.get("interpretation_text", "") or "").strip()
        row: dict[str, Any] = {
            "source_row": int(sample.get("source_row", len(rows))),
            "sample_number": sample["sample_number"],
            "asset_id": asset_id,
            "sample_date": sample_date,
            "machine_model": sample["machine_model"],
            "component": component,
            "equipment_hours": sample["equipment_hours"],
            "fluid_hours": sample["fluid_hours"],
            "days_since_previous_sample": sample["days_since_previous_sample"],
            "lab_history_samples": int(sample["lab_history_samples"]),
            "lab_measurement_count": measurement_count,
            "lab_numeric_available": float(measurement_count > 0),
            "lab_interpretation_code": str(sample.get("interpretation_code", "") or "").upper(),
            "lab_text_available": float(bool(interpretation_text)),
            "lab_text_length_log": float(np.log1p(len(interpretation_text))),
            "work_order_history_available": float(asset_key in wo_assets),
            "prior_corrective_wo_count": prior_count,
        }
        for column in RAW_MEASUREMENT_COLUMNS:
            row[column] = sample[column]
            row[f"{column}_delta"] = sample[f"{column}_delta"]
            row[f"{column}_rate_100h"] = sample[f"{column}_rate_100h"]
            row[f"{column}_mean_3"] = sample[f"{column}_mean_3"]
        text_signal_count = 0.0
        for feature, pattern in TEXT_SIGNAL_PATTERNS.items():
            matched = float(bool(re.search(pattern, interpretation_text, flags=re.IGNORECASE)))
            row[feature] = matched
            text_signal_count += matched
        row["lab_text_signal_count"] = text_signal_count
        asset_telemetry = telemetry_groups.get(asset_key, pd.DataFrame())
        row.update(_telemetry_window_features(asset_telemetry, asset_key, sample_date))
        rows.append(row)

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("sample_date").reset_index(drop=True)
