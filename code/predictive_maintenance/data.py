from __future__ import annotations

from pathlib import Path
from typing import IO, Any

import numpy as np
import pandas as pd


TabularSource = str | Path | IO[bytes] | IO[str]


def resolve_file_path(source: TabularSource) -> TabularSource:
    if isinstance(source, (str, Path)):
        p = Path(source)
        if not p.exists():
            candidates = ["SOSFluidAnalysisSample.xlsx", "SosFluidSample.xlsx"]
            for cand in candidates:
                alt = p.parent / cand
                if alt.exists():
                    return alt
            if p.name == "WorkOrderSample.csv":
                alt = p.parent.parent / "demo" / "Demo_WorkOrders.xlsx"
                if alt.exists():
                    return alt
    return source


def load_table(source: TabularSource) -> pd.DataFrame:
    """Load an Excel or CSV table from a path or uploaded file object."""
    source = resolve_file_path(source)
    name = str(getattr(source, "name", source)).lower()
    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(source)
    if name.endswith(".csv"):
        return pd.read_csv(source)
    raise ValueError(f"Unsupported tabular file: {name}. Use .xlsx, .xls or .csv")


def _clean_identifier(series: pd.Series) -> pd.Series:
    return (
        series.astype("string")
        .str.strip()
        .replace({"": pd.NA, "nan": pd.NA, "None": pd.NA, "<NA>": pd.NA})
    )


def _first_existing(df: pd.DataFrame, candidates: list[str], default: Any = pd.NA) -> pd.Series:
    for column in candidates:
        if column in df.columns:
            return df[column]
    return pd.Series(default, index=df.index)


def _numeric_alias(df: pd.DataFrame, candidates: list[str]) -> pd.Series:
    return pd.to_numeric(_first_existing(df, candidates, np.nan), errors="coerce")


def _parse_calendar_dates(series: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Parse calendar dates without silently turning time-only cells into dates."""
    raw_text = series.astype("string").str.strip()
    time_only = raw_text.str.fullmatch(
        r"\d{1,2}:\d{2}(?::\d{2}(?:\.\d+)?)?", na=False
    )
    missing = series.isna() | raw_text.isna() | raw_text.eq("")
    parsed = pd.to_datetime(series.mask(time_only), errors="coerce")
    out_of_range = parsed.notna() & ((parsed.dt.year < 1950) | (parsed.dt.year > 2035))
    parsed = parsed.mask(out_of_range)

    issue = pd.Series(pd.NA, index=series.index, dtype="string")
    issue.loc[missing] = "Missing calendar date"
    issue.loc[time_only] = "Time-only value; calendar date is missing"
    issue.loc[~missing & ~time_only & parsed.isna()] = "Unparseable calendar date"
    issue.loc[out_of_range] = "Calendar date is outside the accepted 1950–2035 range"
    return parsed, issue


def prepare_sos(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize real or demo S.O.S fluid data into a canonical schema."""
    out = df.copy()
    out["asset_id"] = _clean_identifier(
        _first_existing(out, ["EquipNum", "EquipmentId", "EquipmentID", "AssetId", "AssetName"])
    )
    out["serial_number"] = _clean_identifier(
        _first_existing(out, ["SerialNum", "SerialNumber", "EquipmentSerialNumber"])
    )
    out["machine_model"] = _clean_identifier(
        _first_existing(out, ["EqpModel", "EquipmentModel", "Model"])
    )
    out["component"] = (
        _clean_identifier(_first_existing(out, ["Compartment", "Component", "component"]))
        .str.upper()
        .str.replace(r"\s+", "_", regex=True)
    )
    sample_date_raw = _first_existing(out, ["DateSampled", "SampleDate", "sample_date"])
    out["sample_date_raw"] = sample_date_raw.astype("string")
    out["sample_date"], out["date_quality_issue"] = _parse_calendar_dates(sample_date_raw)
    out["sample_number"] = _clean_identifier(
        _first_existing(out, ["SampleNum", "SampleNumber", "sample_id", "Id"])
    )
    out["interpretation_code"] = (
        _clean_identifier(_first_existing(out, ["OverallInterp", "Severity", "InterpretationCode"]))
        .str.upper()
    )
    out["interpretation_text"] = (
        _first_existing(out, ["InterpText", "InterpretationText", "Recommendation"], "")
        .fillna("")
        .astype(str)
    )
    out["equipment_hours"] = _numeric_alias(out, ["CMeter", "MeterHours", "EquipmentHours"])
    out["fluid_hours"] = _numeric_alias(out, ["CMeterFluid", "OilHours", "FluidHours"])
    out["fluid_changed"] = _clean_identifier(_first_existing(out, ["FluidChanged"], pd.NA))
    out["filter_changed"] = _clean_identifier(_first_existing(out, ["FilterChanged"], pd.NA))
    out["high_priority"] = _clean_identifier(_first_existing(out, ["HighPriority"], pd.NA))
    out["status"] = _clean_identifier(_first_existing(out, ["SampleStatusNew", "Status", "SampleStatus", "AlertStatus"], "New"))
    out["wo_id"] = _clean_identifier(_first_existing(out, ["WorkOrderId", "WO_ID", "ShopJobNo"], pd.NA))
    out["wo_status"] = _clean_identifier(_first_existing(out, ["WOStatus", "WorkOrderStatus"], pd.NA))
    out["site_name"] = _clean_identifier(_first_existing(out, ["SiteName", "EqpJobsite", "JobsiteDesc", "SiteId"], "Unknown"))
    out["is_invalid_date"] = out["date_quality_issue"].notna()

    measurement_aliases = {
        "iron_ppm": ["Fe", "Iron", "iron_ppm", "Fe_ppm"],
        "copper_ppm": ["Cu", "Copper", "copper_ppm", "Cu_ppm"],
        "aluminium_ppm": ["Al", "Aluminum", "Aluminium", "aluminium_ppm", "Al_ppm"],
        "chromium_ppm": ["Cr", "Chromium", "chromium_ppm", "Cr_ppm"],
        "lead_ppm": ["Pb", "Lead", "lead_ppm", "Pb_ppm"],
        "silicon_ppm": ["Si", "Silicon", "silicon_ppm", "Si_ppm"],
        "water_pct": ["Water", "WaterPct", "water_pct"],
        "fuel_dilution_pct": ["FuelDilution", "FuelPct", "fuel_dilution_pct"],
        "soot_pct": ["Soot", "SootPct", "soot_pct"],
        "viscosity_cst": ["Viscosity", "ViscosityCst", "viscosity_cst"],
        "oxidation": ["Oxidation", "oxidation"],
        "tbn": ["TBN", "tbn"],
    }
    for canonical, aliases in measurement_aliases.items():
        out[canonical] = _numeric_alias(out, aliases)

    out["asset_id"] = out["asset_id"].fillna(out["serial_number"])
    out["source_row"] = np.arange(len(out))
    return out


def prepare_telemetry(df: pd.DataFrame | None) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Normalize, deduplicate and derive safe time-series telemetry features."""
    if df is None or df.empty:
        empty_df = pd.DataFrame(
            columns=[
                "asset_id", "serial_number", "machine_model", "event_time",
                "operating_hours", "odometer", "gap_hours", "operating_hours_delta",
                "odometer_delta", "distance_per_operating_hour", "utilization_rate",
                "telemetry_anomaly", "telemetry_anomaly_score"
            ]
        )
        quality = {
            "rows_before_deduplication": 0,
            "rows_after_deduplication": 0,
            "duplicate_asset_timestamp_rows": 0,
            "unique_assets": 0,
            "date_start": pd.NaT,
            "date_end": pd.NaT,
            "stale_or_constant_fields": [],
        }
        return empty_df, quality
    raw = df.copy()
    raw["asset_id"] = _clean_identifier(
        _first_existing(raw, ["TMSAssetID", "EquipmentHeader_EquipmentID", "EquipmentId", "AssetId"])
    )
    raw["serial_number"] = _clean_identifier(
        _first_existing(raw, ["TMSSerialNum", "EquipmentHeader_SerialNumber", "SerialNumber"])
    )
    raw["machine_model"] = _clean_identifier(
        _first_existing(raw, ["EquipmentHeader_Model", "EquipmentModel", "Model"])
    )
    raw["event_time"] = pd.to_datetime(
        _first_existing(
            raw,
            ["Location_Datetime", "CumulativeOperatingHours_Datetime", "Timestamp", "event_time"],
        ),
        errors="coerce",
    )
    raw["modified_time"] = pd.to_datetime(
        _first_existing(raw, ["SynapseModifiedDateTime", "ModifiedOn"], pd.NaT), errors="coerce"
    )
    raw["operating_hours"] = _numeric_alias(
        raw, ["CumulativeOperatingHours_Hour", "OperatingHours", "operating_hours"]
    )
    raw["odometer"] = _numeric_alias(raw, ["Distance_Odometer", "Odometer", "odometer"])
    raw["idle_hours"] = _numeric_alias(
        raw, ["CumulativeIdleHours_Hour", "IdleHours", "idle_hours"]
    )
    raw["fuel_used"] = _numeric_alias(raw, ["FuelUsed_FuelConsumed", "FuelUsed", "fuel_used"])
    raw["latitude"] = _numeric_alias(raw, ["Location_Latitude", "Latitude"])
    raw["longitude"] = _numeric_alias(raw, ["Location_Longitude", "Longitude"])

    rows_before = len(raw)
    duplicate_rows = int(raw.duplicated(["asset_id", "event_time"], keep=False).sum())
    raw = raw.sort_values(["asset_id", "event_time", "modified_time"], na_position="first")
    out = raw.drop_duplicates(["asset_id", "event_time"], keep="last").copy()
    out = out.sort_values(["asset_id", "event_time"]).reset_index(drop=True)

    grouped = out.groupby("asset_id", dropna=False, group_keys=False)
    out["gap_hours"] = grouped["event_time"].diff().dt.total_seconds().div(3600)
    out["operating_hours_delta"] = grouped["operating_hours"].diff()
    out["odometer_delta"] = grouped["odometer"].diff()
    out["idle_hours_delta"] = grouped["idle_hours"].diff()
    out["fuel_used_delta"] = grouped["fuel_used"].diff()
    out["operating_hours_reset"] = out["operating_hours_delta"].lt(0)
    out["odometer_reset"] = out["odometer_delta"].lt(0)
    out["distance_per_operating_hour"] = out["odometer_delta"].div(
        out["operating_hours_delta"].where(out["operating_hours_delta"].gt(0))
    )
    out["utilization_rate"] = out["operating_hours_delta"].div(
        out["gap_hours"].where(out["gap_hours"].gt(0))
    )

    stale_fields: list[str] = []
    for column in ["EngineStatus_Running", "CumulativeIdleHours_Hour", "FuelUsed_FuelConsumed"]:
        if column in df.columns and df[column].nunique(dropna=False) <= 1:
            stale_fields.append(column)

    quality = {
        "rows_before_deduplication": rows_before,
        "rows_after_deduplication": len(out),
        "duplicate_asset_timestamp_rows": duplicate_rows,
        "unique_assets": int(out["asset_id"].nunique(dropna=True)),
        "date_start": out["event_time"].min(),
        "date_end": out["event_time"].max(),
        "stale_or_constant_fields": stale_fields,
    }
    return out, quality


def prepare_work_orders(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize a work-order export and derive conservative corrective/failure flags."""
    out = df.copy()
    out["asset_id"] = _clean_identifier(
        _first_existing(out, ["EquipmentId", "EquipNum", "AssetId", "EquipmentID"])
    )
    out["component"] = (
        _clean_identifier(_first_existing(out, ["Component", "Compartment", "component"]))
        .str.upper()
        .str.replace(r"\s+", "_", regex=True)
    )
    out["wo_id"] = _clean_identifier(
        _first_existing(out, ["WorkOrderId", "WO_ID", "wo_id", "Id"])
    )
    out["opened_date"] = pd.to_datetime(
        _first_existing(out, ["OpenedDate", "OpenDate", "WorkOrderDate", "opened_date"]),
        errors="coerce",
    )
    out["closed_date"] = pd.to_datetime(
        _first_existing(out, ["ClosedDate", "CloseDate", "closed_date"], pd.NaT), errors="coerce"
    )
    out["observation_end_date"] = pd.to_datetime(
        _first_existing(
            out,
            ["ObservationEndDate", "DataThroughDate", "ExtractEndDate"],
            pd.NaT,
        ),
        errors="coerce",
    )
    out["wo_type"] = (
        _clean_identifier(_first_existing(out, ["WorkOrderType", "WOType", "Type"], ""))
        .fillna("")
        .str.upper()
    )
    text_columns = [
        column
        for column in ["ProblemDescription", "Description", "ActionTaken", "TechnicianNotes", "FailureCode"]
        if column in out.columns
    ]
    if text_columns:
        out["wo_text"] = out[text_columns].fillna("").astype(str).agg(" ".join, axis=1)
    else:
        out["wo_text"] = ""
    explicit_raw = _first_existing(out, ["FailureConfirmed", "ConfirmedFailure"], np.nan)
    explicit_text = explicit_raw.astype("string").str.strip().str.upper()
    explicit_failure = pd.to_numeric(explicit_raw, errors="coerce")
    explicit_failure = explicit_failure.fillna(
        explicit_text.map({"TRUE": 1, "T": 1, "Y": 1, "YES": 1,
                           "FALSE": 0, "F": 0, "N": 0, "NO": 0})
    )
    corrective_text = out["wo_type"].str.contains("CORRECT|BREAKDOWN|UNSCHEDULED", regex=True)
    failure_text = out["wo_text"].str.contains(
        r"fail|broken|damage|wear|replace|overhaul|leak|bearing|gear", case=False, regex=True
    )
    out["is_corrective"] = corrective_text | failure_text
    out["failure_label_source"] = np.where(
        explicit_failure.notna(), "explicit", "text_inferred"
    )
    out["confirmed_failure"] = explicit_failure.fillna(failure_text.astype(int)).astype(bool)
    return out


def asset_intersection(sos: pd.DataFrame, telemetry: pd.DataFrame) -> set[str]:
    return set(sos["asset_id"].dropna().astype(str)) & set(telemetry["asset_id"].dropna().astype(str))
