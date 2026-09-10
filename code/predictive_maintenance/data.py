from __future__ import annotations

from pathlib import Path
import re
from typing import IO, Any

import numpy as np
import pandas as pd


TabularSource = str | Path | IO[bytes] | IO[str]


def _is_scalar_missing(value: Any) -> bool:
    """Return True for None/pandas/numpy missing scalars without ambiguous truth tests."""
    if value is None:
        return True
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        return False
    return isinstance(missing, (bool, np.bool_)) and bool(missing)


def safe_text(value: Any, default: str = "") -> str:
    """Convert a spreadsheet scalar to text while treating blank/NA as empty."""
    return default if _is_scalar_missing(value) else str(value)


def safe_bool(value: Any, default: bool = False) -> bool:
    """Convert common spreadsheet boolean values without evaluating pd.NA."""
    if _is_scalar_missing(value):
        return default
    if isinstance(value, str):
        normalized = value.strip().upper()
        if normalized in {"TRUE", "T", "YES", "Y", "1"}:
            return True
        if normalized in {"FALSE", "F", "NO", "N", "0", ""}:
            return False
    try:
        return bool(value)
    except (TypeError, ValueError):
        return default


def safe_float(value: Any, default: float = 0.0) -> float:
    """Convert a spreadsheet scalar to a finite float with a safe fallback."""
    if _is_scalar_missing(value):
        return default
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return default
    return converted if np.isfinite(converted) else default


def load_table(source: TabularSource) -> pd.DataFrame:
    """Load an Excel or CSV table from a path or uploaded file object."""
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


COMPONENT_ALIASES = {
    "ENGINE": "ENG",
    "ENG": "ENG",
    "HYDRAULIC": "HYD",
    "HYDRAULICS": "HYD",
    "HYD": "HYD",
    "TRANSMISSION": "TRANS",
    "TRANS": "TRANS",
    "TR": "TRANS",
}


def _clean_component(series: pd.Series) -> pd.Series:
    """Normalize common component aliases before S.O.S/WO matching."""
    cleaned = (
        _clean_identifier(series)
        .str.upper()
        .str.replace(r"[^A-Z0-9]+", "_", regex=True)
        .str.strip("_")
    )
    return cleaned.replace(COMPONENT_ALIASES)


def _column_key(value: Any) -> str:
    """Normalize spreadsheet headers for case/punctuation-insensitive matching."""
    return re.sub(r"[^a-z0-9]", "", str(value).strip().lower())


def _resolve_column_name(df: pd.DataFrame, candidates: list[str]) -> Any | None:
    for column in candidates:
        if column in df.columns:
            return column
    normalized_columns: dict[str, list[Any]] = {}
    for column in df.columns:
        normalized_columns.setdefault(_column_key(column), []).append(column)
    for candidate in candidates:
        matched = normalized_columns.get(_column_key(candidate), [])
        if len(matched) > 1:
            raise ValueError(
                f"Ambiguous spreadsheet headers {matched!r}; rename one column before analysis."
            )
        if matched:
            return matched[0]
    return None


def _first_existing(df: pd.DataFrame, candidates: list[str], default: Any = pd.NA) -> pd.Series:
    column = _resolve_column_name(df, candidates)
    if column is not None:
        return df[column]
    return pd.Series(default, index=df.index)


def _numeric_alias(df: pd.DataFrame, candidates: list[str]) -> pd.Series:
    return pd.to_numeric(_first_existing(df, candidates, np.nan), errors="coerce")


def enrich_sos_with_test_details(
    sos_raw: pd.DataFrame,
    test_details_raw: pd.DataFrame | None,
) -> pd.DataFrame:
    """Join long-format laboratory results onto one-row-per-sample S.O.S history.

    TMS exports repeat one result across several synchronization snapshots. We
    keep the latest snapshot for each result identifier, pivot supported tests,
    and fill (never overwrite) measurements already present in the history file.
    """
    if test_details_raw is None or test_details_raw.empty:
        return sos_raw.copy()

    history_sample_column = _resolve_column_name(
        sos_raw, ["SampleNum", "SampleNumber", "sample_id", "Id"]
    )
    detail_sample_column = _resolve_column_name(
        test_details_raw, ["SampleNum", "SampleNumber", "sample_id", "Id"]
    )
    result_name_column = _resolve_column_name(
        test_details_raw, ["resultName", "ResultName", "resultAliasName"]
    )
    result_value_column = _resolve_column_name(
        test_details_raw,
        ["resultFormattedEntry", "ResultFormattedEntry", "ResultValue", "Value"],
    )
    if any(
        column is None
        for column in [history_sample_column, detail_sample_column, result_name_column, result_value_column]
    ):
        raise ValueError(
            "S.O.S test-detail enrichment requires sampleNum, resultName and "
            "resultFormattedEntry columns, plus sampleNum in the history file."
        )

    details = test_details_raw.copy()
    details["_sample_key"] = _clean_identifier(details[detail_sample_column])
    details["_result_name"] = (
        details[result_name_column].astype("string").str.strip().str.upper()
    )
    details["_numeric_value"] = pd.to_numeric(
        details[result_value_column], errors="coerce"
    )
    interpretation_column = _resolve_column_name(
        details, ["interpText", "InterpretationText", "Comment", "Comments"]
    )
    details["_analysis_prevented"] = (
        details[interpretation_column]
        .astype("string")
        .str.contains(
            r"prevent(?:s|ed)?\s+(?:most\s+)?analysis|"
            r"unable\s+to\s+(?:complete|perform|analyse|analyze)|"
            r"analysis\s+(?:was\s+)?not\s+(?:completed|performed)",
            case=False,
            regex=True,
            na=False,
        )
        if interpretation_column is not None
        else False
    )
    units_column = _resolve_column_name(details, ["resultUnits", "Units", "Unit"])
    details["_units"] = (
        details[units_column].astype("string").str.strip().str.upper()
        if units_column is not None
        else ""
    )
    modified_column = _resolve_column_name(
        details,
        ["SynapseModifiedDateTime", "resultChangedOn", "ModifiedOn"],
    )
    details["_modified"] = (
        pd.to_datetime(details[modified_column], errors="coerce")
        if modified_column is not None
        else pd.NaT
    )
    result_number_column = _resolve_column_name(
        details, ["resultNumber", "ResultNumber", "ResultId"]
    )
    details = details.sort_values("_modified", na_position="first")
    duplicate_key = ["_sample_key", "_result_name"]
    if result_number_column is not None:
        details["_result_id"] = _clean_identifier(details[result_number_column])
        duplicate_key = ["_sample_key", "_result_id"]
    details = details.drop_duplicates(duplicate_key, keep="last")

    # Some TMS exports populate unperformed infrared-panel results with a
    # numeric zero when contamination prevented the analysis. A zero in this
    # situation is a system placeholder, not a measured concentration. Keep
    # the laboratory text/rule evidence, but exclude those placeholders from
    # numerical trend calculations.
    prevented_placeholder = (
        details["_analysis_prevented"] & details["_numeric_value"].eq(0)
    )
    details.loc[prevented_placeholder, "_numeric_value"] = np.nan

    result_mapping = {
        "FE": "Fe",
        "CU": "Cu",
        "AL": "Al",
        "CR": "Cr",
        "PB": "Pb",
        "SI": "Si",
        "V100": "Viscosity",
        "OXI": "Oxidation",
        "ST": "Soot",
        "IBN": "TBN",
        "FUEL": "FuelDilution",
        "FUEL DILUTION": "FuelDilution",
        "WATER": "WaterPct",
        "TOTAL WATER": "WaterPct",
    }
    details["_output_column"] = details["_result_name"].map(result_mapping)

    # Do not silently mix incompatible units in one trend. Blank units remain
    # accepted for legacy exports, while known fields with explicit units must
    # use an equivalent representation.
    allowed_units = {
        "Fe": {"", "PPM", "MG/KG"},
        "Cu": {"", "PPM", "MG/KG"},
        "Al": {"", "PPM", "MG/KG"},
        "Cr": {"", "PPM", "MG/KG"},
        "Pb": {"", "PPM", "MG/KG"},
        "Si": {"", "PPM", "MG/KG"},
        "Viscosity": {"", "CST", "MM2/S", "MM²/S"},
        "WaterPct": {"", "PPM", "MG/KG", "%", "PCT", "PERCENT"},
        "FuelDilution": {"", "%", "PCT", "PERCENT"},
        "Soot": {"", "%", "PCT", "PERCENT"},
    }
    explicit_unit_supported = details.apply(
        lambda row: (
            pd.isna(row["_output_column"])
            or row["_output_column"] not in allowed_units
            or row["_units"] in allowed_units[row["_output_column"]]
        ),
        axis=1,
    )
    details.loc[~explicit_unit_supported, "_numeric_value"] = np.nan
    supported = details.dropna(
        subset=["_sample_key", "_output_column", "_numeric_value"]
    ).copy()
    if supported.empty:
        return sos_raw.copy()

    # The test-detail export reports water in ppm; the model stores a fraction
    # expressed as percent, so 10,000 ppm equals 1 percent.
    water_ppm = supported["_output_column"].eq("WaterPct") & supported["_units"].isin(
        ["PPM", "MG/KG"]
    )
    supported.loc[water_ppm, "_numeric_value"] = (
        supported.loc[water_ppm, "_numeric_value"] / 10_000.0
    )
    # Prefer the explicit WATER result when both WATER and TOTAL WATER exist.
    supported["_measurement_priority"] = supported["_result_name"].map(
        {"TOTAL WATER": 1, "WATER": 2}
    ).fillna(1)
    supported = supported.sort_values(["_modified", "_measurement_priority"])
    supported = supported.drop_duplicates(
        ["_sample_key", "_output_column"], keep="last"
    )
    wide = supported.pivot(
        index="_sample_key", columns="_output_column", values="_numeric_value"
    )

    enriched = sos_raw.copy()
    sample_keys = _clean_identifier(enriched[history_sample_column])
    for output_column in result_mapping.values():
        if output_column not in wide.columns:
            continue
        mapped_values = sample_keys.map(wide[output_column])
        existing_column = _resolve_column_name(enriched, [output_column])
        if existing_column is None:
            enriched[output_column] = mapped_values
        else:
            existing_values = pd.to_numeric(enriched[existing_column], errors="coerce")
            enriched[existing_column] = existing_values.where(
                existing_values.notna(), mapped_values
            )
    return enriched


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
    out["component"] = _clean_component(
        _first_existing(out, ["Compartment", "Component", "component"])
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
            "schema_issue": None,
        }
        return empty_df, quality
    raw = df.copy()
    asset_column = _resolve_column_name(
        raw,
        [
            "TMSAssetID", "EquipmentHeader_EquipmentID", "EquipmentId",
            "EquipmentID", "EquipNum", "AssetId",
        ],
    )
    event_column = _resolve_column_name(
        raw,
        [
            "Location_Datetime", "CumulativeOperatingHours_Datetime", "Timestamp",
            "EventDateTime", "DateTime", "EventTime", "ReadingDate", "event_time",
        ],
    )
    if asset_column is None or event_column is None:
        empty, quality = prepare_telemetry(None)
        quality["rows_before_deduplication"] = int(len(raw))
        missing: list[str] = []
        if asset_column is None:
            missing.append("asset identifier")
        if event_column is None:
            missing.append("event timestamp")
        quality["schema_issue"] = (
            "Telemetry file was ignored because it has no " + " and ".join(missing) + "."
        )
        return empty, quality
    raw["asset_id"] = _clean_identifier(raw[asset_column])
    raw["serial_number"] = _clean_identifier(
        _first_existing(raw, ["TMSSerialNum", "EquipmentHeader_SerialNumber", "SerialNumber"])
    )
    raw["machine_model"] = _clean_identifier(
        _first_existing(raw, ["EquipmentHeader_Model", "EquipmentModel", "Model"])
    )
    raw["event_time"] = pd.to_datetime(raw[event_column], errors="coerce")
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
        "schema_issue": None,
    }
    return out, quality


def prepare_work_orders(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize a work-order export and derive conservative corrective/failure flags."""
    out = df.copy()
    out["asset_id"] = _clean_identifier(
        _first_existing(out, ["EquipmentId", "EquipNum", "AssetId", "EquipmentID"])
    )
    out["component"] = _clean_component(
        _first_existing(out, ["Component", "Compartment", "component"])
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
