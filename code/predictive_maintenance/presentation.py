from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .features import RAW_MEASUREMENT_COLUMNS, add_sos_trends
from .pipeline import (
    MODE_ALERT,
    MODE_CONDITION,
    MODE_PREDICTION,
    P1_IMMEDIATE,
    P1_WO_TRACKING,
    P2_REPEATED,
    P3_ACTION,
    TIER_CLOSED,
    TIER_IN_PROGRESS,
)


PRIORITY_ORDER = {
    P1_IMMEDIATE: 0,
    P1_WO_TRACKING: 1,
    P2_REPEATED: 2,
    P3_ACTION: 3,
    TIER_IN_PROGRESS: 4,
    TIER_CLOSED: 5,
}

PRIORITY_LABELS = {
    P1_IMMEDIATE: "P1 — Inspect now",
    P1_WO_TRACKING: "P1 — Verify work order",
    P2_REPEATED: "P2 — Repeated unresolved finding",
    P3_ACTION: "P3 — Engineering review",
    TIER_IN_PROGRESS: "In progress",
    TIER_CLOSED: "Closed laboratory record",
}

COMPONENT_LABELS = {
    "ENG": "Engine",
    "ENGINE": "Engine",
    "HYD": "Hydraulic system",
    "HYDRAULIC": "Hydraulic system",
    "TR": "Transmission",
    "TRANS": "Transmission",
    "TRANSMISSION": "Transmission",
    "DIFF_FR": "Front differential",
    "DIFF_RR": "Rear differential",
    "DIF": "Differential",
    "FINAL_DRIVE": "Final drive",
    "FD_LR": "Left-rear final drive",
    "FD_RR": "Right-rear final drive",
}

MEASUREMENT_LABELS = {
    "iron_ppm": "Iron",
    "copper_ppm": "Copper",
    "aluminium_ppm": "Aluminium",
    "chromium_ppm": "Chromium",
    "lead_ppm": "Lead",
    "silicon_ppm": "Silicon",
    "water_pct": "Water",
    "fuel_dilution_pct": "Fuel dilution",
    "soot_pct": "Soot",
    "viscosity_cst": "Viscosity",
    "oxidation": "Oxidation",
    "tbn": "TBN",
}

MEASUREMENT_UNITS = {
    "iron_ppm": "ppm",
    "copper_ppm": "ppm",
    "aluminium_ppm": "ppm",
    "chromium_ppm": "ppm",
    "lead_ppm": "ppm",
    "silicon_ppm": "ppm",
    "water_pct": "%",
    "fuel_dilution_pct": "%",
    "soot_pct": "%",
    "viscosity_cst": "cSt",
    "oxidation": "",
    "tbn": "",
}

FEATURE_LABELS = {
    "iron_ppm": "Current iron concentration",
    "iron_ppm_delta": "Iron change since the previous sample",
    "iron_ppm_rate_100h": "Iron change per 100 fluid hours",
    "copper_ppm": "Current copper concentration",
    "copper_ppm_delta": "Copper change since the previous sample",
    "silicon_ppm": "Current silicon concentration",
    "silicon_ppm_delta": "Silicon change since the previous sample",
    "water_pct": "Current water concentration",
    "fuel_dilution_pct": "Current fuel dilution",
    "soot_pct": "Current soot concentration",
    "viscosity_cst": "Current viscosity",
    "prior_corrective_wo_count": "Previous confirmed corrective work orders",
    "operating_hours_30d": "Operating hours during the previous 30 days",
    "operating_hours_90d": "Operating hours during the previous 90 days",
    "mean_utilization_30d": "Recent utilisation",
    "telemetry_age_days": "Telemetry freshness",
}


def component_label(value: object) -> str:
    raw = str(value or "Unknown").strip().upper()
    return COMPONENT_LABELS.get(raw, raw.replace("_", " ").title())


def feature_label(value: object) -> str:
    raw = str(value or "").split("__")[-1]
    return FEATURE_LABELS.get(raw, raw.replace("_", " ").title())


def _has_value(value: object) -> bool:
    try:
        if pd.isna(value):
            return False
    except (TypeError, ValueError):
        pass
    return str(value).strip() not in {"", "<NA>", "nan", "None"}


def build_mode_summary(result: dict[str, Any], horizon_days: int) -> dict[str, Any]:
    mode = str(result.get("operating_mode", MODE_ALERT))
    enabled = bool(result.get("predictive_risk_enabled")) and mode == MODE_PREDICTION
    readiness = result.get("readiness", pd.DataFrame())
    blockers: list[str] = []
    if isinstance(readiness, pd.DataFrame) and not readiness.empty:
        blockers = readiness.loc[readiness["status"].eq("BLOCKED"), "criterion"].astype(str).tolist()

    if mode == MODE_PREDICTION and enabled:
        title = "Failure Prediction active"
        description = (
            f"The validated model estimates confirmed corrective-maintenance events within "
            f"the next {int(horizon_days)} days. Predictions support engineering review; they "
            "do not authorise automatic shutdown."
        )
    elif mode == MODE_CONDITION:
        title = "Condition Monitoring active"
        description = (
            "Repeated numerical laboratory history supports trend analysis. Failure probability "
            "remains unavailable until all outcome and validation requirements pass."
        )
    else:
        title = "Alert Management active"
        description = (
            "The available data supports laboratory triage and work-order follow-up. An AR code "
            "means Laboratory Action Required; it is not a predicted failure."
        )
    return {
        "mode": mode,
        "title": title,
        "description": description,
        "horizon_days": int(horizon_days),
        "prediction_enabled": enabled,
        "blocking_reasons": blockers,
    }


def _case_confidence(result: dict[str, Any], asset_id: str, component: str) -> tuple[str, str]:
    sos = result.get("sos", pd.DataFrame())
    telemetry = result.get("telemetry", pd.DataFrame())
    work_orders = result.get("work_orders")
    points = 0
    reasons: list[str] = []
    if isinstance(sos, pd.DataFrame) and not sos.empty:
        history = sos[
            sos["asset_id"].astype(str).eq(str(asset_id))
            & sos["component"].astype(str).eq(str(component))
        ]
        valid_history = history.loc[~history["is_invalid_date"].fillna(True)]
        numeric_rows = int(valid_history[RAW_MEASUREMENT_COLUMNS].notna().any(axis=1).sum())
        if numeric_rows >= 3:
            points += 2
            reasons.append(f"{numeric_rows} dated numerical samples")
        elif len(valid_history) >= 2:
            points += 1
            reasons.append(f"{len(valid_history)} dated samples")
        else:
            reasons.append("limited dated sample history")
    if isinstance(telemetry, pd.DataFrame) and not telemetry.empty and telemetry["asset_id"].astype(str).eq(str(asset_id)).any():
        points += 1
        reasons.append("matched telemetry available")
    else:
        reasons.append("no matched telemetry")
    if isinstance(work_orders, pd.DataFrame) and not work_orders.empty and work_orders["asset_id"].astype(str).eq(str(asset_id)).any():
        points += 1
        reasons.append("detailed work-order history available")
    else:
        reasons.append("detailed WO outcome unavailable")
    label = "High" if points >= 4 else "Medium" if points >= 2 else "Low"
    return label, "; ".join(reasons)


def build_case_table(result: dict[str, Any], horizon_days: int) -> pd.DataFrame:
    """Collapse sample rows into one operational case per asset and component."""
    cards = result.get("maintenance_action_cards", pd.DataFrame())
    if not isinstance(cards, pd.DataFrame) or cards.empty:
        return pd.DataFrame()

    frame = cards.copy().reset_index(drop=True)
    frame["_priority_rank"] = frame["priority_tier"].map(PRIORITY_ORDER).fillna(9)
    frame["_sample_date"] = pd.to_datetime(frame.get("sample_date"), errors="coerce")
    frame["_date_rank"] = frame["_sample_date"].fillna(pd.Timestamp("1900-01-01"))
    frame["_row_rank"] = np.arange(len(frame))

    rows: list[dict[str, Any]] = []
    for (asset_id, component), group in frame.groupby(["asset_id", "component"], dropna=False):
        open_group = group.loc[~group["priority_tier"].eq(TIER_CLOSED)]
        candidate_group = open_group if not open_group.empty else group
        selected = candidate_group.sort_values(
            ["_priority_rank", "_date_rank", "_row_rank"],
            ascending=[True, False, False],
        ).iloc[0]
        dated = group.loc[group["_sample_date"].notna()].sort_values("_sample_date")
        latest_date = dated["_sample_date"].max() if not dated.empty else pd.NaT
        probability_rows = group.assign(
            _probability=pd.to_numeric(group.get("failure_probability_pct"), errors="coerce")
        ).dropna(subset=["_probability"])
        if not probability_rows.empty:
            latest_prediction = probability_rows.sort_values(
                ["_date_rank", "_row_rank"], ascending=[False, False]
            ).iloc[0]
            probability = float(latest_prediction["_probability"])
            prediction_data_sources = str(latest_prediction.get("prediction_data_sources", "S.O.S."))
        else:
            probability = np.nan
            prediction_data_sources = str(selected.get("prediction_data_sources", "S.O.S."))
        evidence = selected.get("key_evidence", [])
        evidence_list = evidence if isinstance(evidence, list) else [str(evidence)]
        main_issue = next((str(item) for item in evidence_list if _has_value(item)), "Review laboratory interpretation")
        wo_id = selected.get("wo_id")
        has_wo = _has_value(wo_id)
        confidence, confidence_reason = _case_confidence(result, str(asset_id), str(component))
        priority = str(selected.get("priority_tier", P3_ACTION))
        due = "Today" if priority in {P1_IMMEDIATE, P1_WO_TRACKING} else "Within 3 days" if priority in {P2_REPEATED, P3_ACTION} else "Track" if priority == TIER_IN_PROGRESS else "Closed"
        rows.append({
            "case_id": f"{asset_id}::{component}",
            "priority_rank": int(selected.get("_priority_rank", 9)),
            "priority": PRIORITY_LABELS.get(priority, priority),
            "priority_code": priority,
            "asset_id": str(asset_id),
            "component": str(component),
            "component_name": component_label(component),
            "machine_model": str(selected.get("machine_model", "")),
            "site_name": str(selected.get("site_name", "Unknown")),
            "latest_sample_date": latest_date,
            "lab_status": str(selected.get("lab_status", "Unspecified")),
            "main_issue": main_issue,
            "failure_probability_pct": probability,
            "probability_display": f"{probability:.0f}% / {int(horizon_days)} days" if np.isfinite(probability) else "Not available",
            "prediction_data_sources": prediction_data_sources,
            "wo_id": str(wo_id) if has_wo else "",
            "wo_status_display": "Linked — verify outcome" if has_wo else "No linked WO",
            "required_action": str(selected.get("action_needed", "")).removeprefix("Suggested engineering review:").strip(),
            "due": due,
            "open_sample_count": int(len(open_group)),
            "total_sample_count": int(len(group)),
            "data_confidence": confidence,
            "confidence_reason": confidence_reason,
            "data_quality_status": str(selected.get("data_quality_status", "Valid")),
            "data_quality_issue": str(selected.get("data_quality_issue", "")),
            "sample_number": str(selected.get("sample_number", "")),
            "key_evidence": evidence_list,
        })
    return pd.DataFrame(rows).sort_values(
        ["priority_rank", "failure_probability_pct", "asset_id"],
        ascending=[True, False, True],
        na_position="last",
    ).reset_index(drop=True)


def build_fleet_summary(result: dict[str, Any], cases: pd.DataFrame, horizon_days: int) -> dict[str, Any]:
    sos = result.get("sos", pd.DataFrame())
    work_orders = result.get("work_orders")
    immediate = int(cases["priority_code"].isin([P1_IMMEDIATE, P1_WO_TRACKING]).sum()) if not cases.empty else 0
    active = cases.loc[~cases["priority_code"].eq(TIER_CLOSED)] if not cases.empty else cases
    lab_status = active.get("lab_status", pd.Series(dtype="string")).astype(str).str.lower()
    high = int(lab_status.str.contains("action required|severe", regex=True).sum())
    monitor = int(lab_status.str.contains("warning", regex=True).sum())
    normal = int(lab_status.eq("normal").sum())
    wo_linked = int(active["wo_id"].astype(str).str.len().gt(0).sum()) if not active.empty else 0
    no_wo_p1 = int(active.loc[active["priority_code"].eq(P1_IMMEDIATE), "wo_id"].astype(str).str.len().eq(0).sum()) if not active.empty else 0
    open_corrective = 0
    if isinstance(work_orders, pd.DataFrame) and not work_orders.empty:
        open_corrective = int((work_orders["is_corrective"] & work_orders["closed_date"].isna()).sum())

    confidence_order = {"High": 3, "Medium": 2, "Low": 1}
    confidence = "Not available"
    if not cases.empty:
        average = cases["data_confidence"].map(confidence_order).fillna(1).mean()
        confidence = "High" if average >= 2.6 else "Medium" if average >= 1.6 else "Low"

    mode = str(result.get("operating_mode", MODE_ALERT))
    samples = int(len(sos)) if isinstance(sos, pd.DataFrame) else 0
    assets = int(sos["asset_id"].nunique(dropna=True)) if samples else 0
    cases_count = int(len(cases))
    if immediate:
        recommendation = f"Review {immediate} P1 machine-component case(s) today; {wo_linked} active case(s) have a linked WO and {no_wo_p1} P1 case(s) need WO verification or creation."
    else:
        recommendation = "No P1 case was generated. Review monitoring cases and continue scheduled sampling."
    conclusion = (
        f"{assets} machines, {cases_count} machine-components and {samples} S.O.S samples were analysed. "
        f"{immediate} machine-component case(s) require immediate review. Current active cases include "
        f"{high} high-severity, {monitor} monitor and {normal} normal laboratory results."
    )
    if mode == MODE_PREDICTION:
        conclusion += f" Failure Prediction is active for a {int(horizon_days)}-day horizon."
    else:
        conclusion += f" The application is operating in {mode} mode; predictive probabilities are not available."
    return {
        "assets": assets,
        "machine_components": cases_count,
        "samples": samples,
        "immediate_action": immediate,
        "high": high,
        "monitor": monitor,
        "normal": normal,
        "wo_linked_active": wo_linked,
        "open_corrective_wos": open_corrective,
        "analysis_confidence": confidence,
        "conclusion": conclusion,
        "recommendation": recommendation,
    }


def build_validation_summary(result: dict[str, Any]) -> dict[str, Any] | None:
    model = result.get("trained_model")
    if model is None:
        return None
    metrics = dict(getattr(model, "metrics", {}) or {})
    matrix = metrics.get("confusion_matrix", [[0, 0], [0, 0]])
    try:
        tn, fp = int(matrix[0][0]), int(matrix[0][1])
        fn, tp = int(matrix[1][0]), int(matrix[1][1])
    except (TypeError, ValueError, IndexError):
        tn = fp = fn = tp = 0
    test_rows = int(metrics.get("test_rows", tn + fp + fn + tp))
    threshold = float(metrics.get("decision_threshold", 0.5))
    return {
        "model_name": str(getattr(model, "name", "model")).replace("_", " ").title(),
        "test_rows": test_rows,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "true_positive": tp,
        "decision_threshold": threshold,
        "precision": float(metrics.get("precision_at_threshold", np.nan)),
        "recall": float(metrics.get("recall_at_threshold", np.nan)),
        "f1": float(metrics.get("f1_at_threshold", np.nan)),
        "roc_auc": metrics.get("roc_auc"),
        "average_precision": float(metrics.get("average_precision", np.nan)),
        "baseline_average_precision": float(
            metrics.get("baseline_average_precision", np.nan)
        ),
        "average_precision_lift": float(
            metrics.get("average_precision_lift", np.nan)
        ),
        "brier_score": float(metrics.get("brier_score", np.nan)),
        "baseline_brier_score": float(
            metrics.get("baseline_brier_score", np.nan)
        ),
        "brier_skill_score": float(metrics.get("brier_skill_score", np.nan)),
        "passes_utility_gate": bool(metrics.get("passes_utility_gate", False)),
        "plain_language": (
            f"The selected model was tested on {test_rows} newer historical samples. It correctly "
            f"identified {tp} corrective event(s), missed {fn}, correctly classified {tn} no-event "
            f"sample(s), and generated {fp} false alert(s) at the validation-selected "
            f"{threshold:.2f} decision threshold. Probability output is enabled only because it "
            f"also beat the no-feature baseline on this untouched test period."
        ),
    }


def build_quality_issues(result: dict[str, Any]) -> pd.DataFrame:
    alerts = result.get("alerts", pd.DataFrame())
    sos = result.get("sos", pd.DataFrame())
    telemetry_quality = result.get("telemetry_quality", {}) or {}
    matched = result.get("matched_assets", set()) or set()
    issues: list[dict[str, Any]] = []

    invalid = int(sos["is_invalid_date"].fillna(True).sum()) if isinstance(sos, pd.DataFrame) and not sos.empty else 0
    if invalid:
        issues.append({"severity": "Blocker", "issue": "Invalid or time-only sample dates", "count": invalid, "impact": "Date-dependent trends and response-time analysis exclude these rows.", "fix": "Supply a complete calendar date for DateSampled."})
    missing_assets = int(sos["asset_id"].isna().sum()) if isinstance(sos, pd.DataFrame) and not sos.empty else 0
    if missing_assets:
        issues.append({"severity": "Blocker", "issue": "Missing asset identifiers", "count": missing_assets, "impact": "S.O.S, telemetry and WO records cannot be joined reliably.", "fix": "Populate EquipNum or a stable equipment identifier."})
    numeric_missing = 0
    if isinstance(sos, pd.DataFrame) and not sos.empty:
        numeric_missing = int((~sos[RAW_MEASUREMENT_COLUMNS].notna().any(axis=1)).sum())
    if numeric_missing:
        issues.append({"severity": "Warning", "issue": "Samples without structured numerical laboratory measurements", "count": numeric_missing, "impact": "Engineering text triage remains available, but numerical degradation trends are limited.", "fix": "Include Fe, Cu, Si, Water, FuelDilution, Soot, Viscosity, Oxidation and TBN where available."})
    if isinstance(sos, pd.DataFrame) and not sos.empty:
        unmatched_rows = int((~sos["asset_id"].astype(str).isin(matched)).sum())
        if unmatched_rows:
            issues.append({"severity": "Information", "issue": "S.O.S rows without matched telemetry", "count": unmatched_rows, "impact": "These records use the other available predictors; telemetry is optional.", "fix": "If telemetry exists for these machines, align EquipNum and TMSAssetID or provide an approved mapping table."})
    duplicates = int(telemetry_quality.get("duplicate_asset_timestamp_rows", 0) or 0)
    if duplicates:
        issues.append({"severity": "Information", "issue": "Duplicate telemetry asset/timestamp rows", "count": duplicates, "impact": "Duplicates were resolved by retaining the most recently modified record.", "fix": "Review source-system duplicate generation if the count grows."})
    review_rows = int(alerts.get("data_quality_status", pd.Series(dtype="string")).astype(str).eq("Review required").sum()) if isinstance(alerts, pd.DataFrame) else 0
    if not issues and not review_rows:
        issues.append({"severity": "Information", "issue": "No material quality issue detected", "count": 0, "impact": "The supplied rows passed the configured checks.", "fix": "Continue monitoring data coverage and units."})
    return pd.DataFrame(issues)


def measurement_trend_table(sos: pd.DataFrame, asset_id: str, component: str) -> pd.DataFrame:
    if sos.empty:
        return pd.DataFrame()
    enriched = add_sos_trends(sos)
    subset = enriched[
        enriched["asset_id"].astype(str).eq(str(asset_id))
        & enriched["component"].astype(str).eq(str(component))
        & ~enriched["is_invalid_date"].fillna(True)
    ].sort_values("sample_date")
    return subset


def latest_measurement_changes(trend: pd.DataFrame, limit: int = 4) -> pd.DataFrame:
    if trend.empty:
        return pd.DataFrame()
    latest = trend.iloc[-1]
    previous = trend.iloc[-2] if len(trend) >= 2 else None
    rows: list[dict[str, Any]] = []
    for measurement in RAW_MEASUREMENT_COLUMNS:
        current = pd.to_numeric(pd.Series([latest.get(measurement)]), errors="coerce").iloc[0]
        if pd.isna(current):
            continue
        prev = pd.to_numeric(pd.Series([previous.get(measurement) if previous is not None else np.nan]), errors="coerce").iloc[0]
        delta = current - prev if pd.notna(prev) else np.nan
        pct = (delta / abs(prev) * 100) if pd.notna(delta) and prev != 0 else np.nan
        rate = pd.to_numeric(pd.Series([latest.get(f"{measurement}_rate_100h")]), errors="coerce").iloc[0]
        rows.append({
            "measurement": measurement,
            "measurement_name": MEASUREMENT_LABELS.get(measurement, measurement),
            "unit": MEASUREMENT_UNITS.get(measurement, ""),
            "current": float(current),
            "previous": float(prev) if pd.notna(prev) else np.nan,
            "change": float(delta) if pd.notna(delta) else np.nan,
            "change_pct": float(pct) if pd.notna(pct) else np.nan,
            "rate_per_100h": float(rate) if pd.notna(rate) else np.nan,
            "absolute_change": abs(float(delta)) if pd.notna(delta) else 0.0,
        })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("absolute_change", ascending=False).head(limit).reset_index(drop=True)


def telemetry_asset_insight(telemetry: pd.DataFrame, asset_id: str) -> dict[str, Any]:
    if telemetry.empty:
        return {"status": "Not available", "summary": "No matched telemetry is available for this machine."}
    data = telemetry[telemetry["asset_id"].astype(str).eq(str(asset_id))].sort_values("event_time")
    if data.empty:
        return {"status": "Not available", "summary": "No matched telemetry is available for this machine."}
    latest_time = data["event_time"].max()
    age_days = (pd.Timestamp.now(tz=None) - pd.Timestamp(latest_time).tz_localize(None)).total_seconds() / 86400 if pd.notna(latest_time) else np.nan
    anomalies = int(data.get("telemetry_anomaly", pd.Series(False, index=data.index)).fillna(False).sum())
    resets = int(data.get("operating_hours_reset", pd.Series(False, index=data.index)).fillna(False).sum())
    max_gap = pd.to_numeric(data.get("gap_hours"), errors="coerce").max()
    status = "Monitor" if anomalies or resets or (pd.notna(max_gap) and max_gap > 72) else "Normal"
    summary = (
        f"{len(data):,} telemetry snapshot(s); {anomalies} unusual snapshot(s); "
        f"{resets} operating-hour reset(s); maximum reporting gap "
        f"{max_gap:.1f} hours."
        if pd.notna(max_gap)
        else f"{len(data):,} telemetry snapshot(s) are available."
    )
    return {"status": status, "summary": summary, "age_days": age_days, "data": data}
