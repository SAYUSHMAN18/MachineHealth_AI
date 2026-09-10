from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd

from .data import safe_float, safe_text
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
    raw = safe_text(value, default="Unknown").strip().upper() or "UNKNOWN"
    return COMPONENT_LABELS.get(raw, raw.replace("_", " ").title())


def feature_label(value: object) -> str:
    raw = safe_text(value).split("__")[-1]
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


def build_condition_outlook(
    result: dict[str, Any], asset_id: str, component: str
) -> dict[str, Any]:
    """Create a transparent forward condition outlook without claiming failure probability.

    This deliberately forecasts the *next laboratory condition*, not a machine
    failure. It uses dated status transitions, persistence and measurement
    coverage, so the conclusion is visibly different from copying the latest
    AR/NAR cell.
    """
    alerts = result.get("alerts", pd.DataFrame())
    sos = result.get("sos", pd.DataFrame())
    empty = {
        "trajectory": "Insufficient history",
        "persistence": "No dated sequence",
        "outlook": "A forward condition outlook cannot be formed yet.",
        "checkpoint": "Next scheduled sample",
        "confidence": "Low",
        "basis": "No usable dated laboratory sequence",
    }
    if not isinstance(alerts, pd.DataFrame) or alerts.empty:
        return empty

    history = alerts[
        alerts["asset_id"].astype(str).eq(str(asset_id))
        & alerts["component"].astype(str).eq(str(component))
        & ~alerts["is_invalid_date"].fillna(True)
    ].sort_values("sample_date")
    if history.empty:
        return empty

    status_text = history["lab_status"].astype("string").str.lower()
    severity = pd.Series(np.nan, index=history.index, dtype=float)
    severity.loc[status_text.str.contains("normal", na=False)] = 0.0
    severity.loc[status_text.str.contains("warning|monitor", regex=True, na=False)] = 1.0
    severity.loc[
        status_text.str.contains("action required|severe|critical", regex=True, na=False)
    ] = 2.0
    valid_severity = severity.dropna()
    if valid_severity.empty:
        return {
            **empty,
            "basis": f"{len(history)} dated sample(s), but no recognised laboratory status sequence",
        }

    latest_score = float(valid_severity.iloc[-1])
    prior_score = float(valid_severity.iloc[-2]) if len(valid_severity) >= 2 else np.nan
    consecutive = 1
    for value in reversed(valid_severity.iloc[:-1].tolist()):
        if float(value) != latest_score:
            break
        consecutive += 1

    latest_date = pd.to_datetime(history["sample_date"].iloc[-1], errors="coerce")
    run_start_index = valid_severity.index[-consecutive]
    run_start = pd.to_datetime(history.loc[run_start_index, "sample_date"], errors="coerce")
    run_days = int((latest_date - run_start).days) if pd.notna(latest_date) and pd.notna(run_start) else 0

    if latest_score >= 2 and consecutive >= 2:
        trajectory = "Persistent abnormal"
        outlook = (
            "Elevated chance that the next laboratory sample remains action-required "
            "unless the identified cause is removed and the repair is confirmed by resampling."
        )
    elif pd.notna(prior_score) and latest_score > prior_score:
        trajectory = "Worsening"
        outlook = (
            "The latest status deteriorated; the next sample has an elevated chance of "
            "remaining abnormal until corrective action is verified."
        )
    elif pd.notna(prior_score) and latest_score < prior_score:
        trajectory = "Improving"
        outlook = "The latest status improved, but one confirmation sample is still needed."
    elif latest_score >= 2:
        trajectory = "Abnormal — new"
        outlook = "A repeat sample is needed to determine whether the abnormal condition persists."
    elif latest_score == 1:
        trajectory = "Monitor"
        outlook = "Continue monitoring; no failure probability is justified from this sequence."
    else:
        trajectory = "Stable normal"
        outlook = "No deterioration is evident in the available laboratory-status sequence."

    latest_label = "action-required" if latest_score >= 2 else "warning" if latest_score == 1 else "normal"
    persistence = f"{consecutive} consecutive {latest_label} sample(s)"
    if consecutive >= 2 and run_days > 0:
        persistence += f" across {run_days} days"

    sample_history = sos[
        sos["asset_id"].astype(str).eq(str(asset_id))
        & sos["component"].astype(str).eq(str(component))
        & ~sos["is_invalid_date"].fillna(True)
    ] if isinstance(sos, pd.DataFrame) and not sos.empty else pd.DataFrame()
    numeric_rows = int(sample_history[RAW_MEASUREMENT_COLUMNS].notna().any(axis=1).sum()) if not sample_history.empty else 0
    abnormal_count = int(valid_severity.ge(2).sum())

    latest = history.iloc[-1]
    timing_text = " ".join([
        safe_text(latest.get("recommended_action")),
        safe_text(latest.get("original_interpretation")),
    ])
    timing_match = re.search(r"\b(\d{1,4})\s*(?:operating[- ]?)?hours?\b", timing_text, flags=re.I)
    checkpoint = f"Resample in {timing_match.group(1)} operating hours" if timing_match else "Next scheduled sample"

    telemetry = result.get("telemetry", pd.DataFrame())
    work_orders = result.get("work_orders")
    has_telemetry = isinstance(telemetry, pd.DataFrame) and not telemetry.empty and telemetry["asset_id"].astype(str).eq(str(asset_id)).any()
    has_wo = isinstance(work_orders, pd.DataFrame) and not work_orders.empty and work_orders["asset_id"].astype(str).eq(str(asset_id)).any()
    confidence = (
        "High" if len(valid_severity) >= 12 and numeric_rows >= 6 and has_wo
        else "Medium" if len(valid_severity) >= 6 and numeric_rows >= 3
        else "Low"
    )
    basis = (
        f"{len(valid_severity)} dated statuses; {abnormal_count} action-required; "
        f"{numeric_rows} sample(s) with measured numerical results; "
        f"telemetry {'available' if has_telemetry else 'not available'}; "
        f"WO outcomes {'available' if has_wo else 'not available'}"
    )
    return {
        "trajectory": trajectory,
        "persistence": persistence,
        "outlook": outlook,
        "checkpoint": checkpoint,
        "confidence": confidence,
        "basis": basis,
    }


def compute_machine_health_score(
    result: dict[str, Any], asset_id: str, component: str
) -> dict[str, Any]:
    """Compute a transparent heuristic condition index, never a failure probability."""
    sos = result.get("sos", pd.DataFrame())
    comp_sos = sos[
        sos["asset_id"].astype(str).eq(str(asset_id))
        & sos["component"].astype(str).eq(str(component))
    ] if isinstance(sos, pd.DataFrame) and not sos.empty else pd.DataFrame()

    dated_sos = (
        comp_sos.loc[
            ~comp_sos["is_invalid_date"].fillna(True)
            & pd.to_datetime(comp_sos["sample_date"], errors="coerce").notna()
        ].sort_values("sample_date")
        if not comp_sos.empty
        else pd.DataFrame()
    )
    # Undated rows cannot safely be treated as the newest condition. They still
    # remain visible in the action queue and data-quality report.
    latest_sos = dated_sos.iloc[-1] if not dated_sos.empty else (
        comp_sos.sort_values("source_row").iloc[-1] if not comp_sos.empty else {}
    )
    lab_code = str(latest_sos.get("interpretation_code", "")).upper()
    interp_text = str(latest_sos.get("interpretation_text", "")).lower()

    score = 100.0
    penalties: list[str] = []
    fingerprints: list[str] = []

    # 1. Lab status penalty
    if lab_code == "AR":
        ordered_codes = dated_sos["interpretation_code"].astype(str).str.upper().tolist()
        consecutive_ar = 0
        for code in reversed(ordered_codes):
            if code != "AR":
                break
            consecutive_ar += 1
        consecutive_ar = max(consecutive_ar, 1)
        if consecutive_ar >= 2:
            score -= 45
            penalties.append(f"Repeated lab Action Required ({consecutive_ar}x consecutive)")
        else:
            score -= 30
            penalties.append("Lab Action Required (AR)")
    elif any(k in lab_code for k in ["WARN", "MONITOR", "B", "C"]):
        score -= 15
        penalties.append("Lab Monitor / Warning status")

    # 2. Contaminant inspection
    water_val = latest_sos.get("water_pct", np.nan)
    has_water = (
        (pd.notna(water_val) and safe_float(water_val) > 0.02)
        or "water present" in interp_text
        or "excessive water" in interp_text
    )
    if has_water:
        score -= 25
        penalties.append("Severe water contamination")
        fingerprints.append("Seal/Plug Moisture Ingress")

    if any(k in interp_text for k in ["coolant", "glycol", "antifreeze"]):
        score -= 25
        penalties.append("Coolant / Glycol contamination")
        fingerprints.append("Cooler / Internal Leak")

    if any(k in interp_text for k in ["fuel dilution", "fuel in oil"]):
        score -= 20
        penalties.append("Fuel dilution")
        fingerprints.append("Fuel Injector Dilution")

    # 3. Wear metals inspection
    fe = safe_float(latest_sos.get("iron_ppm", np.nan), np.nan)
    cu = safe_float(latest_sos.get("copper_ppm", np.nan), np.nan)
    al = safe_float(latest_sos.get("aluminium_ppm", np.nan), np.nan)
    si = safe_float(latest_sos.get("silicon_ppm", np.nan), np.nan)

    if np.isfinite(fe) and fe > 150 and np.isfinite(cu) and cu > 15:
        score -= 20
        penalties.append(f"Elevated gear & bushing wear (Fe {fe:.0f}, Cu {cu:.0f} ppm)")
        fingerprints.append("Gear Mesh & Bushing Breakdown")
    elif np.isfinite(fe) and fe > 150:
        score -= 15
        penalties.append(f"Elevated Iron ({fe:.0f} ppm)")
        fingerprints.append("Mechanical Gear Wear")

    if np.isfinite(si) and si > 25 and np.isfinite(al) and al > 15:
        score -= 15
        penalties.append(f"Abrasive dirt ingress (Si {si:.0f}, Al {al:.0f} ppm)")
        fingerprints.append("Abrasive Dust Ingress via Breather")

    # 4. Fluid condition
    if any(k in interp_text for k in ["viscosity", "oxidation", "degraded"]):
        score -= 10
        penalties.append("Fluid degradation / thermal stress")

    final_score = int(max(5, min(100, round(score))))
    if final_score >= 85:
        status_label = "Optimal"
        status_color = "#16a34a"
    elif final_score >= 65:
        status_label = "Monitor"
        status_color = "#d97706"
    else:
        status_label = "Degraded"
        status_color = "#dc2626"

    fingerprint_text = " + ".join(fingerprints) if fingerprints else "Normal Mechanical Baseline"

    return {
        "health_score": final_score,
        "health_status": status_label,
        "health_color": status_color,
        "fingerprint": fingerprint_text,
        "penalties": penalties,
        "method": "Heuristic condition index; not a probability or remaining-life estimate",
    }


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
        latest_row = dated.iloc[-1] if not dated.empty else group.sort_values("_row_rank").iloc[-1]
        latest_date = latest_row.get("_sample_date", pd.NaT)
        finding_date = selected.get("_sample_date", pd.NaT)
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
        condition_outlook = build_condition_outlook(result, str(asset_id), str(component))
        health_info = compute_machine_health_score(result, str(asset_id), str(component))
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
            "finding_sample_date": finding_date,
            "lab_status": str(latest_row.get("lab_status", "Unspecified")),
            "latest_lab_status": str(latest_row.get("lab_status", "Unspecified")),
            "active_finding_lab_status": str(selected.get("lab_status", "Unspecified")),
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
            "condition_trajectory": condition_outlook["trajectory"],
            "condition_persistence": condition_outlook["persistence"],
            "next_condition_outlook": condition_outlook["outlook"],
            "next_checkpoint": condition_outlook["checkpoint"],
            "outlook_confidence": condition_outlook["confidence"],
            "outlook_basis": condition_outlook["basis"],
            "health_score": health_info["health_score"],
            "health_status": health_info["health_status"],
            "health_color": health_info["health_color"],
            "failure_fingerprint": health_info["fingerprint"],
            "health_penalties": health_info["penalties"],
            "health_method": health_info["method"],
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

    avg_fleet_health = int(round(float(active["health_score"].mean()))) if not active.empty and "health_score" in active.columns else 100

    confidence_order = {"High": 3, "Medium": 2, "Low": 1}
    confidence = "Not available"
    if not cases.empty:
        average = cases["data_confidence"].map(confidence_order).fillna(1).mean()
        confidence = "High" if average >= 2.6 else "Medium" if average >= 1.6 else "Low"

    mode = str(result.get("operating_mode", MODE_ALERT))
    samples = int(len(sos)) if isinstance(sos, pd.DataFrame) else 0
    assets = int(sos["asset_id"].nunique(dropna=True)) if samples else 0
    cases_count = int(len(cases))
    p2_count = int(active["priority_code"].eq(P2_REPEATED).sum()) if not active.empty else 0
    p3_count = int(active["priority_code"].eq(P3_ACTION).sum()) if not active.empty else 0
    if immediate:
        recommendation = f"Review {immediate} P1 machine-component case(s) today; {wo_linked} active case(s) have a linked WO and {no_wo_p1} P1 case(s) need WO verification or creation."
    elif p2_count or p3_count:
        recommendation = (
            f"No P1 case was generated. Review {p2_count} repeated P2 case(s) and "
            f"{p3_count} P3 engineering-review case(s) within 3 days."
        )
    else:
        recommendation = "No P1/P2/P3 case was generated. Continue scheduled sampling and monitor open work orders."
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
        "avg_fleet_health": avg_fleet_health,
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

    telemetry_schema_issue = safe_text(telemetry_quality.get("schema_issue")).strip()
    if telemetry_schema_issue:
        issues.append({
            "severity": "Warning",
            "issue": "Invalid telemetry schema",
            "count": int(safe_float(telemetry_quality.get("rows_before_deduplication"))),
            "impact": telemetry_schema_issue,
            "fix": "Provide machine time-series telemetry with an asset identifier and event timestamp; do not upload enum/reference dictionaries.",
        })

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
    duplicates = int(safe_float(telemetry_quality.get("duplicate_asset_timestamp_rows")))
    if duplicates:
        issues.append({"severity": "Information", "issue": "Duplicate telemetry asset/timestamp rows", "count": duplicates, "impact": "Duplicates were resolved by retaining the most recently modified record.", "fix": "Review source-system duplicate generation if the count grows."})
    review_rows = int(alerts.get("data_quality_status", pd.Series(dtype="string")).astype(str).eq("Review required").sum()) if isinstance(alerts, pd.DataFrame) else 0
    if not issues and not review_rows:
        issues.append({"severity": "Information", "issue": "No material quality issue detected", "count": 0, "impact": "The supplied rows passed the configured checks.", "fix": "Continue monitoring data coverage and units."})
    return pd.DataFrame(issues)


def measurement_trend_table(sos: pd.DataFrame, asset_id: str, component: str) -> pd.DataFrame:
    if sos.empty:
        return pd.DataFrame()
    subset = sos[
        sos["asset_id"].astype(str).eq(str(asset_id))
        & sos["component"].astype(str).eq(str(component))
        & ~sos["is_invalid_date"].fillna(True)
    ].copy()
    if subset.empty:
        return subset
    # Calculate trends only for the selected series instead of rescanning the fleet.
    return add_sos_trends(subset).sort_values("sample_date")


def latest_measurement_changes(trend: pd.DataFrame, limit: int = 4) -> pd.DataFrame:
    if trend.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for measurement in RAW_MEASUREMENT_COLUMNS:
        values = trend[["sample_date", measurement, f"{measurement}_rate_100h"]].copy()
        values[measurement] = pd.to_numeric(values[measurement], errors="coerce")
        values = values.dropna(subset=[measurement]).sort_values("sample_date")
        if values.empty:
            continue
        current_row = values.iloc[-1]
        previous_row = values.iloc[-2] if len(values) >= 2 else None
        current = float(current_row[measurement])
        prev = float(previous_row[measurement]) if previous_row is not None else np.nan
        delta = current - prev if pd.notna(prev) else np.nan
        pct = (delta / abs(prev) * 100) if pd.notna(delta) and prev != 0 else np.nan
        rate = pd.to_numeric(
            pd.Series([current_row.get(f"{measurement}_rate_100h")]), errors="coerce"
        ).iloc[0]
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
