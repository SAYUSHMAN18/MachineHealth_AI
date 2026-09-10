from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .data import (
    asset_intersection,
    enrich_sos_with_test_details,
    load_table,
    prepare_sos,
    prepare_telemetry,
    prepare_work_orders,
    safe_bool,
    safe_float,
    safe_text,
)
from .features import (
    RAW_MEASUREMENT_COLUMNS,
    build_scoring_table,
    build_training_table,
    telemetry_asset_summary,
)
from .reporting import generate_maintenance_summary
from .models import score_telemetry_anomalies, train_failure_models
from .rules import evaluate_sos_rules

# ---------------------------------------------------------------------------
# Operating mode constants
# ---------------------------------------------------------------------------
MODE_ALERT = "Alert Management"
MODE_CONDITION = "Condition Monitoring"
MODE_PREDICTION = "Failure Prediction"

# Priority tier labels
P1_IMMEDIATE = "P1 – Immediate Review"
P1_WO_TRACKING = "P1 – WO Tracking"
P2_REPEATED = "P2 – Multiple Unlinked Alerts"
P3_ACTION = "P3 – Action Required"
TIER_IN_PROGRESS = "In Progress"
TIER_CLOSED = "Closed"

# Approved engineering problem categories for InterpText
_CATEGORY_PATTERNS = [
    ("Wear Metal – Iron",       r"\biron\b|\bFe\b|\bbearing wear\b|\bgear wear\b"),
    ("Wear Metal – Copper",     r"\bcopper\b|\bCu\b|\bbronze\b"),
    ("Wear Metal – Aluminium",  r"\balumini?um\b|\bAl\b"),
    ("Wear Metal – Lead",       r"\blead\b|\bPb\b|\bbabbitt\b"),
    ("Silicon / Dirt Ingress",  r"\bsilicon\b|\bSi\b|\bdirt\b|\bingress\b|\bcontamina"),
    ("Water / Coolant",         r"\bwater\b|\bcoolant\b|\bglyco\b|\bantifreeze\b"),
    ("Fuel Dilution",           r"\bfuel\b|\bdilution\b"),
    ("Fluid Degradation",       r"\bviscosity\b|\boxidation\b|\bacidity\b|\bTBN\b|\bsoot\b|\bnigr"),
    ("Resample / History",      r"\bresample\b|\bsample in\b|\bmore.*history\b|\bhistory needed\b"),
    ("Normal / Acceptable",     r"\bnormal\b|\bacceptable\b|\bsatisfactory\b|\bno action\b"),
]


# ---------------------------------------------------------------------------
# Mode selection
# ---------------------------------------------------------------------------

def _select_operating_mode(readiness: pd.DataFrame) -> str:
    """Choose a mode only when every gate for that mode passes."""
    if readiness.empty or "gate" not in readiness.columns:
        return MODE_ALERT
    gates = readiness.set_index("gate")["status"].to_dict()
    prediction_gates = [
        "lab_predictors",
        "explicit_wo_outcomes",
        "labelled_training_rows",
        "chronological_dates",
    ]
    if all(gates.get(gate) == "PASS" for gate in prediction_gates):
        return MODE_PREDICTION
    if gates.get("numeric_trends") == "PASS":
        return MODE_CONDITION
    return MODE_ALERT


# ---------------------------------------------------------------------------
# Interpretation text categorisation
# ---------------------------------------------------------------------------

def _compute_interp_categories(alerts: pd.DataFrame) -> pd.DataFrame:
    """Create one evidence row per matched category and retain its exact sentence."""
    import re as _re
    rows = []
    for _, row in alerts.iterrows():
        text = safe_text(
            row.get("original_interpretation", row.get("interpretation_text"))
        )
        sentences = [s.strip() for s in _re.split(r"(?<=[.!?])\s+", text) if s.strip()]
        matched = False
        for cat_name, pattern in _CATEGORY_PATTERNS:
            evidence_sentence = next(
                (sentence for sentence in sentences if _re.search(pattern, sentence, flags=_re.IGNORECASE)),
                None,
            )
            if evidence_sentence is not None:
                matched = True
                rows.append({
                    "sample_number": row.get("sample_number"),
                    "asset_id": row.get("asset_id"),
                    "component": row.get("component"),
                    "problem_category": cat_name,
                    "category_evidence": evidence_sentence,
                    "full_interpretation": text,
                })
        if not matched:
            rows.append({
                "sample_number": row.get("sample_number"),
                "asset_id": row.get("asset_id"),
                "component": row.get("component"),
                "problem_category": "Other",
                "category_evidence": sentences[0] if sentences else text[:120],
                "full_interpretation": text,
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Priority tiers
# ---------------------------------------------------------------------------

def _compute_priority_tiers(alerts: pd.DataFrame) -> pd.DataFrame:
    """Assign action priority independently from data-quality flags."""
    wo_present = alerts["wo_id"].apply(
        lambda value: pd.notna(value) and str(value).strip() not in ("", "<NA>", "nan", "None")
    )
    new_ar = alerts[
        (alerts["interpretation_code"].astype(str).str.upper() == "AR")
        & (alerts["status"].astype(str).str.upper() == "NEW")
        & ~wo_present
    ]
    repeat_map: dict[tuple, int] = (
        new_ar.groupby(["asset_id", "component"], dropna=False).size().to_dict()
    )

    tiers, reasons, dq_statuses, dq_issues = [], [], [], []
    for _, row in alerts.iterrows():
        code = str(row.get("interpretation_code", "")).upper()
        status = str(row.get("status", "New")).upper()
        hp_raw = safe_text(row.get("high_priority")).upper()
        hp = hp_raw in ("T", "TRUE", "1", "Y", "YES")
        wo = row.get("wo_id")
        has_wo = pd.notna(wo) and str(wo).strip() not in ("", "<NA>", "nan", "None")
        is_inv = safe_bool(row.get("is_invalid_date"))
        asset = row.get("asset_id", "")
        comp = row.get("component", "")
        repeat_count = repeat_map.get((asset, comp), 0)

        raw_date_issue = row.get("date_quality_issue", "")
        date_issue = "" if pd.isna(raw_date_issue) else str(raw_date_issue).strip()
        if is_inv and not date_issue:
            date_issue = "Sample calendar date is missing or invalid"
        dq_status = "Review required" if is_inv else "Valid"

        if status == "CLOSED":
            tier = TIER_CLOSED
            reason = (
                "The laboratory record status is Closed. This does not by itself confirm that "
                "maintenance or repair was completed."
            )
        elif hp and not has_wo:
            tier = P1_IMMEDIATE
            reason = (
                f"HighPriority flag is set (HighPriority=T), status is New, "
                f"and no WorkOrderId is assigned. Requires immediate engineer review."
            )
        elif hp and has_wo:
            tier = P1_WO_TRACKING
            reason = (
                f"HighPriority flag is set (HighPriority=T), status is New, "
                f"and WorkOrderId {wo} is linked. Verify its status, inspection result and outcome urgently."
            )
        elif code == "AR" and status == "NEW" and repeat_count >= 2 and not has_wo:
            tier = P2_REPEATED
            reason = (
                f"This extract contains {repeat_count} New AR records for asset {asset} / "
                f"compartment {comp}; this record has no linked WorkOrderId. Verify the dated "
                "history and maintenance response."
            )
        elif code == "AR" and status == "NEW" and has_wo:
            tier = TIER_IN_PROGRESS
            reason = (
                f"New AR sample has linked WorkOrderId {wo}. A linked identifier does not "
                "confirm inspection, repair, or WO completion."
            )
        elif code == "AR" and status == "NEW" and not has_wo:
            tier = P3_ACTION
            reason = (
                "New AR sample has no linked WorkOrderId. Verify whether a WO exists and route "
                "the laboratory recommendation for engineering review."
            )
        elif has_wo:
            tier = TIER_IN_PROGRESS
            reason = f"WorkOrderId {wo} is linked; its maintenance outcome is not available in this file."
        else:
            tier = TIER_IN_PROGRESS
            reason = f"Status is {status.capitalize()} with interpretation code {code}."

        tiers.append(tier)
        reasons.append(reason)
        dq_statuses.append(dq_status)
        dq_issues.append(date_issue if is_inv else "")

    out = alerts.copy()
    out["priority_tier"] = tiers
    out["priority_reason"] = reasons
    out["workflow_priority"] = tiers
    out["priority_evidence"] = reasons
    out["data_quality_status"] = dq_statuses
    out["data_quality_issue"] = dq_issues
    return out


# ---------------------------------------------------------------------------
# Risk drivers (used only in Failure Prediction mode)
# ---------------------------------------------------------------------------

def _extract_sample_risk_drivers(alert_row: pd.Series, top_model_features: list[str] | None = None) -> list[str]:
    drivers: list[str] = []
    iron = safe_float(alert_row.get("iron_ppm"))
    copper = safe_float(alert_row.get("copper_ppm"))
    silicon = safe_float(alert_row.get("silicon_ppm"))
    water = safe_float(alert_row.get("water_pct"))
    soot = safe_float(alert_row.get("soot_pct"))
    viscosity = safe_float(alert_row.get("viscosity_cst"))
    fuel = safe_float(alert_row.get("fuel_dilution_pct"))
    if iron > 80:
        drivers.append(f"Iron Wear Metal Spike ({iron:.0f} ppm)")
    elif iron > 40:
        drivers.append(f"Elevated Iron Concentration ({iron:.0f} ppm)")
    if copper > 40:
        drivers.append(f"Copper Bearing Wear ({copper:.0f} ppm)")
    if silicon > 25:
        drivers.append(f"Dirt / Silicon Contamination ({silicon:.0f} ppm)")
    if water > 0.1:
        drivers.append(f"Water Fluid Contamination ({water:.1f}%)")
    if soot > 0.5:
        drivers.append(f"High Soot Concentration ({soot:.1f}%)")
    if viscosity > 0 and (viscosity < 10.0 or viscosity > 18.0):
        drivers.append(f"Viscosity Deviation ({viscosity:.1f} cSt)")
    if fuel > 2.0:
        drivers.append(f"Fuel Dilution ({fuel:.1f}%)")
    evidence_text = str(alert_row.get("evidence", ""))
    if evidence_text and evidence_text != "No configured fault signature found in interpretation text":
        for part in evidence_text.split(";"):
            cleaned = part.strip()
            if cleaned and cleaned not in drivers:
                drivers.append(cleaned)
    if top_model_features:
        for feat in top_model_features:
            readable = feat.replace("_", " ").title()
            if readable not in drivers:
                drivers.append(f"Model Driver: {readable}")
    if not drivers:
        drivers = ["Laboratory AR flag — no numerical measurement data available"]
    return drivers[:4]


# ---------------------------------------------------------------------------
# Readiness table
# ---------------------------------------------------------------------------

def _readiness_table(
    sos: pd.DataFrame,
    telemetry: pd.DataFrame,
    work_orders: pd.DataFrame | None,
    matched_assets: set[str],
    training: pd.DataFrame | None,
) -> pd.DataFrame:
    numeric_mask = sos[RAW_MEASUREMENT_COLUMNS].notna().any(axis=1)
    raw_measurements = int(numeric_mask.sum())
    valid_date_mask = ~sos["is_invalid_date"].fillna(True)
    valid_dates = int(valid_date_mask.sum())
    numeric_dated = sos[numeric_mask & valid_date_mask]
    numeric_series = (
        numeric_dated.groupby(["asset_id", "component"], dropna=False).size()
        if not numeric_dated.empty else pd.Series(dtype="int64")
    )
    repeated_numeric_series = int((numeric_series >= 3).sum())
    numeric_trends_ready = raw_measurements >= 3 and repeated_numeric_series >= 1
    text = sos.get("interpretation_text", pd.Series("", index=sos.index)).fillna("").astype(str).str.strip()
    text_rows = int(text.ne("").sum())
    text_variants = int(text.loc[text.ne("")].nunique())
    text_predictors_ready = text_rows >= 15 and text_variants >= 2
    lab_predictors_ready = numeric_trends_ready or text_predictors_ready

    codes = set(sos["interpretation_code"].dropna().astype(str).str.upper().unique())
    has_both_classes = bool(len(codes) >= 2)
    explicit_wo_rows = 0
    explicit_wo_classes = 0
    if work_orders is not None and not work_orders.empty:
        explicit = work_orders.get(
            "failure_label_source", pd.Series("text_inferred", index=work_orders.index)
        ).eq("explicit")
        explicit_wo_rows = int(explicit.sum())
        explicit_wo_classes = int(
            work_orders.loc[explicit, "confirmed_failure"].nunique()
            if explicit.any() else 0
        )
    has_explicit_wo_outcomes = explicit_wo_rows >= 20 and explicit_wo_classes >= 2
    matched_sos_rows = int(sos["asset_id"].astype(str).isin(matched_assets).sum())
    has_telemetry_match = len(matched_assets) >= 2 and matched_sos_rows >= 10
    training_rows = 0 if training is None else len(training)
    training_classes = (
        0 if training is None or training.empty
        else int(training["corrective_wo_within_horizon"].nunique())
    )
    training_positives = (
        0 if training is None or training.empty
        else int(training["corrective_wo_within_horizon"].eq(1).sum())
    )
    training_negatives = training_rows - training_positives
    training_assets = (
        0 if training is None or training.empty
        else int(training["asset_id"].nunique(dropna=True))
    )
    labelled_training_ready = bool(
        training_rows >= 60
        and training_classes >= 2
        and training_positives >= 15
        and training_negatives >= 15
        and training_assets >= 5
    )
    distinct_valid_dates = int(sos.loc[valid_date_mask, "sample_date"].nunique())
    chronological_dates_ready = valid_dates >= 60 and distinct_valid_dates >= 12

    criteria = [
        (
            "lab_predictors",
            "Enough usable S.O.S predictors exist",
            lab_predictors_ready,
            f"{raw_measurements} numerical sample(s); {text_rows} interpretation-text sample(s) with {text_variants} distinct text value(s) — need ≥3 numeric or ≥15 text",
            "Failure Prediction",
        ),
        (
            "both_lab_classes",
            "Normal and abnormal S.O.S interpretation codes both exist (optional)",
            has_both_classes,
            f"{len(codes)} distinct code(s) — need ≥2 (e.g. A and AR)"
            if not has_both_classes else f"{len(codes)} distinct codes found ✓",
            "Optional enrichment",
        ),
        (
            "numeric_trends",
            "Numerical laboratory trend history exists",
            numeric_trends_ready,
            f"{raw_measurements} numerical sample(s); {repeated_numeric_series} asset/component series have ≥3 dated samples — need ≥3 samples",
            "Condition Monitoring + Failure Prediction",
        ),
        (
            "explicit_wo_outcomes",
            "Explicit confirmed work-order outcomes exist",
            has_explicit_wo_outcomes,
            f"{explicit_wo_rows} explicit outcome row(s); {explicit_wo_classes} outcome class(es) — need ≥20 rows and both classes",
            "Failure Prediction",
        ),
        (
            "matched_assets",
            "S.O.S and telemetry IDs match across enough assets (optional)",
            has_telemetry_match,
            f"{len(matched_assets)} matched asset(s), covering {matched_sos_rows} S.O.S row(s) — need ≥5 assets and ≥30 rows",
            "Optional enrichment",
        ),
        (
            "labelled_training_rows",
            "Enough independently labelled training rows exist",
            labelled_training_ready,
            f"{training_rows} labelled row(s); {training_positives} positive, {training_negatives} negative, "
            f"{training_assets} asset(s) — need ≥60 rows, ≥15/class and ≥5 assets",
            "Failure Prediction",
        ),
        (
            "chronological_dates",
            "Enough valid calendar dates exist for chronological validation",
            chronological_dates_ready,
            f"{valid_dates} valid dated sample(s) across {distinct_valid_dates} date(s) — need ≥60 rows and ≥12 dates",
            "Failure Prediction",
        ),
    ]
    return pd.DataFrame(
        [
            {"gate": gate, "criterion": name,
             "status": "PASS" if passed else ("OPTIONAL" if required_for == "Optional enrichment" else "BLOCKED"),
             "detail": detail, "required_for": required_for}
            for gate, name, passed, detail, required_for in criteria
        ]
    )


# ---------------------------------------------------------------------------
# Dataset metrics (9 counters)
# ---------------------------------------------------------------------------

def _compute_dataset_metrics(sos: pd.DataFrame, alerts: pd.DataFrame) -> dict[str, int]:
    """Compute operational metrics without treating AR as a predicted failure."""
    total = len(sos)
    unique_assets = int(sos["asset_id"].nunique(dropna=True))

    lab_ar_count = int((sos["interpretation_code"].astype(str).str.upper() == "AR").sum())

    hp_count = int(
        sos["high_priority"].astype(str).str.upper().isin(["T", "TRUE", "1", "Y", "YES"]).sum()
    )

    status_col = (
        alerts["status"].astype(str).str.upper()
        if "status" in alerts.columns
        else sos["status"].astype(str).str.upper()
    )
    new_count = int((status_col == "NEW").sum())
    closed_count = int((status_col == "CLOSED").sum())

    wo_col = alerts["wo_id"] if "wo_id" in alerts.columns else sos.get("wo_id", pd.Series(dtype=object))
    wo_present = wo_col.apply(lambda v: pd.notna(v) and str(v).strip() not in ("", "<NA>", "nan", "None"))
    wo_linked = int(wo_present.sum())
    without_wo = total - wo_linked

    new_no_wo = int(((status_col == "NEW") & ~wo_present).sum())

    hp = sos["high_priority"].astype(str).str.upper().isin(["T", "TRUE", "1", "Y", "YES"])
    high_priority_new_no_wo = int((hp & (status_col == "NEW") & ~wo_present).sum())
    high_priority_new_with_wo = int((hp & (status_col == "NEW") & wo_present).sum())
    high_priority_closed = int((hp & (status_col == "CLOSED")).sum())

    new_unlinked = alerts[(status_col == "NEW") & ~wo_present].copy()
    if not new_unlinked.empty:
        repeated_sizes = new_unlinked.groupby(["asset_id", "component"], dropna=False)["sample_number"].transform("size")
        repeated_new_unlinked_rows = int((repeated_sizes >= 2).sum())
    else:
        repeated_new_unlinked_rows = 0

    if "is_invalid_date" in sos.columns:
        invalid_dates = int(sos["is_invalid_date"].sum())
    else:
        parsed = pd.to_datetime(sos.get("sample_date", pd.Series(dtype=object)), errors="coerce")
        invalid_dates = int((parsed.isna() | (parsed.dt.year < 1950) | (parsed.dt.year > 2035)).sum())

    return {
        "unique_assets": unique_assets,
        "lab_ar_samples": lab_ar_count,
        "high_priority_samples": hp_count,
        "new_alerts": new_count,
        "closed_alerts": closed_count,
        "wo_linked_samples": wo_linked,
        "samples_without_wo": without_wo,
        "new_alerts_without_wo": new_no_wo,
        "invalid_date_samples": invalid_dates,
        "valid_date_samples": total - invalid_dates,
        "high_priority_new_without_wo": high_priority_new_no_wo,
        "high_priority_new_with_wo": high_priority_new_with_wo,
        "high_priority_closed": high_priority_closed,
        "repeated_new_unlinked_rows": repeated_new_unlinked_rows,
        "p1_immediate_records": int((alerts["priority_tier"] == P1_IMMEDIATE).sum()),
        "p1_wo_tracking_records": int((alerts["priority_tier"] == P1_WO_TRACKING).sum()),
        "p2_multiple_unlinked_records": int((alerts["priority_tier"] == P2_REPEATED).sum()),
        "p3_action_records": int((alerts["priority_tier"] == P3_ACTION).sum()),
    }


# ---------------------------------------------------------------------------
# Main analysis entry point
# ---------------------------------------------------------------------------

def analyze_frames(
    sos_raw: pd.DataFrame,
    telemetry_raw: pd.DataFrame | None = None,
    work_orders_raw: pd.DataFrame | None = None,
    horizon_days: int = 30,
) -> dict[str, Any]:
    # 1. Clean / normalise
    sos = prepare_sos(sos_raw)
    telemetry, telemetry_quality = prepare_telemetry(telemetry_raw)
    base_alerts = evaluate_sos_rules(sos)

    # 2. Enrich with 6-tier priority
    alerts = _compute_priority_tiers(base_alerts)

    # 3. Interpretation categories
    interp_categories = _compute_interp_categories(alerts)

    # 4. Work orders & matched assets
    work_orders = prepare_work_orders(work_orders_raw) if work_orders_raw is not None else None
    matched_assets = asset_intersection(sos, telemetry)

    # 5. Readiness & mode selection
    scoring = build_scoring_table(sos, telemetry, work_orders)
    training = build_training_table(sos, telemetry, work_orders, horizon_days=horizon_days)
    readiness = _readiness_table(sos, telemetry, work_orders, matched_assets, training)
    operating_mode = _select_operating_mode(readiness)
    predictive_risk_enabled = (operating_mode == MODE_PREDICTION)

    # 9 dataset metrics
    dataset_metrics = _compute_dataset_metrics(sos, alerts)

    # 6. ML / telemetry scoring — only in Failure Prediction mode
    trained_model = None
    leaderboard = None
    model_error: str | None = None
    sample_model_probs: dict[int, float] = {}
    top_features: list[str] = []
    telemetry_anomaly_scores: dict[str, float] = {}

    if operating_mode == MODE_PREDICTION:
        if training is not None and not training.empty:
            try:
                trained_model, leaderboard = train_failure_models(training)
                utility_passed = bool(
                    trained_model.metrics.get("passes_utility_gate", False)
                )
                readiness = pd.concat(
                    [
                        readiness,
                        pd.DataFrame([{
                            "gate": "heldout_model_utility",
                            "criterion": "Selected model beats a no-feature baseline on newer data",
                            "status": "PASS" if utility_passed else "BLOCKED",
                            "detail": (
                                f"Average-precision lift "
                                f"{trained_model.metrics.get('average_precision_lift', 0.0):.3f}; "
                                f"Brier skill "
                                f"{trained_model.metrics.get('brier_skill_score', 0.0):.3f}"
                            ),
                            "required_for": "Failure Prediction",
                        }]),
                    ],
                    ignore_index=True,
                )
                if utility_passed:
                    # Anomaly scoring is secondary evidence, so avoid its cost
                    # unless the supervised probability model is deployable.
                    telemetry = score_telemetry_anomalies(telemetry)
                    if not telemetry.empty and "telemetry_anomaly_score" in telemetry.columns:
                        for a_id, grp in telemetry.groupby("asset_id", dropna=False):
                            valid_scores = grp["telemetry_anomaly_score"].dropna()
                            if not valid_scores.empty:
                                telemetry_anomaly_scores[str(a_id)] = float(valid_scores.iloc[-1])
                    # Score every current, dated S.O.S sample—not merely the subset
                    # that had work-order coverage and could be used for training.
                    probs = trained_model.predict_proba(scoring)
                    for source_row, probability in zip(scoring["source_row"], probs):
                        sample_model_probs[int(source_row)] = float(probability)
                    importances = trained_model.get_feature_importances()
                    top_features = list(importances.keys())[:3]
                else:
                    model_error = (
                        "The selected model did not show enough improvement over a "
                        "constant prevalence baseline on the untouched chronological test set."
                    )
                    predictive_risk_enabled = False
                    numeric_gate = readiness.loc[
                        readiness["gate"].eq("numeric_trends"), "status"
                    ]
                    operating_mode = (
                        MODE_CONDITION
                        if not numeric_gate.empty and numeric_gate.iloc[0] == "PASS"
                        else MODE_ALERT
                    )
                    # Do not export or serve a model that failed the utility gate.
                    trained_model = None
            except Exception as exc:
                model_error = str(exc)
                predictive_risk_enabled = False
                numeric_gate = readiness.loc[
                    readiness["gate"].eq("numeric_trends"), "status"
                ]
                operating_mode = (
                    MODE_CONDITION
                    if not numeric_gate.empty and numeric_gate.iloc[0] == "PASS"
                    else MODE_ALERT
                )
                readiness = pd.concat(
                    [
                        readiness,
                        pd.DataFrame([{
                            "gate": "chronological_model_fit",
                            "criterion": "Chronological model training completes successfully",
                            "status": "BLOCKED",
                            "detail": model_error,
                            "required_for": "Failure Prediction",
                        }]),
                    ],
                    ignore_index=True,
                )

    # 7. Build Maintenance Action Cards for every alert row
    maintenance_action_cards: list[dict[str, Any]] = []
    scoring_by_source = (
        scoring.set_index("source_row", drop=False).to_dict(orient="index")
        if not scoring.empty else {}
    )

    for _, alert_row in alerts.iterrows():
        source_row = int(alert_row.get("source_row", -1))
        sample_num = str(alert_row.get("sample_number", ""))
        asset_id = str(alert_row.get("asset_id", "Unknown"))
        component = str(alert_row.get("component", "Unknown"))
        rule_evidence_strength = safe_float(
            alert_row.get("rule_evidence_strength"), default=0.25
        )
        interp_code = str(alert_row.get("interpretation_code", "A")).upper()
        lab_status = str(alert_row.get("lab_status", "Unspecified"))
        priority_tier = str(alert_row.get("priority_tier", alert_row.get("workflow_priority", P3_ACTION)))
        priority_reason = str(alert_row.get("priority_reason", alert_row.get("priority_evidence", "")))
        rec_action = str(alert_row.get("recommended_action", ""))
        evidence_level = str(alert_row.get("evidence_level", "Rule match available"))
        score_evidence = scoring_by_source.get(source_row, {})
        data_sources = ["S.O.S."]
        if safe_float(score_evidence.get("telemetry_available")) > 0:
            data_sources.append("telemetry")
        if safe_float(score_evidence.get("work_order_history_available")) > 0:
            data_sources.append("work-order history")

        if predictive_risk_enabled:
            model_prob = sample_model_probs.get(source_row)
            anom_score = telemetry_anomaly_scores.get(asset_id, None)
            prob_pct: int | None = int(round(model_prob * 100)) if model_prob is not None else None
            if prob_pct is None:
                risk_badge = "🔒 Not Available"
                predicted_risk_status = "Predicted Failure Risk: Not Available for this sample"
            elif prob_pct >= 70:
                risk_badge = "🔴 HIGH MODEL PROBABILITY"
                predicted_risk_status = f"{prob_pct}% model-estimated event probability"
            elif prob_pct >= 40:
                risk_badge = "🟡 MEDIUM MODEL PROBABILITY"
                predicted_risk_status = f"{prob_pct}% model-estimated event probability"
            else:
                risk_badge = "🟢 LOW MODEL PROBABILITY"
                predicted_risk_status = f"{prob_pct}% model-estimated event probability"
            drivers = _extract_sample_risk_drivers(alert_row, top_features)
        else:
            # AR alone → never produces a failure probability
            prob_pct = None
            risk_badge = "🔒 Not Available"
            predicted_risk_status = "Predicted Failure Risk: Not Available — insufficient labelled data"
            drivers = _extract_sample_risk_drivers(alert_row, [])

        maintenance_action_cards.append({
            "source_row": source_row,
            "sample_number": sample_num,
            "asset_id": asset_id,
            "component": component,
            "site_name": str(alert_row.get("site_name", "Unknown")),
            "machine_model": str(alert_row.get("machine_model", "")),
            "sample_date": alert_row.get("sample_date"),
            "sample_date_raw": alert_row.get("sample_date_raw"),
            "is_invalid_date": safe_bool(alert_row.get("is_invalid_date")),
            "data_quality_status": str(alert_row.get("data_quality_status", "Valid")),
            "data_quality_issue": str(alert_row.get("data_quality_issue", "")),
            "interpretation_code": interp_code,
            "lab_status": lab_status,
            "priority_tier": priority_tier,
            "priority_reason": priority_reason,
            "rule_evidence_strength": round(rule_evidence_strength, 3),
            "evidence_level": evidence_level,
            "key_evidence": drivers,
            "action_needed": rec_action,
            "wo_id": alert_row.get("wo_id"),
            "wo_status": alert_row.get("wo_status"),
            "status": str(alert_row.get("status", "")),
            "high_priority": str(alert_row.get("high_priority", "")),
            "predicted_risk_status": predicted_risk_status,
            "failure_probability_pct": prob_pct,
            "risk_badge": risk_badge,
            "telemetry_anomaly_score": telemetry_anomaly_scores.get(asset_id),
            "prediction_data_sources": ", ".join(data_sources),
            "telemetry_available": "telemetry" in data_sources,
            "work_order_history_available": "work-order history" in data_sources,
            # Backward-compat aliases
            "workflow_priority": priority_tier,
            "priority_evidence": priority_reason,
            "operational_priority": priority_tier,
        })

    cards_df = pd.DataFrame(maintenance_action_cards)

    return {
        "operating_mode": operating_mode,
        "mode": operating_mode,
        "predictive_risk_enabled": predictive_risk_enabled,
        "dataset_metrics": dataset_metrics,
        "sos": sos,
        "alerts": alerts,
        "interp_categories": interp_categories,
        "telemetry": telemetry,
        "telemetry_quality": telemetry_quality,
        "telemetry_summary": telemetry_asset_summary(telemetry),
        "work_orders": work_orders,
        "scoring_table": scoring,
        "training_table": training,
        "maintenance_action_cards": cards_df,
        "predictive_cards": cards_df,
        "readiness": readiness,
        "matched_assets": matched_assets,
        "trained_model": trained_model,
        "leaderboard": leaderboard,
        "model_error": model_error,
    }


# ---------------------------------------------------------------------------
# JSON serialisation helper
# ---------------------------------------------------------------------------

def _json_safe(value: Any) -> Any:
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return pd.Timestamp(value).isoformat()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, set):
        return sorted(value)
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


# ---------------------------------------------------------------------------
# CLI / batch entry point
# ---------------------------------------------------------------------------

def run_analysis(
    sos_path: str | Path,
    telemetry_path: str | Path | None = None,
    output_dir: str | Path = "outputs/current",
    work_orders_path: str | Path | None = None,
    horizon_days: int = 30,
    test_details_path: str | Path | None = None,
) -> dict[str, Any]:
    telemetry_raw = load_table(telemetry_path) if telemetry_path else None
    work_orders_raw = load_table(work_orders_path) if work_orders_path else None
    sos_raw = load_table(sos_path)
    if test_details_path:
        sos_raw = enrich_sos_with_test_details(
            sos_raw, load_table(test_details_path)
        )
    result = analyze_frames(
        sos_raw,
        telemetry_raw,
        work_orders_raw,
        horizon_days=horizon_days,
    )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    result["alerts"].to_csv(output / "sos_engineering_alerts.csv", index=False)
    result["telemetry_summary"].to_csv(output / "telemetry_asset_summary.csv", index=False)
    result["telemetry"].to_csv(output / "telemetry_cleaned_scored.csv", index=False)
    result["readiness"].to_csv(output / "model_readiness.csv", index=False)
    result["scoring_table"].to_csv(output / "scoring_features.csv", index=False)
    result["maintenance_action_cards"].to_csv(output / "sample_predictions.csv", index=False)
    if result["training_table"] is not None:
        result["training_table"].to_csv(output / "training_table.csv", index=False)
    if result["trained_model"] is not None:
        result["trained_model"].save(output / "failure_model.joblib")

    maintenance_summary = generate_maintenance_summary(result)
    result["maintenance_summary"] = maintenance_summary
    with (output / "maintenance_summary.md").open("w", encoding="utf-8") as handle:
        handle.write(maintenance_summary)

    summary = {
        "operating_mode": result["operating_mode"],
        "sos_samples": len(result["sos"]),
        "sos_assets": result["sos"]["asset_id"].nunique(dropna=True),
        "telemetry_rows_after_deduplication": len(result["telemetry"]),
        "telemetry_assets": result["telemetry"]["asset_id"].nunique(dropna=True),
        "matched_assets": result["matched_assets"],
        "prediction_rows": int(result["maintenance_action_cards"]["failure_probability_pct"].notna().sum()),
        "trained_model": result["trained_model"].name if result["trained_model"] is not None else None,
        "telemetry_quality": result["telemetry_quality"],
        "dataset_metrics": result["dataset_metrics"],
    }
    with (output / "analysis_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, default=_json_safe)
    return result
