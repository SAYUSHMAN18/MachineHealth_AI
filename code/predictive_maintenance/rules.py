from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .data import safe_bool, safe_text


DEFAULT_RULES_PATH = Path(__file__).resolve().parents[2] / "config" / "rules.yaml"


def load_rules(path: str | Path | None = None) -> dict:
    with Path(path or DEFAULT_RULES_PATH).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def evaluate_sos_rules(sos: pd.DataFrame, rules_path: str | Path | None = None) -> pd.DataFrame:
    """Convert S.O.S interpretation text and metadata into auditable alerts and workflow maintenance priorities."""
    config = load_rules(rules_path)
    severity_codes = {str(k).upper(): float(v) for k, v in config["severity_codes"].items()}

    # Pre-count repeated New AR occurrences per asset & component
    new_ar_counts = (
        sos[
            (sos["interpretation_code"].astype(str).str.upper() == "AR")
            & (sos["status"].astype(str).str.upper() == "NEW")
        ]
        .groupby(["asset_id", "component"], dropna=False)
        .size()
        .to_dict()
    )

    rows: list[dict] = []

    for _, sample in sos.iterrows():
        text = safe_text(sample.get("interpretation_text"))
        code = safe_text(sample.get("interpretation_code")).upper()
        base_score = severity_codes.get(code, 0.25)
        evidence: list[str] = []
        recommendations: list[str] = []
        triggered: list[str] = []
        scores = [base_score]

        for rule in config["rules"]:
            matched = any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in rule["patterns"])
            if matched:
                triggered.append(rule["id"])
                scores.append(float(rule["score"]))
                evidence.append(rule["evidence"])
                recommendations.append(rule["recommendation"])

        if code == "AR":
            evidence.insert(0, "Laboratory interpretation code is AR (Laboratory Action Required)")
            recommendations.insert(0, "Confirm sample parameters and review unit history with data owner.")
        if not evidence:
            evidence.append("No configured fault signature found in interpretation text")
        if not recommendations:
            recommendations.append("Review original laboratory interpretation and continue scheduled monitoring.")

        missing_fluid_hours = pd.isna(sample.get("fluid_hours"))
        history_warning = bool(re.search(r"more sample history needed", text, flags=re.IGNORECASE))
        if missing_fluid_hours or history_warning:
            confidence = "Low"
        elif len(triggered) >= 2:
            confidence = "High"
        else:
            confidence = "Medium"

        score = float(np.clip(max(scores), 0, 1))
        if score >= 0.8:
            evidence_level = "Strong rule match"
        elif score >= 0.5:
            evidence_level = "Moderate rule match"
        else:
            evidence_level = "Basic rule match"

        # Laboratory Status Display Name
        if code == "AR":
            lab_status = "Laboratory Action Required"
        elif code in ["A", "NAR", "NORMAL"]:
            lab_status = "Normal"
        elif code in ["B", "YELLOW", "WATCH"]:
            lab_status = "Warning"
        elif code in ["C", "RED"]:
            lab_status = "Severe"
        else:
            lab_status = f"Code {code}" if code else "Unspecified"

        # Workflow Maintenance Priority Resolution
        sample_date = sample.get("sample_date")
        asset_id = sample.get("asset_id")
        component = sample.get("component")
        status_val = sample.get("status")
        status_str = "New" if pd.isna(status_val) else str(status_val).strip().capitalize()
        high_pri_val = sample.get("high_priority")
        high_pri = "" if pd.isna(high_pri_val) else str(high_pri_val).strip().upper()
        wo_id = sample.get("wo_id")
        has_wo = pd.notna(wo_id) and str(wo_id).strip() not in ["", "<NA>", "nan", "None"]

        is_invalid_date = safe_bool(sample.get("is_invalid_date")) or pd.isna(sample_date) or (
            isinstance(sample_date, pd.Timestamp) and (sample_date.year < 1950 or sample_date.year > 2035)
        )

        if is_invalid_date:
            workflow_priority = "Data quality hold"
            priority_evidence = "Missing or time-only sample date (e.g., year 1899 or unparseable date format)"
        elif status_str.upper() == "CLOSED":
            workflow_priority = "Closed"
            priority_evidence = "SampleStatusNew is Closed"
        elif high_pri in ["T", "TRUE", "1", "Y", "YES"] and status_str.upper() == "NEW" and not has_wo:
            workflow_priority = "Immediate review"
            priority_evidence = "HighPriority=T, Status=New, and no WorkOrderId assigned"
        elif high_pri in ["T", "TRUE", "1", "Y", "YES"] and status_str.upper() == "NEW" and has_wo:
            workflow_priority = "Immediate review"
            priority_evidence = f"HighPriority=T, Status=New, with assigned WorkOrderId ({wo_id})"
        elif code == "AR" and status_str.upper() == "NEW":
            repeat_count = new_ar_counts.get((asset_id, component), 0)
            if repeat_count >= 2:
                workflow_priority = "Priority review"
                wo_info = "without WorkOrderId" if not has_wo else f"with WorkOrderId ({wo_id})"
                priority_evidence = f"Repeated New AR sample ({repeat_count} occurrences) for {asset_id} / {component} {wo_info}"
            elif has_wo:
                workflow_priority = "In progress"
                priority_evidence = f"New AR sample with assigned WorkOrderId ({wo_id})"
            else:
                workflow_priority = "Priority review"
                priority_evidence = "New AR sample without WorkOrderId assigned"
        elif has_wo:
            workflow_priority = "In progress"
            priority_evidence = f"Assigned WorkOrderId ({wo_id})"
        elif code in ["B", "C", "AR", "RED", "YELLOW"]:
            workflow_priority = "Priority review"
            priority_evidence = f"Elevated laboratory code ({code}) without assigned WorkOrderId"
        else:
            workflow_priority = "Closed" if status_str.upper() == "CLOSED" else "In progress"
            priority_evidence = f"Status {status_str}"

        # Non-diagnostic "Suggested engineering review" recommendations
        unique_recommendations = list(dict.fromkeys(recommendations))
        formatted_recommendation = "Suggested engineering review: " + " ".join(
            recommendation.removeprefix("Suggested engineering review:").strip()
            for recommendation in unique_recommendations
        )

        rows.append(
            {
                "source_row": sample.get("source_row"),
                "sample_number": sample.get("sample_number"),
                "asset_id": asset_id,
                "serial_number": sample.get("serial_number"),
                "machine_model": sample.get("machine_model"),
                "component": component,
                "sample_date": sample_date,
                "sample_date_raw": sample.get("sample_date_raw"),
                "is_invalid_date": is_invalid_date,
                "date_quality_issue": sample.get("date_quality_issue"),
                "interpretation_code": code,
                "lab_status": lab_status,
                "workflow_priority": workflow_priority,
                "priority_evidence": priority_evidence,
                "operational_priority": workflow_priority,  # Alias for backward compatibility
                "status": status_str,
                "high_priority": high_pri,
                "wo_id": wo_id if has_wo else None,
                "wo_status": sample.get("wo_status"),
                "site_name": sample.get("site_name", "Unknown"),
                "rule_evidence_strength": round(score, 3),
                "evidence_level": evidence_level,
                "confidence": confidence,
                "triggered_rules": "; ".join(triggered) or "none",
                "evidence": "; ".join(dict.fromkeys(evidence)),
                "recommended_action": formatted_recommendation,
                "original_interpretation": text,
            }
        )

    return pd.DataFrame(rows)
