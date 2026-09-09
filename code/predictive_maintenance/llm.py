from __future__ import annotations

import json
import os
from typing import Any

import pandas as pd

_MODE_ALERT = "Alert Management"
_MODE_CONDITION = "Condition Monitoring"
_MODE_PREDICTION = "Failure Prediction"


def _generate_fallback_insights(analysis_result: dict[str, Any]) -> str:
    """Generate structured expert maintenance recommendations without an active API key."""
    alerts = analysis_result.get("alerts", pd.DataFrame())
    telemetry_quality = analysis_result.get("telemetry_quality", {})
    readiness = analysis_result.get("readiness", pd.DataFrame())
    operating_mode = analysis_result.get("operating_mode", analysis_result.get("mode", _MODE_ALERT))
    dataset_metrics = analysis_result.get("dataset_metrics", {})

    p1_count = p2_count = p3_count = dq_count = 0
    if not alerts.empty and "priority_tier" in alerts.columns:
        p1_count = int(alerts["priority_tier"].isin(["P1 \u2013 Immediate Review", "P1 \u2013 WO Tracking"]).sum())
        p2_count = int((alerts["priority_tier"] == "P2 \u2013 Multiple Unlinked Alerts").sum())
        p3_count = int((alerts["priority_tier"] == "P3 \u2013 Action Required").sum())
        dq_count = int(alerts.get("is_invalid_date", pd.Series(False, index=alerts.index)).sum())

    probability_notice = (
        "> **Important**: `OverallInterp=AR` is a laboratory flag, not the model target. "
        "Displayed probabilities come only from the independently labelled, chronologically tested model."
        if operating_mode == _MODE_PREDICTION
        else "> **Important**: `OverallInterp=AR` is a laboratory flag. It is **not** a failure prediction. "
        "No failure probability is calculated in this report."
    )
    lines = [
        "# Maintenance Triage Summary",
        "",
        "## Executive Summary",
        f"- **Operating Mode**: `{operating_mode}`",
        f"- **Total S.O.S Samples Analysed**: {len(alerts):,}",
        f"- **Unique Assets**: {dataset_metrics.get('unique_assets', 0):,}",
        f"- **Laboratory AR Samples**: {dataset_metrics.get('lab_ar_samples', 0):,}",
        f"- **P1 Immediate / WO Tracking Alerts**: {p1_count}",
        f"- **P2 Multiple Unlinked Alert Records**: {p2_count}",
        f"- **P3 Action Required (New, No WO)**: {p3_count}",
        f"- **Records With Invalid Sample Dates**: {dq_count}",
        "",
        probability_notice,
        "",
        "---",
        "",
        "## Priority Action Summary",
    ]

    if p1_count > 0 and "priority_tier" in alerts.columns:
        p1_rows = alerts[alerts["priority_tier"].isin(["P1 \u2013 Immediate Review", "P1 \u2013 WO Tracking"])]
        for _, row in p1_rows.head(5).iterrows():
            lines.extend([
                f"### {row.get('asset_id', '?')} / {row.get('component', '?')}",
                f"- **Priority**: {row.get('priority_tier', '')}",
                f"- **Priority Reason**: {row.get('priority_reason', row.get('priority_evidence', ''))}",
                f"- **Lab Status**: {row.get('lab_status', 'Laboratory Action Required')}",
                f"- **Rule Evidence**: {row.get('evidence_level', 'Rule match available')} *(not a failure probability)*",
                f"- **Suggested Engineering Review**: {row.get('recommended_action', 'Review laboratory interpretation and raise a work order.')}",
                "",
            ])
    else:
        lines.extend(["No P1 Immediate or WO Tracking alerts detected.", ""])

    if p2_count > 0 and "priority_tier" in alerts.columns:
        lines.append("## P2 – Multiple Unlinked Alert Records")
        p2_rows = (
            alerts[alerts["priority_tier"] == "P2 \u2013 Multiple Unlinked Alerts"]
            .drop_duplicates(["asset_id", "component"])
        )
        for _, row in p2_rows.head(5).iterrows():
            lines.append(
                f"- **{row.get('asset_id', '?')} / {row.get('component', '?')}**: "
                f"{row.get('priority_reason', 'Repeated AR without work order')}"
            )
        lines.append("")

    lines.extend([
        "---", "",
        "## Work-Order Linkage Coverage",
        f"- WO-linked samples: **{dataset_metrics.get('wo_linked_samples', 0):,}**",
        f"- Samples without WO: **{dataset_metrics.get('samples_without_wo', 0):,}**",
        f"- New alerts without WO: **{dataset_metrics.get('new_alerts_without_wo', 0):,}**",
        "", "---", "",
        "## Data Quality",
        f"- Invalid / time-only sample dates: **{dataset_metrics.get('invalid_date_samples', 0):,}** "
        "(excluded from date-dependent analysis)",
        "", "---", "",
        "## Prediction Readiness",
    ])

    if not readiness.empty:
        for _, r in readiness.iterrows():
            lines.append(f"- **{r['status']} — {r['criterion']}**: `{r['detail']}`")

    unlock_msg = {
        _MODE_ALERT: "Resolve the BLOCKED gates above; alert-only data cannot unlock failure prediction.",
        _MODE_CONDITION: "Add sufficient explicit corrective outcomes, then pass chronological validation. Telemetry remains optional enrichment.",
        _MODE_PREDICTION: "All required prediction-readiness gates are PASS — ML failure prediction is active; telemetry is used where available.",
    }.get(operating_mode, "")
    if unlock_msg:
        lines.extend(["", f"> **Next requirement**: {unlock_msg}"])

    lines.extend([
        "", "---",
        "*Deterministic evidence summary; probabilities appear only when the validated prediction mode is active.*",
    ])
    return "\n".join(lines)


def generate_llm_insights(
    analysis_result: dict[str, Any],
    api_key: str | None = None,
    allow_external: bool = False,
) -> str:
    """Generate a local summary by default; call Gemini only after explicit opt-in."""
    if not allow_external:
        return _generate_fallback_insights(analysis_result)
    effective_api_key = api_key or os.environ.get("GEMINI_API_KEY")
    if not effective_api_key:
        return _generate_fallback_insights(analysis_result)

    try:
        import google.generativeai as genai
        genai.configure(api_key=effective_api_key)

        alerts = analysis_result.get("alerts", pd.DataFrame())
        operating_mode = analysis_result.get("operating_mode", analysis_result.get("mode", _MODE_ALERT))
        dataset_metrics = analysis_result.get("dataset_metrics", {})
        readiness = analysis_result.get("readiness", pd.DataFrame())

        p1_rows = pd.DataFrame()
        if not alerts.empty and "priority_tier" in alerts.columns:
            p1_rows = alerts[alerts["priority_tier"].isin(
                ["P1 \u2013 Immediate Review", "P1 \u2013 WO Tracking"]
            )].head(10)

        if operating_mode == _MODE_ALERT:
            mode_instruction = (
                "IMPORTANT: This dataset is in Alert Management mode. Every record has OverallInterp=AR. "
                "You MUST NOT calculate failure probabilities, assign risk percentages, or confirm any diagnosis. "
                "Summarise laboratory evidence only. Use 'Suggested engineering review' language."
            )
        elif operating_mode == _MODE_CONDITION:
            mode_instruction = (
                "This dataset is in Condition Monitoring mode. You may reference measurement trends "
                "to indicate concern levels, but must not produce failure probabilities."
            )
        else:
            mode_instruction = (
                "This dataset is in Failure Prediction mode. ML failure probabilities are available. "
                "Reference them accurately."
            )

        p1_data = "None"
        if not p1_rows.empty:
            # Asset identifiers are deliberately excluded from the external prompt.
            cols = [c for c in ["component", "priority_tier", "priority_reason",
                                "rule_evidence_strength", "recommended_action"] if c in p1_rows.columns]
            p1_data = str(p1_rows[cols].to_dict(orient="records"))

        prompt = f"""You are an expert Senior Reliability Engineer and Heavy Equipment Maintenance Specialist.
{mode_instruction}

Operating Mode: {operating_mode}
Total Samples: {len(alerts):,} | Unique Assets: {dataset_metrics.get('unique_assets', 0):,}
Laboratory AR Samples: {dataset_metrics.get('lab_ar_samples', 0):,}
HighPriority=T: {dataset_metrics.get('high_priority_samples', 0)} | New: {dataset_metrics.get('new_alerts', 0):,} | Closed: {dataset_metrics.get('closed_alerts', 0):,}
WO-Linked: {dataset_metrics.get('wo_linked_samples', 0):,} | Without WO: {dataset_metrics.get('samples_without_wo', 0):,}
Invalid Dates: {dataset_metrics.get('invalid_date_samples', 0):,}

P1 Priority Records: {p1_data}
Readiness Gates: {readiness.to_dict(orient='records') if not readiness.empty else 'None'}

Provide in Markdown with emojis:
1. Executive Summary
2. P1 & P2 Priority Queue (evidence summaries, suggested reviews — NO failure % or diagnoses in Alert/Condition modes)
3. Work-Order Coverage Analysis
4. Data Quality Issues
5. Prediction Readiness & Unlock Requirements
"""
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)
        if response and hasattr(response, "text") and response.text:
            return response.text
    except Exception as exc:
        print(f"Gemini API call failed ({exc}). Falling back to internal AI generator.")

    return _generate_fallback_insights(analysis_result)
