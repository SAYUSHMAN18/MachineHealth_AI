from __future__ import annotations

from typing import Any

import pandas as pd


MODE_ALERT = "Alert Management"
MODE_CONDITION = "Condition Monitoring"
MODE_PREDICTION = "Failure Prediction"


def generate_maintenance_summary(analysis_result: dict[str, Any]) -> str:
    """Build a deterministic local report from the calculated evidence."""
    alerts = analysis_result.get("alerts", pd.DataFrame())
    readiness = analysis_result.get("readiness", pd.DataFrame())
    operating_mode = analysis_result.get(
        "operating_mode", analysis_result.get("mode", MODE_ALERT)
    )
    metrics = analysis_result.get("dataset_metrics", {})

    p1_count = p2_count = p3_count = invalid_dates = 0
    if not alerts.empty and "priority_tier" in alerts.columns:
        p1_count = int(
            alerts["priority_tier"]
            .isin(["P1 – Immediate Review", "P1 – WO Tracking"])
            .sum()
        )
        p2_count = int(
            (alerts["priority_tier"] == "P2 – Multiple Unlinked Alerts").sum()
        )
        p3_count = int(
            (alerts["priority_tier"] == "P3 – Action Required").sum()
        )
        invalid_dates = int(
            alerts.get("is_invalid_date", pd.Series(False, index=alerts.index)).sum()
        )

    probability_notice = (
        "> **Important**: `OverallInterp=AR` is a laboratory flag, not the model target. "
        "Displayed probabilities come only from the independently labelled, "
        "chronologically tested model."
        if operating_mode == MODE_PREDICTION
        else "> **Important**: `OverallInterp=AR` is a laboratory flag. It is **not** "
        "a failure prediction. No failure probability is calculated in this report."
    )
    lines = [
        "# Maintenance Triage Summary",
        "",
        "## Executive Summary",
        f"- **Operating Mode**: `{operating_mode}`",
        f"- **Total S.O.S Samples Analysed**: {len(alerts):,}",
        f"- **Unique Assets**: {metrics.get('unique_assets', 0):,}",
        f"- **Laboratory AR Samples**: {metrics.get('lab_ar_samples', 0):,}",
        f"- **P1 Immediate / WO Tracking Alerts**: {p1_count}",
        f"- **P2 Multiple Unlinked Alert Records**: {p2_count}",
        f"- **P3 Action Required (New, No WO)**: {p3_count}",
        f"- **Records With Invalid Sample Dates**: {invalid_dates}",
        "",
        probability_notice,
        "",
        "## Priority Action Summary",
    ]

    if p1_count and "priority_tier" in alerts.columns:
        p1_rows = alerts[
            alerts["priority_tier"].isin(
                ["P1 – Immediate Review", "P1 – WO Tracking"]
            )
        ]
        for _, row in p1_rows.head(5).iterrows():
            lines.extend(
                [
                    f"### {row.get('asset_id', '?')} / {row.get('component', '?')}",
                    f"- **Priority**: {row.get('priority_tier', '')}",
                    f"- **Priority Reason**: {row.get('priority_reason', row.get('priority_evidence', ''))}",
                    f"- **Lab Status**: {row.get('lab_status', 'Laboratory Action Required')}",
                    f"- **Rule Evidence**: {row.get('evidence_level', 'Rule match available')} *(not a failure probability)*",
                    f"- **Suggested Engineering Review**: {row.get('recommended_action', 'Review laboratory interpretation and raise a work order.')}",
                    "",
                ]
            )
    else:
        lines.extend(["No P1 Immediate or WO Tracking alerts detected.", ""])

    lines.extend(
        [
            "## Work-Order Linkage Coverage",
            f"- WO-linked samples: **{metrics.get('wo_linked_samples', 0):,}**",
            f"- Samples without WO: **{metrics.get('samples_without_wo', 0):,}**",
            f"- New alerts without WO: **{metrics.get('new_alerts_without_wo', 0):,}**",
            "",
            "## Data Quality",
            f"- Invalid / time-only sample dates: **{metrics.get('invalid_date_samples', 0):,}**",
            "",
            "## Prediction Readiness",
        ]
    )
    if not readiness.empty:
        for _, row in readiness.iterrows():
            lines.append(
                f"- **{row['status']} — {row['criterion']}**: `{row['detail']}`"
            )

    unlock_message = {
        MODE_ALERT: "Resolve the BLOCKED gates above; alert-only data cannot unlock failure prediction.",
        MODE_CONDITION: "Add sufficient explicit corrective outcomes, then pass chronological validation.",
        MODE_PREDICTION: "All required gates pass — validated failure prediction is active.",
    }.get(operating_mode, "")
    if unlock_message:
        lines.extend(["", f"> **Next requirement**: {unlock_message}"])
    lines.extend(
        ["", "*Deterministic local evidence summary; no external AI service is used.*"]
    )
    return "\n".join(lines)
