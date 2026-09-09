from __future__ import annotations

import os
import sys
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

CODE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CODE_DIR.parent
sys.path.insert(0, str(CODE_DIR))

from predictive_maintenance.data import load_table  # noqa: E402
from predictive_maintenance.features import RAW_MEASUREMENT_COLUMNS  # noqa: E402
from predictive_maintenance.llm import generate_llm_insights  # noqa: E402
from predictive_maintenance.pipeline import MODE_ALERT, MODE_CONDITION, MODE_PREDICTION, TIER_CLOSED, analyze_frames  # noqa: E402
from predictive_maintenance.presentation import (  # noqa: E402
    MEASUREMENT_LABELS, build_case_table, build_fleet_summary, build_mode_summary,
    build_quality_issues, build_validation_summary, feature_label,
    latest_measurement_changes, measurement_trend_table, telemetry_asset_insight,
)

st.set_page_config(page_title="Machinery Reliability Decision Support", page_icon="🚜", layout="wide")
st.markdown("""<style>
.block-container{padding-top:1.4rem;padding-bottom:3rem}.decision{border-left:6px solid #d97706;background:#fff8e8;padding:1rem 1.2rem;border-radius:.5rem;margin:.4rem 0 1rem}.mode{border:1px solid #cbd5e1;background:#f8fafc;padding:.9rem 1rem;border-radius:.6rem}.note{color:#475569;font-size:.9rem}[data-testid="stMetric"]{border:1px solid #e2e8f0;padding:.7rem;border-radius:.6rem}
</style>""", unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def load_path(path: str) -> pd.DataFrame:
    return load_table(path)


def bundled(source: str):
    if source == "Matched prediction demo":
        base = PROJECT_ROOT / "data/demo"
        return (load_path(str(base / "Demo_SOSFluidAnalysis.xlsx")),
                load_path(str(base / "Demo_TelematicData.xlsx")),
                load_path(str(base / "Demo_WorkOrders.xlsx")))
    base = PROJECT_ROOT / "data/current"
    return load_path(str(base / "SosFluidSample.xlsx")), load_path(str(base / "TelematicDataSample.xlsx")), None


def date_text(value) -> str:
    parsed = pd.to_datetime(value, errors="coerce")
    return parsed.strftime("%d %b %Y") if pd.notna(parsed) else "Date unavailable"


def safe(value) -> str:
    """Escape spreadsheet-derived text before rendering an HTML decision card."""
    return escape(str(value if pd.notna(value) else ""))


def metrics(items):
    for col, (label, value, help_text) in zip(st.columns(len(items)), items):
        col.metric(label, value, help=help_text)


def mode_banner(summary):
    icon = "🔮" if summary["mode"] == MODE_PREDICTION else "📈" if summary["mode"] == MODE_CONDITION else "🔔"
    st.markdown(f'<div class="mode"><b>{icon} {summary["title"]}</b><br><span class="note">{summary["description"]}</span></div>', unsafe_allow_html=True)
    if summary["blocking_reasons"] and not summary["prediction_enabled"]:
        with st.expander("Why failure probability is unavailable"):
            for item in summary["blocking_reasons"]:
                st.write(f"• {item}")


def display_cases(frame):
    if frame.empty:
        return frame
    renamed = frame.rename(columns={"priority":"Priority", "asset_id":"Machine", "component_name":"Component",
        "main_issue":"Main finding", "probability_display":"Failure probability", "wo_status_display":"Work order",
        "prediction_data_sources":"Model inputs", "required_action":"Required action", "due":"When",
        "data_confidence":"Evidence confidence"})
    return renamed[["Priority", "Machine", "Component", "Main finding", "Failure probability", "Model inputs", "Work order", "Required action", "When", "Evidence confidence"]]


with st.sidebar:
    st.title("🚜 Reliability Hub")
    source = st.radio("Data source", ["Supplied current data", "Matched prediction demo", "Upload your own files"])
    sos_file = telemetry_file = wo_file = None
    if source == "Upload your own files":
        sos_file = st.file_uploader("S.O.S fluid file (required)", type=["xlsx", "xls", "csv"])
        telemetry_file = st.file_uploader("Telemetry file (optional)", type=["xlsx", "xls", "csv"])
        wo_file = st.file_uploader("Work-order file (optional)", type=["xlsx", "xls", "csv"])
    horizon = st.select_slider("Prediction horizon (days)", options=[15, 30, 60, 90], value=30,
        help="Used only if work-order outcome and validation gates pass.")
    st.caption("A requested horizon is not proof that prediction is available.")
    with st.expander("Optional external AI summary"):
        external = st.checkbox("Allow aggregated data to be sent to Gemini", value=False)
        api_key = st.text_input("Gemini API key", type="password", disabled=not external,
                                value=os.environ.get("GEMINI_API_KEY", "") if external else "")
        st.caption("Off by default. Core calculations do not require AI.")
    run = st.button("Run analysis", type="primary", width="stretch")

signature = (source, int(horizon))
if run or "analysis" not in st.session_state or st.session_state.get("signature") != signature:
    if source == "Upload your own files" and sos_file is None:
        st.info("Upload the S.O.S file, then select **Run analysis**.")
        st.stop()
    frames = ((load_table(sos_file), load_table(telemetry_file) if telemetry_file else None,
               load_table(wo_file) if wo_file else None) if source == "Upload your own files" else bundled(source))
    with st.status("Running engineering checks…", expanded=True) as status:
        st.write("1. Standardising identifiers, dates, units and duplicate records")
        st.write("2. Joining S.O.S, telemetry and work orders without inventing outcomes")
        analysis = analyze_frames(*frames, horizon_days=horizon)
        st.write("3. Selecting the highest justified operating mode and building machine-level cases")
        analysis["ai_insights"] = generate_llm_insights(analysis, api_key=api_key, allow_external=external)
        status.update(label="Analysis completed", state="complete", expanded=False)
    st.session_state.update(analysis=analysis, signature=signature, analysis_source=source)

result = st.session_state["analysis"]
mode = build_mode_summary(result, horizon)
cases = build_case_table(result, horizon)
fleet = build_fleet_summary(result, cases, horizon)

st.title("Heavy Machinery Condition & Failure Decision Support")
mode_banner(mode)
if st.session_state.get("analysis_source") == "Matched prediction demo":
    st.warning("Training demonstration: these synthetic records must not be used for real maintenance decisions.")

page = st.radio("View", ["Fleet Overview", "Action Queue", "Asset Analysis", "Model Validation", "Data Quality"], horizontal=True, label_visibility="collapsed")

if page == "Fleet Overview":
    st.header("Fleet Overview")
    st.markdown(f'<div class="decision"><b>What the analysis found</b><br>{fleet["conclusion"]}</div>', unsafe_allow_html=True)
    metrics([("P1 cases today", fleet["immediate_action"], "Unique machine-component cases"),
             ("High-severity cases", fleet["high"], None), ("Monitor cases", fleet["monitor"], None),
             ("Normal cases", fleet["normal"], None), ("Open corrective WOs", fleet["open_corrective_wos"], None),
             ("Evidence confidence", fleet["analysis_confidence"], "Coverage of dates, measurements, telemetry and WOs")])
    st.subheader("Recommended next step")
    st.info(fleet["recommendation"])
    left, right = st.columns([1, 1.5])
    with left:
        counts = cases["priority"].value_counts().rename_axis("Priority").reset_index(name="Cases") if not cases.empty else pd.DataFrame()
        if not counts.empty:
            st.plotly_chart(px.bar(counts, x="Cases", y="Priority", orientation="h", title="Unique machine-component cases"), width="stretch")
    with right:
        st.subheader("Highest-priority cases")
        st.dataframe(display_cases(cases.head(10)), hide_index=True, width="stretch")
    with st.expander("How this conclusion was produced"):
        st.markdown("""1. Validate identifiers, dates and numerical fields.  
2. Keep laboratory status separate from confirmed failure outcomes.  
3. Collapse repeated samples into one machine-component case.  
4. Apply engineering priority and WO follow-up logic.  
5. Show probability only after independent WO labels and chronological validation pass.""")
    st.download_button("Download machine-level action queue", cases.to_csv(index=False).encode(), "machine_action_queue.csv", "text/csv")

elif page == "Action Queue":
    st.header("Action Queue")
    if cases.empty:
        st.info("No cases were produced."); st.stop()
    a, b, c = st.columns(3)
    sites = a.multiselect("Site", sorted(cases["site_name"].dropna().unique()))
    priorities = b.multiselect("Priority", list(cases["priority"].dropna().unique()))
    components = c.multiselect("Component", sorted(cases["component_name"].dropna().unique()))
    filtered = cases.copy()
    if sites: filtered = filtered[filtered["site_name"].isin(sites)]
    if priorities: filtered = filtered[filtered["priority"].isin(priorities)]
    if components: filtered = filtered[filtered["component_name"].isin(components)]
    if not st.checkbox("Include closed laboratory records", value=False):
        filtered = filtered[~filtered["priority_code"].eq(TIER_CLOSED)]
    st.caption(f"{len(filtered):,} unique machine-component cases. Repeated samples are not double-counted.")
    st.dataframe(display_cases(filtered), hide_index=True, width="stretch", height=420)
    if not filtered.empty:
        labels = {x.case_id: f"{x.priority} | {x.asset_id} | {x.component_name}" for x in filtered.itertuples()}
        chosen = st.selectbox("Open case", list(labels), format_func=labels.get)
        row = filtered[filtered["case_id"].eq(chosen)].iloc[0]
        st.subheader(f'{row["asset_id"]} — {row["component_name"]}')
        st.markdown(f'<div class="decision"><b>Decision:</b> {safe(row["required_action"])}<br><b>Main evidence:</b> {safe(row["main_issue"])}<br><b>Timing:</b> {safe(row["due"])}</div>', unsafe_allow_html=True)
        metrics([("Priority", row["priority"], None), ("Latest sample", date_text(row["latest_sample_date"]), None),
                 ("Failure probability", row["probability_display"], "Unavailable unless all prediction gates pass"),
                 ("Model inputs", row["prediction_data_sources"], "Optional sources appear only when available for this sample"),
                 ("Work order", row["wo_status_display"], None), ("Evidence confidence", row["data_confidence"], row["confidence_reason"])])

elif page == "Asset Analysis":
    st.header("Asset Analysis")
    if cases.empty:
        st.info("No cases are available."); st.stop()
    asset = st.selectbox("Machine", sorted(cases["asset_id"].unique()))
    ac = cases[cases["asset_id"].eq(asset)]
    component_name = st.selectbox("Component", sorted(ac["component_name"].unique()))
    row = ac[ac["component_name"].eq(component_name)].iloc[0]
    st.markdown(f'<div class="decision"><b>{safe(row["priority"])}</b><br>{safe(row["required_action"])}<br><span class="note">Evidence: {safe(row["main_issue"])}</span></div>', unsafe_allow_html=True)
    metrics([("Lab status", row["lab_status"], None), ("Failure probability", row["probability_display"], None),
             ("WO status", row["wo_status_display"], None), ("Evidence confidence", row["data_confidence"], row["confidence_reason"])])
    trend = measurement_trend_table(result.get("sos", pd.DataFrame()), asset, row["component"])
    changes = latest_measurement_changes(trend, limit=8)
    left, right = st.columns([1, 1.5])
    with left:
        st.subheader("Latest numerical changes")
        if changes.empty:
            st.info("No dated numerical series. This case uses laboratory text and workflow fields only.")
        else:
            shown = changes.rename(columns={"measurement_name":"Measurement", "current":"Current", "previous":"Previous", "change":"Change", "unit":"Unit"})
            st.dataframe(shown[["Measurement", "Current", "Previous", "Change", "Unit"]], hide_index=True, width="stretch")
    with right:
        st.subheader("Laboratory trend")
        available = [x for x in RAW_MEASUREMENT_COLUMNS if x in trend and trend[x].notna().any()]
        if available:
            measure = st.selectbox("Measurement", available, format_func=lambda x: MEASUREMENT_LABELS.get(x, x))
            chart = px.line(trend, x="sample_date", y=measure, markers=True,
                            title=f'{MEASUREMENT_LABELS.get(measure, measure)} — measured trend, not a universal alarm limit')
            st.plotly_chart(chart, width="stretch")
        else:
            st.info("No structured laboratory measurement is available to plot.")
    telem = telemetry_asset_insight(result.get("telemetry", pd.DataFrame()), asset)
    st.subheader("Telemetry evidence")
    st.write(f'**{telem["status"]}:** {telem["summary"]}')
    with st.expander("Raw evidence for audit"):
        st.dataframe(trend, hide_index=True, width="stretch")
        wo = result.get("work_orders")
        if isinstance(wo, pd.DataFrame) and not wo.empty:
            st.dataframe(wo[wo["asset_id"].astype(str).eq(str(asset))], hide_index=True, width="stretch")

elif page == "Model Validation":
    st.header("Model Validation")
    validation = build_validation_summary(result)
    if not mode["prediction_enabled"] or validation is None:
        st.warning("No failure probability was calculated. This is a safety result, not a software failure.")
        st.write("Alert prioritisation, trend review and work-order follow-up remain available.")
        st.dataframe(result.get("readiness", pd.DataFrame()), hide_index=True, width="stretch")
    else:
        st.success(validation["plain_language"])
        metrics([("Newer test samples", validation["test_rows"], None),
                 ("Precision", f'{validation["precision"]:.1%}', "Of predicted events, how many occurred"),
                 ("Recall", f'{validation["recall"]:.1%}', "Of actual events, how many were found"),
                 ("F1", f'{validation["f1"]:.1%}', "Balance of precision and recall"),
                 ("Decision threshold", f'{validation["decision_threshold"]:.2f}', "Selected using validation data only"),
                 ("Brier score", f'{validation["brier_score"]:.3f}', "Probability error; lower is better"),
                 ("AP lift vs baseline", f'{validation["average_precision_lift"]:+.3f}', "Must be at least +0.020"),
                 ("Brier skill", f'{validation["brier_skill_score"]:.1%}', "Improvement over a constant probability; must be positive")])
        matrix = np.array([[validation["true_negative"], validation["false_positive"]], [validation["false_negative"], validation["true_positive"]]])
        fig = go.Figure(go.Heatmap(z=matrix, x=["Predicted no event", "Predicted event"], y=["Actual no event", "Actual event"], text=matrix, texttemplate="%{text}", colorscale="Blues"))
        fig.update_layout(title="Chronological holdout confusion matrix")
        st.plotly_chart(fig, width="stretch")
        trained_model = result.get("trained_model")
        importance_values = trained_model.get_feature_importances() if trained_model is not None else {}
        importance = pd.DataFrame(
            [{"feature": name, "importance": value} for name, value in importance_values.items()]
        )
        if not importance.empty:
            shown = importance.head(10).copy(); feature_col = "feature"
            shown[feature_col] = shown[feature_col].map(feature_label)
            st.subheader("Global model drivers")
            st.caption("Model influence across the dataset—not proof of root cause for one machine.")
            st.dataframe(shown, hide_index=True, width="stretch")
        with st.expander("Candidate-model comparison"):
            st.dataframe(result.get("leaderboard", pd.DataFrame()), hide_index=True, width="stretch")

else:
    st.header("Data Quality & Prediction Readiness")
    issues = build_quality_issues(result); dm = result.get("dataset_metrics", {})
    metrics([("S.O.S rows", f'{len(result.get("sos", [])):,}', None), ("Valid sample dates", f'{dm.get("valid_date_samples", 0):,}', None),
             ("Invalid sample dates", f'{dm.get("invalid_date_samples", 0):,}', None), ("Matched telemetry assets", len(result.get("matched_assets", set())), None)])
    st.subheader("Issues that change what can be concluded")
    shown = issues.rename(columns={"severity":"Severity", "issue":"Issue", "count":"Rows", "impact":"Analysis impact", "fix":"Required fix"})
    st.dataframe(shown, hide_index=True, width="stretch")
    st.subheader("Prediction readiness gates")
    st.dataframe(result.get("readiness", pd.DataFrame()), hide_index=True, width="stretch")
    st.download_button("Download data-quality issues", issues.to_csv(index=False).encode(), "data_quality_issues.csv", "text/csv")
    with st.expander("Laboratory interpretation categories"):
        st.dataframe(result.get("interp_categories", pd.DataFrame()), hide_index=True, width="stretch")

st.divider()
st.caption("Decision support only. Maintenance action remains the responsibility of qualified reliability personnel.")
