from __future__ import annotations

import sys
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

CODE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CODE_DIR.parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(CODE_DIR))

from predictive_maintenance.data import enrich_sos_with_test_details, load_table  # noqa: E402
from predictive_maintenance.features import RAW_MEASUREMENT_COLUMNS  # noqa: E402
from predictive_maintenance.pipeline import (  # noqa: E402
    MODE_ALERT, MODE_CONDITION, MODE_PREDICTION,
    TIER_CLOSED, P1_IMMEDIATE, P1_WO_TRACKING, P2_REPEATED,
    analyze_frames,
)
from predictive_maintenance.presentation import (  # noqa: E402
    MEASUREMENT_LABELS, build_case_table, build_fleet_summary, build_mode_summary,
    build_quality_issues, latest_measurement_changes, measurement_trend_table,
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Machine Health Dashboard", page_icon="🏭", layout="wide")

STYLES = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.block-container { padding-top: 1.2rem; padding-bottom: 3rem; }
.hcard {
    border-radius:16px; padding:1.3rem 1.5rem; margin-bottom:0.9rem;
    display:flex; flex-direction:column; gap:0.25rem;
}
.hcard.green  { background:linear-gradient(135deg,#d1fae5,#a7f3d0); border-left:6px solid #059669; }
.hcard.yellow { background:linear-gradient(135deg,#fef3c7,#fde68a); border-left:6px solid #d97706; }
.hcard.red    { background:linear-gradient(135deg,#fee2e2,#fca5a5); border-left:6px solid #dc2626; }
.hcard .score { font-size:2.2rem; font-weight:800; line-height:1.05; }
.hcard .lbl   { font-size:0.78rem; font-weight:600; text-transform:uppercase; letter-spacing:.05em; opacity:0.65; }
.hcard .name  { font-size:0.95rem; font-weight:700; }
.hcard .sub   { font-size:0.8rem; opacity:0.72; }
.hcard .fp    { font-size:0.8rem; font-weight:600; margin-top:0.2rem; }
.badge { display:inline-block; padding:.15rem .65rem; border-radius:99px; font-size:.75rem; font-weight:700; letter-spacing:.03em; }
.bp1  { background:#dc2626; color:#fff; }
.bp2  { background:#d97706; color:#fff; }
.bp3  { background:#2563eb; color:#fff; }
.bok  { background:#059669; color:#fff; }
.strip { background:#1e293b; color:#e2e8f0; border-radius:12px; padding:.95rem 1.4rem;
         margin-bottom:1.1rem; display:flex; gap:2rem; align-items:center; flex-wrap:wrap; }
.kv   { display:flex; flex-direction:column; align-items:center; }
.kv .v { font-size:1.6rem; font-weight:800; color:#f8fafc; }
.kv .k { font-size:0.68rem; text-transform:uppercase; letter-spacing:.06em; opacity:0.58; }
.finding { background:#fff8e8; border-left:5px solid #d97706; border-radius:8px;
           padding:.85rem 1rem; margin:.5rem 0 .9rem; font-size:.91rem; }
.finding b { color:#92400e; }
.mstrip { background:#f1f5f9; border:1px solid #cbd5e1; border-radius:10px;
          padding:.65rem .95rem; margin-bottom:.85rem; font-size:.86rem; color:#334155; }
[data-testid="stMetric"] { border:1px solid #e2e8f0; border-radius:10px; padding:.65rem .85rem; }
</style>
"""
st.markdown(STYLES, unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_path(path: str) -> pd.DataFrame:
    return load_table(path)


def bundled(source: str):
    base = PROJECT_ROOT / "data/current"
    hist = sorted(base.glob("SampleHistory*.xlsx")) + sorted(base.glob("SosFluidSample*.xlsx"))
    if not hist:
        raise FileNotFoundError(
            "No S.O.S history file found. Expected SampleHistory*.xlsx or SosFluidSample*.xlsx."
        )
    sos_path = hist[0]
    sos = load_path(str(sos_path))
    test = sorted(base.glob("SampleTestDetails*.xlsx")) + sorted(base.glob("SosTestDetails*.xlsx"))
    if test:
        sos = enrich_sos_with_test_details(sos, load_path(str(test[0])))
    tel = sorted(
        file for file in base.glob("*.xlsx")
        if "telemat" in file.name.lower() and "enum" not in file.name.lower()
    )
    tel_path = tel[0] if tel else None
    wo = sorted(
        file for file in base.iterdir()
        if file.is_file()
        and "work" in file.name.lower()
        and file.suffix.lower() in {".xlsx", ".xls", ".csv"}
    )
    wo_path = wo[0] if wo else None
    return (
        sos,
        load_path(str(tel_path)) if tel_path is not None else None,
        load_path(str(wo_path)) if wo_path is not None else None,
    )


def available_sources() -> list[str]:
    choices: list[str] = []
    base = PROJECT_ROOT / "data/current"
    files = list(base.glob("*.xlsx")) + list(base.glob("*.csv"))
    if any(
        "samplehistory" in f.name.lower() or "sosfluidsample" in f.name.lower()
        for f in files
    ):
        choices.append("Current data")
    choices.append("Upload files")
    return choices


def badge(code: str) -> str:
    if code in {P1_IMMEDIATE, P1_WO_TRACKING}:
        return '<span class="badge bp1">P1 Act Now</span>'
    if code == P2_REPEATED:
        return '<span class="badge bp2">P2 Review</span>'
    if code == TIER_CLOSED:
        return '<span class="badge bok">Closed</span>'
    return '<span class="badge bp3">P3 Monitor</span>'


def card_color(score: int) -> str:
    return "green" if score >= 85 else "yellow" if score >= 60 else "red"


def s(v) -> str:
    return escape(str(v if pd.notna(v) else ""))


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🏭 Machine Health")
    st.divider()
    source = st.radio("Source", available_sources(), label_visibility="collapsed")
    sos_file = sos_test_file = tel_file = wo_file = None
    if source == "Upload files":
        sos_file     = st.file_uploader("SOS Sample History (required)", type=["xlsx","xls","csv"])
        sos_test_file= st.file_uploader("SOS Test Details (optional)",   type=["xlsx","xls","csv"])
        tel_file     = st.file_uploader("Telemetry (optional)",          type=["xlsx","xls","csv"])
        wo_file      = st.file_uploader("Work Orders (optional)",        type=["xlsx","xls","csv"])
    st.divider()
    horizon = st.select_slider("Forecast window (days)", options=[15, 30, 60, 90], value=30)
    run = st.button("▶  Run Analysis", type="primary", use_container_width=True)
    st.divider()
    st.caption("Results are for engineering review only.")


# ── Run analysis ──────────────────────────────────────────────────────────────
sig = (source, int(horizon))
if run or "analysis" not in st.session_state or st.session_state.get("sig") != sig:
    if source == "Upload files" and sos_file is None:
        st.info("Upload the SOS Sample History file then click ▶ Run Analysis.")
        st.stop()
    if source == "Upload files":
        raw = load_table(sos_file)
        if sos_test_file:
            raw = enrich_sos_with_test_details(raw, load_table(sos_test_file))
        frames = (raw,
                  load_table(tel_file) if tel_file else None,
                  load_table(wo_file)  if wo_file  else None)
    else:
        frames = bundled(source)
    with st.status("Running analysis...", expanded=True) as status:
        st.write("Cleaning and standardising records")
        st.write("Merging SOS, telemetry and work orders")
        analysis = analyze_frames(*frames, horizon_days=horizon)
        st.write("Building machine health scores and priority cases")
        status.update(label="Analysis complete", state="complete", expanded=False)
    st.session_state.update(analysis=analysis, sig=sig, _src=source)

result   = st.session_state["analysis"]
mode_info= build_mode_summary(result, horizon)
cases    = build_case_table(result, horizon)
fleet    = build_fleet_summary(result, cases, horizon)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("## Machine Health Dashboard")
mi = "🔮" if mode_info["mode"] == MODE_PREDICTION else ("📈" if mode_info["mode"] == MODE_CONDITION else "🔔")
st.markdown(f'<div class="mstrip">{mi} <b>{mode_info["title"]}</b> — {mode_info["description"]}</div>',
            unsafe_allow_html=True)
# Fleet summary strip
avg_h = fleet.get("avg_fleet_health", 100)
h_col = "#059669" if avg_h >= 85 else ("#d97706" if avg_h >= 60 else "#dc2626")
st.markdown(f"""
<div class="strip">
  <div class="kv"><div class="v" style="color:{h_col};">{avg_h}/100</div><div class="k">Fleet Condition</div></div>
  <div class="kv"><div class="v">{fleet['assets']}</div><div class="k">Machines</div></div>
  <div class="kv"><div class="v" style="color:#dc2626;">{fleet['immediate_action']}</div><div class="k">P1 Act Now</div></div>
  <div class="kv"><div class="v" style="color:#d97706;">{fleet['high']}</div><div class="k">High Severity</div></div>
  <div class="kv"><div class="v">{fleet['monitor']}</div><div class="k">Monitor</div></div>
  <div class="kv"><div class="v" style="color:#059669;">{fleet['normal']}</div><div class="k">Normal</div></div>
  <div class="kv"><div class="v">{fleet['open_corrective_wos']}</div><div class="k">Open WOs</div></div>
  <div class="kv"><div class="v">{fleet['analysis_confidence']}</div><div class="k">Confidence</div></div>
</div>""", unsafe_allow_html=True)
st.info(f"What to do next: {fleet['recommendation']}")

tab1, tab2, tab3 = st.tabs(["Fleet Overview", "Machine Detail", "Data Quality"])

# ── TAB 1 — Fleet Overview ────────────────────────────────────────────────────
with tab1:
    if cases.empty:
        st.info("No cases generated.")
    else:
        active = cases[~cases["priority_code"].eq(TIER_CLOSED)]
        st.markdown("### Machine Health Cards")
        st.caption("Color = health level: Green (good) | Yellow (monitor) | Red (action needed)")
        per_row = 3
        for chunk_start in range(0, len(active), per_row):
            chunk = active.iloc[chunk_start:chunk_start + per_row]
            cols  = st.columns(per_row)
            for col, (_, r) in zip(cols, chunk.iterrows()):
                sc  = int(r.get("health_score", 100))
                cc  = card_color(sc)
                bdg = badge(r.get("priority_code", ""))
                fp  = s(r.get("failure_fingerprint", "Normal Baseline"))
                traj= s(r.get("condition_trajectory", "—"))
                col.markdown(f"""
<div class="hcard {cc}">
  <div class="lbl">Heuristic Condition Index</div>
  <div class="score">{sc}/100</div>
  <div class="name">{s(r['asset_id'])}</div>
  <div class="sub">{s(r['component_name'])}</div>
  <div>{bdg}</div>
  <div class="fp">&#128302; {fp}</div>
  <div class="sub">Trend: {traj}</div>
</div>""", unsafe_allow_html=True)

        st.divider()
        col_c, col_t = st.columns([1, 2])
        with col_c:
            st.markdown("### Priority Breakdown")
            counts = active["priority"].value_counts().rename_axis("Priority").reset_index(name="Cases")
            color_map = {
                "P1 — Inspect now": "#dc2626",
                "P1 — Verify work order": "#ef4444",
                "P2 — Repeated unresolved finding": "#d97706",
                "P3 — Engineering review": "#2563eb",
                "In progress": "#7c3aed",
            }
            fig = px.bar(counts, x="Cases", y="Priority", orientation="h",
                         color="Priority", color_discrete_map=color_map,
                         template="plotly_white")
            fig.update_layout(showlegend=False, margin=dict(l=0,r=0,t=8,b=0), height=250)
            st.plotly_chart(fig, use_container_width=True)
        with col_t:
            st.markdown("### Top Priority Cases")
            top = active.head(8)[["priority","asset_id","component_name",
                                   "health_score","failure_fingerprint",
                                   "condition_trajectory","required_action"]].copy()
            top.columns = ["Priority","Machine","Component","Health %",
                           "Diagnostic Fingerprint","Trend","Action"]
            st.dataframe(top, hide_index=True, use_container_width=True)
        st.download_button("Download full action list",
                           cases.to_csv(index=False).encode(),
                           "action_queue.csv", "text/csv")

# ── TAB 2 — Machine Detail ────────────────────────────────────────────────────
with tab2:
    if cases.empty:
        st.info("No cases available.")
    else:
        sel_asset = st.selectbox("Select Machine", sorted(cases["asset_id"].unique()), key="da")
        ac        = cases[cases["asset_id"].eq(sel_asset)]
        sel_comp  = st.selectbox("Select Component", sorted(ac["component_name"].unique()), key="dc")
        row       = ac[ac["component_name"].eq(sel_comp)].iloc[0]

        sc       = int(row.get("health_score", 100))
        st_label = str(row.get("health_status", "Optimal"))
        col_hex  = str(row.get("health_color", "#16a34a"))
        fp       = str(row.get("failure_fingerprint", "Normal Baseline"))
        pens     = row.get("health_penalties", []) or []
        cc       = card_color(sc)
        pen_text = "  |  ".join([s(p) for p in pens]) if pens else "No active deductions"
        finding_date = pd.to_datetime(row.get("finding_sample_date"), errors="coerce")
        finding_date_text = finding_date.date().isoformat() if pd.notna(finding_date) else "Date unavailable"

        st.markdown(f"""
<div class="hcard {cc}" style="flex-direction:row;align-items:center;gap:2rem;">
  <div>
    <div class="lbl">Heuristic Condition Index</div>
    <div class="score">{sc}/100</div>
    <div style="font-size:.95rem;font-weight:700;">{st_label}</div>
  </div>
  <div style="flex:1;">
    <div><b>Machine:</b> {s(row['asset_id'])} &mdash; {s(row['component_name'])}</div>
    <div><b>Diagnostic Fingerprint:</b> {s(fp)}</div>
    <div style="font-size:.81rem;margin-top:.35rem;opacity:.78;">{pen_text}</div>
  </div>
</div>""", unsafe_allow_html=True)

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Latest Lab",    row.get("latest_lab_status", "—"))
        m2.metric("Priority",      row.get("priority", "—"))
        m3.metric("Trend",         row.get("condition_trajectory", "—"))
        m4.metric("Failure Prob.", row.get("probability_display", "Not available"))
        m5.metric("Confidence",    row.get("data_confidence", "—"))

        st.markdown(
            f'<div class="finding"><b>Active Finding:</b> {s(row["main_issue"])}<br>'
            f'<b>Finding sample:</b> {s(finding_date_text)} — {s(row.get("active_finding_lab_status", ""))}<br>'
            f'<b>Action:</b> {s(row["required_action"])}<br>'
            f'<b>When:</b> {s(row["due"])}</div>',
            unsafe_allow_html=True)

        outlook = str(row.get("next_condition_outlook", ""))
        chk     = str(row.get("next_checkpoint", "Next scheduled sample"))
        pers    = str(row.get("condition_persistence", ""))
        if outlook:
            st.info(f"Forward Outlook: {outlook}\n\nPersistence: {pers} | Checkpoint: {chk}")

        st.markdown("#### Lab Measurements Over Time")
        trend = measurement_trend_table(result.get("sos", pd.DataFrame()), sel_asset, row["component"])
        avail = [x for x in RAW_MEASUREMENT_COLUMNS if x in trend and trend[x].notna().any()]
        if avail:
            meas = st.selectbox("Measurement to plot", avail,
                                format_func=lambda x: MEASUREMENT_LABELS.get(x, x),
                                key="dm")
            fig2 = px.line(trend, x="sample_date", y=meas, markers=True,
                           labels={"sample_date": "Sample Date",
                                   meas: MEASUREMENT_LABELS.get(meas, meas)},
                           template="plotly_white")
            fig2.update_traces(line_color=col_hex, marker=dict(size=8))
            fig2.update_layout(margin=dict(l=0,r=0,t=18,b=0), height=270)
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("No numerical measurements available to plot.")

        changes = latest_measurement_changes(trend, limit=8)
        if not changes.empty:
            st.markdown("#### Latest Measurement Changes")
            sh = changes.rename(columns={"measurement_name":"Measurement","current":"Latest",
                                         "previous":"Previous","change":"Change","unit":"Unit"})
            st.dataframe(sh[["Measurement","Latest","Previous","Change","Unit"]],
                         hide_index=True, use_container_width=True)

        with st.expander("Raw Data (Audit Trail)"):
            st.dataframe(trend, hide_index=True, use_container_width=True)
            wo_df = result.get("work_orders")
            if isinstance(wo_df, pd.DataFrame) and not wo_df.empty:
                a_wo = wo_df[wo_df["asset_id"].astype(str).eq(str(sel_asset))]
                if not a_wo.empty:
                    st.markdown("**Work Orders:**")
                    st.dataframe(a_wo, hide_index=True, use_container_width=True)

# ── TAB 3 — Data Quality ──────────────────────────────────────────────────────
with tab3:
    st.markdown("### Data Quality and Readiness")
    st.caption("These checks show whether the data is complete enough for each analysis type.")
    dm = result.get("dataset_metrics", {})
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("SOS Rows",      f'{len(result.get("sos", [])):,}')
    c2.metric("Valid Dates",   f'{dm.get("valid_date_samples", 0):,}')
    c3.metric("Invalid Dates", f'{dm.get("invalid_date_samples", 0):,}')
    c4.metric("Matched Assets",len(result.get("matched_assets", set())))

    issues = build_quality_issues(result)
    st.markdown("#### Issues Found")
    sh = issues.rename(columns={"severity":"Severity","issue":"Issue",
                                 "count":"Rows affected","impact":"Impact","fix":"How to fix"})
    st.dataframe(sh, hide_index=True, use_container_width=True)

    st.markdown("#### Prediction Readiness Gates")
    st.caption("All gates must pass before failure probability can be shown.")
    st.dataframe(result.get("readiness", pd.DataFrame()), hide_index=True, use_container_width=True)
    st.download_button("Download quality report",
                       issues.to_csv(index=False).encode(),
                       "data_quality_issues.csv", "text/csv")

st.divider()
st.caption("Decision support only. Maintenance action remains the responsibility of qualified reliability personnel.")
