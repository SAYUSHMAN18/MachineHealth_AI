# Final Tested Antigravity Prompt

Copy everything inside the block below into Antigravity when reviewing or rebuilding this project.

```text
Act as a senior reliability engineer, data scientist and Python/Streamlit maintainer. Work directly in this Heavy_Machinery_Predictive_Maintenance project. Preserve the existing folder structure and real uploaded files. Do not claim success until the automated tests, CLI run and Streamlit smoke test pass.

BUSINESS GOAL
Build an engineering-safe system with three strictly separated modes:

1. Alert Management
   - Available from an S.O.S file alone.
   - Prioritises existing laboratory records and maintenance follow-up.
   - Never displays failure probabilities.

2. Condition Monitoring
   - Requires at least one asset/component series containing at least three valid dated numerical laboratory samples.
   - Shows Fe, Cu, Si, water, fuel dilution, viscosity and related trends when present.
   - Does not claim failure probability.

3. Failure Prediction
   - Requires all prediction-readiness gates to pass.
   - Requires both normal and abnormal S.O.S history, at least 60 independently labelled rows, explicit confirmed work-order outcomes containing positive and negative classes, at least five matched telemetry assets covering at least 30 S.O.S rows, sufficient valid dates, and successful chronological training/calibration/test partitions.

CURRENT REAL-DATA FACTS
The supplied S.O.S workbook must produce exactly:

- 520 assets
- 1,051 laboratory AR samples
- 10 HighPriority=T samples
- 929 New laboratory records
- 122 Closed laboratory records
- 648 records with WorkOrderId
- 403 records without WorkOrderId
- 364 New records without WorkOrderId
- 499 valid calendar dates
- 552 invalid/time-only dates
- 3 P1 Immediate Review records
- 6 P1 WO Tracking records
- 93 P2 Multiple Unlinked records after higher-priority overrides
- 268 P3 Action Required records
- zero matched S.O.S/telemetry assets
- zero structured numerical laboratory samples
- zero predicted failure probabilities

INTERPRETATION RULES
- OverallInterp=AR means Laboratory Action Required. It is not a predicted failure.
- HighPriority=T is a laboratory workflow flag and must be shown separately.
- WorkOrderId means only that an identifier is linked. It does not prove inspection, repair, failure confirmation or closure.
- SampleStatusNew=Closed means the laboratory record is closed. It does not prove maintenance completion.
- Rule-match strength is evidence metadata, not probability.
- Never display an arbitrary 0.85, 0.95 or fused rule score as a failure probability.
- Never estimate Remaining Useful Life or “safe days” from a classification score.

PRIORITY LOGIC
Priority and data quality must be independent:

- Closed laboratory record -> Closed, with a warning that repair completion is not confirmed.
- HighPriority=T + New + no WorkOrderId -> P1 – Immediate Review.
- HighPriority=T + New + WorkOrderId -> P1 – WO Tracking; verify WO status and outcome.
- Multiple New AR records for the same asset/component where the current record has no WorkOrderId -> P2 – Multiple Unlinked Alerts.
- New AR + WorkOrderId -> In Progress, but linkage is not completion.
- New AR + no WorkOrderId -> P3 – Action Required; verify whether a WO exists before creating one.
- Invalid or time-only dates must add a separate data-quality warning and must never suppress P1/P2/P3 priority.
- Do not call records “consecutive” unless valid dates prove their order.

DATA AND LABEL SAFETY
- Preserve sample_date_raw and create a separate parsed sample_date.
- Time-only values such as 00:00:00 must become an invalid-date flag, never today’s date.
- Never manufacture targets from OverallInterp, severity, Fe, Cu or other model inputs.
- Never flip a label to force two target classes.
- Training labels may come only from future independently confirmed corrective work-order outcomes within the selected horizon.
- Exclude right-censored samples when the work-order extract does not cover the complete future prediction horizon; never label them as negative.
- Prefer an explicit ObservationEndDate/DataThroughDate from the WO export; otherwise use the latest explicit WO date as a conservative observation boundary.
- Example or synthetic WO files must never be automatically mixed into real current-data analysis.

MODEL REQUIREMENTS
- Use chronological train, calibration and untouched test blocks.
- Each block must contain positive and negative outcomes; otherwise block model training with a clear reason.
- Sigmoid-calibrate candidate classifiers before evaluating probability quality.
- Report average precision, precision, recall, F1, Brier score, ROC-AUC when defined, confusion matrix, test size and positive rate.
- Model-estimated probability must come only from the trained calibrated classifier.
- Keep FMECA rules and telemetry anomaly scores as separate evidence; do not blend them into a fake probability.
- If model training fails after readiness checks, disable prediction and expose the failure reason.

TEXT EVIDENCE
- Extract engineering categories from InterpText using auditable rules.
- Support multiple categories per sample.
- Retain the exact sentence supporting each category.
- Suggested actions must be framed as “Suggested engineering review,” use OEM procedures and require qualified engineering judgement for operating restrictions.
- Do not confirm a diagnosis from interpretation text alone.

DASHBOARD REQUIREMENTS
- Show the current operating mode prominently.
- Show accurate dataset and workflow metrics.
- Include Executive Overview, Immediate Action Queue, Asset History, Laboratory Trends, Multiple Alerts, Work-Order Response, Data Quality and Prediction Readiness tabs.
- Hide prediction horizon and probability output outside Failure Prediction mode.
- In Alert Management mode show “Predicted Failure Risk: Not Available.”
- Show numerical trend charts only when structured measurements and usable dates exist; require at least three observations before interpreting a trend.
- Escape all spreadsheet-derived values before inserting them into unsafe HTML.
- Keep the deterministic maintenance summary local by default.
- External Gemini processing must require explicit opt-in, disclose that data leaves the environment, and exclude asset identifiers from the external prompt.

EFFICIENCY
- Do not train ML, run telemetry anomaly detection or call an external LLM when readiness gates fail.
- Cache file loading and reuse computed results across dashboard filters.
- Keep optional Gemini dependencies outside the base requirements file.
- Do not package .venv, .pytest_cache, __pycache__, generated models or previous outputs in the final ZIP.

MANDATORY VALIDATION
Run all of the following and fix every failure:

1. python -m pytest -q
   Expected: 15 passed.

2. python -m compileall -q app.py src tests
   Expected: no errors.

3. python -m predictive_maintenance.cli analyze --sos data/current/SosFluidSample.xlsx --telemetry data/current/TelematicDataSample.xlsx --output outputs/current
   Expected mode: Alert Management; 1,051 samples; 520 assets; 259 cleaned telemetry snapshots; zero matched assets; UTF-8 outputs written successfully.

4. Run the Streamlit AppTest or start Streamlit headlessly and verify there are no application exceptions. Confirm the visible metrics include AR Samples=1,051, HighPriority=T=10, P1 No WO=3, P1 WO Linked=6, P2 Multiple Unlinked=93 and Invalid Dates=552.

5. Confirm failure_probability_pct is null for every current-data record, trained_model is None, and all prediction-readiness gates except valid chronological-date count remain BLOCKED.

Do not change the expected counts to make tests pass. Correct the implementation. At the end, report the files changed, exact test commands, exact results and any remaining data limitations. Do not describe the current workbook as a failure-prediction dataset.
```
