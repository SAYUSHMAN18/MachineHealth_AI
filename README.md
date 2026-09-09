# Heavy Machinery Condition Monitoring

This project analyses S.O.S fluid records, optional telemetry and independently confirmed work-order outcomes in three guarded operating modes. Version 3 presents the result as a machine-level engineering decision workflow instead of a wall of sample-level alerts.

## Simple project layout

```text
MachineHealth_AI-main/
├── code/       # Dashboard and all active Python model code
├── data/       # Current and demo input files
├── outputs/    # Reports, predictions and trained model
├── tests/      # Automated safety/model tests
├── config/     # Engineering rule configuration
├── docs/       # Data request and validation documents
├── scripts/    # Windows/Linux launch scripts
└── archive/    # Preserved legacy phase-based implementation
```

The active application is only `code/app.py` plus `code/predictive_maintenance/`. The archived pipeline is retained for reference and is not imported by the dashboard.

## Start here

On Windows or in the Antigravity terminal:

```powershell
cd "C:\Users\ersay\Downloads\MachineHealth_AI-main"
.\scripts\setup_and_run.bat
```

Open `http://localhost:8501`, select a data source, and choose **Run analysis**. Use:

- **Supplied current data** to analyse the provided alert-only export.
- **Matched prediction demo** to test S.O.S + telemetry + detailed WO prediction end to end. A warning identifies it as synthetic.
- **Upload your own files** to analyse your exports without changing the source code.

The dashboard has five views: Fleet Overview, Action Queue, Asset Analysis, Model Validation and Data Quality.

## What the current data can do

`data/current/SosFluidSample.xlsx` is an alert-only export. All 1,051 rows have `OverallInterp=AR`, so the application runs in **Alert Management** mode and does not calculate failure probabilities.

Verified current-data results:

- 520 assets and 1,051 laboratory AR records
- 10 `HighPriority=T` records
- 929 New and 122 Closed laboratory records
- 648 records with a linked `WorkOrderId`; 403 without one
- 364 New records without a WO link
- 3 P1 Immediate Review records and 6 P1 WO Tracking records
- 93 P2 Multiple Unlinked records after higher-priority overrides
- 499 valid sample dates and 552 time-only/invalid dates
- 952 telemetry rows reduced to 259 snapshots, but no S.O.S/telemetry asset IDs match
- No structured Fe, Cu, Si, water, viscosity or other numerical laboratory columns

`WorkOrderId` linkage does not prove inspection, repair, failure confirmation or closure.

## Operating modes

1. **Alert Management** — prioritises laboratory records and maintenance follow-up. No ML probability.
2. **Condition Monitoring** — requires at least one asset/component series with three dated numerical laboratory samples.
3. **Failure Prediction** — requires usable S.O.S predictors (numerical trends and/or varied interpretation text), at least 60 independently labelled samples, explicit confirmed WO outcomes with both classes, and successful leak-resistant chronological validation. Telemetry and per-machine work-order history are optional enrichments, not admission requirements.

The model never creates labels from S.O.S severity and never flips labels to force two classes.

### Mixed data availability

The prediction table is built from every valid S.O.S sample. Each optional source has an explicit availability feature:

- **S.O.S only**: the model uses current measurements, past laboratory trends, equipment/fluid hours, machine model and component.
- **Noise-resistant trend**: each numerical analyte also gets a transparent three-sample rolling mean.
- **Text-heavy S.O.S**: compact engineering indicators are extracted from interpretation text (wear, contamination, coolant, fuel, degradation, urgency and resampling); `AR` remains an input flag, never the outcome label.
- **S.O.S + work-order history**: past confirmed corrective-event counts are added. Future work orders are used only to create training outcomes.
- **S.O.S + telemetry**: recent 7/30/90-day operating hours, distance, utilisation and telemetry freshness are added.
- **All three**: all available feature groups are used. Missing telemetry is imputed inside the model and marked by `telemetry_available=0`.

Machines absent from the work-order extract are still eligible for scoring after a fleet model has been trained, but they are excluded from supervised training so that “no record” is never mistaken for “no failure.”

## Run on Windows

Open PowerShell or the Antigravity terminal:

```powershell
cd "C:\Users\ersay\Downloads\MachineHealth_AI-main"
.\scripts\setup_and_run.bat
```

The setup script creates `.venv`, installs dependencies, runs the tests and starts Streamlit. Open `http://localhost:8501` if the browser does not open automatically.

Later launches:

```powershell
.\scripts\run_dashboard.bat
```

Stop the server with `Ctrl+C`.

## Manual setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
$env:PYTHONPATH="$PWD\code"
python -m pytest -q
python -m streamlit run code\app.py
```

The deterministic maintenance summary is local by default. Optional Gemini support requires:

```powershell
python -m pip install -r requirements-ai.txt
```

External processing must then be explicitly enabled in the dashboard. Asset identifiers are excluded from the external prompt.

## Command-line analysis

```powershell
$env:PYTHONPATH="$PWD\code"
python -m predictive_maintenance.cli analyze `
  --sos data\current\SosFluidSample.xlsx `
  --telemetry data\current\TelematicDataSample.xlsx `
  --output outputs\current
```

Expected CLI mode: `Alert Management` with zero matched assets.

## Important outputs

- `analysis_summary.json` — data coverage and operational counts
- `model_readiness.csv` — explicit PASS/BLOCKED gates
- `sos_engineering_alerts.csv` — evidence and workflow priorities
- `scoring_features.csv` — past-only, availability-aware features for every dated S.O.S sample
- `sample_predictions.csv` — sample decisions, probabilities (when enabled) and the sources used for each prediction
- `failure_model.joblib` — the selected calibrated model, written only when prediction mode passes
- `ai_insights.md` — deterministic maintenance triage summary
- `telemetry_cleaned_scored.csv` — cleaned telemetry; anomaly scoring remains disabled unless prediction readiness passes

## Methodological protections

- `AR` is never treated as a predicted failure.
- Rule-match strength is never shown as a probability.
- Work-order labels come only from future confirmed corrective outcomes.
- Samples without a fully observable future horizon are excluded instead of being labelled negative.
- Provide `ObservationEndDate` (or `DataThroughDate`) in the detailed WO export when available; it is applied per machine. Otherwise the latest explicit WO date is used conservatively as the extract boundary.
- A classifier never invents remaining safe days or remaining useful life.
- Model probability is not blended with rule or anomaly scores.
- Candidate selection uses a chronological validation/calibration block; only the winner is evaluated on the untouched final test block.
- Predictor eligibility and telemetry coverage are decided from the training block only, so later data availability cannot influence model design.
- Machines are equally weighted during fitting, preventing one frequently sampled machine from dominating the fleet model.
- Failure probability is enabled only when the selected model beats a constant-prevalence baseline on the untouched newer test block (average-precision lift ≥ 0.02, positive Brier skill and non-zero recall).
- The event/no-event decision threshold is selected on validation data for the best F1/recall balance; the test block remains untouched.
- Equal timestamps never cross partitions, and earlier blocks are purged by the prediction horizon to prevent label-window overlap.
- Telemetry is optional. Availability flags and imputation let one fleet model accept different source combinations without inventing sensor readings.
- Extremely sparse labelled telemetry coverage is automatically left out of the classifier so it cannot become a proxy for one machine; it remains visible as operational evidence.
- The Random Forest is deliberately bounded (160 trees, depth 12) to reduce training time and overfitting.
- Invalid dates are flagged separately and never suppress P1 action priority.
- Closed laboratory status does not claim that maintenance was completed.
- External LLM processing is off by default.

## Validation

The included suite covers real-workbook metrics, strict date parsing, independent priority/data-quality logic, multi-label evidence extraction, label leakage prevention, readiness gates, chronological model calibration, UTF-8 CLI output and a Streamlit dashboard smoke test.

```powershell
python -m pytest -q
```

Expected result: `23 passed`.

Synthetic or example work orders are schema demonstrations only and must not be mixed into real fleet analysis.
