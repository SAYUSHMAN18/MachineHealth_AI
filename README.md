# Heavy Machinery Condition Monitoring

This project analyses S.O.S fluid records, optional telemetry and independently confirmed work-order outcomes in three guarded operating modes. Version 3 presents the result as a machine-level engineering decision workflow instead of a wall of sample-level alerts.

## Simple project layout

```text
MachineHealth_AI-main/
├── code/       # Dashboard and all active Python model code
├── data/       # Current input files
├── outputs/    # Reports, predictions and trained model
├── tests/      # Automated safety/model tests
├── config/     # Engineering rule configuration
├── docs/       # Data request and validation documents
└── scripts/    # Windows setup and dashboard launch scripts
```

The active application is `code/app.py`; all calculation and workflow logic is inside `code/predictive_maintenance/`.

## Start here

On Windows or in the Antigravity terminal:

```powershell
cd "C:\Users\ersay\Downloads\MachineHealth_AI-main"
.\scripts\setup_and_run.bat
```

Open `http://localhost:8501`, select a data source, and choose **Run analysis**. Use:

- **Upload your own files** to analyse new exports without changing the source code.
- **Current data** appears only when a supported S.O.S history file exists under `data/current`.

For TMS-style exports:

- Upload `SampleHistory_*.xlsx` as **S.O.S sample history file (required)**.
- Upload `SampleTestDetails_*.xlsx` as **S.O.S test-result detail file (optional)**. Long-format measurements are deduplicated, pivoted and joined by `sampleNum`.
- Do not upload `SampleDetails_*.xlsx` when its latest sample already exists in Sample History.
- Do not upload `Telematics_Enums.xlsx` as telemetry; it is a reference dictionary, not time-series machine data.
- Leave Telemetry and Work-order blank until actual records for those sources are available.

For a no-upload local run, store the exports under `data/current` using their normal
`SampleHistory_*.xlsx` and `SampleTestDetails_*.xlsx` names. Enum/reference files are
never auto-selected as telemetry.

The dashboard has three task-based tabs: Fleet Overview, Machine Detail and Data Quality.

## Current input state

The current folder contains the supplied S.O.S history and long-format test details. The dashboard opens with **Current data** selected. `Telematics_Enums.xlsx` is retained only as a reference dictionary and is not telemetry. No machine time-series telemetry or confirmed work-order outcome file is currently bundled. A linked `WorkOrderId` inside an S.O.S export does not by itself prove inspection, repair, failure confirmation or closure.

## Operating modes

1. **Alert Management** — prioritises laboratory records and maintenance follow-up. No ML probability.
2. **Condition Monitoring** — requires at least one asset/component series with three dated numerical laboratory samples.
3. **Failure Prediction** — requires usable S.O.S predictors, at least 60 independently labelled samples, at least 15 positive and 15 negative outcomes across five machines, at least 12 distinct dates, explicit confirmed WO outcomes, and successful leak-resistant chronological validation. Telemetry is optional enrichment.

Even before labelled failure prediction is available, the dashboard now derives a separate **forward-looking condition outlook** from dated AR/NAR transitions, repeated abnormal findings, resampling timing and numerical-result coverage. This predicts whether the next laboratory condition is likely to remain abnormal; it is explicitly not presented as a machine-failure probability.

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
$env:PYTHONDONTWRITEBYTECODE="1"
python -m pytest -q
python -m streamlit run code\app.py
```

The workflow is fully local and deterministic. It does not call an external AI service.

## Command-line analysis

```powershell
$env:PYTHONPATH="$PWD\code"
python -m predictive_maintenance.cli analyze `
  --sos data\current\SampleHistory_Asset_120-000432.xlsx `
  --test-details data\current\SampleTestDetails_ASSET_120-000432.xlsx `
  --output outputs\current
```

Replace the example paths with the new data files before using the CLI.

## Important outputs

- `analysis_summary.json` — data coverage and operational counts
- `model_readiness.csv` — explicit PASS/BLOCKED gates
- `sos_engineering_alerts.csv` — evidence and workflow priorities
- `scoring_features.csv` — past-only, availability-aware features for every dated S.O.S sample
- `sample_predictions.csv` — sample decisions, probabilities (when enabled) and the sources used for each prediction
- `failure_model.joblib` — the selected calibrated model, written only when prediction mode passes
- `maintenance_summary.md` — deterministic maintenance triage summary
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
- No external LLM processing exists in the runtime workflow.

## Validation

The included suite covers the current workbook schema, strict date parsing, telemetry-schema rejection, independent priority/data-quality logic, sparse measurement trends, label leakage prevention, readiness gates, chronological calibration, UTF-8 CLI output and a Streamlit smoke test.

```powershell
python -m pytest -q
```

The exact passing count is printed by pytest and should have zero failures.

Synthetic or example work orders are schema demonstrations only and must not be mixed into real fleet analysis.
