# Validation Report

## Verified scope

The current application was validated against:

- `data/current/SampleHistory_Asset_120-000432.xlsx`
- `data/current/SampleTestDetails_ASSET_120-000432.xlsx`

`Telematics_Enums.xlsx` is a reference dictionary, not machine telemetry. The dashboard now excludes it during automatic discovery, and the ingestion layer rejects it when supplied manually because it has no machine asset/timestamp schema.

## Implemented safeguards

- S.O.S and work-order components use the same canonical aliases before matching.
- Ambiguous normalized spreadsheet headers fail clearly instead of silently choosing the first column.
- Invalid telemetry schemas are ignored and reported as a data-quality issue.
- Failure labels come only from explicit future confirmed corrective work-order outcomes.
- Right-censored samples are excluded rather than labelled as non-failures.
- Prediction requires at least 60 labelled rows, 15 positive outcomes, 15 negative outcomes, five assets and 12 distinct dates.
- Model selection uses chronological train/calibration/test blocks with prediction-horizon embargo.
- Probability output remains blocked unless the held-out utility gate passes.
- A historical active finding and the latest laboratory sample are displayed with separate dates/statuses.
- The 0–100 display is labelled as a heuristic condition index, not a failure probability or remaining-life estimate.
- Repeated AR deductions now count only the consecutive dated AR sequence.
- Sparse laboratory trends compare the latest two non-null values for each individual measurement.
- Work-order event dates are pre-indexed by asset/component to avoid repeated full-table scans.

## Current real-data result

| Check | Result |
|---|---:|
| Operating mode | Alert Management |
| S.O.S rows | 5 |
| Assets | 1 |
| Valid / invalid dates | 5 / 0 |
| Laboratory AR samples | 2 |
| Machine telemetry rows | 0 |
| Work-order outcome rows | 0 |
| Operational cases | 1 |
| Highest case | P2 repeated unresolved finding |
| Condition trajectory | Persistent abnormal |
| Failure probabilities generated | 0 |

Recommended action from the application: review the repeated P2 case within three days and confirm the laboratory-requested resampling interval.

## Verification executed

```text
python -m pytest -q
Result: 31 passed

Current Excel end-to-end analysis
Result: Alert Management; 5 samples; 1 case; no telemetry; no failure probability

Streamlit AppTest
Result: current-data dashboard completed without an application exception

Synthetic independently-labelled pipeline check
Result: Failure Prediction activated and produced a non-empty validation summary
```

## Remaining data limitation

The software workflow is ready, but this five-row extract cannot validate real-world failure prediction. Production probability requires substantially more independently confirmed work-order outcomes, including both failures and non-failures, across multiple machines and dates. Until those gates pass, the application correctly provides deterministic laboratory triage and condition outlook only.
