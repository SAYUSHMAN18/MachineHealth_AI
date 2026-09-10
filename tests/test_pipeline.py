from pathlib import Path
import numpy as np
import pandas as pd

from predictive_maintenance.data import (
    enrich_sos_with_test_details,
    load_table,
    prepare_sos,
    prepare_telemetry,
    prepare_work_orders,
)
from predictive_maintenance.features import build_scoring_table, build_training_table
from predictive_maintenance.reporting import generate_maintenance_summary
from predictive_maintenance.models import _heldout_utility_metrics, train_failure_models
from predictive_maintenance.pipeline import (
    MODE_ALERT, MODE_CONDITION, MODE_PREDICTION,
    P1_IMMEDIATE, P1_WO_TRACKING, P2_REPEATED, P3_ACTION,
    TIER_IN_PROGRESS, TIER_CLOSED,
    analyze_frames, run_analysis,
)
from predictive_maintenance.presentation import (
    build_case_table, build_condition_outlook, build_fleet_summary, build_mode_summary,
    build_quality_issues, build_validation_summary, compute_machine_health_score,
    latest_measurement_changes, measurement_trend_table,
)

ROOT = Path(__file__).resolve().parents[1]


def _current_sos_raw() -> pd.DataFrame:
    base = ROOT / "data/current"
    history = next(iter(sorted(base.glob("SampleHistory*.xlsx"))), None)
    if history is None:
        history = next(iter(sorted(base.glob("SosFluidSample*.xlsx"))), None)
    assert history is not None, "Current S.O.S history fixture is missing"
    raw = load_table(history)
    details = next(iter(sorted(base.glob("SampleTestDetails*.xlsx"))), None)
    if details is not None:
        raw = enrich_sos_with_test_details(raw, load_table(details))
    return raw


def _synthetic_prediction_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    samples: list[dict] = []
    work_orders: list[dict] = []
    dates = pd.date_range("2023-01-01", periods=18, freq="40D")
    observation_end = dates[-1] + pd.Timedelta(days=60)
    for period, sample_date in enumerate(dates):
        for asset_number in range(10):
            positive = (period + asset_number) % 2 == 0
            asset_id = f"EQ-{asset_number:02d}"
            sample_id = f"S-{period:02d}-{asset_number:02d}"
            samples.append({
                "SampleNum": sample_id,
                "EquipNum": asset_id,
                "Compartment": "ENGINE",
                "OverallInterp": "AR" if positive else "NAR",
                "InterpText": "Elevated iron wear" if positive else "Normal condition",
                "DateSampled": sample_date,
                "Fe": 220.0 if positive else 20.0,
                "Cu": 25.0 if positive else 3.0,
            })
            work_orders.append({
                "WorkOrderId": f"WO-{period:02d}-{asset_number:02d}",
                "EquipmentId": asset_id,
                "Component": "ENG",
                "OpenedDate": sample_date + pd.Timedelta(days=5),
                "ObservationEndDate": observation_end,
                "WorkOrderType": "CORRECTIVE" if positive else "PREVENTIVE",
                "FailureConfirmed": int(positive),
            })
    return pd.DataFrame(samples), pd.DataFrame(work_orders)


def test_sos_sample_automatically_selects_alert_management_mode():
    result = analyze_frames(_current_sos_raw())
    assert result["operating_mode"] == MODE_ALERT
    assert result["mode"] == MODE_ALERT
    assert not result["predictive_risk_enabled"]
    assert len(result["sos"]) > 0
    assert len(result["alerts"]) == len(result["sos"])
    assert result["matched_assets"] == set()


def test_ar_alone_never_produces_failure_probability():
    ar_sos = pd.DataFrame({
        "SampleNum": ["S999"],
        "EquipNum": ["EQ-100"],
        "Compartment": ["ENGINE"],
        "OverallInterp": ["AR"],
        "InterpText": ["Laboratory action required for oil sample"],
        "DateSampled": ["2026-01-15"],
        "Fe": [15.0],
        "Cu": [5.0],
    })
    result = analyze_frames(ar_sos, telemetry_raw=None, work_orders_raw=None)
    alerts = result["alerts"]
    assert len(alerts) == 1
    assert alerts.iloc[0]["lab_status"] == "Laboratory Action Required"
    assert alerts.iloc[0]["rule_evidence_strength"] <= 0.40
    assert alerts.iloc[0]["evidence_level"] != "Strong rule match"

    cards = result["maintenance_action_cards"]
    assert (cards["failure_probability_pct"].isna()).all(), "AR alone must never produce a failure probability"
    assert "Not Available" in cards.iloc[0]["predicted_risk_status"]
    assert result["trained_model"] is None, "ML model must not run when operating mode is Alert Management"


def test_nullable_blank_cells_from_fuzzy_upload_do_not_crash():
    fuzzy_sos = pd.DataFrame({
        "SampleNum": ["S-NA"],
        "EquipNum": ["EQ-NA"],
        "Compartment": ["ENGINE"],
        "OverallInterp": [pd.NA],
        "InterpText": [pd.NA],
        "HighPriority": [pd.NA],
        "SampleStatusNew": [pd.NA],
        "DateSampled": ["2026-08-01"],
        "Fe": [pd.NA],
    })
    result = analyze_frames(fuzzy_sos)
    assert result["operating_mode"] == MODE_ALERT
    assert result["alerts"].iloc[0]["lab_status"] == "Unspecified"
    assert not result["predictive_risk_enabled"]
    assert pd.isna(
        result["maintenance_action_cards"].iloc[0]["failure_probability_pct"]
    )


def test_tms_camelcase_history_and_long_test_details_are_supported():
    history = pd.DataFrame({
        "sampleNum": [101, 102],
        "equipNum": ["120-000432 / 2082"] * 2,
        "serialNum": ["6AB01170"] * 2,
        "eqpModel": ["621E_CAT"] * 2,
        "compartment": ["FD_FR_LT"] * 2,
        "overallInterp": ["NAR", "AR"],
        "interpText": [
            "NO PROBLEMS PRESENTLY ASSOCIATED WITH THIS SAMPLE.",
            "EXCESSIVE WATER PRESENT IN SAMPLE. SAMPLE IN 50 HOURS.",
        ],
        "dateSampled": ["2025-01-01", "2026-01-01"],
        "cMeter": [1000, 1500],
        "cMeterFluid": [500, 1000],
        "sampleStatusNew": ["New", "New"],
    })
    details = pd.DataFrame({
        "sampleNum": [101, 101, 102, 102],
        "resultName": ["Fe", "Fe", "WATER", "V100"],
        "resultFormattedEntry": [10, 20, 1000, 15.2],
        "resultUnits": ["PPM", "PPM", "PPM", "CST"],
        "resultNumber": [9001, 9001, 9002, 9003],
        "SynapseModifiedDateTime": [
            "2025-01-02 10:00:00",
            "2025-01-02 11:00:00",
            "2026-01-02 10:00:00",
            "2026-01-02 10:00:00",
        ],
    })
    enriched = enrich_sos_with_test_details(history, details)
    prepared = prepare_sos(enriched)
    assert prepared["asset_id"].eq("120-000432 / 2082").all()
    assert prepared.loc[prepared["sample_number"].eq("101"), "iron_ppm"].iloc[0] == 20
    assert prepared.loc[prepared["sample_number"].eq("102"), "water_pct"].iloc[0] == 0.1
    assert prepared.loc[prepared["sample_number"].eq("102"), "viscosity_cst"].iloc[0] == 15.2

    result = analyze_frames(enriched)
    assert len(result["sos"]) == 2
    assert result["alerts"].iloc[0]["lab_status"] == "Normal"
    latest = result["alerts"].sort_values("sample_date").iloc[-1]
    assert "water contamination" in latest["evidence"].lower()


def test_unperformed_zero_results_do_not_become_fake_measurements():
    history = pd.DataFrame({
        "sampleNum": [201], "equipNum": ["EQ-1"], "compartment": ["ENG"],
        "overallInterp": ["AR"], "interpText": ["EXCESSIVE WATER PREVENTS MOST ANALYSIS."],
        "dateSampled": ["2026-01-01"],
    })
    details = pd.DataFrame({
        "sampleNum": [201, 201], "resultName": ["WATER", "V100"],
        "resultFormattedEntry": [0, 0], "resultUnits": ["PPM", "CST"],
        "resultNumber": [1, 2],
        "interpText": ["EXCESSIVE WATER PREVENTS MOST ANALYSIS."] * 2,
    })
    prepared = prepare_sos(enrich_sos_with_test_details(history, details))
    assert pd.isna(prepared.iloc[0]["water_pct"])
    assert pd.isna(prepared.iloc[0]["viscosity_cst"])


def test_incompatible_lab_units_are_not_mixed_into_trends():
    history = pd.DataFrame({
        "sampleNum": [301, 302],
        "equipNum": ["EQ-1", "EQ-1"],
        "compartment": ["ENG", "ENG"],
        "overallInterp": ["NAR", "AR"],
        "dateSampled": ["2025-01-01", "2025-02-01"],
    })
    details = pd.DataFrame({
        "sampleNum": [301, 302],
        "resultName": ["FE", "FE"],
        "resultFormattedEntry": [10.0, 99.0],
        "resultUnits": ["PPM", "%"],
        "resultNumber": [1, 2],
    })
    prepared = prepare_sos(enrich_sos_with_test_details(history, details))
    assert prepared.loc[prepared["sample_number"].eq("301"), "iron_ppm"].iloc[0] == 10.0
    assert pd.isna(
        prepared.loc[prepared["sample_number"].eq("302"), "iron_ppm"].iloc[0]
    )


def test_condition_outlook_is_forward_looking_but_not_failure_probability():
    sos = pd.DataFrame({
        "SampleNum": ["S1", "S2", "S3", "S4", "S5"],
        "EquipNum": ["EQ-1"] * 5,
        "Compartment": ["ENG"] * 5,
        "OverallInterp": ["NAR", "NAR", "NAR", "AR", "AR"],
        "InterpText": ["Normal"] * 3 + ["Inspect and resample"] + ["Water entry; sample in 50 hours"],
        "DateSampled": ["2023-01-01", "2024-01-01", "2024-06-01", "2025-08-01", "2026-08-01"],
    })
    result = analyze_frames(sos)
    outlook = build_condition_outlook(result, "EQ-1", "ENG")
    assert outlook["trajectory"] == "Persistent abnormal"
    assert outlook["persistence"].startswith("2 consecutive action-required")
    assert outlook["checkpoint"] == "Resample in 50 operating hours"
    assert outlook["confidence"] == "Low"
    assert result["maintenance_action_cards"]["failure_probability_pct"].isna().all()


def test_6_tier_priority_classification():
    mock_sos = pd.DataFrame({
        "SampleNum": ["S01", "S02", "S03", "S04", "S05", "S06", "S07"],
        "EquipNum":  ["EQ-1", "EQ-2", "EQ-3", "EQ-3", "EQ-4", "EQ-5", "EQ-6"],
        "Compartment": ["ENG", "HYD", "TRANS", "TRANS", "DIFF", "ENG", "HYD"],
        "OverallInterp": ["AR", "AR", "AR", "AR", "AR", "A", "AR"],
        "HighPriority":  ["T",   "T",    "F",    "F",    "F",   "F",   "F"],
        "Status":        ["New", "New",  "New",  "New",  "New", "Closed", "New"],
        "WorkOrderId":   [None,  "WO-10", None,  None,   "WO-20", None,  None],
        "DateSampled":   ["2026-01-10", "2026-01-11", "2026-01-12", "2026-01-13", "2026-01-14", "2026-01-15", "invalid-date"],
    })
    result = analyze_frames(mock_sos)
    alerts = result["alerts"].set_index("sample_number")

    # S01: HighPriority=T, Status=New, no WO -> P1 – Immediate Review
    assert alerts.loc["S01", "priority_tier"] == P1_IMMEDIATE
    assert "HighPriority flag is set" in alerts.loc["S01", "priority_reason"]

    # S02: HighPriority=T, Status=New, has WO -> P1 – WO Tracking
    assert alerts.loc["S02", "priority_tier"] == P1_WO_TRACKING
    assert "Verify its status" in alerts.loc["S02", "priority_reason"]

    # S03 & S04: multiple New AR records for (EQ-3, TRANS) without WO -> P2
    assert alerts.loc["S04", "priority_tier"] == P2_REPEATED
    assert "2 New AR records" in alerts.loc["S04", "priority_reason"]

    # S05: New AR with WorkOrderId -> In Progress
    assert alerts.loc["S05", "priority_tier"] == TIER_IN_PROGRESS

    # S06: Status=Closed -> Closed
    assert alerts.loc["S06", "priority_tier"] == TIER_CLOSED

    # S07: Date quality is separate and does not erase the action priority
    assert alerts.loc["S07", "priority_tier"] == P3_ACTION
    assert alerts.loc["S07", "data_quality_status"] == "Review required"
    assert "Unparseable" in alerts.loc["S07", "data_quality_issue"]


def test_dataset_metrics_are_internally_consistent_for_current_data():
    sos = _current_sos_raw()
    result = analyze_frames(sos, telemetry_raw=None, work_orders_raw=None)
    m = result["dataset_metrics"]

    assert m["unique_assets"] == result["sos"]["asset_id"].nunique(dropna=True)
    assert m["valid_date_samples"] + m["invalid_date_samples"] == len(result["sos"])
    assert m["lab_ar_samples"] == result["sos"]["interpretation_code"].astype(str).str.upper().eq("AR").sum()
    assert sum(
        m[key] for key in [
            "p1_immediate_records", "p1_wo_tracking_records",
            "p2_multiple_unlinked_records", "p3_action_records",
        ]
    ) <= len(result["alerts"])


def test_ml_not_run_in_alert_management_mode():
    sos = _current_sos_raw()
    result = analyze_frames(sos, telemetry_raw=None, work_orders_raw=None)
    assert result["operating_mode"] == MODE_ALERT
    assert result["trained_model"] is None
    assert (result["maintenance_action_cards"]["failure_probability_pct"].isna()).all()


def test_interp_categories_retain_exact_evidence_sentence():
    mock_sos = pd.DataFrame({
        "SampleNum": ["S01"],
        "EquipNum": ["EQ-1"],
        "Compartment": ["ENGINE"],
        "OverallInterp": ["AR"],
        "InterpText": ["High iron wear detected in sample. Inspect bearing and resample within 10 days."],
        "DateSampled": ["2026-01-10"],
    })
    result = analyze_frames(mock_sos)
    cats = result["interp_categories"]
    assert not cats.empty
    assert cats.iloc[0]["problem_category"] == "Wear Metal – Iron"
    assert cats.iloc[0]["category_evidence"] == "High iron wear detected in sample."
    resample = cats[cats["problem_category"] == "Resample / History"]
    assert len(resample) == 1
    assert resample.iloc[0]["category_evidence"] == "Inspect bearing and resample within 10 days."


def test_local_maintenance_summary_contains_mode_header():
    sos = _current_sos_raw()
    result = analyze_frames(sos)
    insights = generate_maintenance_summary(result)
    assert "Maintenance Triage Summary" in insights
    assert "Alert Management" in insights
    assert "OverallInterp=AR" in insights


def test_time_only_date_is_preserved_and_flagged():
    prepared = prepare_sos(pd.DataFrame({
        "SampleNum": ["S01"],
        "EquipNum": ["EQ-1"],
        "Compartment": ["ENG"],
        "OverallInterp": ["AR"],
        "DateSampled": ["00:00:00"],
    }))
    assert pd.isna(prepared.iloc[0]["sample_date"])
    assert prepared.iloc[0]["sample_date_raw"] == "00:00:00"
    assert prepared.iloc[0]["date_quality_issue"] == "Time-only value; calendar date is missing"


def test_invalid_date_never_suppresses_high_priority_action():
    result = analyze_frames(pd.DataFrame({
        "SampleNum": ["S01"],
        "EquipNum": ["EQ-1"],
        "Compartment": ["ENG"],
        "OverallInterp": ["AR"],
        "HighPriority": ["T"],
        "SampleStatusNew": ["New"],
        "WorkOrderId": [None],
        "DateSampled": ["00:00:00"],
    }))
    alert = result["alerts"].iloc[0]
    assert alert["priority_tier"] == P1_IMMEDIATE
    assert alert["data_quality_status"] == "Review required"


def test_training_labels_are_never_manufactured_from_sos_severity():
    sos = prepare_sos(pd.DataFrame({
        "SampleNum": ["S01", "S02", "S03"],
        "EquipNum": ["EQ-1", "EQ-1", "EQ-1"],
        "Compartment": ["ENG", "ENG", "ENG"],
        "OverallInterp": ["AR", "AR", "AR"],
        "DateSampled": ["2026-01-01", "2026-01-02", "2026-01-03"],
        "Fe": [10, 20, 30],
    }))
    telemetry, _ = prepare_telemetry(None)
    assert build_training_table(sos, telemetry, None).empty

    work_orders = prepare_work_orders(pd.DataFrame({
        "WorkOrderId": ["WO-1", "WO-2"],
        "EquipmentId": ["EQ-1", "EQ-1"],
        "Component": ["ENG", "ENG"],
        "OpenedDate": ["2026-01-10", "2026-02-15"],
        "WorkOrderType": ["CORRECTIVE", "PREVENTIVE"],
        "FailureConfirmed": [1, 0],
    }))
    labelled = build_training_table(sos, telemetry, work_orders, horizon_days=30)
    assert len(labelled) == 3
    assert labelled["corrective_wo_within_horizon"].eq(1).all()
    assert labelled["corrective_wo_within_horizon"].nunique() == 1


def test_small_example_wo_file_does_not_unlock_prediction():
    sos = _current_sos_raw()
    asset_id = str(prepare_sos(sos).iloc[0]["asset_id"])
    small_wo = pd.DataFrame({
        "WorkOrderId": ["WO-1", "WO-2"],
        "EquipmentId": [asset_id, asset_id],
        "Component": ["ENG", "ENG"],
        "OpenedDate": ["2026-01-10", "2026-02-10"],
        "ObservationEndDate": ["2026-12-31", "2026-12-31"],
        "WorkOrderType": ["CORRECTIVE", "PREVENTIVE"],
        "FailureConfirmed": [1, 0],
    })
    result = analyze_frames(
        sos,
        telemetry_raw=None,
        work_orders_raw=small_wo,
    )
    gate = result["readiness"].set_index("gate")
    assert gate.loc["explicit_wo_outcomes", "status"] == "BLOCKED"
    assert result["operating_mode"] == MODE_ALERT
    assert not result["predictive_risk_enabled"]


def test_right_censored_samples_are_not_labelled_negative():
    sos = prepare_sos(pd.DataFrame({
        "SampleNum": ["S01"],
        "EquipNum": ["EQ-1"],
        "Compartment": ["ENG"],
        "OverallInterp": ["A"],
        "DateSampled": ["2026-01-20"],
        "Fe": [10],
    }))
    telemetry, _ = prepare_telemetry(None)
    work_orders = prepare_work_orders(pd.DataFrame({
        "WorkOrderId": ["WO-1"],
        "EquipmentId": ["EQ-1"],
        "Component": ["ENG"],
        "OpenedDate": ["2026-01-25"],
        "WorkOrderType": ["PREVENTIVE"],
        "FailureConfirmed": [0],
    }))
    assert build_training_table(sos, telemetry, work_orders, horizon_days=30).empty


def test_observation_end_is_applied_per_asset_when_supplied():
    sos = prepare_sos(pd.DataFrame({
        "SampleNum": ["S-A", "S-B"],
        "EquipNum": ["EQ-A", "EQ-B"],
        "Compartment": ["ENG", "ENG"],
        "OverallInterp": ["A", "A"],
        "DateSampled": ["2026-01-20", "2026-01-20"],
        "Fe": [10, 10],
    }))
    telemetry, _ = prepare_telemetry(None)
    work_orders = prepare_work_orders(pd.DataFrame({
        "WorkOrderId": ["WO-A", "WO-B"],
        "EquipmentId": ["EQ-A", "EQ-B"],
        "Component": ["ENG", "ENG"],
        "OpenedDate": ["2026-02-25", "2026-01-25"],
        "ObservationEndDate": ["2026-02-28", "2026-02-01"],
        "WorkOrderType": ["PREVENTIVE", "PREVENTIVE"],
        "FailureConfirmed": [0, 0],
    }))
    labelled = build_training_table(sos, telemetry, work_orders, horizon_days=30)
    assert set(labelled["asset_id"]) == {"EQ-A"}
    assert labelled["corrective_wo_within_horizon"].eq(0).all()


def test_optional_sources_do_not_block_scoring_or_create_false_negatives():
    sos = prepare_sos(pd.DataFrame({
        "SampleNum": ["S01", "S02", "S03", "S04"],
        "EquipNum": ["EQ-LABELLED", "EQ-LABELLED", "EQ-UNLABELLED", "EQ-UNLABELLED"],
        "Compartment": ["ENG"] * 4,
        "OverallInterp": ["A", "AR", "A", "AR"],
        "DateSampled": ["2025-01-01", "2025-02-01", "2025-01-01", "2025-02-01"],
        "Fe": [10, 20, 12, 24],
    }))
    telemetry, _ = prepare_telemetry(None)
    work_orders = prepare_work_orders(pd.DataFrame({
        "WorkOrderId": ["WO-1", "WO-2"],
        "EquipmentId": ["EQ-LABELLED", "EQ-LABELLED"],
        "Component": ["ENG", "ENG"],
        "OpenedDate": ["2025-02-15", "2025-04-15"],
        "WorkOrderType": ["CORRECTIVE", "PREVENTIVE"],
        "FailureConfirmed": [1, 0],
    }))

    scoring = build_scoring_table(sos, telemetry, work_orders)
    assert len(scoring) == 4
    assert scoring["telemetry_available"].eq(0).all()
    assert scoring.loc[scoring["asset_id"].eq("EQ-UNLABELLED"), "work_order_history_available"].eq(0).all()
    labelled_scoring = scoring[scoring["asset_id"].eq("EQ-LABELLED")].sort_values("sample_date")
    assert labelled_scoring["iron_ppm_mean_3"].tolist() == [10.0, 15.0]

    training = build_training_table(sos, telemetry, work_orders, horizon_days=30)
    assert set(training["asset_id"]) == {"EQ-LABELLED"}


def test_text_heavy_sos_is_a_valid_predictor_source_but_not_a_label():
    rows = 60
    result = analyze_frames(pd.DataFrame({
        "SampleNum": [f"S{i:03d}" for i in range(rows)],
        "EquipNum": [f"EQ-{i % 10}" for i in range(rows)],
        "Compartment": ["ENG"] * rows,
        "OverallInterp": ["AR"] * rows,
        "InterpText": [
            "Elevated iron wear; inspect and resample." if i % 2
            else "Water contamination is suspected; resample."
            for i in range(rows)
        ],
        "DateSampled": pd.date_range("2024-01-01", periods=rows, freq="7D"),
    }))
    gates = result["readiness"].set_index("gate")
    assert gates.loc["lab_predictors", "status"] == "PASS"
    assert gates.loc["numeric_trends", "status"] == "BLOCKED"
    assert result["training_table"].empty
    assert result["operating_mode"] == MODE_ALERT
    scoring = result["scoring_table"]
    assert scoring["lab_text_available"].eq(1).all()
    assert scoring[["lab_text_wear_signal", "lab_text_coolant_signal"]].sum().sum() == rows


def test_prediction_pipeline_runs_without_telemetry_and_scores_all_sos_rows():
    sos, work_orders = _synthetic_prediction_inputs()
    result = analyze_frames(
        sos,
        telemetry_raw=None,
        work_orders_raw=work_orders,
    )
    assert result["operating_mode"] == MODE_PREDICTION
    assert result["predictive_risk_enabled"]
    assert result["matched_assets"] == set()
    assert result["scoring_table"]["telemetry_available"].eq(0).all()
    assert len(result["scoring_table"]) == len(result["sos"])
    assert result["maintenance_action_cards"]["failure_probability_pct"].notna().all()
    assert result["trained_model"].metrics["passes_utility_gate"]
    utility_gate = result["readiness"].set_index("gate").loc["heldout_model_utility"]
    assert utility_gate["status"] == "PASS"


def test_model_training_uses_separate_chronological_calibration_block():
    rows = 90
    training = pd.DataFrame({
        "sample_number": [f"S{i:03d}" for i in range(rows)],
        "asset_id": [f"EQ-{i % 10}" for i in range(rows)],
        "sample_date": pd.date_range("2025-01-01", periods=rows, freq="D"),
        "machine_model": ["MODEL-A" if i % 2 else "MODEL-B" for i in range(rows)],
        "component": ["ENG" if i % 3 else "HYD" for i in range(rows)],
        "equipment_hours": np.arange(rows) * 10.0,
        "fluid_hours": np.arange(rows) % 20,
        "future_only_feature": [0] * 75 + list(range(15)),
        "corrective_wo_within_horizon": [i % 2 for i in range(rows)],
        "horizon_days": [7] * rows,
    })
    model, leaderboard = train_failure_models(training)
    assert model.metrics["calibration_rows"] >= 8
    assert model.metrics["test_rows"] >= 8
    assert model.metrics["embargo_days"] == 7
    assert 0.10 <= model.metrics["decision_threshold"] <= 0.90
    assert "precision_at_threshold" in model.metrics
    assert "baseline_brier_score" in model.metrics
    assert "passes_utility_gate" in model.metrics
    assert model.metrics["asset_balanced_training"]
    assert "future_only_feature" not in model.features
    assert pd.Timestamp(model.metrics["calibration_start"]) < pd.Timestamp(model.metrics["test_start"])
    assert set(leaderboard["model"]) == {"logistic_regression", "random_forest"}
    probabilities = model.predict_proba(training.head(5))
    assert np.isfinite(probabilities).all()
    assert ((probabilities >= 0) & (probabilities <= 1)).all()


def test_model_utility_gate_rejects_a_constant_no_skill_prediction():
    y_true = pd.Series([0, 1] * 10)
    probabilities = np.full(len(y_true), 0.5)
    metrics = {
        "average_precision": 0.5,
        "brier_score": 0.25,
        "recall_at_threshold": 1.0,
    }
    utility = _heldout_utility_metrics(y_true, probabilities, 0.5, metrics)
    assert utility["average_precision_lift"] == 0.0
    assert utility["brier_skill_score"] == 0.0
    assert not utility["passes_utility_gate"]


def test_cli_outputs_are_utf8_serializable(tmp_path):
    base = ROOT / "data/current"
    history = next(iter(sorted(base.glob("SampleHistory*.xlsx"))))
    details = next(iter(sorted(base.glob("SampleTestDetails*.xlsx"))), None)
    result = run_analysis(
        history,
        telemetry_path=None,
        output_dir=tmp_path,
        test_details_path=details,
    )
    assert result["operating_mode"] == MODE_ALERT
    report = (tmp_path / "maintenance_summary.md").read_text(encoding="utf-8")
    assert "Maintenance Triage Summary" in report
    assert (tmp_path / "analysis_summary.json").exists()
    assert (tmp_path / "model_readiness.csv").exists()


def test_presentation_uses_current_mode_and_unique_cases():
    result = analyze_frames(_current_sos_raw())
    mode = build_mode_summary(result, 90)
    cases = build_case_table(result, 90)
    fleet = build_fleet_summary(result, cases, 90)
    assert mode["title"] == "Alert Management active"
    assert not mode["prediction_enabled"]
    assert cases["case_id"].is_unique
    assert len(cases) < len(result["sos"])
    assert fleet["samples"] == len(result["sos"])
    assert fleet["machine_components"] == len(cases)


def test_case_probability_uses_latest_prediction_not_historical_maximum():
    cards = pd.DataFrame({
        "asset_id": ["EQ-1", "EQ-1"],
        "component": ["ENG", "ENG"],
        "priority_tier": [P1_IMMEDIATE, P3_ACTION],
        "sample_date": ["2025-01-01", "2025-02-01"],
        "failure_probability_pct": [90, 20],
        "prediction_data_sources": ["S.O.S., telemetry", "S.O.S."],
        "key_evidence": [["old"], ["latest"]],
        "sample_number": ["S1", "S2"],
        "machine_model": ["M", "M"],
        "site_name": ["A", "A"],
        "lab_status": ["Warning", "Normal"],
        "action_needed": ["Inspect", "Review"],
        "wo_id": [None, None],
        "data_quality_status": ["Valid", "Valid"],
        "data_quality_issue": ["", ""],
    })
    cases = build_case_table({
        "maintenance_action_cards": cards,
        "sos": pd.DataFrame(),
        "telemetry": pd.DataFrame(),
        "work_orders": pd.DataFrame(),
    }, 30)
    assert cases.iloc[0]["failure_probability_pct"] == 20
    assert cases.iloc[0]["prediction_data_sources"] == "S.O.S."
    assert cases.iloc[0]["active_finding_lab_status"] == "Warning"
    assert cases.iloc[0]["latest_lab_status"] == "Normal"
    assert pd.Timestamp(cases.iloc[0]["finding_sample_date"]) == pd.Timestamp("2025-01-01")
    assert pd.Timestamp(cases.iloc[0]["latest_sample_date"]) == pd.Timestamp("2025-02-01")


def test_condition_index_uses_latest_valid_date_and_true_consecutive_ar_count():
    result = analyze_frames(pd.DataFrame({
        "SampleNum": ["S1", "S2", "S3", "S4"],
        "EquipNum": ["EQ-1"] * 4,
        "Compartment": ["ENGINE"] * 4,
        "OverallInterp": ["AR", "NAR", "AR", "NAR"],
        "InterpText": ["Wear", "Normal", "Wear", "Undated normal"],
        "DateSampled": ["2025-01-01", "2025-02-01", "2025-03-01", "invalid"],
    }))
    health = compute_machine_health_score(result, "EQ-1", "ENG")
    assert health["health_score"] == 70
    assert "Lab Action Required (AR)" in health["penalties"]
    assert not any("consecutive" in item for item in health["penalties"])


def test_sparse_measurement_change_uses_last_two_non_null_values():
    result = analyze_frames(pd.DataFrame({
        "SampleNum": ["S1", "S2", "S3"],
        "EquipNum": ["EQ-1"] * 3,
        "Compartment": ["ENG"] * 3,
        "OverallInterp": ["NAR"] * 3,
        "DateSampled": ["2025-01-01", "2025-02-01", "2025-03-01"],
        "Fe": [10.0, np.nan, 30.0],
    }))
    trend = measurement_trend_table(result["sos"], "EQ-1", "ENG")
    changes = latest_measurement_changes(trend, limit=8).set_index("measurement")
    assert changes.loc["iron_ppm", "current"] == 30.0
    assert changes.loc["iron_ppm", "previous"] == 10.0
    assert changes.loc["iron_ppm", "change"] == 20.0


def test_enum_dictionary_is_rejected_as_telemetry():
    enum_path = ROOT / "data/current/Telematics_Enums.xlsx"
    result = analyze_frames(_current_sos_raw(), load_table(enum_path))
    assert result["telemetry"].empty
    assert "ignored" in result["telemetry_quality"]["schema_issue"].lower()
    issues = build_quality_issues(result)
    assert "Invalid telemetry schema" in set(issues["issue"])


def test_streamlit_dashboard_smoke_and_current_mode():
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(str(ROOT / "code/app.py"), default_timeout=90).run()
    assert not app.exception
    assert next(item for item in app.radio if item.label == "Source").value == "Current data"
    assert next(item for item in app.select_slider if item.label == "Forecast window (days)").value == 30
    assert any("Machine Health Dashboard" in item.value for item in app.markdown)
    assert any("Alert Management active" in item.value for item in app.markdown)


def test_synthetic_prediction_has_validation_summary():
    sos, work_orders = _synthetic_prediction_inputs()
    result = analyze_frames(sos, work_orders_raw=work_orders)
    validation = build_validation_summary(result)
    assert result["operating_mode"] == MODE_PREDICTION
    assert validation is not None
    assert validation["passes_utility_gate"]
