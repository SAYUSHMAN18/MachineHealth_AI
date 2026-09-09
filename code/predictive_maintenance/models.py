from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy.stats import beta
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TARGET = "corrective_wo_within_horizon"
EXCLUDED = {
    TARGET,
    "source_row",
    "sample_number",
    "asset_id",
    "sample_date",
    "horizon_days",
}
TELEMETRY_FEATURES = {
    "telemetry_available",
    "operating_hours_7d",
    "operating_hours_30d",
    "operating_hours_90d",
    "distance_30d",
    "mean_utilization_30d",
    "telemetry_age_days",
}


@dataclass
class TrainedModel:
    name: str
    pipeline: Any
    features: list[str]
    metrics: dict[str, Any]
    cutoff_date: pd.Timestamp
    explain_pipeline: Pipeline | None = None

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        return self.pipeline.predict_proba(frame[self.features])[:, 1]

    def get_feature_importances(self) -> dict[str, float]:
        """Extract normalized feature importances or absolute model coefficients."""
        try:
            source = self.explain_pipeline or self.pipeline
            model = source.named_steps["model"]
            preprocess = source.named_steps["preprocess"]

            if hasattr(preprocess, "get_feature_names_out"):
                raw_names = preprocess.get_feature_names_out()
            else:
                raw_names = [f"feature_{i}" for i in range(len(self.features))]

            if hasattr(model, "feature_importances_"):
                scores = model.feature_importances_
            elif hasattr(model, "coef_"):
                scores = np.abs(model.coef_[0])
            else:
                return {}

            clean_scores: dict[str, float] = {}
            for name, score in zip(raw_names, scores):
                clean_name = name.split("__")[-1]
                clean_scores[clean_name] = clean_scores.get(clean_name, 0.0) + float(score)

            total = sum(clean_scores.values())
            if total > 0:
                clean_scores = {k: round(v / total, 4) for k, v in clean_scores.items()}
            return dict(sorted(clean_scores.items(), key=lambda x: x[1], reverse=True))
        except Exception:
            return {}


def _safe_metrics(
    y_true: pd.Series, probabilities: np.ndarray, threshold: float = 0.5
) -> dict[str, Any]:
    predictions = (probabilities >= threshold).astype(int)
    metrics: dict[str, Any] = {
        "average_precision": float(average_precision_score(y_true, probabilities)),
        "decision_threshold": float(threshold),
        "precision_at_threshold": float(precision_score(y_true, predictions, zero_division=0)),
        "recall_at_threshold": float(recall_score(y_true, predictions, zero_division=0)),
        "f1_at_threshold": float(f1_score(y_true, predictions, zero_division=0)),
        "brier_score": float(brier_score_loss(y_true, probabilities)),
        "confusion_matrix": confusion_matrix(y_true, predictions, labels=[0, 1]).tolist(),
        "test_rows": int(len(y_true)),
        "test_positive_rate": float(y_true.mean()),
    }
    metrics["roc_auc"] = (
        float(roc_auc_score(y_true, probabilities)) if y_true.nunique() == 2 else None
    )
    return metrics


def _select_decision_threshold(
    y_true: pd.Series, probabilities: np.ndarray
) -> float:
    """Choose a practical validation threshold without touching test data."""
    candidates = np.linspace(0.10, 0.90, 81)
    scored: list[tuple[float, float, float, float]] = []
    for threshold in candidates:
        predicted = (probabilities >= threshold).astype(int)
        scored.append(
            (
                float(f1_score(y_true, predicted, zero_division=0)),
                float(recall_score(y_true, predicted, zero_division=0)),
                -abs(float(threshold) - 0.5),
                float(threshold),
            )
        )
    return max(scored)[-1]


def _equal_asset_sample_weights(frame: pd.DataFrame) -> np.ndarray:
    """Give every machine equal total influence during model fitting.

    Repeated samples from one frequently tested machine should not overwhelm
    the fleet model. The returned weights have mean 1.0, so estimator
    regularisation remains on approximately the same scale as before.
    """
    if frame.empty or "asset_id" not in frame.columns:
        return np.ones(len(frame), dtype=float)
    asset_keys = frame["asset_id"].astype("string").fillna("__missing_asset__")
    counts = asset_keys.value_counts(dropna=False)
    weights = asset_keys.map(lambda key: 1.0 / float(counts.loc[key])).to_numpy(dtype=float)
    mean_weight = float(weights.mean()) if len(weights) else 1.0
    return weights / mean_weight if mean_weight > 0 else np.ones(len(frame), dtype=float)


def _heldout_utility_metrics(
    y_true: pd.Series,
    probabilities: np.ndarray,
    development_prevalence: float,
    model_metrics: dict[str, Any],
) -> dict[str, Any]:
    """Compare the selected model with a no-feature prevalence baseline.

    The baseline probability comes only from the development period. Test
    labels are used once, here, to decide whether the trained model is useful
    enough to enable live probability output.
    """
    baseline_probability = float(np.clip(development_prevalence, 0.0, 1.0))
    baseline_probabilities = np.full(len(y_true), baseline_probability, dtype=float)
    baseline_average_precision = float(y_true.mean())
    baseline_brier = float(brier_score_loss(y_true, baseline_probabilities))
    average_precision_lift = (
        float(model_metrics["average_precision"]) - baseline_average_precision
    )
    brier_skill_score = (
        1.0 - float(model_metrics["brier_score"]) / baseline_brier
        if baseline_brier > 0 else 0.0
    )
    passes = bool(
        average_precision_lift >= 0.02
        and brier_skill_score > 0.0
        and float(model_metrics.get("recall_at_threshold", 0.0)) > 0.0
    )
    return {
        "baseline_probability": baseline_probability,
        "baseline_average_precision": baseline_average_precision,
        "baseline_brier_score": baseline_brier,
        "average_precision_lift": average_precision_lift,
        "brier_skill_score": brier_skill_score,
        "passes_utility_gate": passes,
        "minimum_average_precision_lift": 0.02,
    }


def train_failure_models(training: pd.DataFrame) -> tuple[TrainedModel, pd.DataFrame]:
    """Train availability-aware baselines with leak-resistant time blocks.

    Model selection uses the middle validation/calibration period. The final
    chronological test block is touched once, after a winner has been chosen.
    Whole timestamps remain in one block and the outcome horizon is purged from
    the end of earlier blocks so future labels cannot cross a boundary.
    """
    if len(training) < 60:
        raise ValueError("At least 60 labelled samples are required for chronological calibration and testing.")
    if training[TARGET].nunique() < 2:
        raise ValueError("Training data must contain both positive and negative outcomes.")

    data = training.copy()
    data["sample_date"] = pd.to_datetime(data["sample_date"], errors="coerce")
    data = data.dropna(subset=["sample_date"]).sort_values("sample_date").reset_index(drop=True)
    unique_dates = pd.Index(data["sample_date"].drop_duplicates().sort_values())
    if len(unique_dates) < 5:
        raise ValueError("At least five distinct sample dates are required for chronological validation.")

    horizon_days = int(pd.to_numeric(data.get("horizon_days", 30), errors="coerce").max())
    horizon = pd.Timedelta(max(horizon_days, 0), unit="D")
    split_candidates: list[tuple[pd.Timestamp, pd.Timestamp, pd.DataFrame, pd.DataFrame, pd.DataFrame]] = []
    train_positions = range(max(1, int(len(unique_dates) * 0.50)), max(2, int(len(unique_dates) * 0.72)))
    calibration_positions = range(max(2, int(len(unique_dates) * 0.72)), max(3, int(len(unique_dates) * 0.90)))
    for train_position in train_positions:
        if train_position >= len(unique_dates):
            continue
        calibration_start = pd.Timestamp(unique_dates[train_position])
        for calibration_position in calibration_positions:
            if calibration_position <= train_position or calibration_position >= len(unique_dates):
                continue
            test_start = pd.Timestamp(unique_dates[calibration_position])
            train = data.loc[data["sample_date"] + horizon < calibration_start].copy()
            calibration = data.loc[
                data["sample_date"].ge(calibration_start)
                & (data["sample_date"] + horizon < test_start)
            ].copy()
            test = data.loc[data["sample_date"].ge(test_start)].copy()
            blocks = [train, calibration, test]
            if all(len(block) >= 8 and block[TARGET].nunique() == 2 for block in blocks):
                split_candidates.append(
                    (calibration_start, test_start, train, calibration, test)
                )
    if not split_candidates:
        raise ValueError(
            "Leak-resistant chronological train, calibration, and test blocks must each contain "
            "at least eight rows and both outcome classes after applying the prediction-horizon embargo."
        )
    calibration_start, test_start, train, calibration, test = split_candidates[
        len(split_candidates) // 2
    ]

    # Decide feature availability from the training period only. Looking at
    # future validation/test rows—even without using their labels—can leak
    # knowledge about which fields and categories appear later in time.
    features = [
        column for column in train.columns
        if column not in EXCLUDED
        and train[column].notna().any()
        and train[column].nunique(dropna=False) > 1
    ]
    # A tiny telemetry pocket can otherwise act like an asset/site proxy. Use
    # telemetry only when its labelled coverage is broad enough to validate.
    if "telemetry_available" in train.columns:
        telemetry_mask = pd.to_numeric(train["telemetry_available"], errors="coerce").fillna(0).gt(0)
        telemetry_rows = int(telemetry_mask.sum())
        telemetry_assets = int(train.loc[telemetry_mask, "asset_id"].nunique())
        minimum_telemetry_rows = max(20, int(np.ceil(len(train) * 0.05)))
        if telemetry_rows < minimum_telemetry_rows or telemetry_assets < 3:
            features = [feature for feature in features if feature not in TELEMETRY_FEATURES]
    if not features:
        raise ValueError("No varying predictor columns are available after data-quality filtering.")
    categorical = [column for column in features if train[column].dtype == "object" or str(train[column].dtype).startswith("string")]
    numeric = [column for column in features if column not in categorical]

    numeric_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    preprocess = ColumnTransformer(
        [("numeric", numeric_pipe, numeric), ("categorical", categorical_pipe, categorical)],
        remainder="drop",
    )
    candidates = {
        "logistic_regression": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42),
        "random_forest": RandomForestClassifier(
            n_estimators=160,
            max_depth=12,
            min_samples_leaf=4,
            max_features="sqrt",
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
        ),
    }
    rows: list[dict[str, Any]] = []
    fitted_bases: dict[str, Pipeline] = {}
    train_weights = _equal_asset_sample_weights(train)
    for name, estimator in candidates.items():
        base_pipeline = Pipeline([("preprocess", clone(preprocess)), ("model", estimator)])
        base_pipeline.fit(
            train[features], train[TARGET], model__sample_weight=train_weights
        )
        validation_probabilities = base_pipeline.predict_proba(calibration[features])[:, 1]
        metrics = _safe_metrics(calibration[TARGET], validation_probabilities)
        metrics["model"] = name
        metrics["train_rows"] = int(len(train))
        metrics["calibration_rows"] = int(len(calibration))
        metrics["evaluation_period"] = "validation"
        rows.append(metrics)
        fitted_bases[name] = base_pipeline

    leaderboard = pd.DataFrame(rows).sort_values(
        ["average_precision", "brier_score"], ascending=[False, True]
    )
    winner = str(leaderboard.iloc[0]["model"])
    winning_base = fitted_bases[winner]
    try:
        from sklearn.frozen import FrozenEstimator
        calibrated = CalibratedClassifierCV(FrozenEstimator(winning_base), method="sigmoid")
    except ImportError:
        calibrated = CalibratedClassifierCV(winning_base, method="sigmoid", cv="prefit")
    calibrated.fit(calibration[features], calibration[TARGET])
    calibrated_validation_probabilities = calibrated.predict_proba(calibration[features])[:, 1]
    decision_threshold = _select_decision_threshold(
        calibration[TARGET], calibrated_validation_probabilities
    )
    test_probabilities = calibrated.predict_proba(test[features])[:, 1]
    best_metrics = _safe_metrics(test[TARGET], test_probabilities, decision_threshold)
    development_prevalence = float(
        pd.concat([train[TARGET], calibration[TARGET]], ignore_index=True).mean()
    )
    best_metrics.update(
        _heldout_utility_metrics(
            test[TARGET], test_probabilities, development_prevalence, best_metrics
        )
    )
    selection_metrics = next(row for row in rows if row["model"] == winner)
    best_metrics.update({
        "model": winner,
        "selection_average_precision": selection_metrics["average_precision"],
        "selection_brier_score": selection_metrics["brier_score"],
        "train_rows": int(len(train)),
        "calibration_rows": int(len(calibration)),
        "calibration_start": calibration_start.isoformat(),
        "test_start": test_start.isoformat(),
        "embargo_days": horizon_days,
        "asset_balanced_training": True,
    })
    model = TrainedModel(
        name=winner,
        pipeline=calibrated,
        features=features,
        metrics=best_metrics,
        cutoff_date=test_start,
        explain_pipeline=winning_base,
    )
    return model, leaderboard


class BayesianComponentBaseline:
    """Empirical-Bayes component event-rate baseline with credible intervals."""

    def __init__(self, prior_strength: float = 12.0):
        self.prior_strength = prior_strength
        self.table_: pd.DataFrame | None = None
        self.fleet_: dict[str, float] | None = None

    def fit(self, training: pd.DataFrame) -> "BayesianComponentBaseline":
        fleet_rate = float(training[TARGET].mean())
        alpha0 = max(0.5, fleet_rate * self.prior_strength)
        beta0 = max(0.5, (1 - fleet_rate) * self.prior_strength)
        grouped = training.groupby(["machine_model", "component"], dropna=False)[TARGET].agg(["sum", "count"])
        grouped["alpha"] = alpha0 + grouped["sum"]
        grouped["beta"] = beta0 + grouped["count"] - grouped["sum"]
        grouped["posterior_mean"] = grouped["alpha"] / (grouped["alpha"] + grouped["beta"])
        grouped["credible_low_90"] = beta.ppf(0.05, grouped["alpha"], grouped["beta"])
        grouped["credible_high_90"] = beta.ppf(0.95, grouped["alpha"], grouped["beta"])
        self.table_ = grouped.reset_index()
        self.fleet_ = {
            "alpha": alpha0 + training[TARGET].sum(),
            "beta": beta0 + len(training) - training[TARGET].sum(),
        }
        return self

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        if self.table_ is None or self.fleet_ is None:
            raise RuntimeError("Fit the Bayesian baseline before prediction.")
        merged = frame[["machine_model", "component"]].merge(
            self.table_, on=["machine_model", "component"], how="left"
        )
        fleet_mean = self.fleet_["alpha"] / (self.fleet_["alpha"] + self.fleet_["beta"])
        merged["posterior_mean"] = merged["posterior_mean"].fillna(fleet_mean)
        merged["credible_low_90"] = merged["credible_low_90"].fillna(
            beta.ppf(0.05, self.fleet_["alpha"], self.fleet_["beta"])
        )
        merged["credible_high_90"] = merged["credible_high_90"].fillna(
            beta.ppf(0.95, self.fleet_["alpha"], self.fleet_["beta"])
        )
        return merged[["posterior_mean", "credible_low_90", "credible_high_90"]]


def score_telemetry_anomalies(telemetry: pd.DataFrame) -> pd.DataFrame:
    """Fit an unsupervised snapshot anomaly model to cleaned telemetry."""
    out = telemetry.copy()
    features = [
        "gap_hours",
        "operating_hours_delta",
        "odometer_delta",
        "distance_per_operating_hour",
        "utilization_rate",
    ]
    usable = out[features].replace([np.inf, -np.inf], np.nan)
    if len(out) < 20 or usable.notna().sum().sum() == 0:
        out["telemetry_anomaly_score"] = np.nan
        out["telemetry_anomaly"] = False
        return out

    matrix = SimpleImputer(strategy="median", add_indicator=True).fit_transform(usable)
    model = IsolationForest(n_estimators=250, contamination=0.05, random_state=42, n_jobs=-1)
    labels = model.fit_predict(matrix)
    raw_score = -model.score_samples(matrix)
    low, high = np.nanmin(raw_score), np.nanmax(raw_score)
    normalized = (raw_score - low) / (high - low) if high > low else np.zeros_like(raw_score)
    out["telemetry_anomaly_score"] = normalized
    out["telemetry_anomaly"] = labels == -1
    return out
