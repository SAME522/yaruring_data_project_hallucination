"""Compare three ways to estimate a 0–1 hallucination risk score.

The original 4,507 questions and 3,605/902 train/test split are unchanged.
Linear regression is an intentionally imperfect probability baseline: its raw
scores can leave [0, 1], so we clip only for probability metrics and display.
All transformations are fitted within training folds, never on held-out labels.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from modeling import SEED, make_preprocessor, prepare_question_data


MODEL_NAMES = ("선형 회귀", "로지스틱 회귀", "의사결정트리")
FEATURE_SETS = ("기본 특징", "기본 + 군집")
MODEL_KEYS = {"선형 회귀": "linear", "로지스틱 회귀": "logistic", "의사결정트리": "tree"}


def make_estimator(name: str):
    if name == "선형 회귀":
        return LinearRegression()
    if name == "로지스틱 회귀":
        # For risk estimation, avoid reweighting the prior prevalence.
        return LogisticRegression(max_iter=1500, solver="liblinear", random_state=SEED)
    if name == "의사결정트리":
        return DecisionTreeClassifier(
            max_depth=5, min_samples_leaf=40, random_state=SEED
        )
    raise ValueError(f"알 수 없는 모델: {name}")


def make_pipeline(name: str, with_clusters: bool) -> Pipeline:
    return Pipeline([
        ("features", make_preprocessor(with_clusters)),
        ("model", make_estimator(name)),
    ])


def predict_risk(model: Pipeline, frame: pd.DataFrame, name: str):
    """Return a bounded displayed risk score and number of invalid raw scores."""
    if name == "선형 회귀":
        raw = np.asarray(model.predict(frame), dtype=float)
    else:
        raw = np.asarray(model.predict_proba(frame)[:, 1], dtype=float)
    outside = int(np.sum((raw < 0.0) | (raw > 1.0)))
    return np.clip(raw, 0.0, 1.0), outside


def calculate_metrics(y_true, risk: np.ndarray, outside: int = 0) -> dict:
    predicted = (risk >= 0.5).astype(int)
    tn, fp, fn, tp = (int(i) for i in confusion_matrix(
        y_true, predicted, labels=[0, 1]
    ).ravel())
    return {
        "brier": float(brier_score_loss(y_true, risk)),
        "average_precision": float(average_precision_score(y_true, risk)),
        "roc_auc": float(roc_auc_score(y_true, risk)),
        "accuracy": float(accuracy_score(y_true, predicted)),
        "precision": float(precision_score(y_true, predicted, zero_division=0)),
        "recall": float(recall_score(y_true, predicted, zero_division=0)),
        "f1": float(f1_score(y_true, predicted, zero_division=0)),
        "out_of_range_count": outside,
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
    }


def audition(raw: pd.DataFrame, n_splits: int = 5):
    train, test = prepare_question_data(raw)
    y_train = train["y"].astype(int)
    y_test = test["y"].astype(int)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)

    # A constant probability is the relevant baseline for a risk score.
    prevalence = float(y_train.mean())
    baseline = calculate_metrics(y_test, np.full(len(test), prevalence))
    prediction_rows = pd.DataFrame({
        "source_row": test["source_row"].values if "source_row" in test else test.index,
        "question": test["question"].values,
        "y": y_test.values,
    })
    rows = []
    fitted = {}
    for features in FEATURE_SETS:
        with_clusters = features == "기본 + 군집"
        suffix = "on" if with_clusters else "off"
        for name in MODEL_NAMES:
            folds = []
            for train_idx, validation_idx in cv.split(train, y_train):
                model = make_pipeline(name, with_clusters)
                part = train.iloc[train_idx]
                validation = train.iloc[validation_idx]
                model.fit(part, y_train.iloc[train_idx])
                risk, outside = predict_risk(model, validation, name)
                folds.append(calculate_metrics(y_train.iloc[validation_idx], risk, outside))

            model = make_pipeline(name, with_clusters)
            model.fit(train, y_train)
            test_risk, test_outside = predict_risk(model, test, name)
            test_metrics = calculate_metrics(y_test, test_risk, test_outside)
            prediction_rows[f"{MODEL_KEYS[name]}_{suffix}"] = test_risk
            fitted[(name, features)] = model

            row = {"모델": name, "특징": features}
            for key, label in [
                ("brier", "CV Brier"),
                ("average_precision", "CV PR-AUC"),
                ("roc_auc", "CV ROC-AUC"),
                ("f1", "CV F1"),
                ("recall", "CV 재현율"),
            ]:
                values = [fold[key] for fold in folds]
                row[label] = float(np.mean(values))
                row[label + " 표준편차"] = float(np.std(values))
            row.update({"시험 " + key: value for key, value in test_metrics.items()})
            rows.append(row)

    table = pd.DataFrame(rows).sort_values(
        ["CV Brier", "CV PR-AUC"], ascending=[True, False], kind="stable"
    ).reset_index(drop=True)
    selected = table.iloc[0][["모델", "특징", "CV Brier", "CV PR-AUC"]].to_dict()
    meta = {
        "seed": SEED,
        "n_splits": n_splits,
        "train_count": len(train), "test_count": len(test),
        "train_yes": int(y_train.sum()), "test_yes": int(y_test.sum()),
        "baseline_train_yes_rate": prevalence,
        "baseline": baseline,
        "selected_by_train_cv": selected,
        "test_predictions": prediction_rows,
        "fitted_pipelines": fitted,
    }
    return table, meta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--out", default="outputs/halueval_probability")
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()
    raw = pd.read_csv(args.csv)
    table, meta = audition(raw, n_splits=args.folds)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "model_comparison_probability.csv",
                 index=False, encoding="utf-8-sig")
    meta["test_predictions"].to_csv(
        out / "model_predictions_probability.csv",
        index=False, encoding="utf-8-sig", float_format="%.6g"
    )
    summary = {k: v for k, v in meta.items()
               if k not in {"fitted_pipelines", "test_predictions"}}
    (out / "model_summary_probability.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(table[["모델", "특징", "CV Brier", "CV PR-AUC",
                 "시험 brier", "시험 average_precision", "시험 recall"]]
          .round(4).to_string(index=False))
    print("Baseline:", summary["baseline"])
    print("Chosen by train CV:", summary["selected_by_train_cv"])


if __name__ == "__main__":
    main()
