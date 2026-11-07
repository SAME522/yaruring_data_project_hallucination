"""Fair, reproducible HaluEval Yes/No model audition.

Use only the saved train/test split. Vectorizers, scalers and encoders are
trained *inside* each CV fold; test labels are never used to select a model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer


SEED = 42
MODEL_NAMES = ("로지스틱 회귀", "kNN", "랜덤 포레스트")
FEATURE_SETS = ("기본 특징", "기본 + 군집")

BASE_NUMERIC = (
    "char_count", "token_count", "unique_token_count",
    "avg_alpha_word_length", "question_mark_count", "number_token_count",
)
CLUSTER_NUMERIC = tuple(f"word_cluster_{i}_share" for i in range(8))
BASE_CATEGORICAL = ("intent_type",)
CLUSTER_CATEGORICAL = ("question_cluster",)


def prepare_question_data(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Validate the existing preprocessed CSV without redefining its split."""
    needed = {"question", "split", "y", *BASE_NUMERIC,
              *CLUSTER_NUMERIC, *BASE_CATEGORICAL, *CLUSTER_CATEGORICAL}
    missing = sorted(needed.difference(raw.columns))
    if missing:
        raise ValueError(f"필수 데이터 열이 없습니다: {', '.join(missing)}")
    data = raw.copy()
    data["question"] = data["question"].fillna("").astype(str)
    data["intent_type"] = data["intent_type"].fillna("other_or_unclear").astype(str)
    data["question_cluster"] = data["question_cluster"].fillna(-1).astype(str)
    for col in BASE_NUMERIC + CLUSTER_NUMERIC:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    data["y"] = pd.to_numeric(data["y"], errors="coerce")
    if data["y"].isna().any() or not data["y"].isin([0, 1]).all():
        raise ValueError("할루시네이션 정답 y가 0/1 이외의 값을 포함합니다.")
    train = data.loc[data["split"].eq("train")].copy()
    test = data.loc[data["split"].eq("test")].copy()
    if len(train) != 3605 or len(test) != 902:
        raise ValueError("저장된 원본 분할(학습 3,605 / 시험 902)이 유지되지 않았습니다.")
    if not train["y"].nunique() == test["y"].nunique() == 2:
        raise ValueError("각 분할에 Yes와 No가 모두 포함되어야 합니다.")
    if "source_row" in data and data["source_row"].duplicated().any():
        raise ValueError("source_row 식별자가 중복됩니다.")
    return train, test


def make_preprocessor(with_clusters: bool) -> ColumnTransformer:
    numeric = list(BASE_NUMERIC)
    categorical = list(BASE_CATEGORICAL)
    if with_clusters:
        numeric.extend(CLUSTER_NUMERIC)
        categorical.extend(CLUSTER_CATEGORICAL)
    numeric_steps = Pipeline([
        ("fill", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    return ColumnTransformer([
        ("tfidf", TfidfVectorizer(max_features=1200, min_df=3,
                                  ngram_range=(1, 2), sublinear_tf=True), "question"),
        ("numeric", numeric_steps, numeric),
        ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical),
    ], sparse_threshold=1.0)


def make_estimator(name: str):
    if name == "로지스틱 회귀":
        return LogisticRegression(max_iter=1500, solver="liblinear",
                                  class_weight="balanced", random_state=SEED)
    if name == "kNN":
        return KNeighborsClassifier(n_neighbors=21, weights="distance",
                                    metric="cosine", algorithm="brute")
    if name == "랜덤 포레스트":
        return RandomForestClassifier(n_estimators=100, max_depth=18,
                                      min_samples_leaf=4, class_weight="balanced_subsample",
                                      random_state=SEED, n_jobs=2)
    raise ValueError(f"알 수 없는 모델: {name}")


def make_pipeline(name: str, with_clusters: bool) -> Pipeline:
    return Pipeline([
        ("features", make_preprocessor(with_clusters)),
        ("model", make_estimator(name)),
    ])


def calculate_metrics(y_true, y_pred, yes_score) -> dict:
    tn, fp, fn, tp = (int(v) for v in confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel())
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, yes_score)),
        "average_precision": float(average_precision_score(y_true, yes_score)),
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
    }


def audition(raw: pd.DataFrame, n_splits: int = 5) -> tuple[pd.DataFrame, dict]:
    train, test = prepare_question_data(raw)
    x_train, y_train = train, train["y"].astype(int)
    x_test, y_test = test, test["y"].astype(int)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    rows = []
    fitted = {}

    # A majority-class baseline provides a sanity check for imbalanced labels.
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(np.zeros((len(train), 1)), y_train)
    base_pred = dummy.predict(np.zeros((len(test), 1)))
    base_prob = np.full(len(test), float(y_train.mean()))
    baseline = calculate_metrics(y_test, base_pred, base_prob)

    for feature_name in FEATURE_SETS:
        with_clusters = feature_name == "기본 + 군집"
        for name in MODEL_NAMES:
            pipeline = make_pipeline(name, with_clusters)
            scores = cross_validate(
                pipeline, x_train, y_train, cv=cv, n_jobs=1,
                scoring={"auc": "roc_auc", "ap": "average_precision",
                         "f1": "f1", "recall": "recall"},
                error_score="raise",
            )
            # Final training uses the entire train split; test stays untouched
            # until all CV scores and model settings have been determined.
            pipeline.fit(x_train, y_train)
            prob = pipeline.predict_proba(x_test)[:, 1]
            pred = (prob >= 0.5).astype(int)
            test_metrics = calculate_metrics(y_test, pred, prob)
            row = {"모델": name, "특징": feature_name}
            for key, label in [("auc", "CV ROC-AUC"), ("ap", "CV PR-AUC"),
                               ("f1", "CV F1"), ("recall", "CV 재현율")]:
                row[label] = float(np.mean(scores[f"test_{key}"]))
                row[label + " 표준편차"] = float(np.std(scores[f"test_{key}"]))
            row.update({"시험 " + key: value for key, value in test_metrics.items()})
            rows.append(row)
            fitted[(name, feature_name)] = pipeline

    table = pd.DataFrame(rows).sort_values(
        by=["CV PR-AUC", "CV ROC-AUC"], ascending=False, kind="stable",
    ).reset_index(drop=True)
    winner = table.iloc[0][["모델", "특징", "CV PR-AUC", "CV ROC-AUC"]].to_dict()
    meta = {
        "seed": SEED,
        "n_splits": n_splits,
        "train_count": len(train), "test_count": len(test),
        "train_yes": int(y_train.sum()), "test_yes": int(y_test.sum()),
        "baseline": baseline,
        "selected_by_train_cv": winner,
        "fitted_pipelines": fitted,
    }
    return table, meta


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Previously generated clustered-question CSV")
    parser.add_argument("--out", default="outputs/halueval_models_09")
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()
    frame = pd.read_csv(args.csv)
    results, meta = audition(frame, n_splits=args.folds)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    results.to_csv(out / "model_comparison_09.csv", index=False, encoding="utf-8-sig")
    summary = {k: v for k, v in meta.items() if k != "fitted_pipelines"}
    (out / "model_summary_09.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(results[["모델", "특징", "CV ROC-AUC", "CV PR-AUC",
                   "시험 roc_auc", "시험 average_precision", "시험 f1", "시험 recall"]]
          .round(4).to_string(index=False))
    print("Baseline:", summary["baseline"])
    print("Selected from train CV:", summary["selected_by_train_cv"])


if __name__ == "__main__":
    main()
