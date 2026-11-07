"""Three follow-up experiments for HaluEval hallucination risk prediction.

The pre-existing 3605/902 train/test split is fixed. Candidate choice is based
solely on five-fold CV over training rows. Calibrators use their own three-fold
CV within each *outer* training fold; the test rows never choose parameters.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import FunctionTransformer

from modeling import SEED, prepare_question_data
from probability_models import make_pipeline


GROUPS = ("1. 규제 강도 C", "2. 확률 보정", "3. 질문 의미 특징")
REGULARIZATION_VALUES = (0.03, 0.1, 0.3, 1.0, 3.0, 10.0)

# All features below are determined only by the question text, not the answer,
# HaluEval model output, source label or held-out statistics.
SEMANTIC_PATTERNS = {
    "current": r"\b(?:today|now|current|latest|recent|currently|this year|202[0-9])\b",
    "source": r"\b(?:source|sources|cite|citation|reference|references|evidence|study|studies|research)\b",
    "number": r"\b(?:how many|how much|percent|percentage|statistics|calculate|population|date|year|amount|number)\b",
    "person_place": r"\b(?:who|where|which country|which city|president|ceo|capital|born|died|located)\b",
    "verification": r"\b(?:true|false|accurate|verify|verification|fact|proof|real|correct)\b",
    "creative": r"\b(?:write|story|poem|fiction|invent|imagine|design|create|compose)\b",
    "explanation": r"\b(?:explain|compare|describe|why|how|difference|similar|meaning)\b",
}


def semantic_flags(questions, names):
    """Return a deterministic numeric design matrix from question wording."""
    text = pd.Series(questions).fillna("").astype(str)
    return np.column_stack([
        text.str.contains(SEMANTIC_PATTERNS[name], case=False, regex=True).to_numpy(dtype=float)
        for name in names
    ])


def factual_flags(questions):
    return semantic_flags(questions, ("current", "source", "number", "person_place"))


def full_flags(questions):
    return semantic_flags(questions, tuple(SEMANTIC_PATTERNS))


EXPERIMENTS = [
    {"group": "기존", "name": "기존 로지스틱 C=1", "kind": "original",
     "reason": "기존 TF-IDF + 질문 길이/유형 + 군집 특징, C=1"},
    *[
        {"group": GROUPS[0], "name": f"C={c:g}", "kind": "regularization", "c": c,
         "reason": f"로지스틱 회귀의 규제 강도 C={c:g} (작을수록 강한 규제)"}
        for c in REGULARIZATION_VALUES if c != 1.0
    ],
    {"group": GROUPS[1], "name": "시그모이드 보정", "kind": "calibration", "method": "sigmoid",
     "reason": "훈련 부분 안쪽 3겹 교차검증으로 시그모이드 확률 보정"},
    {"group": GROUPS[1], "name": "아이소토닉 보정", "kind": "calibration", "method": "isotonic",
     "reason": "훈련 부분 안쪽 3겹 교차검증으로 단조 증가 확률 보정"},
    {"group": GROUPS[2], "name": "사실 질문 단서 4개", "kind": "semantics", "flags": "factual",
     "reason": "질문에 최신성·출처·숫자·사람/장소 표현이 있는지 추가"},
    {"group": GROUPS[2], "name": "의미 단서 7개", "kind": "semantics", "flags": "full",
     "reason": "위 4개에 사실 확인·창작·설명 요청 표현 추가"},
]


def build_model(config):
    model = make_pipeline("로지스틱 회귀", with_clusters=True)
    if config["kind"] == "regularization":
        model.set_params(model__C=config["c"])
    elif config["kind"] == "calibration":
        return CalibratedClassifierCV(
            estimator=model, method=config["method"], cv=3, ensemble=True,
        )
    elif config["kind"] == "semantics":
        transformer = factual_flags if config["flags"] == "factual" else full_flags
        # Added within the existing fold-fitted ColumnTransformer.
        model.named_steps["features"].transformers.append(
            ("semantic", FunctionTransformer(transformer, validate=False), "question")
        )
    return model


def score(y, risk):
    predicted_yes = (risk >= 0.5).astype(int)
    tn, fp, fn, tp = (int(v) for v in confusion_matrix(
        y, predicted_yes, labels=[0, 1]
    ).ravel())
    return {
        "brier": float(brier_score_loss(y, risk)),
        "mae": float(mean_absolute_error(y, risk)),
        "r2": float(r2_score(y, risk)),
        "ap": float(average_precision_score(y, risk)),
        "recall_05": float(recall_score(y, predicted_yes, zero_division=0)),
        "precision_05": float(precision_score(y, predicted_yes, zero_division=0)),
        "f1_05": float(f1_score(y, predicted_yes, zero_division=0)),
        "confusion_05": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
    }


def probability(model, data):
    return np.asarray(model.predict_proba(data)[:, 1], dtype=float)


def validate_candidate(config, train, labels, folds):
    cv_scores = []
    for training_indices, validation_indices in folds:
        fitted = build_model(config)
        fitted.fit(train.iloc[training_indices], labels.iloc[training_indices])
        risk = probability(fitted, train.iloc[validation_indices])
        if np.any(~np.isfinite(risk)) or np.any((risk < 0) | (risk > 1)):
            raise ValueError("확률 예측값이 0~1 범위를 벗어났습니다.")
        cv_scores.append(score(labels.iloc[validation_indices], risk))
    return {
        "cv_brier": float(np.mean([s["brier"] for s in cv_scores])),
        "cv_brier_std": float(np.std([s["brier"] for s in cv_scores])),
        "cv_ap": float(np.mean([s["ap"] for s in cv_scores])),
        "cv_recall_05": float(np.mean([s["recall_05"] for s in cv_scores])),
        "cv_f1_05": float(np.mean([s["f1_05"] for s in cv_scores])),
    }


def run(raw):
    train, test = prepare_question_data(raw)
    y_train, y_test = train["y"].astype(int), test["y"].astype(int)
    folds = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED).split(train, y_train))
    candidates = []
    for config in EXPERIMENTS:
        result = {
            "group": config["group"], "name": config["name"],
            "reason": config["reason"], **validate_candidate(config, train, y_train, folds),
        }
        candidates.append(result)
        print(f"{result['name']}: CV Brier={result['cv_brier']:.6f}; PR-AUC={result['cv_ap']:.5f}", flush=True)

    original = candidates[0]
    selected = []
    # Test scoring follows all candidate CV assessments; the test labels are
    # excluded from choosing each group winner and the overall winner.
    for group in ("기존", *GROUPS):
        winner = min((c for c in candidates if c["group"] == group), key=lambda r: r["cv_brier"])
        config = next(c for c in EXPERIMENTS if c["name"] == winner["name"])
        fitted = build_model(config)
        fitted.fit(train, y_train)
        test_metrics = score(y_test, probability(fitted, test))
        selected.append({**winner, "test": test_metrics,
                         "cv_brier_reduction_vs_original": original["cv_brier"] - winner["cv_brier"]})

    best_by_cv = min(selected, key=lambda row: row["cv_brier"])
    base_probability = float(y_train.mean())
    return {
        "description": "추가 실험 3가지: 로지스틱 C, 내부 교차검증 확률 보정, 질문 글에서만 계산한 의미 단서",
        "seed": SEED, "cv_folds": 5, "calibration_inner_folds": 3,
        "train_count": int(len(train)), "test_count": int(len(test)),
        "train_yes": int(y_train.sum()), "test_yes": int(y_test.sum()),
        "with_clusters": True,
        "selection_metric": "학습 5겹 CV Brier 최저값; 시험 정답은 설정 선택에 사용하지 않음",
        "baseline_constant_probability": base_probability,
        "baseline_test_brier": float(brier_score_loss(y_test, np.full(len(test), base_probability))),
        "original": selected[0],
        "all_candidates": candidates,
        "selected_by_group": selected[1:],
        "best_by_train_cv": {"name": best_by_cv["name"], "group": best_by_cv["group"],
                             "cv_brier": best_by_cv["cv_brier"]},
        "semantic_feature_definitions": SEMANTIC_PATTERNS,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    results = run(pd.read_csv(args.csv))
    target = Path(args.out)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Train-CV winner:", results["best_by_train_cv"])


if __name__ == "__main__":
    main()
