"""Repeatable lesson-10 experiments with training-only model selection.

Run from main/: python model_improvement.py --csv ../data/questions_clustered.csv
The original fixed train/test split, seed and five CV folds are preserved.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score, brier_score_loss, confusion_matrix,
    f1_score, mean_absolute_error, precision_score, r2_score, recall_score,
)
from sklearn.model_selection import StratifiedKFold

from modeling import SEED, prepare_question_data
from probability_models import make_pipeline, predict_risk


ORIGINAL_LOGISTIC = "로지스틱 회귀"
ORIGINAL_TREE = "의사결정트리"

# The baseline parameters are identical to the existing 09 lesson models.
# Each family tests exactly one proposed direction; no test score enters tuning.
EXPERIMENTS = [
    {"group": "기존", "name": "기존 로지스틱", "model": ORIGINAL_LOGISTIC,
     "params": {}, "reason": "기존 TF-IDF 1200개, min_df=3, 1~2그램, 가중치 없음"},
    {"group": "기존", "name": "기존 트리", "model": ORIGINAL_TREE,
     "params": {}, "reason": "기존 깊이 5, 말단 최소 표본 40"},
    {"group": "A. 텍스트 전처리", "name": "A1 단어 범위 확대", "model": ORIGINAL_LOGISTIC,
     "params": {"features__tfidf__max_features": 2400, "features__tfidf__min_df": 2},
     "reason": "TF-IDF 단어 한도 1200→2400, 최소 빈도 3→2"},
    {"group": "A. 텍스트 전처리", "name": "A2 세 단어 조합 추가", "model": ORIGINAL_LOGISTIC,
     "params": {"features__tfidf__max_features": 2400, "features__tfidf__min_df": 2,
                "features__tfidf__ngram_range": (1, 3)},
     "reason": "단어 한도 2400, 최소 빈도 2, 1~3그램"},
    {"group": "A. 텍스트 전처리", "name": "A3 빈도 높은 단어 중심", "model": ORIGINAL_LOGISTIC,
     "params": {"features__tfidf__max_features": 600, "features__tfidf__min_df": 5},
     "reason": "단어 한도 600, 최소 빈도 5로 드문 표현 축소"},
    {"group": "B. Yes 가중치", "name": "B1 Yes 1.5배", "model": ORIGINAL_LOGISTIC,
     "params": {"model__class_weight": {0: 1, 1: 1.5}},
     "reason": "Yes 정답의 학습 가중치를 1.5배"},
    {"group": "B. Yes 가중치", "name": "B2 Yes 2배", "model": ORIGINAL_LOGISTIC,
     "params": {"model__class_weight": {0: 1, 1: 2}},
     "reason": "Yes 정답의 학습 가중치를 2배"},
    {"group": "B. Yes 가중치", "name": "B3 Yes 3배", "model": ORIGINAL_LOGISTIC,
     "params": {"model__class_weight": {0: 1, 1: 3}},
     "reason": "Yes 정답의 학습 가중치를 3배"},
    {"group": "C. 트리 설정", "name": "C1 간단한 트리", "model": ORIGINAL_TREE,
     "params": {"model__max_depth": 3, "model__min_samples_leaf": 80},
     "reason": "깊이 3, 말단 최소 표본 80로 가지치기 강화"},
    {"group": "C. 트리 설정", "name": "C2 중간 트리", "model": ORIGINAL_TREE,
     "params": {"model__max_depth": 4, "model__min_samples_leaf": 60},
     "reason": "깊이 4, 말단 최소 표본 60"},
    {"group": "C. 트리 설정", "name": "C3 깊은 트리", "model": ORIGINAL_TREE,
     "params": {"model__max_depth": 7, "model__min_samples_leaf": 20},
     "reason": "깊이 7, 말단 최소 표본 20로 세부 규칙 허용"},
]


def build_model(config):
    model = make_pipeline(config["model"], with_clusters=True)
    if config["params"]:
        model.set_params(**config["params"])
    return model


def cross_validate_config(config, train, y, folds):
    metrics = []
    for train_ids, val_ids in folds:
        candidate = build_model(config)
        candidate.fit(train.iloc[train_ids], y.iloc[train_ids])
        risk, outside = predict_risk(candidate, train.iloc[val_ids], config["model"])
        if outside:
            raise ValueError("로지스틱 회귀나 트리에서 확률 범위 이탈이 발생했습니다.")
        truth = y.iloc[val_ids]
        metrics.append({
            "brier": float(brier_score_loss(truth, risk)),
            "ap": float(average_precision_score(truth, risk)),
            "recall_05": float(recall_score(truth, risk >= 0.5, zero_division=0)),
        })
    return {
        "cv_brier": float(np.mean([m["brier"] for m in metrics])),
        "cv_brier_std": float(np.std([m["brier"] for m in metrics])),
        "cv_ap": float(np.mean([m["ap"] for m in metrics])),
        "cv_recall_05": float(np.mean([m["recall_05"] for m in metrics])),
    }


def run(raw: pd.DataFrame) -> dict:
    train, test = prepare_question_data(raw)
    y_train, y_test = train["y"].astype(int), test["y"].astype(int)
    splits = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED).split(train, y_train))
    rows = []
    for config in EXPERIMENTS:
        cv_result = cross_validate_config(config, train, y_train, splits)
        row = {"group": config["group"], "name": config["name"],
               "model": config["model"], "reason": config["reason"], **cv_result}
        rows.append(row)
        print(f"{config['name']}: CV Brier {row['cv_brier']:.6f}, AP {row['cv_ap']:.5f}", flush=True)

    original = {r["model"]: r for r in rows if r["group"] == "기존"}
    selected_by_group = []
    for group in ("A. 텍스트 전처리", "B. Yes 가중치", "C. 트리 설정"):
        group_rows = [r for r in rows if r["group"] == group]
        winner = min(group_rows, key=lambda r: r["cv_brier"])
        base = original[winner["model"]]
        winner["cv_brier_reduction_vs_original"] = base["cv_brier"] - winner["cv_brier"]
        winner["cv_ap_change_vs_original"] = winner["cv_ap"] - base["cv_ap"]

        config = next(c for c in EXPERIMENTS if c["name"] == winner["name"])
        refit = build_model(config)
        refit.fit(train, y_train)
        risk, _ = predict_risk(refit, test, config["model"])
        predicted_yes = (risk >= 0.5).astype(int)
        tn, fp, fn, tp = [int(n) for n in confusion_matrix(
            y_test, predicted_yes, labels=[0, 1]
        ).ravel()]
        winner["test_brier"] = float(brier_score_loss(y_test, risk))
        winner["test_mae"] = float(mean_absolute_error(y_test, risk))
        winner["test_r2"] = float(r2_score(y_test, risk))
        winner["test_ap"] = float(average_precision_score(y_test, risk))
        winner["test_recall_05"] = float(recall_score(y_test, predicted_yes, zero_division=0))
        winner["test_precision_05"] = float(precision_score(y_test, predicted_yes, zero_division=0))
        winner["test_f1_05"] = float(f1_score(y_test, predicted_yes, zero_division=0))
        winner["test_confusion_05"] = {"tn": tn, "fp": fp, "fn": fn, "tp": tp}
        selected_by_group.append(dict(winner))

    baseline_mean = float(y_train.mean())
    overall = min([original[ORIGINAL_LOGISTIC], *selected_by_group], key=lambda r: r["cv_brier"])
    return {
        "purpose": "세 가지 성능 개선 가설을 동일한 학습 데이터의 교차검증으로 평가",
        "seed": SEED, "cv_folds": 5,
        "train_count": len(train), "test_count": len(test),
        "train_yes_count": int(y_train.sum()), "test_yes_count": int(y_test.sum()),
        "all_use_cluster_features": True,
        "selection_metric": "cv_brier (낮을수록 우수)",
        "test_usage": "각 실험군의 승자는 학습 CV로만 선택한 뒤 고정 시험 데이터에서 1회 평가",
        "constant_probability_baseline_test_brier": float(
            brier_score_loss(y_test, np.full(len(test), baseline_mean))
        ),
        "original": original,
        "all_candidates": rows,
        "selected_by_group": selected_by_group,
        "overall_by_train_cv": {"name": overall["name"], "cv_brier": overall["cv_brier"],
                                "model": overall["model"]},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    raw = pd.read_csv(args.csv)
    summary = run(raw)
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Selected by CV: {summary['overall_by_train_cv']}")


if __name__ == "__main__":
    main()
