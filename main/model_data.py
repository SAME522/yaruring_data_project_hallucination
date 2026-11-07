"""Shared, read-only experiment results for the Streamlit lesson pages."""

from __future__ import annotations

import io
import json

import pandas as pd
import requests
import streamlit as st


BASE_URL = "https://raw.githubusercontent.com/SAME522/yaruring_data_project_hallucination/main/data"
REPO_URL = "https://github.com/SAME522/yaruring_data_project_hallucination"
MODEL_KEYS = {
    "선형 회귀": "linear",
    "로지스틱 회귀": "logistic",
    "의사결정트리": "tree",
}
ON = "기본 + 군집"
OFF = "기본 특징"


@st.cache_data(ttl=3600)
def load_results() -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    files = [
        "model_comparison_probability.csv",
        "model_summary_probability.json",
        "model_predictions_probability.csv",
    ]
    raw = {}
    for filename in files:
        response = requests.get(f"{BASE_URL}/{filename}", timeout=45)
        response.raise_for_status()
        raw[filename] = response.content
    comparison = pd.read_csv(io.BytesIO(raw[files[0]]))
    summary = json.loads(raw[files[1]].decode("utf-8"))
    predictions = pd.read_csv(io.BytesIO(raw[files[2]]))
    if len(comparison) != 6 or len(predictions) != int(summary["test_count"]):
        raise ValueError("모델 평가 파일의 행 수가 일치하지 않습니다.")
    if set(comparison["모델"]) != set(MODEL_KEYS):
        raise ValueError("최종 비교 모델 세 가지가 모두 기록되지 않았습니다.")
    if not comparison.groupby("모델")["특징"].apply(
        lambda values: set(values) == {ON, OFF}
    ).all():
        raise ValueError("각 모델의 군집 ON/OFF 결과가 필요합니다.")
    required = {"y", "question", *(
        f"{key}_{suffix}" for key in MODEL_KEYS.values() for suffix in ("on", "off")
    )}
    if not required.issubset(predictions.columns):
        raise ValueError("질문별 위험도 예측 결과에 필요한 열이 없습니다.")
    return comparison, summary, predictions


def get_row(table: pd.DataFrame, model: str, cluster_on: bool) -> pd.Series:
    features = ON if cluster_on else OFF
    selected = table.loc[table["모델"].eq(model) & table["특징"].eq(features)]
    if len(selected) != 1:
        raise ValueError(f"{model}의 군집 설정 결과가 정확히 하나여야 합니다.")
    return selected.iloc[0]


def risk_column(model: str, cluster_on: bool) -> str:
    return f"{MODEL_KEYS[model]}_{'on' if cluster_on else 'off'}"


def threshold_counts(y, risk, threshold: float) -> dict:
    actual_yes = pd.Series(y).astype(int).to_numpy() == 1
    predicted_yes = pd.Series(risk).astype(float).to_numpy() >= threshold
    tp = int((actual_yes & predicted_yes).sum())
    tn = int((~actual_yes & ~predicted_yes).sum())
    fp = int((~actual_yes & predicted_yes).sum())
    fn = int((actual_yes & ~predicted_yes).sum())
    return {
        "TP": tp, "TN": tn, "FP": fp, "FN": fn,
        "재현율": tp / (tp + fn) if tp + fn else 0.0,
        "정밀도": tp / (tp + fp) if tp + fp else 0.0,
    }


def confusion_table(counts: dict) -> pd.DataFrame:
    return pd.DataFrame(
        [
            [counts["TN"], counts["FP"]],
            [counts["FN"], counts["TP"]],
        ],
        index=["실제 No", "실제 Yes"],
        columns=["예측 No", "예측 Yes"],
    )
