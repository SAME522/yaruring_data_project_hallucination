"""Examine held-out predictions, calibration, and classification errors."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from model_data import (
    MODEL_KEYS, ON, confusion_table, get_row, load_results,
    risk_column, threshold_counts,
)


st.set_page_config(page_title="예측 위험도·오분류 분석", page_icon="🔍", layout="wide")
st.title("할루시네이션 위험도와 오분류 분석")
st.caption("같은 시험 질문 902개에 대해 모델의 확률 추정과 틀린 예측을 살펴봅니다.")

try:
    comparison, summary, predictions = load_results()
except Exception as exc:
    st.error(f"모델 시험 결과를 불러올 수 없습니다: {exc}")
    st.stop()

col_model, col_setting = st.columns([2, 1])
with col_model:
    model = st.selectbox("시험할 모델", list(MODEL_KEYS), index=1)
with col_setting:
    cluster_on = st.toggle("군집 특징 포함", value=True)

record = get_row(comparison, model, cluster_on)
risk = predictions[risk_column(model, cluster_on)].astype(float)
actual = predictions["y"].astype(int)
baseline_risk = float(summary["baseline_train_yes_rate"])
baseline_brier = float(summary["baseline"]["brier"])
model_brier = float(record["시험 brier"])

st.subheader("확률 평가와 기준선")
m1, m2, m3, m4 = st.columns(4)
m1.metric("현재 모델 Brier ↓", f"{model_brier:.4f}")
m2.metric("일정 확률 기준선 Brier", f"{baseline_brier:.4f}")
m3.metric(
    "기준선 대비 Brier 감소",
    f"{baseline_brier - model_brier:+.4f}",
    help="양수일수록 기준선보다 예측 확률의 평균 제곱 오차가 작습니다.",
)
m4.metric("시험 PR-AUC ↑", f"{float(record['시험 average_precision']):.4f}")
st.caption(
    f"기준선은 질문을 읽지 않고 학습 Yes 비율 {baseline_risk:.1%}를 모든 질문에 표시합니다. "
    "시험 데이터의 정답을 이용해 모델을 다시 학습하거나 최적의 경고 기준을 고르지는 않습니다."
)

st.subheader("예측 위험도와 실제 Yes 비율 비교")
bins = pd.cut(risk, bins=np.linspace(0.0, 1.0, 11), include_lowest=True)
calibration = pd.DataFrame({
    "구간": bins.astype(str),
    "예측 위험도": risk,
    "실제 Yes": actual,
}).groupby("구간", sort=True, observed=False).agg(
    질문_수=("실제 Yes", "size"),
    평균_예측=("예측 위험도", "mean"),
    실제_Yes_비율=("실제 Yes", "mean"),
).reset_index()
calibration = calibration.loc[calibration["질문_수"].gt(0)]
st.dataframe(calibration.round(4), use_container_width=True, hide_index=True)
st.bar_chart(
    calibration.set_index("구간")[["평균_예측", "실제_Yes_비율"]],
    use_container_width=True,
)
st.caption(
    "같은 위험도 구간에서 평균 예측값과 실제 Yes 비율이 비슷할수록 확률 보정이 잘된 편입니다. "
    "질문 수가 적은 구간은 결과가 불안정할 수 있습니다."
)

st.subheader("경고 기준을 바꿨을 때 놓치는 질문")
threshold = st.slider(
    "Yes 경고 기준 (시험 결과를 살펴보기 위한 설정)",
    min_value=0.05, max_value=0.95, value=0.20, step=0.05,
)
counts = threshold_counts(actual, risk, threshold)
left, right = st.columns(2)
with left:
    st.dataframe(confusion_table(counts), use_container_width=True)
with right:
    st.metric("Yes 재현율", f"{counts['재현율']:.1%}")
    st.metric("Yes 정밀도", f"{counts['정밀도']:.1%}")
    st.write(f"잘못 경고한 No(FP) {counts['FP']}개, 놓친 Yes(FN) {counts['FN']}개")
st.warning(
    "경고 기준 0.2는 비교를 돕기 위한 임시 값입니다. "
    "실제 서비스의 경고 기준은 별도로 학습 데이터의 교차검증을 통해 정해야 합니다."
)

st.subheader("어떤 질문을 틀렸을까?")
error_case = st.selectbox(
    "확인할 사례", ["놓친 Yes (FN)", "잘못 경고한 No (FP)", "전체 시험 질문"],
)
predicted_yes = risk.ge(threshold)
view = predictions[["source_row", "question", "y"]].copy()
view["예측 위험도"] = risk
view["경고 여부"] = predicted_yes.map({True: "Yes", False: "No"})
view["실제 정답"] = actual.map({1: "Yes", 0: "No"})
if error_case == "놓친 Yes (FN)":
    view = view.loc[actual.eq(1) & ~predicted_yes]
elif error_case == "잘못 경고한 No (FP)":
    view = view.loc[actual.eq(0) & predicted_yes]
view = view.sort_values("예측 위험도", ascending=False)
st.write(f"조건에 해당하는 질문: {len(view):,}개")
st.dataframe(
    view.rename(columns={"source_row": "원본 행", "question": "사용자 질문"}),
    hide_index=True, use_container_width=True,
    column_config={"예측 위험도": st.column_config.NumberColumn(format="%.3f")},
)

st.subheader("시험 질문 하나씩 보기")
question_number = st.number_input(
    "시험 질문 번호", min_value=1, max_value=len(predictions), value=1, step=1,
)
picked = predictions.iloc[int(question_number) - 1]
st.text_area("사용자 질문", value=str(picked["question"]), height=100, disabled=True)
other_column = risk_column(model, not cluster_on)
a, b, c = st.columns(3)
a.metric("현재 위험도", f"{float(picked[risk_column(model, cluster_on)]):.1%}")
b.metric("반대 군집 설정", f"{float(picked[other_column]):.1%}")
c.metric("실제 정답", "Yes" if int(picked["y"]) else "No")
st.caption(
    "여기의 질문은 이미 정답이 알려진 시험 데이터입니다. "
    "새 프롬프트를 실시간 판정하려면 학습된 전처리 변환기와 모델을 따로 저장해야 합니다. "
    "HaluEval에서 얻은 위험도는 다른 AI의 실제 할루시네이션 확률을 보장하지 않습니다."
)

st.download_button(
    "현재 화면의 질문 분석 CSV 다운로드",
    data=view.to_csv(index=False).encode("utf-8-sig"),
    file_name="halueval_model_error_inspection.csv",
    mime="text/csv",
)
