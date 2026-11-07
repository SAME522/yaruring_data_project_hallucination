"""Experiment: does adding the previously learned clusters help prediction?"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from model_data import (
    MODEL_KEYS, OFF, ON, confusion_table, get_row, load_results,
    risk_column, threshold_counts,
)


st.set_page_config(page_title="군집 ON/OFF 실험", page_icon="🔀", layout="wide")
st.title("군집 특징을 넣으면 예측 성능이 달라질까?")
st.write(
    "기본적인 질문 길이, 단어 수, TF-IDF 및 요청 유형은 그대로 유지하고 "
    "질문군집 Q와 단어군집 W 비중을 추가했을 때의 차이를 비교합니다. "
    "군집화는 최종 예측 모델에 입력할 특징을 만든 전처리 과정입니다."
)

try:
    results, _summary, predictions = load_results()
except Exception as exc:
    st.error(f"모델 평가 자료를 불러올 수 없습니다: {exc}")
    st.stop()

model = st.selectbox("예측 모델", list(MODEL_KEYS))
use_clusters = st.toggle(
    "군집 특징 사용 (ON은 군집 포함, OFF는 기본 특징만)",
    value=True,
)
current = get_row(results, model, use_clusters)
other = get_row(results, model, not use_clusters)
on = get_row(results, model, True)
off = get_row(results, model, False)
st.caption(
    "두 설정 모두 같은 3,605개 질문으로 학습하고 같은 902개 질문으로 시험했습니다. "
    "ON/OFF를 바꾸면 해당 설정의 저장된 결과가 표시됩니다."
)

metrics = [
    ("CV Brier", "CV Brier", True),
    ("시험 Brier", "시험 brier", True),
    ("시험 PR-AUC", "시험 average_precision", False),
    ("시험 ROC-AUC", "시험 roc_auc", False),
]
for col, (label, key, lower_is_better) in zip(st.columns(4), metrics):
    difference = float(current[key]) - float(other[key])
    col.metric(
        label,
        f"{current[key]:.4f}",
        delta=f"{difference:+.4f}",
        delta_color="inverse" if lower_is_better else "normal",
    )

st.subheader("ON/OFF 결과를 나란히 비교")
fields = [
    ("교차검증 Brier", "CV Brier"),
    ("교차검증 PR-AUC", "CV PR-AUC"),
    ("시험 Brier", "시험 brier"),
    ("시험 PR-AUC", "시험 average_precision"),
    ("시험 ROC-AUC", "시험 roc_auc"),
    ("시험 정확도 (경고 기준 0.5)", "시험 accuracy"),
    ("시험 Yes 재현율 (경고 기준 0.5)", "시험 recall"),
    ("시험 F1 (경고 기준 0.5)", "시험 f1"),
    ("원래 예측값의 0~1 범위 이탈 개수", "시험 out_of_range_count"),
]
comparison = pd.DataFrame([
    {
        "지표": title,
        "군집 ON": round(float(on[key]), 4),
        "군집 OFF": round(float(off[key]), 4),
        "ON - OFF": round(float(on[key]) - float(off[key]), 4),
    }
    for title, key in fields
])
st.dataframe(comparison, hide_index=True, use_container_width=True)
st.caption(
    "Brier가 내려가면 예측 확률의 오차가 줄어들었다는 뜻입니다. "
    "군집을 추가해 일부 지표가 개선되어도 모든 지표에서 좋아지는 것은 아닙니다."
)

st.subheader("경고 기준을 바꾸면 무엇이 달라질까?")
threshold = st.slider(
    "Yes 경고를 출력할 위험도 기준", min_value=0.05, max_value=0.95,
    value=0.20, step=0.05,
)
current_scores = predictions[risk_column(model, use_clusters)]
other_scores = predictions[risk_column(model, not use_clusters)]
a = threshold_counts(predictions["y"], current_scores, threshold)
b = threshold_counts(predictions["y"], other_scores, threshold)
left, right = st.columns(2)
with left:
    st.write(f"현재 설정: {'군집 ON' if use_clusters else '군집 OFF'}")
    st.metric("실제 Yes 재현율", f"{a['재현율']:.1%}")
    st.write(f"잘못된 경고(FP) {a['FP']}개, 놓친 Yes(FN) {a['FN']}개")
    st.dataframe(confusion_table(a), use_container_width=True)
with right:
    st.write(f"비교 설정: {'군집 OFF' if use_clusters else '군집 ON'}")
    st.metric("실제 Yes 재현율", f"{b['재현율']:.1%}")
    st.write(f"잘못된 경고(FP) {b['FP']}개, 놓친 Yes(FN) {b['FN']}개")
    st.dataframe(confusion_table(b), use_container_width=True)
st.warning(
    "이 슬라이더는 결과를 이해하기 위한 시험 데이터 실험입니다. "
    "시험 정답을 보고 최종 경고 기준을 골라서는 안 되며, "
    "그 기준은 학습 데이터 내부 검증에서 정해야 합니다."
)

if model == "선형 회귀":
    st.write(
        f"선형 회귀는 군집 ON에서 {int(on['시험 out_of_range_count'])}개, "
        f"OFF에서 {int(off['시험 out_of_range_count'])}개의 원래 예측값이 "
        "0~1 범위를 벗어났습니다. 위험도 표시에는 이를 0~1로 제한했지만 "
        "선형 회귀가 올바른 확률을 보장한다는 뜻은 아닙니다."
    )

st.subheader("해석")
on_brier = float(on["CV Brier"])
off_brier = float(off["CV Brier"])
change = off_brier - on_brier
if change > 0:
    st.write(
        f"{model}은 학습 데이터 교차검증에서 군집을 포함했을 때 "
        f"Brier가 {change:.4f} 낮아졌습니다. 다만 차이가 작은 만큼 "
        "추가 검증 없이 군집의 효과가 확실하다고 단정할 수는 없습니다."
    )
elif change < 0:
    st.write(
        f"{model}은 학습 데이터 교차검증에서 군집을 포함했을 때 "
        f"Brier가 {-change:.4f} 높아졌습니다. 이번 설정에서는 군집의 추가가 "
        "확률 예측 오차를 개선하지 못했습니다."
    )
else:
    st.write(f"{model}에서 군집 ON/OFF의 교차검증 Brier 점수는 같았습니다.")
