"""Lesson model audition: three candidates under the same conditions."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from model_data import OFF, ON, REPO_URL, get_row, load_results, risk_column


st.set_page_config(page_title="모델 후보와 기준선", page_icon="📋", layout="wide")
st.title("모델 후보·기준선 평가")
st.caption("세 모델의 성능을 같은 조건에서 비교하고, 수업의 평가 항목을 확인합니다.")

try:
    results, info, predictions = load_results()
except Exception as exc:
    st.error(f"모델 평가 자료를 불러올 수 없습니다: {exc}")
    st.stop()

st.subheader("우리 연구 문제는 무엇인가?")
st.write(
    "사용자 질문의 특징을 이용해 할루시네이션 Yes가 발생할 가능성을 "
    "0부터 1까지의 수치로 예측하고자 합니다. 현재 데이터에 주어진 정답은 Yes/No이므로 "
    "로지스틱 회귀와 의사결정트리 분류기를 이용해 확률을 추정할 수 있습니다. "
    "일반 선형 회귀는 연속값 예측을 적용했을 때 나타나는 한계를 비교하는 모델입니다. "
    "기존의 군집화는 최종 예측 모델에 입력할 질문의 특징을 만드는 전처리 단계입니다."
)

with st.expander("모델 세 가지의 원리와 일반적인 활용 사례", expanded=True):
    st.write(
        "선형 회귀는 여러 특징에 가중치를 곱해 더한 결과로 0 또는 1을 예측합니다. "
        "수치 사이의 관계를 알아볼 때 많이 사용하지만 0~1 범위를 벗어날 수 있어 "
        "이번 연구에서는 비교 기준으로만 사용합니다."
    )
    st.write(
        "로지스틱 회귀는 계산한 값을 시그모이드 함수에 넣어 Yes의 추정 확률을 구합니다. "
        "스팸 여부나 질병 발생 여부처럼 정답이 둘로 나뉘는 문제에 많이 사용합니다. "
        "질문이 텍스트이고 정답이 Yes/No인 우리 데이터와 잘 맞는 후보입니다."
    )
    st.write(
        "의사결정트리는 특징에 관한 질문을 반복하면서 데이터를 작은 집단으로 나눕니다. "
        "조건별 판단 과정을 이해하기 쉬워 고객 분류나 위험 판정처럼 "
        "결정 기준을 설명해야 하는 문제에 사용할 수 있습니다."
    )

left, middle, right = st.columns(3)
left.metric("학습 질문", f"{info['train_count']:,}")
middle.metric("시험 질문", f"{info['test_count']:,}")
right.metric("학습 내부 교차검증", f"{info['n_splits']}겹")
st.caption(f"동일한 학습·시험 분할과 특징 처리 방식, 난수값 {info['seed']}를 사용했습니다.")

st.subheader("모델 평가 조건")
cluster_on = st.toggle("군집 특징을 포함해 비교", value=True)
features = ON if cluster_on else OFF
exam = results.loc[results["특징"].eq(features)].copy().sort_values("CV Brier")
base = info["baseline"]

# All six trained setups and the constant-risk baseline share the same 902 labels.
# Binary y and a predicted risk in [0, 1] allow numerical error measurements.
y_test = predictions["y"].astype(int).to_numpy()
constant_probability = np.full(len(y_test), float(info["baseline_train_yes_rate"]))
baseline_r2 = float(r2_score(y_test, constant_probability))
baseline_mae = float(mean_absolute_error(y_test, constant_probability))
baseline_mse = float(mean_squared_error(y_test, constant_probability))

if not np.isclose(baseline_mse, float(base["brier"]), atol=1e-8):
    st.error("기준선의 원본 예측값과 저장된 Brier 점수가 일치하지 않습니다.")
    st.stop()

metric_rows = []
for _, record in results.iterrows():
    uses_clusters = record["특징"] == ON
    scores = predictions[risk_column(record["모델"], uses_clusters)].astype(float).to_numpy()
    r2 = float(r2_score(y_test, scores))
    mae = float(mean_absolute_error(y_test, scores))
    mse = float(mean_squared_error(y_test, scores))
    if not np.isclose(mse, float(record["시험 brier"]), atol=1e-6):
        st.error(f"{record['모델']}의 예측값과 저장된 평가 점수가 다릅니다.")
        st.stop()
    metric_rows.append({
        "모델": record["모델"],
        "특징": record["특징"],
        "R²": r2,
        "MAE": mae,
        "MSE": mse,
        "R² 차이": r2 - baseline_r2,
        "MAE 감소": baseline_mae - mae,
        "MSE 감소": baseline_mse - mse,
        "MSE 개선율(%)": 100 * (baseline_mse - mse) / baseline_mse,
    })
metric_table = pd.DataFrame(metric_rows)
current_metrics = metric_table.loc[metric_table["특징"].eq(features)].copy()
current_metrics = current_metrics.sort_values("MSE")
baseline_row = pd.DataFrame([{
    "모델": "평균 예측 베이스라인",
    "특징": "학습 Yes 비율",
    "R²": baseline_r2,
    "MAE": baseline_mae,
    "MSE": baseline_mse,
    "R² 차이": 0.0,
    "MAE 감소": 0.0,
    "MSE 감소": 0.0,
    "MSE 개선율(%)": 0.0,
}])
regression_view = pd.concat([baseline_row, current_metrics], ignore_index=True)

st.subheader("R²·MAE·MSE로 위험도 예측 평가")
st.write(
    f"학습 데이터의 Yes 비율 {info['baseline_train_yes_rate']:.1%}를 모든 질문에 "
    "똑같이 출력하는 평균 예측 모델을 회귀형 기준선으로 사용했습니다. "
    "시험 데이터 902개와 각 모델의 저장된 위험도 수치를 직접 비교해 계산했습니다."
)
with st.expander("세 지표의 계산 방법과 해석", expanded=True):
    st.latex(r"\mathrm{MAE}=\frac{1}{n}\sum_{i=1}^{n}|y_i-\hat{p}_i|")
    st.write("MAE는 실제 정답(0 또는 1)과 예측 위험도의 차이를 절댓값으로 나타낸 뒤 평균한 값입니다. 0에 가까울수록 좋습니다.")
    st.latex(r"\mathrm{MSE}=\frac{1}{n}\sum_{i=1}^{n}(y_i-\hat{p}_i)^2")
    st.write("MSE는 차이를 제곱해 평균한 값이라 큰 오차에 더 민감합니다. 이진 정답에 대한 확률 예측에서는 Brier 점수와 같은 값입니다.")
    st.latex(r"R^2=1-\frac{\sum_i(y_i-\hat{p}_i)^2}{\sum_i(y_i-\bar y)^2}")
    st.write("R²는 1이 가장 좋고, 0은 시험 데이터의 평균 정답값만 출력하는 기준과 비슷한 수준입니다. 음수이면 그 기준보다 제곱 오차가 큰 것입니다.")
    st.caption("로지스틱 회귀와 의사결정트리는 이진 분류 모델입니다. 여기의 회귀 지표는 모델이 출력한 Yes 확률을 연속값으로 보고 계산한 참고 지표입니다.")

display_columns = ["모델", "R²", "MAE", "MSE", "R² 차이", "MAE 감소", "MSE 감소", "MSE 개선율(%)"]
st.dataframe(regression_view[display_columns].round(4), hide_index=True, use_container_width=True)
st.caption("차이·감소 항목에서 양수는 베이스라인보다 개선되었다는 뜻이며, 음수는 악화되었다는 뜻입니다.")
st.bar_chart(regression_view.set_index("모델")[["MSE"]], use_container_width=True)

selected_model = info["selected_by_train_cv"]
selected_metrics = metric_table.loc[
    metric_table["모델"].eq(selected_model["모델"])
    & metric_table["특징"].eq(selected_model["특징"])
].iloc[0]
st.write(
    f"학습 데이터 교차검증으로 정한 우선 후보는 {selected_model['모델']} "
    f"({'군집 포함' if selected_model['특징'] == ON else '군집 제외'})입니다. "
    f"시험 R² {selected_metrics['R²']:.4f}, MAE {selected_metrics['MAE']:.4f}, "
    f"MSE {selected_metrics['MSE']:.4f}로, 기준선 대비 시험 MSE가 "
    f"{selected_metrics['MSE 개선율(%)']:.1f}% 낮아졌습니다."
)
st.write(
    "선형 회귀는 일부 큰 오차 때문에 MAE가 기준선보다 조금 낮더라도 MSE는 더 높았습니다. "
    "의사결정트리 역시 군집을 포함한 설정의 MSE가 기준선보다 높았습니다. "
    "따라서 지표를 하나만 보고 판단하지 않았습니다."
)
if cluster_on:
    st.caption(
        "일반 선형 회귀는 시험 질문 902개 중 229개의 원래 예측값이 0~1 범위를 벗어났습니다. "
        "저장된 위험도는 표시를 위해 0~1로 제한한 값이므로 선형 회귀의 확률을 그대로 신뢰할 수 없습니다."
    )

st.subheader("정확도·정밀도·재현율·F1로 Yes 탐지 평가")
classification_columns = [
    "모델", "시험 accuracy", "시험 precision", "시험 recall", "시험 f1", "시험 average_precision",
]
majority = pd.DataFrame([{
    "모델": "항상 No 베이스라인",
    "시험 accuracy": float(base["accuracy"]),
    "시험 precision": 0.0,
    "시험 recall": 0.0,
    "시험 f1": 0.0,
    "시험 average_precision": float(base["average_precision"]),
}])
classification_view = pd.concat([majority, exam[classification_columns]], ignore_index=True)
st.dataframe(classification_view.round(4), hide_index=True, use_container_width=True)
st.caption(
    "여기서 정밀도·재현율·F1·정확도는 위험도 0.5 이상을 Yes로 간주해 계산했습니다. "
    "PR-AUC(AP)는 경고 기준 하나에 묶이지 않은 별도의 분류 지표입니다."
)
st.write(
    "항상 No라고 답하는 기준선의 정확도는 높지만 실제 Yes를 한 건도 찾지 못합니다. "
    "로지스틱 회귀도 위험도 수치의 오차는 작지만 0.5 기준에서 Yes 재현율이 낮습니다. "
    "경고 기준을 바꾸면 재현율과 잘못된 경고 횟수가 어떻게 달라지는지는 '위험도·오분류 분석' 페이지에서 확인할 수 있습니다."
)

st.subheader("세 모델의 교차검증 성적표")
exam["기준선보다 낮은 Brier"] = (base["brier"] - exam["시험 brier"]).round(4)
exam["기준선보다 높은 PR-AUC"] = (
    exam["시험 average_precision"] - base["average_precision"]
).round(4)
cols = [
    "모델", "CV Brier", "CV PR-AUC", "CV ROC-AUC", "시험 brier",
    "시험 average_precision", "시험 roc_auc", "시험 recall", "시험 f1",
    "기준선보다 낮은 Brier", "기준선보다 높은 PR-AUC",
]
st.dataframe(exam[cols].round(4), hide_index=True, use_container_width=True)
st.caption(
    "Brier 점수는 낮을수록, PR-AUC·ROC-AUC·재현율·F1은 높을수록 좋습니다. "
    "Brier 개선 폭은 양수면 기준선보다 개선된 것이고 음수면 악화된 것입니다."
)

st.subheader("기준선이 예측하는 값")
st.write(
    f"확률 예측의 기준선은 모든 질문에 학습 데이터의 Yes 비율인 "
    f"{info['baseline_train_yes_rate']:.1%}를 똑같이 출력합니다. "
    f"기준선의 시험 Brier는 {base['brier']:.4f}, PR-AUC는 "
    f"{base['average_precision']:.4f}입니다. 이 점수를 넘어야 "
    "질문의 특징을 분석한 이점이 있다고 말할 수 있습니다."
)
st.write(
    f"반면 Yes/No를 직접 판정할 때의 기준선은 모든 질문을 No로 예측하며 "
    f"시험 정확도는 {base['accuracy']:.1%}입니다. 그러나 실제 Yes를 하나도 "
    "찾지 못해 재현율은 0%입니다. 정확도만으로 모델을 고르면 안 되는 이유입니다."
)

selected = info["selected_by_train_cv"]
chosen = get_row(results, selected["모델"], selected["특징"] == ON)
st.subheader("현재 선정한 모델")
st.write(
    f"학습 데이터의 교차검증 Brier 점수가 가장 낮은 "
    f"{selected['모델']} ({'군집 포함' if selected['특징'] == ON else '군집 제외'})를 "
    f"우선 후보로 정했습니다. CV Brier는 {selected['CV Brier']:.4f}이고 "
    f"시험 Brier는 {chosen['시험 brier']:.4f}입니다."
)
st.write(
    "로지스틱 회귀는 확률을 직접 추정하고 여러 단어 및 수치형 변수를 함께 사용할 수 있어 "
    "이 문제에 적합했습니다. 선형 회귀는 0~1 범위를 벗어나는 예측이 발생했고, "
    "의사결정트리는 이번 교차검증에서 Brier 점수가 더 높아 우선 후보로 선택하지 않았습니다."
)
st.warning(
    "로지스틱 회귀와 군집을 제외한 버전 사이의 Brier 차이는 작습니다. "
    "군집이 효과적이라고 확정하거나 새로운 프롬프트에서 같은 결과가 나온다고 보장할 수 없습니다."
)

st.subheader("수업 평가 기준 확인")
conditions = [
    "1. 문제 유형: 목표는 0~1 위험도 추정이며, 현재 정답은 Yes/No인 이진 분류 데이터입니다.",
    "2. 세 모델 비교: 선형 회귀·로지스틱 회귀·의사결정트리를 같은 분할, 특징, 난수에서 비교했습니다.",
    "3. 후보 간 차이: 위 표의 교차검증 및 시험 지표로 수치 차이를 확인할 수 있습니다.",
    "4. 선정 이유: 세 모델의 작동 원리와 일반적인 사용 사례, 로지스틱 회귀의 적합성을 설명했습니다.",
    "5. 기준선: 항상 No라는 분류 기준선과 학습 Yes 비율을 그대로 내는 확률 기준선을 확인했습니다.",
    "6. 개선 수치: 시험 R²·MAE·MSE, Brier와 PR-AUC가 기준선과 얼마나 다른지 계산했습니다.",
    "7. 기록: 비교 코드를 GitHub에 저장하고 09.md에 선택 및 제외 이유를 남겼습니다.",
]
for item in conditions:
    st.write(item)

st.caption("모델 평가는 이미 구한 시험 데이터 성적표를 표시합니다. 새 모델을 임의로 학습한 결과가 아닙니다.")
st.markdown(
    f"[비교 코드]({REPO_URL}/blob/main/main/probability_models.py) / "
    f"[연구 일지]({REPO_URL}/blob/main/log/09.md) / "
    f"[원본 데이터]({REPO_URL}/blob/main/data/README.md)"
)
st.download_button(
    "R²·MAE·MSE와 기준선 비교표 CSV 다운로드",
    data=pd.concat([baseline_row, metric_table], ignore_index=True).to_csv(index=False).encode("utf-8-sig"),
    file_name="halueval_regression_metric_comparison.csv",
    mime="text/csv",
)
st.download_button(
    "교차검증 및 분류 평가표 CSV 다운로드",
    data=exam.to_csv(index=False).encode("utf-8-sig"),
    file_name="halueval_model_audition.csv",
    mime="text/csv",
)
