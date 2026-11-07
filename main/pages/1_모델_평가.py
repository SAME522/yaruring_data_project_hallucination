"""Lesson model audition: three candidates under the same conditions."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from model_data import OFF, ON, REPO_URL, get_row, load_results


st.set_page_config(page_title="모델 후보와 기준선", page_icon="📋", layout="wide")
st.title("모델 후보·기준선 평가")
st.caption("세 모델의 성능을 같은 조건에서 비교하고, 수업의 평가 항목을 확인합니다.")

try:
    results, info, _predictions = load_results()
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

st.subheader("세 모델의 동일 조건 비교")
cluster_on = st.toggle("군집 특징을 포함해 비교", value=True)
features = ON if cluster_on else OFF
exam = results.loc[results["특징"].eq(features)].copy().sort_values("CV Brier")
base = info["baseline"]
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
    "6. 개선 수치: 각 후보의 시험 Brier와 PR-AUC가 기준선과 얼마나 다른지 계산했습니다.",
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
    "모델별 비교표 CSV 다운로드",
    data=exam.to_csv(index=False).encode("utf-8-sig"),
    file_name="model_evaluation.csv",
    mime="text/csv",
)
