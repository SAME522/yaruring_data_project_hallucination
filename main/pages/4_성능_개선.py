"""Lesson 10: compare three pre-declared improvement experiments fairly."""

from __future__ import annotations

import json

import pandas as pd
import requests
import streamlit as st
from sklearn.metrics import mean_absolute_error, r2_score

from model_data import REPO_URL, get_row, load_results, risk_column


RESULT_URL = (
    "https://raw.githubusercontent.com/SAME522/yaruring_data_project_hallucination/"
    "main/data/improvement_summary.json"
)

st.set_page_config(page_title="성능 개선 실험", page_icon="🛠️", layout="wide")
st.title("성능 개선 실험")
st.write(
    "9차시의 모델을 기준으로 TF-IDF 특징 변경, Yes 가중치 조정, "
    "의사결정트리 설정 변경을 각각 실행한 결과입니다. "
    "새로운 데이터를 추가하지 않고 기존의 학습·시험 데이터 분할을 그대로 사용했습니다."
)


@st.cache_data(ttl=3600)
def load_improvements():
    response = requests.get(RESULT_URL, timeout=45)
    response.raise_for_status()
    return json.loads(response.content.decode("utf-8"))


try:
    improvements = load_improvements()
    previous, previous_info, previous_predictions = load_results()
except Exception as exc:
    st.error(f"실험 결과 자료를 불러오지 못했습니다: {exc}")
    st.stop()

if (int(improvements["train_count"]) != int(previous_info["train_count"])
        or int(improvements["test_count"]) != int(previous_info["test_count"])):
    st.error("실험 결과의 학습·시험 질문 수가 기존 모델과 일치하지 않습니다.")
    st.stop()

a, b, c, d = st.columns(4)
a.metric("학습 질문", f"{improvements['train_count']:,}")
b.metric("시험 질문", f"{improvements['test_count']:,}")
c.metric("5겹 교차검증", f"{improvements['cv_folds']}겹")
d.metric("난수값", improvements["seed"])
st.caption(
    "군집 특징 ON을 고정했습니다. 각 실험의 후보는 학습 데이터 교차검증 Brier가 가장 낮은 것으로 "
    "결정한 뒤 동일한 시험 데이터에서 평가했습니다. 시험 점수로 설정을 고르지 않았습니다."
)

rows = []
for winner in improvements["selected_by_group"]:
    model = winner["model"]
    previous_row = get_row(previous, model, True)
    previous_cv = float(previous_row["CV Brier"])
    previous_test = float(previous_row["시험 brier"])
    saved_risk = previous_predictions[risk_column(model, True)].astype(float)
    saved_label = previous_predictions["y"].astype(int)
    rows.append({
        "개선 방법": winner["group"],
        "교차검증으로 선택한 설정": winner["name"],
        "기존 CV Brier": previous_cv,
        "새 CV Brier": float(winner["cv_brier"]),
        "CV Brier 감소": previous_cv - float(winner["cv_brier"]),
        "기존 시험 Brier": previous_test,
        "새 시험 Brier": float(winner["test_brier"]),
        "시험 Brier 감소": previous_test - float(winner["test_brier"]),
        "기존 시험 R²": float(r2_score(saved_label, saved_risk)),
        "새 시험 R²": float(winner["test_r2"]),
        "기존 시험 MAE": float(mean_absolute_error(saved_label, saved_risk)),
        "새 시험 MAE": float(winner["test_mae"]),
        "기존 시험 재현율": float(previous_row["시험 recall"]),
        "새 시험 재현율": float(winner["test_recall_05"]),
        "기존 시험 정밀도": float(previous_row["시험 precision"]),
        "새 시험 정밀도": float(winner["test_precision_05"]),
        "기존 시험 F1": float(previous_row["시험 f1"]),
        "새 시험 F1": float(winner["test_f1_05"]),
        "기존 시험 FP": int(previous_row["시험 fp"]),
        "새 시험 FP": int(winner["test_confusion_05"]["fp"]),
        "기존 시험 TP": int(previous_row["시험 tp"]),
        "새 시험 TP": int(winner["test_confusion_05"]["tp"]),
        "기존 시험 PR-AUC": float(previous_row["시험 average_precision"]),
        "새 시험 PR-AUC": float(winner["test_ap"]),
    })
summary_table = pd.DataFrame(rows)

st.subheader("기존 모델과 성능 비교")
st.write(
    "Brier는 예측한 Yes 확률과 실제 Yes/No의 제곱 오차로, 낮을수록 좋습니다. "
    "표의 Brier 감소는 양수이면 개선, 음수이면 악화된 것입니다."
)
probability_tab, warning_tab = st.tabs(["확률 예측 성능", "Yes 탐지 성능(기준 0.5)"])
with probability_tab:
    st.dataframe(summary_table[[
        "개선 방법", "교차검증으로 선택한 설정", "기존 CV Brier", "새 CV Brier",
        "CV Brier 감소", "기존 시험 Brier", "새 시험 Brier", "시험 Brier 감소",
        "기존 시험 PR-AUC", "새 시험 PR-AUC",
    ]].round(5), hide_index=True, use_container_width=True)
    st.write("MSE는 0/1 정답에 대한 확률 예측에서 Brier와 같으므로 같은 값을 사용합니다.")
    st.dataframe(summary_table[[
        "개선 방법", "기존 시험 R²", "새 시험 R²",
        "기존 시험 MAE", "새 시험 MAE", "기존 시험 Brier", "새 시험 Brier",
    ]].rename(columns={"기존 시험 Brier": "기존 시험 MSE", "새 시험 Brier": "새 시험 MSE"}).round(5),
        hide_index=True, use_container_width=True)
with warning_tab:
    st.dataframe(summary_table[[
        "개선 방법", "기존 시험 재현율", "새 시험 재현율",
        "기존 시험 정밀도", "새 시험 정밀도", "기존 시험 F1", "새 시험 F1",
        "기존 시험 TP", "새 시험 TP", "기존 시험 FP", "새 시험 FP",
    ]].round(4), hide_index=True, use_container_width=True)

previous_logistic = get_row(previous, "로지스틱 회귀", True)
st.write(
    f"기존 로지스틱 회귀의 시험 Brier는 {previous_logistic['시험 brier']:.5f}이고, "
    f"일정 확률만 출력하는 평균 모델은 {previous_info['baseline']['brier']:.5f}입니다."
)
for column, row in zip(st.columns(3), rows):
    with column:
        st.metric(
            row["개선 방법"],
            f"{row['새 시험 Brier']:.5f}",
            delta=f"{row['시험 Brier 감소']:+.5f} (기존 대비 감소)",
            delta_color="normal",
        )
st.caption("위 변화량은 개선량(기존 점수 − 새 점수)이므로 양수가 좋은 결과입니다.")

st.subheader("각 실험에서 실제로 무엇을 바꿨을까?")
groups = ["A. 텍스트 전처리", "B. Yes 가중치", "C. 트리 설정"]
tabs = st.tabs(groups)
for tab, group in zip(tabs, groups):
    with tab:
        candidates = pd.DataFrame([
            {
                "실험 설정": r["name"], "바꾼 점": r["reason"],
                "CV Brier": r["cv_brier"], "CV Brier 표준편차": r["cv_brier_std"],
                "CV PR-AUC": r["cv_ap"], "CV 재현율(0.5 기준)": r["cv_recall_05"],
            }
            for r in improvements["all_candidates"] if r["group"] == group
        ]).sort_values("CV Brier")
        st.dataframe(candidates.round(5), hide_index=True, use_container_width=True)
        st.bar_chart(candidates.set_index("실험 설정")[["CV Brier"]])
        selected = next(x for x in improvements["selected_by_group"] if x["group"] == group)
        st.write(f"학습 교차검증으로 선택한 설정: {selected['name']}. {selected['reason']}.")

st.subheader("결과 해석")
a_result, b_result, c_result = rows
st.write(
    f"A 실험은 특징 추출에 사용하는 단어 수와 단어 조합을 바꾸었고, "
    f"CV Brier가 {a_result['CV Brier 감소']:+.6f}만큼 낮아졌습니다. "
    f"하지만 시험 Brier 감소는 {a_result['시험 Brier 감소']:+.6f}이어서 "
    "같은 개선이 시험 데이터에서도 나타나지는 않았습니다."
)
st.write(
    f"B 실험은 Yes의 학습 가중치를 높였습니다. 시험 재현율은 "
    f"{b_result['기존 시험 재현율']:.1%}에서 {b_result['새 시험 재현율']:.1%}로 "
    f"높아졌지만 시험 Brier 개선량은 {b_result['시험 Brier 감소']:+.6f}이었습니다. "
    f"잘못된 경고(FP)는 {b_result['기존 시험 FP']}개에서 "
    f"{b_result['새 시험 FP']}개로 바뀌었습니다. "
    "Brier 개선량이 음수이므로 확률 오차는 악화된 것입니다."
)
st.write(
    f"C 실험은 트리를 더 단순하게 만들어 과적합을 줄이려고 했습니다. "
    f"CV Brier는 {c_result['CV Brier 감소']:+.6f}, 시험 Brier는 "
    f"{c_result['시험 Brier 감소']:+.6f}만큼 낮아졌습니다. "
    "그러나 개선된 트리도 평균 예측 기준선보다 시험 Brier가 높습니다."
)
st.info(
    "현재로서는 기존 로지스틱 회귀를 안정적인 비교 기준으로 유지하겠습니다. "
    "A 실험은 학습 교차검증에서 작은 개선을 보였지만 시험 성능 향상까지 확인되지 않았습니다. "
    "B 실험은 탐지 목적에서만 일부 이점이 있으며, C 실험은 트리 성능을 개선했지만 "
    "로지스틱 회귀보다 확률 예측 성능이 낮습니다."
)
st.caption(
    "시험 데이터가 고정된 동일한 HaluEval 응답에 관한 결과입니다. "
    "새로운 질문이나 다른 AI 서비스에서도 같은 성능을 보인다는 뜻은 아닙니다. "
    "이번 실험에서는 시험 정답으로 가중치나 모델 구조를 다시 선택하지 않았습니다."
)

st.download_button(
    "세 개선 방법 비교표 CSV 다운로드",
    data=summary_table.to_csv(index=False).encode("utf-8-sig"),
    file_name="halueval_improvement_comparison.csv",
    mime="text/csv",
)
st.markdown(
    f"[개선 실험 코드]({REPO_URL}/blob/main/main/model_improvement.py) / "
    f"[실험 결과 원본]({REPO_URL}/blob/main/data/improvement_summary.json) / "
    f"[이전 모델 평가 결과]({REPO_URL}/blob/main/data/model_comparison_probability.csv)"
)
