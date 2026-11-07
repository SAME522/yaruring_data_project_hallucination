"""Explore HaluEval and compare numerical hallucination-risk estimates.

Run from the main/ directory: streamlit run main.py
"""

from __future__ import annotations

import io
import json

import numpy as np
import pandas as pd
import requests
import streamlit as st

from modeling import prepare_question_data
from probability_models import MODEL_KEYS, audition


SOURCE_URL = "https://raw.githubusercontent.com/RUCAIBox/HaluEval/main/data/general_data.json"
REPO_DATA = "https://raw.githubusercontent.com/SAME522/yaruring_data_project_hallucination/main/data"
CLUSTERED_CSV_URL = f"{REPO_DATA}/questions_clustered.csv"
RESULTS_CSV_URL = f"{REPO_DATA}/model_comparison_probability.csv"
RESULTS_JSON_URL = f"{REPO_DATA}/model_summary_probability.json"
PREDICTIONS_CSV_URL = f"{REPO_DATA}/model_predictions_probability.csv"

st.set_page_config(page_title="HaluEval 전처리와 위험도 예측", page_icon="🔎", layout="wide")


@st.cache_data(ttl=3600, show_spinner="데이터 다운로드 중...")
def download_bytes(url: str) -> bytes:
    if not url.startswith("https://"):
        raise ValueError("HTTPS 주소만 허용합니다.")
    response = requests.get(url, timeout=45)
    response.raise_for_status()
    return response.content


@st.cache_data(ttl=3600)
def fetch_original(url: str) -> pd.DataFrame:
    return pd.read_json(io.BytesIO(download_bytes(url)), lines=True, dtype=False)


@st.cache_data(ttl=3600)
def fetch_clustered() -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(download_bytes(CLUSTERED_CSV_URL)))


@st.cache_data(ttl=3600)
def fetch_saved_results() -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    table = pd.read_csv(io.BytesIO(download_bytes(RESULTS_CSV_URL)))
    summary = json.loads(download_bytes(RESULTS_JSON_URL).decode("utf-8"))
    predictions = pd.read_csv(io.BytesIO(download_bytes(PREDICTIONS_CSV_URL)))
    return table, summary, predictions


def enrich(df: pd.DataFrame) -> pd.DataFrame:
    """Preserve the previous data exploration and its five numerical features."""
    out = df.copy()
    texts = out["user_query"].fillna("").astype(str)
    tokens = texts.str.findall(r"[A-Za-z]+(?:'[A-Za-z]+)*")
    out["질문_글자수"] = texts.str.len()
    out["질문_단어수"] = tokens.str.len()
    out["고유_단어수"] = tokens.apply(lambda ws: len(set(w.lower() for w in ws)))
    out["평균_단어길이"] = tokens.apply(
        lambda ws: round(sum(len(w) for w in ws) / len(ws), 3) if ws else 0.0
    )
    out["물음표_수"] = texts.str.count(r"\?")
    if "hallucination" in out:
        label = out["hallucination"].astype("string").str.strip().str.lower()
        out["할루시네이션_발생"] = label.map({"yes": 1, "no": 0}).astype("Float64")
    return out


def show_preprocessing() -> None:
    st.title("HaluEval 데이터 탐색·전처리")
    st.caption("원본 데이터의 결측값, 수치형 특징, Yes/No 분포를 확인합니다.")
    url = st.text_input("HaluEval 원본 Raw 주소", value=SOURCE_URL)
    try:
        original = fetch_original(url)
    except Exception as exc:
        st.error(f"원본 데이터를 불러오지 못했습니다: {exc}")
        st.stop()
    if "user_query" not in original.columns:
        st.error("필수 열 user_query가 없습니다. 원본 HaluEval 데이터 주소인지 확인해 주세요.")
        st.stop()
    data = enrich(original)
    yes_count = int(data.get("할루시네이션_발생", pd.Series(dtype=float)).eq(1).sum())
    no_count = int(data.get("할루시네이션_발생", pd.Series(dtype=float)).eq(0).sum())

    a, b, c, d = st.columns(4)
    a.metric("원본 행", f"{len(original):,}")
    b.metric("원본 열", str(len(original.columns)))
    c.metric("생성한 수치형 특징", "5개")
    d.metric("Yes 비율", f"{yes_count / (yes_count + no_count):.2%}"
             if yes_count + no_count else "알 수 없음")

    tab1, tab2, tab3 = st.tabs(["원본·결측치", "수치형 변수·중앙값", "라벨·질문 비교"])
    with tab1:
        st.subheader("원본 열의 자료형과 결측치")
        overview = pd.DataFrame({
            "열 이름": original.columns,
            "자료형": [str(typ) for typ in original.dtypes],
            "결측치 수": original.isna().sum().to_numpy(),
            "결측 비율(%)": (original.isna().mean() * 100).round(2).to_numpy(),
            "고유 값 수": original.apply(
                lambda col: col.dropna().map(str).nunique()
            ).to_numpy(),
        })
        st.dataframe(overview, use_container_width=True, hide_index=True)
        st.write(f"중복 질문: {original['user_query'].duplicated().sum():,}개")
        st.caption("빈 문자열·빈 리스트는 결측치(NaN)와 구분합니다.")
        st.dataframe(original.head(30), use_container_width=True)
    with tab2:
        numeric = ["질문_글자수", "질문_단어수", "고유_단어수", "평균_단어길이", "물음표_수"]
        stats = data[numeric].agg(["count", "mean", "median", "std", "min", "max"]).T
        stats.columns = ["개수", "평균", "중앙값", "표준편차", "최솟값", "최댓값"]
        st.dataframe(stats.round(2), use_container_width=True)
        selected = st.selectbox("분포를 확인할 변수", numeric)
        hist = pd.cut(data[selected], bins=20, duplicates="drop").value_counts(sort=False)
        hist_table = hist.rename_axis("구간").reset_index(name="질문 수")
        hist_table["구간"] = hist_table["구간"].astype(str)
        st.bar_chart(hist_table.set_index("구간"))
        st.caption("수치형 열은 원본 문장으로부터 계산한 파생 변수입니다.")
        st.dataframe(data[["user_query"] + numeric].head(30), use_container_width=True)
    with tab3:
        if "hallucination" not in data.columns:
            st.warning("할루시네이션 라벨이 없습니다.")
        else:
            labels = data["hallucination"].astype("string").str.lower().str.strip().value_counts()
            st.bar_chart(labels.rename_axis("라벨").reset_index(name="질문 수").set_index("라벨"))
            st.write(f"Yes: {yes_count:,}개, No: {no_count:,}개")
            by_label = (data.loc[data["할루시네이션_발생"].notna()]
                        .groupby("hallucination")[["질문_글자수", "질문_단어수", "고유_단어수"]]
                        .median())
            st.subheader("라벨별 질문 특징 중앙값")
            st.dataframe(by_label, use_container_width=True)
            st.caption("Yes 비율은 데이터셋에 수록된 과거 응답에 한정됩니다.")
    st.download_button("전처리 데이터 CSV 다운로드",
                       data=data.to_csv(index=False).encode("utf-8-sig"),
                       file_name="halueval_preprocessed.csv", mime="text/csv")
    st.caption("원본 제공: RUCAIBox/HaluEval (MIT License)")


def confusion_from_scores(y_true: pd.Series, risk: pd.Series, threshold: float):
    actual_yes = y_true.astype(int).eq(1)
    predicted_yes = risk.astype(float).ge(threshold)
    tp = int((actual_yes & predicted_yes).sum())
    fp = int((~actual_yes & predicted_yes).sum())
    fn = int((actual_yes & ~predicted_yes).sum())
    tn = int((~actual_yes & ~predicted_yes).sum())
    matrix = pd.DataFrame(
        [[tn, fp], [fn, tp]],
        index=["실제 No", "실제 Yes"],
        columns=["예측 No", "예측 Yes"],
    )
    recall = tp / (tp + fn) if tp + fn else 0.0
    return matrix, recall, fp, fn


def show_models() -> None:
    st.title("할루시네이션 발생 가능성 수치 예측")
    st.info(
        "최종 목표는 질문으로부터 0~1 사이의 위험도 수치를 추정하는 것입니다. "
        "현재 정답이 Yes/No이므로 로지스틱 회귀와 의사결정트리는 확률을 출력하는 분류기로 학습하고, "
        "일반 선형 회귀는 비교용으로 사용합니다. 기존 질문·단어 군집은 입력 특징을 만드는 전처리 과정입니다."
    )
    with st.expander("비교하는 세 모델과 주의할 점", expanded=True):
        st.markdown(
            "- 선형 회귀: 특징의 가중합으로 0/1 정답을 예측합니다. 원래 값은 0~1을 벗어날 수 있어 "
            "위험도 표시와 확률 평가에만 범위 제한을 적용합니다.\n"
            "- 로지스틱 회귀: 시그모이드 함수를 이용해 0~1 사이의 Yes 확률을 계산합니다.\n"
            "- 의사결정트리: 질문 특징에 따른 분기 후 말단 노드의 Yes 비율을 확률로 사용합니다.\n"
            "\n세 모델 모두 동일한 질문 TF-IDF·수치형 특징·요청 유형을 사용하고, "
            "군집 ON에서만 질문군집 Q와 단어군집 W 비중을 추가합니다."
        )

    try:
        table, meta, predictions = fetch_saved_results()
    except Exception as exc:
        st.error(f"모델 평가 결과를 가져오지 못했습니다: {exc}")
        st.stop()
    if "probability_live_table" in st.session_state:
        table = st.session_state["probability_live_table"]
        meta = st.session_state["probability_live_meta"]
        predictions = st.session_state["probability_live_predictions"]
        st.success("현재 CSV로 다시 학습한 결과를 보여 줍니다.")

    a, b, c, d = st.columns(4)
    a.metric("학습 질문", f"{int(meta['train_count']):,}")
    b.metric("시험 질문", f"{int(meta['test_count']):,}")
    c.metric("학습 Yes 비율", f"{meta['baseline_train_yes_rate']:.1%}")
    d.metric("학습 데이터 교차검증", f"{int(meta['n_splits'])}겹")

    st.subheader("동일 조건에서 세 모델 비교")
    st.caption(
        "Brier 점수는 예측한 확률과 실제 0/1 정답의 차이를 측정하며 낮을수록 좋습니다. "
        "PR-AUC(AP)는 Yes 탐지 성능을 평가하며 높을수록 좋습니다. "
        "최종 후보는 시험 데이터가 아닌 학습 데이터의 교차검증 Brier 점수로 골랐습니다."
    )
    cols = [
        "모델", "특징", "CV Brier", "CV PR-AUC", "CV ROC-AUC",
        "시험 brier", "시험 average_precision", "시험 roc_auc",
        "시험 recall", "시험 f1", "시험 out_of_range_count",
    ]
    st.dataframe(table[cols].round(4), use_container_width=True, hide_index=True)
    scores = table[["모델", "특징", "CV Brier"]].copy()
    scores["모델 · 특징"] = scores["모델"] + " · " + scores["특징"]
    st.bar_chart(scores.set_index("모델 · 특징")[["CV Brier"]])

    st.subheader("군집 특징 ON/OFF 비교")
    model = st.selectbox(
        "비교할 모델", options=list(MODEL_KEYS), key="comparison_probability_model",
    )
    use_clusters = st.toggle(
        "군집 특징 사용 (ON = 군집 포함, OFF = 기본 특징만)",
        value=True,
        key="cluster_probability_switch",
    )
    feature = "기본 + 군집" if use_clusters else "기본 특징"
    subset = table.loc[table["모델"].eq(model)]
    chosen = subset.loc[subset["특징"].eq(feature)].iloc[0]
    other = subset.loc[subset["특징"].ne(feature)].iloc[0]
    st.caption("현재 선택한 설정과 반대 설정의 차이를 확인합니다. 스위치를 눌러도 모델을 재학습하지는 않습니다.")

    quick_metrics = [
        ("CV Brier ↓", "CV Brier", True),
        ("시험 Brier ↓", "시험 brier", True),
        ("시험 PR-AUC ↑", "시험 average_precision", False),
        ("시험 Yes 재현율(기준 0.5)", "시험 recall", False),
    ]
    for box, (label, field, lower_better) in zip(st.columns(4), quick_metrics):
        diff = float(chosen[field]) - float(other[field])
        box.metric(
            label, f"{float(chosen[field]):.4f}",
            delta=f"{diff:+.4f}",
            delta_color="inverse" if lower_better else "normal",
        )

    fields = [
        ("교차검증 Brier", "CV Brier"),
        ("교차검증 PR-AUC", "CV PR-AUC"),
        ("시험 Brier", "시험 brier"),
        ("시험 PR-AUC", "시험 average_precision"),
        ("시험 ROC-AUC", "시험 roc_auc"),
        ("시험 재현율 (기준 0.5)", "시험 recall"),
        ("시험 F1 (기준 0.5)", "시험 f1"),
        ("선형 회귀 원래 값의 범위 초과 건수", "시험 out_of_range_count"),
    ]
    on = subset.loc[subset["특징"].eq("기본 + 군집")].iloc[0]
    off = subset.loc[subset["특징"].eq("기본 특징")].iloc[0]
    comparison = pd.DataFrame([
        {
            "지표": label,
            "군집 ON": round(float(on[key]), 4),
            "군집 OFF": round(float(off[key]), 4),
            "ON − OFF": round(float(on[key]) - float(off[key]), 4),
        }
        for label, key in fields
    ])
    st.dataframe(comparison, use_container_width=True, hide_index=True)

    if model == "선형 회귀":
        st.warning(
            f"선형 회귀는 시험 질문 {len(predictions):,}개 중 군집 ON에서 "
            f"{int(on['시험 out_of_range_count']):,}개, OFF에서 "
            f"{int(off['시험 out_of_range_count']):,}개의 원래 예측값이 0~1 범위를 벗어났습니다. "
            "이를 0~1로 잘라 표시한 값이 바로 올바른 확률이라는 뜻은 아닙니다."
        )

    suffix = "on" if use_clusters else "off"
    alternative_suffix = "off" if use_clusters else "on"
    chosen_risk = predictions[f"{MODEL_KEYS[model]}_{suffix}"]
    other_risk = predictions[f"{MODEL_KEYS[model]}_{alternative_suffix}"]

    st.subheader("시험 질문으로 위험도 수치 확인")
    row_number = st.number_input(
        f"확인할 시험 질문 번호 (1~{len(predictions):,})",
        min_value=1, max_value=len(predictions), value=1, step=1,
    )
    sample = predictions.iloc[int(row_number) - 1]
    st.text_area("해당 질문", value=str(sample["question"]), height=100, disabled=True)
    p = float(chosen_risk.iloc[int(row_number) - 1])
    q = float(other_risk.iloc[int(row_number) - 1])
    left, right = st.columns(2)
    left.metric(f"현재 설정: {'군집 ON' if use_clusters else '군집 OFF'} 위험도",
                f"{p:.1%}", delta=f"{p-q:+.1%} (반대 설정 대비)")
    right.metric("실제로 기록된 정답", "Yes" if int(sample["y"]) == 1 else "No")
    st.caption(
        "위 예시는 이미 정답이 있는 시험 질문에 대한 예측입니다. 새 질문을 실시간으로 검사하려면 "
        "학습 때 사용한 군집 변환기를 별도로 보존·재사용해야 합니다."
    )

    st.subheader("경고 기준에 따른 결과 변화")
    threshold = st.slider(
        "위험 경고 기준", 0.05, 0.95, 0.20, step=0.05,
        help="기준보다 예측 위험도가 높으면 Yes 경고를 표시한다고 가정합니다.",
    )
    primary_matrix, primary_recall, primary_fp, primary_fn = confusion_from_scores(
        predictions["y"], chosen_risk, threshold
    )
    other_matrix, other_recall, other_fp, other_fn = confusion_from_scores(
        predictions["y"], other_risk, threshold
    )
    primary_col, other_col = st.columns(2)
    with primary_col:
        st.write(f"현재 설정 (재현율 {primary_recall:.1%}, FP {primary_fp}, FN {primary_fn})")
        st.dataframe(primary_matrix, use_container_width=True)
    with other_col:
        st.write(f"반대 설정 (재현율 {other_recall:.1%}, FP {other_fp}, FN {other_fn})")
        st.dataframe(other_matrix, use_container_width=True)
    st.caption(
        "이 슬라이더는 시험 데이터의 경고 횟수 변화를 관찰하는 용도입니다. "
        "최종 경고 기준을 선택할 때는 시험 정답이 아닌 학습 데이터 내부의 검증 결과를 사용해야 합니다."
    )

    st.subheader("확률 예측의 신뢰성")
    buckets = pd.cut(
        chosen_risk.astype(float),
        bins=np.linspace(0, 1, 11), include_lowest=True,
    )
    calibration = pd.DataFrame({
        "구간": buckets.astype(str),
        "예측": chosen_risk.astype(float),
        "실제": predictions["y"].astype(int),
    }).groupby("구간", sort=True, as_index=False).agg(
        질문_수=("실제", "size"),
        평균_예측_위험도=("예측", "mean"),
        실제_Yes_비율=("실제", "mean"),
    )
    st.dataframe(calibration.round(3), use_container_width=True, hide_index=True)
    st.caption(
        "예측한 위험도와 실제 Yes 비율이 비슷할수록 확률 추정이 잘 맞는 편입니다. "
        "표본이 적은 구간의 값은 불안정할 수 있습니다."
    )

    selected_model = meta["selected_by_train_cv"]
    st.subheader("교차검증으로 선정한 우선 모델")
    st.write(
        f"{selected_model['모델']} / {selected_model['특징']}. "
        f"5겹 CV Brier {selected_model['CV Brier']:.4f} "
        f"(낮을수록 좋음), CV PR-AUC {selected_model['CV PR-AUC']:.4f}."
    )
    baseline = meta["baseline"]
    st.write(
        f"기준선은 모든 질문에 학습 Yes 비율인 "
        f"{meta['baseline_train_yes_rate']:.1%}를 예측합니다. "
        f"기준선의 시험 Brier는 {baseline['brier']:.4f}입니다."
    )
    st.caption(
        "현재 데이터에서의 확률 추정이 실제 모든 AI·질문에서의 할루시네이션 발생 확률을 "
        "보장하지는 않습니다. HaluEval은 특정 응답의 Yes/No 기록을 이용한 데이터입니다."
    )

    with st.expander("동일한 데이터로 세 모델 다시 학습"):
        if st.button("다시 학습하고 비교하기"):
            try:
                with st.spinner("학습 데이터에서 5겹 교차검증 중입니다..."):
                    frame = fetch_clustered()
                    prepare_question_data(frame)
                    fresh_table, fresh_meta = audition(frame, n_splits=5)
                st.session_state["probability_live_table"] = fresh_table
                st.session_state["probability_live_predictions"] = fresh_meta["test_predictions"]
                st.session_state["probability_live_meta"] = {
                    key: value for key, value in fresh_meta.items()
                    if key not in {"fitted_pipelines", "test_predictions"}
                }
                st.rerun()
            except Exception as exc:
                st.error(f"재학습 실패: {exc}")

    st.download_button(
        "세 모델 비교 결과 다운로드 (CSV)",
        data=table.to_csv(index=False).encode("utf-8-sig"),
        file_name="halueval_probability_model_comparison.csv",
        mime="text/csv",
    )


menu = st.sidebar.radio("분석 메뉴", ["데이터 탐색·전처리", "모델 학습·위험도 평가"])
if menu == "데이터 탐색·전처리":
    show_preprocessing()
else:
    show_models()
