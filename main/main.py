"""HaluEval exploration and preprocessing homepage.

Run from this directory with: streamlit run main.py
The sibling pages/ directory provides separate model-analysis screens.
"""

from __future__ import annotations

import io

import pandas as pd
import requests
import streamlit as st


SOURCE_URL = "https://raw.githubusercontent.com/RUCAIBox/HaluEval/main/data/general_data.json"
CLUSTER_URL = (
    "https://raw.githubusercontent.com/SAME522/yaruring_data_project_hallucination/"
    "main/data/questions_clustered.csv"
)

st.set_page_config(page_title="HaluEval 데이터 전처리", page_icon="🔎", layout="wide")


@st.cache_data(ttl=3600, show_spinner="HaluEval 데이터 불러오는 중...")
def read_source(url: str) -> pd.DataFrame:
    if not url.startswith("https://"):
        raise ValueError("HTTPS 데이터 주소만 사용할 수 있습니다.")
    response = requests.get(url, timeout=45)
    response.raise_for_status()
    return pd.read_json(io.BytesIO(response.content), lines=True, dtype=False)


@st.cache_data(ttl=3600)
def read_clustered() -> pd.DataFrame:
    response = requests.get(CLUSTER_URL, timeout=45)
    response.raise_for_status()
    return pd.read_csv(io.BytesIO(response.content))


def enrich(frame: pd.DataFrame) -> pd.DataFrame:
    """Create numerical question features, without changing the source data."""
    data = frame.copy()
    questions = data["user_query"].fillna("").astype(str)
    tokens = questions.str.findall(r"[A-Za-z]+(?:'[A-Za-z]+)*")
    data["질문_글자수"] = questions.str.len()
    data["질문_단어수"] = tokens.str.len()
    data["고유_단어수"] = tokens.apply(lambda words: len(set(w.lower() for w in words)))
    data["평균_단어길이"] = tokens.apply(
        lambda words: round(sum(map(len, words)) / len(words), 3) if words else 0.0
    )
    data["물음표_수"] = questions.str.count(r"\?")
    if "hallucination" in data:
        labels = data["hallucination"].astype("string").str.lower().str.strip()
        data["할루시네이션_발생"] = labels.map({"yes": 1, "no": 0}).astype("Float64")
    return data


st.title("HaluEval 원본 데이터와 전처리")
st.write(
    "이 화면에서는 데이터를 살펴보고 질문 길이, 단어 수 등의 숫자 특징을 만듭니다. "
    "왼쪽 페이지 메뉴에서 모델별 성적표, 군집 ON/OFF, 위험도·오분류 분석을 확인할 수 있습니다."
)
source_url = st.text_input("원본 데이터 주소", SOURCE_URL)
try:
    original = read_source(source_url)
except Exception as exc:
    st.error(f"원본 데이터 다운로드에 실패했습니다: {exc}")
    st.stop()

if "user_query" not in original.columns:
    st.error("원본 파일에 user_query 열이 없습니다.")
    st.stop()

data = enrich(original)
yes = int(data.get("할루시네이션_발생", pd.Series(dtype=float)).eq(1).sum())
no = int(data.get("할루시네이션_발생", pd.Series(dtype=float)).eq(0).sum())
one, two, three, four = st.columns(4)
one.metric("원본 질문 수", f"{len(original):,}")
two.metric("원본 열 수", len(original.columns))
three.metric("만든 숫자 특징", "5개")
four.metric("Yes 비율", f"{yes / (yes + no):.1%}" if yes + no else "알 수 없음")

tab1, tab2, tab3, tab4 = st.tabs([
    "원본·결측", "파생 특징·분포", "Yes/No 분포", "기존 군집 데이터",
])
with tab1:
    overview = pd.DataFrame({
        "열": original.columns,
        "자료형": [str(dtype) for dtype in original.dtypes],
        "결측 수": original.isna().sum().to_numpy(),
        "결측 비율(%)": (original.isna().mean() * 100).round(2).to_numpy(),
        "서로 다른 값 수": original.apply(
            lambda series: series.dropna().map(str).nunique()
        ).to_numpy(),
    })
    st.dataframe(overview, hide_index=True, use_container_width=True)
    st.write(f"중복 질문: {original['user_query'].duplicated().sum():,}개")
    st.caption("결측값과 빈 문자열은 서로 다르게 취급했습니다.")
    st.dataframe(original.head(30), use_container_width=True)

with tab2:
    numeric = ["질문_글자수", "질문_단어수", "고유_단어수", "평균_단어길이", "물음표_수"]
    statistics = data[numeric].agg(["count", "mean", "median", "std", "min", "max"]).T
    statistics.columns = ["개수", "평균", "중앙값", "표준편차", "최솟값", "최댓값"]
    st.dataframe(statistics.round(2), use_container_width=True)
    feature = st.selectbox("분포를 볼 특징", numeric)
    distribution = pd.cut(data[feature], bins=20, duplicates="drop").value_counts(sort=False)
    histogram = distribution.rename_axis("구간").reset_index(name="질문 수")
    histogram["구간"] = histogram["구간"].astype(str)
    st.bar_chart(histogram.set_index("구간"))
    st.dataframe(data[["user_query", *numeric]].head(30), use_container_width=True)

with tab3:
    if "hallucination" in data.columns:
        labels = data["hallucination"].astype("string").str.strip().str.lower().value_counts()
        st.bar_chart(labels.rename_axis("라벨").reset_index(name="질문 수").set_index("라벨"))
        st.write(f"Yes: {yes:,}개, No: {no:,}개")
        by_label = (
            data.loc[data["할루시네이션_발생"].notna()]
            .groupby("hallucination")[["질문_글자수", "질문_단어수", "고유_단어수"]]
            .median()
        )
        st.dataframe(by_label, use_container_width=True)
    else:
        st.warning("원본 데이터에 hallucination 라벨이 없습니다.")

with tab4:
    st.write(
        "기존 질문군집과 단어군집은 예측을 위한 추가 입력 특징으로 사용합니다. "
        "군집을 포함한 경우와 제외한 경우의 모델 성능은 별도 페이지에서 비교합니다."
    )
    if st.button("GitHub에 저장된 군집 CSV 읽기"):
        try:
            clustered = read_clustered()
            st.write(f"행 {len(clustered):,}개, 열 {len(clustered.columns)}개")
            st.dataframe(clustered.head(30), use_container_width=True)
            st.write("군집 데이터의 열", list(clustered.columns))
        except Exception as exc:
            st.error(f"군집 데이터 다운로드에 실패했습니다: {exc}")

st.download_button(
    "질문 특징을 추가한 데이터 다운로드 (CSV)",
    data=data.to_csv(index=False).encode("utf-8-sig"),
    file_name="halueval_preprocessed.csv",
    mime="text/csv",
)
st.caption("원본: RUCAIBox/HaluEval. 시험 결과는 이 페이지의 숫자 특징뿐 아니라 TF-IDF 등의 특징도 사용합니다.")
