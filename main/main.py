"""HaluEval 1.0: preprocessing exploration and fair model comparison.

Deploy from this directory with: streamlit run main.py
"""

from __future__ import annotations

import io

import pandas as pd
import requests
import streamlit as st

from modeling import audition, prepare_question_data


SOURCE_URL = "https://raw.githubusercontent.com/RUCAIBox/HaluEval/main/data/general_data.json"
REPO_DATA = "https://raw.githubusercontent.com/SAME522/yaruring_data_project_hallucination/main/data"
CLUSTERED_CSV_URL = f"{REPO_DATA}/questions_clustered.csv"
RESULTS_CSV_URL = f"{REPO_DATA}/model_comparison_09.csv"
RESULTS_JSON_URL = f"{REPO_DATA}/model_summary_09.json"

st.set_page_config(page_title="HaluEval 데이터 분석과 모델 비교", page_icon="🔎", layout="wide")


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
def fetch_saved_results() -> tuple[pd.DataFrame, dict]:
    import json

    table = pd.read_csv(io.BytesIO(download_bytes(RESULTS_CSV_URL)))
    meta = json.loads(download_bytes(RESULTS_JSON_URL).decode("utf-8"))
    return table, meta


def enrich(df: pd.DataFrame) -> pd.DataFrame:
    """Preserve the original EDA preprocessing and its five numeric features."""
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
    st.title("🔎 HaluEval 1.0 데이터 탐색·전처리")
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
        st.write(f"중복 질문: **{original['user_query'].duplicated().sum():,}개**")
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
        hist_table["구간"] = hist_table["구간"].astype(str)  # Altair Interval fix
        st.bar_chart(hist_table.set_index("구간"))
        st.caption("수치형 열은 원본 문장으로부터 계산한 파생 변수입니다.")
        st.dataframe(data[["user_query"] + numeric].head(30), use_container_width=True)
    with tab3:
        if "hallucination" not in data.columns:
            st.warning("할루시네이션 라벨이 없습니다.")
        else:
            labels = data["hallucination"].astype("string").str.lower().str.strip().value_counts()
            st.bar_chart(labels.rename_axis("라벨").reset_index(name="질문 수").set_index("라벨"))
            st.write(f"Yes: **{yes_count:,}개**, No: **{no_count:,}개**")
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


def show_models() -> None:
    st.title("🧪 할루시네이션 Yes/No 분류 모델 비교")
    st.info("**문제 유형: 지도학습의 이진 분류.** 정답이 Yes/No 두 가지이기 때문입니다. "
            "로지스틱 회귀는 이름에 '회귀'가 들어가지만, 여기서는 Yes 확률을 구하는 분류 모델입니다. "
            "앞서 진행한 군집화는 정답을 사용하지 않는 비지도학습으로서 입력 특징을 만드는 역할을 합니다.")

    with st.expander("모델 세 가지의 동작 원리", expanded=True):
        st.markdown("""
- **로지스틱 회귀:** 여러 특징의 가중합을 시그모이드 함수에 넣어 Yes 점수를 계산합니다.
- **kNN:** 새로운 질문과 가까운 학습 질문 21개를 찾아 거리 가중 투표를 합니다.
- **랜덤 포레스트:** 특징을 나누는 결정트리 100개를 학습하고 그 결과를 종합합니다.

세 모델 모두 **동일한 질문 텍스트 TF-IDF + 질문 길이·단어 수·요청 유형**을 기본으로 사용하며,
추가 실험에서 **기존 Q군집 및 W단어군집 비중**을 포함해 군집화의 효과를 확인합니다.
""")

    try:
        table, meta = fetch_saved_results()
    except Exception as exc:
        st.error(f"저장된 모델 실험 결과를 불러오지 못했습니다: {exc}")
        st.info("GitHub data/ 폴더에 모델 비교 CSV와 JSON이 있는지 확인해 주세요.")
        st.stop()

    a, b, c, d = st.columns(4)
    a.metric("학습 질문", f"{int(meta['train_count']):,}")
    b.metric("시험 질문", f"{int(meta['test_count']):,}")
    c.metric("학습 Yes", f"{int(meta['train_yes']):,}")
    d.metric("학습 내부 검증", f"{int(meta['n_splits'])}겹")

    st.subheader("모델 오디션: 동일한 조건에서 3개 모델 × 2개 특징 조합")
    st.caption("**선정 기준:** 학습 데이터의 교차검증 PR-AUC(Yes가 드문 데이터에 적합). "
               "시험 결과는 모델 선택 기준에 사용하지 않았습니다. 표의 지표는 0~1 범위입니다.")

    if "live_table" in st.session_state:
        table = st.session_state["live_table"]
        meta = st.session_state["live_meta"]
        st.success("현재 CSV로 다시 학습한 성적표를 표시합니다.")

    cols = ["모델", "특징", "CV PR-AUC", "CV ROC-AUC", "CV F1", "CV 재현율",
            "시험 average_precision", "시험 roc_auc", "시험 f1", "시험 recall", "시험 accuracy"]
    st.dataframe(table[cols].round(3), use_container_width=True, hide_index=True)
    bar = table[["모델", "특징", "CV PR-AUC"]].copy()
    bar["모델 · 특징"] = bar["모델"] + " · " + bar["특징"]
    st.bar_chart(bar.set_index("모델 · 특징")[["CV PR-AUC"]])

    st.divider()
    st.subheader("군집 특징 ON/OFF 비교")
    st.caption(
        "같은 모델·학습/시험 분할·기본 변수를 유지하고, "
        "ON일 때만 기존 질문군집 Q와 단어군집 W 비중을 추가합니다. "
        "스위치는 저장된 실험 결과를 바꿔 보여주므로 재학습을 기다릴 필요가 없습니다."
    )
    picked_model = st.selectbox(
        "비교할 예측 모델",
        options=table["모델"].drop_duplicates().tolist(),
        key="comparison_model",
    )
    use_clusters = st.toggle(
        "군집 특징 사용 (ON = 군집 포함 / OFF = 기본 특징만)",
        value=True,
        key="cluster_feature_switch",
    )
    selected_features = "기본 + 군집" if use_clusters else "기본 특징"
    other_features = "기본 특징" if use_clusters else "기본 + 군집"
    matching_rows = table.loc[table["모델"].eq(picked_model)]
    active = matching_rows.loc[matching_rows["특징"].eq(selected_features)].iloc[0]
    alternative = matching_rows.loc[matching_rows["특징"].eq(other_features)].iloc[0]
    st.write(
        f"현재 선택: **{picked_model} / {'군집 ON' if use_clusters else '군집 OFF'}** "
        "(증감은 현재 설정 − 반대 설정)"
    )

    quick_metrics = [
        ("CV PR-AUC", "CV PR-AUC", False),
        ("시험 PR-AUC(AP)", "시험 average_precision", False),
        ("Yes 재현율", "시험 recall", True),
        ("시험 F1", "시험 f1", False),
    ]
    for metric_col, (title, field, as_percent) in zip(st.columns(4), quick_metrics):
        gap = float(active[field]) - float(alternative[field])
        if as_percent:
            current_value = f"{float(active[field]):.1%}"
            change = f"{gap * 100:+.1f}%p"
        else:
            current_value = f"{float(active[field]):.3f}"
            change = f"{gap:+.3f}"
        metric_col.metric(title, current_value, delta=change)

    comparison_fields = [
        ("5겹 CV PR-AUC", "CV PR-AUC"),
        ("시험 PR-AUC(AP)", "시험 average_precision"),
        ("시험 ROC-AUC", "시험 roc_auc"),
        ("시험 정확도", "시험 accuracy"),
        ("시험 Yes 재현율", "시험 recall"),
        ("시험 F1", "시험 f1"),
        ("잘못된 경고(FP)", "시험 fp"),
        ("놓친 Yes(FN)", "시험 fn"),
    ]
    on_row = matching_rows.loc[matching_rows["특징"].eq("기본 + 군집")].iloc[0]
    off_row = matching_rows.loc[matching_rows["특징"].eq("기본 특징")].iloc[0]
    comparison = pd.DataFrame([
        {
            "평가 지표": title,
            "군집 ON": round(float(on_row[field]), 3),
            "군집 OFF": round(float(off_row[field]), 3),
            "ON − OFF": round(float(on_row[field]) - float(off_row[field]), 3),
        }
        for title, field in comparison_fields
    ])
    st.dataframe(comparison, use_container_width=True, hide_index=True)

    def confusion_for(row: pd.Series) -> pd.DataFrame:
        return pd.DataFrame(
            [[int(row["시험 tn"]), int(row["시험 fp"])],
             [int(row["시험 fn"]), int(row["시험 tp"])]],
            index=["실제 No", "실제 Yes"],
            columns=["예측 No", "예측 Yes"],
        )

    primary, secondary = st.columns(2)
    with primary:
        st.markdown(f"**현재 설정: {'군집 ON' if use_clusters else '군집 OFF'}**")
        st.dataframe(confusion_for(active), use_container_width=True)
    with secondary:
        st.markdown(f"**비교 설정: {'군집 OFF' if use_clusters else '군집 ON'}**")
        st.dataframe(confusion_for(alternative), use_container_width=True)
    st.caption(
        "FP는 정상 질문에 잘못 경고한 개수, FN은 할루시네이션 Yes를 놓친 개수입니다. "
        "성능 차이가 작으면 군집 특징의 효과가 확실하다고 단정하지 않습니다."
    )

    winner = meta["selected_by_train_cv"]
    st.subheader("학습 데이터로 선정한 우선 모델")
    st.write(f"**{winner['모델']} / {winner['특징']}** "
             f"(5겹 CV PR-AUC {winner['CV PR-AUC']:.3f})")
    selected = table.loc[(table["모델"] == winner["모델"]) &
                         (table["특징"] == winner["특징"])].iloc[0]
    st.write(f"시험 데이터의 ROC-AUC **{selected['시험 roc_auc']:.3f}**, "
             f"PR-AUC **{selected['시험 average_precision']:.3f}**, "
             f"Yes 재현율 **{selected['시험 recall']:.1%}**, "
             f"F1 **{selected['시험 f1']:.3f}**")
    confusion = pd.DataFrame(
        [[int(selected["시험 tn"]), int(selected["시험 fp"])],
         [int(selected["시험 fn"]), int(selected["시험 tp"])]],
        index=["실제 No", "실제 Yes"], columns=["예측 No", "예측 Yes"],
    )
    st.dataframe(confusion, use_container_width=True)
    baseline = meta["baseline"]
    st.warning(f"아무 질문이나 No로 답하는 기준 모델은 시험 정확도가 "
               f"{baseline['accuracy']:.1%}이지만 Yes 재현율은 0%입니다. "
               "따라서 정확도뿐 아니라 재현율·F1·ROC-AUC·PR-AUC를 함께 봐야 합니다.")

    st.subheader("기존 전처리 CSV와 결과 재현")
    st.caption("시험 라벨을 빼고 고정된 학습 3,605개만으로 벡터화·표준화·교차검증합니다. "
               "결과표에 기록된 시험 점수는 설정을 정한 다음 한 번 평가한 값입니다.")
    if st.button("현재 데이터로 모델 3종 다시 학습·검증", type="primary"):
        try:
            with st.spinner("동일한 학습/시험 분할로 6가지 실험을 실행하고 있습니다..."):
                clustered = fetch_clustered()
                prepare_question_data(clustered)  # validate split and feature schema
                fresh_table, fresh_meta = audition(clustered, n_splits=5)
            st.session_state["live_table"] = fresh_table
            st.session_state["live_meta"] = {k: v for k, v in fresh_meta.items()
                                             if k != "fitted_pipelines"}
            st.rerun()
        except Exception as exc:
            st.error(f"재학습 실패: {exc}")
            st.info("data/questions_clustered.csv가 전처리·군집 특징을 포함하는지 확인해 주세요.")

    st.download_button("모델 성적표 다운로드 (CSV)",
                       data=table.to_csv(index=False).encode("utf-8-sig"),
                       file_name="halueval_model_audition.csv", mime="text/csv")
    st.caption("이 실험은 과거 응답의 Yes/No를 예측한 것입니다. "
               "점수는 실제 서비스에서 보장되는 할루시네이션 발생 확률이 아니며, "
               "새 프롬프트 실시간 검사는 추가 검증 이후 구현할 예정입니다.")


menu = st.sidebar.radio("분석 메뉴", ["데이터 탐색·전처리", "모델 학습·성능 비교"])
if menu == "데이터 탐색·전처리":
    show_preprocessing()
else:
    show_models()
