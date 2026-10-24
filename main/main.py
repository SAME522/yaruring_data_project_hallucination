    out["질문_글자수"] = texts.str.len()
    out["질문_단어수"] = tokens.str.len()
    out["고유_단어수"] = tokens.apply(lambda ws: len(set(w.lower() for w in ws)))
    out["평균_단어길이"] = tokens.apply(lambda ws: round(sum(len(w) for w in ws) / len(ws), 3) if ws else 0.0)
    out["물음표_수"] = texts.str.count(r"\?")
    if "hallucination" in out:
        mapped = out["hallucination"].astype("string").str.lower().str.strip().map({"yes": 1, "no": 0})
        out["할루시네이션_발생"] = mapped.astype("Float64")
    return out


st.title("🔎 HaluEval 1.0 데이터 기초통계 분석")
st.caption("사용자 질문과 과거 ChatGPT 응답에 대한 할루시네이션 라벨을 탐색하는 앱")
url = st.text_input("HaluEval 원본 Raw 데이터 주소", value=DEFAULT_URL)

try:
    original = fetch_data(url)
except Exception as exc:
    st.error(f"데이터를 불러오지 못했습니다: {exc}")
    st.info("GitHub에서 `general_data.json`을 열고 Raw 주소를 복사했는지 확인해 주세요.")
    st.stop()

if "user_query" not in original.columns:
    st.error("필수 열 `user_query`가 없습니다. HaluEval general_data.json 주소인지 확인해 주세요.")
    st.stop()

data = enrich(original)
yes_count = int(data.get("할루시네이션_발생", pd.Series(dtype=float)).eq(1).sum())
no_count = int(data.get("할루시네이션_발생", pd.Series(dtype=float)).eq(0).sum())
raw_cols = len(original.columns)
a, b, c, d = st.columns(4)
a.metric("전체 행", f"{len(original):,}")
b.metric("원본 열", f"{raw_cols:,}")
c.metric("생성된 수치형 특징", "5개")
d.metric("Yes 비율", f"{yes_count/(yes_count+no_count):.2%}" if yes_count+no_count else "알 수 없음")

tab1, tab2, tab3 = st.tabs(["원본 데이터 · 결측치", "수치형 변수 · 중앙값", "라벨 및 질문 분석"])

with tab1:
    st.subheader("원본 열의 자료형과 결측치")
    overview = pd.DataFrame({
        "열 이름": original.columns,
        "자료형": [str(t) for t in original.dtypes],
        "결측치 수": original.isna().sum().to_numpy(),
        "결측 비율(%)": (original.isna().mean()*100).round(2).to_numpy(),
        "고유 값 수": original.nunique(dropna=True).to_numpy(),
    })
    st.dataframe(overview, use_container_width=True, hide_index=True)
    st.write(f"질문 텍스트 중복 행: **{original['user_query'].duplicated().sum():,}개**")
    st.caption("문자열 빈칸과 빈 리스트는 pandas의 결측치(NaN)와 다를 수 있습니다.")
    st.subheader("원본 데이터 미리보기")
    st.dataframe(original.head(30), use_container_width=True)

with tab2:
    st.subheader("질문에서 만든 수치형 변수의 기초통계")
    numeric = ["질문_글자수", "질문_단어수", "고유_단어수", "평균_단어길이", "물음표_수"]
    summary = data[numeric].agg(["count", "mean", "median", "std", "min", "max"]).T
    summary.columns = ["개수", "평균", "중앙값", "표준편차", "최솟값", "최댓값"]
    st.dataframe(summary.round(2), use_container_width=True)
    selected = st.selectbox("분포를 확인할 변수", numeric)
    hist = pd.cut(data[selected], bins=20, duplicates="drop").value_counts(sort=False)
    st.bar_chart(hist.rename_axis("구간").reset_index(name="질문 수").set_index("구간"))
    st.caption("원본에는 질문 글자 수 등의 수치 열이 없어서 질문 텍스트에서 새로 계산했습니다.")
    st.dataframe(data[["user_query"] + numeric].head(30), use_container_width=True, hide_index=True)

with tab3:
    st.subheader("할루시네이션 라벨 분포")
    if "hallucination" in data.columns:
        labels = data["hallucination"].astype("string").str.strip().str.lower().value_counts(dropna=False)
        st.bar_chart(labels.rename_axis("라벨").reset_index(name="질문 수").set_index("라벨"))
        st.write(f"Yes: **{yes_count:,}개**, No: **{no_count:,}개**")
        st.caption("이 비율은 HaluEval에 수록된 과거 응답의 비율이며, 모든 AI 질문에서의 일반적인 발생률은 아닙니다.")
        group = data.loc[data["할루시네이션_발생"].notna()].groupby("hallucination", dropna=False)
        by_label = group[["질문_글자수", "질문_단어수", "고유_단어수"]].median()
        st.subheader("라벨별 질문 특징 중앙값")
        st.dataframe(by_label, use_container_width=True)
    else:
        st.warning("할루시네이션 라벨 열이 없습니다.")

st.divider()
st.download_button("전처리 데이터 CSV 다운로드", data=data.to_csv(index=False).encode("utf-8-sig"),
                   file_name="halueval_preprocessed.csv", mime="text/csv")
st.caption("원본: RUCAIBox/HaluEval (general_data.json). 앱은 데이터 탐색용이며 머신러닝 예측 모델은 포함하지 않습니다.")
