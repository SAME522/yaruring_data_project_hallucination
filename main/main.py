"""Entrypoint with visible top navigation for all HaluEval study pages.

Run: streamlit run main.py
"""

import streamlit as st


pages = [
    st.Page(
        "pages/0_데이터_전처리.py",
        title="데이터 전처리",
        icon="🔎",
        default=True,
    ),
    st.Page(
        "pages/1_모델_평가.py",
        title="모델 평가",
        icon="📋",
    ),
    st.Page(
        "pages/2_군집_효과_비교.py",
        title="군집 ON/OFF",
        icon="🔀",
    ),
    st.Page(
        "pages/3_위험도_세부_분석.py",
        title="위험도·오분류 분석",
        icon="🔍",
    ),
    st.Page(
        "pages/4_성능_개선.py",
        title="성능 개선",
        icon="🛠️",
    ),
]

current_page = st.navigation(pages, position="top")
current_page.run()
