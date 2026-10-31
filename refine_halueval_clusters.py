"""Cross-analyze existing HaluEval question/word clusters with topic and task intent.

Inputs are prior UNMODIFIED question-level and token-level preprocessing files.
Topic/intent tags are transparent keyword heuristics, not trained ground truth.
All hallucination-rate comparisons use the existing TRAIN split only.

Usage: python refine_halueval_clusters.py
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "outputs" / "halueval_prep"
OUT = ROOT / "outputs" / "halueval_cross"

# Deliberately retain negation and question words (not/no/never, what/why/how).
# These affect what the question asks and are often over-removed by generic lists.
STOP_GROUPS = {
    "관사·한정사": "a an the this that these those some any both another other such".split(),
    "접속사": "and or but nor yet so although though whereas since".split(),
    "전치사": "of in on at to from for by with within into onto upon among through across during about around toward towards against off".split(),
    "조동사·be동사": "am is are was were be been being do does did have has had having can could will would shall should may might must".split(),
    "일반 대명사": "i me my mine you your yours he him his she her hers it its we us our ours they them their theirs myself yourself ourselves themselves".split(),
    "요청 관용어": "please kindly".split(),
}
STOPWORDS = set().union(*(set(x) for x in STOP_GROUPS.values()))
GROUP_MAP = {word: group for group, ws in STOP_GROUPS.items() for word in ws}

# Intent words, requested formats and constraints form a SECOND, interpretable
# hierarchy under the original context-based eight word clusters. This lets us
# ask whether a 'science' term in W4 is used with e.g. 'explain' vs 'generate'.
ACTION_WORDS = set("""explain describe define compare contrast discuss analyze analyse identify
give provide tell answer clarify generate create compose write invent design draft craft
produce develop formulate construct build make propose recommend suggest list name
enumerate brainstorm calculate compute solve evaluate determine summarize summarise
translate paraphrase rewrite rephrase reword shorten proofread convert transform
simplify correct edit show demonstrate illustrate find classify categorize organise
organize predict estimate""".split())
QUESTION_WORDS = set("what why how who where when which whether".split())
CONSTRAINT_WORDS = set("""only not no never without unless if exactly precisely minimum maximum
least most fewer greater less than more fewer under over between before after
until given provided including include excluding exclude avoid ensure specified
specific requirement requirements limit length maximum minimum mandatory optional
stepwise step by step first second third numbered using based assume hypothetical
true false accurate fact factual source citation reference references""".split())
OUTPUT_WORDS = set("""table chart graph diagram list bullet paragraph sentence sentences
story poem poetry essay report summary article code script program function array
json xml html csv markdown format template outline email letter tweet headline
dialogue lyrics song recipe presentation slide slides dataset dataframe equation
formula plot illustration infographic text output""".split())

# Substantive topic keywords intentionally avoid overly polysemous forms like
# 'life', 'state', 'solution', 'power', 'energy', 'force', 'light', 'model'.
TOPIC_KEYWORDS = {
    "과학·자연": [
        "science", "scientific", "physics", "physicist", "chemistry", "chemical", "biology",
        "biological", "biologist", "experiment", "laboratory", "lab", "atom", "atomic",
        "molecule", "molecular", "photosynthesis", "bacteria", "bacterial", "ecosystem",
        "evolution", "evolutionary", "gravity", "gravitational", "quantum", "electromagnetic",
        "electricity", "electron", "proton", "neutron", "periodic table", "element",
        "carbon", "hydrogen", "oxygen", "nitrogen", "genetic", "genetics", "genome",
        "dna", "rna", "cellular", "enzyme", "microorganism", "organism", "species",
        "botany", "geology", "geological", "climate", "weather", "meteorology",
        "astronomy", "astronomical", "planet", "galaxy", "solar system", "universe",
        "fossil", "volcano", "earthquake", "temperature", "thermodynamics",
        "acid", "alkaline", "reaction", "reactions", "magnetic field", "radiation",
    ],
    "수학·컴퓨팅": [
        "mathematics", "math", "mathematical", "algebra", "calculus", "geometry",
        "equation", "equations", "statistics", "probability", "fraction", "integer",
        "derivative", "integral", "algorithm", "programming", "python", "javascript",
        "java", "html", "css", "software", "coding", "code", "computer", "database",
        "machine learning", "neural network", "artificial intelligence", "data structure",
        "recursion", "binary", "function", "variable", "array", "matrix",
    ],
    "의학·건강": [
        "medicine", "medical", "disease", "symptom", "virus", "viral", "vaccine",
        "vaccination", "health", "healthcare", "hospital", "patient", "diet", "nutrition",
        "vitamin", "protein", "cholesterol", "blood pressure", "diabetes", "cancer",
        "surgery", "mental health", "psychology", "anxiety", "depression",
        "exercise", "workout", "sleep", "heart", "muscle", "brain", "anatomy",
    ],
    "역사·인문": [
        "history", "historical", "war", "revolution", "ancient", "medieval", "empire",
        "civilization", "century", "dynasty", "constitution", "philosophy",
        "philosophical", "religion", "religious", "archaeology", "historian",
        "renaissance", "world war", "roman empire", "greek", "mythology",
    ],
    "언어·예술": [
        "poem", "poetry", "literature", "literary", "novel", "fiction", "story",
        "song", "lyrics", "music", "painting", "artwork", "artist", "drawing",
        "character", "metaphor", "simile", "grammar", "synonym", "antonym",
        "homophone", "pronunciation", "linguistics", "language", "english", "french",
        "spanish", "translation", "translate", "rhyme", "haiku", "narrative",
    ],
    "사회·경제": [
        "economics", "economic", "economy", "finance", "financial", "stock market",
        "investment", "investing", "business", "marketing", "advertising", "startup",
        "entrepreneur", "currency", "inflation", "tax", "taxation", "politics",
        "political", "government", "election", "democracy", "policy", "law",
        "legal", "society", "sociology", "trade", "sales", "budget",
    ],
    "일상·생활": [
        "cooking", "recipe", "ingredients", "meal", "shopping", "travel", "hotel",
        "tourism", "vacation", "hobby", "household", "home", "family", "friends",
        "birthday", "wedding", "clothes", "fashion", "restaurant", "food",
        "gardening", "pet", "pets", "sports", "football", "basketball",
    ],
}

TOPIC_PRIORITY = list(TOPIC_KEYWORDS)
TOPIC_PATTERNS = {
    topic: [(kw, re.compile(r"(?<![a-z])" + re.escape(kw) + r"(?![a-z])", re.I)) for kw in words]
    for topic, words in TOPIC_KEYWORDS.items()
}
WORD_TOPICS = defaultdict(list)
for topic, words in TOPIC_KEYWORDS.items():
    for term in words:
        if " " not in term:
            WORD_TOPICS[term].append(topic)


def word_role(word: str) -> str:
    if word[0].isdigit():
        return "숫자·수치"
    if word in ACTION_WORDS:
        return "요청 동사"
    if word in QUESTION_WORDS:
        return "의문·질문어"
    if word in CONSTRAINT_WORDS:
        return "조건·제약어"
    if word in OUTPUT_WORDS:
        return "출력 형식어"
    if word in WORD_TOPICS:
        return "주제 전문어"
    return "기타 내용어"


def word_topic(word: str) -> str:
    return "/".join(WORD_TOPICS[word]) if word in WORD_TOPICS else "해당 없음"

# Match leading instructions first. This avoids "Explain how to create" being
# mislabeled as creation merely because "create" appears later.
INTENT_RULES = [
    ("요약·변환", r"(?:summari[sz]e|translate|paraphrase|rewrite|rephrase|reword|shorten|proofread|convert|transform|simplify|correct|edit)"),
    ("계산·문제풀이", r"(?:calculate|compute|solve|evaluate|find the value of|determine the value of)"),
    ("창작·생성", r"(?:generate|create|compose|write|invent|design|draft|craft|produce|develop|formulate|construct|build)"),
    ("목록·추천", r"(?:list|suggest|recommend|enumerate|name|brainstorm|give examples|provide examples|identify examples)"),
    ("지식·설명", r"(?:explain|describe|define|compare|contrast|discuss|analy[sz]e|identify|what|why|how|who|where|when|which|is|are|can|does|do|tell me)"),
]
INTENT_REGEX = [(name, re.compile(r"^(?:please\s+|could you\s+|can you\s+|would you\s+|i need you to\s+|i want you to\s+|provide\s+me\s+with\s+)?" + pattern + r"\b", re.I)) for name, pattern in INTENT_RULES]

Q_NAMES = {
    0: "시·비유·구조화 중심", 1: "목록·일상 제안 중심",
    2: "문장·형식 변환 중심", 3: "여러 유형이 섞인 대규모 혼합",
    4: "예시·상황 제시 중심", 5: "서사·형식 생성 중심",
}
W_NAMES = {
    0: "문장·서사", 1: "개념·아이디어", 2: "정보·조건·계산",
    3: "기사·요약", 4: "사례·데이터", 5: "코드·수학",
    6: "목록·생활", 7: "시·운문", -1: "기존 단어군집 할당 없음",
}
TOPIC_EN = {"과학·자연": "Science", "수학·컴퓨팅": "Math/CS", "의학·건강": "Health",
            "역사·인문": "History", "언어·예술": "Arts/Language", "사회·경제": "Social/Business",
            "일상·생활": "Everyday", "미분류": "Unclassified"}
REQUEST_EN = {"요약·변환": "Summarize/Convert", "계산·문제풀이": "Calculate/Solve",
              "창작·생성": "Create/Generate", "목록·추천": "List/Recommend",
              "지식·설명": "Explain/Ask", "미분류": "Unclassified"}


def detect_topic(text: str) -> tuple[str, str, str]:
    matches = defaultdict(list)
    for topic, patterns in TOPIC_PATTERNS.items():
        matches[topic] = [term for term, p in patterns if p.search(text)]
    active = [topic for topic in TOPIC_PRIORITY if matches[topic]]
    if not active:
        return "미분류", "", "근거 부족"
    # Prefer the topic with the most distinct keyword hits. Tie-breaking is
    # deterministic; additional matching topics remain visible to reviewers.
    primary = sorted(active, key=lambda t: (-len(matches[t]), TOPIC_PRIORITY.index(t)))[0]
    evidence = "; ".join(f"{t}: {', '.join(matches[t])}" for t in active)
    return primary, evidence, "복합 주제" if len(active) > 1 else "단일 주제"


def detect_intent(text: str) -> tuple[str, str]:
    t = re.sub(r"\s+", " ", text.strip().lower())
    for name, pattern in INTENT_REGEX:
        m = pattern.search(t)
        if m:
            return name, m.group(0)
    # Only fall back to question format; avoid searching for a random verb in
    # the middle of a sentence and calling it the user's request.
    if "?" in t or re.match(r"^(?:give|provide|tell)\s+(?:me\s+)?(?:an?\s+)?(?:explanation|description|difference|definition)\b", t):
        return "지식·설명", "질문 문장 또는 설명 요청"
    if re.match(r"^(?:give|provide)\s+(?:me\s+)?(?:a\s+)?list\b", t):
        return "목록·추천", "list 요청"
    return "미분류", "명확한 시작 동사·의문형 없음"


def add_descriptive_rates(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    train = df.loc[df["split"].eq("train")]
    grouped = (train.groupby(keys, dropna=False, observed=True)
               .agg(train_n=("source_row", "size"), train_yes=("y", "sum"))
               .reset_index())
    grouped["train_no"] = grouped.train_n - grouped.train_yes
    grouped["train_yes_rate"] = (grouped.train_yes / grouped.train_n).round(5)
    grouped["reliability_note"] = np.where(grouped.train_n >= 30, "표본 30개 이상: 기술통계", "표본 30개 미만: 해석 주의")
    return grouped.sort_values(["train_n"] + keys, ascending=[False] + [True] * len(keys)).reset_index(drop=True)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    questions = pd.read_csv(SOURCE / "question_features_for_regression.csv", encoding="utf-8-sig")
    tokens = pd.read_csv(SOURCE / "token_occurrences.csv", encoding="utf-8-sig")
    assert len(questions) == questions.source_row.nunique() == 4507
    assert set(tokens.source_row) <= set(questions.source_row)

    topic_details = [detect_topic(q) for q in questions.question.fillna("")]
    questions["topic"] = [r[0] for r in topic_details]
    questions["topic_rule_evidence"] = [r[1] for r in topic_details]
    questions["topic_ambiguity"] = [r[2] for r in topic_details]
    intent_details = [detect_intent(q) for q in questions.question.fillna("")]
    questions["request_type"] = [r[0] for r in intent_details]
    questions["request_rule_evidence"] = [r[1] for r in intent_details]
    questions["topic_x_request"] = questions.topic + " × " + questions.request_type

    tokens["is_excluded"] = tokens.token.isin(STOPWORDS)
    tokens["exclusion_reason"] = tokens.token.map(GROUP_MAP).fillna("")
    excluded = tokens.loc[tokens.is_excluded]
    excluded_stats = (excluded.groupby(["token", "exclusion_reason"], observed=True)
        .agg(removed_occurrences=("source_row", "size"), affected_questions=("source_row", "nunique"))
        .reset_index().rename(columns={"token": "excluded_word"})
        .sort_values(["removed_occurrences", "excluded_word"], ascending=[False, True]).reset_index(drop=True))

    kept = tokens.loc[~tokens.is_excluded].copy()
    kept["word_role"] = kept.token.map(word_role)
    kept["word_topic"] = kept.token.map(word_topic)
    kept["word_hierarchical_cluster"] = ("W" + kept.word_cluster.astype(str) + " × " + kept.word_role)
    kept_counts = kept.groupby("source_row").size().rename("kept_token_count")
    removed_counts = excluded.groupby("source_row").size().rename("removed_token_count")
    questions = questions.merge(kept_counts, on="source_row", how="left").merge(removed_counts, on="source_row", how="left")
    questions[["kept_token_count", "removed_token_count"]] = questions[["kept_token_count", "removed_token_count"]].fillna(0).astype(int)
    questions["removed_token_share"] = (questions.removed_token_count / questions.token_count.clip(lower=1)).round(5)
    kept_sentence = (kept.sort_values(["source_row", "position"]).groupby("source_row").token.agg(" ".join))
    questions["filtered_tokens"] = questions.source_row.map(kept_sentence).fillna("")
    # Count each role per QUESTION; word roles and original W clusters are
    # descriptive features independent of the Yes/No outcome.
    role_counts = pd.crosstab(kept.source_row, kept.word_role)
    for role, name in [
        ("주제 전문어", "domain_word_count"), ("요청 동사", "action_word_count"),
        ("조건·제약어", "constraint_word_count"), ("출력 형식어", "format_word_count"),
        ("기타 내용어", "other_content_word_count"), ("숫자·수치", "numeric_word_count"),
        ("의문·질문어", "question_word_count"),
    ]:
        questions[name] = questions.source_row.map(role_counts[role] if role in role_counts else {}).fillna(0).astype(int)
        questions[name.replace("_count", "_share")] = (questions[name] / questions.kept_token_count.clip(lower=1)).round(5)

    # Previously learned word clusters were fitted WITHOUT label information.
    # Their 'dominant' assignment reflects the number of matching token
    # occurrences; 0-length/missing has explicit cluster -1.
    counts = (kept.loc[kept.word_cluster.ge(0)]
              .groupby(["source_row", "word_cluster"]).size().rename("count").reset_index())
    if len(counts):
        dominant = (counts.sort_values(["source_row", "count", "word_cluster"], ascending=[True, False, True])
                    .drop_duplicates("source_row").set_index("source_row").word_cluster)
        questions["dominant_word_cluster"] = questions.source_row.map(dominant).fillna(-1).astype(int)
    else:
        questions["dominant_word_cluster"] = -1
    questions["question_cluster_name"] = questions.question_cluster.map(Q_NAMES)
    questions["word_cluster_name"] = questions.dominant_word_cluster.map(W_NAMES)
    questions["question_x_word_cluster"] = questions.question_cluster.astype(str).radd("Q") + " × W" + questions.dominant_word_cluster.astype(str)
    questions["fine_group"] = questions.question_x_word_cluster + " × " + questions.topic_x_request
    questions["review_needed"] = np.where(questions.topic.eq("미분류") | questions.request_type.eq("미분류") | questions.topic_ambiguity.eq("복합 주제"), 1, 0)

    cross = add_descriptive_rates(questions, ["question_cluster", "dominant_word_cluster"])
    cross["question_cluster_name"] = cross.question_cluster.map(Q_NAMES)
    cross["word_cluster_name"] = cross.dominant_word_cluster.map(W_NAMES)
    cross["group_id"] = "Q" + cross.question_cluster.astype(str) + "-W" + cross.dominant_word_cluster.astype(str)
    descriptions = (questions.loc[questions.split.eq("train")]
                    .groupby(["question_cluster", "dominant_word_cluster"])
                    .agg(example=("question", "first"), typical_topic=("topic", lambda s: s.value_counts().index[0]),
                         typical_request=("request_type", lambda s: s.value_counts().index[0]))
                    .reset_index())
    cross = cross.merge(descriptions, on=["question_cluster", "dominant_word_cluster"], how="left")

    topic_intent = add_descriptive_rates(questions, ["topic", "request_type"])
    topic_intent["sample_question"] = topic_intent.apply(
        lambda row: questions.loc[questions.split.eq("train") & questions.topic.eq(row.topic)
                                  & questions.request_type.eq(row.request_type), "question"].iloc[0][:180], axis=1)
    fine = add_descriptive_rates(questions, ["question_cluster", "dominant_word_cluster", "topic", "request_type"])
    fine["fine_group"] = ("Q" + fine.question_cluster.astype(str) + "-W" + fine.dominant_word_cluster.astype(str)
                          + " | " + fine.topic + " | " + fine.request_type)

    # Each word counts at most once per question; avoid counting repeated words
    # as independent trials for the descriptive Yes rate.
    unique_token = kept[["source_row", "token"]].drop_duplicates()
    unique_token = unique_token.merge(questions[["source_row", "split", "y", "topic", "request_type"]], on="source_row", validate="many_to_one")
    word_intent = (unique_token.loc[unique_token.split.eq("train")]
                   .groupby(["token", "topic", "request_type"])
                   .agg(train_n=("source_row", "size"), train_yes=("y", "sum")).reset_index())
    word_intent["train_yes_rate"] = (word_intent.train_yes / word_intent.train_n).round(5)
    sci_vocab = {word for word, topics in WORD_TOPICS.items() if "과학·자연" in topics}
    science_terms = (word_intent.loc[word_intent.token.isin(sci_vocab) & word_intent.train_n.ge(2)]
                     .sort_values(["train_n", "token"], ascending=[False, True]))
    science_terms = science_terms.head(700)

    # Exact word-pair evidence, not a claim that either token caused the label.
    pair_rows = []
    label_lookup = questions.set_index("source_row")
    for source_row, frame in kept.groupby("source_row"):
        meta = label_lookup.loc[source_row]
        if meta["split"] != "train":
            continue
        word_set = set(frame.token)
        topics_here = sorted(word_set & sci_vocab)
        actions_here = sorted(word_set & ACTION_WORDS)
        for term in topics_here:
            for action in actions_here:
                pair_rows.append((term, action, int(source_row), int(meta["y"]), str(meta["request_type"])))
    if pair_rows:
        science_pairs = pd.DataFrame(pair_rows, columns=["scientific_word", "action_word", "source_row", "y", "request_type"])
        science_pairs = (science_pairs.groupby(["scientific_word", "action_word"])
                         .agg(train_n=("source_row", "size"), train_yes=("y", "sum"))
                         .reset_index())
        science_pairs["train_yes_rate"] = (science_pairs.train_yes / science_pairs.train_n).round(5)
        science_pairs = science_pairs.sort_values(["train_n", "scientific_word", "action_word"], ascending=[False, True, True])
    else:
        science_pairs = pd.DataFrame(columns=["scientific_word", "action_word", "train_n", "train_yes", "train_yes_rate"])

    role_context = (kept[["source_row", "token", "word_cluster", "word_role", "word_topic"]]
                    .drop_duplicates(["source_row", "token"])
                    .merge(questions[["source_row", "split", "y"]], on="source_row", validate="many_to_one"))
    role_context = role_context.loc[role_context.split.eq("train")]
    role_context = (role_context.groupby(["token", "word_cluster", "word_role", "word_topic"])
                    .agg(train_n=("source_row", "size"), train_yes=("y", "sum"))
                    .reset_index().rename(columns={"token": "word"}))
    role_context["train_yes_rate"] = (role_context.train_yes / role_context.train_n).round(5)
    role_context["hierarchical_cluster"] = "W" + role_context.word_cluster.astype(str) + " × " + role_context.word_role
    role_context = role_context.sort_values(["train_n", "word"], ascending=[False, True]).reset_index(drop=True)

    role_groups = add_descriptive_rates(
        kept[["source_row", "word_cluster", "word_role"]].drop_duplicates()
        .merge(questions[["source_row", "split", "y"]], on="source_row", validate="many_to_one"),
        ["word_cluster", "word_role"])
    role_groups["word_context_label"] = role_groups.word_cluster.map(W_NAMES)
    role_groups["hierarchical_cluster"] = "W" + role_groups.word_cluster.astype(str) + " × " + role_groups.word_role

    # This is the user's requested central cross: within EACH original question
    # cluster Q, subdivide its words by the original context group W and role R.
    # One question can contribute to several Q×W×R cells; rates count a
    # question ONCE per cell, even when multiple words from that cell occur.
    # W=-1 is NOT one of the eight learned groups: it means unseen / omitted
    # by the original TF-IDF vocabulary. Keep it as a visible 'unassigned'
    # bucket so constraint words like 'not' are still analysed, never deleted.
    qwr_tokens = kept[["source_row", "token", "word_cluster", "word_role", "question_cluster"]]
    qwr_question_presence = (qwr_tokens[["source_row", "question_cluster", "word_cluster", "word_role"]]
                             .drop_duplicates())
    qwr_question_presence = qwr_question_presence.merge(questions[["source_row", "split", "y"]],
                                                       on="source_row", validate="many_to_one")
    qwr = add_descriptive_rates(qwr_question_presence, ["question_cluster", "word_cluster", "word_role"])
    qwr["question_cluster_name"] = qwr.question_cluster.map(Q_NAMES)
    qwr["word_cluster_name"] = qwr.word_cluster.map(W_NAMES)
    qwr["subgroup_id"] = ("Q" + qwr.question_cluster.astype(str) + " × W" + qwr.word_cluster.astype(str)
                          + " × " + qwr.word_role)
    top_words_in_group = (qwr_tokens[["source_row", "question_cluster", "word_cluster", "word_role", "token"]]
        .drop_duplicates().groupby(["question_cluster", "word_cluster", "word_role"]).token
        .agg(lambda s: ", ".join(s.value_counts().head(8).index)).rename("frequent_words").reset_index())
    qwr = qwr.merge(top_words_in_group, on=["question_cluster", "word_cluster", "word_role"], how="left")

    qword = (qwr_tokens.drop_duplicates(["source_row", "token"])
             .merge(questions[["source_row", "split", "y", "request_type", "topic"]],
                    on="source_row", validate="many_to_one"))
    qword = (qword.loc[qword.split.eq("train")]
             .groupby(["question_cluster", "word_cluster", "word_role", "token"])
             .agg(train_n=("source_row", "size"), train_yes=("y", "sum")).reset_index())
    qword["train_yes_rate"] = (qword.train_yes / qword.train_n).round(5)
    qword["subgroup_id"] = ("Q" + qword.question_cluster.astype(str) + " × W" + qword.word_cluster.astype(str)
                             + " × " + qword.word_role)
    qword = qword.sort_values(["train_n", "token"], ascending=[False, True]).reset_index(drop=True)

    # Per-question interaction matrix for the optional Q×W×R predictive check.
    # Fit supervised weights only inside CV folds. The original Q/W labels were
    # learned in the prior train-only unsupervised preprocessing pass.
    qwr_features = (qwr_tokens.groupby(["source_row", "question_cluster", "word_cluster", "word_role"])
                    .size().rename("count").reset_index())
    qwr_features["feature"] = ("Q" + qwr_features.question_cluster.astype(str) + "_W"
                               + qwr_features.word_cluster.astype(str) + "_" + qwr_features.word_role)
    xqwr = qwr_features.pivot_table(index="source_row", columns="feature", values="count", fill_value=0)
    xqwr = xqwr.reindex(index=questions.source_row, fill_value=0)
    xqwr = xqwr.div(questions.kept_token_count.clip(lower=1).to_numpy(), axis=0)
    xqwr.index = questions.index

    # Assess whether the WORDS, or merely broad question clusters, are useful.
    # CV is restricted to TRAIN. This is an exploratory estimate, not a causal
    # estimate, and it does not reuse held-out TEST labels.
    train_for_cv = questions.loc[questions.split.eq("train")].copy()
    target = train_for_cv.y.astype(int)
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    models = [
        ("질문 군집만", Pipeline([
            ("prepare", ColumnTransformer([
                ("categorical", OneHotEncoder(handle_unknown="ignore"), ["question_cluster"]),
            ])), ("model", LogisticRegression(max_iter=1000))
        ]), train_for_cv[["question_cluster"]]),
        ("질문+단어 군집", Pipeline([
            ("prepare", ColumnTransformer([
                ("categorical", OneHotEncoder(handle_unknown="ignore"), ["question_cluster", "dominant_word_cluster"]),
                ("word_share", "passthrough", [f"word_cluster_{i}_share" for i in range(8)]),
            ])), ("model", LogisticRegression(max_iter=1000))
        ]), train_for_cv[["question_cluster", "dominant_word_cluster"] + [f"word_cluster_{i}_share" for i in range(8)]]),
        ("단어 역할+기존 군집", Pipeline([
            ("prepare", ColumnTransformer([
                ("categorical", OneHotEncoder(handle_unknown="ignore"), ["question_cluster", "dominant_word_cluster", "topic", "request_type"]),
                ("role", StandardScaler(), ["domain_word_share", "action_word_share", "constraint_word_share", "format_word_share", "numeric_word_share", "other_content_word_share", "question_word_share"]),
                ("word_share", "passthrough", [f"word_cluster_{i}_share" for i in range(8)]),
            ])), ("model", LogisticRegression(max_iter=1000))
        ]), train_for_cv[["question_cluster", "dominant_word_cluster", "topic", "request_type", "domain_word_share", "action_word_share", "constraint_word_share", "format_word_share", "numeric_word_share", "other_content_word_share", "question_word_share"] + [f"word_cluster_{i}_share" for i in range(8)]]),
        ("질문군집×단어군집×역할", LogisticRegression(max_iter=1500, C=.5),
         xqwr.loc[train_for_cv.index]),
        ("개별 단어·단어쌍 TF-IDF", Pipeline([
            ("tfidf", TfidfVectorizer(lowercase=True, stop_words=list(STOPWORDS), min_df=3, max_df=.9,
                                     ngram_range=(1, 2), sublinear_tf=True, token_pattern=r"(?u)\b\w\w+\b")),
            ("model", LogisticRegression(max_iter=1500, C=1.0))
        ]), train_for_cv.question),
    ]
    benchmarks = []
    for name, pipeline, x in models:
        proba = cross_val_predict(pipeline, x, target, cv=folds, method="predict_proba", n_jobs=1)[:, 1]
        benchmarks.append({"model": name, "train_rows": len(target), "folds": 5,
                           "cv_auc": float(roc_auc_score(target, proba)),
                           "cv_average_precision": float(average_precision_score(target, proba)),
                           "interpretation": "학습 데이터 내부 5분할 검증, 정답 유출 없이 비교"})
    bench_df = pd.DataFrame(benchmarks)

    questions.to_csv(OUT / "question_cross_features.csv", index=False, encoding="utf-8-sig")
    kept.to_csv(OUT / "tokens_after_stopwords.csv", index=False, encoding="utf-8-sig")
    excluded_stats.to_csv(OUT / "excluded_words_with_counts.csv", index=False, encoding="utf-8-sig")
    cross.to_csv(OUT / "question_word_cross_groups.csv", index=False, encoding="utf-8-sig")
    topic_intent.to_csv(OUT / "topic_request_groups.csv", index=False, encoding="utf-8-sig")
    fine.to_csv(OUT / "fine_intersection_groups.csv", index=False, encoding="utf-8-sig")
    science_terms.to_csv(OUT / "science_words_by_request.csv", index=False, encoding="utf-8-sig")
    science_pairs.to_csv(OUT / "science_word_action_pairs.csv", index=False, encoding="utf-8-sig")
    role_context.to_csv(OUT / "word_hierarchical_dictionary.csv", index=False, encoding="utf-8-sig")
    role_groups.to_csv(OUT / "word_role_context_groups.csv", index=False, encoding="utf-8-sig")
    qwr.to_csv(OUT / "Q_question_W_word_R_role_cross.csv", index=False, encoding="utf-8-sig")
    qword.to_csv(OUT / "words_within_question_word_role_groups.csv", index=False, encoding="utf-8-sig")
    bench_df.to_csv(OUT / "predictive_cv_benchmarks.csv", index=False, encoding="utf-8-sig")

    # Data-only JSON used by the JavaScript spreadsheet authoring tool.
    for name, frame in {
        "question_cross_features": questions,
        "excluded_words_with_counts": excluded_stats,
        "question_word_cross_groups": cross,
        "topic_request_groups": topic_intent,
        "fine_intersection_groups": fine,
        "science_words_by_request": science_terms,
        "science_word_action_pairs": science_pairs,
        "word_hierarchical_dictionary": role_context,
        "word_role_context_groups": role_groups,
        "Q_question_W_word_R_role_cross": qwr,
        "words_within_question_word_role_groups": qword,
        "predictive_cv_benchmarks": bench_df,
    }.items():
        (OUT / f"{name}.json").write_text(frame.replace({np.nan: None}).to_json(orient="records", force_ascii=False), encoding="utf-8")

    train = questions.loc[questions.split.eq("train")]
    matrix = (pd.crosstab(train.question_cluster, train.dominant_word_cluster)
              .reindex(index=range(6), columns=range(-1, 8), fill_value=0))
    fig, ax = plt.subplots(figsize=(12, 5.8))
    im = ax.imshow(matrix.values, cmap="Blues")
    ax.set_xticks(range(len(matrix.columns)), ["Unassigned" if x == -1 else f"W{x}" for x in matrix.columns]); ax.set_yticks(range(6), [f"Q{x}" for x in matrix.index])
    ax.set(xlabel="Dominant word-context cluster", ylabel="Original question cluster", title="Original question × word cluster (train questions)")
    for i in range(6):
        for j in range(len(matrix.columns)):
            v = matrix.iat[i, j]
            ax.text(j, i, f"{v:,}" if v else "", ha="center", va="center", fontsize=9, color="white" if v > matrix.values.max() * .48 else "#223148")
    fig.colorbar(im, ax=ax, shrink=.8, label="Question count")
    fig.tight_layout(); fig.savefig(OUT / "01_question_word_cross_heatmap.png", dpi=170); plt.close(fig)

    top_grid = (pd.pivot_table(train, index="topic", columns="request_type", values="source_row", aggfunc="count", fill_value=0))
    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(top_grid.to_numpy(), cmap="YlGnBu")
    ax.set_xticks(range(len(top_grid.columns)), [REQUEST_EN[x] for x in top_grid.columns], rotation=25, ha="right")
    ax.set_yticks(range(len(top_grid.index)), [TOPIC_EN[x] for x in top_grid.index])
    ax.set(title="Topic × request type (training set)")
    for i in range(len(top_grid.index)):
        for j in range(len(top_grid.columns)):
            v = top_grid.iat[i, j]
            if v: ax.text(j, i, f"{v:,}", ha="center", va="center", fontsize=9,
                          color="white" if v > top_grid.to_numpy().max() * .48 else "#152436")
    fig.colorbar(im, ax=ax, shrink=.75, label="Question count")
    fig.tight_layout(); fig.savefig(OUT / "02_topic_request_heatmap.png", dpi=170); plt.close(fig)

    science_view = topic_intent.loc[topic_intent.topic.eq("과학·자연") & topic_intent.train_n.ge(3)].sort_values("train_n", ascending=False)
    if len(science_view):
        fig, ax = plt.subplots(figsize=(10, 4.8))
        pos = np.arange(len(science_view))
        ax.barh(pos, science_view.train_yes_rate * 100, color="#4389ac")
        ax.set_yticks(pos, [f"{REQUEST_EN[r.request_type]} (n={r.train_n})" for r in science_view.itertuples()])
        ax.invert_yaxis(); ax.set(xlabel="Observed hallucination Yes rate (%)", title="Science-topic requests (train only, descriptive)")
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout(); fig.savefig(OUT / "03_science_intent_rates.png", dpi=170); plt.close(fig)

    report = {
        "source_questions": int(len(questions)), "train_questions": int(len(train)),
        "test_questions_not_used_for_descriptive_rates": int(questions.split.eq("test").sum()),
        "question_clusters": int(questions.question_cluster.nunique()),
        "word_clusters_learned": 8,
        "nonempty_question_word_intersections_train": int(len(cross)),
        "nonempty_topic_request_intersections_train": int(len(topic_intent)),
        "nonempty_fine_intersections_train": int(len(fine)),
        "fine_groups_at_least_20_train_questions": int(fine.train_n.ge(20).sum()),
        "stopword_policy_total_terms": int(len(STOPWORDS)),
        "stopword_types_observed": int(len(excluded_stats)),
        "removed_token_occurrences": int(len(excluded)),
        "removed_ratio_all_token_occurrences": float(len(excluded) / len(tokens)),
        "questions_without_existing_word_cluster": int(questions.dominant_word_cluster.eq(-1).sum()),
        "questions_needing_topic_or_intent_review": int(questions.review_needed.sum()),
        "science_train_counts": {str(k): int(v) for k, v in train.loc[train.topic.eq("과학·자연")].request_type.value_counts().items()},
        "word_hierarchical_groups_train": int(len(role_groups)),
        "nonempty_Q_W_role_groups_train": int(len(qwr)),
        "words_in_Q_W_role_groups_train": int(len(qword)),
        "word_role_lexicon_entries": int(len(role_context)),
        "distinct_science_word_action_pairs_train": int(len(science_pairs)),
        "cv_benchmarks": benchmarks,
        "science_train_examples": [
            {"type": str(row.request_type), "n": int(row.train_n), "yes_rate": float(row.train_yes_rate), "sample": str(row.sample_question)}
            for row in science_view.itertuples()
        ],
        "stopword_words_observed": excluded_stats.excluded_word.tolist(),
        "stopword_policy_words_absent_from_data": sorted(STOPWORDS - set(excluded_stats.excluded_word)),
        "note": "Topic and task intent are keyword heuristics, not ground-truth classes. Training-set Yes rates are correlations, not causal estimates; never use y as a predictor or cluster input.",
    }
    (OUT / "analysis_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
