"""HaluEval: question-only topic clustering and hallucination prediction.

Run: python halueval_project.py --input general_data.json --output results
Dependencies: pandas numpy scipy scikit-learn
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.cluster import MiniBatchKMeans
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score, silhouette_score,
)
from sklearn.model_selection import train_test_split


def load_data(path: Path) -> pd.DataFrame:
    with path.open(encoding="utf-8") as handle:
        raw = [json.loads(line) for line in handle if line.strip()]
    frame = pd.DataFrame(raw)
    label_col = "hallucination" if "hallucination" in frame else "hallucination_label"
    frame = frame[["user_query", label_col]].rename(columns={label_col: "label"})
    frame["label"] = frame["label"].astype(str).str.strip().str.lower()
    frame = frame.loc[frame.label.isin(["yes", "no"]) & frame.user_query.notna()].copy()
    frame["question"] = frame.user_query.astype(str).str.strip()
    frame = frame.loc[frame.question.ne("")].copy()
    frame["y"] = (frame.label == "yes").astype(int)
    return frame.reset_index(drop=True)


def scores(name, model, x_train, y_train, x_test, y_test) -> dict:
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    prob = model.predict_proba(x_test)[:, 1]
    return {
        "model": name,
        "accuracy": accuracy_score(y_test, pred),
        "balanced_accuracy": balanced_accuracy_score(y_test, pred),
        "precision_yes": precision_score(y_test, pred, zero_division=0),
        "recall_yes": recall_score(y_test, pred, zero_division=0),
        "f1_yes": f1_score(y_test, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, prob),
        "predicted_yes": int(pred.sum()),
    }, pred, prob


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--clusters", type=int, default=6)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    data = load_data(args.input)
    # Exact repeats can put the same text in both splits. Drop these first;
    # contradictory labels are excluded rather than resolved using test labels.
    conflicts = data.groupby("question").y.nunique()
    conflict_questions = set(conflicts[conflicts > 1].index)
    data = data.loc[~data.question.isin(conflict_questions)]
    duplicates = int(data.question.duplicated().sum())
    data = data.drop_duplicates("question").reset_index(drop=True)

    train_idx, test_idx = train_test_split(
        np.arange(len(data)), test_size=0.2, random_state=42, stratify=data.y
    )
    train, test = data.iloc[train_idx].copy(), data.iloc[test_idx].copy()
    vectorizer = TfidfVectorizer(
        lowercase=True, strip_accents="unicode", stop_words="english",
        ngram_range=(1, 2), min_df=3, max_df=0.95, max_features=5000,
        sublinear_tf=True,
    )
    x_train = vectorizer.fit_transform(train.question)
    x_test = vectorizer.transform(test.question)

    # Unsupervised cluster fit uses training-question features ONLY.
    km = MiniBatchKMeans(n_clusters=args.clusters, random_state=42, n_init=10, batch_size=256)
    train_groups = km.fit_predict(x_train)
    test_groups = km.predict(x_test)
    g_train = csr_matrix(np.eye(args.clusters)[train_groups])
    g_test = csr_matrix(np.eye(args.clusters)[test_groups])

    candidates = [
        ("항상 No (기준선)", DummyClassifier(strategy="most_frequent"), x_train, x_test),
        ("군집만 사용", LogisticRegression(class_weight="balanced", max_iter=1500, random_state=42), g_train, g_test),
        ("질문 TF-IDF", LogisticRegression(class_weight="balanced", max_iter=1500, random_state=42), x_train, x_test),
        ("질문 TF-IDF + 군집", LogisticRegression(class_weight="balanced", max_iter=1500, random_state=42), hstack([x_train, g_train]).tocsr(), hstack([x_test, g_test]).tocsr()),
    ]
    metrics=[]
    predictions=pd.DataFrame({"question":test.question.to_numpy(), "actual_yes":test.y.to_numpy(), "cluster":test_groups})
    for name,model,xtr,xte in candidates:
        row,pred,prob=scores(name,model,xtr,train.y.to_numpy(),xte,test.y.to_numpy())
        metrics.append(row)
        if name=="질문 TF-IDF + 군집":
            predictions["predicted_yes"] = pred
            predictions["predicted_probability"] = prob

    terms=vectorizer.get_feature_names_out()
    groups=[]
    for k in range(args.clusters):
        selected=train.iloc[np.where(train_groups==k)[0]]
        top=terms[np.argsort(km.cluster_centers_[k])[-10:][::-1]].tolist()
        groups.append({"cluster":k,"train_questions":len(selected),"train_yes":int(selected.y.sum()),
                       "train_yes_rate":float(selected.y.mean()) if len(selected) else 0,
                       "top_terms":", ".join(top)})
    pd.DataFrame(metrics).to_csv(args.output/"model_comparison.csv",index=False,encoding="utf-8-sig")
    pd.DataFrame(groups).to_csv(args.output/"cluster_summary_train_only.csv",index=False,encoding="utf-8-sig")
    predictions.to_csv(args.output/"test_predictions.csv",index=False,encoding="utf-8-sig")
    overview={"original_labeled_questions":int(len(load_data(args.input))),"conflicting_question_texts_excluded":len(conflict_questions),
              "duplicate_rows_excluded":duplicates,"usable_unique_questions":len(data),"train":len(train),"test":len(test),
              "train_yes":int(train.y.sum()),"test_yes":int(test.y.sum()),"k":args.clusters,
              "caution":"Benchmark conditional association, not causal risk; labels concern historical ChatGPT responses."}
    (args.output/"dataset_overview.json").write_text(json.dumps(overview,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(overview,ensure_ascii=False,indent=2))
    print(pd.DataFrame(metrics).to_string(index=False,float_format=lambda x:f"{x:.4f}"))
    print(pd.DataFrame(groups).to_string(index=False))


if __name__=="__main__":
    main()
