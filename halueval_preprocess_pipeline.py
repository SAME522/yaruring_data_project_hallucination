"""Prepare HaluEval general queries for question/word clustering and regression.

Usage:
  python halueval_preprocess_pipeline.py --input general_data.json --output outputs/halueval_prep

The train split alone fits TF-IDF, SVD and K-means. Hallucination labels are
never used to create cluster membership or predictor columns.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer

WORD_PATTERN = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)*|\d+(?:\.\d+)?")


def read_jsonl(path: Path) -> pd.DataFrame:
    with path.open(encoding="utf-8") as src:
        rows = [json.loads(line) for line in src if line.strip()]
    df = pd.DataFrame(rows)
    needed = {"ID", "user_query", "hallucination"}
    if not needed.issubset(df.columns):
        raise ValueError(f"Missing required fields: {sorted(needed - set(df.columns))}")
    df.insert(0, "source_row", np.arange(1, len(df) + 1))
    df["label"] = df["hallucination"].astype(str).str.strip().str.lower()
    df["question"] = df["user_query"].fillna("").astype(str).str.strip()
    df["y"] = df["label"].map({"yes": 1, "no": 0})
    df["original_id"] = df["ID"].astype(str)
    return df.loc[df["y"].notna() & df["question"].ne("")].copy().reset_index(drop=True)


def lexical_type(word: str) -> str:
    if word[0].isdigit():
        return "number"
    if word in ENGLISH_STOP_WORDS:
        return "function_word"
    return "content_word"


INTENT_RULES = [
    ("rewrite_or_summarize", r"\b(translate|summarize|paraphrase|rewrite|convert|shorten)\b"),
    ("math_or_programming", r"\b(python|code|algorithm|program|calculate|equation|mathematics|math|solve)\b"),
    ("creative_writing", r"\b(poem|story|fiction|song|essay|dialogue|creative|lyrics)\b"),
    ("list_or_recommend", r"\b(list|suggest|recommend|brainstorm|generate|enumerate)\b"),
    ("explain_or_compare", r"\b(explain|describe|what|why|how|define|compare|difference)\b"),
]


def rough_intent(text: str) -> str:
    """Transparent, mutually exclusive heuristic; NOT a ground-truth topic label."""
    lower=text.lower()
    for name,pattern in INTENT_RULES:
        if re.search(pattern,lower): return name
    return "other_or_unclear"


def make_question_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[tuple[int,int,str,str]]]:
    token_rows=[]
    features=[]
    for row in df.itertuples(index=False):
        matches=list(WORD_PATTERN.finditer(row.question))
        tokens=[m.group(0).lower() for m in matches]
        words=[w for w in tokens if not w[0].isdigit()]
        features.append({
            "source_row":row.source_row,
            "char_count":len(row.question),
            "token_count":len(tokens),
            "unique_token_count":len(set(tokens)),
            "avg_alpha_word_length":sum(len(w) for w in words)/len(words) if words else 0.,
            "question_mark_count":row.question.count("?"),
            "number_token_count":sum(w[0].isdigit() for w in tokens),
            "function_word_count":sum(w in ENGLISH_STOP_WORDS for w in words),
            "content_word_count":sum(w not in ENGLISH_STOP_WORDS for w in words),
        })
        for position,word in enumerate(tokens,1):
            token_rows.append((int(row.source_row),position,word,lexical_type(word)))
    return pd.DataFrame(features),token_rows


def plot_bar_labels(df, out):
    counts=df["label"].value_counts().reindex(["yes","no"],fill_value=0)
    fig,ax=plt.subplots(figsize=(6,4))
    bars=ax.bar(["Hallucination: yes","Hallucination: no"],counts,color=["#f38d6f","#42a6a0"])
    for b in bars: ax.text(b.get_x()+b.get_width()/2,b.get_height()+22,f"{int(b.get_height()):,}",ha="center",fontsize=11)
    ax.set_ylim(0,max(counts)*1.13);ax.set_ylabel("Questions");ax.set_title("HaluEval: original label distribution")
    ax.spines[["top","right"]].set_visible(False)
    fig.tight_layout();fig.savefig(out/"01_label_distribution.png",dpi=170);plt.close(fig)


def plot_question_lengths(df,out):
    fig,ax=plt.subplots(figsize=(7.5,4.5))
    for label,color in [("no","#42a6a0"),("yes","#f38d6f")]:
        x=df.loc[df.label==label,"token_count"].clip(upper=100)
        ax.hist(x,bins=35,density=True,alpha=.55,label=f"{label} (n={len(x):,})",color=color)
    ax.set(xlabel="Tokens per question (100+ clipped)",ylabel="Density",title="Question length by original label")
    ax.legend();ax.spines[["top","right"]].set_visible(False)
    fig.tight_layout();fig.savefig(out/"02_question_length.png",dpi=170);plt.close(fig)


def plot_question_clusters(summary,out):
    x=np.arange(len(summary))
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    axes[0].bar(x,summary.train_questions,color="#4f8bb2")
    axes[0].set(xlabel="Question cluster",ylabel="Training questions",title="Cluster size (training set)")
    axes[1].bar(x,summary.train_yes_rate*100,color="#ee9874")
    axes[1].axhline(summary.train_yes.sum()/summary.train_questions.sum()*100,color="#666",ls="--",label="Training-set average")
    axes[1].set(xlabel="Question cluster",ylabel="Hallucination yes (%)",title="Observed yes rate by cluster")
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.set_xticks(x,summary.question_cluster)
        ax.spines[["top","right"]].set_visible(False)
    fig.tight_layout();fig.savefig(out/"03_question_clusters.png",dpi=170);plt.close(fig)


def plot_word_clusters(summary,out):
    fig,ax=plt.subplots(figsize=(8,4))
    ax.bar(summary.word_cluster.astype(str),summary.word_types,color="#9a83bd")
    ax.set(xlabel="Context-based word cluster",ylabel="Vocabulary entries",title="Words per learned cluster")
    ax.spines[["top","right"]].set_visible(False)
    fig.tight_layout();fig.savefig(out/"04_word_clusters.png",dpi=170);plt.close(fig)


def plot_rough_intents(summary,out):
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    names=summary.intent_type.tolist();x=np.arange(len(names))
    axes[0].bar(x,summary.train_questions,color="#4f8bb2")
    axes[1].bar(x,summary.train_yes_rate*100,color="#ee9874")
    for ax in axes:
        ax.set_xticks(x,names,rotation=28,ha="right")
        ax.spines[["top","right"]].set_visible(False)
    axes[0].set(title="Heuristic question intent: counts",ylabel="Training questions")
    axes[1].set(title="Heuristic intent: observed yes rate",ylabel="Yes (%)")
    fig.tight_layout();fig.savefig(out/"05_question_intent_rules.png",dpi=170);plt.close(fig)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--input",required=True,type=Path)
    p.add_argument("--output",required=True,type=Path)
    p.add_argument("--question-clusters",type=int,default=6)
    p.add_argument("--word-clusters",type=int,default=8)
    a=p.parse_args()
    out=a.output;out.mkdir(parents=True,exist_ok=True)
    df=read_jsonl(a.input)
    original_count=len(df)
    # Exact-duplicate questions must not leak between train and test.
    conflicting=df.groupby("question").y.nunique()
    bad=set(conflicting[conflicting>1].index)
    clean=df.loc[~df.question.isin(bad)].drop_duplicates("question").copy()
    clean["split"]="train"
    i_train,i_test=train_test_split(clean.index.to_numpy(),test_size=.2,random_state=42,stratify=clean.y)
    clean.loc[i_test,"split"]="test"
    clean=clean.sort_values("source_row").reset_index(drop=True)
    feat,tokens=make_question_features(clean)
    clean=clean.merge(feat,on="source_row",validate="one_to_one")
    clean["intent_type"]=clean.question.map(rough_intent)
    train_mask=clean.split.eq("train").to_numpy()

    # Train-only vocabulary, no labels in unsupervised steps.
    vec=TfidfVectorizer(lowercase=True,strip_accents="unicode",stop_words="english",
                        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z']+\b",min_df=4,max_df=.9,
                        max_features=4000,ngram_range=(1,1),sublinear_tf=True)
    xtr=vec.fit_transform(clean.loc[train_mask,"question"])
    xte=vec.transform(clean.loc[~train_mask,"question"])
    xd=np.empty((len(clean),min(48,max(2,xtr.shape[1]-1))))
    svd=TruncatedSVD(n_components=xd.shape[1],random_state=42)
    doc_normalizer=Normalizer()
    ztr=doc_normalizer.fit_transform(svd.fit_transform(xtr))
    zte=doc_normalizer.transform(svd.transform(xte))
    qkm=MiniBatchKMeans(n_clusters=a.question_clusters,random_state=42,n_init=15,batch_size=256)
    qtr=qkm.fit_predict(ztr);qte=qkm.predict(zte)
    qlabels=np.empty(len(clean),dtype=int);qlabels[train_mask]=qtr;qlabels[~train_mask]=qte
    clean["question_cluster"]=qlabels

    # Words are grouped by shared QUESTION contexts using train documents only.
    # This is distributional grouping; it does not label true dictionary meaning.
    term_svd=TruncatedSVD(n_components=min(32,max(2,xtr.shape[0]-1)),random_state=42)
    w_embed=Normalizer().fit_transform(term_svd.fit_transform(xtr.T))
    wkm=MiniBatchKMeans(n_clusters=a.word_clusters,random_state=42,n_init=15,batch_size=256)
    w_labels=wkm.fit_predict(w_embed)
    vocab=np.asarray(vec.get_feature_names_out())
    word_map={word:int(k) for word,k in zip(vocab,w_labels)}

    token_frame=pd.DataFrame(tokens,columns=["source_row","position","token","lexical_type"])
    token_frame["word_cluster"]=token_frame.token.map(word_map).fillna(-1).astype(int)
    token_frame.loc[token_frame.lexical_type.ne("content_word"),"word_cluster"]=-1
    token_frame=token_frame.merge(clean[["source_row","split","question_cluster"]],on="source_row",validate="many_to_one")
    token_frame.to_csv(out/"token_occurrences.csv",index=False,encoding="utf-8-sig")

    # Count each question once per word for descriptive label rates (train only).
    unique_words=token_frame.loc[token_frame.split.eq("train") & token_frame.word_cluster.ge(0),["source_row","token","word_cluster"]].drop_duplicates(["source_row","token"])
    label_join=unique_words.merge(clean[["source_row","y"]],on="source_row",validate="many_to_one")
    vocab_stats=label_join.groupby(["token","word_cluster"]).agg(train_questions=("source_row","size"),train_yes=("y","sum")).reset_index()
    vocab_stats["train_yes_rate"]=vocab_stats.train_yes/vocab_stats.train_questions
    vocab_stats=vocab_stats.rename(columns={"token":"word"})
    vocab_stats.to_csv(out/"word_vocabulary_clusters.csv",index=False,encoding="utf-8-sig")

    question_cluster=[]
    for k in range(a.question_clusters):
        g=clean.loc[train_mask & clean.question_cluster.eq(k)]
        # Top terms by within-cluster training TF-IDF weight.
        idx=np.where(qtr==k)[0]
        weights=np.asarray(xtr[idx].mean(axis=0)).ravel() if len(idx) else np.zeros(xtr.shape[1])
        top=vocab[np.argsort(weights)[-12:][::-1]].tolist()
        question_cluster.append({"question_cluster":k,"train_questions":len(g),"train_yes":int(g.y.sum()),
            "train_yes_rate":g.y.mean() if len(g) else np.nan,"top_terms":", ".join(top),
            "sample_question":str(g.iloc[0].question)[:150] if len(g) else ""})
    qsummary=pd.DataFrame(question_cluster)
    qsummary.to_csv(out/"question_cluster_summary.csv",index=False,encoding="utf-8-sig")
    qsummary.to_json(out/"question_cluster_summary.json",orient="records",force_ascii=False)

    word_summary=[]
    for k in range(a.word_clusters):
        words=vocab[w_labels==k]
        subset=vocab_stats.loc[vocab_stats.word_cluster.eq(k)].sort_values("train_questions",ascending=False)
        word_summary.append({"word_cluster":k,"word_types":len(words),
          "top_words_by_frequency":", ".join(subset.word.head(15).tolist())})
    wsummary=pd.DataFrame(word_summary)
    wsummary.to_csv(out/"word_cluster_summary.csv",index=False,encoding="utf-8-sig")
    wsummary.to_json(out/"word_cluster_summary.json",orient="records",force_ascii=False)
    isummary=(clean.loc[train_mask].groupby("intent_type")
        .agg(train_questions=("source_row","size"),train_yes=("y","sum")).reset_index())
    isummary["train_yes_rate"]=isummary.train_yes/isummary.train_questions
    isummary.to_csv(out/"intent_type_summary.csv",index=False,encoding="utf-8-sig")
    isummary.to_json(out/"intent_type_summary.json",orient="records",force_ascii=False)

    # Model-ready numeric columns: cluster dummies and per-question word cluster proportions.
    counts=pd.crosstab(token_frame.source_row,token_frame.word_cluster)
    for k in range(a.word_clusters):
        ct=counts[k] if k in counts else pd.Series(0,index=counts.index)
        clean[f"word_cluster_{k}_share"]=(clean.source_row.map(ct).fillna(0).to_numpy()/clean.token_count.clip(lower=1).to_numpy())
    for k in range(a.question_clusters):
        clean[f"question_cluster_{k}"]=(clean.question_cluster==k).astype(int)
    clean["function_word_share"]=clean.function_word_count/clean.token_count.clip(lower=1)
    clean["content_word_share"]=clean.content_word_count/clean.token_count.clip(lower=1)
    for name,_ in INTENT_RULES:
        clean[f"intent_{name}"]=(clean.intent_type==name).astype(int)
    clean["intent_other_or_unclear"]=(clean.intent_type=="other_or_unclear").astype(int)
    # Keep metadata+label in the same file but clearly name the target y.
    columns=["source_row","original_id","split","question","label","y","question_cluster","intent_type",
       "char_count","token_count","unique_token_count","avg_alpha_word_length",
       "question_mark_count","number_token_count","function_word_count","content_word_count",
       "function_word_share","content_word_share"]
    columns += [f"question_cluster_{k}" for k in range(a.question_clusters)]
    columns += [f"word_cluster_{k}_share" for k in range(a.word_clusters)]
    columns += [f"intent_{name}" for name,_ in INTENT_RULES]
    columns += ["intent_other_or_unclear"]
    clean[columns].to_csv(out/"question_features_for_regression.csv",index=False,encoding="utf-8-sig")
    clean[columns].to_json(out/"question_features_for_regression.json",orient="records",force_ascii=False)
    vocab_stats.to_json(out/"word_vocabulary_clusters.json",orient="records",force_ascii=False)

    plot_bar_labels(clean,out)
    plot_question_lengths(clean,out)
    plot_question_clusters(qsummary,out)
    plot_word_clusters(wsummary,out)
    plot_rough_intents(isummary,out)
    result={"original_labeled_rows":original_count,"usable_rows":len(clean),
        "train_rows":int(train_mask.sum()),"test_rows":int((~train_mask).sum()),
        "yes":int(clean.y.sum()),"no":int((1-clean.y).sum()),
        "question_clusters":a.question_clusters,"word_clusters":a.word_clusters,
        "unique_words_in_train_vocabulary":len(vocab),"token_occurrences":len(token_frame),
        "percent_content_tokens_clustered":round(100*token_frame.loc[token_frame.lexical_type.eq("content_word"),"word_cluster"].ge(0).mean(),2),
        "question_cluster_train_sizes":qsummary.train_questions.tolist(),
        "heuristic_intent_types":len(isummary),
        "source":"https://github.com/RUCAIBox/HaluEval/blob/main/data/general_data.json",
        "warning":"Question/word clusters are unsupervised, not manually verified semantics. Do not use y/label as predictor."}
    (out/"preprocessing_report.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(result,ensure_ascii=False,indent=2))
    print(qsummary[["question_cluster","train_questions","train_yes_rate","top_terms"]].to_string(index=False))
    print(wsummary.to_string(index=False))

if __name__=="__main__":
    main()
