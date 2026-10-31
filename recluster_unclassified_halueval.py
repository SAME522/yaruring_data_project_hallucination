"""Second-pass UNSUPERVISED clustering of the 1,138 HaluEval Q'1 leftovers.

Only question text is used for vectorizing, fitting and naming. The yes/no
label is copied into the output exclusively for eventual outcome analysis.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize

ROOT=Path(__file__).resolve().parent
IN=ROOT/'outputs'/'halueval_topic_expansion'/'expanded_question_tags.csv'
OUT=ROOT/'outputs'/'halueval_secondpass_unclassified'
OUT.mkdir(parents=True,exist_ok=True)

# Common answer directives are masked in *features only*. The original text
# remains unchanged in deliverables. Technical/topic/output tokens survive.
GENERIC={
 'given','following','provide','provided','generate','generated','create',
 'created','make','made','write','written','explain','describe','identify',
 'specify','list','example','examples','similar','different','suitable',
 'include','including','following','using','used','please','produce',
 'construct','choose','select','appropriate','input','output','answer',
 'answers','request','task','topic','topics','related','ways','way',
 'sentence','sentences','word','words','paragraph','paragraphs',
 'length','number','numbers','following','information','based',
 'show','tell','find','possible','part','parts','short','long','interesting',
 'new','suggest','consider','provide','example','specific','individual',
 'content','question','questions','person','people',
 'first','second','third','fourth','five','three','two','one','ten',
 'include','find','come','complete','fill','additional',
}
STOP=frozenset(ENGLISH_STOP_WORDS | GENERIC)
URL=re.compile(r'https?://\S+|www\.\S+')

def preprocess(s:str)->str:
    return URL.sub(' ',str(s)).lower()

def save(name:str,data:pd.DataFrame)->None:
    data.to_csv(OUT/f'{name}.csv',index=False,encoding='utf-8-sig')
    (OUT/f'{name}.json').write_text(data.to_json(orient='records',force_ascii=False),encoding='utf-8')


def cluster_metrics(x:np.ndarray,k:int):
    model=KMeans(n_clusters=k,random_state=49,n_init=20,max_iter=400)
    labs=model.fit_predict(x)
    sizes=np.bincount(labs,minlength=k)
    sil=silhouette_score(x,labs,metric='cosine',sample_size=min(len(x),800),random_state=49)
    return model,labs,{
       'k':k,'cosine_silhouette':round(float(sil),5),
       'min_size':int(sizes.min()),'max_size':int(sizes.max()),
       'max_share':round(float(sizes.max()/len(x)),4),
       'sizes':','.join(map(str,sorted(sizes,reverse=True))),
    }

def main():
    source=pd.read_csv(IN,encoding='utf-8-sig')
    frame=source.loc[source.expanded_topic_count.eq(0)].copy().reset_index(drop=True)
    assert len(frame)==1138 and frame.source_row.is_unique and frame.split.eq('train').all()
    clean=[preprocess(q) for q in frame.question.fillna('')]
    tf=TfidfVectorizer(stop_words=list(STOP),ngram_range=(1,2),min_df=2,
                       max_df=0.75,sublinear_tf=True,token_pattern=r'(?u)\b[a-z][a-z]+\b')
    sparse=tf.fit_transform(clean)
    reduced=TruncatedSVD(n_components=min(70,sparse.shape[1]-1),random_state=49).fit_transform(sparse)
    reduced=normalize(reduced)
    grid=[]
    fits={}
    for k in (6,8,10,12,14,16):
        model,labs,metrics=cluster_metrics(reduced,k)
        grid.append(metrics);fits[k]=(model,labs)
    diag=pd.DataFrame(grid)
    save('cluster_k_diagnostics',diag)

    # Pre-registered selection: prioritize interpretable clusters with neither
    # a massive catchall nor tiny fragments, then cosine silhouette. No labels.
    eligible=diag.loc[(diag.max_share<=0.32)&(diag.min_size>=12)]
    choice=int(eligible.sort_values('cosine_silhouette',ascending=False).iloc[0].k) if len(eligible) else int(diag.sort_values('cosine_silhouette',ascending=False).iloc[0].k)
    model,lab=fits[choice]
    frame['secondpass_cluster']=lab

    allterms=np.asarray(tf.get_feature_names_out())
    cluster_tfidf=normalize(sparse)
    centers=normalize(model.cluster_centers_)
    similarities=(reduced @ centers.T)
    frame['cluster_cosine']=similarities[np.arange(len(frame)),lab]
    frame['cluster_margin']=np.sort(similarities,axis=1)[:,-1]-np.sort(similarities,axis=1)[:,-2]

    # TOP tokens per group are derived from the member questions only.
    rows=[];represent=[]
    for cl in range(choice):
        idx=np.where(lab==cl)[0]
        v=np.asarray(sparse[idx].mean(axis=0)).ravel()
        order=v.argsort()[::-1]
        terms=[allterms[i] for i in order[:18]]
        membership=frame.iloc[idx]
        nearest=idx[np.argsort(-frame.cluster_cosine.to_numpy()[idx])[:10]]
        rand=membership.sample(min(10,len(membership)),random_state=49)
        rows.append({'cluster':cl,'questions':len(idx), 'share':round(len(idx)/len(frame),5),
                     'mean_similarity':round(float(frame.cluster_cosine.iloc[idx].mean()),4),
                     'mean_margin':round(float(frame.cluster_margin.iloc[idx].mean()),4),
                     'top_words':', '.join(terms),
                     'sample_requests':', '.join(Counter(membership.request_type).keys()),
                     'tentative_label':'검토 후 확정'})
        for j,ix in enumerate(nearest):
            represent.append({'cluster':cl,'sample_kind':'중심에 가까운 질문','rank':j+1,
                              'source_row':int(frame.loc[ix,'source_row']),
                              'cosine_to_center':round(float(frame.loc[ix,'cluster_cosine']),4),
                              'question':frame.loc[ix,'question'],
                              'request_type':frame.loc[ix,'request_type']})
        for j,rec in enumerate(rand.itertuples()):
            represent.append({'cluster':cl,'sample_kind':'무작위 점검 질문','rank':j+1,
                              'source_row':int(rec.source_row),'cosine_to_center':round(float(rec.cluster_cosine),4),
                              'question':rec.question, 'request_type':rec.request_type})
    summary=pd.DataFrame(rows).sort_values('questions',ascending=False)
    samples=pd.DataFrame(represent)
    save('cluster_summary_draft',summary)
    save('cluster_examples_draft',samples)
    save('question_clusters_draft',frame)
    meta={'rows':len(frame),'features':int(sparse.shape[1]),'components':int(reduced.shape[1]),'selected_k':choice,
          'selection':'cosine silhouette, max cluster <= 32%, min cluster >= 12 if possible (without y labels)',
          'stops_added':sorted(GENERIC)}
    (OUT/'draft_report.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(meta,ensure_ascii=False,indent=2))
    print(diag.to_string(index=False));print(summary[['cluster','questions','mean_similarity','top_words']].to_string(index=False))

if __name__=='__main__':main()
