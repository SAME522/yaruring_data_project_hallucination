"""Compare existing W groups to new content-word group candidates, without y."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from sklearn.cluster import BisectingKMeans, KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import normalize

BASE = Path(__file__).resolve().parent
OUT = BASE / 'outputs/halueval_word_quality_improved'
OUT.mkdir(parents=True, exist_ok=True)

tokens = pd.read_csv(BASE / 'outputs/halueval_cross/tokens_after_stopwords.csv', encoding='utf-8-sig')
tokens = tokens.loc[tokens['split'].eq('train'), ['source_row','token']].drop_duplicates()
old = pd.read_csv(BASE / 'outputs/halueval_cross/word_hierarchical_dictionary.csv', encoding='utf-8-sig')
role_overrides = set('''
one two three four five six seven eight nine ten twenty hundred thousand million
first second third fourth last next more less few many several much most least
following given above below under over among between each every all as than
new different same some another using used use without within only just exactly
short long little recent current best important specific particular possible
example examples question questions answer answers information detail details
how what why when where which who whom whose
'''.split())
url_artifacts = set('https http www com org html htm url urls link links'.split())
base_words = old.loc[
    (old['train_n'] >= 8)
    & (old['word_role'].isin(['기타 내용어','주제 전문어']))
    & ~old['word'].isin(role_overrides | url_artifacts)
].copy()
words = sorted(set(base_words['word']))
word_index = {w:i for i,w in enumerate(words)}

records = tokens.loc[tokens.token.isin(words)].copy()
train_rows = sorted(tokens.source_row.unique().tolist())
train_row_map = {source_row: i for i, source_row in enumerate(train_rows)}
assert len(train_row_map) == 3605
row_ix = records.source_row.map(train_row_map).to_numpy(int)
col_ix = records.token.map(word_index).to_numpy(int)
mat = coo_matrix((np.ones(len(records), dtype='float32'),(row_ix,col_ix)),shape=(3605,len(words))).tocsr()
mat.data[:]=1
docfreq = np.asarray(mat.sum(axis=0)).ravel().astype('float64')

cooc = (mat.T @ mat).toarray().astype('float64')
np.fill_diagonal(cooc,0)
expected = np.outer(docfreq,docfreq)/3605
ppmi = np.maximum(np.log((cooc+1e-6)/(expected+1e-6)),0)
ppmi *= np.log1p(cooc)
np.fill_diagonal(ppmi,0)

views = {}
for name, base in [('cooccur_PPMI',ppmi),('word_question',mat.T)]:
    svd = TruncatedSVD(n_components=min(50,base.shape[1]-1),n_iter=8,random_state=17)
    emb = normalize(svd.fit_transform(base))
    views[name] = (emb,float(svd.explained_variance_ratio_.sum()))

old_index = old.set_index('word')
old_label = np.array([int(old_index.loc[w,'word_cluster']) for w in words])
has_W = old_label>=0
scores = []
for view_name,(X,expvar) in views.items():
    old_sil = silhouette_score(X[has_W],old_label[has_W],metric='cosine')
    print('VIEW',view_name,'words',len(words),'old_known',sum(has_W),'old_sil',round(old_sil,4),'variance',round(expvar,3),flush=True)
    for alg in ['KMeans','BisectingKMeans']:
        for k in [7,9,11,13,15]:
            labels=[]
            for seed in [17,41,73]:
                m=KMeans(n_clusters=k,n_init=12,random_state=seed) if alg=='KMeans' else BisectingKMeans(
                    n_clusters=k,n_init=12,random_state=seed,bisecting_strategy='largest_cluster')
                labels.append(m.fit_predict(X))
            sizes=np.bincount(labels[0])
            sil=silhouette_score(X,labels[0],metric='cosine')
            stability=np.mean([adjusted_rand_score(labels[0],l) for l in labels[1:]])
            shared=float(silhouette_score(X[has_W],labels[0][has_W],metric='cosine'))
            result=dict(view=view_name,alg=alg,k=k,words=len(words),old_W_silhouette_known=old_sil,
                silhouette_all=float(sil),silhouette_shared_old_W_words=shared,ARI_stability=float(stability),
                min_size=int(min(sizes)),max_size=int(max(sizes)),
                max_share=float(max(sizes)/len(words)),sizes=sizes.tolist())
            scores.append(result)
            print('EXPERIMENT',view_name,alg,k,round(sil,4),round(shared,4),round(stability,4),sorted(sizes,reverse=True),flush=True)
            if ((alg=='KMeans' and k in [9,13]) or
                (alg=='BisectingKMeans' and k==13 and view_name=='cooccur_PPMI')):
                for i in np.argsort(sizes)[::-1]:
                    terms=[words[j] for j in range(len(words)) if labels[0][j]==i]
                    terms.sort(key=lambda w:-old_index.loc[w,'train_n'])
                    print('  C',i,len(terms),', '.join(terms[:12]),flush=True)

pd.DataFrame(scores).to_csv(OUT/'candidate_diagnostics.csv',index=False,encoding='utf-8-sig')
(OUT/'candidate_diagnostics.json').write_text(json.dumps(scores,ensure_ascii=False,indent=2),encoding='utf-8')
