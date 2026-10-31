"""Controlled HaluEval question-cluster experiment: mask five template tokens.

Re-fits *question clusters only* with the same TF-IDF, SVD, Normalizer and
MiniBatchKMeans parameters as the original pipeline. The extra masking terms
are excluded from the feature-building vectorizer, never from question text.

Uses training questions alone to learn TF-IDF, SVD and KMeans; predict labels
for test questions only afterwards. Yes/No is NOT used for clustering.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import adjusted_rand_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer, OneHotEncoder

ROOT=Path(__file__).resolve().parent
SRC=ROOT/'outputs'/'halueval_cross'
OUT=ROOT/'outputs'/'halueval_remask'
MASK=('create','list','following','example','generate')

Q_OLD={0:'시·비유·구조화',1:'목록·일상',2:'문장·형식',3:'혼합 일반',4:'예시·상황',5:'서사·생성'}

def save(frame:pd.DataFrame,name:str)->None:
    frame.to_csv(OUT/f'{name}.csv',index=False,encoding='utf-8-sig')
    (OUT/f'{name}.json').write_text(frame.to_json(orient='records',force_ascii=False),encoding='utf-8')


def top_word_stats(qdf:pd.DataFrame,tokens:pd.DataFrame,clust_col:str,cluster_id:int)->dict:
    subset=qdf.loc[qdf[clust_col].eq(cluster_id),['source_row']]
    if subset.empty:
        return {'top_word':'','top_word_question_count':0,'top_word_share':np.nan}
    word_subset=tokens.loc[tokens.source_row.isin(set(subset.source_row))]
    freq=word_subset[['source_row','token']].drop_duplicates().token.value_counts()
    return {'top_word':str(freq.index[0]),'top_word_question_count':int(freq.iloc[0]),
            'top_word_share':float(freq.iloc[0]/len(subset))}


def make_cluster_summary(qdf:pd.DataFrame,tokens:pd.DataFrame,col:str,kind:str,n_clusters:int=6)->pd.DataFrame:
    rows=[]
    for k in range(n_clusters):
        g=qdf.loc[qdf[col].eq(k)]
        frequent=top_word_stats(qdf,tokens,col,k)
        classified=g.loc[g.topic.ne('미분류')]
        topic_vals=classified.topic.value_counts()
        req_vals=g.request_type.value_counts()
        rows.append({
          'kind':kind,'cluster':k,'train_questions':int(len(g)),
          **frequent,
          'top_topic_if_classified':str(topic_vals.index[0]) if len(topic_vals) else '없음',
          'top_topic_share_classified':float(topic_vals.iloc[0]/len(classified)) if len(classified) else np.nan,
          'topic_classified_questions':int(len(classified)),
          'top_request_type':str(req_vals.index[0]) if len(req_vals) else '없음',
          'top_request_share_all':float(req_vals.iloc[0]/len(g)) if len(g) else np.nan,
          'example_question':str(g.question.iloc[0])[:165] if len(g) else '',
        })
    return pd.DataFrame(rows)


def main()->None:
    OUT.mkdir(parents=True,exist_ok=True)
    allq=pd.read_csv(SRC/'question_cross_features.csv',encoding='utf-8-sig')
    tokens=pd.read_csv(SRC/'tokens_after_stopwords.csv',encoding='utf-8-sig')
    assert len(allq)==4507 and allq.source_row.nunique()==4507
    train_mask=allq.split.eq('train').to_numpy()
    train=allq.loc[train_mask].copy()
    assert len(train)==3605
    vocab_stopwords=sorted(set(ENGLISH_STOP_WORDS).union(MASK))

    # Match original modeling settings EXCEPT for these five additional terms.
    vectorizer=TfidfVectorizer(lowercase=True,strip_accents='unicode',
       stop_words=vocab_stopwords,
       token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z']+\b", min_df=4,max_df=.9,
       max_features=4000,ngram_range=(1,1),sublinear_tf=True)
    x_train=vectorizer.fit_transform(allq.loc[train_mask,'question'])
    x_test=vectorizer.transform(allq.loc[~train_mask,'question'])
    components=min(48,max(2,x_train.shape[1]-1))
    svd=TruncatedSVD(n_components=components,random_state=42)
    norm=Normalizer()
    embed_train=norm.fit_transform(svd.fit_transform(x_train))
    embed_test=norm.transform(svd.transform(x_test))
    kmeans=MiniBatchKMeans(n_clusters=6,random_state=42,n_init=15,batch_size=256)
    q_new_train=kmeans.fit_predict(embed_train)
    q_new_test=kmeans.predict(embed_test)
    new_ids=np.empty(len(allq),dtype=int)
    new_ids[train_mask]=q_new_train
    new_ids[~train_mask]=q_new_test
    allq['question_cluster_original']=allq.question_cluster.astype(int)
    allq['question_cluster_revised']=new_ids
    train=allq.loc[train_mask].copy()
    vocab=vectorizer.get_feature_names_out()

    # Raw frequencies use filtered tokens from original text, even when a
    # keyword is masked from the new *model* vectorizer. This is intentional.
    train_tokens=tokens.loc[tokens.split.eq('train')].copy()
    train_tokens=train_tokens.merge(train[['source_row','question_cluster_revised']],on='source_row',validate='many_to_one')
    train_tokens['question_cluster_original']=train_tokens.question_cluster

    orig=make_cluster_summary(train,train_tokens,'question_cluster_original','기존 Q')
    new=make_cluster_summary(train,train_tokens,'question_cluster_revised','수정 Q')
    orig['top_tfidf_terms_model']='기존 전처리 모델은 그대로 유지'
    top_terms=[]
    for k in range(6):
        idx=np.where(q_new_train==k)[0]
        means=np.asarray(x_train[idx].mean(axis=0)).ravel() if len(idx) else np.zeros(len(vocab))
        terms=vocab[np.argsort(means)[-15:][::-1]]
        top_terms.append(', '.join(terms.tolist()))
    new['top_tfidf_terms_model']=top_terms
    orig['cluster_description']=orig.cluster.map(Q_OLD)
    new['cluster_description']='수정 실험: 번호에 고정 의미 없음'
    summaries=pd.concat([orig,new],ignore_index=True)
    save(summaries,'original_vs_revised_cluster_summary')

    transition=pd.crosstab(train.question_cluster_original,train.question_cluster_revised)
    transition=transition.reindex(index=range(6),columns=range(6),fill_value=0)
    transitions=[]
    for i in range(6):
        for j in range(6):
            n=int(transition.loc[i,j]);
            transitions.append({'original_Q':i,'revised_Q':j,'train_questions':n,
               'original_row_share':round(n/int(transition.loc[i].sum()),5),
               'total_train_share':round(n/len(train),5)})
    save(pd.DataFrame(transitions),'question_cluster_movement')

    # Substantive document frequencies within new clusters, with original W
    # and R retained to support the next cross-analysis stage.
    tc=train_tokens[['source_row','question_cluster_revised','token','word_cluster','word_role','word_topic']]
    df_unique=tc.drop_duplicates(['source_row','token'])
    tf=tc.groupby(['question_cluster_revised','token']).size().rename('token_occurrences')
    dfreq=df_unique.groupby(['question_cluster_revised','token']).size().rename('questions_with_word')
    globaldf=df_unique.groupby('token').size().rename('overall_questions_with_word')
    freq=pd.concat([tf,dfreq],axis=1).reset_index()
    freq=freq.merge(df_unique[['token','word_cluster','word_role','word_topic']].drop_duplicates('token'),on='token')
    freq['cluster_questions']=freq.question_cluster_revised.map(train.groupby('question_cluster_revised').size())
    freq['overall_questions_with_word']=freq.token.map(globaldf)
    freq['per_100_questions']=(100*freq.questions_with_word/freq.cluster_questions).round(2)
    freq['lift_vs_train_average']=((freq.questions_with_word/freq.cluster_questions)
                      /(freq.overall_questions_with_word/len(train))).round(3)
    freq=freq.rename(columns={'token':'word'})
    freq=freq.sort_values(['question_cluster_revised','questions_with_word','word'],ascending=[True,False,True])
    save(freq,'revised_cluster_word_frequencies')

    top=[]
    for k in range(6):
        g=freq.loc[freq.question_cluster_revised.eq(k)].head(25).copy()
        g.insert(1,'rank',range(1,len(g)+1));top.append(g)
    save(pd.concat(top,ignore_index=True),'top_revised_cluster_words')

    qwr=train_tokens[['source_row','question_cluster_revised','word_cluster','word_role']].drop_duplicates()
    qwr=qwr.groupby(['question_cluster_revised','word_cluster','word_role']).size().rename('questions_containing_group').reset_index()
    qwr['cluster_questions']=qwr.question_cluster_revised.map(train.groupby('question_cluster_revised').size())
    qwr['per_100_questions']=(100*qwr.questions_containing_group/qwr.cluster_questions).round(2)
    qwr['group_label']=('NewQ'+qwr.question_cluster_revised.astype(str)+' × W'+qwr.word_cluster.astype(str)+' × '+qwr.word_role)
    qwr=qwr.sort_values(['question_cluster_revised','questions_containing_group'],ascending=[True,False])
    save(qwr,'revised_Q_original_W_role_intersections')

    important_words=[]
    old_anchor_counts=[]
    new_anchor_counts=[]
    for term in MASK:
        for kind,col in [('기존 Q','question_cluster_original'),('수정 Q','question_cluster_revised')]:
            q_with=set(train_tokens.loc[train_tokens.token.eq(term),'source_row'])
            for k in range(6):
                g=train.loc[train[col].eq(k)]
                n=int(g.source_row.isin(q_with).sum());den=len(g)
                important_words.append({'term':term,'kind':kind,'cluster':k,
                      'questions_containing_term':n,'cluster_questions':den,
                      'share_within_cluster':n/den if den else np.nan})
    save(pd.DataFrame(important_words),'masked_template_term_audit')

    # Compare only the *cluster-as-feature* logistic models on the same train
    # folds. Target used here for evaluation; never fed into clustering.
    target=train.y.astype(int)
    folds=StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
    cv=[]
    for kind,col in [('기존 질문 군집','question_cluster_original'),('수정 질문 군집','question_cluster_revised')]:
        pipeline=Pipeline([('enc',OneHotEncoder(handle_unknown='ignore')),
                           ('model',LogisticRegression(max_iter=1000))])
        oof=cross_val_predict(pipeline,train[[col]],target,cv=folds,method='predict_proba')[:,1]
        cv.append({'model':kind,'rows':len(train),'folds':5,'roc_auc':round(float(roc_auc_score(target,oof)),5)})
    save(pd.DataFrame(cv),'cluster_only_cv_comparison')

    compact=allq[['source_row','original_id','split','question','label','y',
         'question_cluster_original','question_cluster_revised','topic','request_type',
         'topic_ambiguity','kept_token_count','removed_token_count']].copy()
    compact['cluster_changed_number']=(compact.question_cluster_original!=compact.question_cluster_revised).astype(int)
    save(compact,'question_level_revised_clusters')

    # Plots. Original/new cluster ids are arbitrary and should not be treated
    # as matched labels; use the movement heatmap to show their relationship.
    fig,ax=plt.subplots(figsize=(8,5.6))
    im=ax.imshow(transition.to_numpy(),cmap='Blues')
    ax.set_xticks(range(6),[f'New {i}' for i in range(6)])
    ax.set_yticks(range(6),[f'Old {i}' for i in range(6)])
    ax.set(title='Old vs revised question clusters (train only)',xlabel='Revised cluster',ylabel='Original cluster')
    for i in range(6):
        for j in range(6):
            value=int(transition.iat[i,j]);
            if value:ax.text(j,i,str(value),ha='center',va='center',fontsize=9,
               color='white' if value>transition.to_numpy().max()*.5 else '#233952')
    fig.colorbar(im,ax=ax,shrink=.8,label='Questions')
    fig.tight_layout();fig.savefig(OUT/'01_question_cluster_movement_heatmap.png',dpi=170);plt.close(fig)

    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    for ax,df,title in [(axes[0],orig,'Before masking 5 words'),(axes[1],new,'After masking 5 words')]:
        ax.bar([f'{k}' for k in df.cluster],df.top_word_share*100,color='#4385AF')
        for x,val in enumerate(df.top_word_share*100):ax.text(x,val+1,f'{val:.0f}%',ha='center',fontsize=8)
        ax.set(ylim=(0,110),title=title,xlabel='Cluster ID',ylabel='Top word used in questions (%)')
        ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Does removing template verbs reduce single-word domination?',fontsize=13)
    fig.tight_layout();fig.savefig(OUT/'02_top_word_dominance_comparison.png',dpi=170);plt.close(fig)

    fig,ax=plt.subplots(figsize=(10,4.6))
    x=np.arange(6);width=.35
    ax.bar(x-width/2,orig.train_questions,width,label='Old',color='#4385AF')
    ax.bar(x+width/2,new.train_questions,width,label='Revised',color='#E69C61')
    ax.set(xticks=x,xticklabels=[f'Cluster {i}' for i in x],ylabel='Training questions',title='Cluster size before and after re-fitting')
    ax.legend();ax.spines[['top','right']].set_visible(False)
    fig.tight_layout();fig.savefig(OUT/'03_cluster_size_before_after.png',dpi=170);plt.close(fig)

    train_ari=float(adjusted_rand_score(train.question_cluster_original,train.question_cluster_revised))
    result={
       'total_questions':len(allq),'train_questions':len(train),'test_questions':int((~train_mask).sum()),
       'masked_extra_terms':list(MASK),'original_question_clusters':6,'revised_question_clusters':6,
       'train_only_unsupervised_fit':True,
       'old_top_word_shares':[round(float(x),4) for x in orig.top_word_share],
       'new_top_word_shares':[round(float(x),4) for x in new.top_word_share],
       'old_more_than_90_percent_clusters':int(orig.top_word_share.ge(.9).sum()),
       'new_more_than_90_percent_clusters':int(new.top_word_share.ge(.9).sum()),
       'old_max_cluster_questions':int(orig.train_questions.max()),
       'new_max_cluster_questions':int(new.train_questions.max()),
       'adjusted_rand_index_old_vs_new':round(train_ari,4),
       'question_cluster_only_cv':cv,
       'note':'Cluster ID numbers are not semantically aligned. Masking is only applied to TF-IDF, not to original questions, token-frequency tables or labels. This is a controlled single-intervention experiment, not proof of improved semantic clustering.'
    }
    (OUT/'experiment_report.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
