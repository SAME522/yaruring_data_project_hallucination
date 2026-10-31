"""Recluster HaluEval unlabeled-topic prompts by dense text neighborhoods.

Prior subject tags, hallucination outcomes and request classes are excluded from
fit. Unlike KMeans, HDBSCAN can leave dissimilar prompts unassigned (noise).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import HDBSCAN
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from recluster_unclassified_halueval import ROOT, IN, STOP, preprocess

OUT=ROOT/'outputs'/'halueval_secondpass_density'
OUT.mkdir(parents=True,exist_ok=True)
N_COMPONENTS=30
MIN_CLUSTER_SIZE=15
MIN_SAMPLES=3

# Analyst interpretations based on examples from question text only. They are
# candidates for human review, NOT truth labels or measured precision.
INTERPRET={
 16:('수학 계산·도형 문제','주제·문제','비교적 명확'),
 17:('색상·로고·디자인 표현','혼합','혼합·검토 필요'),
 8:('의미·시간·일반 설명','혼합','혼합·검토 필요'),
 21:('문제 해결·환경 개선','혼합','혼합·검토 필요'),
 15:('데이터 처리·표·보안','혼합','혼합·검토 필요'),
 10:('SNS 글·트윗 작성','요청·표현 형식','비교적 명확'),
 23:('이야기·서사 창작','요청·표현 형식','비교적 명확'),
 3:('개념·이론 설명','혼합','혼합·검토 필요'),
 24:('인물·캐릭터 설정','요청·표현 형식','비교적 명확'),
 14:('상황·도표·과정','혼합','혼합·검토 필요'),
 12:('항목·준비물 목록','요청·표현 형식','혼합·검토 필요'),
 19:('책·문학 작품','주제·문제','비교적 명확'),
 13:('게임·기기 특성','혼합','혼합·검토 필요'),
 2:('AI·언어모델','주제·문제','혼합·검토 필요'),
 1:('언어·표현','혼합','혼합·검토 필요'),
 20:('핵심 내용 요약','요청·표현 형식','혼합·검토 필요'),
 9:('종류·유형 설명','혼합','혼합·검토 필요'),
 5:('행동·태도 묘사','혼합','혼합·검토 필요'),
 0:('차이·비교 설명','혼합','혼합·검토 필요'),
 6:('기사·문서 요약','요청·표현 형식','비교적 명확'),
 7:('장면·대상 묘사','요청·표현 형식','비교적 명확'),
 22:('창의적 활용·제안','요청·표현 형식','비교적 명확'),
 18:('진술문·주장 작성','요청·표현 형식','비교적 명확'),
 11:('음악·시 창작','요청·표현 형식','혼합·검토 필요'),
 4:('대화·회의 상황','요청·표현 형식','혼합·검토 필요'),
}

def save(name,frame):
    frame.to_csv(OUT/f'{name}.csv',index=False,encoding='utf-8-sig')
    (OUT/f'{name}.json').write_text(frame.to_json(orient='records',force_ascii=False),encoding='utf-8')

def main():
    src=pd.read_csv(IN,encoding='utf-8-sig')
    source=src.loc[src.expanded_topic_count.eq(0)].copy().reset_index(drop=True)
    assert len(source)==1138 and source.source_row.nunique()==len(source)
    tf=TfidfVectorizer(stop_words=list(STOP),ngram_range=(1,2),min_df=2,
                       max_df=.75,sublinear_tf=True,token_pattern=r'(?u)\b[a-z][a-z]+\b')
    mat=tf.fit_transform(source.question.fillna('').map(preprocess))
    # Completely unseen content words (after min_df and stopwords) yield an
    # all-zero row. Never pretend that zero vectors form a semantic cluster.
    valid=np.asarray(mat.getnnz(axis=1)).reshape(-1)>0
    svd=TruncatedSVD(n_components=N_COMPONENTS,random_state=49)
    emb_valid=normalize(svd.fit_transform(mat[valid]))
    clusterer=HDBSCAN(min_cluster_size=MIN_CLUSTER_SIZE,min_samples=MIN_SAMPLES,
                      metric='euclidean',copy=True)
    labels_valid=clusterer.fit_predict(emb_valid)
    label=np.full(len(source),-1,dtype=int)
    label[valid]=labels_valid
    emb=np.zeros((len(source),N_COMPONENTS),dtype=float)
    emb[valid]=emb_valid
    source['subcluster_id']=label
    source['clustered']=np.where(label>=0, '밀집 군집', '유사 문장 부족·미배정')
    source['evidence_token_count']=np.asarray(mat.getnnz(axis=1)).reshape(-1)
    source['no_shared_keywords']=np.where(valid,0,1)
    source['candidate_cluster_name']=[INTERPRET.get(int(i),('미배정','미분류','군집 없음'))[0] for i in label]
    source['candidate_cluster_kind']=[INTERPRET.get(int(i),('미배정','미분류','군집 없음'))[1] for i in label]
    source['candidate_review_state']=[INTERPRET.get(int(i),('미배정','미분류','군집 없음'))[2] for i in label]
    terms=np.asarray(tf.get_feature_names_out())
    mean_global=np.asarray(mat.mean(axis=0)).ravel()
    summary=[];samples=[]
    for cl in sorted(set(label)-{-1}):
        idx=np.where(label==cl)[0]
        center=normalize(emb[idx].mean(axis=0).reshape(1,-1))[0]
        similarity=emb[idx]@center
        mean=np.asarray(mat[idx].mean(axis=0)).ravel()
        # Combined relevance: frequent in cluster and disproportionately
        # represented compared with all 1,138 prompts.
        rel=mean*np.log1p((mean+0.0001)/(mean_global+0.005))
        best=np.argsort(rel)[::-1][:15]
        prominent=terms[best]
        summary.append({'subcluster_id':int(cl),'question_count':len(idx),
            'share_of_1138':float(len(idx)/len(source)),
            'top_terms':', '.join(prominent),
            'representative_question':str(source.question.iloc[idx[similarity.argmax()]])[:220],
            'proposed_theme':INTERPRET[int(cl)][0],
            'theme_type':INTERPRET[int(cl)][1],
            'review_note':INTERPRET[int(cl)][2]})
        centers=idx[np.argsort(-similarity)[:6]]
        random=source.iloc[idx].sample(min(6,len(idx)),random_state=103)
        for rank,rowid in enumerate(centers,1):
            samples.append({'subcluster_id':int(cl),'kind':'중심 질문','rank':rank,
                'source_row':int(source.source_row.iloc[rowid]),
                'question':str(source.question.iloc[rowid]),
                'request_type':str(source.request_type.iloc[rowid])})
        for rank,rec in enumerate(random.itertuples(),1):
            samples.append({'subcluster_id':int(cl),'kind':'무작위 질문','rank':rank,
                'source_row':int(rec.source_row),
                'question':str(rec.question),
                'request_type':str(rec.request_type)})
    summary=pd.DataFrame(summary).sort_values('question_count',ascending=False)
    examples=pd.DataFrame(samples)
    save('density_cluster_summary_draft',summary)
    save('density_cluster_examples_draft',examples)
    save('density_question_assignments_draft',source)
    save('unassigned_questions',source.loc[source.subcluster_id.eq(-1),[
         'source_row','question','request_type','evidence_token_count','no_shared_keywords',
         'question_cluster_original','label','y']])
    save('cluster_by_request_type',source.loc[source.subcluster_id.ge(0)]
         .groupby(['subcluster_id','candidate_cluster_name','request_type']).size()
         .rename('questions').reset_index().sort_values(['subcluster_id','questions'],ascending=[True,False]))
    save('cluster_by_previous_Q',source.loc[source.subcluster_id.ge(0)]
         .groupby(['subcluster_id','candidate_cluster_name','question_cluster_original']).size()
         .rename('questions').reset_index().sort_values(['subcluster_id','questions'],ascending=[True,False]))
    # For future word-cluster crossing, retain data-derived term evidence.
    term_entries=[]
    for entry in summary.itertuples(index=False):
        for pos,w in enumerate(entry.top_terms.split(', '),1):
            term_entries.append({'subcluster_id':entry.subcluster_id,
                                 'proposed_theme':entry.proposed_theme,
                                 'rank':pos,'word_or_phrase':w})
    save('cluster_term_dictionary',pd.DataFrame(term_entries))
    coverage=pd.DataFrame([
        {'status':'후보 군집 배정','questions':int((label>=0).sum())},
        {'status':'유사성 부족·미배정','questions':int((label==-1).sum())}
    ])
    coverage['share']=coverage.questions/len(source)
    save('coverage',coverage)
    # Preserve the earlier 14 topic tags unchanged. These density groups mix
    # subjects with output styles and must be modeled on a SEPARATE axis.
    joined=src.merge(source[[
        'source_row','subcluster_id','candidate_cluster_name','candidate_cluster_kind',
        'candidate_review_state','clustered','no_shared_keywords'
    ]],on='source_row',how='left',validate='one_to_one')
    joined['secondpass_status']=np.where(joined.expanded_topic_count.gt(0),
       '기존 14개 주제 태그 보유',
       np.where(joined.subcluster_id.ge(0),'2차 문장 유사도 군집 후보','여전히 미배정'))
    assert len(joined)==2734 and joined.source_row.is_unique
    save('all_Q1_integrated_assignments',joined)

    fig,axes=plt.subplots(1,2,figsize=(13,5))
    axes[0].bar(['Density groups','Unassigned'],coverage.questions,color=['#4288AD','#C1CBD0'])
    for i,n in enumerate(coverage.questions):axes[0].text(i,n+12,str(int(n)),ha='center')
    axes[0].set_ylim(0,800)
    axes[0].set(title="Second-pass text clustering",ylabel='Questions')
    top=summary.sort_values('question_count',ascending=True).tail(12)
    axes[1].barh([f"G{r.subcluster_id}" for r in top.itertuples()],top.question_count,color='#5198A9')
    axes[1].set(title='Largest 12 candidate groups',xlabel='Questions')
    for ax in axes:ax.spines[['top','right']].set_visible(False)
    fig.tight_layout();fig.savefig(OUT/'secondpass_group_counts.png',dpi=180);plt.close(fig)
    meta={'total':len(source),'clusters':len(summary),
          'assigned':int((label>=0).sum()),'noise':int((label==-1).sum()),
          'zero_vector_questions':int((~valid).sum()),
          'already_assigned_original_topics':int(src.expanded_topic_count.gt(0).sum()),
          'integrated_with_any_tag_or_candidate':int(joined.secondpass_status.ne('여전히 미배정').sum()),
          'clearly_interpretable_by_analyst_sample_only':int(sum(
              INTERPRET[int(r.subcluster_id)][2]=='비교적 명확' for r in summary.itertuples())),
          'text_features':int(mat.shape[1]),'svd_dims':N_COMPONENTS,
          'min_cluster_size':MIN_CLUSTER_SIZE,'min_samples':MIN_SAMPLES,
          'selection_basis':'Text only, TF-IDF 1-2grams, 30 SVD dims, HDBSCAN density; hallucination y untouched'}
    (OUT/'draft_report.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(meta,ensure_ascii=False,indent=2))
    print(summary[['subcluster_id','question_count','top_terms','representative_question']].to_string(index=False,max_colwidth=140))

if __name__=='__main__':main()
