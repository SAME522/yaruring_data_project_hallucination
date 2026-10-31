"""Third-pass unsupervised lexical grouping of 459 HaluEval residual prompts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import HDBSCAN
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from recluster_unclassified_halueval import ROOT, STOP, preprocess

PARENT=ROOT/'outputs'/'halueval_secondpass_density'
OUT=ROOT/'outputs'/'halueval_thirdpass_459'
OUT.mkdir(parents=True,exist_ok=True)

# Exploratory names grounded in representative/raw sampled prompt texts. For
# mixed groups the label is just a retrieval hint, not a validated category.
LABELS={
 4:('창의적 활용·상황 응용','혼합·검토 필요'),
 11:('생각·논리·실생활','혼합·검토 필요'),
 9:('사용자·양식·탐사','혼합·검토 필요'),
 16:('근거·중요성·재활용','혼합·검토 필요'),
 2:('계절·시·문장 창작','혼합·검토 필요'),
 15:('스토리 구성·요약','비교적 명확'),
 7:('수수께끼·퀴즈','비교적 명확'),
 17:('글쓰기 프롬프트','비교적 명확'),
 19:('대화·대인 질문','혼합·검토 필요'),
 5:('태양·이미지·운율','혼합·검토 필요'),
 20:('문학 캐릭터·판타지','혼합·검토 필요'),
 0:('리머릭 시 창작','비교적 명확'),
 21:('해야 할 일·작업 목록','비교적 명확'),
 12:('인물·삶·인용문','혼합·검토 필요'),
 8:('연도·스크립트·기간','혼합·검토 필요'),
 23:('일반 지식·질문','혼합·검토 필요'),
 18:('긍정적 가치·경험','혼합·검토 필요'),
 22:('문맥·상황에 맞춘 작성','혼합·검토 필요'),
 1:('영어 표현·서신','혼합·검토 필요'),
 13:('사건 묘사·유머','혼합·검토 필요'),
 6:('사실 정보 나열','비교적 명확'),
 3:('에너지·환경 변화','혼합·검토 필요'),
 14:('차이점·자연 현상','혼합·검토 필요'),
 10:('사물·대상 분류','혼합·검토 필요'),
}

def save(n,df):
    df.to_csv(OUT/f'{n}.csv',index=False,encoding='utf-8-sig')
    (OUT/f'{n}.json').write_text(df.to_json(orient='records',force_ascii=False),encoding='utf-8')

def main():
    questions=pd.read_csv(PARENT/'unassigned_questions.csv',encoding='utf-8-sig')
    assert len(questions)==459 and questions.source_row.is_unique
    vectorizer=TfidfVectorizer(stop_words=list(STOP),ngram_range=(1,2),
        min_df=1,max_df=.7,sublinear_tf=True,token_pattern=r'(?u)\b[a-z][a-z]+\b')
    mat=vectorizer.fit_transform(questions.question.fillna('').map(preprocess))
    valid=np.asarray(mat.getnnz(axis=1)).ravel()>0
    dim=min(25,mat.shape[0]-1,mat.shape[1]-1)
    embedding=normalize(TruncatedSVD(n_components=dim,random_state=49).fit_transform(mat[valid]))
    rows=[];fits={}
    for n in (6,8,10,12):
        for min_samples in (2,3):
            model=HDBSCAN(min_cluster_size=n,min_samples=min_samples,metric='euclidean',copy=True)
            lab=model.fit_predict(embedding)
            keys,counts=np.unique(lab,return_counts=True)
            rows.append({'min_cluster_size':n,'min_samples':min_samples,
                'cluster_count':int((keys>=0).sum()),
                'assigned_questions':int(counts[keys>=0].sum()),
                'left_unassigned':int((~valid).sum()+counts[keys==-1].sum() if -1 in keys else (~valid).sum()),
                'max_cluster_size':int(counts[keys>=0].max()) if (keys>=0).any() else 0})
            fits[(n,min_samples)]=lab
    diag=pd.DataFrame(rows)
    save('density_parameter_comparison',diag)
    # Small minimum and weak density requirement are suitable for the sparse
    # residue; all choices made WITHOUT the target label, but must be audited.
    size=8;ms=2
    fitted=fits[(size,ms)]
    assignment=np.full(len(questions),-1,dtype=int)
    assignment[valid]=fitted
    questions['thirdpass_cluster']=assignment
    questions['thirdpass_status']=np.where(assignment>=0,'3차 유사 문장 후보','미분류 유지')
    questions['thirdpass_label']=[LABELS.get(int(c),('미배정','군집 없음'))[0] for c in assignment]
    questions['thirdpass_quality']=[LABELS.get(int(c),('미배정','군집 없음'))[1] for c in assignment]
    questions['feature_terms']=np.asarray(mat.getnnz(axis=1)).ravel()
    save('thirdpass_question_assignments',questions)
    term=np.array(vectorizer.get_feature_names_out())
    global_mean=np.asarray(mat.mean(axis=0)).ravel()
    emb_all=np.zeros((len(questions),dim));emb_all[valid]=embedding
    cluster_rows=[];samples=[]
    for cl in sorted(set(assignment)-{-1}):
        idx=np.where(assignment==cl)[0]
        group=mat[idx]
        mean=np.asarray(group.mean(axis=0)).ravel()
        weight=mean*np.log1p((mean+.0001)/(global_mean+.004))
        important=term[weight.argsort()[::-1][:12]]
        center=normalize(emb_all[idx].mean(axis=0).reshape(1,-1))[0]
        proximities=emb_all[idx]@center
        closest=idx[np.argsort(-proximities)[:5]]
        cluster_rows.append({'thirdpass_cluster':int(cl),'questions':int(len(idx)),
           'top_words':', '.join(important),
           'representative_question':str(questions.question.iloc[closest[0]])[:220],
           'candidate_label':LABELS[int(cl)][0],
           'quality_note':LABELS[int(cl)][1]})
        for j,v in enumerate(closest,1):
            samples.append({'thirdpass_cluster':int(cl),'sample_type':'중심 질문','rank':j,
                            'source_row':int(questions.source_row.iloc[v]),
                            'question':str(questions.question.iloc[v])})
        random=questions.iloc[idx].sample(min(5,len(idx)),random_state=49)
        for j,r in enumerate(random.itertuples(),1):
            samples.append({'thirdpass_cluster':int(cl),'sample_type':'무작위 질문','rank':j,
                            'source_row':int(r.source_row),'question':str(r.question)})
    summary=pd.DataFrame(cluster_rows).sort_values('questions',ascending=False)
    save('thirdpass_cluster_summary_draft',summary)
    save('thirdpass_examples_draft',pd.DataFrame(samples))
    save('still_unclassified_168',questions.loc[questions.thirdpass_cluster.eq(-1)])
    save('thirdpass_by_request',questions.loc[questions.thirdpass_cluster.ge(0)]
         .groupby(['thirdpass_cluster','thirdpass_label','request_type']).size()
         .rename('questions').reset_index().sort_values(['thirdpass_cluster','questions'],ascending=[True,False]))

    before=pd.read_csv(PARENT/'all_Q1_integrated_assignments.csv',encoding='utf-8-sig')
    merged=before.merge(questions[['source_row','thirdpass_cluster','thirdpass_label',
                                  'thirdpass_quality','thirdpass_status']],
                        how='left',on='source_row',validate='one_to_one')
    merged['final_review_stage']=np.select([
        merged.expanded_topic_count.gt(0),
        merged.subcluster_id.ge(0),
        merged.thirdpass_cluster.ge(0)],
        ['1단계: 14개 사전 주제','2단계: 밀도 군집 후보','3단계: 소군집 후보'],
        default='계속 미분류')
    assert len(merged)==2734 and merged.source_row.is_unique
    save('integrated_2734_with_three_stages',merged)
    stage_counts=merged.final_review_stage.value_counts().rename_axis('stage').reset_index(name='questions')
    stage_counts['share']=stage_counts.questions/len(merged)
    save('stage_coverage',stage_counts)
    meta={'original_remaining':459,'newly_assigned':int((assignment>=0).sum()),
          'still_unassigned':int((assignment==-1).sum()),
          'groups':int(len(summary)), 'zero_content_rows':int((~valid).sum()),
          'clear_candidate_groups':int(summary.quality_note.eq('비교적 명확').sum()),
          'clear_candidate_question_count':int(summary.loc[summary.quality_note.eq('비교적 명확'),'questions'].sum()),
          'staged_2734_counts':{str(r.stage):int(r.questions) for r in stage_counts.itertuples()},
          'parameters':{'svd_dim':dim,'min_cluster_size':size,'min_samples':ms},
          'notice':'No Yes/No labels were used to create or select clusters.'}
    (OUT/'thirdpass_report.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(meta,ensure_ascii=False,indent=2))
    print(diag.to_string(index=False))
    print(summary[['thirdpass_cluster','questions','top_words','representative_question']].to_string(index=False,max_colwidth=100))

if __name__=='__main__':main()
