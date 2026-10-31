"""Single, predeclared, label-free lexical refinement of mixed S01/S03/S06.

Audit three mixed semantic groups using word meanings and question text, keep
original S13 features, and add a small transparent subtopic axis. Compare once
on 5-fold train CV; do NOT use the untouched 902 held-out test labels.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix,hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score,average_precision_score,brier_score_loss
from sklearn.model_selection import StratifiedKFold

BASE=Path(__file__).resolve().parent
OUT=BASE/'outputs/halueval_final_refinement'
OUT.mkdir(parents=True,exist_ok=True)

# Fixed before looking at the new model scores. Unspecified words retain their
# prior S group and are marked "unresolved"; none are deleted or forced.
LEXICON={
 'S01_AI_컴퓨팅':set('language learning computer model machine gpt system assistant models processing input'.split()),
 'S01_과학_연구':set('science scientific research hypothesis physical paper test sample field cause effect body'.split()),
 'S03_비즈니스_웹':set('customer business website service online web company logo page project marketing brand products team budget money offer'.split()),
 'S03_건강_교육_생활':set('healthy environment student students school health life home vacation self protect'.split()),
 'S06_데이터_SQL':set('data user users query database account names valid structure types network neural'.split()),
 'S06_수학_기하':set('area triangle mathematical expression probability cm side width total methods'.split()),
 'S06_금융_거래':set('price stock market rate cost sales income interest store'.split()),
 'S06_연도_시간':set('year years old age past date month months week'.split()),
}
PARENTS={name:name.split('_')[0] for name in LEXICON}
names=list(LEXICON)
assert len(names)==8

words=pd.read_csv(BASE/'outputs/halueval_word_quality_improved/word_cluster_dictionary_refined.csv',encoding='utf-8-sig')
q=pd.read_csv(BASE/'outputs/halueval_cross/question_cross_features.csv',encoding='utf-8-sig')
q=q.loc[q.split.eq('train'),['source_row','question','question_cluster','y']].sort_values('source_row').reset_index(drop=True)
tokens=pd.read_csv(BASE/'outputs/halueval_cross/tokens_after_stopwords.csv',encoding='utf-8-sig')
tokens=tokens.loc[tokens.split.eq('train'),['source_row','token']].drop_duplicates()
assert len(q)==3605 and q.y.sum()==652

word_map=words.set_index('word')
selected=words.loc[words.semantic_group_id.isin(['S01','S03','S06'])].copy()
def subtopic(word,parent):
 matched=[group for group,tags in LEXICON.items() if PARENTS[group]==parent and word in tags]
 assert len(matched)<=1,(word,parent,matched)
 return matched[0] if matched else None

selected['proposed_subtopic']=[subtopic(r.word,r.semantic_group_id) for r in selected.itertuples()]
selected['subtopic_status']=np.where(selected.proposed_subtopic.notna(),'명시적 의미어','혼합·다의어: 기존 S 유지')
selected['review_reason']=np.where(selected.proposed_subtopic.notna(),
 '의미 기반 사전; Yes/No 라벨 미사용','용례가 넓거나 경계가 애매해 임의 재분류 보류')
selected=selected[['word','semantic_group_id','semantic_group_label','train_n',
 'proposed_subtopic','subtopic_status','review_reason','train_yes','train_yes_rate']]

word_sub=selected.set_index('word').proposed_subtopic.dropna().to_dict()
qpos={rid:i for i,rid in enumerate(q.source_row)}
groups=sorted(words.semantic_group_id.dropna().unique())
gidx={s:i for i,s in enumerate(groups)}
rnames=sorted(x for x in words.role_refined.unique() if isinstance(x,str) and x.startswith('R_') and x!='R_URL')
ridx={x:i for i,x in enumerate(rnames)}
nidx={n:i for i,n in enumerate(names)}
Q=np.eye(6,dtype=np.float32)[q.question_cluster.to_numpy(dtype=int)]
S=np.zeros((len(q),len(groups)),dtype=np.float32)
R=np.zeros((len(q),len(rnames)),dtype=np.float32)
N=np.zeros((len(q),len(names)),dtype=np.float32)
matches=defaultdict(set)
for rec in tokens.itertuples(index=False):
 i=qpos[int(rec.source_row)]
 if rec.token not in word_map.index:continue
 w=word_map.loc[rec.token]
 if isinstance(w.semantic_group_id,str):S[i,gidx[w.semantic_group_id]]=1
 if w.role_refined in ridx:R[i,ridx[w.role_refined]]=1
 sub=word_sub.get(rec.token)
 if sub:
  N[i,nidx[sub]]=1
  matches[(int(rec.source_row),sub)].add(rec.token)

X=(Q[:,:,None]*S[:,None,:]).reshape(len(q),78)
QN=(Q[:,:,None]*N[:,None,:]).reshape(len(q),48)
data=[
 ('Q_S_baseline',np.hstack([Q,S])),
 ('Q_S_interactions_roles',np.hstack([Q,S,X,R])),
 ('Q_S_roles_plus_8_subtopics',np.hstack([Q,S,R,N])),
 ('Q_S_cross_plus_8_subtopics',np.hstack([Q,S,X,R,N])),
 ('Q_S_cross_plus_8_subtopics_Qcross',np.hstack([Q,S,X,R,N,QN])),
 ('Q_S_cross_subtopics_Qcross_C0p3',np.hstack([Q,S,X,R,N,QN])),
 ('TFIDF_reference',None),
 ('TFIDF_plus_8_subtopics',None),
]
y=q.y.to_numpy(dtype=int)
cv=StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
rows=[]
preds={name:np.zeros(len(q)) for name,_ in data}
for fold,(it,iv) in enumerate(cv.split(Q,y),1):
 tv=TfidfVectorizer(ngram_range=(1,2),min_df=2,max_features=12000,
  sublinear_tf=True,strip_accents='unicode',stop_words='english')
 Ttr=tv.fit_transform(q.question.iloc[it].fillna(''))
 Tva=tv.transform(q.question.iloc[iv].fillna(''))
 for name,mat in data:
  if name=='TFIDF_reference':Xtr,Xva=Ttr,Tva
  elif name=='TFIDF_plus_8_subtopics':
   Xtr=hstack([Ttr,csr_matrix(N[it]*.4)],format='csr')
   Xva=hstack([Tva,csr_matrix(N[iv]*.4)],format='csr')
  else:Xtr,Xva=mat[it],mat[iv]
  reg=.3 if name.endswith('C0p3') else 1.0
  clf=LogisticRegression(C=reg,max_iter=2500,solver='liblinear',random_state=17)
  clf.fit(Xtr,y[it]);pp=clf.predict_proba(Xva)[:,1]
  preds[name][iv]=pp
  rows.append({'model':name,'fold':fold,'roc_auc':roc_auc_score(y[iv],pp),
   'average_precision':average_precision_score(y[iv],pp),
   'brier_score':brier_score_loss(y[iv],pp)})
 print(f'Fold {fold}/5 finished',flush=True)

folds=pd.DataFrame(rows)
summary=[]
for name,_ in data:
 d=folds.loc[folds.model.eq(name)]
 pp=preds[name]
 top=np.argsort(pp)[::-1][:int(np.ceil(len(y)*.1))]
 summary.append({
  'model':name,'roc_auc_mean':float(d.roc_auc.mean()),
  'roc_auc_std':float(d.roc_auc.std(ddof=1)),
  'ap_mean':float(d.average_precision.mean()),
  'ap_std':float(d.average_precision.std(ddof=1)),
  'brier_mean':float(d.brier_score.mean()),
  'oof_roc_auc':float(roc_auc_score(y,pp)),
  'oof_ap':float(average_precision_score(y,pp)),
  'top10_precision':float(y[top].mean()),
  'top10_yes_captured':int(y[top].sum()),
 })
models=pd.DataFrame(summary).sort_values('roc_auc_mean',ascending=False)

sub_rows=[];audit=[]
for name in names:
 j=nidx[name]
 ix=np.flatnonzero(N[:,j]>0)
 parent=PARENTS[name]
 for sourcei in sorted(ix,key=lambda i:(-len(matches[(int(q.source_row.iloc[i]),name)]),q.source_row.iloc[i]))[:5]:
  row=q.iloc[sourcei];audit.append({'subtopic':name,'source_row':int(row.source_row),
   'matching_words':', '.join(sorted(matches[(int(row.source_row),name)])),
   'question':row.question,'y':int(row.y)})
 n=len(ix);ny=int(y[ix].sum());p=ny/n if n else float('nan')
 original_ix=np.flatnonzero(S[:,gidx[parent]]>0)
 pp=float(y[original_ix].mean())
 candidates=selected.loc[selected.proposed_subtopic.eq(name)].sort_values('train_n',ascending=False)
 sub_rows.append({'subtopic':name,'parent_S':parent,'lexicon_words':len(candidates),
  'frequent_words':', '.join(candidates.word.head(14)),
  'question_count':n,'yes_questions':ny,'yes_rate':p,'parent_yes_rate':pp,
  'difference_pp':float((p-pp)*100),
  'review_note':'표본이 작으면 해석 금지' if n<30 else '임시 의미 사전, 원문 검토 필요'})
substats=pd.DataFrame(sub_rows)

revised_questions=[]
for i,r in enumerate(q.itertuples(index=False)):
 matched_sub=[names[j] for j in np.flatnonzero(N[i])]
 revised_questions.append({'source_row':int(r.source_row),'question_cluster':int(r.question_cluster),
  'original_semantic_groups':' | '.join(groups[j] for j in np.flatnonzero(S[i])),
  'refined_subtopics':' | '.join(matched_sub),
  'q_x_subtopic_groups':' | '.join(f'Q{r.question_cluster}x{z}' for z in matched_sub),
  'subtopic_count':len(matched_sub),
  'question':r.question,'y':int(r.y)})
questions=pd.DataFrame(revised_questions)

base_result=models.set_index('model')
report={
 'train_questions':int(len(q)),'train_yes':int(y.sum()),
 'holdout_test_untouched':902,
 'focus_mixed_semantic_groups':['S01','S03','S06'],
 'focus_vocabulary':int(len(selected)),
 'assigned_to_8_subtopics':int(selected.proposed_subtopic.notna().sum()),
 'unresolved_kept_original_S':int(selected.proposed_subtopic.isna().sum()),
 'subtopics':len(names),'train_questions_with_subtopics':int((N.sum(axis=1)>0).sum()),
 'models_compared':len(models),'cross_validation_folds':5,
 'best_model':str(models.iloc[0]['model']),
 'tfidf_reference_auc':float(base_result.loc['TFIDF_reference','roc_auc_mean']),
 'tfidf_plus_8_auc':float(base_result.loc['TFIDF_plus_8_subtopics','roc_auc_mean']),
 'baseline_QS_roles_interactions_auc':float(base_result.loc['Q_S_interactions_roles','roc_auc_mean']),
 'best_non_text_auc':float(models.loc[~models.model.str.startswith('TFIDF'),'roc_auc_mean'].max()),
 'decision':'Do not extend clustering beyond this round; freeze analysis, retain best CV baseline, conduct one final external holdout evaluation',
 'caution':'The subtopic lexicon was designed after examining train-group patterns and is exploratory; no test labels used. Prior Q/S groups were fitted on train corpus before CV; final fold-refit or external test needed.',
}
for name,df in {
 'final_word_audit_186':selected,
 'subtopic_8_summary':substats,
 'subtopic_sample_questions_40':pd.DataFrame(audit),
 'question_subtopic_assignments':questions,
 'final_model_5fold':folds,
 'final_model_comparison':models,
}.items():
 df.to_csv(OUT/f'{name}.csv',index=False,encoding='utf-8-sig')
 df.to_json(OUT/f'{name}.json',orient='records',force_ascii=False)
(OUT/'final_refinement_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(report,ensure_ascii=False,indent=2))
print(substats.to_string(index=False))
print(models.to_string(index=False))
