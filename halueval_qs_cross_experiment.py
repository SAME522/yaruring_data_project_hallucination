"""HaluEval: question-cluster Q × refined semantic word-cluster S experiment.

Exploratory train-only cross-validation. The 902 held-out test labels are NEVER
loaded into the modeling dataframe or used in feature/threshold selection.
Word and Q partitions were learned in earlier unsupervised train-wide stages;
5-fold results are diagnostic, not a leakage-free external performance claim.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             log_loss, roc_auc_score)
from sklearn.model_selection import StratifiedKFold


ROOT=Path(__file__).resolve().parent
OUT=ROOT/'outputs/halueval_qs_cross_experiment'
OUT.mkdir(parents=True,exist_ok=True)
MASTER=ROOT/'outputs/halueval_word_quality_improved/word_cluster_dictionary_refined.csv'
QUESTIONS=ROOT/'outputs/halueval_cross/question_cross_features.csv'
TOKENS=ROOT/'outputs/halueval_cross/tokens_after_stopwords.csv'

qall=pd.read_csv(QUESTIONS,encoding='utf-8-sig')
q=qall.loc[qall.split.eq('train'),[
 'source_row','question','question_cluster','question_cluster_name','y',
 'char_count','token_count','question_mark_count']].copy()
q=q.sort_values('source_row').reset_index(drop=True)
assert len(q)==3605 and int(q.y.sum())==652
assert q.source_row.is_unique and q.question_cluster.between(0,5).all()

w=pd.read_csv(MASTER,encoding='utf-8-sig')
assert w.word.is_unique
t=pd.read_csv(TOKENS,encoding='utf-8-sig')
t=t.loc[t.split.eq('train'),['source_row','token']].drop_duplicates()
assert t.source_row.isin(q.source_row).all()

semantic=sorted(w.semantic_group_id.dropna().unique())
assert semantic==[f'S{i:02d}' for i in range(1,14)]
qgroups=sorted(q.question_cluster.unique())
roles=sorted(x for x in w.role_refined.dropna().unique() if x.startswith('R_') and x!='R_URL')
word_to_sem=w.set_index('word').semantic_group_id.to_dict()
word_to_role=w.set_index('word').role_refined.to_dict()
word_to_old=w.set_index('word').word_cluster.to_dict()
sem_names=w.dropna(subset=['semantic_group_id']).drop_duplicates('semantic_group_id').set_index('semantic_group_id').semantic_group_label.to_dict()
q_names=q.drop_duplicates('question_cluster').set_index('question_cluster').question_cluster_name.to_dict()

qpos={rid:i for i,rid in enumerate(q.source_row)}
semidx={x:i for i,x in enumerate(semantic)}
roleidx={x:i for i,x in enumerate(roles)}
sq=np.zeros((len(q),len(semantic)),dtype=np.float32)
rq=np.zeros((len(q),len(roles)),dtype=np.float32)
wq=np.zeros((len(q),8),dtype=np.float32)
matched=defaultdict(set)
for rec in t.itertuples(index=False):
 pos=qpos[int(rec.source_row)]
 s=word_to_sem.get(rec.token)
 if isinstance(s,str):
  sq[pos,semidx[s]]=1
  matched[(int(rec.source_row),s)].add(rec.token)
 role=word_to_role.get(rec.token)
 if role in roleidx:rq[pos,roleidx[role]]=1
 old=word_to_old.get(rec.token,-1)
 if pd.notna(old) and int(old)>=0:wq[pos,int(old)]=1

qidx=q.question_cluster.to_numpy(dtype=int)
qq=np.eye(6,dtype=np.float32)[qidx]
xx=(qq[:,:,None]*sq[:,None,:]).reshape(len(q),78).astype(np.float32)

assignment=[]
for i,rec in enumerate(q.itertuples(index=False)):
 ss=[semantic[j] for j in np.flatnonzero(sq[i])]
 rr=[roles[j] for j in np.flatnonzero(rq[i])]
 assignment.append({
  'source_row':int(rec.source_row),'question_cluster':int(rec.question_cluster),
  'question_cluster_name':rec.question_cluster_name,
  'semantic_groups':' | '.join(ss),'semantic_group_count':len(ss),
  'cross_groups':' | '.join(f'Q{rec.question_cluster}x{s}' for s in ss),
  'role_groups':' | '.join(rr),'question':rec.question,
  'char_count':int(rec.char_count),'token_count':int(rec.token_count),
  'y':int(rec.y),
 })
assign=pd.DataFrame(assignment)
assert int((assign.semantic_group_count>0).sum())==3341

cross_rows=[]
examples=[]
for qid in range(6):
 parent=q.loc[q.question_cluster.eq(qid)]
 pn=int(len(parent)); py=int(parent.y.sum()); pr=py/pn
 for s in semantic:
  j=semidx[s]; subset=np.flatnonzero((qidx==qid)&(sq[:,j]>0))
  n=len(subset); yes=int(q.y.iloc[subset].sum()); no=n-yes
  rate=yes/n if n else float('nan')
  prior_weight=25
  smooth=(yes+prior_weight*pr)/(n+prior_weight)
  # Wilson score interval; descriptive and not an independent-group test.
  z=1.96
  denom=1+z*z/n if n else 1
  centre=(rate+z*z/(2*n))/denom if n else float('nan')
  half=z*np.sqrt(rate*(1-rate)/n+z*z/(4*n*n))/denom if n else float('nan')
  ok=n>=30 and yes>=5 and no>=5
  status=('주요 검토 후보' if ok else '소규모: 탐색만' if n>=15 else '희소: 통합·보류')
  cid=f'Q{qid}x{s}'
  ranked=sorted(subset,key=lambda ix:(-len(matched[(int(q.source_row.iloc[ix]),s)]),int(q.source_row.iloc[ix])))
  first=ranked[0] if ranked else None
  cross_rows.append({
   'cross_group':cid,'question_group':f'Q{qid}',
   'question_cluster_name':q_names[qid],
   'semantic_group':s,'semantic_group_name':sem_names[s],
   'train_questions':n,'train_yes':yes,'train_no':no,
   'train_yes_rate':rate,'parent_Q_questions':pn,'parent_Q_yes':py,
   'parent_Q_yes_rate':pr,'difference_from_parent_pp':100*(rate-pr),
   'smoothed_yes_rate':smooth,'smoothed_difference_pp':100*(smooth-pr),
   'wilson_95_low':centre-half,'wilson_95_high':centre+half,
   'analysis_status':status,
   'example_source_row':int(q.source_row.iloc[first]) if first is not None else None,
   'example_question':str(q.question.iloc[first])[:500] if first is not None else '',
   'example_matching_words':', '.join(sorted(matched[(int(q.source_row.iloc[first]),s)])) if first is not None else '',
  })
  for rnk,ix in enumerate(ranked[:3],1):
   rid=int(q.source_row.iloc[ix])
   examples.append({'cross_group':cid,'rank':rnk,'source_row':rid,
     'matching_words':', '.join(sorted(matched[(rid,s)])),
     'question':str(q.question.iloc[ix]),'y':int(q.y.iloc[ix])})

cross=pd.DataFrame(cross_rows)
audit=pd.DataFrame(examples)
assert len(cross)==78 and len(audit)==234
status_counts=cross.analysis_status.value_counts().to_dict()
q3=cross.loc[cross.question_group.eq('Q3')].copy().sort_values('train_questions',ascending=False)
prioritized=cross.loc[cross.analysis_status.eq('주요 검토 후보')].copy()
prioritized['abs_adjusted_difference_pp']=prioritized.smoothed_difference_pp.abs()
prioritized=prioritized.sort_values('abs_adjusted_difference_pp',ascending=False)

feature_sets={
 'A_Q6_only':qq,
 'B_S13_only':sq,
 'C_Q6_plus_S13':np.hstack([qq,sq]),
 'D_QS_full78':np.hstack([qq,sq,xx]),
 'E_QS_interaction_min30':None,
 'F_QS_interaction_plus_roles':np.hstack([qq,sq,xx,rq]),
 'G_old_W8_plus_roles':np.hstack([wq,rq]),
 'H_TFIDF_1to2gram':None,
 'I_TFIDF_plus_QS':None,
}

def model():
 return LogisticRegression(C=1.0,max_iter=2500,solver='liblinear',class_weight=None,random_state=17)

y=q.y.to_numpy(dtype=int)
cv_splitter=StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
fold_records=[]
out_of_fold={name:np.zeros(len(q),dtype=float) for name in feature_sets}
selected_counts=[]
for fold,(it,iv) in enumerate(cv_splitter.split(qq,y),1):
 # Fit text vocabulary on training fold only. The earlier Q/S unsupervised
 # assignments are frozen exploratory features, as reported explicitly.
 tfidf=TfidfVectorizer(ngram_range=(1,2),min_df=2,max_features=12000,
                        sublinear_tf=True,strip_accents='unicode',stop_words='english')
 text_train=tfidf.fit_transform(q.question.iloc[it].fillna(''))
 text_valid=tfidf.transform(q.question.iloc[iv].fillna(''))
 valid_interactions=(xx[it].sum(axis=0)>=30)
 selected_counts.append(int(valid_interactions.sum()))
 for name,x in feature_sets.items():
  if name=='E_QS_interaction_min30':
   Xtr=np.hstack([qq[it],sq[it],xx[it][:,valid_interactions]])
   Xva=np.hstack([qq[iv],sq[iv],xx[iv][:,valid_interactions]])
  elif name=='H_TFIDF_1to2gram':Xtr,Xva=text_train,text_valid
  elif name=='I_TFIDF_plus_QS':
   Xtr=hstack([text_train,csr_matrix(np.hstack([qq[it],sq[it],xx[it]])*.4)],format='csr')
   Xva=hstack([text_valid,csr_matrix(np.hstack([qq[iv],sq[iv],xx[iv]])*.4)],format='csr')
  else:Xtr,Xva=x[it],x[iv]
  est=model().fit(Xtr,y[it]);pp=est.predict_proba(Xva)[:,1]
  out_of_fold[name][iv]=pp
  k=int(np.ceil(len(iv)*.10))
  top=np.argsort(pp)[::-1][:k]
  fold_records.append({
   'model':name,'fold':fold,'n_valid':len(iv),'valid_yes':int(y[iv].sum()),
   'roc_auc':roc_auc_score(y[iv],pp),
   'average_precision':average_precision_score(y[iv],pp),
   'brier_score':brier_score_loss(y[iv],pp),
   'log_loss':log_loss(y[iv],pp,labels=[0,1]),
   'top10_precision':float(y[iv][top].mean()),
   'top10_yes_captured':int(y[iv][top].sum()),
  })
 print(f'Completed fold {fold}/5; min30 interaction features: {int(valid_interactions.sum())}/78',flush=True)

folds=pd.DataFrame(fold_records)
summary=[]
for name in feature_sets:
 r=folds.loc[folds.model.eq(name)]
 pp=out_of_fold[name]
 top=np.argsort(pp)[::-1][:int(np.ceil(len(y)*.1))]
 summary.append({
  'model':name,'fold_mean_roc_auc':float(r.roc_auc.mean()),
  'fold_sd_roc_auc':float(r.roc_auc.std(ddof=1)),
  'fold_mean_average_precision':float(r.average_precision.mean()),
  'fold_sd_average_precision':float(r.average_precision.std(ddof=1)),
  'fold_mean_brier':float(r.brier_score.mean()),
  'out_of_fold_roc_auc':float(roc_auc_score(y,pp)),
  'out_of_fold_average_precision':float(average_precision_score(y,pp)),
  'out_of_fold_top10_precision':float(y[top].mean()),
  'out_of_fold_top10_yes_captured':int(y[top].sum()),
  'out_of_fold_top10_recall':float(y[top].sum()/y.sum()),
 })
models=pd.DataFrame(summary).sort_values('fold_mean_roc_auc',ascending=False)

report={
 'n_train_questions':len(q),'n_train_yes':int(y.sum()),
 'train_yes_rate':float(y.mean()),'n_holdout_test_untouched':int((qall.split=='test').sum()),
 'question_clusters':len(qgroups),'semantic_clusters':len(semantic),
 'observed_intersections':len(cross),'possible_intersections':len(qgroups)*len(semantic),
 'cross_n_ge_30':int((cross.train_questions>=30).sum()),
 'cross_n_ge_50':int((cross.train_questions>=50).sum()),
 'cross_n_lt_15':int((cross.train_questions<15).sum()),
 'main_analysis_groups_with_yes_no_ge_5':int((cross.analysis_status=='주요 검토 후보').sum()),
 'train_questions_with_S_group':int((sq.sum(axis=1)>0).sum()),
 'Q3_train_size':int((qidx==3).sum()),'Q3_semantic_covered':int(((qidx==3)&(sq.sum(axis=1)>0)).sum()),
 'Q3_baseline_yes_rate':float(y[qidx==3].mean()),
 'CV_scheme':'StratifiedKFold(n_splits=5,shuffle=True,random_state=42)',
 'CV_C':1.0,'CV_model':'LogisticRegression(liblinear,L2,class_weight=None)',
 'CV_min30_fold_features':selected_counts,
 'S_semantic_labels_frozen_exploratory':True,
 'test_labels_never_used':True,
 'caution':'Q/S clusters are pre-fitted without labels on train text; this 5fold comparison is exploratory, not a fully fold-refit pipeline or an untouched external test',
 'intersection_warning':'S is multilabel per question; groups overlap and their Yes counts cannot be summed into population totals',
 'inference_warning':'Cross-group rate differences are observational, posthoc and not independent evidence of causal hallucination triggers',
 'next_step':'Manual review high-support and mixed clusters, refine only from training data; then final single external holdout evaluation',
}

for fn,df in {
 'cross_78_summary':cross,
 'Q3_cross_13_summary':q3,
 'cross_78_examples_234':audit,
 'cross_priority_review':prioritized,
 'train_question_multilabel_assignments':assign,
 'model_5fold_results':folds,
 'model_comparison_summary':models,
}.items():
 df.to_csv(OUT/f'{fn}.csv',index=False,encoding='utf-8-sig')
 df.to_json(OUT/f'{fn}.json',orient='records',force_ascii=False)
(OUT/'cross_experiment_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(report,ensure_ascii=False,indent=2))
print(models.to_string(index=False))
