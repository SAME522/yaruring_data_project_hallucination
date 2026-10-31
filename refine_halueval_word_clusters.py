"""Refine HaluEval W word clusters using train-only word contexts.

The fitted semantic groups are exploratory and do not use yes/no. Functional
roles are kept as a separate axis. Rare, URL-like and weak-context words are
never silently forced into a confident semantic label.
"""

from __future__ import annotations

import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from sklearn.cluster import BisectingKMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import adjusted_rand_score, average_precision_score, roc_auc_score, silhouette_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.pipeline import make_pipeline

BASE = Path(__file__).resolve().parent
OUT = BASE / 'outputs/halueval_word_quality_improved'
OUT.mkdir(parents=True, exist_ok=True)

dictionary = pd.read_csv(BASE/'outputs/halueval_cross/word_hierarchical_dictionary.csv', encoding='utf-8-sig')
unified = pd.read_csv(BASE/'outputs/halueval_unified_word_clusters/unified_word_index.csv', encoding='utf-8-sig')
unified = unified[['word','subject_groups','format_groups']].drop_duplicates('word')
dictionary = dictionary.merge(unified,on='word',how='left',validate='one_to_one')

qs = pd.read_csv(BASE/'outputs/halueval_prep/question_features_for_regression.csv',encoding='utf-8-sig')
qs = qs[qs.split.eq('train')].sort_values('source_row').reset_index(drop=True)
assert len(qs)==3605 and int(qs.y.sum())==652
tokens = pd.read_csv(BASE/'outputs/halueval_cross/tokens_after_stopwords.csv', encoding='utf-8-sig')
tokens = tokens[tokens.split.eq('train')][['source_row','token']].drop_duplicates()
question_row = {row_id:i for i,row_id in enumerate(qs.source_row)}

WORDS_NUM = set('one two three four five six seven eight nine ten twenty hundred thousand million first second third fourth last next more less few many several much most least'.split())
WORDS_QUESTION = set('how what why when where which who whom whose'.split())
WORDS_CONDITION = set('following given above below under over among between each every all as than same some another using used use without within only just exactly'.split())
WORDS_MODIFIER = set('new different short long little recent current best important specific particular possible example examples question questions answer answers information detail details'.split())
URL_ARTIFACTS = set('https http www com org html htm url urls link links noinput th'.split())
SYMBOL_WORDS = set('x y b c d s'.split())
ROLE_OVERRIDES = WORDS_NUM | WORDS_QUESTION | WORDS_CONDITION | WORDS_MODIFIER
OLD_ROLE_MAP = {
    '요청 동사':'R_REQUEST','의문·질문어':'R_QUESTION',
    '조건·제약어':'R_CONDITION','출력 형식어':'R_FORMAT',
    '숫자·수치':'R_NUMBER','주제 전문어':'TOPIC_WORD',
    '기타 내용어':'CONTENT_WORD',
}
ROLE_NAMES = {
    'R_REQUEST':'요청 동사','R_QUESTION':'의문·질문 표현',
    'R_CONDITION':'조건·제약 표현','R_FORMAT':'출력 형식 표현',
    'R_NUMBER':'수량·숫자 표현','R_MODIFIER':'정도·비교·표현 제약',
    'R_URL':'URL·형식 잡음','R_SYMBOL':'기호·코드 변수',
    'TOPIC_WORD':'주제 전문어',
    'CONTENT_WORD':'일반 내용어',
}


def refine_role(word: str, prior: str) -> str:
    if word in URL_ARTIFACTS:return 'R_URL'
    if word in SYMBOL_WORDS:return 'R_SYMBOL'
    if word in WORDS_QUESTION:return 'R_QUESTION'
    if word in WORDS_NUM:return 'R_NUMBER'
    if word in WORDS_CONDITION:return 'R_CONDITION'
    if word in WORDS_MODIFIER:return 'R_MODIFIER'
    return OLD_ROLE_MAP.get(prior,'CONTENT_WORD')


dictionary['role_refined'] = [refine_role(r.word,r.word_role) for r in dictionary.itertuples()]
dictionary['role_name_refined'] = dictionary.role_refined.map(ROLE_NAMES)
eligible = dictionary.loc[(dictionary.train_n>=8) & dictionary.role_refined.isin(['TOPIC_WORD','CONTENT_WORD'])].copy()
words = sorted(eligible.word.tolist())
word_idx = {w:i for i,w in enumerate(words)}
filtered = tokens.loc[tokens.token.isin(words)]
rows = filtered.source_row.map(question_row).to_numpy(int)
cols = filtered.token.map(word_idx).to_numpy(int)
x = coo_matrix((np.ones(len(rows),dtype='float32'),(rows,cols)),shape=(3605,len(words))).tocsr()
x.data[:]=1
dfreq = np.asarray(x.sum(axis=0)).ravel().astype('float64')
cooc = (x.T@x).toarray().astype('float64')
np.fill_diagonal(cooc,0)
expected = np.outer(dfreq,dfreq)/3605
ppmi = np.maximum(np.log((cooc+1e-6)/(expected+1e-6)),0)*np.log1p(cooc)
np.fill_diagonal(ppmi,0)
svd = TruncatedSVD(n_components=min(50,len(words)-1),n_iter=8,random_state=17)
emb = normalize(svd.fit_transform(ppmi))

K=13
models = [BisectingKMeans(n_clusters=K,n_init=12,bisecting_strategy='largest_cluster',random_state=seed).fit(emb)
          for seed in [17,41,73]]
k8_models = [BisectingKMeans(n_clusters=8,n_init=12,bisecting_strategy='largest_cluster',random_state=seed).fit(emb)
             for seed in [17,41,73]]
labels = models[0].labels_
sizes = np.bincount(labels)
old_W = dictionary.set_index('word').loc[words,'word_cluster'].to_numpy(dtype=int)
old_present = old_W>=0
old_sil = float(silhouette_score(emb[old_present],old_W[old_present],metric='cosine'))
new_sil_shared = float(silhouette_score(emb[old_present],labels[old_present],metric='cosine'))
new_sil_all = float(silhouette_score(emb,labels,metric='cosine'))
stability = float(np.mean([adjusted_rand_score(labels,model.labels_) for model in models[1:]]))
k8_sil_shared=float(silhouette_score(emb[old_present],k8_models[0].labels_[old_present],metric='cosine'))
k8_stability=float(np.mean([adjusted_rand_score(k8_models[0].labels_,model.labels_) for model in k8_models[1:]]))

provisional_names = {
    0:'수학·코드·알고리즘',1:'식품·일상 물품',2:'수치·연도·가격 혼합',
    3:'국가·역사·경제',4:'앱·디지털 콘텐츠',5:'서비스·사람·웹 혼합',
    6:'환경·건강·개선',7:'사회·기술·뉴스 혼합',8:'AI·개념·언어모델 혼합',
    9:'시·문학 표현',10:'창작 표현·상황',11:'서사·작품·인물',
    12:'시간·동물·이야기 혼합',
}

# Re-number according to corpus usage, to make S01 the most represented cluster.
freq_by_word = dictionary.set_index('word')['train_n']
raw_cluster_to_group = {
    raw_id:f'S{position+1:02d}' for position,raw_id in enumerate(sorted(
        range(K),key=lambda c:(-sum(float(freq_by_word[words[i]]) for i in np.where(labels==c)[0]),c)))
}
centers = normalize(models[0].cluster_centers_)
sim = np.einsum('ij,ij->i',emb,centers[labels])
cutoffs = {cid:float(np.quantile(sim[labels==cid],.15)) for cid in range(K)}

learned = pd.DataFrame({
    'word':words,'semantic_cluster_raw':labels,
    'semantic_group_id':[raw_cluster_to_group[int(n)] for n in labels],
    'semantic_group_label':[provisional_names.get(int(n),'검토 필요') for n in labels],
    'cosine_to_center':np.round(sim,4),
    'low_similarity_review':[bool(score<cutoffs[int(c)]) for score,c in zip(sim,labels)],
})
dictionary=dictionary.merge(learned,on='word',how='left',validate='one_to_one')
dictionary['classification_status']=np.select([
    dictionary.role_refined.eq('R_URL'),
    dictionary.semantic_group_id.notna() & dictionary.low_similarity_review.fillna(False),
    dictionary.semantic_group_id.notna(),
    dictionary.role_refined.str.startswith('R_'),
    dictionary.train_n.lt(8),
],[
    'URL·잡음 별도','의미 군집 후보·검토 필요',
    '의미 군집 후보','문장 기능·역할 별도',
    '문맥 부족(<8개 질문)',
],default='문맥 부족 또는 미분류')

manual_tag = dictionary.set_index('word')['subject_groups'].to_dict()


def purity_from_labels(selected_words,groups) -> tuple[float,int]:
    pairs = [(str(group),str(manual_tag[w])) for w,group in zip(selected_words,groups)
             if isinstance(manual_tag.get(w),str) and str(group)!='-1']
    by_group = defaultdict(list)
    for group,tag in pairs:by_group[group].append(tag)
    n=sum(len(v) for v in by_group.values())
    if n==0:return float('nan'),0
    return sum(max(pd.Series(t).value_counts()) for t in by_group.values())/n,n

tag_purity_old,tag_count_old=purity_from_labels(np.array(words)[old_present],old_W[old_present])
tag_purity_new,tag_count_new=purity_from_labels(words,labels)

node_rows=[]
for raw_id in range(K):
    selected = dictionary.loc[dictionary.semantic_cluster_raw.eq(raw_id)].sort_values('train_n',ascending=False)
    rep = selected['word'].head(18).tolist()
    center_ranks = np.argsort(sim[labels==raw_id])[::-1]
    subset_words = np.array(words)[labels==raw_id]
    nearest = subset_words[center_ranks][:12].tolist()
    manual_counts = selected.subject_groups.dropna().value_counts()
    node_rows.append({
        'semantic_group_id':raw_cluster_to_group[raw_id],
        'raw_cluster_id':raw_id,'name_provisional':provisional_names[raw_id],
        'distinct_words':len(selected),
        'train_word_question_occurrences':int(selected.train_n.sum()),
        'median_word_question_count':float(selected.train_n.median()),
        'low_similarity_words':int(selected.low_similarity_review.sum()),
        'frequent_words':', '.join(rep),
        'center_nearest_words':', '.join(nearest),
        'manual_topic_evidence':', '.join(f'{k}({v})' for k,v in manual_counts.head(3).items()),
        'manual_topic_tagged_words':int(manual_counts.sum()),
        'review_note':'주제 혼합 의심 / 원문 검토' if '혼합' in provisional_names[raw_id] else '임시 라벨 / 사람이 확인 필요',
    })
nodes=pd.DataFrame(node_rows).sort_values('semantic_group_id').reset_index(drop=True)

dic_map=dictionary.set_index('word')[['semantic_group_id','role_refined']]
joined=tokens.join(dic_map,on='token',validate='many_to_one')
groups=[]
for row in joined.itertuples():
    if isinstance(row.semantic_group_id,str):
        groups.append((row.source_row,row.semantic_group_id))
    if isinstance(row.role_refined,str) and row.role_refined.startswith('R_') and row.role_refined!='R_URL':
        groups.append((row.source_row,row.role_refined))
group_q=pd.DataFrame(groups,columns=['source_row','group_id']).drop_duplicates()
group_q=group_q.merge(qs[['source_row','y']],on='source_row',how='left',validate='many_to_one')
gstats=group_q.groupby('group_id').agg(train_questions=('source_row','nunique'),train_yes=('y','sum')).reset_index()
gstats['train_yes']=gstats.train_yes.astype(int)
gstats['train_no']=gstats.train_questions-gstats.train_yes
gstats['train_yes_rate']=gstats.train_yes/gstats.train_questions
gstats['group_name']=gstats.group_id.map(dict(zip(nodes.semantic_group_id,nodes.name_provisional))|ROLE_NAMES)
gstats['rate_caution']=np.where(gstats.train_questions>=30,'표본 30개 이상','표본 30개 미만: 해석 주의')
gstats=gstats[['group_id','group_name','train_questions','train_yes','train_no','train_yes_rate','rate_caution']]

# Compare fitted group encodings only as an exploratory downstream sanity test.
# Note: the unsupervised word embedding is fitted on all 3605 *training* prompts;
# final results must refit preprocessing within a fold or use untouched test data.
old_token_map=dictionary.set_index('word')['word_cluster'].to_dict()
old_matrix=np.zeros((len(qs),8),dtype='float32')
groups_new=sorted(gstats.group_id.tolist())
new_index={g:i for i,g in enumerate(groups_new)}
new_matrix=np.zeros((len(qs),len(groups_new)),dtype='float32')
for row in tokens.itertuples():
    qpos=question_row[int(row.source_row)]
    w=old_token_map.get(row.token,-1)
    if w>=0:old_matrix[qpos,w]=1
for row in group_q.itertuples():
    new_matrix[question_row[int(row.source_row)],new_index[row.group_id]]=1
role_cols=[new_index[g] for g in groups_new if g.startswith('R_')]
role_matrix=new_matrix[:,role_cols]

splits=list(StratifiedKFold(n_splits=5,shuffle=True,random_state=42).split(old_matrix,qs.y))
cv=[]
for name,features in [('old_W8_presence',old_matrix),('refined_S13_plus_roles',new_matrix),
                      ('old_W8_plus_roles',np.hstack([old_matrix,role_matrix])),
                      ('old_W8_plus_refined',np.hstack([old_matrix,new_matrix]))]:
    for fold,(it,iv) in enumerate(splits,1):
        model=make_pipeline(StandardScaler(),LogisticRegression(max_iter=2500,C=1.0))
        model.fit(features[it],qs.y.iloc[it])
        pp=model.predict_proba(features[iv])[:,1]
        cv.append({'model':name,'fold':fold,'ROC_AUC':roc_auc_score(qs.y.iloc[iv],pp),
                   'PR_AUC':average_precision_score(qs.y.iloc[iv],pp)})
cv_df=pd.DataFrame(cv)
cv_sum=cv_df.groupby('model').agg(ROC_AUC=('ROC_AUC','mean'),PR_AUC=('PR_AUC','mean')).reset_index()

# Audit examples for each group: questions containing several of the central
# words, so a human can determine whether the provisional group is coherent.
audits=[]
for nr in nodes.itertuples():
    subset_words=learned.loc[learned.semantic_group_id.eq(nr.semantic_group_id),'word'].tolist()
    qcounts=joined.loc[joined.token.isin(subset_words)].groupby('source_row').token.nunique().sort_values(ascending=False)
    examples=qs.set_index('source_row').reindex(qcounts.index[:5]).question.fillna('').tolist()
    for j,qtext in enumerate(examples,1):
        audits.append({'semantic_group_id':nr.semantic_group_id,'candidate_label':nr.name_provisional,
                       'rank':j,'question':qtext,'distinct_group_words_in_question':int(qcounts.iloc[j-1])})
audit_df=pd.DataFrame(audits)

report={
    'source_train_questions':len(qs),'source_yes':int(qs.y.sum()),
    'train_observed_vocabulary':len(dictionary),
    'train_questions_containing_semantic_group_words':int(joined.semantic_group_id.notna().groupby(joined.source_row).any().sum()),
    'train_questions_containing_refined_role_words':int(joined.role_refined.str.startswith('R_').groupby(joined.source_row).any().sum()),
    'min_document_frequency_for_context_cluster':8,
    'semantic_word_candidates':len(words),'old_W_assigned_on_same_word_set':int(old_present.sum()),
    'semantic_groups':K,'low_similarity_review_count':int(learned.low_similarity_review.sum()),
    'rare_words_below_8_questions':int((dictionary.train_n<8).sum()),
    'old_W_silhouette_same_labeled_subset':old_sil,
    'candidate_new_K8_silhouette_same_labeled_subset':k8_sil_shared,
    'candidate_new_K8_ARI_stability_3_seeds':k8_stability,
    'new_silhouette_same_labeled_subset':new_sil_shared,
    'new_silhouette_all_candidates':new_sil_all,
    'new_ARI_stability_3_seeds':stability,
    'old_manual_topic_purity_sample':tag_purity_old,
    'new_manual_topic_purity_sample':tag_purity_new,
    'old_manual_topic_tagged_count':tag_count_old,
    'new_manual_topic_tagged_count':tag_count_new,
    'role_override_words':sorted(ROLE_OVERRIDES),
    'url_artifacts':sorted(URL_ARTIFACTS),
    'symbol_words':sorted(SYMBOL_WORDS),
    'metric_warning':'silhouette is geometrical separation, not verified semantic accuracy or prediction lift',
    'label_warning':'word meanings and subject labels are provisional; associations do not establish causality',
    'cv_warning':'internal cross-validation exploratory only; unsupervised vocabulary fitted on full training questions, test 902 labels not used',
}

for name,data in {
    'word_cluster_dictionary_refined':dictionary,
    'semantic_cluster_13_summary':nodes,
    'refined_group_question_frequency':gstats,
    'manual_audit_questions_65':audit_df,
    'model_comparison_5fold':cv_df,
    'model_comparison_summary':cv_sum,
}.items():
    data.to_csv(OUT/f'{name}.csv',index=False,encoding='utf-8-sig')
    data.to_json(OUT/f'{name}.json',orient='records',force_ascii=False)
(OUT/'improvement_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in report.items() if not isinstance(v,list)},ensure_ascii=False,indent=2))
print(nodes[['semantic_group_id','name_provisional','distinct_words','low_similarity_words','frequent_words']].to_string(index=False))
print(cv_sum.to_string(index=False))
