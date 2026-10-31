"""Stage 1: word-frequency profiles *within* existing HaluEval question clusters.

Uses the previous question cluster Q (0–5), word-context cluster W (0–7,
-1 for unassigned), and word-role R. The input is the existing TRAIN split.
We do NOT use hallucination Yes/No labels during this frequency analysis.

Run: python analyze_halueval_cluster_frequencies.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
SRC=ROOT / 'outputs' / 'halueval_cross'
OUT=ROOT / 'outputs' / 'halueval_freq'

Q_LABELS={
  0:'시·비유·구조화 중심',
  1:'목록·일상 제안 중심',
  2:'문장·형식 변환 중심',
  3:'여러 유형이 섞인 대규모 혼합',
  4:'예시·상황 제시 중심',
  5:'서사·형식 생성 중심',
}
ROLE_EN={
    '요청 동사':'Action', '주제 전문어':'Topic term', '조건·제약어':'Constraint',
    '출력 형식어':'Output format', '의문·질문어':'Question word',
    '숫자·수치':'Number', '기타 내용어':'Other content',
}


def save(frame:pd.DataFrame,filename:str)->None:
    frame.to_csv(OUT / f'{filename}.csv',index=False,encoding='utf-8-sig')
    (OUT / f'{filename}.json').write_text(frame.to_json(orient='records',force_ascii=False),encoding='utf-8')


def main()->None:
    OUT.mkdir(parents=True,exist_ok=True)
    questions=pd.read_csv(SRC/'question_cross_features.csv',encoding='utf-8-sig')
    tokens=pd.read_csv(SRC/'tokens_after_stopwords.csv',encoding='utf-8-sig')
    assert len(questions)==4507
    assert not tokens.is_excluded.any()
    train_q=questions.loc[questions.split.eq('train')].copy()
    train_tokens=tokens.loc[tokens.split.eq('train')].copy()
    n_q=train_q.groupby('question_cluster').source_row.nunique().rename('cluster_question_count')
    assert int(n_q.sum())==3605
    assert set(train_tokens.source_row)<=set(train_q.source_row)

    # Count a word once per question for document frequency (DF); raw token
    # count (TF) is kept separately so repeated words aren't hidden.
    uniq=train_tokens[['question_cluster','source_row','token','word_cluster','word_role','word_topic']].drop_duplicates(['source_row','token'])
    q_tf=(train_tokens.groupby(['question_cluster','token']).size().rename('token_occurrences').reset_index())
    q_df=(uniq.groupby(['question_cluster','token']).size().rename('document_frequency').reset_index())
    overall_df=uniq.groupby('token').size().rename('overall_document_frequency')
    metadata=(uniq[['token','word_cluster','word_role','word_topic']].drop_duplicates('token'))
    assert metadata.token.nunique()==len(metadata)

    word_stats=q_tf.merge(q_df,on=['question_cluster','token'],validate='one_to_one')
    word_stats=word_stats.merge(metadata,on='token',validate='many_to_one')
    word_stats=word_stats.merge(n_q,on='question_cluster',validate='many_to_one')
    word_stats['overall_document_frequency']=word_stats.token.map(overall_df).astype(int)
    word_stats['document_frequency_per_100_questions']=(100*word_stats.document_frequency/word_stats.cluster_question_count).round(2)
    word_stats['token_occurrences_per_100_questions']=(100*word_stats.token_occurrences/word_stats.cluster_question_count).round(2)
    word_stats['cluster_document_share']=(word_stats.document_frequency/word_stats.cluster_question_count).round(6)
    word_stats['overall_document_share']=(word_stats.overall_document_frequency/len(train_q)).round(6)
    word_stats['relative_frequency_lift']=((word_stats.document_frequency/word_stats.cluster_question_count)
         /(word_stats.overall_document_frequency/len(train_q))).round(3)
    word_stats['minimum_sample_check']=np.where((word_stats.document_frequency>=5)&(word_stats.overall_document_frequency>=15),'대표 비교 가능','낮은 빈도·해석 주의')
    word_stats['question_cluster_label']=word_stats.question_cluster.map(Q_LABELS)
    word_stats=word_stats.rename(columns={'token':'word'}).sort_values(
        ['question_cluster','document_frequency','word'],ascending=[True,False,True]).reset_index(drop=True)
    save(word_stats,'all_words_by_question_cluster')

    # Frequent and distinctive are different questions. Frequency is the
    # number of distinct questions that contain the term, not the TF count.
    frequency_ranks=[]
    distinctive_ranks=[]
    for q in sorted(n_q.index):
        g=word_stats.loc[word_stats.question_cluster.eq(q)].copy()
        top=g.sort_values(['document_frequency','word'],ascending=[False,True]).head(25).copy()
        top.insert(0,'rank',range(1,len(top)+1))
        frequency_ranks.append(top)
        # Restrict enrichment ranking to a minimum of 5 questions in the
        # cluster and 15 globally to reduce one-off/rare-word artifacts.
        qualified=g.loc[g.document_frequency.ge(5)&g.overall_document_frequency.ge(15)]
        distinct=qualified.sort_values(['relative_frequency_lift','document_frequency','word'],ascending=[False,False,True]).head(25).copy()
        distinct.insert(0,'rank',range(1,len(distinct)+1))
        distinctive_ranks.append(distinct)
    top_freq=pd.concat(frequency_ranks,ignore_index=True)
    top_lift=pd.concat(distinctive_ranks,ignore_index=True)
    save(top_freq,'top_words_by_frequency')
    save(top_lift,'distinctive_words_by_lift')

    # Q×W×R joint incidence. A question can contain multiple intersections;
    # each intersection has its own question count denominator.
    qwr_keys=['question_cluster','word_cluster','word_role']
    role_tf=(train_tokens.groupby(qwr_keys).size().rename('token_occurrences').reset_index())
    role_question=(train_tokens[qwr_keys+['source_row']].drop_duplicates()
        .groupby(qwr_keys).size().rename('questions_containing_group').reset_index())
    qwr=role_tf.merge(role_question,on=qwr_keys,validate='one_to_one').merge(n_q,on='question_cluster',validate='many_to_one')
    qwr['questions_per_100']=(100*qwr.questions_containing_group/qwr.cluster_question_count).round(2)
    qwr['token_occurrences_per_100_questions']=(100*qwr.token_occurrences/qwr.cluster_question_count).round(2)
    qwr['group_label']=('Q'+qwr.question_cluster.astype(str)+' × W'+qwr.word_cluster.astype(str)+' × '+qwr.word_role)
    qwr['question_cluster_label']=qwr.question_cluster.map(Q_LABELS)
    qwr=qwr.sort_values(['question_cluster','questions_containing_group','word_cluster','word_role'],ascending=[True,False,True,True])
    save(qwr,'question_word_role_frequency')

    role_keys=['question_cluster','word_role']
    rtf=train_tokens.groupby(role_keys).size().rename('token_occurrences').reset_index()
    rdf=(train_tokens[role_keys+['source_row']].drop_duplicates()
         .groupby(role_keys).size().rename('questions_containing_role').reset_index())
    role=rtf.merge(rdf,on=role_keys).merge(n_q,on='question_cluster')
    role['role_question_share']=(role.questions_containing_role/role.cluster_question_count).round(5)
    role['role_tokens_per_100_questions']=(100*role.token_occurrences/role.cluster_question_count).round(2)
    role['role_questions_per_100']=(100*role.questions_containing_role/role.cluster_question_count).round(2)
    role['question_cluster_label']=role.question_cluster.map(Q_LABELS)
    save(role.sort_values(['question_cluster','questions_containing_role'],ascending=[True,False]),'word_roles_by_question_cluster')

    summary=(train_q.groupby('question_cluster').agg(question_count=('source_row','size'),
             average_question_tokens=('token_count','mean'),average_kept_tokens=('kept_token_count','mean'))
             .reset_index())
    grouped_word=train_tokens.groupby('question_cluster').agg(
        kept_token_occurrences=('token','size'),distinct_vocabulary=('token','nunique')).reset_index()
    summary=summary.merge(grouped_word,on='question_cluster')
    summary['average_question_tokens']=summary.average_question_tokens.round(2)
    summary['average_kept_tokens']=summary.average_kept_tokens.round(2)
    summary['question_cluster_label']=summary.question_cluster.map(Q_LABELS)
    summary['top_10_frequent_words']=summary.question_cluster.map(
        top_freq.loc[top_freq['rank'].le(10)].groupby('question_cluster').word.agg(', '.join))
    summary['top_10_distinctive_words']=summary.question_cluster.map(
        top_lift.loc[top_lift['rank'].le(10)].groupby('question_cluster').word.agg(', '.join))
    save(summary,'question_cluster_frequency_summary')

    # Science terms are a specific example of the requested word-within-Q
    # comparison; include all Q intersections rather than preselecting one.
    scientific=word_stats.loc[word_stats.word_topic.fillna('').str.contains('과학·자연',regex=False)].copy()
    save(scientific.sort_values(['document_frequency','question_cluster'],ascending=[False,True]),'science_terms_by_question_cluster')

    # Visualization 1: use normalized question incidence for fair cluster-size
    # comparisons; panels are English-only because local render fonts lack KR.
    fig,axes=plt.subplots(2,3,figsize=(14.8,10.2))
    for q,ax in zip(range(6),axes.ravel()):
        g=top_freq.loc[top_freq.question_cluster.eq(q)].head(12).iloc[::-1]
        ax.barh(g.word,g.document_frequency_per_100_questions,color='#418DB7')
        ax.set(title=f'Q{q} | {int(n_q[q]):,} questions',xlabel='Questions with word / 100 questions')
        ax.tick_params(axis='y',labelsize=9)
        ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Most frequent words within each original question cluster',fontsize=15)
    fig.tight_layout();fig.savefig(OUT/'01_top_words_per_cluster.png',dpi=170);plt.close(fig)

    fig,axes=plt.subplots(2,3,figsize=(14.8,10.2))
    for q,ax in zip(range(6),axes.ravel()):
        g=top_lift.loc[top_lift.question_cluster.eq(q)].head(10).iloc[::-1]
        ax.barh(g.word,g.relative_frequency_lift,color='#6F9F80')
        ax.axvline(1.0,color='#777777',ls='--',lw=1)
        ax.set(title=f'Q{q} | lift vs entire training set',xlabel='Document-frequency lift (1 = average)')
        ax.tick_params(axis='y',labelsize=9)
        ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Words unusually common in each question cluster (n≥5 local, n≥15 overall)',fontsize=14)
    fig.tight_layout();fig.savefig(OUT/'02_distinctive_words_lift.png',dpi=170);plt.close(fig)

    matrix=role.pivot(index='question_cluster',columns='word_role',values='role_questions_per_100').fillna(0)
    fig,ax=plt.subplots(figsize=(11,5))
    im=ax.imshow(matrix.values,cmap='Blues',vmin=0,vmax=100)
    ax.set_xticks(np.arange(matrix.shape[1]),[ROLE_EN.get(x,x) for x in matrix.columns],rotation=25,ha='right')
    ax.set_yticks(np.arange(matrix.shape[0]),[f'Q{k}' for k in matrix.index])
    ax.set(title='Word roles in original question clusters',ylabel='Question cluster')
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value=matrix.iat[i,j]
            ax.text(j,i,f'{value:.1f}%',ha='center',va='center',color='white' if value>58 else '#163653',fontsize=8)
    fig.colorbar(im,ax=ax,shrink=.78,label='Questions with this role (%)')
    fig.tight_layout();fig.savefig(OUT/'03_question_role_frequency.png',dpi=170);plt.close(fig)

    report={
      'train_questions':int(len(train_q)),
      'test_questions_untouched':int(questions.split.eq('test').sum()),
      'cluster_sizes':{str(k):int(v) for k,v in n_q.items()},
      'kept_train_token_occurrences':int(len(train_tokens)),
      'distinct_kept_words_train':int(train_tokens.token.nunique()),
      'word_cluster_pairs':int(len(word_stats)),
      'Q_W_R_intersections_observed':int(len(qwr)),
      'science_word_cluster_pairs':int(len(scientific)),
      'frequency_definition':'TF counts every token occurrence; DF counts each question at most once per word.',
      'lift_definition':'(DF_in_Q / questions_in_Q) / (DF_overall / all_train_questions).',
      'top_frequency_per_cluster':{str(q):top_freq.loc[top_freq.question_cluster.eq(q),'word'].head(10).tolist() for q in range(6)},
      'top_lift_per_cluster':{str(q):top_lift.loc[top_lift.question_cluster.eq(q),'word'].head(10).tolist() for q in range(6)},
      'note':'No hallucination labels were used to rank words. Association with yes/no is reserved for the next analysis stage.'
    }
    (OUT/'frequency_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
