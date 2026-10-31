"""Consolidate HaluEval word clusters, roles, topic lexicons and group keywords.

Important: The W cluster, manual topic dictionary and words describing Q groups
are separate feature axes. Train outcomes are used ONLY for post-hoc frequency
and rate summaries, never for assigning new clusters.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
PREP=ROOT/'outputs'/'halueval_prep'
CROSS=ROOT/'outputs'/'halueval_cross'
FREQ=ROOT/'outputs'/'halueval_freq'
TOP=ROOT/'outputs'/'halueval_keyword_topics'
EXP=ROOT/'outputs'/'halueval_topic_expansion'
SECOND=ROOT/'outputs'/'halueval_secondpass_density'
THIRD=ROOT/'outputs'/'halueval_thirdpass_459'
OUT=ROOT/'outputs'/'halueval_unified_word_clusters'
OUT.mkdir(parents=True,exist_ok=True)

def read(path):return pd.read_csv(path,encoding='utf-8-sig')

def save(name,frame):
    frame.to_csv(OUT/f'{name}.csv',index=False,encoding='utf-8-sig')
    (OUT/f'{name}.json').write_text(frame.to_json(orient='records',force_ascii=False),encoding='utf-8')

def main():
    words=read(CROSS/'word_hierarchical_dictionary.csv')
    tokens=read(CROSS/'tokens_after_stopwords.csv')
    excluded=read(CROSS/'excluded_words_with_counts.csv')
    existing=read(TOP/'topic_keyword_dictionary.csv')
    extended=read(EXP/'new_group_dictionary.csv')
    q=read(PREP/'question_features_for_regression.csv')
    integrated=read(THIRD/'integrated_2734_with_three_stages.csv')
    assert len(words)==5650 and words.word.is_unique
    assert len(excluded)==90 and len(integrated)==2734
    assert q.source_row.is_unique
    trainq=q.loc[q.split.eq('train'),['source_row','y']]
    assert len(trainq)==3605
    q1ids=set(integrated.source_row)
    tr=tokens.loc[tokens.split.eq('train')].copy()
    tr=tr.merge(trainq,on='source_row',how='left',validate='many_to_one')
    assert tr.y.notna().all() and tr.source_row.nunique()<=len(trainq)
    tr['q1']=tr.source_row.isin(q1ids)

    base=words.set_index('word')
    vocab_names={
      -1:'기존 W 미할당',0:'문장·서사',1:'개념·아이디어',2:'조건·구성',
      3:'기사·요약',4:'사례·데이터',5:'코드·계산',6:'목록·생활',7:'시·표현'
    }

    tag_group=defaultdict(set);style_group=defaultdict(set)
    lexicon=[]
    for r in existing.itertuples(index=False):
        term=str(r.keyword).strip().lower()
        tag_group[term].add(str(r.topic))
        lexicon.append({'vocabulary_axis':'기존 7개 주제어','group':r.topic,
            'term':term,'term_type':'구절' if ' ' in term else '단어',
            'source':'초기 주제 사전'})
    for r in extended.itertuples(index=False):
        term=str(r.keyword).strip().lower()
        if r.kind=='new_topic':
            tag_group[term].add(str(r.group));axis='추가 7개 주제어'
        else:
            style_group[term].add(str(r.group));axis='출력·표현 형식어'
        lexicon.append({'vocabulary_axis':axis,'group':r.group,
            'term':term,'term_type':'구절' if ' ' in term or '-' in term else '단어',
            'source':'주제 사전 확장'})
    lex=pd.DataFrame(lexicon).drop_duplicates().sort_values(['vocabulary_axis','group','term'])
    save('manual_topic_and_style_lexicon',lex)

    wcontext=read(CROSS/'word_role_context_groups.csv')
    context=read(CROSS/'Q_question_W_word_R_role_cross.csv')
    wsummary=[]
    for wid,grp in tr.groupby('word_cluster'):
        doc=grp[['source_row','y']].drop_duplicates('source_row')
        wsummary.append({'word_cluster':int(wid),'word_cluster_name':vocab_names.get(int(wid),'미정'),
           'distinct_words':int(grp.token.nunique()),'token_occurrences':int(len(grp)),
           'train_questions_with_any_word':len(doc),'train_yes':int(doc.y.sum()),
           'train_no':int(len(doc)-doc.y.sum()),
           'train_yes_rate':float(doc.y.mean()),'note':'군집에 포함된 단어 1개 이상 사용한 질문'})
    save('W_cluster_summary',pd.DataFrame(wsummary).sort_values('word_cluster'))
    save('W_by_role_cross',wcontext.sort_values(['word_cluster','train_n'],ascending=[True,False]))
    save('Q_by_W_by_role_cross',context.sort_values(['question_cluster','train_n'],ascending=[True,False]))

    # Representative term from second- and third-pass *question* clusters.
    # These are descriptive lexical markers, not memberships of W word groups.
    reps=[]
    second=read(SECOND/'cluster_term_dictionary.csv')
    for r in second.itertuples(index=False):
        reps.append({'stage':'2차 질문 유사도','question_group_id':f'2-{int(r.subcluster_id)}',
            'group_name':str(r.proposed_theme),'term':str(r.word_or_phrase).lower(),
            'rank':int(r.rank),'term_type':'구절' if ' ' in str(r.word_or_phrase) else '단어'})
    third=read(THIRD/'thirdpass_cluster_summary_draft.csv')
    for r in third.itertuples(index=False):
        for rank,term in enumerate(str(r.top_words).split(', '),1):
            reps.append({'stage':'3차 질문 유사도','question_group_id':f'3-{int(r.thirdpass_cluster)}',
                'group_name':str(r.candidate_label),'term':term.lower().strip(),
                'rank':rank,'term_type':'구절' if ' ' in term else '단어'})
    rep=pd.DataFrame(reps).sort_values(['stage','question_group_id','rank'])
    save('question_cluster_representative_terms',rep)
    rep_by_word=defaultdict(list)
    for r in rep.loc[rep.term_type.eq('단어')].itertuples(index=False):
        rep_by_word[str(r.term)].append((r.stage,r.question_group_id,r.group_name,int(r.rank)))

    counts=tr.groupby('token').agg(train_occurrences=('token','size'),
                    train_questions=('source_row','nunique')).reset_index().rename(columns={'token':'word'})
    yes_doc=tr.loc[tr.y.eq(1)].groupby('token').source_row.nunique()
    counts['train_yes']=counts.word.map(yes_doc).fillna(0).astype(int)
    counts['train_no']=counts.train_questions-counts.train_yes
    counts['train_yes_rate']=counts.train_yes/counts.train_questions

    sub=tr.loc[tr.q1]
    q1=sub.groupby('token').agg(Q1_occurrences=('token','size'),Q1_questions=('source_row','nunique')).reset_index().rename(columns={'token':'word'})
    q1_yes=sub.loc[sub.y.eq(1)].groupby('token').source_row.nunique()
    q1['Q1_yes']=q1.word.map(q1_yes).fillna(0).astype(int)
    q1['Q1_no']=q1.Q1_questions-q1.Q1_yes
    q1['Q1_yes_rate']=q1.Q1_yes/q1.Q1_questions

    # Exact observed words + manual topic/style lexicon terms (including
    # unseen words) + excluded words; multi-word phrases live only in lexicon.
    unique_terms=set(base.index)
    unique_terms.update(x for x in tag_group if ' ' not in x and '-' not in x)
    unique_terms.update(x for x in style_group if ' ' not in x and '-' not in x)
    unique_terms.update(excluded.excluded_word.astype(str))
    master=pd.DataFrame({'word':sorted(unique_terms)})
    master=master.join(base[['word_cluster','word_role','word_topic','train_n','hierarchical_cluster']],on='word')
    master=master.merge(counts,on='word',how='left',validate='one_to_one')
    master=master.merge(q1,on='word',how='left',validate='one_to_one')
    miss=excluded.set_index('excluded_word')
    master['excluded_from_analysis']=master.word.isin(set(miss.index))
    master['exclusion_reason']=master.word.map(miss.exclusion_reason)
    master['excluded_token_occurrences']=master.word.map(miss.removed_occurrences).fillna(0).astype(int)
    master['excluded_questions_affected']=master.word.map(miss.affected_questions).fillna(0).astype(int)
    master['W_group_name']=master.word_cluster.map(vocab_names)
    master['subject_groups']=master.word.map(lambda s:' / '.join(sorted(tag_group.get(s,[]))))
    master['format_groups']=master.word.map(lambda s:' / '.join(sorted(style_group.get(s,[]))))
    master['representative_Q_groups']=master.word.map(
      lambda s:'; '.join(f'{qg} {name} (top {rank})' for _stage,qg,name,rank in rep_by_word.get(s,[])))
    master['has_topic_or_format_seed']=master.subject_groups.ne('')|master.format_groups.ne('')
    master['review_note']=np.select([
        master.excluded_from_analysis,
        master.word_cluster.isna(),
        master.word_cluster.eq(-1),
        master.representative_Q_groups.ne('')],
        ['원문 유지·분석 토큰만 제외','사전에 있지만 학습 문항에는 안 나옴',
         '기존 W 미할당: 주제어 자동 부여 아님',
         '질문군집 대표어: W 배정과 별도'],default='기존 W 어휘군집')
    for col in ['train_occurrences','train_questions','train_yes','train_no',
                'Q1_occurrences','Q1_questions','Q1_yes','Q1_no']:
        master[col]=master[col].fillna(0).astype(int)
    master['W_group_id']=master.word_cluster.map(lambda w: '해당 없음' if pd.isna(w) else f'W{int(w)}')
    master['word_role']=master.word_role.fillna('해당 없음')
    master['word_topic']=master.word_topic.fillna('해당 없음')
    master['train_yes_rate']=master.train_yes_rate.fillna(np.nan)
    master['Q1_yes_rate']=master.Q1_yes_rate.fillna(np.nan)
    master=master.sort_values(['train_questions','train_occurrences','word'],ascending=[False,False,True]).reset_index(drop=True)
    master.insert(0,'frequency_rank',range(1,len(master)+1))
    save('unified_word_index',master)

    # Safe comparisons use a per-question denominator and sample support.
    eligible=master.loc[(master.train_questions>=10)&(~master.excluded_from_analysis)].copy()
    eligible=eligible.sort_values(['train_yes_rate','train_questions'],ascending=[False,False])
    save('words_min_10_questions',eligible)
    top=master.loc[(master.train_questions>=30)&(~master.excluded_from_analysis)].sort_values('train_questions',ascending=False)
    save('frequent_words_30plus',top)
    save('top_words_by_original_Q',read(FREQ/'top_words_by_frequency.csv'))
    save('excluded_90_words',excluded)

    report={'training_questions':int(len(trainq)),'train_yes_questions':int(trainq.y.sum()),
      'original_unified_words':int(len(base)),
      'master_unique_terms':int(len(master)),
      'excluded_words':int(len(excluded)),
      'manual_topic_or_style_terms':int(len(lex)),
      'representative_term_occurrences':int(len(rep)),
      'W_clusters':8,
      'W_minus_one':'Unassigned to original W lexical clusters; not a substantive topic class',
      'Q1_size':int(len(integrated)),
      'Q1_label_positive':int(integrated.y.sum()),
      'scope':'Trained word/label statistical counts on 3605 training questions; Q1 counts on 2734 revised mixed-cluster train questions; zero test outcomes accessed',
      'interpretation':'Word frequency and conditional hallucination rates are associations, not causal effects. Topic labels are manual keywords; second/third-pass representative terms are NOT W word cluster membership.',
      'note':'Stopword exclusions affect only analytic tokens; original questions and labels remain intact.'}
    (OUT/'consolidation_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
