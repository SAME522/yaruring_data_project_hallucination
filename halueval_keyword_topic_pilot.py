"""Topic-keyword weak labeling for the 2,734-query HaluEval mixed cluster.

Separate topic tags (possibly multiple) from existing intent labels. Keep
unmatched queries unclassified. Never read yes/no to assign topic tags.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
SRC=ROOT/'outputs'/'halueval_remask'
OUT=ROOT/'outputs'/'halueval_keyword_topics'

# One-term keywords are direct subject evidence; the list is editable and the
# method deliberately declines to force every prompt into a topic class.
VOCAB={
 '과학·자연':'''science scientific physics chemistry biology biological chemical chemicals
  molecule molecules atomic atom atoms quantum gravity gravitational electromagnetic
  electricity electrons electron proton neutron thermodynamics temperature climate
  weather meteorology astronomy astronomical galaxy galaxies planet planetary planets
  ecosystem ecosystems biodiversity environmental ecology ecological photosynthesis
  fossil fossils volcano volcanic earthquake seismic solar geological geology
  evolution evolutionary genetics genetic genome dna rna enzymes enzyme bacteria
  bacterial species organism organisms oceanography carbon hydrogen oxygen nitrogen
  magnetism astrophysics astronomy greenhouse climate carbon dioxide radiation''',
 '수학·컴퓨팅':'''mathematics mathematical math algebra calculus geometry trigonometry
  equation equations polynomial polynomials derivative derivatives integral integrals
  probability probabilities statistics statistical matrix matrices vector vectors
  python javascript java html css sql programming programmer program programs code coding software
  algorithm algorithms computer computers computing database databases dataset
  recursion recursive binary array arrays variable variables function functions
  machine learning neural network artificial intelligence ai chatbot bots neural
  regression classifier classification tensor tensors optimization cryptography''',
 '의학·건강':'''medicine medical health healthcare diseases disease symptom symptoms
  patient patients diagnosis diagnostic vaccine vaccines virus viruses vaccination
  nutrition nutritional diet dietary calories calorie vitamin vitamins protein
  diabetes cancer tumor tumors depression anxiety psychiatry psychiatric psychology
  psychological mental wellness stress insomnia fitness workout workouts exercise
  exercises blood cholesterol heart cardiovascular obesity infection infections
  hospital hospitals anatomy brain brains neuron neurons immune immunology surgery''',
 '역사·인문':'''history historical historian historians ancient medieval renaissance
  civilization civilizations dynasty dynasties empire empires colonial revolution
  revolutionary industrialization prehistoric archaeology archaeological relic relics
  war wars battle battles monarchy monarch monarchs century centuries timeline
  philosophy philosophical philosopher philosophers ethics ethical religion religious
  bible biblical mythology myth myths constitution constitutional democracy
  enlightenment medieval feudalism roman greek egyptian viking historical''',
 '사회·경제':'''economy economic economics finance financial business businesses
  marketing advertising advertisement advertisements sales customer customers
  products product entrepreneur entrepreneurial startup startups investment investing
  investor investors capital currency currencies inflation inflationary tax taxes
  taxation budget budgets profits profit revenue price prices trade trading
  government politics political election elections policy policies democracy
  law legal legislation society social sociology globalization employer employees''',
 '언어·예술':'''grammar grammatical synonym synonyms antonym antonyms homophone
  homophones linguistics linguistic phonetics pronunciation vocabulary noun nouns
  adjective adjectives adverb adverbs verb verbs metaphor metaphors simile similes
  analogy analogies idiom idioms literature literary poetry poetic novelist
  novel novels fiction fictional symbolism symbolic motif motifs narrative narratology
  painting paintings sculpture sculptures architecture architectural painter
  artworks artwork artists artist art cinematic cinematography melody melodies''',
 '일상·생활':'''cooking recipe recipes ingredient ingredients food foods meal meals
  breakfast lunch dinner kitchen restaurant restaurants grocery groceries shopping
  travel traveling travelling tourism tourist tourists vacation vacations hotel
  hotels destination destinations hobby hobbies gardening garden gardens pet pets
  dog dogs cat cats family families wedding weddings birthday birthdays furniture
  clothing clothes fashion cleaning household home homes school classroom schoolwork
  sport sports basketball football soccer camping hiking transportation commute'''
}
VOCAB={k:set(v.split()) for k,v in VOCAB.items()}
PHRASES={
 '과학·자연':['solar system','climate change','periodic table','greenhouse gas','nuclear fusion','chemical reaction','natural selection','black hole'],
 '수학·컴퓨팅':['machine learning','artificial intelligence','deep learning','neural network','data structure','web development','computer science','data science','linear equation'],
 '의학·건강':['mental health','blood pressure','immune system','heart disease','public health'],
 '역사·인문':['world war','roman empire','ancient greek','civil rights','industrial revolution'],
 '사회·경제':['stock market','supply chain','economic growth','small business','market share'],
 '언어·예술':['literary devices','parts of speech','figurative language','fine arts'],
 '일상·생활':['personal care','meal plan','daily routine','pet care'],
}

# These may be output formats even when the *subject* is scientific, e.g.
# "Write a poem about gravity". Keep them out of subject evidence.
FORMATS=set('''poem poems story stories essay essays paragraph paragraphs table tables
  list lists chart charts graph graphs diagram diagrams script scripts presentation
  presentations report reports summary summaries dialogue dialogues song songs
  lyrics format formats image images infographic poster posters letter letters'''.split())
REQUEST_ANCHORS={'generate','create','write','compose','explain','describe','define',
  'calculate','solve','summarize','list','recommend','suggest','identify','compare'}
SKIP_CANDIDATE={'given','following','new','way','ways','common','different',
  'one','two','three','four','five','ten','many','much','some','other','example',
  'provide','please','make','use','used','using','input','output','question',
  'questions','words','word','short','long','information','help','user','based',
  'can','will','should','topic','topics','possible','good','best','main','first',
  'second','third','types','type','people','someone','person','interesting',
  'related','need','real','https','www','com','html','comprehensive'}

TOKEN_RE=re.compile(r"\b[a-z]+(?:'[a-z]+)?\b",re.I)
PHRASE_PATTERNS={k:[(p,re.compile(r'(?<![a-z])'+re.escape(p)+r'(?![a-z])',re.I)) for p in terms]
                 for k,terms in PHRASES.items()}

def canonical(word:str)->str:
    """Conservative suffix fallback, only when it matches a seed dictionary."""
    if any(word in words for words in VOCAB.values()):return word
    candidates=[]
    if word.endswith('ies') and len(word)>5:candidates.append(word[:-3]+'y')
    if word.endswith('es') and len(word)>5:candidates.append(word[:-2])
    if word.endswith('s') and len(word)>4:candidates.append(word[:-1])
    if word.endswith('ing') and len(word)>6:candidates.append(word[:-3])
    if word.endswith('ed') and len(word)>5:candidates.append(word[:-2])
    for c in candidates:
        if any(c in terms for terms in VOCAB.values()):return c
    return word

def tag(text:str)->tuple[list[str],dict[str,list[str]],list[str]]:
    words=[m.group(0).lower() for m in TOKEN_RE.finditer(text)]
    tokens={canonical(w) for w in words}
    evidence={}
    for cat,seeds in VOCAB.items():
        matches={w for w in tokens if w in seeds and w not in FORMATS}
        matches.update(phrase for phrase,pattern in PHRASE_PATTERNS[cat] if pattern.search(text))
        # "data science" is normally computing/data analysis, not evidence
        # of natural science by itself. Keep science if other evidence exists.
        if cat=='과학·자연' and re.search(r'\bdata science\b',text,re.I):
            matches.discard('science')
        if matches:evidence[cat]=sorted(matches)
    return list(evidence),evidence,words

def save(frame:pd.DataFrame,name:str)->None:
    frame.to_csv(OUT/f'{name}.csv',index=False,encoding='utf-8-sig')
    (OUT/f'{name}.json').write_text(frame.to_json(orient='records',force_ascii=False),encoding='utf-8')

def main()->None:
    OUT.mkdir(parents=True,exist_ok=True)
    df=pd.read_csv(SRC/'question_level_revised_clusters.csv',encoding='utf-8-sig')
    train=df.loc[df.split.eq('train')].copy()
    target=train.loc[train.question_cluster_revised.eq(1)].copy()
    assert len(target)==2734
    tagged=[tag(s) for s in target.question.fillna('')]
    target['topic_tags']=[' / '.join(r[0]) for r in tagged]
    target['topic_count']=[len(r[0]) for r in tagged]
    target['topic_status']=np.select([target.topic_count.eq(0),target.topic_count.eq(1),target.topic_count.ge(2)],
                                    ['미분류','단일 주제','복합 주제'],default='미분류')
    target['matched_keywords']=['; '.join(f'{k}: {", ".join(v)}' for k,v in r[1].items()) for r in tagged]
    target['science_keyword_present']=[int('과학·자연' in r[0]) for r in tagged]
    target['reasoning_note']=np.where(target.topic_count.eq(0),'주제 사전과 일치하는 단어 없음','사전 일치: 사람 검토 필요')
    target['tokenized']= [' '.join(r[2]) for r in tagged]
    save(target[['source_row','original_id','question','split','question_cluster_original',
                 'question_cluster_revised','request_type','topic_status','topic_tags','topic_count',
                 'matched_keywords','science_keyword_present','reasoning_note','tokenized','label','y']],
         'mixed_cluster_question_topic_tags')

    distribution=target.topic_status.value_counts().rename_axis('status').reset_index(name='questions')
    distribution['share']=distribution.questions/len(target)
    save(distribution,'classification_coverage')

    # Multi-label aggregation: one question can contribute to two topics.
    expanded=[]
    for row, result in zip(target.itertuples(index=False),tagged):
        for topic in result[0]:
            expanded.append({'source_row':int(row.source_row),'topic':topic,
                             'evidence':', '.join(result[1][topic]),
                             'request_type':str(row.request_type),
                             'question':str(row.question)})
    assigned=pd.DataFrame(expanded)
    topic_sums=assigned.groupby('topic').agg(questions=('source_row','nunique')).reset_index()
    topic_sums['share_of_all_Q1']=topic_sums.questions/len(target)
    topic_sums=topic_sums.sort_values('questions',ascending=False)
    save(topic_sums,'topic_counts_multilabel')
    save(assigned.sort_values(['topic','source_row']),'topic_question_membership')

    topic_intent=(assigned.groupby(['topic','request_type']).source_row.nunique().rename('questions').reset_index())
    topic_intent=topic_intent.sort_values(['topic','questions'],ascending=[True,False])
    save(topic_intent,'topic_by_request_type')
    sci=assigned.loc[assigned.topic.eq('과학·자연')].sort_values('source_row')
    save(sci,'science_question_examples')
    unclassified=target.loc[target.topic_status.eq('미분류')].copy()
    save(unclassified[['source_row','question','request_type','tokenized','topic_status','reasoning_note']],
         'unclassified_question_examples')

    # Candidate vocabulary for HUMAN REVIEW ONLY. Document-level word
    # enrichment in tagged questions, without hallucination labels.
    # Never silently add suggested words to the tagging dictionary.
    corpus=defaultdict(set)
    for row,result in zip(target.itertuples(index=False),tagged):
        for w in set(result[2]):corpus[canonical(w)].add(int(row.source_row))
    suggestions=[]
    categories=list(VOCAB)
    for topic in categories:
        topic_ids=set(assigned.loc[assigned.topic.eq(topic),'source_row'])
        n_topic=len(topic_ids)
        if n_topic<10:continue
        for word,ids in corpus.items():
            if word in SKIP_CANDIDATE or word in FORMATS or word in REQUEST_ANCHORS:continue
            if any(word in seed for seed in VOCAB.values()):continue
            if len(word)<4 or len(ids)<8:continue
            jointly=len(ids & topic_ids)
            if jointly<3:continue
            lift=(jointly/n_topic)/(len(ids)/len(target))
            if lift<1.5:continue
            suggestions.append({'topic':topic,'candidate_word':word,
                     'questions_in_Q1':len(ids),'tagged_topic_questions':n_topic,
                     'cooccurring_questions':jointly,'cooccurrence_lift':round(lift,3),
                     'status':'후보만 제시: 자동 분류에 사용하지 않음'})
    suggested=pd.DataFrame(suggestions)
    if not suggested.empty:
        suggested=suggested.sort_values(['topic','cooccurrence_lift','cooccurring_questions'],ascending=[True,False,False])
        suggested=suggested.groupby('topic',group_keys=False).head(20)
    save(suggested,'candidate_topic_keywords_for_review')

    lexicon=[]
    for topic,terms in VOCAB.items():
        for word in sorted(terms):lexicon.append({'topic':topic,'keyword':word,'match_type':'single token'})
        for phrase in PHRASES[topic]:lexicon.append({'topic':topic,'keyword':phrase,'match_type':'phrase'})
    save(pd.DataFrame(lexicon),'topic_keyword_dictionary')

    # Visualize coverage and potentially overlapping topic counts.
    fig,axes=plt.subplots(1,2,figsize=(11,4.3))
    ordered=distribution.set_index('status').reindex(['단일 주제','복합 주제','미분류'],fill_value=0)
    names=['One topic','Multiple topics','Unclassified']
    axes[0].bar(names,ordered.questions,color=['#4081a9','#8b80bb','#c1c6cb'])
    axes[0].set(title='Mixed-cluster keyword coverage',ylabel='Questions')
    for i,n in enumerate(ordered.questions):axes[0].text(i,n+12,str(int(n)),ha='center')
    topic_en={'과학·자연':'Natural science','수학·컴퓨팅':'Math/Computing',
       '의학·건강':'Health/Medicine','역사·인문':'History/Humanities','사회·경제':'Social/Business',
       '언어·예술':'Language/Arts','일상·생활':'Everyday life'}
    axes[1].barh([topic_en[x] for x in topic_sums.topic.iloc[::-1]],topic_sums.questions.iloc[::-1],color='#418DB7')
    axes[1].set(title='Topic tags (multi-label)',xlabel='Questions containing topic keywords')
    for ax in axes:ax.spines[['top','right']].set_visible(False)
    fig.tight_layout();fig.savefig(OUT/'01_keyword_topic_coverage.png',dpi=180);plt.close(fig)

    if len(sci):
        count=sci.request_type.value_counts()
        req_en={'창작·생성':'Generate/Create','지식·설명':'Explain/Ask','목록·추천':'List/Recommend',
                '계산·문제풀이':'Calculate/Solve','요약·변환':'Summarize/Rewrite','미분류':'Unclear'}
        fig,ax=plt.subplots(figsize=(7.5,4.5))
        ax.barh([req_en.get(x,x) for x in count.index[::-1]],count.values[::-1],color='#6D9BA8')
        ax.set(title='Science-topic requests: instructions remain separate',xlabel='Questions')
        ax.spines[['top','right']].set_visible(False)
        fig.tight_layout();fig.savefig(OUT/'02_science_question_request_types.png',dpi=180);plt.close(fig)

    report={
      'population':'revised question cluster Q1, training subset',
      'questions':int(len(target)),
      'status_counts':{str(r.status):int(r.questions) for r in distribution.itertuples()},
      'tagged_any':int(target.topic_count.ge(1).sum()),
      'tagged_any_share':round(float(target.topic_count.ge(1).mean()),4),
      'topic_count_multilabel':{str(r.topic):int(r.questions) for r in topic_sums.itertuples()},
      'science_request_count':{str(k):int(v) for k,v in sci.request_type.value_counts().items()},
      'previous_rule_unclassified_questions_in_Q1':int(train.loc[train.question_cluster_revised.eq(1)].topic.eq('미분류').sum()),
      'candidate_new_words_for_human_review':int(len(suggested)),
      'note':'Only explicit lexicon matches are tagged. Tags are tentative and can overlap; no forced assignment or hallucination label use. Coverage is not accuracy. Topic word suggestions are not auto-applied.'
    }
    (OUT/'topic_labeling_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
