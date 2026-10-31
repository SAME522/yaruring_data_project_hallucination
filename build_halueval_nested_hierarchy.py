"""Experimental, label-blind, nested question clusters for HaluEval.

Three fitted thematic levels (KMeans fitted inside each parent), followed by
an independent rule-derived request-intent grouping.  Ground-truth y is never
used to construct clusters or choose their parameters; it only summarizes the
training distribution after clustering.  The reserved test y is not exported.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.cluster import BisectingKMeans, KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize


ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'outputs' / 'halueval_nested_hierarchy'
OUT.mkdir(parents=True, exist_ok=True)
QUESTIONS = ROOT / 'outputs/halueval_prep/question_features_for_regression.csv'
TOKENS = ROOT / 'outputs/halueval_cross/tokens_after_stopwords.csv'
MANUAL = ROOT / 'outputs/halueval_thirdpass_459/integrated_2734_with_three_stages.csv'
EXCLUSIONS = ROOT / 'outputs/halueval_cross/excluded_words_with_counts.csv'

SEED = 20261031
TOP_K = 9
MIDDLE_K = 3
MIN_DEEP_SPLIT = 35

# Common boilerplate is removed *only from the thematic clustering features*;
# the request type is retained as a separate, explicit fourth hierarchy level.
GENERIC_REQUEST_WORDS = set('''
    please can could would should tell give provide write create generate make
    produce list explain describe discuss summarize summary compare example examples
    question questions answer answers help want need ask asked show respond response
    provide include including based following given specify specified using used
    information details detail something anything someone thing things kind kinds
    several few some many one two three also about regarding related
'''.split())

INTENT_LABELS = {
    'creative_writing': '창작·생성',
    'explain_or_compare': '설명·비교',
    'list_or_recommend': '목록·추천',
    'math_or_programming': '계산·프로그래밍',
    'rewrite_or_summarize': '수정·요약',
    'other_or_unclear': '기타·미확정 요청',
}

raw = pd.read_csv(QUESTIONS, encoding='utf-8-sig').sort_values('source_row').reset_index(drop=True)
assert len(raw) == 4507, len(raw)
train = raw['split'].eq('train').to_numpy()
assert train.sum() == 3605 and (~train).sum() == 902
assert raw.loc[train, 'y'].sum() == 652
assert raw['source_row'].is_unique

excluded = pd.read_csv(EXCLUSIONS, encoding='utf-8-sig')
common_words = set(excluded['excluded_word'].astype(str).str.lower())
stop_words = sorted(common_words | GENERIC_REQUEST_WORDS)

vectorizer = TfidfVectorizer(
    lowercase=True, strip_accents='unicode', stop_words=stop_words,
    token_pattern=r'(?u)\b[a-z][a-z0-9]+\b',
    ngram_range=(1, 2), min_df=3, max_df=0.60,
    sublinear_tf=True, max_features=16000,
)
texts = raw['question'].fillna('').astype(str)
x_sparse = vectorizer.fit_transform(texts[train])
if x_sparse.shape[1] < 2:
    raise ValueError('Not enough thematic vocabulary')
svd = TruncatedSVD(n_components=min(90, x_sparse.shape[1] - 1), n_iter=8, random_state=SEED)
x_train = normalize(svd.fit_transform(x_sparse)).astype('float32')
x_test_sparse = vectorizer.transform(texts[~train])
x_test = normalize(svd.transform(x_test_sparse)).astype('float32')
x_all = np.empty((len(raw), x_train.shape[1]), dtype='float32')
x_all[train] = x_train
x_all[~train] = x_test
feature_names = np.array(vectorizer.get_feature_names_out())

# A truly hierarchical family: every child is fitted only within its parent,
# not a collection of unrelated KMeans outcomes subsequently cross-tabulated.
node_members: dict[str, np.ndarray] = {}
node_depth: dict[str, int] = {}
node_parent: dict[str, str] = {}
centroids: dict[str, np.ndarray] = {}
all_positions = np.arange(len(raw))
train_positions = all_positions[train]
test_positions = all_positions[~train]
node_members['ROOT'] = train_positions.copy()
node_depth['ROOT'] = 0
node_parent['ROOT'] = ''
centroids['ROOT'] = normalize(x_train.mean(axis=0).reshape(1, -1))[0]

per_question_paths: dict[int, list[str]] = {int(i): [] for i in all_positions}


def fit_children(parent_id: str, groups: int, depth: int):
    members = node_members[parent_id]
    if len(members) < max(2 * MIN_DEEP_SPLIT, groups * 13):
        return []
    groups = min(groups, len(members) // 13)
    if groups < 2:
        return []
    if depth == 1:
        # Recursive splits of the largest remaining group avoid the massive
        # catchall branch produced by ordinary flat KMeans on this corpus.
        fitted = BisectingKMeans(
            n_clusters=groups, bisecting_strategy='largest_cluster',
            random_state=SEED, n_init=12,
        )
    else:
        fitted = KMeans(n_clusters=groups, random_state=SEED + depth * 17 + len(members), n_init=12)
    train_assignments = fitted.fit_predict(x_all[members])
    # Assign reserved test data by traversing only learned training centroids.
    # Parent test path was decided in the preceding stage.
    parent_test = np.array([
        i for i in test_positions
        if (parent_id == 'ROOT' or (per_question_paths[int(i)] and per_question_paths[int(i)][-1] == parent_id))
    ], dtype='int32')
    test_assignments = fitted.predict(x_all[parent_test]) if len(parent_test) else np.array([], dtype=int)

    children = []
    # Sort child IDs by size, not arbitrary KMeans numeric labels.
    old_ids = sorted(range(groups), key=lambda c: (-int((train_assignments == c).sum()), c))
    for ordinal, old in enumerate(old_ids, 1):
        cid = f'T{ordinal:02d}' if depth == 1 else f'{parent_id}.{ordinal:02d}'
        child_train = members[train_assignments == old]
        if not len(child_train):
            continue
        node_members[cid] = child_train
        node_depth[cid] = depth
        node_parent[cid] = parent_id
        centroids[cid] = normalize(x_all[child_train].mean(axis=0).reshape(1, -1))[0]
        children.append(cid)
        for i in child_train:
            per_question_paths[int(i)].append(cid)
        for i in parent_test[test_assignments == old]:
            per_question_paths[int(i)].append(cid)
    return children


top_nodes = fit_children('ROOT', TOP_K, 1)
middle_nodes = []
for tid in top_nodes:
    middle_nodes.extend(fit_children(tid, MIDDLE_K, 2))

small_nodes = []
for mid in middle_nodes:
    size = len(node_members[mid])
    k = 3 if size >= 105 else 2 if size >= 2 * MIN_DEEP_SPLIT else 1
    small_nodes.extend(fit_children(mid, k, 3))

# No third split is silently invented for small subsets; a terminal mid node
# remains a terminal node and can still have request-intent leaf children.
terminal_topic_nodes = small_nodes + [mid for mid in middle_nodes if not any(node_parent.get(s) == mid for s in small_nodes)]
for i in all_positions:
    if not per_question_paths[int(i)]:
        raise AssertionError(f'Question {i} has no thematic branch')

topic_text_matrix: csr_matrix = vectorizer.transform(texts)
root_tf = np.asarray(topic_text_matrix[train].sum(axis=0)).ravel()
total_docs = int(train.sum())
train_y = raw['y'].to_numpy()


def top_terms(ids: np.ndarray, count: int = 6) -> list[str]:
    if len(ids) == 0:
        return []
    sub = topic_text_matrix[ids]
    weights = np.asarray(sub.sum(axis=0)).ravel()
    denom = root_tf / total_docs + 0.02
    salient = (weights / max(len(ids), 1)) / denom
    salient *= np.sqrt(np.maximum(weights, 0))
    # Group labels are exploratory: prioritize strongly distinctive content,
    # but penalize a phrase seen in <3 distinct questions in the node.
    present = np.asarray(sub.getnnz(axis=0)).ravel()
    salient[present < min(3, len(ids))] = 0.0
    chosen = []
    for idx in np.argsort(salient)[::-1]:
        term = str(feature_names[idx])
        if salient[idx] <= 0:
            break
        if any(term in other.split() and ' ' not in term for other in chosen):
            continue
        if any(term == other or term in other for other in chosen):
            continue
        chosen.append(term)
        if len(chosen) >= count:
            break
    return chosen


def word_cluster_summary(positions: np.ndarray) -> tuple[str, str]:
    cols = [f'word_cluster_{j}_share' for j in range(8)]
    shares = raw.iloc[positions][cols].mean().to_numpy(dtype=float)
    top = np.argsort(shares)[::-1][:3]
    return ', '.join(f'W{i}' for i in top), ', '.join(f'W{i}: {shares[i]:.1%}' for i in top)


def score_label(cid: str) -> str:
    terms = top_terms(node_members[cid], 4)
    return (' / '.join(terms[:3]) if terms else '의미 단서 부족')


name_by_cluster = {c: score_label(c) for c in node_members if c != 'ROOT'}
child_of = {k: [] for k in node_members}
for cid, parent in node_parent.items():
    if parent:
        child_of[parent].append(cid)

manual = pd.read_csv(MANUAL, encoding='utf-8-sig')
manual = manual[['source_row', 'expanded_topic_tags', 'expanded_topic_status', 'output_style_tags']]
merged = raw.merge(manual, on='source_row', how='left', validate='one_to_one')

rows = []
intent_rows = []
for cid in ['ROOT'] + top_nodes + middle_nodes + small_nodes:
    pos = node_members[cid]
    sub = merged.iloc[pos]
    yes = int(sub['y'].sum())
    total = len(pos)
    rep = np.argsort(x_all[pos] @ centroids[cid])[::-1][:3]
    examples = [str(texts.iloc[pos[j]]) for j in rep]
    wlabels, wshare = word_cluster_summary(pos)
    terms = top_terms(pos)
    depth = node_depth[cid]
    is_terminal = not child_of[cid]
    rates_note = '30개 미만: 수치 주의' if total < 30 else ''
    rows.append({
        'depth': depth, 'path': cid,
        'parent_path': node_parent[cid],
        'kind': '전체' if cid == 'ROOT' else ('대분류' if depth == 1 else '중분류' if depth == 2 else '소분류'),
        'keyword_label_draft': '전체 질문' if cid == 'ROOT' else name_by_cluster[cid],
        'label_status': '자동 추출·검토 필요' if cid != 'ROOT' else '전체',
        'train_questions': total, 'train_yes': yes, 'train_no': total - yes,
        'train_yes_rate': yes / total if total else None,
        'children': len(child_of[cid]),
        'top_topic_terms': ', '.join(terms),
        'main_W_clusters': wlabels,
        'mean_W_shares': wshare,
        'dominant_previous_Q': ', '.join(map(str, sub['question_cluster'].value_counts().head(2).index.tolist())),
        'top_existing_manual_topics': ', '.join(sub['expanded_topic_tags'].dropna().value_counts().head(2).index.astype(str).tolist()),
        'example1': examples[0] if examples else '',
        'example2': examples[1] if len(examples) > 1 else '',
        'example3': examples[2] if len(examples) > 2 else '',
        'review_note': rates_note + (' / 하위 미분할' if depth == 2 and is_terminal else ''),
    })
    if is_terminal and cid != 'ROOT':
        for intent, group in sub.groupby('intent_type', sort=True):
            n = len(group)
            y = int(group['y'].sum())
            intent_rows.append({
                'path': f'{cid}::{intent}', 'parent_path': cid,
                'intent_code': intent, 'intent_name': INTENT_LABELS.get(intent, str(intent)),
                'depth': 4, 'train_questions': n, 'train_yes': y,
                'train_no': n-y, 'train_yes_rate': y/n,
                'review_note': '표본 10개 미만 / 예시 수동 검토' if n < 10 else '요청유형은 규칙기반 분류',
                'example1': str(group['question'].iloc[0]),
            })

node_df = pd.DataFrame(rows)
intent_df = pd.DataFrame(intent_rows).sort_values(['parent_path', 'intent_name'])

# All questions: test rows are assigned by train-fitted clusters, but their
# ground-truth outcomes are deliberately hidden to maintain a blind test set.
items = []
for i, row in merged.iterrows():
    path = per_question_paths[i]
    last = path[-1]
    intent = str(row['intent_type'])
    similarity = float(x_all[i] @ centroids[last])
    items.append({
        'source_row': int(row['source_row']), 'split': row['split'],
        'question': row['question'],
        'top_path': path[0], 'middle_path': path[1] if len(path) > 1 else '',
        'small_path': path[2] if len(path) > 2 else '',
        'terminal_topic_path': last,
        'intent': INTENT_LABELS.get(intent, intent),
        'full_nested_path': f"{' > '.join(path)} > {INTENT_LABELS.get(intent,intent)}",
        'top_label': name_by_cluster.get(path[0], ''),
        'middle_label': name_by_cluster.get(path[1], '') if len(path) > 1 else '',
        'small_label': name_by_cluster.get(path[2], '') if len(path) > 2 else '',
        'existing_question_cluster_Q': int(row['question_cluster']),
        'existing_manual_topics': row.get('expanded_topic_tags'),
        'existing_output_style': row.get('output_style_tags'),
        'cosine_to_terminal_center': round(similarity, 4),
        'review_bottom_20pct': False,
        'train_label_for_posthoc_summary_only': int(row['y']) if row['split']=='train' else None,
    })
question_df = pd.DataFrame(items)
for cid, grp in question_df.loc[question_df['split']=='train'].groupby('terminal_topic_path'):
    cutoff = float(grp['cosine_to_terminal_center'].quantile(.20))
    positions = question_df['terminal_topic_path'].eq(cid)
    question_df.loc[positions, 'review_bottom_20pct'] = (
        question_df.loc[positions, 'cosine_to_terminal_center'] < cutoff
    )

assert question_df['source_row'].is_unique and len(question_df)==4507
assert int(question_df['train_label_for_posthoc_summary_only'].sum()) == 652
assert question_df.loc[question_df.split.eq('test'),'train_label_for_posthoc_summary_only'].isna().all()
assert sum(node_df.loc[node_df.depth.eq(1),'train_questions']) == 3605
for cid in node_members:
    if child_of[cid]:
        assert sum(len(node_members[c]) for c in child_of[cid]) == len(node_members[cid])

# In-place nested display, with one row per actual node; indents are cosmetic.
tree = []
def build_tree(cid: str):
    r = node_df.loc[node_df.path.eq(cid)].iloc[0].to_dict()
    r['indented_title'] = '    ' * r['depth'] + f"{cid}  {r['keyword_label_draft']}"
    tree.append(r)
    for c in child_of[cid]:
        build_tree(c)
    if not child_of[cid] and cid != 'ROOT':
        for _, it in intent_df.loc[intent_df.parent_path.eq(cid)].iterrows():
            tree.append({
                'depth': 4, 'path': it['path'], 'parent_path': cid,
                'indented_title': '    ' * 4 + f"{it['intent_name']}",
                'kind': '요청유형', 'keyword_label_draft': it['intent_name'],
                'label_status': '기존 규칙 기반',
                'train_questions': int(it['train_questions']),
                'train_yes': int(it['train_yes']),
                'train_no': int(it['train_no']),
                'train_yes_rate': float(it['train_yes_rate']),
                'top_topic_terms': '', 'main_W_clusters': '',
                'example1': it['example1'], 'review_note': it['review_note'],
            })
build_tree('ROOT')
tree_df = pd.DataFrame(tree)

# Existing W word clusters × thematic tree, based on the already cleaned
# token table; this is a cross-tab, not another hierarchical fitting stage.
tokens = pd.read_csv(TOKENS, encoding='utf-8-sig')
tokens = tokens[tokens['split'].eq('train')].copy()
path_map = question_df.set_index('source_row')[['top_path','middle_path','terminal_topic_path']]
tokens = tokens.join(path_map, on='source_row', validate='many_to_one')
cross = []
for hierarchy_col in ['top_path','middle_path','terminal_topic_path']:
    grouped = tokens.groupby([hierarchy_col,'word_cluster'], dropna=False)
    for (cid, w), sample in grouped:
        if pd.isna(cid):
            continue
        cross.append({
            'tree_level': hierarchy_col, 'topic_path': str(cid),
            'word_cluster_W': f'W{int(w)}',
            'token_occurrences': len(sample),
            'distinct_question_count': sample['source_row'].nunique(),
            'representative_words': ', '.join(sample['token'].value_counts().head(8).index.tolist()),
        })
cross_df = pd.DataFrame(cross).sort_values(['tree_level','topic_path','token_occurrences'], ascending=[True, True, False])

sample_n = min(650,len(x_train))
top_silhouette = float(silhouette_score(x_train, [per_question_paths[int(i)][0] for i in train_positions], sample_size=sample_n, random_state=SEED))
report = {
    'train_questions': 3605, 'test_questions_unlabeled': 902,
    'root_top_groups': len(top_nodes), 'middle_groups': len(middle_nodes),
    'small_groups': len(small_nodes),
    'terminal_topic_groups': len(terminal_topic_nodes),
    'intent_child_groups': len(intent_df),
    'tree_nodes_including_root': len(tree_df),
    'top_cluster_silhouette_sample_650': top_silhouette,
    'tfidf_features': int(x_sparse.shape[1]),
    'LSA_components': int(x_train.shape[1]),
    'LSA_explained_variance_ratio': float(svd.explained_variance_ratio_.sum()),
    'thematic_feature_excluded_common_words': sorted(common_words),
    'thematic_feature_extra_masked_request_words': sorted(GENERIC_REQUEST_WORDS),
    'manual_review_question_count': int(question_df.loc[question_df.split.eq('train'),'review_bottom_20pct'].sum()),
    'word_cluster_note': 'W=-1은 기존 어휘 미할당, 새 계층 군집이 아님',
    'disclaimer': '탐색적 군집; 기계적 라벨은 정답이 아님. train y는 사후 발생빈도에만 사용. test y는 은닉.',
}

exports = {
    'nested_tree': tree_df,
    'topic_nodes': node_df,
    'intent_leaf_nodes': intent_df,
    'question_memberships_4507': question_df,
    'word_cluster_cross': cross_df,
    'review_queue': question_df[(question_df.split=='train') & question_df.review_bottom_20pct].copy(),
}
for stem, data in exports.items():
    data.to_csv(OUT / f'{stem}.csv', index=False, encoding='utf-8-sig')
    data.to_json(OUT / f'{stem}.json', orient='records', force_ascii=False)

(OUT / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(report, ensure_ascii=False, indent=2))
print(node_df.loc[node_df.depth.eq(1), ['path','keyword_label_draft','train_questions','train_yes_rate','top_topic_terms']].to_string(index=False))
