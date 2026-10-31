"""Expand the previously fitted keyword-topic groups without using labels.

The topic groups below are auditable, manually specified *tags*, not clusters
learned by unsupervised algorithms. Output formats are encoded on a separate
axis. The held-out test set is never inspected or tagged here.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from halueval_keyword_topic_pilot import VOCAB, PHRASES, canonical, tag, TOKEN_RE

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "outputs" / "halueval_keyword_topics"
OUT = ROOT / "outputs" / "halueval_topic_expansion"

# Non-overlapping in *meaning* where possible; multi-labels are intentional.
# Generic words like people, time, person, story, data, chart, output, and
# examples are not accepted as evidence for a topic by themselves.
NEW_TOPICS = {
    "영화·방송": """movie movies film films cinema cinematic television tv
        theater theaters actress actresses actor actors animation
        documentary documentaries sitcom sitcoms episodes episode""",
    "지리·국가": """geography geographical geographic countries country nation nations
        continent continents oceania africa europe asia antarctica african
        europeans asian america american japan japanese china chinese korea
        korean canada canadian brazil australia australian regional region
        regions city cities capital-city island islands maps map""",
    "교육·학습": """education educational educating teacher teachers student students
        professor professors classroom classrooms curriculum
        homework assignment assignments exam exams examination examinations
        college colleges university universities semester semesters lecture
        lectures scholarship scholarships schooling""",
    "기술·공학": """technology technological engineering engineer engineers robot robots
        robotics mechanical hardware electronics electronic smartphone
        smartphones gadget gadgets sensor sensors device devices drone drones
        aerospace aircraft airplane airplanes microservices website websites
        webpage webpages internet application applications webapp""",
    "동식물·생태": """animal animals wildlife rainforest rainforests
        plant plants tree trees flower flowers bird birds fish fishes
        insect insects butterfly butterflies butterflys mammal mammals
        reptile reptiles amphibian amphibians habitat habitats forest forests
        dolphin dolphins whale whales elephant elephants tiger tigers""",
    "인간·관계": """friend friends friendship relationships relationship
        romance romantic love affection grief anger happiness sadness
        empathy compassion loneliness emotional emotion emotions
        marriage married boyfriend girlfriend dating roommate roommates
        sibling siblings parent parents communication bullying""",
    "뉴스·시사": """news headline headlines newspaper newspapers journalism journalist
        journalists breaking-news newsletter newsletters topicality""",
}
NEW_TOPICS = {k: set(v.split()) for k, v in NEW_TOPICS.items()}
NEW_PHRASES = {
    "영화·방송": ["movie theaters", "television series", "tv show", "film festival"],
    "지리·국가": ["new york city", "capital city", "united states", "united kingdom",
                   "south america", "north america", "middle east", "population of"],
    "교육·학습": ["school course", "class schedule", "high school", "middle school",
                  "university course", "college entrance"],
    "기술·공학": ["web page", "web design", "mobile app", "software engineering",
                    "aeronautical engineering", "computer hardware", "smart home"],
    "동식물·생태": ["animal kingdom", "amazon rainforest", "living organisms"],
    "인간·관계": ["best friend", "social relationships", "family relationships"],
    "뉴스·시사": ["current events", "current event", "recent news", "breaking news",
                  "recent event in the news"],
}

# This is a SEPARATE output/style axis, not a subject topic. The user can
# later cross it with both original/revised Q clusters and topic tags.
FORMATS = {
    "이야기·문학": """story stories poem poems haiku haikus limerick limericks
        sonnet sonnets fairy-tale ballad ballads monologue monologues
        plot plots narrative narratives fable fables fiction fictional""",
    "표·데이터": """chart charts table tables graph graphs dataset datasets
        dataframe data-frame spreadsheet spreadsheets visualization visualizations
        histogram histograms diagram diagrams plot plots""",
    "소셜·메시지": """tweet tweets twitter post posts email emails
        newsletter newsletters announcement announcements invitation invitations
        greeting greetings sms hashtag hashtags""",
    "웹·프로그램": """website websites webpage webpages webapp app apps
        javascript html python pseudocode algorithm algorithms code coding
        software database databases query queries""",
    "이미지·디자인": """logo logos illustration illustrations poster posters screenshot
        screenshots image images infographic infographics design designs
        sketch sketches drawing drawings painting paintings""",
}
FORMATS = {k: set(v.split()) for k, v in FORMATS.items()}
FORMAT_PHRASES = {
    "이야기·문학": ["short story", "fairy tale", "character development"],
    "표·데이터": ["data set", "data frame", "bar chart", "word cloud", "decision tree",
                    "data visualization", "flow chart"],
    "소셜·메시지": ["social media", "twitter post", "thank you note", "text message"],
    "웹·프로그램": ["web page", "mobile app", "source code", "sql query"],
    "이미지·디자인": ["color scheme", "graphic design", "web design", "visual design"],
}


def exact_match(tokens: set[str], text: str, words: set[str], phrases: list[str]) -> set[str]:
    found = words.intersection(tokens)
    for phrase in phrases:
        if re.search(r"(?<![a-z])" + re.escape(phrase) + r"(?![a-z])", text, re.I):
            found.add(phrase)
    return found


def find_new_topics(text: str, words: set[str]) -> dict[str, list[str]]:
    matches = {}
    for topic, vocab in NEW_TOPICS.items():
        hits = exact_match(words, text, vocab, NEW_PHRASES.get(topic, []))
        if topic == "지리·국가":
            # Only count country/city if these words occur literally; never
            # infer geographic topics from a bare date or population number.
            # Population alone can be a statistics question.
            hits.discard("population of") if not re.search(
                r"\bpopulation of\s+(?:the\s+)?(?:[a-z]+\s+){0,3}(?:country|city|china|india|japan|korea|[A-Z][a-z]+)",
                text, re.I) else None
        if topic == "교육·학습":
            # Don't interpret "machine learning" as a classroom topic.
            if re.search(r"\bmachine learning\b", text, re.I):
                hits.discard("learning")
        if topic == "동식물·생태" and re.search(r"\bdecision tree\b", text, re.I):
            hits.discard("tree")
            hits.discard("trees")
        if topic == "인간·관계":
            # "communication protocol" is computer engineering, not friendship.
            if re.search(r"\bcommunication (?:protocol|network|system)\b", text, re.I):
                hits.discard("communication")
            # Causal relationships are not interpersonal relationships.
            if re.search(r"\bcause\s+and\s+effect\b", text, re.I):
                hits.discard("relationship")
                hits.discard("relationships")
            if re.search(r"\b(?:cats?|dogs?|animals?)\s+love\b", text, re.I):
                hits.discard("love")
        if hits:
            matches[topic] = sorted(hits)
    return matches


def find_formats(text: str, words: set[str]) -> dict[str, list[str]]:
    matches = {}
    for fmt, vocab in FORMATS.items():
        hits = exact_match(words, text, vocab, FORMAT_PHRASES.get(fmt, []))
        # A "plot" can mean a story arc or mathematical plot; let both output
        # styles remain tentative, to be audited in a later quality sample.
        if hits:
            matches[fmt] = sorted(hits)
    return matches


def save(name: str, data: pd.DataFrame) -> None:
    data.to_csv(OUT / f"{name}.csv", index=False, encoding="utf-8-sig")
    (OUT / f"{name}.json").write_text(data.to_json(orient="records", force_ascii=False), encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    old = pd.read_csv(SRC / "mixed_cluster_question_topic_tags.csv", encoding="utf-8-sig")
    assert len(old) == 2734 and old.split.eq("train").all()
    records = []
    topic_occurrences = []
    for rec in old.to_dict(orient="records"):
        q = str(rec["question"])
        # A URL slug (e.g. /news/ or /technology/) is not a question word.
        # Retain the unmodified text in output for traceability.
        body = re.sub(r"https?://\S+|www\.\S+", " ", q)
        words = set(w.group(0).lower() for w in TOKEN_RE.finditer(body))
        new = find_new_topics(body, words)
        fmt = find_formats(body, words)
        previous = [] if pd.isna(rec["topic_tags"]) else str(rec["topic_tags"]).split(" / ")
        updated = previous + list(new)
        # Keep the previous classification untouched and append new evidence.
        group = "미분류" if not updated else "단일 주제" if len(updated) == 1 else "복합 주제"
        rec.update({
            "previous_topic_tags": " / ".join(previous),
            "additional_topic_tags": " / ".join(new),
            "expanded_topic_tags": " / ".join(updated),
            "expanded_topic_count": len(updated),
            "expanded_topic_status": group,
            "new_topic_evidence": "; ".join(f"{k}: {', '.join(v)}" for k, v in new.items()),
            "output_style_tags": " / ".join(fmt),
            "output_style_evidence": "; ".join(f"{k}: {', '.join(v)}" for k, v in fmt.items()),
            "newly_assigned": int(len(previous) == 0 and len(new) > 0),
        })
        records.append(rec)
        for topic, matched in new.items():
            topic_occurrences.append({"source_row": rec["source_row"], "topic": topic,
                                      "evidence": ", ".join(matched),
                                      "question": q, "request_type": rec["request_type"],
                                      "previous_status": rec["topic_status"]})

    result = pd.DataFrame.from_records(records)
    save("expanded_question_tags", result)
    grouped = (
        result.groupby(["topic_status", "expanded_topic_status"])
        .size().rename("questions").reset_index()
        .sort_values(["topic_status", "expanded_topic_status"])
    )
    save("before_after_group_status", grouped)
    changes = pd.DataFrame([
        {"stage": "기존 7주제", "total": len(result),
         "assigned": int(result.topic_count.gt(0).sum()),
         "unclassified": int(result.topic_count.eq(0).sum()),
         "multi_topic": int(result.topic_count.gt(1).sum())},
        {"stage": "확장 14주제", "total": len(result),
         "assigned": int(result.expanded_topic_count.gt(0).sum()),
         "unclassified": int(result.expanded_topic_count.eq(0).sum()),
         "multi_topic": int(result.expanded_topic_count.gt(1).sum())},
    ])
    changes["assigned_pct"] = changes.assigned / changes.total
    save("coverage_comparison", changes)

    membership = pd.DataFrame(topic_occurrences)
    if membership.empty:
        membership = pd.DataFrame(columns=["source_row", "topic", "evidence", "question",
                                            "request_type", "previous_status"])
    save("new_topic_question_membership", membership)
    cross = membership.merge(result[["source_row", "question_cluster_original"]],
                             on="source_row", how="left", validate="many_to_one")
    cross = (cross.groupby(["topic", "question_cluster_original", "request_type"])
             .size().rename("questions").reset_index()
             .sort_values(["topic", "questions"], ascending=[True, False]))
    save("new_topic_by_old_Q_and_request", cross)
    summary = membership.groupby("topic").agg(
        questions=("source_row", "nunique"),
        new_from_unclassified=("previous_status", lambda s: int(s.eq("미분류").sum())),
    ).reset_index().sort_values("questions", ascending=False)
    summary["of_2734_pct"] = summary.questions / len(old)
    save("new_topic_group_summary", summary)

    # For every new topic, use a deterministic, readable sample of examples
    # (never hallucination labels) so users can verify its true meaning.
    audit = []
    for topic, grp in membership.groupby("topic"):
        grp = grp.sort_values("source_row")
        chosen = pd.concat([grp.head(8), grp.iloc[8:].sample(min(7, max(0, len(grp)-8)), random_state=7)])
        for row in chosen.itertuples(index=False):
            audit.append({"new_topic": topic, "source_row": row.source_row,
                          "matched": row.evidence, "question": row.question,
                          "previously_unclassified": int(row.previous_status == "미분류"),
                          "review_result": "미검토"})
    save("human_audit_examples", pd.DataFrame(audit))

    only_new = result.loc[result.newly_assigned.eq(1),
        ["source_row", "question", "request_type", "additional_topic_tags",
         "new_topic_evidence", "output_style_tags", "label", "y"]]
    save("newly_assigned_questions", only_new)
    remaining = result.loc[result.expanded_topic_count.eq(0),
        ["source_row", "question", "request_type", "output_style_tags",
         "output_style_evidence", "label", "y"]]
    save("remaining_unclassified", remaining)

    style_data = []
    for fmt in FORMATS:
        style_data.append({"style": fmt, "questions": int(result.output_style_tags.fillna("").str.split(" / ")
                            .apply(lambda ls: fmt in ls).sum()),
                           "unclassified_topics": int(result.loc[result.expanded_topic_count.eq(0),
                               "output_style_tags"].fillna("").str.split(" / ")
                               .apply(lambda ls: fmt in ls).sum())})
    save("output_style_summary", pd.DataFrame(style_data).sort_values("questions", ascending=False))

    lexicon_rows = []
    for kind, topic_words, phrase_dict in [
        ("new_topic", NEW_TOPICS, NEW_PHRASES), ("output_style", FORMATS, FORMAT_PHRASES)
    ]:
        for topic, terms in topic_words.items():
            for term in sorted(terms):
                lexicon_rows.append({"kind": kind, "group": topic,
                                    "keyword": term, "match_type": "token"})
            for phrase in phrase_dict.get(topic, []):
                lexicon_rows.append({"kind": kind, "group": topic,
                                    "keyword": phrase, "match_type": "phrase"})
    save("new_group_dictionary", pd.DataFrame(lexicon_rows))

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.9))
    axes[0].bar(changes.stage, changes.assigned, color=["#98AABE", "#2E729D"])
    for idx, row in enumerate(changes.itertuples()):
        axes[0].text(idx, row.assigned+20, f"{row.assigned:,} ({row.assigned_pct:.1%})", ha="center")
    axes[0].set_ylim(0, 3000)
    axes[0].set_ylabel("Number of train Q'1 questions")
    axes[0].set_title("Topic coverage before / after")
    axes[0].set_xticks([0,1], ["7 topic tags", "14 topic tags"])
    names = {
        "영화·방송":"Film & TV", "지리·국가":"Geography", "교육·학습":"Education",
        "기술·공학":"Engineering", "동식물·생태":"Plants & animals",
        "인간·관계":"Human relations", "뉴스·시사":"News & events"}
    asc = summary.iloc[::-1]
    axes[1].barh([names.get(t, t) for t in asc.topic], asc.questions, color="#3D87A5")
    axes[1].set_title("Additional topic tags (overlap allowed)")
    axes[1].set_xlabel("Questions")
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "topic_expansion_summary.png", dpi=170)
    plt.close(fig)

    report = {
        "population": "previous mixed revised question cluster Q'1; train rows only",
        "questions": int(len(result)),
        "topic_groups_before": len(VOCAB), "new_topic_groups": list(NEW_TOPICS),
        "topic_groups_after": len(VOCAB) + len(NEW_TOPICS),
        "previous_tagged": int(changes.iloc[0].assigned),
        "expanded_tagged": int(changes.iloc[1].assigned),
        "newly_tagged": int(result.newly_assigned.sum()),
        "unclassified_before": int(changes.iloc[0].unclassified),
        "unclassified_after": int(changes.iloc[1].unclassified),
        "new_topic_counts": {r.topic: int(r.questions) for r in summary.itertuples()},
        "new_from_unclassified": {r.topic: int(r.new_from_unclassified) for r in summary.itertuples()},
        "newly_tagged_with_multi_topics": int(result.loc[result.newly_assigned.eq(1), "expanded_topic_count"].gt(1).sum()),
        "note": "Lexicon tags are provisional, not ML-learned clusters; coverage is not accuracy. No labels used in dictionary construction. Test split untouched. Some output-format keywords are tracked on a separate axis.",
    }
    (OUT / "group_expansion_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
