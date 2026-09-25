from typing import Optional

# Distinctive multi-word phrases per source document, deliberately specific
# rather than single generic words ("leave", "policy") that show up across
# multiple documents and would risk a false-confident match. A query only
# gets routed to one document when it matches that document's phrases and no
# other document's - any other case (ambiguous, cross-policy, or no match at
# all) falls back to the full, unfiltered retriever. This mirrors the lesson
# already paid for by the score-threshold/rank-cutoff experiments in
# RAG_IMPROVEMENT_ROADMAP.md section 1: blindly trimming the candidate pool
# costs recall/faithfulness for a marginal precision gain, so filtering here
# only fires on a high-confidence signal instead of being mandatory.
_SOURCE_KEYWORDS: dict[str, list[str]] = {
    "data/Attendance policy.pdf": [
        "attendance", "punctual", "late arrival", "check-in", "check in",
        "office hours", "office timing", "grace period", "late sitting",
        "shift roster", "payroll attendance",
    ],
    # Bare "leave" is deliberately excluded here: it's also a verb ("leave
    # the office by 7pm" is an Attendance Policy late-sitting question), and
    # including it causes more false-positive misroutes (e.g. "What time
    # should I leave the office by?" wrongly narrowed to Leave Policy) than
    # the compound cross-policy phrasings it would catch ("leave and
    # attendance rules") - those just fall back to unfiltered instead, which
    # is the safe default.
    "data/Revised Leave Policy - REV02.pdf": [
        "sick leave", "casual leave", "maternity leave", "paternity leave",
        "pilgrimage leave", "annual leave", "leave policy", "leave balance",
        "leave year", "leave without pay", "leave of absence",
    ],
    "data/Work From Home Policy.pdf": [
        "work from home", "wfh", "work from anywhere", "remote work",
        "working remotely", "wfa",
    ],
    "data/Laptop Policy Monit.pdf": [
        "laptop", "gate pass", "administrator", "it security",
        "company laptop",
    ],
}


def infer_source(query: str) -> Optional[str]:
    lowered = query.lower()
    matched = [
        source
        for source, keywords in _SOURCE_KEYWORDS.items()
        if any(keyword in lowered for keyword in keywords)
    ]
    return matched[0] if len(matched) == 1 else None
