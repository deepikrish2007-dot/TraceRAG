
import math
import re


# --------------------------------------------------
# TEXT NORMALIZATION
# --------------------------------------------------

def normalize_source(source):
    """Normalize document names for reliable comparison."""

    source = str(source or "").strip().replace("\\", "/")
    return source.split("/")[-1].casefold()


def normalize_page(page):
    """Normalize page numbers into comparable strings."""

    page = str(page or "").strip()

    if page.endswith(".0"):
        page = page[:-2]

    return page


def normalize_text(text):
    """Normalize text and punctuation for keyword matching."""

    text = str(text or "").casefold()
    text = re.sub(r"[^a-z0-9.]+", " ", text)

    return " ".join(text.split())


def contains_keyword(text, keyword):
    """Check whether a normalized keyword occurs in normalized text."""

    normalized_text = normalize_text(text)
    normalized_keyword = normalize_text(keyword)

    if not normalized_keyword:
        return False

    return (
        f" {normalized_keyword} "
        in f" {normalized_text} "
    )


# --------------------------------------------------
# BUILD EXPECTED EVIDENCE
# --------------------------------------------------

def build_expected_evidence(expected_evidence):
    """
    Convert expected evidence into a dictionary.

    Each key represents one unique source-page pair.
    """

    evidence_map = {}

    for item in expected_evidence:

        source = normalize_source(item.get("source"))

        pages = item.get("pages", [])
        keywords = item.get("keywords", [])

        min_matches = item.get(
            "min_keyword_matches",
            1
        )

        for page in pages:

            page = normalize_page(page)
            key = (source, page)

            if key not in evidence_map:
                evidence_map[key] = {
                    "keywords": set(),
                    "min_keyword_matches": min_matches
                }

            evidence_map[key]["keywords"].update(
                normalize_text(keyword)
                for keyword in keywords
                if normalize_text(keyword)
            )

            evidence_map[key]["min_keyword_matches"] = max(
                evidence_map[key]["min_keyword_matches"],
                min_matches
            )

    return evidence_map


# --------------------------------------------------
# CHECK PASSAGE RELEVANCE
# --------------------------------------------------

def is_relevant_passage(passage, evidence_map):
    """Check whether a passage matches expected page evidence."""

    source = normalize_source(
        passage.get("source")
    )

    page = normalize_page(
        passage.get("page")
    )

    key = (source, page)

    if key not in evidence_map:
        return False

    expected = evidence_map[key]

    keywords = expected["keywords"]
    minimum = expected["min_keyword_matches"]

    # If no keywords were specified, page match is sufficient.
    if not keywords:
        return True

    text = passage.get("text", "")

    matched_count = sum(
        1
        for keyword in keywords
        if contains_keyword(text, keyword)
    )

    return matched_count >= minimum


# --------------------------------------------------
# EVALUATION METRICS
# --------------------------------------------------

def evaluate_retrieval(
    expected_evidence,
    retrieved,
    k=5
):
    """
    Calculate:
        1. Precision@K
        2. Recall@K
        3. MRR
        4. Hit Rate@K
        5. nDCG@K
        6. Source Coverage
    """

    if k <= 0:
        raise ValueError("k must be greater than zero.")

    top_k = retrieved[:k]

    evidence_map = build_expected_evidence(
        expected_evidence
    )

    expected_pairs = set(evidence_map.keys())

    expected_sources = {
        source for source, page in expected_pairs
    }

    found_pairs = set()
    found_sources = set()

    relevant_flags = []

    # --------------------------------------------------
    # CHECK EACH RETRIEVED PASSAGE
    # --------------------------------------------------

    for passage in top_k:

        source = normalize_source(
            passage.get("source")
        )

        page = normalize_page(
            passage.get("page")
        )

        pair = (source, page)

        relevant = (
            pair not in found_pairs
            and is_relevant_passage(
                passage,
                evidence_map
            )
        )

        relevant_flags.append(relevant)

        if relevant:
            found_pairs.add(pair)
            found_sources.add(source)

    # --------------------------------------------------
    # 1. PRECISION@K
    # --------------------------------------------------

    relevant_count = sum(relevant_flags)

    precision = relevant_count / k

    # --------------------------------------------------
    # 2. RECALL@K
    # --------------------------------------------------

    recall = (
        len(found_pairs) / len(expected_pairs)
        if expected_pairs else 0.0
    )

    # --------------------------------------------------
    # 3. MEAN RECIPROCAL RANK
    # --------------------------------------------------

    mrr = 0.0

    for rank, relevant in enumerate(
        relevant_flags,
        start=1
    ):
        if relevant:
            mrr = 1.0 / rank
            break

    # --------------------------------------------------
    # 4. HIT RATE@K
    # --------------------------------------------------

    hit_rate = (
        1.0 if relevant_count > 0 else 0.0
    )

    # --------------------------------------------------
    # 5. nDCG@K
    # --------------------------------------------------

    dcg = 0.0

    for rank, relevant in enumerate(
        relevant_flags,
        start=1
    ):
        if relevant:
            dcg += 1.0 / math.log2(rank + 1)

    ideal_relevant_count = min(
        len(expected_pairs),
        k
    )

    ideal_dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank in range(
            1,
            ideal_relevant_count + 1
        )
    )

    ndcg = (
        dcg / ideal_dcg
        if ideal_dcg > 0 else 0.0
    )

    # --------------------------------------------------
    # 6. SOURCE COVERAGE
    # --------------------------------------------------

    source_coverage = (
        len(found_sources) / len(expected_sources)
        if expected_sources else 0.0
    )

    return {
        "precision": precision,
        "recall": recall,
        "mrr": mrr,
        "hit_rate": hit_rate,
        "ndcg": ndcg,
        "source_coverage": source_coverage,
        "relevant_count": relevant_count,
        "retrieved_count": len(top_k),
        "expected_evidence_count": len(expected_pairs),
        "covered_evidence_count": len(found_pairs)
    }
