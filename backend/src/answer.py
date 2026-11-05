import os
import re
from pathlib import Path
from typing import Any
import chromadb
from dotenv import load_dotenv
from google import genai
from sentence_transformers import SentenceTransformer
# ============================================================
# 1. CONFIGURATION
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")
API_KEY = os.getenv("GEMINI_API_KEY")
MODEL_NAME = os.getenv("GEMINI_MODEL", "").strip()
DB_DIR = PROJECT_ROOT / ".chroma"
COLLECTION_NAME = "tracerag_chunks"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
CANDIDATE_COUNT = 30
PER_SOURCE_CANDIDATE_COUNT = 10
DEFAULT_TOP_K = 6
BROAD_QUESTION_TOP_K = 8
MAX_CONTEXT_CHARS = 18000
if not API_KEY:
    raise ValueError("GEMINI_API_KEY is missing. Check backend/.env.")
if not MODEL_NAME:
    raise ValueError("GEMINI_MODEL is missing. Check backend/.env.")
# ============================================================
# 2. INITIALIZE GEMINI
# ============================================================
client = genai.Client(api_key=API_KEY)
# ============================================================
# 3. CONNECT TO CHROMADB
# ============================================================
DB_DIR.mkdir(parents=True, exist_ok=True)
db_client = chromadb.PersistentClient(path=str(DB_DIR))
def get_collection():
    """Return the current TraceRAG ChromaDB collection."""
    return db_client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
# Keep this variable because api.py may import collection.
collection = get_collection()
# ============================================================
# 4. LOAD EMBEDDING MODEL ONCE
# ============================================================
print("Loading TraceRAG embedding model...")
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
print("TraceRAG embedding model loaded.")
# ============================================================
# 5. QUESTION INTENT
# ============================================================
FUTURE_TERMS = (
    "future work",
    "future research",
    "further work",
    "future direction",
    "future directions",
    "future improvement",
    "future improvements",
    "future extension",
    "future extensions",
    "improve",
    "improvements",
    "extend",
    "extensions",
    "could be improved",
    "will be explored",
    "remains future work",
)
CONCLUSION_TERMS = (
    "conclusion",
    "conclusions",
    "concluding remarks",
    "we conclude",
    "in conclusion",
    "main findings",
    "key findings",
    "takeaway",
    "takeaways",
    "what does the paper conclude",
    "overall findings",
)
LIMITATION_TERMS = (
    "limitation",
    "limitations",
    "drawback",
    "drawbacks",
    "shortcoming",
    "shortcomings",
    "weakness",
    "weaknesses",
    "threats to validity",
)
CONTRIBUTION_TERMS = (
    "main contribution",
    "main contributions",
    "key contribution",
    "key contributions",
    "contribution",
    "contributions",
    "proposed method",
    "proposed approach",
    "we propose",
    "we introduce",
    "our approach",
    "our method",
    "novel",
    "state of the art",
    "state-of-the-art",
    "results show",
    "experimental results",
)
BROAD_TERMS = (
    "objective",
    "purpose",
    "aim",
    "contribution",
    "contributions",
    "problem",
    "limitation",
    "limitations",
    "conclusion",
    "summary",
    "overview",
    "key findings",
    "main findings",
    "future work",
    "future research",
    "improvement",
    "improvements",
    "main points",
)
COMPARISON_TERMS = (
    "compare",
    "comparison",
    "compare and contrast",
    "difference between",
    "differences between",
    "similarities between",
    "versus",
    " vs ",
    "both papers",
    "both documents",
    "each paper",
    "two papers",
    "two documents",
    "compare the main contributions",
)
SECTION_PATTERNS = {
    "conclusion": re.compile(
        r"(?im)^\s*(?:\d+(?:\.\d+)*\.?\s+)?"
        r"(?:conclusions?|concluding remarks)\s*:?\s*$"
    ),
    "future": re.compile(
        r"(?im)^\s*(?:\d+(?:\.\d+)*\.?\s+)?"
        r"(?:future work|future research|future directions?|"
        r"future improvements?|further work)\s*:?\s*$"
    ),
    "limitations": re.compile(
        r"(?im)^\s*(?:\d+(?:\.\d+)*\.?\s+)?"
        r"(?:limitations?|threats to validity)\s*:?\s*$"
    ),
    "contribution": re.compile(
        r"(?im)^\s*(?:\d+(?:\.\d+)*\.?\s+)?"
        r"(?:abstract|introduction|contributions?|our contributions|results|experimental results|discussion)\s*:?\s*$"
    ),
}

EXCLUDED_HEADING_PATTERN = re.compile(
    r"(?im)^\s*(?:\d+(?:\.\d+)*\.?\s+)?"
    r"(?:references|bibliography|acknowledgements?|"
    r"author contributions|conflict of interest|"
    r"declaration of interests)\s*$"
)

def detect_intent(question: str) -> str:
    """Identify the kind of evidence needed for a question."""
    text = question.casefold()
    asks_conclusion = any(term in text for term in CONCLUSION_TERMS)
    asks_future = any(term in text for term in FUTURE_TERMS)
    asks_limitations = any(term in text for term in LIMITATION_TERMS)
    if asks_conclusion and asks_future:
        return "conclusion_future"
    if asks_future:
        return "future"
    if asks_conclusion:
        return "conclusion"
    if asks_limitations:
        return "limitations"
    if any(term in text for term in CONTRIBUTION_TERMS):
        return "contribution"
    return "general"
def is_comparison_question(question: str) -> bool:
    """Return True when a question likely needs evidence from multiple PDFs."""
    text = f" {question.casefold()} "
    return any(term in text for term in COMPARISON_TERMS)
# ============================================================
# 6. SECTION DETECTION AND RELEVANCE SCORING
# ============================================================
def has_section_heading(text: str, section: str) -> bool:
    """Find a section heading anywhere in a chunk."""
    pattern = SECTION_PATTERNS.get(section)
    return bool(pattern and pattern.search(text))
def heading_bonus(text: str, intent: str) -> float:
    """Lower scores are better; reward chunks with relevant headings."""
    bonus = 0.0
    if intent in ("conclusion", "conclusion_future"):
        if has_section_heading(text, "conclusion"):
            bonus -= 0.85
    if intent in ("future", "conclusion_future"):
        if has_section_heading(text, "future"):
            bonus -= 0.85
    if intent == "limitations" and has_section_heading(text, "limitations"):
        bonus -= 0.85
    if intent == "contribution" and has_section_heading(text, "contribution"):
        bonus -= 0.55
    first_lines = "\n".join(
        line.strip() for line in text.splitlines() if line.strip()
    ).splitlines()[:8]
    if EXCLUDED_HEADING_PATTERN.search("\n".join(first_lines)):
        bonus += 0.50
    return bonus
def contribution_signal_bonus(text: str) -> float:
    """Reward passages that directly explain a paper's contribution."""
    normalized = " ".join(text.casefold().split())
    bonus = 0.0

    strong_signals = (
        "we propose",
        "we introduce",
        "we present",
        "we develop",
        "we design",
        "this paper proposes",
        "this paper introduces",
        "in this paper, we",
        "our work proposes",
        "our work introduces",
        "openml is a place",
        "openml is",
        "share data",
        "share their code and results",
        "share code and results",
        "stored and organized",
        "collaborate on a global scale",
        "based solely on attention",
        "attention mechanisms",
        "dispensing with recurrence",
        "dispensing with convolutions",
    )

    matched = sum(1 for phrase in strong_signals if phrase in normalized)
    bonus -= min(matched, 5) * 0.15

    if "abstract" in normalized[:500]:
        bonus -= 0.12
    if "introduction" in normalized[:500]:
        bonus -= 0.08

    metadata_signals = (
        "arxiv preprint",
        "copyright",
        "all rights reserved",
        "corresponding author",
    )
    if sum(1 for phrase in metadata_signals if phrase in normalized) >= 2:
        bonus += 0.45

    return bonus

def keyword_score(text: str, question: str, intent: str) -> float:
    """Give a small score improvement for relevant keywords."""
    normalized = " ".join(text.casefold().split())
    question_words = {
        word
        for word in re.findall(r"[a-z0-9]+", question.casefold())
        if len(word) > 2
    }
    passage_words = set(re.findall(r"[a-z0-9]+", normalized))
    overlap = len(question_words & passage_words)
    score = min(overlap, 10) * 0.015
    intent_terms = {
        "future": FUTURE_TERMS,
        "conclusion": CONCLUSION_TERMS,
        "conclusion_future": CONCLUSION_TERMS + FUTURE_TERMS,
        "limitations": LIMITATION_TERMS,
        "contribution": CONTRIBUTION_TERMS,
        "general": (),
    }.get(intent, ())
    matched_terms = sum(1 for term in intent_terms if term in normalized)
    score += min(matched_terms, 4) * 0.025
    return score
# ============================================================
# 7. HYBRID RETRIEVAL
# ============================================================
_STOP_WORDS = {
    "the", "and", "for", "that", "with", "from", "this", "these", "those",
    "what", "which", "when", "where", "who", "why", "how", "does", "did",
    "are", "was", "were", "is", "be", "been", "being", "about", "into",
    "their", "they", "them", "its", "it", "paper", "document", "according",
    "explain", "describe", "discuss", "give", "tell", "please", "can", "could",
}
def lexical_relevance(text: str, question: str) -> float:
    """Return a bounded lexical relevance bonus (higher is better).*
    Semantic distance remains the main ranking signal. This bonus helps exact
    technical terms and named concepts win when several chunks are semantically
    similar, without allowing keyword overlap to dominate retrieval.
    """
    q_norm = " ".join(question.casefold().split())
    p_norm = " ".join(text.casefold().split())
    q_terms = {
        token for token in re.findall(r"[a-z0-9][a-z0-9_-]*", q_norm)
        if len(token) > 2 and token not in _STOP_WORDS
    }
    if not q_terms:
        return 0.0
    p_terms = set(re.findall(r"[a-z0-9][a-z0-9_-]*", p_norm))
    coverage = len(q_terms & p_terms) / len(q_terms)
    phrase_bonus = 0.12 if len(q_norm) > 12 and q_norm in p_norm else 0.0
    # Reward exact multi-word concepts (e.g., "citation tracking") modestly.
    pairs = [" ".join(tokens) for tokens in zip(q_norm.split(), q_norm.split()[1:])]
    pair_bonus = min(sum(1 for pair in pairs if pair in p_norm), 3) * 0.025
    return min(0.32, coverage * 0.24 + phrase_bonus + pair_bonus)
def _metadata_source(metadata: dict[str, Any]) -> str:
    """Get a consistent source filename from chunk metadata."""
    return str(metadata.get("source") or metadata.get("filename") or "Unknown")
def _find_requested_source(question: str, sources: list[str]) -> str | None:
    """Return a source filename explicitly named in the question, if any."""
    q = question.casefold()
    # Prefer exact filename matches; filenames may contain spaces.
    matches = [source for source in sources if source.casefold() in q]
    if matches:
        return max(matches, key=len)
    return None
def _page_number(value: Any) -> str:
    """Normalize page metadata to a simple integer string when possible."""
    if value is None:
        return "Unknown"
    try:
        number = int(value)
        return str(number) if number >= 1 else "Unknown"
    except (TypeError, ValueError):
        text = str(value).strip()
        match = re.fullmatch(r"(?:page\s*)?(\d+)", text, flags=re.IGNORECASE)
        return match.group(1) if match else "Unknown"
def _validate_citations(
    answer: str,
    passages: list[dict[str, Any]],
) -> str:
    """Validate citations against the source/page metadata in retrieved passages.

    Supported examples:
      (Source: paper.pdf, p. 3)
      (Source: paper.pdf, page 3)
      (Source: paper.pdf, p. 2, 3, 5)
      (Source: paper.pdf, p. 2-5)
      [paper.pdf · p. 3]
      [**paper name.pdf · p. 3**](http://...)

    This validates metadata consistency only; it does not prove that a passage
    semantically supports the associated claim.
    """
    valid_pairs: set[tuple[str, str]] = set()
    for item in passages:
        source = str(item.get("source") or item.get("filename") or "").strip()
        page = _page_number(item.get("page"))
        if source and source.casefold() != "unknown" and page != "Unknown":
            valid_pairs.add((source.casefold(), page))

    def parse_pages(page_text: str) -> list[str]:
        """Parse individual pages and ranges, including '1 - p. 8'."""
        text = page_text.strip()

        # A repeated page label means two separate page references, not a range.
        repeated_labels = re.fullmatch(
            r"\s*(\d+)\s*[-–,;]\s*p\.?\s*(\d+)\s*",
            text,
            flags=re.IGNORECASE,
        )
        if repeated_labels:
            return [repeated_labels.group(1), repeated_labels.group(2)]

        pages: list[str] = []
        parts = re.split(r"\s*[,;]\s*", text)
        for part in parts:
            part = re.sub(r"^p\.?\s*", "", part, flags=re.IGNORECASE).strip()
            range_match = re.fullmatch(r"(\d+)\s*[-–]\s*(\d+)", part)
            if range_match:
                first, last = map(int, range_match.groups())
                if first <= last and last - first <= 100:
                    pages.extend(str(number) for number in range(first, last + 1))
                else:
                    pages.append(part)
            else:
                pages.append(part)
        return pages

    def citation_is_valid(source: str, page_text: str) -> bool:
        source_key = source.strip().strip("`*[] ").casefold()
        pages = parse_pages(page_text)
        return bool(pages) and all(
            (source_key, _page_number(page)) in valid_pairs
            for page in pages
        )

    def mark_unverified(source: str, page_text: str) -> str:
        return (
            f"[Unverified citation: {source.strip()} · "
            f"p. {page_text.strip()}]"
        )

    # Parenthesized format: (Source: paper.pdf, p. 3)
    parenthesized_pattern = re.compile(
        r"\(\s*Source:\s*(.+?\.pdf)\s*,\s*"
        r"(?:p\.?|page)\s*"
        r"(\d+(?:\s*[-–,;]\s*(?:p\.?\s*)?\d+)*)\s*\)",
        flags=re.IGNORECASE,
    )

    def mark_parenthesized(match: re.Match[str]) -> str:
        source, page_text = match.group(1).strip(), match.group(2).strip()
        pages = parse_pages(page_text)
        verified = [page for page in pages if (
            source.casefold(), _page_number(page)
        ) in valid_pairs]
        unverified = [page for page in pages if (
            source.casefold(), _page_number(page)
        ) not in valid_pairs]
        rendered = [f"(Source: {source}, p. {page})" for page in verified]
        if unverified:
            rendered.append(mark_unverified(source, ", ".join(unverified)))
        return " ".join(rendered) if rendered else match.group(0)

    answer = parenthesized_pattern.sub(mark_parenthesized, answer)

    # Markdown format, optionally bold and/or followed by a Markdown URL.
    markdown_pattern = re.compile(
        r"\[(?:\*\*)?(?P<source>[^\]\n]+?\.pdf)"
        r"\s*[·•]\s*p\.?\s*"
        r"(?P<page>\d+(?:\s*[-–,;]\s*(?:p\.?\s*)?\d+)*)"
        r"(?:\*\*)?\](?:\([^)]+\))?",
        flags=re.IGNORECASE,
    )

    def mark_markdown(match: re.Match[str]) -> str:
        source = match.group("source").strip()
        page_text = match.group("page").strip()
        if citation_is_valid(source, page_text):
            return match.group(0)
        return mark_unverified(source, page_text)

    answer = markdown_pattern.sub(mark_markdown, answer)

    if "[Unverified citation:" in answer and "could not be matched" not in answer:
        answer += (
            "\n\nNote: One or more citations could not be matched to the "
            "exact source/page metadata in the retrieved passages. "
            "Verify them against the original PDF."
        )

    return answer

def _add_candidate(
    candidates: dict[str, dict[str, Any]],
    text: str,
    metadata: dict[str, Any] | None,
    distance: float | None,
    fallback_source: str = "Unknown",
) -> None:
    """Add a chunk while preserving genuine semantic-distance provenance."""
    if not text or not text.strip():
        return

    metadata = metadata or {}
    normalized_text = " ".join(text.casefold().split())
    source = _metadata_source(metadata)
    if source == "Unknown":
        source = fallback_source
    page = _page_number(metadata.get("page"))

    normalized_key = f"{source.casefold()}|{page}|{normalized_text}"
    candidate_distance = float(distance) if distance is not None else None
    existing = candidates.get(normalized_key)

    if existing is not None:
        existing_distance = existing.get("distance")
        if candidate_distance is not None:
            if existing_distance is None:
                existing["distance"] = candidate_distance
            else:
                existing["distance"] = min(existing_distance, candidate_distance)

        if existing["source"] == "Unknown" and source != "Unknown":
            existing["source"] = source
        return

    candidates[normalized_key] = {
        "text": text.strip(),
        "source": source,
        "page": page,
        "distance": candidate_distance,
        "metadata": metadata,
    }

def retrieve_passages(
    question: str,
    top_k: int = DEFAULT_TOP_K,
) -> list[dict[str, Any]]:
    """Combine semantic search, supplemental scanning, and source balancing."""
    question = question.strip()
    if not question:
        return []
    current_collection = get_collection()
    total_chunks = current_collection.count()
    if total_chunks == 0:
        return []
    # Discover indexed source filenames and honor an explicit filename in the question.
    try:
        metadata_result = current_collection.get(include=["metadatas"])
        all_metadata_for_sources = metadata_result.get("metadatas") or []
        available_sources = sorted({
            _metadata_source(meta)
            for meta in all_metadata_for_sources
            if meta and _metadata_source(meta) != "Unknown"
        })
    except Exception:
        available_sources = []
    # Detect all indexed filenames explicitly mentioned in the question.
    mentioned_sources = [
        source
        for source in available_sources
        if source.casefold() in question.casefold()
    ]
    detected_source = _find_requested_source(question, available_sources)
    intent = detect_intent(question)
    # Comparison takes priority when the question names multiple indexed PDFs.
    # Otherwise preserve the existing behavior for a single explicitly named PDF.
    comparison = is_comparison_question(question) and (
        len(mentioned_sources) >= 2 or detected_source is None
    )
    requested_source = None if comparison else detected_source
    query_embedding = embedding_model.encode(
        [question], normalize_embeddings=True, show_progress_bar=False
    ).tolist()
    # Do not block comparison questions just because they mention filenames.
    if (
        any(filename.casefold() in question.casefold() for filename in available_sources)
        and not requested_source
        and not comparison
    ):
        return []
    candidates: dict[str, dict[str, Any]] = {}
    # A. Standard semantic search; an explicitly named PDF limits retrieval to that source.
    semantic_count = min(CANDIDATE_COUNT, total_chunks)
    query_kwargs = {
        "query_embeddings": query_embedding,
        "n_results": semantic_count,
        "include": ["documents", "metadatas", "distances"],
    }
    if requested_source:
        query_kwargs["where"] = {"source": requested_source}
    results = current_collection.query(**query_kwargs)
    documents = (results.get("documents") or [[]])[0]
    metadatas = (results.get("metadatas") or [[]])[0]
    distances = (results.get("distances") or [[]])[0]
    for index, text in enumerate(documents):
        metadata = metadatas[index] if index < len(metadatas) else {}
        distance = distances[index] if index < len(distances) else None
        _add_candidate(candidates, text, metadata, distance)
     # B. For comparison questions, independently retrieve
    # relevant passages from every indexed source.
    if comparison:
        try:
            metadata_result = current_collection.get(include=["metadatas"])
            all_metadata = metadata_result.get("metadatas") or []
            sources = sorted({
                _metadata_source(meta)
                for meta in all_metadata
                if meta and _metadata_source(meta) != "Unknown"
            })
            # Search each source with several focused queries. A single
            # generic query can retrieve secondary benefits or experiment
            # details instead of the paper's central contribution.
            comparison_queries = [question]
            if intent == "contribution":
                comparison_queries.extend([
                    "abstract introduction central idea research problem "
                    "what this paper proposes introduces or presents",
                    "main contribution proposed method approach system "
                    "framework architecture methodology and design",
                    "what is the central idea of this research paper "
                    "and what does it contribute",
                    "problem addressed and the new solution or platform "
                    "introduced by the authors",
                ])
            elif intent == "limitations":
                comparison_queries.append(
                    "limitations, weaknesses, challenges, "
                    "shortcomings and drawbacks"
                )
            elif intent == "future":
                comparison_queries.append(
                    "future work, future directions, "
                    "possible improvements and research opportunities"
                )
            elif intent == "conclusion":
                comparison_queries.append(
                    "conclusions, main findings, key findings, "
                    "overall results and implications"
                )
            for source in sources:
                for search_query in comparison_queries:
                    search_embedding = embedding_model.encode(
                        [search_query],
                        normalize_embeddings=True,
                        show_progress_bar=False,
                    ).tolist()
                    source_results = current_collection.query(
                        query_embeddings=search_embedding,
                        n_results=min(
                            PER_SOURCE_CANDIDATE_COUNT,
                            total_chunks,
                        ),
                        where={"source": source},
                        include=["documents", "metadatas", "distances"],
                    )
                    source_docs = (
                        source_results.get("documents") or [[]]
                    )[0]
                    source_metas = (
                        source_results.get("metadatas") or [[]]
                    )[0]
                    source_distances = (
                        source_results.get("distances") or [[]]
                    )[0]
                    for index, passage_text in enumerate(source_docs):
                        metadata = (
                            source_metas[index]
                            if index < len(source_metas)
                            else {}
                        )
                        distance = (
                            source_distances[index]
                            if index < len(source_distances)
                            else None
                        )
                        _add_candidate(
                            candidates,
                            passage_text,
                            metadata,
                            distance,
                            fallback_source=source,
                        )
        except Exception as error:
            print(f"Source-balanced retrieval warning: {error}")
    # C. Scan stored chunks to recover important section passages.
    try:
        stored = current_collection.get(include=["documents", "metadatas"])
        all_documents = stored.get("documents") or []
        all_metadatas = stored.get("metadatas") or []
        relevant_terms = {
            "future": FUTURE_TERMS,
            "conclusion": CONCLUSION_TERMS,
            "conclusion_future": CONCLUSION_TERMS + FUTURE_TERMS,
            "limitations": LIMITATION_TERMS,
            "contribution": CONTRIBUTION_TERMS,
            "general": (),
        }.get(intent, ())
        for index, text in enumerate(all_documents):
            if not text or not text.strip():
                continue
            metadata = all_metadatas[index] if index < len(all_metadatas) else {}
            if requested_source and _metadata_source(metadata) != requested_source:
                continue
            normalized = " ".join(text.casefold().split())
            matching_section = (
                (intent in ("conclusion", "conclusion_future") and has_section_heading(text, "conclusion"))
                or (intent in ("future", "conclusion_future") and has_section_heading(text, "future"))
                or (intent == "limitations" and has_section_heading(text, "limitations"))
                or (intent == "contribution" and has_section_heading(text, "contribution"))
            )
            has_relevant_term = any(term in normalized for term in relevant_terms)
            # Add section-specific chunks; for comparison, per-source semantic
            # search above already supplies general-purpose candidates.
            if matching_section or has_relevant_term:
                _add_candidate(candidates, text, metadata, None)
    except Exception as error:
        print(f"Supplemental text scan warning: {error}")
    # D. Score candidates. Lower scores rank higher.
    ranked = []
    for item in candidates.values():
        # Supplemental-only candidates have no semantic distance. The fallback
        # is a ranking penalty only; it is never stored as a semantic distance.
        semantic_distance = item.get("distance")
        base_score = semantic_distance if semantic_distance is not None else 1.0

        score = (
            base_score
            + heading_bonus(item["text"], intent)
            - keyword_score(item["text"], question, intent)
            - lexical_relevance(item["text"], question)
        )

        if comparison and intent == "contribution":
            score += contribution_signal_bonus(item["text"])

        ranked.append({**item, "score": score})

    ranked.sort(key=lambda item: item["score"])
    # E. Select page-diverse passages.
    # First pass: select the best-ranked chunk from each page.
    # Second pass: fill remaining positions with additional chunks.
    def select_page_diverse(items, limit):
        selected = []
        selected_ids = set()
        seen_pages = set()
        # First pass: one chunk per source-page pair.
        for item in items:
            page_key = (
                str(item.get("source", "")).casefold(),
                str(item.get("page", "Unknown"))
            )
            if page_key not in seen_pages:
                selected.append(item)
                selected_ids.add(id(item))
                seen_pages.add(page_key)
                if len(selected) >= limit:
                    return selected
        # Second pass: fill remaining positions with other chunks.
        for item in items:
            if id(item) not in selected_ids:
                selected.append(item)
                selected_ids.add(id(item))
                if len(selected) >= limit:
                    break
        return selected
    # F. Comparison-specific selection.
    if comparison:
        grouped = {}
        for item in ranked:
            source = str(item["source"])
            grouped.setdefault(source, []).append(item)

        selected = []
        selected_ids = set()
        seen_pages = set()

        # First guarantee the best-ranked passage from each indexed source.
        for source, source_items in grouped.items():
            if source_items:
                best_item = source_items[0]
                selected.append(best_item)
                selected_ids.add(id(best_item))
                seen_pages.add((
                    source.casefold(),
                    str(best_item.get("page", "Unknown")),
                ))

        if len(selected) >= top_k:
            return selected[:top_k]

        # Fill remaining positions globally, preferring new pages.
        for item in ranked:
            if id(item) in selected_ids:
                continue
            page_key = (
                str(item.get("source", "")).casefold(),
                str(item.get("page", "Unknown")),
            )
            if page_key not in seen_pages:
                selected.append(item)
                selected_ids.add(id(item))
                seen_pages.add(page_key)
                if len(selected) >= top_k:
                    return selected

        # If necessary, allow additional chunks from already represented pages.
        for item in ranked:
            if id(item) not in selected_ids:
                selected.append(item)
                selected_ids.add(id(item))
                if len(selected) >= top_k:
                    break

        return selected
    # Standard questions: prefer different pages first.
    return select_page_diverse(ranked, top_k)
# ============================================================
# 8. GENERATE AN EVIDENCE-BASED ANSWER
# ============================================================
def answer_question(question: str) -> str:
    """Answer questions using retrieved passages and citations."""
    question = question.strip()
    if not question:
        return "Please enter a question."
    current_collection = get_collection()
    if current_collection.count() == 0:
        return (
            "No indexed document passages were found. "
            "Please upload a readable PDF and index it first."
        )
    intent = detect_intent(question)
    comparison = is_comparison_question(question)
    is_broad_question = any(term in question.casefold() for term in BROAD_TERMS)
    top_k = BROAD_QUESTION_TOP_K if (is_broad_question or comparison) else DEFAULT_TOP_K
    passages = retrieve_passages(question, top_k=top_k)
    if not passages:
        # Resolve the source locally for a helpful message; do not depend on a
        # variable that exists only inside retrieve_passages().
        try:
            metadata_result = current_collection.get(include=["metadatas"])
            source_names = sorted({
                _metadata_source(meta)
                for meta in (metadata_result.get("metadatas") or [])
                if meta and _metadata_source(meta) != "Unknown"
            })
            requested_source = _find_requested_source(question, source_names)
        except Exception:
            requested_source = None
        if requested_source:
            return (
                f"No passages could be retrieved from {requested_source}. "
                "Please check that this PDF is indexed and that its source metadata "
                "matches the filename."
            )
        return (
            "I could not retrieve relevant passages from the indexed documents. "
            "Please check PDF extraction and indexing."
        )
    # Build bounded context with exact source/page metadata.
    context_parts = []
    used_chars = 0
    for index, passage in enumerate(passages, start=1):
        source = str(passage["source"])
        page = str(passage["page"])
        text = passage["text"]
        part = (
            f"[Passage {index}]\n"
            f"Source: {source}\n"
            f"Page: {page}\n"
            f"Text:\n{text}\n"
        )
        remaining = MAX_CONTEXT_CHARS - used_chars
        if len(part) > remaining:
            # Do not truncate a passage mid-sentence; retain only complete passages.
            break
        context_parts.append(part)
        used_chars += len(part)
    context = "\n".join(context_parts)
    if intent == "future":
        focus = (
            "Focus on explicitly proposed future work, planned improvements, "
            "extensions, and recommendations. Distinguish planned work from "
            "features already described."
        )
    elif intent == "conclusion":
        focus = (
            "Find the actual conclusion section and summarize its central message, "
            "contributions, and stated implications. Do not answer using only the "
            "introduction or background. If a Conclusions heading appears, use "
            "the text following that heading as primary evidence."
        )
    elif intent == "conclusion_future":
        focus = (
            "Answer in separate sections: Conclusion and Future Work. Use the "
            "actual conclusion section for the first part and explicit future "
            "proposals for the second part."
        )
    elif intent == "limitations":
        focus = (
            "Report limitations, drawbacks, or threats to validity only when "
            "supported by the supplied passages. Distinguish explicitly stated "
            "limitations from inferred limitations. If the passages do not "
            "establish any, say that the provided evidence does not identify them. "
            "Do not claim no limitations exist merely because no dedicated section "
            "was found."
        )
    elif comparison:
        focus = (
            "This is a multi-document comparison question. Use evidence from each "
            "relevant source represented in the supplied passages. Keep the papers' "
            "claims separate and attribute every claim to the correct filename. "
            "For contribution-evidence questions, identify the contribution and its "
            "supporting evidence separately for each paper. If passages from one "
            "requested paper are missing, state that clearly; do not fabricate a "
            "comparison or assume a paper is absent from the index."
        )
    elif intent == "contribution":
        focus = (
            "Identify the paper's explicitly supported main contribution and the "
            "evidence supporting it. Prefer relevant abstract, introduction, results, "
            "discussion, and conclusion passages when present. Separate the authors' "
            "claims from experimental evidence, and cite the exact source filename and "
            "page for each claim. Do not infer a contribution from references or "
            "administrative material alone."
        )
    else:
        focus = "Answer the exact question using the most directly relevant evidence from the supplied passages."
    # -------------------------------------------------
    # BUILD AN EXACT CITATION ALLOWLIST
    # -------------------------------------------------
    allowed_pairs = {
        (
            str(item.get("source", "")).strip(),
            str(_page_number(item.get("page")))
        )
        for item in passages
        if str(item.get("source", "")).strip()
        and str(item.get("source", "")).strip().casefold() != "unknown"
        and str(_page_number(item.get("page"))) != "Unknown"
    }
    allowed_citations = "\n".join(
        f"- (Source: {source}, p. {page})"
        for source, page in sorted(
            allowed_pairs,
            key=lambda pair: (pair[0].casefold(), int(pair[1]))
        )
    )
    if not allowed_citations:
        return (
            "I could not find verified source and page information "
            "in the retrieved passages. Please try another question."
        )
    # -------------------------------------------------
    # GEMINI PROMPT
    # -------------------------------------------------
    prompt = f"""
You are TraceRAG, a careful and accurate research-paper assistant.
Your task is to answer the user's question using ONLY the supplied
document passages. Do not use outside knowledge.
QUESTION FOCUS:
{focus}
STRICT RULES:
1\\. Answer the exact question directly.
2\\. Use clear, concise, well-organized language.
3\\. Synthesize relevant passages when useful.
4\\. Never invent facts, experiments, results, contributions, or proposals.
5\\. Distinguish explicit statements from reasonable inference.
6\\. Cite important factual claims using exactly this format:
   (Source: exact_filename.pdf, p. 10)
7\\. CITATION ALLOWLIST:
   You may ONLY use the following source and page combinations:
{allowed_citations}
8\\. Copy each filename exactly as it appears in the citation allowlist.
9\\. Never create a filename, modify a filename, or substitute a different PDF.
10\\. Never cite a page number that is not listed for that exact source.
11\\. Never combine source filenames or page numbers.
12\\. Never use decimal page numbers.
13\\. Do not generate citations such as:
    [Unverified citation: ...]
    (Source: invented.pdf, p. 5)
14\\. If a claim has no supporting passage, state that the supplied
    evidence does not establish it.
15\\. Do not claim the entire paper lacks information merely because
    the supplied passages do not contain it.
16\\. For conclusion questions, prioritize the actual Conclusions
    section if it appears in the supplied passages.
17\\. For future-work questions, report only explicitly proposed work.
18\\. For limitations questions, do not invent limitations.
19\\. If asked about conclusion and future work, use separate headings.
20\\. For contribution questions, prioritize passages that state
    contributions and passages that provide direct supporting evidence.
21\\. For comparison questions, distinguish the documents clearly.
22\\. Never attribute a passage to a different document.
23\\. If only one source appears in the supplied passages, do not
    pretend you compared multiple papers.
24\\. If the user names exactly one PDF, answer only from that PDF.
25\\. If the user explicitly compares multiple named PDFs, use evidence
    from each named PDF when available.
26\\. Every citation must match one exact source/page combination
    from the citation allowlist above.
27\\. If different sections of a paper report conflicting values,
    identify the discrepancy and cite each relevant passage separately.
28\\. Do not silently resolve contradictions between the paper's
    abstract, main text, tables, or conclusion.
29\\. Do not repeat the same citation unnecessarily.
DOCUMENT PASSAGES:
{context}
USER QUESTION:
{question}
FINAL INSTRUCTION:
Write a direct, evidence-based answer.
Use only citations from the citation allowlist.
If evidence is insufficient, explain the limitation instead of guessing.
""".strip()
    # -------------------------------------------------
    # GENERATE AND VALIDATE ANSWER
    # -------------------------------------------------
    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt
        )
        answer = (response.text or "").strip()
        if not answer:
            return "The AI returned an empty answer. Please try again."
        return _validate_citations(answer, passages)
    except Exception as error:
        print(f"Gemini API error: {error}")
        raise RuntimeError(
            "Answer generation failed. Check the Gemini model name, "
            "API quota, internet connection, and backend/.env."
        ) from error
# ============================================================
# 9. OPTIONAL TERMINAL INTERFACE
# ============================================================
if __name__ == "__main__":
    print("\n====================================")
    print("       TraceRAG Research Assistant")
    print("====================================")
    try:
        print(f"Indexed chunks: {get_collection().count()}")
    except Exception as error:
        print(f"Could not read the collection: {error}")
    print("Type 'exit' to stop.\n")
    while True:
        try:
            question = input("You: ").strip()
            if question.casefold() in {"exit", "quit", "stop"}:
                print("TraceRAG stopped.")
                break
            if not question:
                print("Please enter a question.\n")
                continue
            print("\nTraceRAG:")
            print(answer_question(question))
            print()
        except KeyboardInterrupt:
            print("\nTraceRAG stopped.")
            break
        except Exception as error:
            print(f"\nError: {error}\n")
