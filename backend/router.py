"""
router.py
---------
Decides WHICH data source(s) a user query needs:
  - SQL (structured facts: fees, hours, difficulty, location)
  - Text semantic search (descriptive/mood-based queries)
  - Image search (query includes an uploaded image)
  - Hybrid (a mix of the above — most natural-language questions end up here)

Approach: simple, explainable rule-based classification using keyword
matching and entity extraction. This is deliberately NOT a black box —
you can point to exactly why a query was routed a certain way, which is
easy to describe and defend in your report. (An LLM-based classifier is
a possible extension — see the note at the bottom of this file.)
"""

import re
import sqlite3
from pathlib import Path
from dataclasses import dataclass, field

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "db" / "tourism.db"

# Keywords that suggest the user wants a specific FACT (structured/SQL)
STRUCTURED_KEYWORDS = [
    "fee", "cost", "price", "how much", "entrance", "ticket",
    "hours", "open", "close", "timing", "when does",
    "height", "difficulty", "accessible", "accessibility",
    "district", "located", "location of", "how far",
    "list", "which beaches", "which temples", "which historical",
    "free", "cheapest", "most expensive",
]

# Keywords that suggest a MOOD/DESCRIPTIVE query (semantic/vector)
SEMANTIC_KEYWORDS = [
    "peaceful", "romantic", "relaxing", "adventurous", "quiet",
    "secluded", "lively", "recommend", "suggest", "best for",
    "similar to", "vibe", "atmosphere", "feel like", "family-friendly",
    "hidden gem", "off the beaten path", "spiritual", "scenic",
]

# Keywords that suggest the user wants to SEE something (image search),
# even without uploading a photo. CLIP lets us search the image collection
# using text directly, so these don't require an uploaded image.
IMAGE_KEYWORDS = [
    "show me", "what does it look like", "what do they look like",
    "picture", "photo", "photos", "image", "images", "look like",
    "visual", "scenery", "view of", "what it looks like",
]

# Regex to catch "under 1000", "less than 500 rupees", etc.
FEE_PATTERN = re.compile(r"(under|less than|below|cheaper than)\s+(\d+)", re.IGNORECASE)


@dataclass
class RouteDecision:
    use_sql: bool = False
    use_text_semantic: bool = False
    use_image: bool = False
    matched_destination_names: list = field(default_factory=list)
    max_fee_lkr: int | None = None
    category_filter: str | None = None
    reasoning: list = field(default_factory=list)


def _load_destination_names():
    """Used for simple entity matching: does the query mention a place by name?"""
    conn = sqlite3.connect(DB_PATH)
    names = [row[0] for row in conn.execute("SELECT name FROM destinations;")]
    conn.close()
    return names


def classify_query(query_text: str, image_provided: bool = False) -> RouteDecision:
    """
    Main entry point. Returns a RouteDecision describing which retrieval
    functions (from retrieve.py) should be called for this query.
    """
    decision = RouteDecision()
    text_lower = query_text.lower()

    # --- Image provided => image retrieval is needed ---
    if image_provided:
        decision.use_image = True
        decision.reasoning.append("An image was provided, so image similarity search is used.")

    # --- Entity matching: does the query name a specific destination? ---
    # Checks both the full name ("Temple of the Sacred Tooth Relic") and its
    # significant individual words ("Sigiriya" alone should still match
    # "Sigiriya Rock Fortress"). Short/common words are skipped to avoid
    # false positives (e.g. matching "Beach" or "Temple" as if it were a name).
    STOPWORDS = {"the", "of", "and", "at", "in", "temple", "beach", "fort",
                 "ancient", "city", "rock", "cave", "sacred"}
    all_names = _load_destination_names()
    for name in all_names:
        name_lower = name.lower()
        if name_lower in text_lower:
            decision.matched_destination_names.append(name)
            continue
        significant_words = [
            w for w in re.findall(r"[a-z']+", name_lower)
            if len(w) > 3 and w not in STOPWORDS
        ]
        if any(w in text_lower for w in significant_words):
            decision.matched_destination_names.append(name)

    if decision.matched_destination_names:
        decision.use_sql = True
        decision.reasoning.append(
            f"Query mentions specific destination(s): {decision.matched_destination_names} "
            "-> structured lookup."
        )

    # --- Category filter (helps narrow SQL and can help semantic too) ---
    if "beach" in text_lower:
        decision.category_filter = "beach"
    elif "temple" in text_lower or "historical" in text_lower or "heritage" in text_lower:
        decision.category_filter = "historical_site"

    # --- Fee constraint extraction ---
    fee_match = FEE_PATTERN.search(text_lower)
    if fee_match:
        decision.max_fee_lkr = int(fee_match.group(2))
        decision.use_sql = True
        decision.reasoning.append(f"Detected a fee constraint: under {decision.max_fee_lkr} LKR.")
    if "free" in text_lower:
        decision.max_fee_lkr = 0
        decision.use_sql = True
        decision.reasoning.append("Detected 'free' -> filtering entrance_fee_lkr = 0.")

    # --- Keyword-based classification ---
    structured_hit = any(kw in text_lower for kw in STRUCTURED_KEYWORDS)
    semantic_hit = any(kw in text_lower for kw in SEMANTIC_KEYWORDS)
    image_hit = any(kw in text_lower for kw in IMAGE_KEYWORDS)

    if structured_hit:
        decision.use_sql = True
        decision.reasoning.append("Query contains factual/structured keywords.")
    if semantic_hit:
        decision.use_text_semantic = True
        decision.use_image = True
        decision.reasoning.append(
            "Query contains descriptive/mood keywords -> semantic search, "
            "plus a representative image via CLIP text-to-image."
        )
    if image_hit:
        decision.use_image = True
        decision.reasoning.append(
            "Query asks to see/visualize something -> image search via CLIP text-to-image."
        )

    # --- Fallback: if nothing matched, default to full hybrid (SQL + semantic + image) ---
    # Most open-ended natural language questions benefit from structured
    # facts, descriptive context, AND a representative photo, so this is a
    # safe default rather than returning nothing.
    if not (decision.use_sql or decision.use_text_semantic or decision.use_image):
        decision.use_sql = True
        decision.use_text_semantic = True
        decision.use_image = True
        decision.reasoning.append(
            "No specific signal detected -> defaulting to full hybrid "
            "(SQL + semantic + image) to give the LLM the richest possible context."
        )

    return decision


def describe_route(decision: RouteDecision) -> str:
    parts = []
    if decision.use_sql:
        parts.append("SQL")
    if decision.use_text_semantic:
        parts.append("Text-Semantic")
    if decision.use_image:
        parts.append("Image")
    label = " + ".join(parts) if parts else "None"
    return f"[Route: {label}]  Reasoning: {'; '.join(decision.reasoning)}"


# ---------------------------------------------------------------------------
# NOTE / possible extension for your report:
# Instead of keyword rules, you could ask the LLM itself to classify the
# query type by prompting it to output one of {structured, semantic, image,
# hybrid} before generation. That's more flexible but less transparent and
# costs an extra API call per query. Keyword rules are simpler to explain,
# debug, and evaluate for a project like this — worth mentioning both
# approaches and your tradeoff reasoning in the report.
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_queries = [
        "What is the entrance fee for Sigiriya?",
        "Suggest a peaceful place for meditation",
        "Which beaches are free to enter?",
        "Tell me about a good beach for surfing with a lively atmosphere",
        "What time does the Temple of the Sacred Tooth Relic open?",
    ]
    for q in test_queries:
        d = classify_query(q)
        print(f"\nQuery: {q}")
        print(describe_route(d))