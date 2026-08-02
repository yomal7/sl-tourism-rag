
from backend.retrieve import retrieve, RetrievalContext
from backend.router import describe_route
from backend.llm_client import generate

SYSTEM_PROMPT = """You are a knowledgeable, friendly Sri Lanka travel assistant.

Rules you must follow:
- Answer ONLY using the information given to you in the "Context" section below.
- If the context doesn't fully answer the question, say what you don't know
  rather than inventing facts.
- Do not make up entrance fees, hours, or other specifics that aren't in the context.
- Keep your answer natural and conversational, not just a list of facts.
- If images are mentioned in the context, refer to them naturally by the
  destination's name only (e.g. "as you can see in the photo of Mirissa
  Beach"). NEVER mention filenames, file extensions, or any technical
  identifiers — the person reading your answer should never see things
  like "toothtemple_1.jpg".
"""


def format_sql_results(rows: list[dict]) -> str:
    if not rows:
        return ""
    lines = ["Structured facts (from database):"]
    for r in rows:
        lines.append(
            f"- {r['name']} ({r['category']}, {r['location']}, {r['district']}): "
            f"entrance fee {r['entrance_fee_lkr']} LKR, "
            f"accessibility: {r['accessibility']}, "
            f"opening hours: {r['opening_hours']}, "
            f"best time to visit: {r['best_time_to_visit']}, "
            f"notable: {r['significance_or_activities']}."
        )
    return "\n".join(lines)


def format_text_results(rows: list[dict]) -> str:
    if not rows:
        return ""
    lines = ["Descriptive information (from semantic search):"]
    for r in rows:
        lines.append(f"- {r['name']}: {r['passage']}")
    return "\n".join(lines)


def format_image_results(rows: list[dict]) -> str:
    if not rows:
        return ""
    lines = ["Related images found (from image search), most visually relevant first:"]
    for r in rows:
        lines.append(f"- {r['name']}")
    return "\n".join(lines)


def build_context_block(ctx: RetrievalContext) -> str:
    sections = [
        format_sql_results(ctx.sql_results),
        format_text_results(ctx.text_results),
        format_image_results(ctx.image_results),
    ]
    sections = [s for s in sections if s]  # drop empty sections
    if not sections:
        return "No relevant information was found in the knowledge base for this query."
    return "\n\n".join(sections)


def generate_response(ctx: RetrievalContext) -> str:
    context_block = build_context_block(ctx)

    user_prompt = f"""Context:
{context_block}

Question: {ctx.query_text}

Write a helpful, natural-language answer using only the context above."""

    return generate(SYSTEM_PROMPT, user_prompt)


def answer_query(query_text: str, uploaded_image_path: str | None = None) -> dict:
    ctx = retrieve(query_text, uploaded_image_path=uploaded_image_path)
    answer_text = generate_response(ctx)

    seen_paths = set()
    image_paths = []
    for r in ctx.image_results:
        path = r.get("filepath")
        if path and path not in seen_paths:
            seen_paths.add(path)
            image_paths.append({"filepath": path, "name": r.get("name", "")})

    return {
        "query": query_text,
        "answer": answer_text,
        "route": describe_route(ctx.route),
        "sql_results": ctx.sql_results,
        "text_results": ctx.text_results,
        "image_results": ctx.image_results,
        "image_paths_to_display": image_paths,
    }


if __name__ == "__main__":
    demo_queries = [
        "What is the entrance fee for Sigiriya and when is it open?",
        "Suggest a peaceful place for meditation with some background on it",
        "Tell me about a good beach for surfing with a lively atmosphere and show me what it looks like",
    ]

    for q in demo_queries:
        print(f"\n{'='*70}")
        print(f"Query: {q}")
        result = answer_query(q)
        print(result["route"])
        print("\nAnswer:\n" + result["answer"])