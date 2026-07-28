"""
generate_response.py
---------------------
Phase 6: Context Integration + LLM Generation.

Takes a RetrievalContext (from retrieve.py) — which may contain SQL rows,
semantic text matches, and/or image matches — formats it all into one
clear text block, and asks a local Ollama model to write a grounded,
natural-language answer using ONLY that context.

Requires Ollama running locally (https://ollama.com):
    1. Install Ollama, then run:  ollama pull llama3.1
    2. Make sure the Ollama app/service is running (it runs a local
       server at http://localhost:11434 automatically once installed)
    3. python scripts/generate_response.py   (runs a demo query)

You can swap MODEL_NAME for any model you've pulled, e.g. "mistral",
"llama3.1:8b", "phi3", etc. Smaller models are faster but less fluent —
worth mentioning your choice and why in the report.
"""

from pathlib import Path
import sys

import ollama

sys.path.insert(0, str(Path(__file__).resolve().parent))
from retrieve import retrieve, RetrievalContext
from router import describe_route

MODEL_NAME = "llama3.1"

SYSTEM_PROMPT = """You are a knowledgeable, friendly Sri Lanka travel assistant.

Rules you must follow:
- Answer ONLY using the information given to you in the "Context" section below.
- If the context doesn't fully answer the question, say what you don't know
  rather than inventing facts.
- Do not make up entrance fees, hours, or other specifics that aren't in the context.
- Keep your answer natural and conversational, not just a list of facts.
- If images are mentioned in the context, refer to them naturally
  (e.g. "as you can see in the photo of Mirissa Beach").
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
    lines = ["Related images found (from image search):"]
    for r in rows:
        lines.append(f"- {r['name']} — image file: {r['filename'] if 'filename' in r else r.get('filepath', 'unknown')}")
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
    """
    Sends the retrieved context + user question to the local Ollama model
    and returns the generated answer text.
    """
    context_block = build_context_block(ctx)

    user_prompt = f"""Context:
{context_block}

Question: {ctx.query_text}

Write a helpful, natural-language answer using only the context above."""

    response = ollama.chat(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )
    return response["message"]["content"]


def answer_query(query_text: str, uploaded_image_path: str | None = None) -> dict:
    """
    Full end-to-end pipeline for one query: retrieve -> integrate context
    -> generate. Returns a dict with everything useful for display in the
    Streamlit app (Phase 7): the answer text, which route was used, and
    which images to show alongside it.
    """
    ctx = retrieve(query_text, uploaded_image_path=uploaded_image_path)
    answer_text = generate_response(ctx)

    image_paths = list({r.get("filepath") for r in ctx.image_results if r.get("filepath")})

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
