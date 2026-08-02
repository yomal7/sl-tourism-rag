import sqlite3
from pathlib import Path
from dataclasses import dataclass, field

import chromadb
from sentence_transformers import SentenceTransformer
from PIL import Image

from backend.router import classify_query, describe_route, RouteDecision

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "db" / "tourism.db"
CHROMA_PATH = PROJECT_ROOT / "vector_store" / "chroma"

TEXT_COLLECTION_NAME = "destinations_text"
IMAGE_COLLECTION_NAME = "destinations_image"
TEXT_EMBED_MODEL = "all-MiniLM-L6-v2"
CLIP_MODEL = "clip-ViT-B-32"

_text_model = None
_clip_model = None
_chroma_client = None


def _get_text_model():
    global _text_model
    if _text_model is None:
        _text_model = SentenceTransformer(TEXT_EMBED_MODEL)
    return _text_model


def _get_clip_model():
    global _clip_model
    if _clip_model is None:
        _clip_model = SentenceTransformer(CLIP_MODEL)
    return _clip_model


def _get_chroma_client():
    global _chroma_client
    if _chroma_client is None:
        _chroma_client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    return _chroma_client


# SQL retrieval
def query_sql(decision: RouteDecision, limit: int = 5) -> list[dict]:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    clauses, params = [], []

    if decision.matched_destination_names:
        placeholders = ",".join("?" for _ in decision.matched_destination_names)
        clauses.append(f"name IN ({placeholders})")
        params.extend(decision.matched_destination_names)

    if decision.category_filter:
        clauses.append("category = ?")
        params.append(decision.category_filter)

    if decision.max_fee_lkr is not None:
        clauses.append("entrance_fee_lkr <= ?")
        params.append(decision.max_fee_lkr)

    sql = "SELECT * FROM destinations"
    if clauses:
        sql += " WHERE " + " AND ".join(clauses)
    sql += f" LIMIT {limit};"

    rows = [dict(r) for r in cur.execute(sql, params).fetchall()]
    conn.close()
    return rows


# Text semantic retrieval
def query_text_semantic(query_text: str, top_k: int = 3) -> list[dict]:
    model = _get_text_model()
    client = _get_chroma_client()
    collection = client.get_collection(TEXT_COLLECTION_NAME)

    embedding = model.encode(query_text).tolist()
    results = collection.query(query_embeddings=[embedding], n_results=top_k)

    matches = []
    for doc, meta, dist in zip(
        results["documents"][0], results["metadatas"][0], results["distances"][0]
    ):
        matches.append({**meta, "passage": doc, "distance": dist})
    return matches



# Image retrieval
def query_image_by_image(image_path: str, top_k: int = 3, category_filter: str | None = None) -> list[dict]:
    model = _get_clip_model()
    client = _get_chroma_client()
    collection = client.get_collection(IMAGE_COLLECTION_NAME)

    img = Image.open(image_path).convert("RGB")
    embedding = model.encode(img).tolist()
    where = {"category": category_filter} if category_filter else None
    results = collection.query(query_embeddings=[embedding], n_results=top_k, where=where)

    return [
        {**meta, "distance": dist}
        for meta, dist in zip(results["metadatas"][0], results["distances"][0])
    ]


def query_image_by_text(query_text: str, top_k: int = 3, category_filter: str | None = None) -> list[dict]:
    model = _get_clip_model()
    client = _get_chroma_client()
    collection = client.get_collection(IMAGE_COLLECTION_NAME)

    embedding = model.encode(query_text).tolist()
    where = {"category": category_filter} if category_filter else None
    results = collection.query(query_embeddings=[embedding], n_results=top_k, where=where)

    return [
        {**meta, "distance": dist}
        for meta, dist in zip(results["metadatas"][0], results["distances"][0])
    ]


def query_images_for_destinations(names: list[str], top_k: int = 3) -> list[dict]:
    if not names:
        return []

    client = _get_chroma_client()
    collection = client.get_collection(IMAGE_COLLECTION_NAME)
    got = collection.get(where={"name": {"$in": names}})

    metadatas = got.get("metadatas", []) or []

    by_name: dict[str, dict] = {}
    for meta in metadatas:
        by_name.setdefault(meta["name"], meta)

    ordered = []
    for name in names:
        if name in by_name and len(ordered) < top_k:
            ordered.append({**by_name[name], "distance": None, "matched_via": "text/SQL retrieval"})

    return ordered


@dataclass
class RetrievalContext:
    query_text: str
    route: RouteDecision
    sql_results: list = field(default_factory=list)
    text_results: list = field(default_factory=list)
    image_results: list = field(default_factory=list)

def retrieve(query_text: str, uploaded_image_path: str | None = None,
             sql_limit: int = 20, semantic_top_k: int = 3, image_top_k: int = 3) -> RetrievalContext:
    decision = classify_query(query_text, image_provided=uploaded_image_path is not None)
    ctx = RetrievalContext(query_text=query_text, route=decision)

    if decision.use_sql:
        ctx.sql_results = query_sql(decision, limit=sql_limit)

    if decision.use_text_semantic:
        ctx.text_results = query_text_semantic(query_text, top_k=semantic_top_k)

    if decision.use_image:
        if uploaded_image_path:
            ctx.image_results = query_image_by_image(
                uploaded_image_path, top_k=image_top_k, category_filter=decision.category_filter
            )
        else:
            anchor_names = []
            for r in ctx.sql_results + ctx.text_results:
                if r["name"] not in anchor_names:
                    anchor_names.append(r["name"])

            if anchor_names:
                ctx.image_results = query_images_for_destinations(anchor_names, top_k=image_top_k)
            else:
                ctx.image_results = query_image_by_text(
                    query_text, top_k=image_top_k, category_filter=decision.category_filter
                )

    return ctx


if __name__ == "__main__":
    test_queries = [
        "What is the entrance fee for Sigiriya?",
        "Suggest a peaceful place for meditation",
        "Tell me about a good beach for surfing with a lively atmosphere",
    ]
    for q in test_queries:
        print(f"\n{'='*70}\nQuery: {q}")
        ctx = retrieve(q)
        print(describe_route(ctx.route))
        print(f"SQL results: {[r['name'] for r in ctx.sql_results]}")
        print(f"Text results: {[r['name'] for r in ctx.text_results]}")
        print(f"Image results: {[r['name'] for r in ctx.image_results]}")