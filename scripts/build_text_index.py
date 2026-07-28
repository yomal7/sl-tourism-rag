"""
build_text_index.py
--------------------
Builds the TEXT vector database for semantic retrieval.

What it does:
1. Reads destinations from db/tourism.db (the SQLite database from Phase 2)
2. For each destination, builds a rich text passage (name + category +
   location + description + activities) — richer input = better embeddings
3. Embeds each passage using a Sentence Transformer model
4. Stores the embeddings + metadata in a persistent ChromaDB collection

Run from the project root (after setup_db.py has been run):
    python scripts/build_text_index.py

Re-running this script is safe — it clears and rebuilds the collection
each time, so it always reflects the latest database contents.
"""

import sqlite3
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "db" / "tourism.db"
CHROMA_PATH = PROJECT_ROOT / "vector_store" / "chroma"
COLLECTION_NAME = "destinations_text"

# A small, fast, well-regarded general-purpose embedding model.
# 384-dimensional vectors, runs fine on CPU, no GPU needed.
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"


def fetch_destinations():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute("SELECT * FROM destinations;")
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows


def build_passage(row: dict) -> str:
    """
    Combine several fields into one text passage to embed.
    Including name/category/location helps the model match queries that
    mention a place type or region, not just descriptive wording.
    """
    parts = [
        row.get("name", ""),
        f"Category: {row.get('category', '')}",
        f"Location: {row.get('location', '')}, {row.get('district', '')}",
        row.get("description", "") or "",
        f"Activities/significance: {row.get('significance_or_activities', '')}",
    ]
    return ". ".join(p for p in parts if p)


def build_text_index():
    if not DB_PATH.exists():
        raise FileNotFoundError(
            f"{DB_PATH} not found. Run scripts/setup_db.py first (Phase 2)."
        )

    print("Loading destinations from SQLite...")
    rows = fetch_destinations()
    print(f"Found {len(rows)} destinations.")

    print(f"Loading embedding model '{EMBEDDING_MODEL_NAME}' (first run downloads it)...")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    passages = [build_passage(r) for r in rows]
    ids = [str(r["id"]) for r in rows]
    metadatas = [
        {
            "name": r["name"],
            "category": r["category"],
            "subcategory": r.get("subcategory") or "",
            "location": r.get("location") or "",
            "district": r.get("district") or "",
            "entrance_fee_lkr": r.get("entrance_fee_lkr") or 0,
            "image_filenames": r.get("image_filenames") or "",
        }
        for r in rows
    ]

    print("Generating embeddings...")
    embeddings = model.encode(passages, show_progress_bar=True).tolist()

    print(f"Writing to ChromaDB at {CHROMA_PATH} ...")
    CHROMA_PATH.parent.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))

    # Fresh start each run so re-running always reflects the latest DB
    existing = [c.name for c in client.list_collections()]
    if COLLECTION_NAME in existing:
        client.delete_collection(COLLECTION_NAME)

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"description": "Text embeddings of Sri Lanka tourism destinations"},
    )

    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=passages,
        metadatas=metadatas,
    )

    print(f"Indexed {collection.count()} passages into collection '{COLLECTION_NAME}'.")

    # ---- Sanity check: run a couple of sample semantic queries ----
    print("\n--- Sample semantic query: 'peaceful place for meditation and reflection' ---")
    results = collection.query(
        query_texts=["peaceful place for meditation and reflection"], n_results=3
    )
    for name, dist in zip(
        [m["name"] for m in results["metadatas"][0]], results["distances"][0]
    ):
        print(f"  {name}  (distance={dist:.4f})")

    print("\n--- Sample semantic query: 'good beach for surfing with a lively atmosphere' ---")
    results = collection.query(
        query_texts=["good beach for surfing with a lively atmosphere"], n_results=3
    )
    for name, dist in zip(
        [m["name"] for m in results["metadatas"][0]], results["distances"][0]
    ):
        print(f"  {name}  (distance={dist:.4f})")

    print("\nDone.")


if __name__ == "__main__":
    build_text_index()
