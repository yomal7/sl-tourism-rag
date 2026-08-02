import sqlite3
from pathlib import Path

import chromadb
from PIL import Image
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "db" / "tourism.db"
IMAGES_ROOT = PROJECT_ROOT / "data" / "images"
CHROMA_PATH = PROJECT_ROOT / "vector_store" / "chroma"
COLLECTION_NAME = "destinations_image"

CLIP_MODEL_NAME = "clip-ViT-B-32"

CATEGORY_TO_FOLDER = {
    "beach": "beaches",
    "historical_site": "historical_sites",
    "mountain": "mountains",
    "national_park": "national_parks",
    "waterfall": "waterfalls",
    "temple": "temples",
}


def fetch_destinations():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute("SELECT * FROM destinations;")
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows


def find_image_path(filename: str, category: str) -> Path | None:
    folder = CATEGORY_TO_FOLDER.get(category)
    if folder:
        candidate = IMAGES_ROOT / folder / filename
        if candidate.exists():
            return candidate

    for sub in IMAGES_ROOT.iterdir():
        if sub.is_dir():
            candidate = sub / filename
            if candidate.exists():
                return candidate

    return None


def build_image_index():
    if not DB_PATH.exists():
        raise FileNotFoundError(f"{DB_PATH} not found. Run scripts/setup_db.py first.")
    if not IMAGES_ROOT.exists():
        raise FileNotFoundError(f"{IMAGES_ROOT} not found.")

    print("Loading destinations from SQLite...")
    rows = fetch_destinations()

    print(f"Loading CLIP model '{CLIP_MODEL_NAME}' (first run downloads it)...")
    model = SentenceTransformer(CLIP_MODEL_NAME)

    ids, embeddings, metadatas = [], [], []
    missing_files = []

    for row in rows:
        filenames_raw = row.get("image_filenames") or ""
        filenames = [f.strip() for f in filenames_raw.split(";") if f.strip()]

        for idx, filename in enumerate(filenames):
            path = find_image_path(filename, row["category"])
            if path is None:
                missing_files.append(filename)
                continue

            try:
                img = Image.open(path).convert("RGB")
            except Exception as e:
                print(f"  Skipping {path} — could not open ({e})")
                continue

            embedding = model.encode(img).tolist()

            image_id = f"{row['id']}_{idx}"
            ids.append(image_id)
            embeddings.append(embedding)
            metadatas.append(
                {
                    "destination_id": row["id"],
                    "name": row["name"],
                    "category": row["category"],
                    "location": row.get("location") or "",
                    "filename": filename,
                    "filepath": str(path),
                }
            )

    print(f"\nSuccessfully embedded {len(ids)} images.")
    if missing_files:
        print(f"WARNING: {len(missing_files)} image file(s) listed in the CSV were not "
              f"found under data/images/: {missing_files}")
        print("Double check filenames match exactly (case-sensitive) and are in the "
              "right subfolder.")

    if not ids:
        print("No images were indexed — nothing to write to ChromaDB. Stopping.")
        return

    print(f"\nWriting to ChromaDB at {CHROMA_PATH} ...")
    CHROMA_PATH.parent.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))

    existing = [c.name for c in client.list_collections()]
    if COLLECTION_NAME in existing:
        client.delete_collection(COLLECTION_NAME)

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"description": "CLIP image embeddings of Sri Lanka tourism destinations"},
    )
    collection.add(ids=ids, embeddings=embeddings, metadatas=metadatas)
    print(f"Indexed {collection.count()} images into collection '{COLLECTION_NAME}'.")

    print("\n--- Sample text-to-image query: 'golden sandy beach with palm trees' ---")
    text_embedding = model.encode("golden sandy beach with palm trees").tolist()
    results = collection.query(query_embeddings=[text_embedding], n_results=3)
    for name, dist in zip(
        [m["name"] for m in results["metadatas"][0]], results["distances"][0]
    ):
        print(f"  {name}  (distance={dist:.4f})")

    print("\n--- Sample text-to-image query: 'ancient stone ruins and statues' ---")
    text_embedding = model.encode("ancient stone ruins and statues").tolist()
    results = collection.query(query_embeddings=[text_embedding], n_results=3)
    for name, dist in zip(
        [m["name"] for m in results["metadatas"][0]], results["distances"][0]
    ):
        print(f"  {name}  (distance={dist:.4f})")

    print("\nDone.")


if __name__ == "__main__":
    build_image_index()
