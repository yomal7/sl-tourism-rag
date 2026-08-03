from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "db" / "tourism.db"
CHROMA_PATH = PROJECT_ROOT / "vector_store" / "chroma"
IMAGES_ROOT = PROJECT_ROOT / "data" / "images"

TEXT_COLLECTION_NAME = "destinations_text"
IMAGE_COLLECTION_NAME = "destinations_image"
TEXT_EMBED_MODEL = "all-MiniLM-L6-v2"
CLIP_MODEL = "clip-ViT-B-32"
CLIP_MODEL_NAME = CLIP_MODEL

CATEGORY_TO_FOLDER = {
    "beach": "beaches",
    "historical_site": "historical_sites",
    "mountain": "mountains",
    "national_park": "national_parks",
    "waterfall": "waterfalls",
    "temple": "temples",
}