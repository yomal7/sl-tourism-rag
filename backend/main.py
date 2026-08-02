import tempfile
from pathlib import Path

from fastapi import FastAPI, Form, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from backend.config import settings
from backend.generate_response import answer_query

app = FastAPI(
    title="Sri Lanka Tourism RAG API",
    description="Structured (SQL) + semantic (text) + image retrieval, "
    "combined and passed to an LLM for a grounded answer.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health():
    """Quick check that the API is up and which LLM provider it's using."""
    return {"status": "ok", "llm_provider": settings.llm_provider}


@app.post("/api/query")
async def query(
    query: str = Form(...),
    image: UploadFile | None = File(None),
):
    image_path = None
    if image is not None:
        suffix = Path(image.filename).suffix
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tmp.write(await image.read())
        tmp.close()
        image_path = tmp.name

    try:
        result = answer_query(query, uploaded_image_path=image_path)
    finally:
        if image_path:
            Path(image_path).unlink(missing_ok=True)

    return result
