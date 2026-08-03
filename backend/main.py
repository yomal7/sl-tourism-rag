import tempfile
from pathlib import Path

from fastapi import FastAPI, Form, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from backend.config import settings
from backend.generate_response import answer_query
from backend.llm_client import LLMServiceError

app = FastAPI(
    title="Sri Lanka Tourism RAG API",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# Endpoint for health check.
@app.get("/api/health")
def health():
    return {"status": "ok", "llm_provider": settings.llm_provider}

# Endpoint for answering queries with optional image upload.
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
    except LLMServiceError as e:
        raise HTTPException(status_code=e.status_code, detail=e.user_message)
    finally:
        if image_path:
            Path(image_path).unlink(missing_ok=True)

    return result
