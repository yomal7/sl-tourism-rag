# Sri Lanka Tourism — Multimodal RAG System

A Retrieval-Augmented Generation system for Sri Lanka tourism (beaches +
historical sites/temples), combining a relational database (SQLite),
a text vector database (ChromaDB + Sentence Transformers), and an image
vector database (ChromaDB + CLIP). Retrieved context is passed to an LLM
(Gemini API or a local Ollama model — configurable per teammate) to
generate the final answer.

## Project structure
```
sl-tourism-rag/
├── backend/                     # FastAPI app: retrieval + LLM generation
│   ├── main.py                  # API endpoints (/api/health, /api/query)
│   ├── config.py                # reads .env into typed settings
│   ├── llm_client.py            # Gemini / Ollama switch, per LLM_PROVIDER
│   ├── router.py                # classifies query type (SQL/semantic/image)
│   ├── retrieve.py              # retrieval functions per source
│   └── generate_response.py     # context integration + LLM call
├── frontend/
│   └── streamlit_app.py         # calls the backend over HTTP
├── scripts/                     # one-off build scripts (run once, or after data changes)
│   ├── setup_db.py              # builds SQLite from destinations.csv
│   ├── build_text_index.py      # embeds descriptions into ChromaDB
│   └── build_image_index.py     # embeds images (CLIP) into ChromaDB
├── data/
│   ├── destinations.csv         # master spreadsheet (facts + descriptions)
│   └── images/                  # source images, organized by category
├── db/                          # SQLite database (generated, not in git)
├── vector_store/                # ChromaDB storage (generated, not in git)
├── notes/                       # evaluation logs, image source citations
├── pyproject.toml               # dependencies (managed with uv)
├── .env.example                 # copy to .env and fill in
└── .gitignore
```

## Setup instructions (run once per teammate)

### 1. Clone the repo
```bash
git clone <repo-url>
cd sl-tourism-rag
```

### 2. Install uv (if you don't have it)
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 3. Create the virtual environment and install dependencies
```bash
uv venv
uv sync
```
This creates `.venv/` and installs everything from `pyproject.toml`
(pinned exactly via `uv.lock`, so every teammate gets identical versions).
`sentence-transformers` pulls in PyTorch, so the first `uv sync` can take
a few minutes.

### 4. Configure your `.env`
```bash
cp .env.example .env
```
Then edit `.env`:
- **No local Ollama install?** Set `LLM_PROVIDER=gemini` and add a free
  key from [Google AI Studio](https://aistudio.google.com/apikey) as
  `GEMINI_API_KEY`.
- **Have Ollama running locally?** Set `LLM_PROVIDER=ollama`, then:
  ```bash
  ollama pull llama3.1
  ollama list   # confirms the server is running
  ```

Each teammate's `.env` can differ — the code reads `LLM_PROVIDER` at
runtime and picks the right path, so this is not something you need to
agree on as a team or hardcode.

### 5. Build the databases
Run these **in order** — each one depends on the previous step:
```bash
uv run python scripts/setup_db.py           # builds SQLite from data/destinations.csv
uv run python scripts/build_text_index.py    # builds the text vector index
uv run python scripts/build_image_index.py   # builds the image vector index (CLIP)
```
The first run of steps 2 and 3 downloads the embedding models (~90MB for
MiniLM, ~350MB for CLIP) — needs internet access, only happens once, then
they're cached locally. Each script prints sanity-check output (row
counts, sample queries) — check that these look correct before moving on.

### 6. Run the app (two processes)
In one terminal:
```bash
uv run uvicorn backend.main:app --reload --port 8000
```
Confirm it's up at http://localhost:8000/docs (FastAPI's interactive
API docs — useful for testing `/api/query` directly without the UI).

In a second terminal:
```bash
uv run streamlit run frontend/streamlit_app.py
```
Opens automatically at http://localhost:8501. The sidebar shows whether
it successfully connected to the backend and which LLM provider is active.

## Re-running after changes
- **Edited a backend/frontend `.py` file?** `uvicorn --reload` and
  Streamlit both auto-detect changes.
- **Edited `data/destinations.csv` or added/changed images?** Re-run the
  build scripts from Step 5 — they're safe to re-run anytime, they
  rebuild the database/index from scratch each time.
- **Added a dependency?** `uv add <package>` (updates `pyproject.toml`
  and `uv.lock` together — don't hand-edit `uv.lock`).

## Notes
- `db/` and `vector_store/` are gitignored since they're large, generated,
  and machine-specific. Every teammate runs Step 5 locally after cloning.
- `uv.lock` **is** committed — that's what makes `uv sync` reproducible
  across your Kubuntu machine and your teammate's Windows machine.
- Image sources and licenses are logged in `notes/image_sources.csv` for
  acknowledgement in the report.
- Containerization (Docker) is a separate, later phase — not covered here.
