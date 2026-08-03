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
├── backend/               
│   ├── main.py                  
│   ├── config.py 
|   ├── constants.py                
│   ├── llm_client.py            
│   ├── router.py                
│   ├── retrieve.py              
│   └── generate_response.py     
├── frontend/
│   └── streamlit_app.py     
├── scripts/             
│   ├── setup_db.py     
│   ├── build_text_index.py 
│   └── build_image_index.py 
├── data/
│   ├── destinations.csv         
│   └── images/                  
├── db/                          
├── vector_store/                
├── notes/                       
├── pyproject.toml               
├── .env.example                 
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
  ollama list
  ```

Each teammate's `.env` can differ — the code reads `LLM_PROVIDER` at
runtime and picks the right path, so this is not something you need to
agree on as a team or hardcode.

### 5. Build the databases
Run these **in order** — each one depends on the previous step:
```bash
uv run python scripts/setup_db.py
uv run python scripts/build_text_index.py
uv run python scripts/build_image_index.py
```
The first run of steps 2 and 3 downloads the embedding models (~90MB for
MiniLM, ~350MB for CLIP) — needs internet access, only happens once, then
they're cached locally. Each script prints sanity-check output (row
counts, sample queries).

### 6. Run the app (two processes)
In one terminal:
```bash
uv run uvicorn backend.main:app --reload --port 8000
```

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

