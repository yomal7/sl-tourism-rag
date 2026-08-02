import os

import requests
import streamlit as st
from PIL import Image

# Same machine by default (both processes run locally for now). Overridable
# via an environment variable so this also works once backend/frontend move
# into separate Docker containers in the next phase.
BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000")

st.set_page_config(page_title="Sri Lanka Tourism RAG", page_icon="🏝️", layout="wide")

st.title("🏝️ Sri Lanka Tourism — Multimodal RAG Assistant")
st.caption(
    "Ask about beaches, temples, and historical sites in Sri Lanka. "
    "Answers are generated from a relational database + vector database, "
    "not from the LLM's general knowledge."
)

with st.sidebar:
    st.header("About this system")
    st.markdown(
        """
This assistant retrieves information from **three sources** before
generating an answer:

- 🗄️ **Relational database (SQLite)** — facts like entrance fees, hours,
  accessibility
- 📖 **Text vector database (ChromaDB)** — semantic search over
  descriptions
- 🖼️ **Image vector database (ChromaDB + CLIP)** — visual similarity search

A rule-based **router** decides which source(s) each query needs, then
an **LLM** (Gemini API or a local Ollama model, set in `.env`) generates
the final answer using only the retrieved context.
        """
    )

    try:
        health = requests.get(f"{BACKEND_URL}/api/health", timeout=3).json()
        st.success(f"Backend connected — LLM provider: **{health['llm_provider']}**")
    except requests.exceptions.RequestException:
        st.error(
            f"Can't reach the backend at {BACKEND_URL}. "
            "Start it with: `uv run uvicorn backend.main:app --reload --port 8000`"
        )

    st.divider()
    st.subheader("Try these example queries")
    example_queries = [
        "What is the entrance fee for Sigiriya and when is it open?",
        "Suggest a peaceful place for meditation",
        "Which beaches are free to enter?",
        "Tell me about a good beach for surfing with a lively atmosphere",
        "What ancient ruins can I visit near Anuradhapura?",
    ]
    for eq in example_queries:
        if st.button(eq, use_container_width=True):
            st.session_state["query_input"] = eq

# Query input
query = st.text_input(
    "Ask a question about Sri Lankan tourist destinations:",
    key="query_input",
    placeholder="e.g. Suggest a peaceful place for meditation",
)

# Image upload
uploaded_image = st.file_uploader(
    "Optional: upload a photo to find visually similar destinations",
    type=["jpg", "jpeg", "png"],
)

col1, col2 = st.columns([1, 5])
with col1:
    submitted = st.button("Ask", type="primary")

if uploaded_image is not None:
    st.image(uploaded_image, caption="Uploaded image", width=250)

if submitted:
    if not query and uploaded_image is None:
        st.warning("Please enter a question or upload an image.")
    else:
        effective_query = query or "Find destinations that look like this image."

        with st.spinner("Retrieving context and generating answer..."):
            data = {"query": effective_query}
            files = None
            if uploaded_image is not None:
                files = {
                    "image": (
                        uploaded_image.name,
                        uploaded_image.getvalue(),
                        uploaded_image.type,
                    )
                }

            result = None
            try:
                resp = requests.post(
                    f"{BACKEND_URL}/api/query", data=data, files=files, timeout=120
                )
                resp.raise_for_status()
                result = resp.json()
            except requests.exceptions.RequestException as e:
                st.error(f"Couldn't reach the backend: {e}")

        if result:
            st.subheader("Answer")
            st.write(result["answer"])

            with st.expander("🔍 How this answer was retrieved (routing + raw context)"):
                st.markdown(f"**Routing decision:** `{result['route']}`")

                if result["sql_results"]:
                    st.markdown("**Structured facts (SQL):**")
                    st.table(
                        [
                            {
                                "Name": r["name"],
                                "Category": r["category"],
                                "Fee (LKR)": r["entrance_fee_lkr"],
                                "Hours": r["opening_hours"],
                                "Location": r["location"],
                            }
                            for r in result["sql_results"]
                        ]
                    )

                if result["text_results"]:
                    st.markdown("**Semantic text matches:**")
                    for r in result["text_results"]:
                        st.markdown(f"- **{r['name']}** (distance={r['distance']:.4f}): {r['passage']}")

                if result["image_results"]:
                    st.markdown("**Image matches:**")
                    for r in result["image_results"]:
                        if r.get("distance") is not None:
                            st.markdown(f"- **{r['name']}** (distance={r['distance']:.4f})")
                        else:
                            st.markdown(f"- **{r['name']}** (matched via {r.get('matched_via', 'retrieval')})")

            if result["image_paths_to_display"]:
                st.subheader("Related images")
                images_to_show = result["image_paths_to_display"]
                cols = st.columns(min(len(images_to_show), 4))
                for i, item in enumerate(images_to_show):
                    try:
                        img = Image.open(item["filepath"])
                        with cols[i % len(cols)]:
                            st.image(img, caption=item["name"], use_container_width=True)
                    except Exception:
                        pass