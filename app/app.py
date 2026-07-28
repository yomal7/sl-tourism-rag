"""
app.py
------
Phase 7: Web interface for the Sri Lanka Tourism Multimodal RAG system.

Lets you:
  - Type a natural language question (structured, semantic, or hybrid)
  - Optionally upload an image to search for visually similar destinations
  - See the generated answer, which retrieval route was used, and the
    underlying retrieved facts/text/images (useful for your demo, since it
    shows the examiner exactly how retrieval worked, not just the final text)

Run from the project root:
    streamlit run app/app.py
"""

import sys
import tempfile
from pathlib import Path

import streamlit as st
from PIL import Image

# Make scripts/ importable from here
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from generate_response import answer_query  # noqa: E402

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
a locally-running **LLM (Ollama)** generates the final answer using only
the retrieved context.
        """
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

# ---- Query input ----
query = st.text_input(
    "Ask a question about Sri Lankan tourist destinations:",
    key="query_input",
    placeholder="e.g. Suggest a peaceful place for meditation",
)

uploaded_image = st.file_uploader(
    "Optional: upload a photo to find visually similar destinations",
    type=["jpg", "jpeg", "png"],
)

col1, col2 = st.columns([1, 5])
with col1:
    submitted = st.button("Ask", type="primary")

if uploaded_image is not None:
    st.image(uploaded_image, caption="Uploaded image", width=250)

# ---- Handle submission ----
if submitted:
    if not query and uploaded_image is None:
        st.warning("Please enter a question or upload an image.")
    else:
        with st.spinner("Retrieving context and generating answer..."):
            image_path = None
            if uploaded_image is not None:
                # Save the uploaded image to a temp file so query_image_by_image
                # (which expects a filesystem path) can read it
                suffix = Path(uploaded_image.name).suffix
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
                tmp.write(uploaded_image.getvalue())
                tmp.close()
                image_path = tmp.name

            effective_query = query or "Find destinations that look like this image."

            try:
                result = answer_query(effective_query, uploaded_image_path=image_path)
            except Exception as e:
                st.error(f"Something went wrong: {e}")
                result = None

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
                        st.markdown(f"- **{r['name']}** (distance={r['distance']:.4f})")

            # Show retrieved images, if any, as actual thumbnails
            if result["image_paths_to_display"]:
                st.subheader("Related images")
                cols = st.columns(min(len(result["image_paths_to_display"]), 4))
                for i, path in enumerate(result["image_paths_to_display"]):
                    try:
                        img = Image.open(path)
                        with cols[i % len(cols)]:
                            st.image(img, caption=Path(path).stem, use_container_width=True)
                    except Exception:
                        pass  # skip images that fail to load rather than breaking the page
