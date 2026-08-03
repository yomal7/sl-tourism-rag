import os
from pathlib import Path

import requests
import streamlit as st
from PIL import Image

APP_NAME = "Serendib"
APP_SUBTITLE = "Multimodal RAG Assistant"
APP_TAGLINE = "Your guide to Sri Lanka's beaches, temples and hidden spots."

BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000")

EXAMPLE_QUERIES = [
    {"label": "Sigiriya fees & hours", "query": "What is the entrance fee for Sigiriya and when is it open?"},
    {"label": "A peaceful spot", "query": "Suggest a peaceful place for meditation"},
    {"label": "Free beaches", "query": "Which beaches are free to enter?"},
    {"label": "Lively surf beach", "query": "Tell me about a good beach for surfing with a lively atmosphere"},
    {"label": "Anuradhapura", "query": "What ancient ruins can I visit near Anuradhapura?"},
]

st.set_page_config(page_title=f"{APP_NAME} — {APP_SUBTITLE}", page_icon="🧭", layout="wide")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Fraunces:wght@600;700&family=Inter:wght@400;500;600&display=swap');

    html, body, [class*="css"]  { font-family: 'Inter', sans-serif; }

    /* Pull the whole page up a bit — default top padding reads as dead space */
    .block-container { padding-top: 2.5rem; max-width: 1100px; }

    /* Bump base body copy site-wide. Targeting real HTML tags (p, li, label,
       input, textarea) rather than Streamlit's internal CSS class names,
       since those class names churn between Streamlit versions and tag
       selectors don't. */
    p, li, label, .stMarkdown { font-size: 1.05rem !important; line-height: 1.65; }
    small { font-size: 0.95rem !important; }

    input, textarea { font-size: 1.05rem !important; }
    div.stButton > button { font-size: 0.95rem !important; padding: 0.55rem 1.1rem !important; }
    div.stButton > button p { font-size: 0.95rem !important; }

    h1, h2, h3 { font-family: 'Fraunces', serif !important; color: #0B4F4A; }
    h2 { font-size: 1.9rem !important; }
    h3 { font-size: 1.35rem !important; }

    .app-title {
        font-family: 'Fraunces', serif;
        font-size: 2.8rem;
        font-weight: 700;
        color: #0B4F4A;
        letter-spacing: -0.5px;
        line-height: 1.15;
    }
    .app-subtitle {
        font-family: 'Inter', sans-serif;
        font-size: 1.3rem;
        font-weight: 500;
        color: #4F7873;
    }
    .app-tagline {
        color: #55605E;
        font-size: 1.1rem !important;
        margin: 0.4rem 0 0.6rem 0;
    }
    .status-pill {
        display: inline-block;
        padding: 0.35rem 0.9rem;
        border-radius: 999px;
        font-size: 0.85rem !important;
        font-weight: 600;
        float: right;
        margin-top: 0.9rem;
    }
    .status-ok   { background: #E4F3EE; color: #0B4F4A; }
    .status-bad  { background: #FBEAEA; color: #9B2C2C; }

    div.stButton > button {
        border-radius: 8px;
        border: 1px solid #D8DDDB;
    }
    div.stButton > button[kind="primary"] {
        background-color: #0B4F4A;
        border: none;
    }
    div.stButton > button[kind="primary"]:hover {
        background-color: #0E6259;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

header_left, header_right = st.columns([5, 1])

with header_left:
    st.markdown(
        f'<div class="app-title">{APP_NAME} <span class="app-subtitle">— {APP_SUBTITLE}</span></div>',
        unsafe_allow_html=True,
    )
    st.markdown(f'<div class="app-tagline">{APP_TAGLINE}</div>', unsafe_allow_html=True)

with header_right:
    try:
        health = requests.get(f"{BACKEND_URL}/api/health", timeout=3).json()
        st.markdown(
            f'<div class="status-pill status-ok">● {health["llm_provider"]}</div>',
            unsafe_allow_html=True,
        )
        backend_ok = True
    except requests.exceptions.RequestException:
        st.markdown('<div class="status-pill status-bad">● backend offline</div>', unsafe_allow_html=True)
        backend_ok = False

st.caption("Answers are pulled from a real destinations database, not just what the model already knows.")

if not backend_ok:
    st.error(
        f"Can't reach the backend at {BACKEND_URL}. "
        "Start it with: `uv run uvicorn backend.main:app --reload --port 8000`"
    )

st.write("")

st.caption("Try one of these, or type your own question below:")
chip_cols = st.columns(len(EXAMPLE_QUERIES))
for col, example in zip(chip_cols, EXAMPLE_QUERIES):
    with col:
        if st.button(example["label"], width="stretch", key=f"chip_{example['label']}"):
            st.session_state["query_input"] = example["query"]

# Query input and optional image upload
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
    submitted = st.button("Ask", type="primary", width="stretch")

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
                if resp.status_code == 200:
                    result = resp.json()
                else:
                    try:
                        detail = resp.json().get("detail", "Something went wrong generating the answer.")
                    except ValueError:
                        detail = "Something went wrong generating the answer."
                    if resp.status_code == 429:
                        st.warning(f"⏳ {detail}")
                    else:
                        st.error(f"⚠️ {detail}")
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
                            st.image(img, caption=item["name"], width="stretch")
                    except Exception:
                        pass
