import sys
import os

# Add project root to path so src/ modules are importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import streamlit as st
from src.rag_chain import ask_query

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MedMind RAG",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🩺 MedMind RAG")
    st.markdown(
        """
        **Medical Question Answering**
        Powered by [MedQuAD](https://github.com/abachaa/MedQuAD) + FAISS + GPT-4o

        ---
        **How it works**
        1. Your question is embedded using `all-MiniLM-L12-v2`
        2. Top-5 relevant passages are retrieved from the FAISS index
        3. GPT-4o synthesises an answer grounded in those passages

        ---
        **Disclaimer**
        This tool is for *educational purposes only* and is **not** a substitute for professional medical advice.
        """
    )

    st.divider()
    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    show_sources = st.toggle("Show retrieved sources", value=True)

# ── Session state ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("## 🩺 MedMind — Medical RAG Assistant")
st.caption("Ask any medical question and get evidence-grounded answers from the MedQuAD dataset.")

# ── Render chat history ───────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and show_sources and msg.get("sources"):
            with st.expander(f"📄 Retrieved sources ({len(msg['sources'])})"):
                for i, src in enumerate(msg["sources"], 1):
                    similarity = round((1 - src["distance"]) * 100, 1)
                    st.markdown(f"**Source {i}** — similarity: `{similarity}%`")
                    st.markdown(f"> {src['text'][:400]}{'…' if len(src['text']) > 400 else ''}")
                    meta = src.get("metadata", {})
                    if meta.get("question"):
                        st.caption(f"Related question: {meta['question']}")
                    st.divider()

# ── Chat input ────────────────────────────────────────────────────────────────
if prompt := st.chat_input("Ask a medical question…"):
    # Render user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Query the RAG chain
    with st.chat_message("assistant"):
        with st.spinner("Searching knowledge base…"):
            try:
                result = ask_query(prompt)
                answer = result["answer"]
                sources = result.get("source", [])

                st.markdown(answer)

                if show_sources and sources:
                    with st.expander(f"📄 Retrieved sources ({len(sources)})"):
                        for i, src in enumerate(sources, 1):
                            similarity = round((1 - src["distance"]) * 100, 1)
                            st.markdown(f"**Source {i}** — similarity: `{similarity}%`")
                            st.markdown(f"> {src['text'][:400]}{'…' if len(src['text']) > 400 else ''}")
                            meta = src.get("metadata", {})
                            if meta.get("question"):
                                st.caption(f"Related question: {meta['question']}")
                            st.divider()

            except Exception as e:
                answer = f"⚠️ Error: {e}"
                sources = []
                st.error(answer)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )
