"""Streamlit chatbot UI for Samsung TV Manual RAG."""

from __future__ import annotations

import sys
from pathlib import Path

import requests
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import get_settings

st.set_page_config(page_title="Samsung TV Manual Chatbot", page_icon="📺", layout="centered")
st.title("Samsung TV Manual Assistant")
st.caption("RAG chatbot adapted from RMIT WALert — answers grounded in official manuals.")

settings = get_settings()
default_model = settings.default_tv_model
api_base = st.sidebar.text_input("API base URL", value=f"http://127.0.0.1:{settings.api_port}")
use_local = st.sidebar.checkbox("Call local Python pipeline (no API)", value=True)
model = st.sidebar.text_input("TV model", value=default_model)
st.sidebar.markdown(
    f"[Official support page]({settings.samsung_manual_source})"
)

if "messages" not in st.session_state:
    st.session_state.messages = []


def ask(question: str, tv_model: str | None) -> dict:
    if use_local:
        from rag.generator import EchoLLM, OllamaClient
        from rag.pipeline import RagPipeline

        settings_local = get_settings()
        collection = settings_local.resolve(settings_local.collection_path)
        if not collection.exists():
            raise FileNotFoundError(
                f"Missing {collection}. Run: python scripts/ingest_manuals.py --hash-embeddings"
            )

        index_dir = settings_local.resolve(settings_local.vector_database_path)
        backend = settings_local.retriever_backend
        use_hash = False
        try:
            import sentence_transformers  # noqa: F401
        except ImportError:
            use_hash = True
            if backend == "dense":
                backend = "bm25"
        if backend == "dense" and not (index_dir / "index.faiss").exists():
            backend = "bm25"

        try:
            llm = OllamaClient(settings_local.ollama_base_url, settings_local.llm_model)
            # Probe Ollama quickly
            requests.get(f"{settings_local.ollama_base_url}/api/tags", timeout=2)
        except Exception:
            llm = EchoLLM()
            st.warning("Ollama not reachable — using EchoLLM stub for demo.")
        pipeline = RagPipeline(
            settings=settings_local,
            llm=llm,
            backend=backend,  # type: ignore[arg-type]
            use_hash_embeddings=use_hash,
        )
        return pipeline.ask(question, model=tv_model or None).to_dict()

    response = requests.post(
        f"{api_base.rstrip('/')}/api/chat",
        json={"question": question, "model": tv_model or None},
        timeout=120,
    )
    response.raise_for_status()
    return response.json()


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            st.markdown("**Sources**")
            for src in message["sources"]:
                page = src.get("page")
                page_txt = f"p. {page}" if page is not None else "page n/a"
                st.markdown(
                    f"- {src.get('manual', 'Manual')} — {src.get('section', '')} — {page_txt}"
                )

prompt = st.chat_input("Ask about your Samsung TV…")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching the manual…"):
            try:
                result = ask(prompt, model.strip() or None)
                answer = result.get("answer", "")
                sources = result.get("sources", [])
                st.markdown(answer)
                if sources:
                    st.markdown("**Sources**")
                    for src in sources:
                        page = src.get("page")
                        page_txt = f"p. {page}" if page is not None else "page n/a"
                        st.markdown(
                            f"- {src.get('manual', 'Manual')} ({src.get('model', '')}) — "
                            f"{src.get('section', '')} — {page_txt}"
                        )
                st.session_state.messages.append(
                    {"role": "assistant", "content": answer, "sources": sources}
                )
            except Exception as exc:  # noqa: BLE001
                st.error(f"Error: {exc}")
