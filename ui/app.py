"""
Cortex RAG — Streamlit UI (Phase 1)

Tabs:
  🔍 Ask      — streaming Q&A with inline citations and source cards
  📥 Ingest   — upload documents or provide a directory path
  🩺 System   — health check and collection statistics
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

import requests
import streamlit as st

# ── Config ────────────────────────────────────────────────────
API_BASE = "http://localhost:8000"

st.set_page_config(
    page_title="Cortex RAG",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Styles ────────────────────────────────────────────────────
st.markdown("""
<style>
    .cortex-title  { font-size:2.2rem; font-weight:700; margin-bottom:0; }
    .cortex-sub    { color:#6b7280; font-size:1rem; margin-top:0; }
    .source-card   { background:#f8fafc; border:1px solid #e2e8f0;
                     border-radius:8px; padding:12px 16px; margin-bottom:8px; }
    .score-badge   { background:#dbeafe; color:#1e40af; border-radius:4px;
                     padding:2px 8px; font-size:0.78rem; font-weight:600; }
    .chunk-snippet { font-size:0.85rem; color:#4b5563;
                     border-left:3px solid #93c5fd; padding-left:10px;
                     margin-top:6px; }
    .metric-row    { display:flex; gap:16px; margin-bottom:12px; }
    div[data-testid="stSpinner"] { margin-top: 0 !important; }
</style>
""", unsafe_allow_html=True)


# ── Session state defaults ────────────────────────────────────
def _init_state():
    defaults = {
        "messages":     [],     # list of {role, content, chunks}
        "ingest_log":   [],
        "health":       None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()

def _render_source_cards_raw(chunks: list[dict]):
    st.caption("**Retrieved passages**")
    cols = st.columns(min(len(chunks), 3))
    for i, chunk in enumerate(chunks):
        with cols[i % len(cols)]:
            score_pct = int(chunk.get("score", 0) * 100)
            title = chunk.get("title", "Unknown")
            source = Path(chunk.get("source", "")).name
            snippet = chunk.get("text_snippet", "")[:160]
            st.markdown(f"""
<div class="source-card">
  <strong>[{i+1}] {title}</strong>
  <span class="score-badge" style="float:right">{score_pct}%</span><br/>
  <small style="color:#6b7280">{source}</small>
  <div class="chunk-snippet">{snippet}…</div>
</div>""", unsafe_allow_html=True)


def _render_source_cards(chunks: list[dict]):
    """Replay version — same cards but from stored history."""
    _render_source_cards_raw(chunks)


# ── Sidebar ────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🧠 Cortex RAG")
    st.caption("Phase 1 · Dense retrieval · Groq/Llama 3.3-70B")
    st.divider()
    top_k = st.slider("Retrieve top-k chunks", 3, 20, 10)
    st.divider()
    if st.button("🗑️ Clear conversation"):
        st.session_state.messages = []
        st.rerun()
    st.caption(f"API: `{API_BASE}`")


# ── Header ─────────────────────────────────────────────────────
st.markdown('<p class="cortex-title">Cortex RAG</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="cortex-sub">Production-grade RAG · Phase 1: Dense retrieval + streaming generation</p>',
    unsafe_allow_html=True
)
st.divider()

tab_ask, tab_ingest, tab_system = st.tabs(["🔍 Ask", "📥 Ingest", "🩺 System"])


# ─────────────────────────────────────────────────────────────
# TAB 1 — ASK
# ─────────────────────────────────────────────────────────────
with tab_ask:
    # Replay conversation history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("chunks"):
                _render_source_cards(msg["chunks"])

    query = st.chat_input("Ask anything about your documents…")

    if query:
        # Append and display user message
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        # Fetch streamed response
        with st.chat_message("assistant"):
            answer_placeholder = st.empty()
            sources_placeholder = st.empty()
            status_placeholder  = st.empty()

            answer_text = ""
            retrieved_chunks = []

            try:
                with requests.post(
                    f"{API_BASE}/query/stream",
                    json={"query": query, "top_k": top_k, "stream": True},
                    stream=True,
                    timeout=60,
                ) as resp:
                    resp.raise_for_status()

                    for raw_line in resp.iter_lines():
                        if not raw_line:
                            continue
                        line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
                        if not line.startswith("data: "):
                            continue
                        payload = json.loads(line[6:])

                        event_type = payload.get("type")

                        if event_type == "chunk_meta":
                            retrieved_chunks = payload.get("chunks", [])
                            status_placeholder.caption(
                                f"📚 Retrieved {len(retrieved_chunks)} relevant passages"
                            )

                        elif event_type == "token":
                            answer_text += payload.get("text", "")
                            answer_placeholder.markdown(answer_text + "▌")

                        elif event_type == "sources":
                            # Replace cursor and append sources
                            answer_placeholder.markdown(answer_text)
                            sources_placeholder.markdown(payload.get("text", ""))
                            status_placeholder.empty()

                        elif event_type == "done":
                            answer_placeholder.markdown(answer_text)
                            status_placeholder.empty()
                            break

                        elif event_type == "error":
                            st.error(f"API error: {payload.get('message')}")
                            break

            except requests.exceptions.ConnectionError:
                st.error(
                    "⚠️ Cannot reach the Cortex API. "
                    "Make sure `uvicorn api.main:app` is running on port 8000."
                )
                answer_text = "_Connection error — see above._"
            except Exception as exc:
                st.error(f"Unexpected error: {exc}")
                answer_text = "_Error — see above._"

            # Render source cards inline
            if retrieved_chunks:
                _render_source_cards_raw(retrieved_chunks)

        # Save to conversation history
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer_text,
            "chunks": retrieved_chunks,
        })


# ─────────────────────────────────────────────────────────────
# TAB 2 — INGEST
# ─────────────────────────────────────────────────────────────
with tab_ingest:
    st.subheader("Ingest documents into the knowledge base")
    st.caption(
        "Supported formats: **PDF**, **HTML**, **TXT**, **Markdown**. "
        "Files are deduplicated automatically."
    )

    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        st.markdown("#### Option A — Provide a server path")
        ingest_path = st.text_input(
            "Path on server",
            placeholder="data/documents  or  /abs/path/to/file.pdf",
            help="Relative or absolute path accessible by the API process.",
        )
        recursive = st.checkbox("Recursive (include subdirectories)", value=True)

        if st.button("🚀 Start ingestion", type="primary", disabled=not ingest_path):
            with st.spinner("Ingesting…"):
                try:
                    resp = requests.post(
                        f"{API_BASE}/ingest",
                        json={"path": ingest_path, "recursive": recursive},
                        timeout=300,
                    )
                    resp.raise_for_status()
                    result = resp.json()
                    st.success(
                        f"✅ {result['documents_processed']} documents processed, "
                        f"{result['chunks_stored']} chunks stored."
                    )
                    if result.get("documents_skipped"):
                        st.info(f"ℹ️ {result['documents_skipped']} documents already existed — skipped.")
                    if result.get("errors"):
                        st.warning(f"⚠️ {len(result['errors'])} errors:")
                        for err in result["errors"]:
                            st.code(json.dumps(err, indent=2))
                    st.session_state.ingest_log.append(result)
                except requests.exceptions.ConnectionError:
                    st.error("Cannot reach the API. Is uvicorn running?")
                except Exception as exc:
                    st.error(f"Ingestion failed: {exc}")

    with col_right:
        st.markdown("#### Ingestion log")
        if st.session_state.ingest_log:
            for i, entry in enumerate(reversed(st.session_state.ingest_log[-5:])):
                with st.expander(f"Run {len(st.session_state.ingest_log) - i}", expanded=(i==0)):
                    st.json(entry)
        else:
            st.caption("No ingestion runs yet.")


# ─────────────────────────────────────────────────────────────
# TAB 3 — SYSTEM HEALTH
# ─────────────────────────────────────────────────────────────
with tab_system:
    st.subheader("System health")

    if st.button("🔄 Refresh health"):
        st.session_state.health = None

    if st.session_state.health is None:
        try:
            resp = requests.get(f"{API_BASE}/health", timeout=5)
            resp.raise_for_status()
            st.session_state.health = resp.json()
        except Exception as exc:
            st.session_state.health = {"error": str(exc)}

    health = st.session_state.health
    if health:
        if "error" in health:
            st.error(f"Cannot reach API: {health['error']}")
        else:
            status = health.get("status", "unknown")
            icon = "✅" if status == "ok" else "⚠️"
            st.markdown(f"**Overall status**: {icon} `{status}`")

            col_a, col_b, col_c = st.columns(3)
            with col_a:
                milvus = health.get("milvus", "unknown")
                st.metric("Milvus", "✅ ok" if milvus == "ok" else f"❌ {milvus}")
            with col_b:
                embedder = health.get("embedder", "unknown")
                st.metric("Embedder", "✅ loaded" if embedder == "loaded" else "⏳ not loaded")
            with col_c:
                stats = health.get("collection_stats", {})
                st.metric("Chunks indexed", stats.get("entity_count", "—"))

            st.divider()
            st.markdown("**Raw health response**")
            st.json(health)
