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
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import get_settings

import requests
import streamlit as st

# ── Config ────────────────────────────────────────────────────
cfg = get_settings()
API_BASE = f"http://{cfg.api_host}:{cfg.api_port}"
REDIS_URL = cfg.redis_url

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
            retriever = chunk.get("retriever", "dense")
            retriever_colors = {
                "dense": "#dbeafe:#1e40af",
                "bm25": "#dcfce7:#166534",
                "dense+bm25": "#f3e8ff:#6b21a8",
                "bm25+dense": "#f3e8ff:#6b21a8",
                "graph": "#fef9c3:#854d0e",
                "web_search": "#fee2e2:#991b1b",
            }
            ret_style = retriever_colors.get(retriever, "#f3f4f6:#374151")
            ret_bg, ret_fg = ret_style.split(":")

            st.markdown(f"""
<div class="source-card">
  <strong>[{i+1}] {title}</strong>
  <span class="score-badge" style="float:right">{score_pct}%</span><br/>
  <small style="color:#6b7280">{source}</small> &nbsp;
  <span style="background:{ret_bg};color:{ret_fg};border-radius:4px;padding:1px 6px;font-size:0.72rem;font-weight:600">{retriever}</span>
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

tab_ask, tab_ingest, tab_eval, tab_system = st.tabs(["🔍 Ask", "📥 Ingest", "📊 Evaluation", "🩺 System"])


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
                    timeout=300,
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
                            routing = payload.get("routing", {})
                            intent = routing.get("intent", "")
                            strategies = routing.get("strategies", [])
                            hits = routing.get("retriever_hits", {})
                            hits_str = "  ·  ".join(f"{k}: {v}" for k, v in hits.items())
                            strategy_str = " + ".join(s.upper() for s in strategies)
                            status_placeholder.caption(
                                f"🧭 **{intent}** → {strategy_str}  |  📚 {len(retrieved_chunks)} passages  |  {hits_str}"
                            )

                        elif event_type == "token":
                            answer_text += payload.get("text", "")
                            answer_placeholder.markdown(answer_text + "▌")

                        elif event_type == "sources":
                            # Replace cursor and append sources
                            answer_placeholder.markdown(answer_text)
                            sources_placeholder.markdown(payload.get("text", ""))
                            status_placeholder.empty()

                        elif event_type == "crag_update":
                            grade = payload.get("grade", "")
                            rewritten = payload.get("rewritten_query")
                            web_used = payload.get("web_search_used", False)
                            reasoning = payload.get("reasoning", "")
                            icon = {"POOR": "🔄", "ABSENT": "🌐"}.get(grade, "ℹ️")
                            msg = f"{icon} **CRAG {grade}**: {reasoning[:100]}"
                            if rewritten:
                                msg += "  \n\u21a9 Rewritten: *" + rewritten + "*"
                            if web_used:
                                msg += "  \n\U0001f310 Web search fallback used"
                            status_placeholder.info(msg)

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
# TAB 3 — EVALUATION DASHBOARD
# ─────────────────────────────────────────────────────────────
with tab_eval:
    st.subheader("RAG evaluation dashboard")
    st.caption("Metrics update automatically after each query. RAGAS scores compute in the background (~5s after response).")

    if st.button("🔄 Refresh metrics"):
        st.session_state.pop("metrics_data", None)

    if "metrics_data" not in st.session_state:
        try:
            resp = requests.get(f"{API_BASE}/metrics?limit=200&days=14", timeout=5)
            resp.raise_for_status()
            st.session_state.metrics_data = resp.json()
        except Exception as exc:
            st.session_state.metrics_data = {"error": str(exc)}

    mdata = st.session_state.get("metrics_data", {})

    if "error" in mdata:
        st.error(f"Cannot reach API: {mdata['error']}")
    else:
        summary = mdata.get("summary", {})
        cache   = mdata.get("cache", {})

        # ── Header KPI row ─────────────────────────────────────
        k1, k2, k3, k4, k5, k6 = st.columns(6)
        k1.metric("Total queries",    summary.get("total_queries", 0))
        k2.metric("Faithfulness",     f"{summary.get('avg_faithfulness', 0):.2f}")
        k3.metric("Answer relevancy", f"{summary.get('avg_answer_relevancy', 0):.2f}")
        k4.metric("Context precision",f"{summary.get('avg_context_precision', 0):.2f}")
        k5.metric("Avg latency",      f"{summary.get('avg_latency_ms', 0):.0f} ms")
        k6.metric("Cache hit rate",   f"{cache.get('hit_rate', 0):.0%}" if cache.get('enabled') else "off")

        st.divider()

        # ── Metric timeseries ──────────────────────────────────
        ts = mdata.get("timeseries", [])
        if ts:
            import pandas as pd
            df_ts = pd.DataFrame(ts)
            df_ts["hour"] = df_ts["hour_bucket"]
            st.markdown("#### RAGAS metrics over time")
            st.line_chart(
                df_ts.set_index("hour")[["faithfulness", "answer_relevancy", "context_precision"]],
                height=220,
            )
        else:
            st.info("No evaluation data yet. Run some queries to populate the dashboard.")

        st.divider()

        col_left, col_right = st.columns(2, gap="large")

        with col_left:
            # ── CRAG grade distribution ────────────────────────
            grade_dist = summary.get("crag_grade_dist", {})
            if grade_dist:
                import pandas as pd
                st.markdown("#### CRAG grade distribution")
                df_grades = pd.DataFrame(
                    list(grade_dist.items()), columns=["Grade", "Count"]
                )
                st.bar_chart(df_grades.set_index("Grade"), height=180)

            # ── Strategy distribution ──────────────────────────
            strat_dist = summary.get("strategy_dist", {})
            if strat_dist:
                import pandas as pd
                st.markdown("#### Retrieval strategy mix")
                rows = []
                for strat_json, cnt in strat_dist.items():
                    try:
                        import json as _json
                        label = "+".join(_json.loads(strat_json)).upper()
                    except Exception:
                        label = strat_json
                    rows.append({"Strategy": label, "Count": cnt})
                df_strat = pd.DataFrame(rows)
                st.bar_chart(df_strat.set_index("Strategy"), height=180)

        with col_right:
            # ── Cache stats ────────────────────────────────────
            st.markdown("#### Cache")
            if cache.get("enabled"):
                c1, c2 = st.columns(2)
                c1.metric("Hits",   cache.get("hits", 0))
                c2.metric("Misses", cache.get("misses", 0))
                st.caption(f"TTL: {cache.get('ttl_s', 0)//60} min")
                if st.button("🗑️ Flush cache"):
                    try:
                        r = requests.post(f"{REDIS_URL}/cache/flush", timeout=5)
                        st.success(f"Flushed {r.json().get('deleted', 0)} entries.")
                        st.session_state.pop("metrics_data", None)
                    except Exception as e:
                        st.error(str(e))
            else:
                st.caption("Redis not connected. Start Redis to enable caching.")
                st.code("docker run -d -p 6379:6379 redis:7-alpine", language="bash")

        st.divider()

        # ── Recent query log table ─────────────────────────────
        recent = mdata.get("recent", [])
        if recent:
            import pandas as pd
            st.markdown("#### Recent queries")
            rows = []
            for r in recent[:50]:
                rows.append({
                    "Query":       r.get("query", "")[:60],
                    "Intent":      r.get("intent", ""),
                    "CRAG":        r.get("crag_grade", ""),
                    "Faithful":    f"{r['faithfulness']:.2f}"      if r.get("faithfulness")      else "—",
                    "Relevancy":   f"{r['answer_relevancy']:.2f}"  if r.get("answer_relevancy")  else "—",
                    "Precision":   f"{r['context_precision']:.2f}" if r.get("context_precision") else "—",
                    "Latency ms":  f"{r.get('latency_ms', 0):.0f}",
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────────────────────
# TAB 4 — SYSTEM HEALTH
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
            graph_stats = health.get("graph_stats", {})
            if graph_stats:
                col_d, col_e = st.columns(2)
                with col_d:
                    st.metric("Graph nodes", graph_stats.get("nodes", "—"))
                with col_e:
                    st.metric("Graph edges", graph_stats.get("edges", "—"))
            st.divider()
            st.markdown("**Raw health response**")
            st.json(health)
