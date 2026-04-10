"""
Cortex RAG — Evaluation Store (SQLite)

Two tables:
  query_logs   — one row per query: routing, CRAG grade, latency, chunk scores
  eval_metrics — one row per query: RAGAS scores (written async after generation)

SQLite is the right choice here: zero infrastructure, works on Railway/Render
out of the box, and a dashboard corpus of ~10k queries fits in <50MB.
Swap to Postgres trivially later by changing the connection string.

The store is intentionally append-only. No deletes, no updates.
This preserves the full history for trend analysis in the dashboard.
"""
from __future__ import annotations

import json
import logging
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_DEFAULT_DB_PATH = Path("data/cortex_eval.db")

# ── Schema ─────────────────────────────────────────────────────

_DDL = """
CREATE TABLE IF NOT EXISTS query_logs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp       REAL    NOT NULL,
    query           TEXT    NOT NULL,
    intent          TEXT,
    strategies      TEXT,           -- JSON list
    retriever_hits  TEXT,           -- JSON dict
    crag_grade      TEXT,
    crag_rewritten  INTEGER DEFAULT 0,   -- bool
    web_search_used INTEGER DEFAULT 0,   -- bool
    num_chunks      INTEGER DEFAULT 0,
    top_chunk_score REAL    DEFAULT 0.0,
    latency_ms      REAL    DEFAULT 0.0,
    model           TEXT,
    extractor       TEXT
);

CREATE TABLE IF NOT EXISTS eval_metrics (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    query_log_id        INTEGER NOT NULL REFERENCES query_logs(id),
    timestamp           REAL    NOT NULL,
    faithfulness        REAL,   -- 0-1: does answer contradict context?
    answer_relevancy    REAL,   -- 0-1: does answer address the question?
    context_precision   REAL,   -- 0-1: are retrieved chunks relevant?
    context_utilisation REAL,   -- 0-1: fraction of chunks cited in answer
    mean_chunk_score    REAL    -- average retrieval score of final chunks
);

CREATE INDEX IF NOT EXISTS idx_query_logs_ts   ON query_logs(timestamp);
CREATE INDEX IF NOT EXISTS idx_eval_metrics_id ON eval_metrics(query_log_id);
"""


# ── Dataclasses ────────────────────────────────────────────────

@dataclass
class QueryLogEntry:
    query: str
    intent: str = ""
    strategies: list[str] = None
    retriever_hits: dict = None
    crag_grade: str = ""
    crag_rewritten: bool = False
    web_search_used: bool = False
    num_chunks: int = 0
    top_chunk_score: float = 0.0
    latency_ms: float = 0.0
    model: str = ""
    extractor: str = ""

    def __post_init__(self):
        if self.strategies is None:
            self.strategies = []
        if self.retriever_hits is None:
            self.retriever_hits = {}


@dataclass
class EvalMetricEntry:
    query_log_id: int
    faithfulness: Optional[float] = None
    answer_relevancy: Optional[float] = None
    context_precision: Optional[float] = None
    context_utilisation: Optional[float] = None
    mean_chunk_score: Optional[float] = None


# ── Store ──────────────────────────────────────────────────────

class EvalStore:
    """
    Thread-safe SQLite-backed store for query logs and eval metrics.

    Usage:
        store = EvalStore()
        log_id = store.log_query(entry)
        store.log_metrics(EvalMetricEntry(query_log_id=log_id, faithfulness=0.92, ...))
    """

    def __init__(self, db_path: str | Path = _DEFAULT_DB_PATH) -> None:
        self._path = Path(db_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    # ── Write ──────────────────────────────────────────────────

    def log_query(self, entry: QueryLogEntry) -> int:
        """Insert a query log row. Returns the new row id."""
        with self._conn() as conn:
            cur = conn.execute(
                """INSERT INTO query_logs
                   (timestamp, query, intent, strategies, retriever_hits,
                    crag_grade, crag_rewritten, web_search_used,
                    num_chunks, top_chunk_score, latency_ms, model, extractor)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    time.time(),
                    entry.query,
                    entry.intent,
                    json.dumps(entry.strategies),
                    json.dumps(entry.retriever_hits),
                    entry.crag_grade,
                    int(entry.crag_rewritten),
                    int(entry.web_search_used),
                    entry.num_chunks,
                    entry.top_chunk_score,
                    entry.latency_ms,
                    entry.model,
                    entry.extractor,
                ),
            )
            return cur.lastrowid

    def log_metrics(self, entry: EvalMetricEntry) -> None:
        """Insert an eval_metrics row."""
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO eval_metrics
                   (query_log_id, timestamp, faithfulness, answer_relevancy,
                    context_precision, context_utilisation, mean_chunk_score)
                   VALUES (?,?,?,?,?,?,?)""",
                (
                    entry.query_log_id,
                    time.time(),
                    entry.faithfulness,
                    entry.answer_relevancy,
                    entry.context_precision,
                    entry.context_utilisation,
                    entry.mean_chunk_score,
                ),
            )

    # ── Read ───────────────────────────────────────────────────

    def get_recent_queries(self, limit: int = 100) -> list[dict]:
        """Last N query logs joined with their eval metrics (if available)."""
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT q.id, q.timestamp, q.query, q.intent, q.strategies,
                          q.crag_grade, q.web_search_used, q.num_chunks,
                          q.top_chunk_score, q.latency_ms,
                          e.faithfulness, e.answer_relevancy,
                          e.context_precision, e.context_utilisation,
                          e.mean_chunk_score
                   FROM query_logs q
                   LEFT JOIN eval_metrics e ON e.query_log_id = q.id
                   ORDER BY q.timestamp DESC
                   LIMIT ?""",
                (limit,),
            ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def get_metric_timeseries(self, days: int = 7) -> list[dict]:
        """
        Hourly-bucketed metric averages over the last N days.
        Used for the trend line chart in the dashboard.
        """
        since = time.time() - days * 86400
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT
                       CAST((q.timestamp - ?) / 3600 AS INTEGER) AS hour_bucket,
                       AVG(e.faithfulness)        AS faithfulness,
                       AVG(e.answer_relevancy)    AS answer_relevancy,
                       AVG(e.context_precision)   AS context_precision,
                       AVG(e.mean_chunk_score)    AS mean_chunk_score,
                       COUNT(*)                   AS query_count
                   FROM query_logs q
                   JOIN eval_metrics e ON e.query_log_id = q.id
                   WHERE q.timestamp > ?
                   GROUP BY hour_bucket
                   ORDER BY hour_bucket""",
                (since, since),
            ).fetchall()
        return [dict(zip(
            ["hour_bucket", "faithfulness", "answer_relevancy",
             "context_precision", "mean_chunk_score", "query_count"], r
        )) for r in rows]

    def get_summary_stats(self) -> dict:
        """Aggregate stats for the dashboard header metrics."""
        with self._conn() as conn:
            total = conn.execute("SELECT COUNT(*) FROM query_logs").fetchone()[0]
            with_metrics = conn.execute("SELECT COUNT(*) FROM eval_metrics").fetchone()[0]
            avgs = conn.execute(
                """SELECT AVG(faithfulness), AVG(answer_relevancy),
                          AVG(context_precision), AVG(mean_chunk_score)
                   FROM eval_metrics"""
            ).fetchone()
            grade_dist = conn.execute(
                """SELECT crag_grade, COUNT(*) as cnt
                   FROM query_logs WHERE crag_grade != ''
                   GROUP BY crag_grade"""
            ).fetchall()
            strategy_dist = conn.execute(
                """SELECT strategies, COUNT(*) as cnt
                   FROM query_logs GROUP BY strategies"""
            ).fetchall()
            avg_latency = conn.execute(
                "SELECT AVG(latency_ms) FROM query_logs WHERE latency_ms > 0"
            ).fetchone()[0]

        return {
            "total_queries": total,
            "evaluated_queries": with_metrics,
            "avg_faithfulness":      round(avgs[0] or 0, 3),
            "avg_answer_relevancy":  round(avgs[1] or 0, 3),
            "avg_context_precision": round(avgs[2] or 0, 3),
            "avg_chunk_score":       round(avgs[3] or 0, 3),
            "avg_latency_ms":        round(avg_latency or 0, 1),
            "crag_grade_dist":  {r[0]: r[1] for r in grade_dist},
            "strategy_dist":    {r[0]: r[1] for r in strategy_dist},
        }

    # ── Init ───────────────────────────────────────────────────

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.executescript(_DDL)
        logger.info("EvalStore ready at %s", self._path)

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self._path, timeout=10, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    @staticmethod
    def _row_to_dict(row) -> dict:
        d = dict(row)
        for key in ("strategies",):
            if d.get(key):
                try:
                    d[key] = json.loads(d[key])
                except Exception:
                    pass
        return d
