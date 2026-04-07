const API = '';   // same origin — FastAPI serves this file
let isStreaming = false;
let conversation = [];

// ── Nav routing ──────────────────────────────────────────────
document.querySelectorAll('.nav-item').forEach(item => {
  item.addEventListener('click', () => {
    const tab = item.dataset.tab;
    document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));
    document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
    item.classList.add('active');
    document.getElementById(`tab-${tab}`).classList.add('active');
    if (tab === 'eval') loadMetrics();
    if (tab === 'system') loadHealth();
  });
});

// ── Health check ─────────────────────────────────────────────
async function checkHealth() {
  try {
    const r = await fetch(`${API}/health`);
    const d = await r.json();
    const dot = document.getElementById('statusDot');
    const lbl = document.getElementById('statusLabel');
    if (d.status === 'ok') {
      dot.className = 'status-dot ok';
      lbl.textContent = `${d.collection_stats?.entity_count ?? 0} chunks`;
    } else {
      dot.className = 'status-dot err';
      lbl.textContent = 'degraded';
    }
  } catch {
    document.getElementById('statusDot').className = 'status-dot err';
    document.getElementById('statusLabel').textContent = 'offline';
  }
}
checkHealth();
setInterval(checkHealth, 30000);

// ── Toast ─────────────────────────────────────────────────────
function toast(msg, dur = 3000) {
  const t = document.getElementById('toast');
  t.textContent = msg;
  t.classList.add('show');
  setTimeout(() => t.classList.remove('show'), dur);
}

// ══════════════════════════════════════════════════════════════
// CHAT
// ══════════════════════════════════════════════════════════════
const chatMessages = document.getElementById('chatMessages');
const chatInput    = document.getElementById('chatInput');
const sendBtn      = document.getElementById('sendBtn');
const streamStatus = document.getElementById('streamStatus');
const sourcesList  = document.getElementById('sourcesList');

// Auto-resize textarea
chatInput.addEventListener('input', () => {
  chatInput.style.height = 'auto';
  chatInput.style.height = Math.min(chatInput.scrollHeight, 160) + 'px';
});

chatInput.addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); }
});

sendBtn.addEventListener('click', sendMessage);
document.getElementById('clearChat').addEventListener('click', clearChat);
document.getElementById('clearChat2').addEventListener('click', clearChat);

function clearChat() {
  conversation = [];
  chatMessages.innerHTML = `
    <div class="message">
      <div class="msg-avatar ai">cx</div>
      <div class="msg-body">
        <div class="msg-role">CORTEX</div>
        <div class="msg-text">Ready. Ask anything about your ingested documents.</div>
      </div>
    </div>`;
  sourcesList.innerHTML = `<div class="empty-sources">
    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" opacity="0.3">
      <path d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"/>
    </svg>
    <span>Sources will appear here after your first query</span>
  </div>`;
  streamStatus.textContent = '';
}

function appendMessage(role, text, metaBadges = []) {
  const isAI = role === 'ai';
  const div = document.createElement('div');
  div.className = 'message';
  div.innerHTML = `
    <div class="msg-avatar ${role}">${isAI ? 'cx' : 'you'}</div>
    <div class="msg-body">
      <div class="msg-role">${isAI ? 'CORTEX' : 'YOU'}</div>
      <div class="msg-text" id="mt-${Date.now()}"></div>
      <div class="info-bar" id="mb-${Date.now()}"></div>
    </div>`;
  chatMessages.appendChild(div);
  const textEl  = div.querySelector('.msg-text');
  const badgeEl = div.querySelector('.info-bar');
  textEl.textContent = text;
  metaBadges.forEach(b => {
    const s = document.createElement('span');
    s.className = `badge badge-${b.color}`;
    s.textContent = b.text;
    badgeEl.appendChild(s);
  });
  chatMessages.scrollTop = chatMessages.scrollHeight;
  return { textEl, badgeEl };
}

function renderSourceCards(chunks) {
  if (!chunks.length) return;
  sourcesList.innerHTML = '';
  chunks.forEach((c, i) => {
    const score = Math.min(Math.max(c.score, 0), 1);
    const pct   = Math.round(score * 100);
    const ret   = c.retriever || 'dense';
    const retColors = { dense:'blue', bm25:'green', graph:'amber', 'web_search':'red', 'dense+bm25':'purple', 'bm25+dense':'purple' };
    const retColor  = retColors[ret] || 'muted';
    const card = document.createElement('div');
    card.className = 'source-card';
    card.innerHTML = `
      <div class="source-num">[${i+1}] <span class="badge badge-${retColor}" style="font-size:9px">${ret}</span></div>
      <div class="source-title" title="${c.title}">${c.title}</div>
      <div class="source-snippet">${c.text_snippet || ''}</div>
      <div class="source-meta">
        <div class="score-bar-wrap"><div class="score-bar" style="width:${pct}%"></div></div>
        <span class="score-val">${pct}%</span>
      </div>`;
    sourcesList.appendChild(card);
  });
}

async function sendMessage() {
  const query = chatInput.value.trim();
  if (!query || isStreaming) return;
  const topK = parseInt(document.getElementById('topkInput').value) || 10;

  chatInput.value = '';
  chatInput.style.height = 'auto';
  isStreaming = true;
  sendBtn.disabled = true;
  sendBtn.textContent = '…';

  // User message
  appendMessage('user', query);
  conversation.push({ role: 'user', content: query });

  // AI message skeleton
  const { textEl, badgeEl } = appendMessage('ai', '');
  const cursor = document.createElement('span');
  cursor.className = 'cursor-blink';
  textEl.appendChild(cursor);

  streamStatus.textContent = 'retrieving…';

  let fullText = '';

  try {
    const resp = await fetch(`${API}/query/stream`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query, top_k: topK, stream: true }),
    });

    if (!resp.ok) throw new Error(`API ${resp.status}`);

    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buf = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });

      const lines = buf.split('\n');
      buf = lines.pop(); // keep incomplete line

      for (const line of lines) {
        if (!line.startsWith('data: ')) continue;
        let evt;
        try { evt = JSON.parse(line.slice(6)); } catch { continue; }

        if (evt.type === 'chunk_meta') {
          const chunks = evt.chunks || [];
          const routing = evt.routing || {};
          renderSourceCards(chunks);
          streamStatus.textContent = `retrieved ${chunks.length} passages`;

          // Routing badges
          if (routing.intent) {
            const b = document.createElement('span');
            b.className = 'badge badge-amber';
            b.textContent = routing.intent;
            badgeEl.appendChild(b);
          }
          (routing.strategies || []).forEach(s => {
            const b = document.createElement('span');
            b.className = 'badge badge-blue';
            b.textContent = s.toUpperCase();
            badgeEl.appendChild(b);
          });
          Object.entries(routing.retriever_hits || {}).forEach(([k, v]) => {
            const b = document.createElement('span');
            b.className = 'badge badge-muted';
            b.textContent = `${k}: ${v}`;
            badgeEl.appendChild(b);
          });
        }

        else if (evt.type === 'crag_update') {
          const gradeColors = { GOOD: 'green', POOR: 'amber', ABSENT: 'red' };
          const b = document.createElement('span');
          b.className = `badge badge-${gradeColors[evt.grade] || 'muted'}`;
          b.textContent = `CRAG: ${evt.grade}`;
          badgeEl.appendChild(b);
          if (evt.web_search_used) {
            const wb = document.createElement('span');
            wb.className = 'badge badge-red';
            wb.textContent = '🌐 web fallback';
            badgeEl.appendChild(wb);
          }
          if (evt.rewritten_query) {
            streamStatus.textContent = `rewritten → "${evt.rewritten_query.slice(0, 50)}…"`;
          }
        }

        else if (evt.type === 'token') {
          fullText += evt.text || '';
          // Remove cursor, set text, re-add cursor
          cursor.remove();
          textEl.textContent = fullText;
          textEl.appendChild(cursor);
          chatMessages.scrollTop = chatMessages.scrollHeight;
        }

        else if (evt.type === 'sources') {
          // Append sources block after answer
          const src = document.createElement('div');
          src.style.cssText = 'margin-top:12px;padding-top:10px;border-top:1px solid var(--border);font-family:var(--mono);font-size:11px;color:var(--muted);line-height:1.8;white-space:pre-wrap';
          src.textContent = evt.text || '';
          textEl.parentElement.appendChild(src);
        }

        else if (evt.type === 'done') {
          cursor.remove();
          textEl.textContent = fullText;
          streamStatus.textContent = '';
        }

        else if (evt.type === 'error') {
          cursor.remove();
          textEl.textContent = `Error: ${evt.message}`;
          textEl.style.color = 'var(--red)';
          streamStatus.textContent = '';
        }
      }
    }
  } catch (err) {
    cursor.remove();
    textEl.textContent = `Connection error: ${err.message}. Is the API running on port 8000?`;
    textEl.style.color = 'var(--red)';
    streamStatus.textContent = '';
  }

  conversation.push({ role: 'assistant', content: fullText });
  isStreaming = false;
  sendBtn.disabled = false;
  sendBtn.textContent = 'send';
  chatMessages.scrollTop = chatMessages.scrollHeight;
}

// ══════════════════════════════════════════════════════════════
// INGEST
// ══════════════════════════════════════════════════════════════
document.getElementById('ingestBtn').addEventListener('click', async () => {
  const path = document.getElementById('ingestPath').value.trim();
  if (!path) { toast('Enter a server path first'); return; }
  const recursive = document.getElementById('ingestRecursive').checked;

  const btn  = document.getElementById('ingestBtn');
  const prog = document.getElementById('ingestProgress');
  const res  = document.getElementById('ingestResult');
  btn.disabled = true;
  btn.textContent = 'running…';
  prog.style.display = 'block';
  res.style.display  = 'none';

  try {
    const r = await fetch(`${API}/ingest`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ path, recursive }),
    });
    const d = await r.json();

    prog.style.display = 'none';
    res.style.display  = 'block';

    const errHtml = (d.errors || []).map(e =>
      `<div class="error-row">⚠ ${e.source}: ${e.error}</div>`
    ).join('');

    res.innerHTML = `
      <h4>ingestion complete</h4>
      <div class="stat-grid">
        <div class="stat-cell"><div class="stat-val">${d.documents_processed}</div><div class="stat-key">DOCS PROCESSED</div></div>
        <div class="stat-cell"><div class="stat-val">${d.chunks_stored}</div><div class="stat-key">CHUNKS STORED</div></div>
        <div class="stat-cell"><div class="stat-val">${d.bm25_indexed || 0}</div><div class="stat-key">BM25 INDEXED</div></div>
        <div class="stat-cell"><div class="stat-val">${d.documents_skipped}</div><div class="stat-key">SKIPPED</div></div>
        <div class="stat-cell"><div class="stat-val">${d.graph_entities || 0}</div><div class="stat-key">ENTITIES</div></div>
        <div class="stat-cell"><div class="stat-val">${d.graph_triples || 0}</div><div class="stat-key">TRIPLES</div></div>
      </div>
      ${errHtml ? `<div class="ingest-errors">${errHtml}</div>` : ''}`;

    // Log entry
    const logList = document.getElementById('ingestLogList');
    const entry   = document.createElement('div');
    entry.className = 'log-entry';
    entry.innerHTML = `<span class="log-ts">${new Date().toLocaleTimeString()}</span>${path} → ${d.chunks_stored} chunks`;
    logList.prepend(entry);

    checkHealth();
    toast(`✓ ${d.documents_processed} docs, ${d.chunks_stored} chunks indexed`);
  } catch (err) {
    prog.style.display = 'none';
    toast(`Error: ${err.message}`);
  }

  btn.disabled = false;
  btn.textContent = 'run ingestion';
});

// ══════════════════════════════════════════════════════════════
// EVAL DASHBOARD
// ══════════════════════════════════════════════════════════════
document.getElementById('refreshMetrics').addEventListener('click', loadMetrics);
document.getElementById('flushCache').addEventListener('click', async () => {
  try {
    const r = await fetch(`${API}/cache/flush`, { method: 'POST' });
    const d = await r.json();
    toast(`Cache flushed — ${d.deleted} entries deleted`);
    loadMetrics();
  } catch { toast('Cache flush failed'); }
});

async function loadMetrics() {
  const body = document.getElementById('evalBody');
  body.innerHTML = '<div style="color:var(--muted);font-family:var(--mono);font-size:12px;padding:40px 0;text-align:center">loading…</div>';
  try {
    const r = await fetch(`${API}/metrics?limit=50&days=14`);
    const d = await r.json();
    renderEvalDashboard(d);
  } catch (err) {
    body.innerHTML = `<div style="color:var(--red);font-family:var(--mono);font-size:12px;padding:40px 0;text-align:center">Error: ${err.message}</div>`;
  }
}

function renderEvalDashboard(d) {
  const body   = document.getElementById('evalBody');
  const s      = d.summary || {};
  const cache  = d.cache   || {};
  const recent = d.recent  || [];
  const grades = s.crag_grade_dist || {};
  const totalG = Object.values(grades).reduce((a, b) => a + b, 0) || 1;

  const fmt = v => (v != null && !isNaN(v)) ? v.toFixed(2) : '—';

  // KPI row
  const kpiHtml = `
    <div class="kpi-row">
      <div class="kpi-card"><div class="kpi-val amber">${s.total_queries ?? 0}</div><div class="kpi-label">TOTAL QUERIES</div></div>
      <div class="kpi-card"><div class="kpi-val green">${fmt(s.avg_faithfulness)}</div><div class="kpi-label">FAITHFULNESS</div></div>
      <div class="kpi-card"><div class="kpi-val blue">${fmt(s.avg_answer_relevancy)}</div><div class="kpi-label">ANSWER RELEVANCY</div></div>
      <div class="kpi-card"><div class="kpi-val">${fmt(s.avg_context_precision)}</div><div class="kpi-label">CTX PRECISION</div></div>
      <div class="kpi-card"><div class="kpi-val">${s.avg_latency_ms ? Math.round(s.avg_latency_ms) + ' ms' : '—'}</div><div class="kpi-label">AVG LATENCY</div></div>
      <div class="kpi-card"><div class="kpi-val ${cache.enabled ? 'green' : ''}">${cache.enabled ? (cache.hit_rate*100).toFixed(0)+'%' : 'off'}</div><div class="kpi-label">CACHE HIT RATE</div></div>
    </div>`;

  // Metric bars
  const metrics = [
    { name:'faithfulness',      val: s.avg_faithfulness,      color:'#34d399' },
    { name:'answer_relevancy',  val: s.avg_answer_relevancy,  color:'#60a5fa' },
    { name:'ctx_precision',     val: s.avg_context_precision, color:'#a78bfa' },
    { name:'avg_chunk_score',   val: s.avg_chunk_score,       color:'#f59e0b' },
  ];
  const metricBarsHtml = metrics.map(m => {
    const pct = m.val != null ? Math.round(m.val * 100) : 0;
    return `<div class="metric-row">
      <span class="metric-name">${m.name}</span>
      <div class="metric-bar-wrap"><div class="metric-bar" style="width:${pct}%;background:${m.color}"></div></div>
      <span class="metric-num">${m.val != null ? m.val.toFixed(2) : '—'}</span>
    </div>`;
  }).join('');

  // CRAG grade dist
  const gradeOrder = ['GOOD', 'POOR', 'ABSENT'];
  const gradeBarsHtml = gradeOrder.map(g => {
    const cnt = grades[g] || 0;
    const pct = Math.round((cnt / totalG) * 100);
    return `<div class="grade-row">
      <span class="grade-label">${g}</span>
      <div class="grade-bar-wrap">
        <div class="grade-bar ${g.toLowerCase()}" style="width:${pct}%">${cnt > 0 ? cnt : ''}</div>
      </div>
    </div>`;
  }).join('');

  // Cache info
  const cacheHtml = cache.enabled ? `
    <div class="cache-row">
      <div class="cache-stat">${cache.hits ?? 0}<span>HITS</span></div>
      <div class="cache-stat">${cache.misses ?? 0}<span>MISSES</span></div>
      <div class="cache-stat">${cache.ttl_s ? Math.round(cache.ttl_s/60) + ' min' : '—'}<span>TTL</span></div>
    </div>` : `<span style="color:var(--muted);font-family:var(--mono);font-size:11px">Redis not connected — <code style="color:var(--amber)">docker run -d -p 6379:6379 redis:7-alpine</code></span>`;

  // Recent queries table
  const tableRows = recent.slice(0, 30).map(r => `
    <tr>
      <td class="query-col" title="${r.query}">${r.query}</td>
      <td class="mono" style="white-space:nowrap;font-size:10px;color:var(--muted)">${r.intent || '—'}</td>
      <td>${r.crag_grade ? `<span class="badge badge-${r.crag_grade==='GOOD'?'green':r.crag_grade==='POOR'?'amber':'red'}" style="font-size:10px">${r.crag_grade}</span>` : '—'}</td>
      <td class="${r.faithfulness!=null?'score-col':'na'}">${fmt(r.faithfulness)}</td>
      <td class="${r.answer_relevancy!=null?'score-col':'na'}">${fmt(r.answer_relevancy)}</td>
      <td class="${r.context_precision!=null?'score-col':'na'}">${fmt(r.context_precision)}</td>
      <td class="mono" style="color:var(--muted);font-size:11px">${r.latency_ms ? Math.round(r.latency_ms) : '—'} ms</td>
    </tr>`).join('');

  body.innerHTML = `
    ${kpiHtml}
    <div class="eval-grid">
      <div class="eval-card">
        <h4>ragas metrics</h4>
        ${metricBarsHtml || '<span style="color:var(--muted);font-size:11px">No eval data yet. Run some queries first.</span>'}
      </div>
      <div class="eval-card">
        <h4>crag grade distribution</h4>
        <div class="grade-bars">${gradeBarsHtml}</div>
      </div>
      <div class="eval-card">
        <h4>cache</h4>
        ${cacheHtml}
      </div>
      <div class="eval-card">
        <h4>strategy mix</h4>
        ${renderStrategyMix(s.strategy_dist || {})}
      </div>
    </div>
    <div class="query-table-wrap">
      <h4>recent queries</h4>
      ${tableRows ? `<table>
        <thead><tr><th>Query</th><th>Intent</th><th>CRAG</th><th>Faithful</th><th>Relevancy</th><th>Precision</th><th>Latency</th></tr></thead>
        <tbody>${tableRows}</tbody>
      </table>` : '<div style="padding:20px;color:var(--muted);font-family:var(--mono);font-size:12px">No queries logged yet.</div>'}
    </div>`;
}

function renderStrategyMix(dist) {
  const entries = Object.entries(dist);
  if (!entries.length) return '<span style="color:var(--muted);font-size:11px">No data yet.</span>';
  const total = entries.reduce((a, [, v]) => a + v, 0);
  return entries.map(([k, v]) => {
    let label = k;
    try { label = JSON.parse(k).join('+').toUpperCase(); } catch {}
    const pct = Math.round((v / total) * 100);
    return `<div class="grade-row">
      <span class="grade-label" style="width:90px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis" title="${label}">${label}</span>
      <div class="grade-bar-wrap">
        <div class="grade-bar good" style="width:${pct}%">${v}</div>
      </div>
    </div>`;
  }).join('');
}

// ══════════════════════════════════════════════════════════════
// SYSTEM HEALTH
// ══════════════════════════════════════════════════════════════
document.getElementById('refreshSystem').addEventListener('click', loadHealth);

async function loadHealth() {
  const body = document.getElementById('systemBody');
  try {
    const r = await fetch(`${API}/health`);
    const d = await r.json();
    const cs = d.collection_stats || {};
    const gs = d.graph_stats || {};

    body.innerHTML = `
      <div class="system-grid">
        <div class="system-card">
          <div class="sys-name">OVERALL STATUS</div>
          <div class="sys-val" style="color:${d.status==='ok'?'var(--green)':'var(--red)'}">${d.status}</div>
        </div>
        <div class="system-card">
          <div class="sys-name">MILVUS</div>
          <div class="sys-val" style="color:${d.milvus==='ok'?'var(--green)':'var(--red)'}">${d.milvus === 'ok' ? '●' : '✕'}</div>
          <div class="sys-sub">${cs.entity_count ?? 0} chunks indexed</div>
        </div>
        <div class="system-card">
          <div class="sys-name">EMBEDDER</div>
          <div class="sys-val" style="color:${d.embedder==='loaded'?'var(--green)':'var(--amber)'}">${d.embedder}</div>
        </div>
        <div class="system-card">
          <div class="sys-name">GRAPH NODES</div>
          <div class="sys-val">${gs.nodes ?? '—'}</div>
          <div class="sys-sub">${gs.edges ?? 0} edges · ${gs.extractor ?? '—'}</div>
        </div>
        <div class="system-card">
          <div class="sys-name">COLLECTION</div>
          <div class="sys-val">${cs.collection ?? '—'}</div>
        </div>
        <div class="system-card">
          <div class="sys-name">CHUNKS</div>
          <div class="sys-val amber">${cs.entity_count ?? 0}</div>
        </div>
      </div>
      <div class="json-block">${JSON.stringify(d, null, 2)}</div>`;
  } catch (err) {
    body.innerHTML = `<div style="color:var(--red);font-family:var(--mono);font-size:12px;padding:40px 0;text-align:center">Cannot reach API: ${err.message}</div>`;
  }
}