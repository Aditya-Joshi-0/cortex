/* THEME */
const html = document.documentElement;
const themeBtn = document.getElementById('themeBtn');
function applyTheme(t){
  html.dataset.theme=t;
  themeBtn.textContent=t==='dark'?'☀':'☾';
  localStorage.setItem('cortex-theme',t);
}
const stored=localStorage.getItem('cortex-theme');
const prefersDark=window.matchMedia('(prefers-color-scheme: dark)').matches;
applyTheme(stored||(prefersDark?'dark':'light'));
themeBtn.addEventListener('click',()=>applyTheme(html.dataset.theme==='dark'?'light':'dark'));
window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change',e=>{
  if(!localStorage.getItem('cortex-theme')) applyTheme(e.matches?'dark':'light');
});

/* RESPONSIVE NAV BINDINGS */
const sidebar = document.getElementById('sidebar');
const mobileNavToggles = document.querySelectorAll('.mobile-nav-toggle');
const mobileOverlay = document.getElementById('mobileOverlay');

function openMobileSidebar() {
  sidebar.classList.add('mobile-open');
  mobileOverlay.classList.add('show');
}
function closeMobilePanels() {
  sidebar.classList.remove('mobile-open');
  sourcesPanel.classList.remove('mobile-open');
  mobileOverlay.classList.remove('show');
}

mobileNavToggles.forEach(btn => btn.addEventListener('click', openMobileSidebar));
mobileOverlay.addEventListener('click', closeMobilePanels);

/* COLLAPSIBLE SIDEBARS (Desktop) & Sources Panel */
const sidebarToggle=document.getElementById('sidebarToggle');
const sourcesPanel=document.getElementById('sourcesPanel');
const sourcesToggle=document.getElementById('sourcesToggle');
const mobileSourcesBtn=document.getElementById('mobileSourcesBtn');

sidebarToggle.addEventListener('click',()=>{
  const c=sidebar.classList.toggle('collapsed');
  sidebarToggle.textContent=c?'▶':'◀';
  sidebarToggle.title=c?'Expand':'Collapse';
});

// For desktop
sourcesToggle.addEventListener('click',()=>{
  const c=sourcesPanel.classList.toggle('collapsed');
  sourcesToggle.textContent=c?'◀':'▶';
  sourcesToggle.title=c?'Expand':'Collapse';
});

// For mobile
if(mobileSourcesBtn) {
  mobileSourcesBtn.addEventListener('click',()=>{
    sourcesPanel.classList.add('mobile-open');
    mobileOverlay.classList.add('show');
  });
}

/* NAV ROUTING */
document.querySelectorAll('.nav-item').forEach(item=>{
  item.addEventListener('click',()=>{
    document.querySelectorAll('.nav-item').forEach(n=>n.classList.remove('active'));
    document.querySelectorAll('.tab-panel').forEach(p=>p.classList.remove('active'));
    item.classList.add('active');
    document.getElementById('tab-'+item.dataset.tab).classList.add('active');
    if(item.dataset.tab==='eval') loadMetrics();
    if(item.dataset.tab==='system') loadHealth();
    
    // Auto-close sidebar on mobile after navigating
    if(window.innerWidth <= 768) {
        closeMobilePanels();
    }
  });
});

/* HEALTH */
async function checkHealth(){
  try{
    const r=await fetch('/health');
    const d=await r.json();
    const dot=document.getElementById('statusDot');
    const lbl=document.getElementById('statusLabel');
    if(d.status==='ok'){dot.className='status-dot ok';lbl.textContent=(d.collection_stats?.entity_count??0)+' chunks';}
    else{dot.className='status-dot err';lbl.textContent='degraded';}
  }catch{document.getElementById('statusDot').className='status-dot err';document.getElementById('statusLabel').textContent='offline';}
}
checkHealth();setInterval(checkHealth,30000);

/* TOAST */
function toast(msg,dur=3000){
  const t=document.getElementById('toast');t.textContent=msg;t.classList.add('show');
  clearTimeout(t._tid);t._tid=setTimeout(()=>t.classList.remove('show'),dur);
}

/* MARKED */
marked.setOptions({breaks:true,gfm:true});

function renderMarkdown(text){return marked.parse(text);}

function linkifyCitations(html,n){
  return html.replace(/\[(\d+)\]/g,(match,num)=>{
    const i=parseInt(num);
    if(i<1||i>n) return match;
    return '<a class="cite-link" onclick="highlightSource('+i+')" title="Jump to source '+i+'">['+i+']</a>';
  });
}

function highlightSource(n){
  const card=document.getElementById('src-card-'+n);
  if(!card) return;
  
  // Handle desktop collapse logic
  if(window.innerWidth > 768 && sourcesPanel.classList.contains('collapsed')){
    sourcesPanel.classList.remove('collapsed');
    sourcesToggle.textContent='▶';
  }
  // Handle mobile slide-in logic
  if(window.innerWidth <= 768 && !sourcesPanel.classList.contains('mobile-open')){
    sourcesPanel.classList.add('mobile-open');
    mobileOverlay.classList.add('show');
  }

  card.scrollIntoView({behavior:'smooth',block:'nearest'});
  card.classList.remove('highlighted');
  void card.offsetWidth;
  card.classList.add('highlighted');
  setTimeout(()=>card.classList.remove('highlighted'),1500);
}

/* CHAT */
const chatMessages=document.getElementById('chatMessages');
const chatInput=document.getElementById('chatInput');
const sendBtn=document.getElementById('sendBtn');
const streamStatus=document.getElementById('streamStatus');
const sourcesList=document.getElementById('sourcesList');
let isStreaming=false;
let currentChunks=[];

chatInput.addEventListener('input',()=>{
  chatInput.style.height='auto';
  chatInput.style.height=Math.min(chatInput.scrollHeight,150)+'px';
});
chatInput.addEventListener('keydown',e=>{if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();sendMessage();}});
sendBtn.addEventListener('click',sendMessage);

document.getElementById('clearChatBtn').addEventListener('click',()=>{
  chatMessages.innerHTML='<div class="message"><div class="msg-avatar ai">cx</div><div class="msg-body"><div class="msg-role">CORTEX</div><div class="msg-text">Cleared. Ask anything.</div></div></div>';
  sourcesList.innerHTML='<div class="empty-sources"><svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"/></svg><span>Retrieved passages will appear here</span></div>';
  streamStatus.textContent='';currentChunks=[];
});

function renderSourceCards(chunks){
  currentChunks=chunks;
  if(!chunks.length) return;
  sourcesList.innerHTML='';
  chunks.forEach((c,i)=>{
    const n=i+1;
    const pct=Math.round(Math.min(Math.max(c.score,0),1)*100);
    const card=document.createElement('div');
    card.className='source-card';card.id='src-card-'+n;
    card.innerHTML='<div class="source-num">['+n+']</div><div class="source-title" title="'+escHtml(c.title)+'">'+escHtml(c.title)+'</div><div class="source-snippet">'+escHtml(c.text_snippet||'')+'</div><div class="source-meta"><div class="score-bar-wrap"><div class="score-bar" style="width:'+pct+'%"></div></div><span class="score-val">'+pct+'%</span></div>';
    sourcesList.appendChild(card);
  });
}

function buildSourcePills(chunks){
  if(!chunks.length) return '';
  const pills=chunks.map((c,i)=>{
    const n=i+1;
    const snippet=(c.text_snippet||'').slice(0,175);
    const pct=Math.round(Math.min(Math.max(c.score,0),1)*100);
    const fname=(c.source||'').split('/').pop()||c.source;
    const title=c.title.slice(0,24)+(c.title.length>24?'…':'');
    return '<span class="source-pill" onclick="highlightSource('+n+')">['+n+'] '+escHtml(title)+'<div class="pill-tip"><div class="pill-tip-title">'+escHtml(c.title)+'</div><div class="pill-tip-snippet">'+escHtml(snippet)+(snippet.length>=175?'…':'')+'</div><div class="pill-tip-score">'+escHtml(fname)+' · '+pct+'% relevance</div></div></span>';
  }).join('');
  return '<div class="source-pills">'+pills+'</div>';
}

async function sendMessage(){
  const query=chatInput.value.trim();
  if(!query||isStreaming) return;
  chatInput.value='';chatInput.style.height='auto';
  isStreaming=true;sendBtn.disabled=true;sendBtn.textContent='…';currentChunks=[];

  // User bubble
  const ud=document.createElement('div');ud.className='message';
  ud.innerHTML='<div class="msg-avatar user">you</div><div class="msg-body"><div class="msg-role">YOU</div><div class="msg-text">'+escHtml(query)+'</div></div>';
  chatMessages.appendChild(ud);

  // AI bubble
  const ad=document.createElement('div');ad.className='message';
  ad.innerHTML='<div class="msg-avatar ai">cx</div><div class="msg-body"><div class="msg-role">CORTEX</div><div class="msg-text streaming" id="live-text"></div><div class="info-bar" id="live-badges"></div></div>';
  chatMessages.appendChild(ad);
  chatMessages.scrollTop=chatMessages.scrollHeight;

  const liveText=document.getElementById('live-text');
  const liveBadges=document.getElementById('live-badges');
  const cursor=document.createElement('span');cursor.className='cursor-blink';
  liveText.appendChild(cursor);

  let rawText='';
  streamStatus.textContent='…';

  try{
    const resp=await fetch('/query/stream',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({query,top_k:10,stream:true})});
    if(!resp.ok) throw new Error('HTTP '+resp.status);
    const reader=resp.body.getReader();
    const decoder=new TextDecoder();
    let buf='';

    while(true){
      const{done,value}=await reader.read();
      if(done) break;
      buf+=decoder.decode(value,{stream:true});
      const lines=buf.split('\n');buf=lines.pop();

      for(const line of lines){
        if(!line.startsWith('data: ')) continue;
        let evt;try{evt=JSON.parse(line.slice(6));}catch{continue;}

        if(evt.type==='chunk_meta'){
          const chunks=evt.chunks||[];
          const routing=evt.routing||{};
          renderSourceCards(chunks);
          streamStatus.textContent='generating…';
          if(routing.intent) addBadge(liveBadges,routing.intent,'amber');
          (routing.strategies||[]).forEach(s=>addBadge(liveBadges,s.toUpperCase(),'blue'));
        }
        else if(evt.type==='crag_update'){
          const gc={GOOD:'green',POOR:'amber',ABSENT:'red'};
          addBadge(liveBadges,'CRAG: '+(evt.grade||''),gc[evt.grade]||'muted');
          if(evt.web_search_used) addBadge(liveBadges,'🌐 web','red');
          if(evt.rewritten_query) streamStatus.textContent='rewritten: "'+evt.rewritten_query.slice(0,50)+'…"';
        }
        else if(evt.type==='token'){
          const tok=evt.text||'';
          rawText+=tok;
          cursor.before(document.createTextNode(tok));
          chatMessages.scrollTop=chatMessages.scrollHeight;
        }
        else if(evt.type==='sources'){
          cursor.remove();
          liveText.classList.remove('streaming');
          liveText.innerHTML=linkifyCitations(renderMarkdown(rawText),currentChunks.length);
          liveText.insertAdjacentHTML('afterend',buildSourcePills(currentChunks));
          streamStatus.textContent='';
          chatMessages.scrollTop=chatMessages.scrollHeight;
        }
        else if(evt.type==='done'){
          if(cursor.isConnected){
            cursor.remove();
            liveText.classList.remove('streaming');
            liveText.innerHTML=linkifyCitations(renderMarkdown(rawText),currentChunks.length);
            if(currentChunks.length) liveText.insertAdjacentHTML('afterend',buildSourcePills(currentChunks));
          }
          streamStatus.textContent='';
        }
        else if(evt.type==='error'){
          cursor.remove();
          liveText.textContent='Error: '+evt.message;
          liveText.style.color='var(--red)';
          streamStatus.textContent='';
        }
      }
    }
  }catch(err){
    cursor.remove();
    liveText.textContent='Connection error: '+err.message;
    liveText.style.color='var(--red)';
    streamStatus.textContent='';
  }

  liveText.removeAttribute('id');liveBadges.removeAttribute('id');
  isStreaming=false;sendBtn.disabled=false;sendBtn.textContent='send';
  chatMessages.scrollTop=chatMessages.scrollHeight;
}

function addBadge(container,text,color){
  const b=document.createElement('span');b.className='badge badge-'+color;b.textContent=text;container.appendChild(b);
}
function escHtml(s){return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');}

/* INGEST TABS */
document.querySelectorAll('.ingest-tab').forEach(tab=>{
  tab.addEventListener('click',()=>{
    const sec=tab.dataset.section;
    document.querySelectorAll('.ingest-tab').forEach(t=>t.classList.remove('active'));
    document.querySelectorAll('.ingest-section').forEach(s=>s.classList.remove('active'));
    tab.classList.add('active');
    document.getElementById('ingest-section-'+sec).classList.add('active');
  });
});

/* FILE UPLOAD */
let selectedFiles=[];

function fmtSize(b){
  if(b<1024) return b+'B';
  if(b<1048576) return (b/1024).toFixed(1)+'KB';
  return (b/1048576).toFixed(1)+'MB';
}

function renderFileList(){
  const list=document.getElementById('fileList');
  const count=document.getElementById('uploadCount');
  const uploadBtn=document.getElementById('uploadBtn');
  const clearBtn=document.getElementById('clearFilesBtn');
  list.innerHTML=selectedFiles.map((f,i)=>
    '<div class="file-item"><span class="file-item-name">'+escHtml(f.name)+'</span><span class="file-item-size">'+fmtSize(f.size)+'</span><button class="file-item-remove" data-i="'+i+'" title="Remove">✕</button></div>'
  ).join('');
  list.querySelectorAll('.file-item-remove').forEach(btn=>{
    btn.addEventListener('click',()=>{
      selectedFiles.splice(parseInt(btn.dataset.i),1);
      renderFileList();
    });
  });
  uploadBtn.disabled=selectedFiles.length===0;
  clearBtn.style.display=selectedFiles.length?'':'none';
  count.textContent=selectedFiles.length?selectedFiles.length+' file'+(selectedFiles.length>1?'s':'')+' selected':'';
}

function addFiles(newFiles){
  const allowed=new Set(['.pdf','.html','.htm','.txt','.md']);
  Array.from(newFiles).forEach(f=>{
    const ext=f.name.slice(f.name.lastIndexOf('.')).toLowerCase();
    if(!allowed.has(ext)){toast('Skipped '+f.name+' — unsupported type');return;}
    if(!selectedFiles.find(x=>x.name===f.name&&x.size===f.size)) selectedFiles.push(f);
  });
  renderFileList();
}

const dropZone=document.getElementById('dropZone');
const fileInput=document.getElementById('fileInput');

dropZone.addEventListener('click',()=>fileInput.click());
fileInput.addEventListener('change',()=>{addFiles(fileInput.files);fileInput.value='';});

dropZone.addEventListener('dragover',e=>{e.preventDefault();dropZone.classList.add('drag-over');});
dropZone.addEventListener('dragleave',()=>dropZone.classList.remove('drag-over'));
dropZone.addEventListener('drop',e=>{
  e.preventDefault();dropZone.classList.remove('drag-over');
  addFiles(e.dataTransfer.files);
});

document.getElementById('clearFilesBtn').addEventListener('click',()=>{selectedFiles=[];renderFileList();});

document.getElementById('uploadBtn').addEventListener('click',async()=>{
  if(!selectedFiles.length) return;
  const btn=document.getElementById('uploadBtn');
  const prog=document.getElementById('uploadProgress');
  const res=document.getElementById('ingestResult');
  btn.disabled=true;btn.textContent='uploading…';prog.style.display='block';res.style.display='none';

  const form=new FormData();
  selectedFiles.forEach(f=>form.append('files',f,f.name));

  try{
    const r=await fetch('/ingest/upload',{method:'POST',body:form});
    if(!r.ok){const e=await r.json();throw new Error(e.detail||'Upload failed');}
    const d=await r.json();
    prog.style.display='none';res.style.display='block';
    showIngestResult(d,'upload: '+selectedFiles.map(f=>f.name).join(', '));
    selectedFiles=[];renderFileList();checkHealth();
  }catch(err){
    prog.style.display='none';
    toast('Error: '+err.message);
    btn.disabled=false;btn.textContent='upload & ingest';
  }
  btn.disabled=false;btn.textContent='upload & ingest';
});

/* SERVER PATH INGEST */
document.getElementById('ingestBtn').addEventListener('click',async()=>{
  const path=document.getElementById('ingestPath').value.trim();
  if(!path){toast('Enter a server path first');return;}
  const recursive=document.getElementById('ingestRecursive').checked;
  const btn=document.getElementById('ingestBtn');
  const prog=document.getElementById('ingestProgress');
  const res=document.getElementById('ingestResult');
  btn.disabled=true;btn.textContent='running…';prog.style.display='block';res.style.display='none';
  try{
    const r=await fetch('/ingest',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({path,recursive})});
    const d=await r.json();
    prog.style.display='none';res.style.display='block';
    showIngestResult(d,path);checkHealth();
  }catch(err){prog.style.display='none';toast('Error: '+err.message);}
  btn.disabled=false;btn.textContent='run ingestion';
});

function showIngestResult(d,label){
  const res=document.getElementById('ingestResult');
  res.style.display='block';
  const errHtml=(d.errors||[]).map(e=>'<div class="error-row">⚠ '+escHtml(e.source)+': '+escHtml(e.error)+'</div>').join('');
  res.innerHTML='<h4>ingestion complete</h4><div class="stat-grid"><div class="stat-cell"><div class="stat-val">'+d.documents_processed+'</div><div class="stat-key">DOCS</div></div><div class="stat-cell"><div class="stat-val">'+d.chunks_stored+'</div><div class="stat-key">CHUNKS</div></div><div class="stat-cell"><div class="stat-val">'+(d.bm25_indexed||0)+'</div><div class="stat-key">BM25</div></div><div class="stat-cell"><div class="stat-val">'+d.documents_skipped+'</div><div class="stat-key">SKIPPED</div></div><div class="stat-cell"><div class="stat-val">'+(d.graph_entities||0)+'</div><div class="stat-key">ENTITIES</div></div><div class="stat-cell"><div class="stat-val">'+(d.graph_triples||0)+'</div><div class="stat-key">TRIPLES</div></div></div>'+(errHtml?'<div style="margin-top:9px">'+errHtml+'</div>':'');
  const ll=document.getElementById('ingestLogList');
  const le=document.createElement('div');le.className='log-entry';
  le.innerHTML='<span class="log-ts">'+new Date().toLocaleTimeString()+'</span>'+escHtml(label.slice(0,60))+' → '+d.chunks_stored+' chunks';
  ll.prepend(le);
  toast('✓ '+d.documents_processed+' docs, '+d.chunks_stored+' chunks');
}

/* EVAL */
document.getElementById('refreshMetrics').addEventListener('click',loadMetrics);
document.getElementById('flushCache').addEventListener('click',async()=>{
  try{const r=await fetch('/cache/flush',{method:'POST'});const d=await r.json();toast('Cache flushed — '+d.deleted+' entries');loadMetrics();}catch{toast('Flush failed');}
});

async function loadMetrics(){
  const body=document.getElementById('evalBody');
  body.innerHTML='<div style="color:var(--muted);font-family:var(--mono);font-size:12px;padding:40px 0;text-align:center">loading…</div>';
  try{const r=await fetch('/metrics?limit=50&days=14');const d=await r.json();renderEvalDashboard(d);}
  catch(err){body.innerHTML='<div style="color:var(--red);font-family:var(--mono);font-size:12px;padding:40px 0;text-align:center">Error: '+escHtml(err.message)+'</div>';}
}

function renderEvalDashboard(d){
  const body=document.getElementById('evalBody');
  const s=d.summary||{};const cache=d.cache||{};const recent=d.recent||[];
  const grades=s.crag_grade_dist||{};const totalG=Object.values(grades).reduce((a,b)=>a+b,0)||1;
  const fmt=v=>(v!=null&&!isNaN(v))?Number(v).toFixed(2):'—';

  const kpi='<div class="kpi-row"><div class="kpi-card"><div class="kpi-val amber">'+(s.total_queries??0)+'</div><div class="kpi-label">QUERIES</div></div><div class="kpi-card"><div class="kpi-val green">'+fmt(s.avg_faithfulness)+'</div><div class="kpi-label">FAITHFULNESS</div></div><div class="kpi-card"><div class="kpi-val blue">'+fmt(s.avg_answer_relevancy)+'</div><div class="kpi-label">RELEVANCY</div></div><div class="kpi-card"><div class="kpi-val">'+fmt(s.avg_context_precision)+'</div><div class="kpi-label">CTX PRECISION</div></div><div class="kpi-card"><div class="kpi-val">'+(s.avg_latency_ms?Math.round(s.avg_latency_ms)+'ms':'—')+'</div><div class="kpi-label">AVG LATENCY</div></div><div class="kpi-card"><div class="kpi-val '+(cache.enabled?'green':'')+'">'+( cache.enabled?Math.round((cache.hit_rate||0)*100)+'%':'off')+'</div><div class="kpi-label">CACHE HIT</div></div></div>';

  const mbars=[['faithfulness',s.avg_faithfulness,'#34d399'],['answer_relevancy',s.avg_answer_relevancy,'#60a5fa'],['ctx_precision',s.avg_context_precision,'#a78bfa'],['chunk_score',s.avg_chunk_score,'#f59e0b']].map(([name,val,color])=>{
    const pct=val!=null?Math.round(val*100):0;
    return '<div class="metric-row"><span class="metric-name">'+name+'</span><div class="metric-bar-wrap"><div class="metric-bar" style="width:'+pct+'%;background:'+color+'"></div></div><span class="metric-num">'+fmt(val)+'</span></div>';
  }).join('');

  const gbars=['GOOD','POOR','ABSENT'].map(g=>{
    const cnt=grades[g]||0;const pct=Math.round((cnt/totalG)*100);
    return '<div class="grade-row"><span class="grade-label">'+g+'</span><div class="grade-bar-wrap"><div class="grade-bar '+g.toLowerCase()+'" style="width:'+pct+'%">'+(cnt||'')+'</div></div></div>';
  }).join('');

  const cacheInfo=cache.enabled?'<div class="cache-row"><div class="cache-stat">'+(cache.hits??0)+'<span>HITS</span></div><div class="cache-stat">'+(cache.misses??0)+'<span>MISSES</span></div><div class="cache-stat">'+(cache.ttl_s?Math.round(cache.ttl_s/60)+'m':'—')+'<span>TTL</span></div></div>':'<span style="color:var(--muted);font-family:var(--mono);font-size:11px">Redis not connected</span>';

  const stratDist=s.strategy_dist||{};const stTotal=Object.values(stratDist).reduce((a,b)=>a+b,0)||1;
  const stratBars=Object.entries(stratDist).map(([k,v])=>{
    let label=k;try{label=JSON.parse(k).join('+').toUpperCase();}catch{}
    const pct=Math.round((v/stTotal)*100);
    return '<div class="grade-row"><span class="grade-label" style="width:78px;overflow:hidden;text-overflow:ellipsis">'+escHtml(label)+'</span><div class="grade-bar-wrap"><div class="grade-bar good" style="width:'+pct+'%">'+v+'</div></div></div>';
  }).join('')||'<span style="color:var(--muted);font-size:11px">No data</span>';

  const gc={GOOD:'green',POOR:'amber',ABSENT:'red'};
  const tableRows=recent.slice(0,30).map(r=>'<tr><td class="query-col" title="'+escHtml(r.query)+'">'+escHtml(r.query)+'</td><td class="mono" style="font-size:10px;color:var(--muted)">'+(r.intent||'—')+'</td><td>'+(r.crag_grade?'<span class="badge badge-'+(gc[r.crag_grade]||'muted')+'" style="font-size:9px">'+escHtml(r.crag_grade)+'</span>':'—')+'</td><td class="'+(r.faithfulness!=null?'mono':'na')+'">'+fmt(r.faithfulness)+'</td><td class="'+(r.answer_relevancy!=null?'mono':'na')+'">'+fmt(r.answer_relevancy)+'</td><td class="mono" style="color:var(--muted)">'+(r.latency_ms?Math.round(r.latency_ms)+'ms':'—')+'</td></tr>').join('');

  body.innerHTML=kpi+'<div class="eval-grid"><div class="eval-card"><h4>ragas metrics</h4>'+(mbars||'<span style="color:var(--muted);font-size:11px">No data yet</span>')+'</div><div class="eval-card"><h4>crag grade distribution</h4><div class="grade-bars">'+gbars+'</div></div><div class="eval-card"><h4>cache</h4>'+cacheInfo+'</div><div class="eval-card"><h4>retrieval strategy mix</h4><div class="grade-bars">'+stratBars+'</div></div></div><div class="query-table-wrap"><h4>recent queries</h4>'+(tableRows?'<table><thead><tr><th>Query</th><th>Intent</th><th>CRAG</th><th>Faithful</th><th>Relevancy</th><th>Latency</th></tr></thead><tbody>'+tableRows+'</tbody></table>':'<div style="padding:16px;color:var(--muted);font-family:var(--mono);font-size:12px">No queries yet.</div>')+'</div>';
}

/* SYSTEM */
document.getElementById('refreshSystem').addEventListener('click',loadHealth);
async function loadHealth(){
  const body=document.getElementById('systemBody');
  try{
    const r=await fetch('/health');const d=await r.json();
    const cs=d.collection_stats||{};const gs=d.graph_stats||{};
    body.innerHTML='<div class="system-grid"><div class="system-card"><div class="sys-name">STATUS</div><div class="sys-val" style="color:'+(d.status==='ok'?'var(--green)':'var(--red)')+'">'+d.status+'</div></div><div class="system-card"><div class="sys-name">MILVUS</div><div class="sys-val" style="color:'+(d.milvus==='ok'?'var(--green)':'var(--red)')+'">'+( d.milvus==='ok'?'●':'✕')+'</div><div class="sys-sub">'+(cs.entity_count??0)+' chunks</div></div><div class="system-card"><div class="sys-name">EMBEDDER</div><div class="sys-val" style="color:'+(d.embedder==='loaded'?'var(--green)':'var(--amber)')+'">'+escHtml(d.embedder)+'</div></div><div class="system-card"><div class="sys-name">GRAPH NODES</div><div class="sys-val">'+(gs.nodes??'—')+'</div><div class="sys-sub">'+(gs.edges??0)+' edges · '+escHtml(gs.extractor??'—')+'</div></div><div class="system-card"><div class="sys-name">COLLECTION</div><div class="sys-val" style="font-size:13px">'+escHtml(cs.collection??'—')+'</div></div><div class="system-card"><div class="sys-name">CHUNKS</div><div class="sys-val" style="color:var(--amber)">'+(cs.entity_count??0)+'</div></div></div><div class="json-block">'+escHtml(JSON.stringify(d,null,2))+'</div>';
  }catch(err){body.innerHTML='<div style="color:var(--red);font-family:var(--mono);font-size:12px;padding:40px 0;text-align:center">Error: '+escHtml(err.message)+'</div>';}
}