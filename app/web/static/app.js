/* 知新 DayOne · 前端交互
 * 纯原生 JS，无构建步骤：打开页面就能用，方便内网部署。 */

const state = {
  // 演示身份：真实部署从登录态取。改一次会记住，刷新不丢
  ownerId: localStorage.getItem('dayone-owner') || 'emp-9',
  sessionId: null,
  sources: [],
  weak: new Set(),
  answerText: '',
  busy: false,
  lastQuestion: '',
  apiKey: localStorage.getItem('kb_api_key') || '',
  topK: Number(localStorage.getItem('kb_topk') || 5),
};

const $ = (id) => document.getElementById(id);
const messagesEl = $('messages');
const sourcesEl = $('sources');
const inputEl = $('input');
const sendBtn = $('send');

/* ---------------- 基础工具 ---------------- */
function esc(s) {
  return String(s == null ? '' : s)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

/* 片段是纯文本展示：去掉 markdown 强调符号，并把表格行压成可读的一行 */
function plain(text) {
  return String(text || '')
    .replace(/\*\*|__|`/g, '')
    .replace(/^\s{0,3}#{1,6}\s*/gm, '')
    .split('\n')
    .filter((line) => !/^\s*\|?[\s:|-]{3,}\|?\s*$/.test(line)) // 丢掉 |---|---| 分隔行
    .join('\n')
    .replace(/[ \t]*\|[ \t]*/g, ' · ')
    .replace(/(^|\n)[ \t]*·[ \t]*/g, '$1')
    .replace(/[ \t]*·[ \t]*$/gm, '')
    .trim();
}

function headers() {
  const h = { 'Content-Type': 'application/json' };
  if (state.apiKey) h['X-API-Key'] = state.apiKey;
  return h;
}

/* ---------------- Markdown 轻量渲染 ---------------- */
function renderMarkdown(text, weakSet) {
  const lines = String(text || '').split('\n');
  let html = '';
  let listType = null;

  const closeList = () => {
    if (listType) { html += `</${listType}>`; listType = null; }
  };

  const inline = (s) => {
    let out = esc(s);
    // 代码
    out = out.replace(/`([^`]+)`/g, '<code>$1</code>');
    // 粗体
    out = out.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
    // 引用标记 [[1]]
    out = out.replace(/\[\[(\d{1,2})\]\]/g, (_, n) => {
      const cls = weakSet && weakSet.has(Number(n)) ? 'cite weak' : 'cite';
      return `<span class="${cls}" data-cite="${n}" role="button" tabindex="0">${n}</span>`;
    });
    return out;
  };

  for (const raw of lines) {
    const line = raw.replace(/\s+$/, '');
    if (!line.trim()) { closeList(); continue; }
    const ol = line.match(/^\s*(\d+)[.、)]\s+(.*)$/);
    const ul = line.match(/^\s*[-*•]\s+(.*)$/);
    if (ul) {
      if (listType !== 'ul') { closeList(); html += '<ul>'; listType = 'ul'; }
      html += `<li>${inline(ul[1])}</li>`;
    } else if (ol) {
      if (listType !== 'ol') { closeList(); html += '<ol>'; listType = 'ol'; }
      html += `<li>${inline(ol[2])}</li>`;
    } else {
      closeList();
      html += `<p>${inline(line)}</p>`;
    }
  }
  closeList();
  return html;
}

/* ---------------- 消息气泡 ---------------- */
function addUserMessage(text) {
  const el = document.createElement('div');
  el.className = 'msg user';
  el.innerHTML = `<div class="avatar">你</div><div class="bubble">${renderMarkdown(text)}</div>`;
  messagesEl.appendChild(el);
  scrollToBottom();
}

function addBotShell() {
  const el = document.createElement('div');
  el.className = 'msg bot';
  el.innerHTML = `
    <div class="avatar">知</div>
    <div class="bubble">
      <div class="thinking"><span class="spinner"></span><span class="thinking-text">正在检索知识库…</span></div>
      <div class="answer" hidden></div>
      <div class="footnote" hidden></div>
    </div>`;
  messagesEl.appendChild(el);
  scrollToBottom();
  return {
    root: el,
    bubble: el.querySelector('.bubble'),
    thinking: el.querySelector('.thinking'),
    thinkingText: el.querySelector('.thinking-text'),
    answer: el.querySelector('.answer'),
    footnote: el.querySelector('.footnote'),
  };
}

function scrollToBottom() {
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

/* ---------------- 来源面板 ---------------- */
const STOP = new Set(['什么', '怎么', '如何', '哪些', '可以', '需要', '请问', '应该', '多少']);

function bigrams(text) {
  const runs = String(text).match(/[\u4e00-\u9fff]+/g) || [];
  const out = new Set();
  for (const run of runs) {
    for (let i = 0; i < run.length - 1; i++) {
      const g = run.slice(i, i + 2);
      if (!STOP.has(g)) out.add(g);
    }
  }
  return [...out];
}

function escRe(s) { return s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'); }

function highlightSnippet(text, terms) {
  const hits = terms.filter((t) => text.includes(t)).sort((a, b) => b.length - a.length).slice(0, 10);
  if (!hits.length) return esc(text);
  const re = new RegExp(`(${hits.map(escRe).join('|')})`, 'g');
  let out = '';
  let last = 0;
  text.replace(re, (m, _g, idx) => {
    out += esc(text.slice(last, idx)) + `<mark>${esc(m)}</mark>`;
    last = idx + m.length;
    return m;
  });
  out += esc(text.slice(last));
  return out;
}

function renderSources(sources, meta) {
  state.sources = sources || [];
  $('source-count').textContent = state.sources.length;

  const metaEl = $('retrieval-meta');
  metaEl.innerHTML = '';
  if (meta) {
    // 编排链路没有"检索置信度"这回事（它可能是查档案/系统），别摆两个 0.00 误导人
    const chips = meta.answer_source === 'orchestrated'
      ? []
      : [`置信度 ${(meta.confidence ?? 0).toFixed(2)}`,
         `覆盖度 ${(meta.coverage ?? 0).toFixed(2)}`];
    if (meta.standalone_query) chips.push(`检索用问题：${meta.standalone_query}`);
    metaEl.innerHTML = chips.map((c) => `<span>${esc(c)}</span>`).join('');
  }

  if (!state.sources.length) {
    const src = (meta && meta.answer_source) || 'knowledge_base';
    const why = src === 'tool'
      ? '这次由系统查询作答（工单 / 人员 / 流程），没有检索文档。'
      : src === 'memory'
        ? '这次由你的入职档案作答，没有检索文档。'
        : src === 'orchestrated'
          ? '这次任务只用到档案与系统数据，没有引用文档。'
          : '这次没有召回到相关文档。';
    metaEl.innerHTML = '';
    sourcesEl.innerHTML = `<div class="empty"><p>${esc(why)}</p></div>`;
    return;
  }

  sourcesEl.innerHTML = state.sources
    .map((s) => `
      <div class="source-card ${s.used ? 'used' : 'unused'}" data-n="${s.n}">
        <div class="source-head">
          <span class="source-num">${s.n}</span>
          <span class="source-title" title="${esc(s.doc_title)}">${esc(s.doc_title)}</span>
          <span class="source-score">${(s.score ?? 0).toFixed(2)}</span>
        </div>
        ${s.heading ? `<div class="source-heading">${esc(s.heading)}</div>` : ''}
        <div class="source-snippet">${esc(plain(s.snippet))}</div>
        <div class="source-foot">
          <span class="tag ${s.used ? 'used' : ''}">${s.used ? `被引用 ${s.hits || 1} 次` : '未使用'}</span>
          <span class="tag">${esc(s.doc_id || '')}</span>
        </div>
      </div>`)
    .join('');

  sourcesEl.querySelectorAll('.source-card').forEach((card) => {
    card.addEventListener('click', () => activateSource(Number(card.dataset.n)));
  });
}

function activateSource(n, terms) {
  const card = sourcesEl.querySelector(`.source-card[data-n="${n}"]`);
  if (!card) return;
  sourcesEl.querySelectorAll('.source-card').forEach((c) => c.classList.remove('active'));
  card.classList.add('active');

  const snippetEl = card.querySelector('.source-snippet');
  const source = state.sources.find((s) => s.n === n);
  if (snippetEl && source) {
    snippetEl.innerHTML = highlightSnippet(plain(source.snippet), terms || bigrams(source.heading || ''));
  }
  card.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

/* 点答案里的角标 → 定位来源 + 高亮该结论涉及的字词 */
document.addEventListener('click', (e) => {
  const chip = e.target.closest('.cite');
  if (!chip) return;
  const n = Number(chip.dataset.cite);
  const sentence = chip.closest('p, li')?.textContent || state.answerText;
  activateSource(n, bigrams(sentence));
});

/* ---------------- SSE 流式请求 ---------------- */
async function streamChat(question, shell) {
  const resp = await fetch('/api/chat/stream', {
    method: 'POST',
    headers: headers(),
    body: JSON.stringify({
      question,
      session_id: state.sessionId,
      top_k: state.topK,
      owner_id: state.ownerId,
    }),
  });

  if (!resp.ok || !resp.body) {
    const err = await resp.text().catch(() => '');
    throw new Error(`HTTP ${resp.status} ${err.slice(0, 120)}`);
  }

  const reader = resp.body.getReader();
  const decoder = new TextDecoder('utf-8');
  let buffer = '';
  let answer = '';
  let donePayload = null;

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    const chunks = buffer.split('\n\n');
    buffer = chunks.pop() || '';

    for (const chunk of chunks) {
      const line = chunk.trim();
      if (!line.startsWith('data:')) continue;
      const data = line.slice(5).trim();
      if (data === '[DONE]') continue;

      let ev;
      try { ev = JSON.parse(data); } catch { continue; }

      if (ev.type === 'start') {
        state.sessionId = ev.session_id;
        $('session-hint').textContent = `会话 ${ev.session_id}`;
      } else if (ev.type === 'status') {
        shell.thinkingText.textContent = ev.message || '处理中…';
      } else if (ev.type === 'sources') {
        renderSources(ev.sources, ev);
        shell.thinkingText.textContent = ev.sufficient ? '正在生成回答…' : '没有找到可靠依据';
      } else if (ev.type === 'delta') {
        if (!answer) {
          shell.thinking.hidden = true;
          shell.answer.hidden = false;
        }
        answer += ev.text;
        state.answerText = answer;
        shell.answer.innerHTML = renderMarkdown(answer, state.weak);
        scrollToBottom();
      } else if (ev.type === 'done') {
        donePayload = ev;
      } else if (ev.type === 'error') {
        throw new Error(ev.message || '服务返回错误');
      }
    }
  }
  return { answer, payload: donePayload };
}

/* 写操作：提议只是提议，必须点确认才执行。卡片会把结果换成工单号 */
function renderConfirmCard(shell, action) {
  const card = document.createElement('div');
  card.className = 'confirm-card';
  card.innerHTML = `
    <div class="confirm-head">
      <span class="confirm-badge">待确认</span>
      <span class="confirm-hint">确认后才会提交，不会自动执行</span>
    </div>
    <div class="confirm-summary">${esc(action.summary || '')}</div>
    <div class="confirm-foot">
      <button class="confirm-btn primary" data-act="confirm">确认执行</button>
      <button class="confirm-btn" data-act="cancel">取消</button>
      <span class="confirm-exp">${esc((action.expires_at || '').slice(0, 16).replace('T', ' '))} 前有效</span>
    </div>`;
  shell.bubble.appendChild(card);

  card.addEventListener('click', async (e) => {
    const act = e.target.dataset && e.target.dataset.act;
    if (!act) return;
    const buttons = card.querySelectorAll('button');
    buttons.forEach((b) => { b.disabled = true; });

    const url = act === 'confirm' ? '/api/tools/confirm' : '/api/tools/cancel';
    try {
      const resp = await fetch(url, {
        method: 'POST',
        headers: headers(),
        body: JSON.stringify({ owner_id: state.ownerId, action_id: action.action_id }),
      });
      const data = await resp.json().catch(() => ({}));
      if (!resp.ok) {
        const msg = (data.detail && data.detail.message) || data.detail || `HTTP ${resp.status}`;
        throw new Error(typeof msg === 'string' ? msg : JSON.stringify(msg));
      }
      if (act === 'cancel') {
        card.innerHTML = '<div class="confirm-done">已取消，没有提交任何内容。</div>';
        return;
      }
      const res = (data.result && data.result.data) || {};
      const ticket = res.ticket_no ? `<b>${esc(res.ticket_no)}</b>` : '已完成';
      card.innerHTML = `<div class="confirm-done">${
        data.replayed ? '这次是重复确认，已按上次结果处理（未重复提交）' : '已提交'
      }：工单 ${ticket}${res.status ? ` · 状态 ${esc(res.status)}` : ''}</div>`;
    } catch (err) {
      card.querySelector('.confirm-foot').innerHTML =
        `<span class="confirm-error">${esc(err.message)}</span>`;
    }
  });
}

/* 工具名对用户没意义，换成业务说法 */
const TOOL_LABELS = {
  get_my_onboarding_status: '入职流程',
  lookup_colleague: '组织目录',
  get_my_tickets: '工单记录',
  submit_permission_request: '权限申请',
};

function toolLabel(name) {
  return TOOL_LABELS[name] || name || '';
}

/* 多步编排：把「它走了哪几步」摊开给用户看，而不是只给一个答案 */
const STEP_LABELS = { memory: '档案', doc: '文档', tool: '系统', propose: '申请' };

function renderPlan(shell, p) {
  const plan = (p && p.plan) || [];
  if (p.answer_source !== 'orchestrated' || !plan.length) return;
  const failed = plan.filter((x) => !x.ok).length;
  const steps = plan
    .map((x) => `<span class="plan-step ${x.ok ? '' : 'bad'}">
        <b>${esc(STEP_LABELS[x.kind] || x.kind)}</b>${esc(x.step)}
      </span>`)
    .join('<span class="plan-arrow">→</span>');
  const box = document.createElement('div');
  box.className = 'plan-box';
  box.innerHTML = `
    <div class="plan-head">执行了 ${plan.length} 步${failed ? `（${failed} 步失败）` : ''}</div>
    <div class="plan-steps">${steps}</div>`;
  shell.bubble.appendChild(box);
}

function renderFootnote(shell, p) {
  if (!p) return;
  // 工具/档案作答没有引用体系，硬塞「引用 0/0、有据率 0.00」只会让人误解为质量差
  if (p.answer_source === 'orchestrated') {
    const plan = p.plan || [];
    const failed = plan.filter((x) => !x.ok).length;
    shell.footnote.innerHTML =
      `<span>编排 <b>${plan.length}</b> 步${failed ? ` · <b>${failed}</b> 步失败` : ' · 全部成功'}</span>` +
      `<span>耗时 <b>${p.latency_ms || 0}</b> ms</span>` +
      (p.requires_confirmation ? '<span>尚未提交，等待确认</span>' : '');
    shell.footnote.hidden = false;
    return;
  }

  if (p.answer_source === 'tool' || p.answer_source === 'memory') {
    const toolName = (p.tool && p.tool.name) || '';
    const label = p.answer_source === 'memory'
      ? '来自你的入职档案'
      : `系统查询${toolName ? `：${toolLabel(toolName)}` : ''}`;
    shell.footnote.innerHTML =
      `<span>${esc(label)}</span><span>耗时 <b>${p.latency_ms || 0}</b> ms</span>` +
      (p.requires_confirmation ? '<span>尚未提交，等待确认</span>' : '');
    shell.footnote.hidden = false;
    return;
  }
  const used = (p.citations || []).filter((c) => c.used).length;
  const bits = [
    `<span>引用 <b>${used}</b>/${(p.citations || []).length} 条来源</span>`,
    `<span>有据率 <b>${(p.grounding_score ?? 0).toFixed(2)}</b></span>`,
    `<span>置信度 <b>${(p.confidence ?? 0).toFixed(2)}</b></span>`,
    `<span>耗时 <b>${p.latency_ms || 0}</b> ms</span>`,
  ];
  if (p.llm && p.llm.provider === 'mock') bits.push('<span>离线基线模式</span>');
  shell.footnote.innerHTML = bits.join('');
  shell.footnote.hidden = false;
}

/* ---------------- 发送 ---------------- */
async function send(question) {
  if (state.busy) return;
  const q = (question || inputEl.value).trim();
  if (!q) return;

  state.busy = true;
  sendBtn.disabled = true;
  state.lastQuestion = q;
  state.weak = new Set();
  inputEl.value = '';
  inputEl.style.height = 'auto';
  $('btn-feedback').hidden = true;

  const welcome = $('welcome');
  if (welcome) welcome.remove();

  addUserMessage(q);
  const shell = addBotShell();

  try {
    const { answer, payload } = await streamChat(q, shell);
    shell.thinking.hidden = true;
    shell.answer.hidden = false;

    if (payload) {
      state.weak = new Set((payload.weak_citations || []).map((w) => w.n));
      shell.answer.innerHTML = renderMarkdown(payload.answer || answer, state.weak);
      state.answerText = payload.answer || answer;
      if (payload.refused) shell.bubble.classList.add('refused');
      renderSources(payload.citations, {
        confidence: payload.confidence,
        coverage: payload.coverage,
        standalone_query: payload.standalone_query,
        answer_source: payload.answer_source,
      });
      renderFootnote(shell, payload);
      $('btn-feedback').hidden = !payload.refused;
      renderPlan(shell, payload);
      if (payload.requires_confirmation && payload.pending_action) {
        renderConfirmCard(shell, payload.pending_action);
      }
    }
    if (!answer && !(payload && payload.answer)) {
      shell.answer.innerHTML = '<p>没有拿到回答，请稍后重试。</p>';
    }
  } catch (err) {
    shell.thinking.hidden = true;
    shell.answer.hidden = false;
    shell.answer.innerHTML = `<p style="color:#c0392b">请求失败：${esc(err.message)}</p>`;
  } finally {
    state.busy = false;
    sendBtn.disabled = false;
    scrollToBottom();
    inputEl.focus();
  }
}

/* ---------------- 事件绑定 ---------------- */
sendBtn.addEventListener('click', () => send());

inputEl.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    send();
  }
});

inputEl.addEventListener('input', () => {
  inputEl.style.height = 'auto';
  inputEl.style.height = `${Math.min(inputEl.scrollHeight, 160)}px`;
});

document.querySelectorAll('#suggestions button').forEach((btn) => {
  btn.addEventListener('click', () => send(btn.dataset.q));
});

$('btn-new').addEventListener('click', async () => {
  if (state.sessionId) {
    fetch('/api/chat/reset', {
      method: 'POST',
      headers: headers(),
      body: JSON.stringify({ session_id: state.sessionId }),
    }).catch(() => {});
  }
  state.sessionId = null;
  $('session-hint').textContent = '新会话';
  $('source-count').textContent = '0';
  $('retrieval-meta').innerHTML = '';
  sourcesEl.innerHTML = `<div class="empty"><p>回答生成后会在这里列出参考的文档片段。</p></div>`;
  $('btn-feedback').hidden = true;
  location.reload();
});

$('btn-feedback').addEventListener('click', async () => {
  const r = await fetch('/api/feedback', {
    method: 'POST',
    headers: headers(),
    body: JSON.stringify({ question: state.lastQuestion, session_id: state.sessionId }),
  });
  if (r.ok) {
    $('btn-feedback').textContent = '已记录，谢谢反馈 ✓';
    $('btn-feedback').disabled = true;
  }
});

/* 设置 */
const dlg = $('settings-dialog');
$('btn-settings').addEventListener('click', () => {
  $('set-apikey').value = state.apiKey;
  $('set-topk').value = state.topK;
  dlg.showModal();
});
dlg.addEventListener('close', () => {
  if (dlg.returnValue !== 'ok') return;
  state.apiKey = $('set-apikey').value.trim();
  state.topK = Math.max(1, Math.min(20, Number($('set-topk').value) || 5));
  localStorage.setItem('kb_api_key', state.apiKey);
  localStorage.setItem('kb_topk', String(state.topK));
  checkHealth();
});

/* ---------------- 健康检查 ---------------- */
async function checkHealth() {
  const dot = $('status-dot');
  const text = $('status-text');
  try {
    const r = await fetch('/api/health', { headers: headers() });
    const j = await r.json();
    const kb = j.knowledge_base || {};
    const degraded = j.llm?.degraded || j.embedding?.degraded;
    dot.className = `dot ${degraded ? 'warn' : 'ok'}`;
    text.textContent = `${kb.documents || 0} 篇 / ${kb.chunks || 0} 段 · ${j.llm?.model || '-'}`;
    text.title = degraded
      ? '当前使用离线兜底模型（未配置 API Key），效果仅供联调验证'
      : `LLM: ${j.llm?.provider}/${j.llm?.model} | Embedding: ${j.embedding?.provider}`;
  } catch (e) {
    dot.className = 'dot err';
    text.textContent = '服务不可用';
  }
}

checkHealth();
inputEl.focus();

/* 支持 #q=问题 的深链，方便把某个问题直接发给同事/截图 */
(function autoAskFromHash() {
  const m = location.hash.match(/^#q=(.+)$/);
  if (!m) return;
  const q = decodeURIComponent(m[1]);
  if (q.trim()) setTimeout(() => send(q), 300);
})();
