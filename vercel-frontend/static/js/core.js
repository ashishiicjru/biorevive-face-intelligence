/* Bio-Revive 360 Vision Suite - shared core */
'use strict';
const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const esc = s => String(s ?? '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const clamp = (x, a, b) => Math.max(a, Math.min(b, x));
const fmt = (x, d = 1) => (x == null || Number.isNaN(x)) ? '—' : Number(x).toFixed(d);

const ICON = {
  home: '<path d="M3 11l9-8 9 8v9a1 1 0 0 1-1 1h-5v-6H9v6H4a1 1 0 0 1-1-1z"/>',
  smile: '<circle cx="12" cy="12" r="9"/><path d="M8 14s1.5 2.5 4 2.5S16 14 16 14"/><path d="M9 9.5h.01M15 9.5h.01"/>',
  phone: '<rect x="7" y="2" width="10" height="20" rx="2.5"/><path d="M11 18h2"/>',
  book: '<path d="M4 5a2 2 0 0 1 2-2h13v16H6a2 2 0 0 0-2 2z"/><path d="M4 19V5"/><path d="M9 8h6"/>',
  layers: '<path d="M12 3l9 5-9 5-9-5z"/><path d="M3 13l9 5 9-5"/>',
  dna: '<path d="M7 3c0 6 10 6 10 12M17 3c0 6-10 6-10 12"/><path d="M9 7h6M9 17h6M7 21c0-3 2-4 5-4s5 1 5 4"/>',
  chart: '<path d="M4 20V10M10 20V4M16 20v-7M22 20H2"/>',
  cpu: '<rect x="6" y="6" width="12" height="12" rx="2"/><path d="M9 2v3M15 2v3M9 19v3M15 19v3M2 9h3M2 15h3M19 9h3M19 15h3"/>',
  info: '<circle cx="12" cy="12" r="9"/><path d="M12 11v6M12 7.5h.01"/>',
  cube: '<path d="M12 2l9 5v10l-9 5-9-5V7z"/><path d="M12 12l9-5M12 12L3 7M12 12v10"/>',
  cam: '<path d="M4 7h3l2-3h6l2 3h3a1 1 0 0 1 1 1v11a1 1 0 0 1-1 1H4a1 1 0 0 1-1-1V8a1 1 0 0 1 1-1z"/><circle cx="12" cy="13" r="3.5"/>',
  stop: '<rect x="6" y="6" width="12" height="12" rx="2"/>',
  save: '<path d="M5 3h11l3 3v15H5z"/><path d="M8 3v6h7V3M8 21v-7h8v7"/>',
  warn: '<path d="M12 3l10 18H2z"/><path d="M12 10v5M12 18h.01"/>',
  bolt: '<path d="M13 2L4 14h7l-1 8 9-12h-7z"/>',
  shield: '<path d="M12 2l8 3v6c0 5-3.5 9-8 11-4.5-2-8-6-8-11V5z"/><path d="M9 12l2 2 4-4"/>',
};
const svgi = (n, sw = 1.8) => `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="${sw}" stroke-linecap="round" stroke-linejoin="round">${ICON[n] || ''}</svg>`;

const NAV = [
  ['/', 'home', 'Home', 'Overview'],
  ['/expression', 'smile', 'Expression Lab', 'Vision labs'],
  ['/phone', 'phone', 'Phone Detector', ''],
  ['/study', 'book', 'Study Focus', ''],
  ['/attributes', 'layers', 'Age · Gender · Texture', ''],
  ['/biosuite', 'dna', 'Bio-Suite', 'Bio-Revive'],
  ['/dashboard', 'chart', 'Dashboard', 'Insights'],
  ['/models', 'cpu', 'Model Manager', ''],
  ['/about', 'info', 'About & Remarks', ''],
];

function mountShell() {
  const b = document.body, path = location.pathname.replace(/\/$/, '') || '/';
  const title = b.dataset.title || 'Bio-Revive 360', sub = b.dataset.sub || '';
  let last = '';
  const links = NAV.map(([href, ic, label, sep]) => {
    const head = sep && sep !== last ? `<div class="sep">${sep}</div>` : ''; if (sep) last = sep;
    return `${head}<a href="${href}" class="${href === path ? 'on' : ''}">${svgi(ic)}<span>${label}</span></a>`;
  }).join('');
  const side = document.createElement('aside');
  side.className = 'side'; side.id = 'side';
  side.innerHTML = `<div class="brand"><div class="logo">${svgi('cube', 2)}</div><div><b>Bio-Revive 360</b><span>Face Intelligence Suite · v3</span></div></div>
    <nav class="nav">${links}</nav>
    <div class="foot"><div class="row" style="gap:8px"><span class="status-dot" id="sysDot"></span><span id="sysTxt">Checking models…</span></div>
    <div style="margin-top:8px">Prototype · not a medical device</div></div>`;
  const main = document.createElement('div'); main.className = 'main';
  const content = document.createElement('div'); content.className = 'content';
  while (b.firstChild) content.appendChild(b.firstChild);
  const top = document.createElement('header'); top.className = 'top';
  top.innerHTML = `<button class="burger" id="burger" aria-label="menu">☰</button><div><h1>${esc(title)}</h1><div class="sub">${esc(sub)}</div></div><div class="grow"></div><span class="chip warn" title="Read the Remarks page">${svgi('shield', 2).replace('<svg', '<svg width="14" height="14"')} Not medical advice</span>`;
  const bar = document.createElement('div'); bar.className = 'disclaimer-bar'; bar.style.marginTop = '16px';
  bar.innerHTML = `${svgi('warn').replace('<svg', '<svg width="18" height="18"')}<span><b>Research prototype.</b> Outputs are estimates from visual cues and are <b>not medical, psychological or diagnostic</b> advice. See <a href="/about">About &amp; Remarks</a>.</span>`;
  const foot = document.createElement('footer'); foot.className = 'page-foot';
  foot.innerHTML = 'Bio-Revive 360 Vision Suite · runs on your own machine · camera frames are analysed locally and are not stored unless you save a snapshot.';
  main.append(top, bar, content, foot);
  const app = document.createElement('div'); app.className = 'app'; app.append(side, main);
  b.appendChild(app);
  const toast = document.createElement('div'); toast.className = 'toast-host'; toast.id = 'toasts'; b.appendChild(toast);
  $('#burger').onclick = () => side.classList.toggle('open');
  content.addEventListener('click', () => side.classList.remove('open'));
  sysStatus();
}

async function api(path, opts) {
  const r = await fetch(path, opts);
  let d = null; try { d = await r.json(); } catch (e) { }
  if (!r.ok) throw new Error((d && (d.detail?.[0]?.msg || d.detail)) || r.statusText || 'Request failed');
  return d;
}
function toast(msg, kind = '') {
  const t = document.createElement('div'); t.className = 'toast ' + kind; t.textContent = msg;
  $('#toasts').appendChild(t); setTimeout(() => t.remove(), 4800);
}
async function sysStatus() {
  try {
    const c = await api('/api/vision/capabilities');
    const n = ['expression', 'phone', 'age'].filter(k => c[k]).length;
    $('#sysDot').className = 'status-dot ' + (n === 3 ? 'ok' : n ? 'warn' : 'bad');
    $('#sysTxt').textContent = n === 3 ? 'Core models ready' : `${n}/3 core models installed`;
    window.CAPS = c; document.dispatchEvent(new CustomEvent('caps', { detail: c }));
  } catch (e) { $('#sysDot').className = 'status-dot bad'; $('#sysTxt').textContent = 'Server unreachable'; }
}
function installBanner(el, caps, needed) {
  const miss = needed.filter(k => !caps[k]);
  if (!miss.length) { el.innerHTML = ''; return; }
  el.innerHTML = `<div class="note bad">Missing model${miss.length > 1 ? 's' : ''}: <b>${miss.join(', ')}</b>. Open the <a href="/models">Model Manager</a> and click “Install recommended”, or run <code>python -m app.download_models</code>.</div>`;
}

/* ---------- small renderers ---------- */
const EMO = {
  Happiness: ['😄', '#34d399', 'Happy'], Neutral: ['😐', '#8fa3c4', 'Neutral'], Sadness: ['😢', '#4f8cff', 'Sad'],
  Anger: ['😠', '#fb7185', 'Angry'], Surprise: ['😮', '#fbbf24', 'Surprised'], Fear: ['😨', '#8b5cf6', 'Fearful'],
  Disgust: ['🤢', '#a3e635', 'Disgusted'], Contempt: ['😒', '#f472b6', 'Contempt'],
};
function renderBars(el, probs, opts = {}) {
  const ent = Object.entries(probs).sort((a, b) => b[1] - a[1]);
  el.innerHTML = '<div class="bars">' + ent.map(([k, v], i) =>
    `<div class="bar ${i === 0 ? 'top' : ''}"><span>${esc(opts.names?.[k] || EMO[k]?.[2] || k)}</span><div class="t"><i style="width:${(v * 100).toFixed(1)}%;${opts.colors && EMO[k] ? `background:${EMO[k][1]}` : ''}"></i></div><span>${(v * 100).toFixed(0)}%</span></div>`).join('') + '</div>';
}
function renderGauge(el, score, color) {
  const r = 78, c = 2 * Math.PI * r, v = clamp(score || 0, 0, 100);
  el.innerHTML = `<svg width="190" height="190" viewBox="0 0 190 190"><defs><linearGradient id="gg" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#22d3ee"/><stop offset="1" stop-color="#8b5cf6"/></linearGradient></defs>
  <circle cx="95" cy="95" r="${r}" fill="none" stroke="rgba(255,255,255,.08)" stroke-width="14"/>
  <circle cx="95" cy="95" r="${r}" fill="none" stroke="${color || 'url(#gg)'}" stroke-width="14" stroke-linecap="round" stroke-dasharray="${c}" stroke-dashoffset="${c * (1 - v / 100)}" style="transition:stroke-dashoffset .6s,stroke .4s"/></svg>
  <div class="v"><div><b>${score == null ? '—' : Math.round(v)}</b><small>attention proxy</small></div></div>`;
}
const stateClass = s => 's-' + String(s || '').split(' ')[0];
const stateColor = s => ({ Focused: '#34d399', 'Partially attentive': '#fbbf24', Distracted: '#fb7185', Drowsy: '#8b5cf6', 'Phone in use': '#fb7185', Away: '#8fa3c4' }[s] || '#8fa3c4');

function beep(freq = 880, ms = 160, vol = .08) {
  try {
    const a = beep.ctx || (beep.ctx = new (window.AudioContext || window.webkitAudioContext)());
    const o = a.createOscillator(), g = a.createGain(); o.frequency.value = freq; o.type = 'sine'; g.gain.value = vol;
    o.connect(g); g.connect(a.destination); o.start(); o.stop(a.currentTime + ms / 1000);
  } catch (e) { }
}

function tabs(root) {
  const btns = $$('.tabs button', root), panes = $$('.pane', root);
  btns.forEach(b => b.onclick = () => {
    btns.forEach(x => x.classList.toggle('on', x === b)); panes.forEach(p => p.classList.toggle('on', p.id === b.dataset.pane));
    document.dispatchEvent(new CustomEvent('tab', { detail: b.dataset.pane }));
  });
}

/* ---------- MediaPipe landmarker (vendored, optional model download) ---------- */
let _lmPromise = null;
async function loadLandmarker(setStatus) {
  if (_lmPromise) return _lmPromise;
  _lmPromise = (async () => {
    try {
      setStatus?.('Loading face landmarker…');
      const mod = await import('/static/vendor/mediapipe/vision_bundle.mjs');
      const fs = await mod.FilesetResolver.forVisionTasks('/static/vendor/mediapipe/wasm');
      let local = false; try { local = (await fetch('/models/face_landmarker.task', { method: 'HEAD' })).ok; } catch (e) { }
      const path = local ? '/models/face_landmarker.task' : 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task';
      let lm;
      for (const delegate of ['GPU', 'CPU']) {
        try { lm = await mod.FaceLandmarker.createFromOptions(fs, { baseOptions: { modelAssetPath: path, delegate }, runningMode: 'VIDEO', numFaces: 1, outputFaceBlendshapes: true }); break; } catch (e) { if (delegate === 'CPU') throw e; }
      }
      setStatus?.('Landmarker ready (' + (local ? 'local model' : 'CDN model') + ')');
      return lm;
    } catch (e) { console.warn('landmarker unavailable', e); setStatus?.('Landmarker unavailable → limited mode (server-side detection only)'); return null; }
  })();
  return _lmPromise;
}

let CONN = null;
async function loadConnections() { if (!CONN) CONN = await (await fetch('/static/data/face_connections.json')).json(); return CONN; }

/* ---------- VisionRig: camera + landmarks + throttled server analysis ---------- */
class VisionRig {
  constructor(o) {
    Object.assign(this, { intervalMs: 800, modules: ['expression'], phoneSize: 608, phoneConf: .4, maxW: 640, useLandmarks: true, session: null, onResult: () => { }, onStatus: () => { }, onFrame: null, draw: {} }, o);
    this.running = false; this.inflight = false; this.lastSent = 0; this.lm = null; this.lastLm = null; this.lastRes = null; this.fps = 0;
    this.cv = document.createElement('canvas');
  }
  async start() {
    this.stream = await navigator.mediaDevices.getUserMedia({ video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: 'user' }, audio: false });
    this.video.srcObject = this.stream; await this.video.play();
    this.ov.width = this.video.videoWidth; this.ov.height = this.video.videoHeight;
    this.running = true;
    if (this.useLandmarks) loadLandmarker(this.onStatus).then(l => { this.lm = l; });
    loadConnections();
    this._raf = requestAnimationFrame(t => this.loop(t));
  }
  stop() {
    this.running = false; cancelAnimationFrame(this._raf);
    this.stream?.getTracks().forEach(t => t.stop()); this.video.srcObject = null;
    const c = this.ov.getContext('2d'); c.clearRect(0, 0, this.ov.width, this.ov.height);
  }
  loop(now) {
    if (!this.running) return;
    const v = this.video;
    if (v.readyState >= 2) {
      if (this.lm) {
        try {
          const r = this.lm.detectForVideo(v, now);
          this.lastLm = r.faceLandmarks?.[0] || null; this.lastBlend = r.faceBlendshapes?.[0]?.categories || null;
        } catch (e) { this.lastLm = null; }
      }
      this.render();
      if (!this.inflight && !this.paused && now - this.lastSent >= this.intervalMs) this.send(now);
    }
    this._raf = requestAnimationFrame(t => this.loop(t));
  }
  frameBlob(q = .82) {
    const v = this.video, s = Math.min(1, this.maxW / v.videoWidth);
    this.cv.width = Math.round(v.videoWidth * s); this.cv.height = Math.round(v.videoHeight * s);
    this.cv.getContext('2d').drawImage(v, 0, 0, this.cv.width, this.cv.height);
    return new Promise(res => this.cv.toBlob(res, 'image/jpeg', q));
  }
  async send(now) {
    this.inflight = true; this.lastSent = now;
    const t0 = performance.now();
    try {
      const blob = await this.frameBlob();
      const fd = new FormData(); fd.append('image', blob, 'f.jpg');
      fd.append('modules', this.modules.join(','));
      fd.append('phone_size', this.phoneSize); fd.append('phone_conf', this.phoneConf); if (this.allObjects) fd.append('all_objects', 'true');
      if (this.session) fd.append('session_id', this.session);
      if (this.lastLm) fd.append('landmarks', JSON.stringify(this.lastLm.map(p => [+p.x.toFixed(5), +p.y.toFixed(5), +p.z.toFixed(5)])));
      const res = await api('/api/vision/analyze', { method: 'POST', body: fd });
      this.lastRes = res; this.latency = performance.now() - t0;
      this.onResult(res, this.lastLm);
    } catch (e) { this.onStatus('Analysis error: ' + e.message); }
    this.inflight = false;
  }
  render() {
    const c = this.ov, g = c.getContext('2d'), W = c.width, H = c.height;
    g.clearRect(0, 0, W, H);
    const d = this.draw, res = this.lastRes;
    if (d.mesh !== false && this.lastLm && CONN) drawMesh(g, this.lastLm, W, H, d.tess);
    if (res) drawResult(g, res, W, H, d);
    this.onFrame?.(g, W, H);
  }
  async snapshot(extra = {}) {
    const c = document.createElement('canvas'), v = this.video; c.width = v.videoWidth; c.height = v.videoHeight;
    c.getContext('2d').drawImage(v, 0, 0);
    const blob = await new Promise(r => c.toBlob(r, 'image/jpeg', .9));
    const fd = new FormData(); fd.append('image', blob, 's.jpg'); fd.append('source', this.snapSource || 'live');
    fd.append('caption', extra.caption || ''); fd.append('analysis', JSON.stringify(extra.analysis || {}));
    return api('/api/snapshots', { method: 'POST', body: fd });
  }
}

function drawMesh(g, lm, W, H, tess) {
  const mx = p => (1 - p.x) * W, my = p => p.y * H;
  g.lineWidth = 1;
  if (tess) {
    g.strokeStyle = 'rgba(34,211,238,.22)'; g.beginPath();
    for (const [a, b] of CONN.FACE_LANDMARKS_TESSELATION) { g.moveTo(mx(lm[a]), my(lm[a])); g.lineTo(mx(lm[b]), my(lm[b])); }
    g.stroke();
  }
  g.strokeStyle = 'rgba(139,92,246,.9)'; g.lineWidth = 1.6; g.beginPath();
  for (const k of ['FACE_LANDMARKS_FACE_OVAL', 'FACE_LANDMARKS_LIPS', 'FACE_LANDMARKS_LEFT_EYE', 'FACE_LANDMARKS_RIGHT_EYE', 'FACE_LANDMARKS_LEFT_EYEBROW', 'FACE_LANDMARKS_RIGHT_EYEBROW'])
    for (const [a, b] of CONN[k]) { g.moveTo(mx(lm[a]), my(lm[a])); g.lineTo(mx(lm[b]), my(lm[b])); }
  g.stroke();
  g.fillStyle = '#22d3ee';
  if (lm.length > 477) for (const i of [468, 473]) { g.beginPath(); g.arc(mx(lm[i]), my(lm[i]), 2.4, 0, 7); g.fill(); }
}

function cornerBox(g, x, y, w, h, col, label, sub) {
  const L = Math.min(w, h) * .22; g.strokeStyle = col; g.lineWidth = 3; g.lineCap = 'round'; g.beginPath();
  for (const [px, py, dx, dy] of [[x, y, 1, 1], [x + w, y, -1, 1], [x, y + h, 1, -1], [x + w, y + h, -1, -1]]) { g.moveTo(px + dx * L, py); g.lineTo(px, py); g.lineTo(px, py + dy * L); }
  g.stroke();
  if (label) {
    g.font = '600 15px Inter,Segoe UI,sans-serif';
    const t = label + (sub ? '  ' + sub : ''), tw = g.measureText(t).width + 16, ty = y - 30 < 0 ? y + h + 4 : y - 30;
    g.fillStyle = 'rgba(3,8,18,.82)'; g.beginPath(); g.roundRect ? g.roundRect(x, ty, tw, 26, 8) : g.rect(x, ty, tw, 26); g.fill();
    g.fillStyle = col; g.fillText(t, x + 8, ty + 18);
  }
}
function drawResult(g, res, W, H, d) {
  const mirror = d.mirror !== false;
  const bx = b => mirror ? [(1 - b[0] - b[2]) * W, b[1] * H, b[2] * W, b[3] * H] : [b[0] * W, b[1] * H, b[2] * W, b[3] * H];
  if (d.faces !== false) res.faces.forEach(f => {
    const e = f.expression, col = e ? (EMO[e.label]?.[1] || '#22d3ee') : '#22d3ee';
    const [x, y, w, h] = bx(f.box);
    let label = '', sub = '';
    if (d.labels !== false) {
      if (e) { label = (EMO[e.label]?.[0] || '') + ' ' + (EMO[e.label]?.[2] || e.label); sub = Math.round(e.confidence * 100) + '%'; }
      if (f.age) label += (label ? '  ·  ' : '') + '~' + Math.round(f.age.apparent_age) + 'y';
      if (f.gender) label += '  ·  ' + f.gender.label.replace('-presenting', '');
    }
    cornerBox(g, x, y, w, h, col, label, sub);
  });
  if (d.phones !== false) res.phones.forEach(p => {
    const [x, y, w, h] = bx(p.box_norm); const col = p.is_phone ? '#fb7185' : '#fbbf24';
    g.strokeStyle = col; g.lineWidth = 3; g.strokeRect(x, y, w, h);
    g.font = '700 14px Inter,sans-serif'; const t = (p.is_phone ? '📱 phone ' : p.label + ' ') + Math.round(p.confidence * 100) + '%';
    g.fillStyle = col; g.fillRect(x, y - 24, g.measureText(t).width + 14, 24); g.fillStyle = '#1b0508'; g.fillText(t, x + 7, y - 7);
  });
}

/* ---------- utilities ---------- */
function downloadText(name, text, mime = 'text/plain') {
  const a = document.createElement('a'); a.href = URL.createObjectURL(new Blob([text], { type: mime })); a.download = name; a.click(); setTimeout(() => URL.revokeObjectURL(a.href), 3000);
}
function tiltCards() {
  $$('.tilt').forEach(el => {
    el.addEventListener('mousemove', e => { const r = el.getBoundingClientRect(), x = (e.clientX - r.left) / r.width - .5, y = (e.clientY - r.top) / r.height - .5; el.style.transform = `perspective(900px) rotateY(${x * 12}deg) rotateX(${-y * 12}deg) translateZ(6px)`; });
    el.addEventListener('mouseleave', () => el.style.transform = '');
  });
}
function fileToImage(file) { return new Promise((res, rej) => { const i = new Image(); i.onload = () => res(i); i.onerror = rej; i.src = URL.createObjectURL(file); }); }
function modal(html) {
  let m = $('#modalBg'); if (!m) { m = document.createElement('div'); m.id = 'modalBg'; m.className = 'modal-bg'; m.innerHTML = '<div class="card modal" id="modalBody"></div>'; document.body.appendChild(m); m.onclick = e => { if (e.target === m) m.classList.remove('on'); }; }
  $('#modalBody').innerHTML = html; m.classList.add('on'); return m;
}
document.addEventListener('DOMContentLoaded', () => { mountShell(); tiltCards(); });

/* ---------- Chart.js theme ---------- */
function chartTheme() {
  if (!window.Chart) return;
  Chart.defaults.color = '#8fa3c4'; Chart.defaults.borderColor = 'rgba(160,190,255,.12)';
  Chart.defaults.font.family = "Inter,'Segoe UI',system-ui,sans-serif"; Chart.defaults.animation.duration = 350;
  Chart.defaults.plugins.legend.labels.boxWidth = 10; Chart.defaults.plugins.legend.labels.usePointStyle = true;
}
const PALETTE = ['#22d3ee', '#4f8cff', '#8b5cf6', '#f472b6', '#fbbf24', '#34d399', '#fb7185', '#a3e635'];
document.addEventListener('DOMContentLoaded', chartTheme);
async function postImage(path, file, fields = {}) {
  const fd = new FormData(); fd.append('image', file, file.name || 'image.jpg');
  for (const [k, v] of Object.entries(fields)) fd.append(k, v);
  return api(path, { method: 'POST', body: fd });
}
