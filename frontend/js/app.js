/* ── Config ──────────────────────────────────────────────────────────────── */
const API = 'http://localhost:8000';

/* ── DOM refs ────────────────────────────────────────────────────────────── */
const $  = id => document.getElementById(id);
const dropZone      = $('drop-zone');
const fileInput     = $('file-input');
const previewBox    = $('preview-box');
const previewImg    = $('preview-img');
const previewMeta   = $('preview-meta');
const uploadPanel   = $('upload-panel');
const resultsPanel  = $('results-panel');
const loadingOverlay= $('loading-overlay');
const ttaSlider     = $('tta-slider');
const ttaVal        = $('tta-val');
const statusPill    = $('status-pill');

/* ── State ───────────────────────────────────────────────────────────────── */
let selectedFile = null;

/* ── Health check ────────────────────────────────────────────────────────── */
async function checkHealth() {
  try {
    const r = await fetch(`${API}/health`, { signal: AbortSignal.timeout(3000) });
    const d = await r.json();
    if (d.model_loaded) {
      statusPill.textContent = '● Model Ready';
      statusPill.className   = 'pill pill--green';
    } else {
      statusPill.textContent = '● No Checkpoint';
      statusPill.className   = 'pill pill--warn';
      statusPill.style.color  = '#fbbf24';
    }
  } catch {
    statusPill.textContent = '● API Offline';
    statusPill.className   = 'pill';
    statusPill.style.cssText = 'background:rgba(248,113,113,.12);color:#f87171;border:1px solid rgba(248,113,113,.25);font-size:11px;font-weight:600;padding:4px 10px;border-radius:999px;font-family:var(--mono)';
  }
}
checkHealth();
setInterval(checkHealth, 8000);

/* ── TTA slider ──────────────────────────────────────────────────────────── */
ttaSlider.addEventListener('input', () => {
  ttaVal.textContent = ttaSlider.value;
});

/* ── Drag-and-drop ───────────────────────────────────────────────────────── */
['dragenter', 'dragover'].forEach(e =>
  dropZone.addEventListener(e, ev => {
    ev.preventDefault();
    dropZone.classList.add('drag-over');
  })
);
['dragleave', 'drop'].forEach(e =>
  dropZone.addEventListener(e, ev => {
    ev.preventDefault();
    dropZone.classList.remove('drag-over');
  })
);
dropZone.addEventListener('drop', ev => {
  const file = ev.dataTransfer.files[0];
  if (file) handleFile(file);
});
dropZone.addEventListener('click', () => fileInput.click());

fileInput.addEventListener('change', () => {
  if (fileInput.files[0]) handleFile(fileInput.files[0]);
});

/* ── File handling ───────────────────────────────────────────────────────── */
function handleFile(file) {
  if (!file.type.match(/image\/(jpeg|png)/)) {
    alert('Please upload a JPG or PNG image.');
    return;
  }
  selectedFile = file;
  const url    = URL.createObjectURL(file);
  previewImg.src = url;
  previewMeta.textContent =
    `${file.name} · ${(file.size / 1024).toFixed(1)} KB`;

  dropZone.classList.add('hidden');
  previewBox.classList.remove('hidden');
}

/* ── Clear ───────────────────────────────────────────────────────────────── */
$('clear-btn').addEventListener('click', reset);

function reset() {
  selectedFile = null;
  fileInput.value = '';
  previewImg.src  = '';
  dropZone.classList.remove('hidden');
  previewBox.classList.add('hidden');
  resultsPanel.classList.add('hidden');
  uploadPanel.classList.remove('hidden');
}

/* ── Retry ───────────────────────────────────────────────────────────────── */
$('retry-btn').addEventListener('click', () => {
  resultsPanel.classList.add('hidden');
  uploadPanel.classList.remove('hidden');
  dropZone.classList.remove('hidden');
  previewBox.classList.add('hidden');
  selectedFile = null;
  fileInput.value = '';
});

/* ── Analyze ─────────────────────────────────────────────────────────────── */
$('analyze-btn').addEventListener('click', runInference);

async function runInference() {
  if (!selectedFile) return;

  // Show loading
  loadingOverlay.classList.remove('hidden');
  setLoadingMessage();

  const formData = new FormData();
  formData.append('file', selectedFile);
  const tta = parseInt(ttaSlider.value, 10);

  try {
    const res = await fetch(`${API}/predict?tta_passes=${tta}`, {
      method: 'POST',
      body:   formData,
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || 'Server error');
    }

    const data = await res.json();
    renderResults(data);

  } catch (err) {
    alert(`Error: ${err.message}`);
  } finally {
    loadingOverlay.classList.add('hidden');
  }
}

/* ── Loading messages ────────────────────────────────────────────────────── */
const LOADING_MSGS = [
  'Running hair removal…',
  'Loading ConvNeXt-Small…',
  'Running TTA augmentations…',
  'Computing Grad-CAM++…',
  'Aggregating predictions…',
];

function setLoadingMessage() {
  let i = 0;
  const el = $('loading-text');
  el.textContent = LOADING_MSGS[0];
  const iv = setInterval(() => {
    i = (i + 1) % LOADING_MSGS.length;
    el.textContent = LOADING_MSGS[i];
  }, 1200);
  // Store so we can clear if needed (simplified: just let it run)
  loadingOverlay._interval = iv;
  const origHide = loadingOverlay.classList.add.bind(loadingOverlay.classList);
  loadingOverlay._clearFn = () => clearInterval(iv);
}

/* ── Render results ──────────────────────────────────────────────────────── */
function renderResults(data) {
  const top = data.predictions[0];

  // ── Banner ──────────────────────────────────────────────────────────────
  $('banner-name').textContent = top.name;
  $('banner-code').textContent = `[${top.class}]`;

  const riskBadge = $('risk-badge');
  riskBadge.textContent  = top.risk;
  riskBadge.style.cssText = riskStyle(top.risk);

  // Confidence ring
  const pct      = Math.round(top.probability * 100);
  const circ     = 201;
  const offset   = circ - (pct / 100) * circ;
  $('ring-pct').textContent = `${pct}%`;
  setTimeout(() => {
    $('ring-fill').style.strokeDashoffset = offset;
    $('ring-fill').style.stroke           = top.color;
  }, 60);

  // ── Images ──────────────────────────────────────────────────────────────
  $('res-original').src = `data:image/jpeg;base64,${data.original_b64}`;
  $('res-clean').src    = `data:image/jpeg;base64,${data.hair_removed_b64}`;
  $('res-gradcam').src  = `data:image/jpeg;base64,${data.gradcam_overlay_b64}`;

  // ── Confidence bars ──────────────────────────────────────────────────────
  const barsEl = $('bars');
  barsEl.innerHTML = '';
  data.predictions.forEach((pred, idx) => {
    const pctVal = (pred.probability * 100).toFixed(1);
    const isTop  = idx === 0;
    const row    = document.createElement('div');
    row.className = 'bar-row';
    row.innerHTML = `
      <span class="bar-label ${isTop ? 'top' : ''}">${pred.name.split(' ').slice(0, 2).join(' ')}</span>
      <div class="bar-track">
        <div class="bar-fill" data-pct="${pred.probability * 100}"
             style="background:${pred.color}; width:0%"></div>
      </div>
      <span class="bar-pct ${isTop ? 'top' : ''}">${pctVal}%</span>
    `;
    barsEl.appendChild(row);
  });

  // Animate bars after render
  requestAnimationFrame(() => {
    document.querySelectorAll('.bar-fill').forEach(el => {
      el.style.width = `${el.dataset.pct}%`;
    });
  });

  // ── Info card ────────────────────────────────────────────────────────────
  const infoCard = $('info-card');
  infoCard.style.borderLeftColor = top.color;
  $('info-title').textContent = `About ${top.name}`;
  $('info-title').style.color = top.color;
  $('info-desc').textContent  = top.desc;

  // ── Show results ─────────────────────────────────────────────────────────
  uploadPanel.classList.add('hidden');
  resultsPanel.classList.remove('hidden');
  window.scrollTo({ top: 0, behavior: 'smooth' });
}

/* ── Risk badge style ────────────────────────────────────────────────────── */
function riskStyle(risk) {
  const map = {
    'Malignant':    'background:rgba(248,113,113,.15);color:#f87171;border:1px solid rgba(248,113,113,.3);',
    'Pre-cancerous':'background:rgba(251,191,36,.15);color:#fbbf24;border:1px solid rgba(251,191,36,.3);',
    'Benign':       'background:rgba(52,211,153,.15);color:#34d399;border:1px solid rgba(52,211,153,.3);',
  };
  const base = 'padding:4px 14px;border-radius:999px;font-size:12px;font-weight:700;font-family:var(--mono);text-transform:uppercase;letter-spacing:.5px;';
  return base + (map[risk] || '');
}
