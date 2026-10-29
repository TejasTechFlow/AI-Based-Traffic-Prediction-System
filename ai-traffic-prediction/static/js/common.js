/* Shared helpers: API calls, chart defaults, formatting, mobile menu. */
(function () {
  const $ = (s, r = document) => r.querySelector(s);
  window.$ = $;
  window.$$ = (s, r = document) => [...r.querySelectorAll(s)];

  // mobile sidebar
  const sb = $('#sidebar'), scrim = $('#scrim'), btn = $('#menuBtn');
  const toggle = (open) => { sb.classList.toggle('open', open); scrim.classList.toggle('show', open); };
  if (btn) { btn.addEventListener('click', () => toggle(!sb.classList.contains('open'))); scrim.addEventListener('click', () => toggle(false)); }

  window.C = { blue: '#2563eb', teal: '#0d9488', purple: '#7c3aed', orange: '#f97316', grey: '#94a3b8',
               blueSoft: 'rgba(37,99,235,.12)', grid: '#eef1f6', ink: '#475569' };
  window.fmt = (n) => (n === null || n === undefined || isNaN(n)) ? '–' : Number(n).toLocaleString('en-US');
  window.esc = (s) => String(s ?? '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  window.catBadge = (cat) => `<span class="cat-badge cat-${esc(cat)}"><span class="cat-dot"></span>${esc(cat)}</span>`;

  window.api = async function (url, opts = {}) {
    let res;
    try { res = await fetch(url, opts); }
    catch (e) { const err = new Error('Could not reach the server. Is the Flask app still running?'); err.data = {}; throw err; }
    let data = {};
    try { data = await res.json(); } catch (e) { /* non-JSON body */ }
    if (!res.ok) { const err = new Error(data.error || `Request failed (${res.status})`); err.data = data; err.status = res.status; throw err; }
    return data;
  };

  const chartsOk = typeof Chart !== 'undefined';
  if (chartsOk) {
    Chart.defaults.font.family = "'Inter', system-ui, sans-serif";
    Chart.defaults.font.size = 12;
    Chart.defaults.color = '#64748b';
    Chart.defaults.borderColor = '#eef1f6';
    Chart.defaults.maintainAspectRatio = false;
    Chart.defaults.plugins.legend.labels.usePointStyle = true;
    Chart.defaults.plugins.legend.labels.boxWidth = 8;
    Chart.defaults.plugins.tooltip.backgroundColor = '#0f172a';
    Chart.defaults.plugins.tooltip.padding = 10;
    Chart.defaults.plugins.tooltip.cornerRadius = 8;
    Chart.defaults.plugins.tooltip.titleFont = { weight: '600' };
    Chart.defaults.interaction = { mode: 'index', intersect: false };
    Chart.defaults.elements.line.borderWidth = 2;
    Chart.defaults.elements.line.tension = 0.3;
    Chart.defaults.elements.point.radius = 0;
    Chart.defaults.elements.point.hoverRadius = 5;
    Chart.defaults.elements.bar.borderRadius = 4;
  }

  const registry = {};
  /** Create or replace a chart. Shows a friendly message if Chart.js did not load. */
  window.mkChart = function (id, config) {
    const el = document.getElementById(id);
    if (!el) return null;
    if (!chartsOk) {
      el.parentElement.innerHTML = '<div class="text-muted small p-3">Chart.js could not be loaded (internet needed for the CDN, or place the library under static/vendor/). Data tables remain available.</div>';
      return null;
    }
    if (registry[id]) registry[id].destroy();
    const base = { options: { responsive: true, scales: {} } };
    registry[id] = new Chart(el, config);
    return registry[id];
  };
  window.yAxis = (title) => ({ beginAtZero: true, grid: { color: C.grid }, border: { display: false },
                               ticks: { callback: v => fmt(v) }, title: title ? { display: true, text: title } : undefined });
  window.xAxis = (extra = {}) => ({ grid: { display: false }, ...extra });
  window.showError = (el, msg) => { el.innerHTML = `<div class="notice notice-err"><i class="bi bi-x-octagon-fill"></i><div>${esc(msg)}</div></div>`; };
})();
