(function () {
  const form = $('#hf');
  const qs = () => { const p = new URLSearchParams(); new FormData(form).forEach((v, k) => { if (v) p.append(k, v); }); return p.toString(); };
  async function load() {
    $('#exportBtn').href = '/api/history/export.csv?' + qs();
    try {
      const d = await api('/api/history?' + qs());
      $('#hBody').innerHTML = d.rows.length ? d.rows.map(r => `<tr><td>${esc(r.created_at)}</td><td>${esc(r.date)}</td><td>${String(r.hour).padStart(2, '0')}:00</td><td>${esc(r.day_of_week)}</td>
        <td>${r.holiday ? 'Yes' : 'No'}</td><td>${esc(r.weather)}</td><td class="text-end">${r.temperature_c === null ? '–' : Number(r.temperature_c).toFixed(1)}</td>
        <td class="text-end fw-semibold">${fmt(r.predicted_volume)}</td><td>${catBadge(r.category)}</td></tr>`).join('')
        : '<tr><td colspan="9" class="text-center text-muted py-4">No predictions found. Make one on the <a href="/predict">Predict traffic</a> page.</td></tr>';
      $('#hCount').textContent = `${d.count} record(s) shown (newest first, max 500).`;
    } catch (e) { $('#hBody').innerHTML = `<tr><td colspan="9" class="text-danger">${esc(e.message)}</td></tr>`; }
  }
  form.addEventListener('submit', (e) => { e.preventDefault(); load(); });
  let t; $('#h_q').addEventListener('input', () => { clearTimeout(t); t = setTimeout(load, 250); });
  $('#hReset').addEventListener('click', () => { form.reset(); load(); });
  $('#clearBtn').addEventListener('click', async () => {
    if (!confirm('Delete all saved predictions? This cannot be undone.')) return;
    try { await api('/api/history/clear', { method: 'POST' }); load(); } catch (e) { alert(e.message); }
  });
  load();
})();
