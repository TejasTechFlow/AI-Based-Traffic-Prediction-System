(function () {
  let M = null, uploadId = null;

  async function load() {
    try { M = await api('/api/performance'); }
    catch (e) { showError($('#perfError'), e.message); return; }
    $('#perfBody').style.display = '';
    const m = M.metrics, s = M.split, ds = M.dataset;
    $('#pAlgo').textContent = M.algorithm; $('#pRule').textContent = 'Lowest cross-validated RMSE';
    $('#pRows').innerHTML = `${fmt(s.train_rows)}<small> rows</small>`;
    $('#pSplit').textContent = `${s.train_pct}% train / ${s.test_pct}% test (chronological) · ${fmt(s.test_rows)} test rows`;
    $('#pErr').textContent = `${m.mae.toFixed(0)} / ${m.rmse.toFixed(0)}`;
    $('#pR2').textContent = m.r2.toFixed(3);
    $('#pTrained').textContent = 'Trained ' + M.trained_at;
    $('#pDatasetNote').innerHTML = `<i class="bi bi-${ds.synthetic ? 'exclamation-triangle-fill' : 'database-check'}"></i><div><strong>${esc(ds.label)}.</strong>
      ${fmt(ds.rows_clean)} clean hourly records, ${esc(ds.date_min.slice(0, 10))} to ${esc(ds.date_max.slice(0, 10))}. Training period ${esc(s.train_range[0].slice(0, 10))} → ${esc(s.train_range[1].slice(0, 10))};
      test period ${esc(s.test_range[0].slice(0, 10))} → ${esc(s.test_range[1].slice(0, 10))}.
      ${ds.synthetic ? 'Metrics describe how well the model learned the <em>simulated</em> patterns, not real-world accuracy.' : ''}</div>`;
    $('#pDatasetNote').className = 'notice ' + (ds.synthetic ? 'notice-warn' : 'notice-info') + ' mb-3';

    $('#pTable').innerHTML = M.comparison.map(r => `<tr class="${r.selected ? 'best' : ''}"><td><strong>${esc(r.model)}</strong></td>
      <td class="text-end">${r.cv_rmse.toFixed(1)}</td><td class="text-end">${r.test_mae.toFixed(1)}</td><td class="text-end">${r.test_rmse.toFixed(1)}</td>
      <td class="text-end">${r.test_r2.toFixed(3)}</td><td class="text-end">${r.train_r2.toFixed(3)}</td>
      <td>${r.selected ? '<span class="pill pill-info py-0"><i class="bi bi-check2-circle"></i>Selected</span>' : ''}</td></tr>`).join('');

    const cl = ds.cleaning;
    $('#pClean').innerHTML = [`Raw rows: <strong>${fmt(cl.rows_in)}</strong> → clean rows: <strong>${fmt(cl.rows_out)}</strong>`,
      `Duplicate records removed: <strong>${fmt(cl.duplicates_removed)}</strong>`,
      `Impossible readings blanked and imputed (e.g. 0 K temperature, rainfall > 100 mm/h): <strong>${fmt(cl.outliers_nulled || 0)}</strong>`,
      'Missing values imputed with the <em>training-set</em> median / "Unknown" inside the pipeline (no leakage)',
      'Holiday flag spread to every hour of the holiday; temperature converted to °C',
      'Features: hour, weekday, weekday×hour, season, holiday, temperature, rain, snow, cloud cover, weather type',
      'No future or target-derived features are used'].map(t => `<li class="mb-1">${t}</li>`).join('');

    const th = M.thresholds.values;
    const rows = [['Low', `< ${fmt(Math.round(th[0]))}`], ['Moderate', `${fmt(Math.round(th[0]))} - ${fmt(Math.round(th[1]))}`], ['High', `${fmt(Math.round(th[1]))} - ${fmt(Math.round(th[2]))}`], ['Severe', `≥ ${fmt(Math.round(th[2]))}`]];
    $('#pThSrc').textContent = 'From ' + M.thresholds.source + ' (configurable in config.py)';
    $('#pTh').innerHTML = rows.map(([c, r]) => `<div class="d-flex justify-content-between align-items-center border rounded-3 px-3 py-2">${catBadge(c)}<span class="fw-semibold">${r} <span class="text-muted fw-normal small">veh/h</span></span></div>`).join('');

    const imp = M.feature_importance;
    mkChart('pImp', { type: 'bar', data: { labels: imp.map(i => i.feature), datasets: [{ label: 'Share of total importance', data: imp.map(i => +(i.share * 100).toFixed(1)), backgroundColor: C.purple }] },
      options: { indexAxis: 'y', plugins: { legend: { display: false }, tooltip: { callbacks: { label: c => ` ${c.parsed.x}% of total (RMSE +${imp[c.dataIndex].importance.toFixed(0)})` } } },
        scales: { x: { grid: { color: C.grid }, border: { display: false }, ticks: { callback: v => v + '%' } }, y: { grid: { display: false } } } } });
    avp('week');
  }

  function avp(view) {
    const s = M.series;
    if (view === 'scatter') {
      const mx = Math.max(...s.scatter.map(p => Math.max(p.x, p.y)));
      mkChart('pAvp', { type: 'scatter', data: { datasets: [
        { label: 'Test hours', data: s.scatter, backgroundColor: 'rgba(37,99,235,.45)', pointRadius: 3 },
        { label: 'Perfect prediction', type: 'line', data: [{ x: 0, y: 0 }, { x: mx, y: mx }], borderColor: C.orange, borderDash: [5, 4], pointRadius: 0 }] },
        options: { interaction: { mode: 'nearest', intersect: true }, plugins: { legend: { position: 'top', align: 'end' } },
          scales: { x: { title: { display: true, text: 'Actual (veh/h)' }, grid: { color: C.grid } }, y: { title: { display: true, text: 'Predicted (veh/h)' }, grid: { color: C.grid } } } } });
      return;
    }
    const d = view === 'week' ? s.last_week : s.daily;
    mkChart('pAvp', { type: 'line', data: { labels: d.labels, datasets: [
      { label: 'Actual', data: d.actual, borderColor: C.blue }, { label: 'Predicted', data: d.predicted, borderColor: C.orange, borderDash: [5, 4] }] },
      options: { plugins: { legend: { position: 'top', align: 'end' } }, scales: { x: xAxis({ ticks: { maxTicksLimit: 8, maxRotation: 0 } }), y: yAxis('Vehicles per hour') } } });
  }
  $('#avpSwitch').addEventListener('click', (e) => {
    const b = e.target.closest('button'); if (!b || !M) return;
    $$('#avpSwitch button').forEach(x => x.classList.toggle('active', x === b)); avp(b.dataset.v);
  });

  const msg = (cls, html) => { $('#upMsg').innerHTML = `<div class="notice notice-${cls}">${html}</div>`; };
  $('#upBtn').addEventListener('click', async () => {
    const f = $('#upFile').files[0];
    if (!f) { msg('warn', '<i class="bi bi-exclamation-triangle-fill"></i><div>Choose a CSV file first.</div>'); return; }
    const fd = new FormData(); fd.append('file', f);
    msg('info', '<span class="spinner-border spinner-border-sm"></span><div>Validating…</div>');
    try {
      const r = await api('/api/upload', { method: 'POST', body: fd });
      uploadId = r.upload_id; $('#retrainBtn').disabled = false;
      msg('info', `<i class="bi bi-check-circle-fill"></i><div><strong>Valid file.</strong> ${fmt(r.rows)} rows, ${esc(r.date_min)} → ${esc(r.date_max)}. Optional columns found: ${r.optional_columns.length ? esc(r.optional_columns.join(', ')) : 'none'}.</div>`);
    } catch (e) {
      uploadId = null; $('#retrainBtn').disabled = true;
      const probs = (e.data && e.data.problems) ? '<ul class="mb-0">' + e.data.problems.map(p => `<li>${esc(p)}</li>`).join('') + '</ul>' : '';
      msg('err', `<i class="bi bi-x-octagon-fill"></i><div>${esc(e.message)}${probs}</div>`);
    }
  });
  async function retrain(body, btn) {
    const old = btn.innerHTML; $$('#retrainBtn,#retrainCurrent,#upBtn').forEach(b => b.disabled = true);
    btn.innerHTML = '<span class="spinner-border spinner-border-sm me-2"></span>Training…';
    msg('info', '<span class="spinner-border spinner-border-sm"></span><div>Training and comparing models - this usually takes under a minute.</div>');
    try {
      const r = await api('/api/retrain', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
      msg('info', `<i class="bi bi-check-circle-fill"></i><div><strong>Done.</strong> Selected ${esc(r.algorithm)} (R² ${r.metrics.r2.toFixed(3)}). Reloading…</div>`);
      setTimeout(() => location.reload(), 900);
    } catch (e) { msg('err', `<i class="bi bi-x-octagon-fill"></i><div>${esc(e.message)}</div>`); }
    finally { btn.innerHTML = old; $('#upBtn').disabled = false; $('#retrainCurrent').disabled = false; $('#retrainBtn').disabled = !uploadId; }
  }
  $('#retrainBtn').addEventListener('click', (e) => retrain({ upload_id: uploadId }, e.currentTarget));
  $('#retrainCurrent').addEventListener('click', (e) => retrain({}, e.currentTarget));
  load();
})();
