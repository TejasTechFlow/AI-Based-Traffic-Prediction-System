(function () {
  const form = $('#predForm');
  const DAYS = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'];
  const LEVEL_ORDER = window.CATEGORIES;

  function syncDow() {
    const v = $('#f_date').value;
    const d = v ? new Date(v + 'T00:00:00') : null;
    $('#f_dow').value = d && !isNaN(d) ? DAYS[(d.getDay() + 6) % 7] : '';
  }
  $('#f_date').addEventListener('input', syncDow); syncDow();

  function clearErrors() {
    $$('.is-invalid', form).forEach(e => e.classList.remove('is-invalid'));
    $$('.invalid-feedback', form).forEach(e => e.textContent = '');
    $('#formError').innerHTML = '';
  }
  function fieldErrors(map) {
    const ids = { date: 'f_date', hour: 'f_hour', day_of_week: 'f_dow', weather_main: 'f_weather', temperature_c: 'f_temp',
                  rain_1h: 'f_rain', snow_1h: 'f_snow', clouds_all: 'f_clouds' };
    Object.entries(map).forEach(([k, msg]) => {
      const el = document.getElementById(ids[k]);
      if (!el) return;
      el.classList.add('is-invalid');
      const fb = el.parentElement.querySelector('.invalid-feedback'); if (fb) fb.textContent = msg;
      if (['rain_1h', 'snow_1h', 'clouds_all'].includes(k)) { const c = document.getElementById('adv'); if (c) c.classList.add('show'); }
    });
  }
  function payload() {
    const g = (id) => { const e = document.getElementById(id); return e ? e.value : undefined; };
    return { date: g('f_date'), hour: g('f_hour'), day_of_week: g('f_dow'), holiday: $('#f_holiday').checked,
             weather_main: g('f_weather'), temperature_c: g('f_temp'), rain_1h: g('f_rain'), snow_1h: g('f_snow'), clouds_all: g('f_clouds') };
  }
  function clientValidate(p) {
    const err = {};
    if (!p.date) err.date = 'Please choose a date.';
    if ($('#f_temp') && (p.temperature_c === '' || isNaN(+p.temperature_c))) err.temperature_c = 'Enter a temperature in °C.';
    else if ($('#f_temp') && (+p.temperature_c < -60 || +p.temperature_c > 60)) err.temperature_c = 'Temperature must be between -60 and 60 °C.';
    return err;
  }

  const show = (which) => {
    $('#resEmpty').style.display = which === 'empty' ? '' : 'none';
    $('#resLoading').classList.toggle('show', which === 'loading');
    $('#resBody').style.display = which === 'result' ? '' : 'none';
  };

  function render(r) {
    const color = r.color;
    $('#resBody').style.setProperty('--meter', color);
    $('#rVol').textContent = fmt(r.predicted_volume);
    $('#rCat').innerHTML = catBadge(r.category) + ` <span class="hint ms-1">${esc(r.inputs.day_of_week)}, ${esc(r.inputs.date)} · ${String(r.inputs.hour).padStart(2, '0')}:00</span>`;
    $('#rPct').textContent = `(${r.intensity_pct}% of a typical peak)`;
    const m = $('#rMeter'); m.style.width = '0'; m.style.background = color; requestAnimationFrame(() => m.style.width = r.intensity_pct + '%');
    $('#rLane').innerHTML = LEVEL_ORDER.map((c, i) => `<i class="${i <= r.category_level ? 'on' : ''}" style="height:${18 + i * 7}px" title="${c}"></i>`).join('');
    $('#rReco').textContent = r.recommendation;
    $('#rPeak').textContent = `Busiest hour that day: ${String(r.peak_hour).padStart(2, '0')}:00 (~${fmt(r.peak_volume)} veh/h)`;
    $('#rAssume').textContent = r.assumptions.length ? 'Defaults used: ' + r.assumptions.join('; ') + '.' : '';
    $('#rWarn').innerHTML = r.outside_training_range
      ? '<div class="notice notice-info mt-2"><i class="bi bi-info-circle-fill"></i><div>This date is later than the data the model learned from. It has no year/trend term, so the result shows the <em>typical</em> pattern for this hour, weekday, season and weather - not long-term growth or one-off events.</div></div>' : '';
    const cols = r.profile.categories.map((c, h) => h === r.inputs.hour ? '#0f172a' : (window.CATEGORY_COLORS[c] + 'cc'));
    mkChart('chDay', { type: 'bar', data: { labels: r.profile.hours.map(h => String(h).padStart(2, '0')), datasets: [{
      label: 'Predicted volume (veh/h)', data: r.profile.volumes, backgroundColor: cols }] },
      options: { plugins: { legend: { display: false }, tooltip: { callbacks: { title: i => i[0].label + ':00',
        afterLabel: c => 'Level: ' + r.profile.categories[c.dataIndex] + (c.dataIndex === r.inputs.hour ? ' (selected hour)' : '') } } },
        scales: { x: xAxis({ ticks: { maxRotation: 0, autoSkip: true, maxTicksLimit: 12 } }), y: yAxis() } } });
    show('result');
  }

  form.addEventListener('submit', async (ev) => {
    ev.preventDefault(); clearErrors();
    const p = payload(), ce = clientValidate(p);
    if (Object.keys(ce).length) { fieldErrors(ce); return; }
    const btn = $('#predBtn'); btn.disabled = true; btn.innerHTML = '<span class="spinner-border spinner-border-sm me-2"></span>Predicting…';
    show('loading');
    try {
      const r = await api('/api/predict', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(p) });
      render(r);
    } catch (e) {
      show('empty');
      if (e.data && e.data.fields) fieldErrors(e.data.fields);
      showError($('#formError'), e.message);
    } finally { btn.disabled = false; btn.innerHTML = '<i class="bi bi-lightning-charge-fill me-1"></i>Predict Traffic'; }
  });

  // sample scenarios (still go through the real model)
  api('/api/sample-scenarios').then(list => {
    $('#samples').innerHTML = list.map((s, i) => `<button type="button" class="btn btn-soft btn-sm py-0 px-2" data-i="${i}">${esc(s.name)}</button>`).join('');
    $('#samples').addEventListener('click', (e) => {
      const b = e.target.closest('button'); if (!b) return;
      const s = list[b.dataset.i];
      $('#f_date').value = s.date; $('#f_hour').value = s.hour; $('#f_holiday').checked = s.holiday;
      if ($('#f_weather')) $('#f_weather').value = s.weather_main;
      if ($('#f_temp')) $('#f_temp').value = s.temperature_c;
      ['f_rain', 'f_snow', 'f_clouds'].forEach(id => { const el = document.getElementById(id); if (el) el.value = ''; });
      syncDow(); form.requestSubmit();
    });
  }).catch(() => { $('#samples').innerHTML = '<span class="hint m-0">unavailable until the model is trained</span>'; });
})();
