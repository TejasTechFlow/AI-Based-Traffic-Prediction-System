(function () {
  const form = $('#filters');
  const qs = () => {
    const p = new URLSearchParams();
    new FormData(form).forEach((v, k) => { if (v !== '' && !(k === 'day_type' && v === 'all')) p.append(k, v); });
    return p.toString();
  };

  async function load() {
    $('#filterError').innerHTML = '';
    $('#reportBtn').href = '/api/report.csv?' + qs();
    let d;
    try { d = await api('/api/analytics?' + qs()); }
    catch (e) { showError($('#filterError'), e.message); return; }
    if (d.empty) { $('#anaBody').style.display = 'none'; $('#anaEmpty').style.display = ''; $('#anaEmptyMsg').textContent = d.message + ' Try widening the date range or clearing a filter.'; return; }
    $('#anaBody').style.display = ''; $('#anaEmpty').style.display = 'none';

    const k = d.kpis;
    $('#aRows').textContent = fmt(d.rows); $('#aPeriod').textContent = k.period;
    $('#aAvg').innerHTML = `${fmt(k.avg)}<small> veh/h</small>`; $('#aMax').textContent = `Range ${fmt(k.min)} - ${fmt(k.max)}`;
    $('#aPeak').textContent = k.peak_hours.join(' · '); $('#aPeakAvg').textContent = `Busiest averages ${fmt(k.peak_avg)} veh/h`;
    const ww = d.weekday_weekend;
    $('#aWW').textContent = `${ww.weekday_avg === null ? '–' : fmt(ww.weekday_avg)} / ${ww.weekend_avg === null ? '–' : fmt(ww.weekend_avg)}`;
    const peaks = new Set(k.peak_hours);
    const noTicks = { ticks: { maxRotation: 0, autoSkip: true, maxTicksLimit: 12 } };

    mkChart('cHour', { type: 'line', data: { labels: d.hourly.labels, datasets: [
      { label: '25th percentile', data: d.hourly.p25, borderWidth: 0, backgroundColor: 'transparent', pointRadius: 0, fill: false },
      { label: '25th-75th percentile', data: d.hourly.p75, borderWidth: 0, backgroundColor: C.blueSoft, fill: '-1', pointRadius: 0 },
      { label: 'Average volume', data: d.hourly.avg, borderColor: C.blue, pointRadius: d.hourly.labels.map(l => peaks.has(l) ? 5 : 0), pointBackgroundColor: C.orange, pointBorderColor: '#fff', pointBorderWidth: 2 }] },
      options: { plugins: { legend: { position: 'top', align: 'end', labels: { filter: i => i.text !== '25th percentile' } } }, scales: { x: xAxis(noTicks), y: yAxis() } } });

    mkChart('cWW', { type: 'line', data: { labels: ww.labels, datasets: [
      { label: 'Weekdays', data: ww.weekday, borderColor: C.blue }, { label: 'Weekends', data: ww.weekend, borderColor: C.orange } ] },
      options: { plugins: { legend: { position: 'top', align: 'end' } }, scales: { x: xAxis(noTicks), y: yAxis() } } });

    $('#tDaily').textContent = d.daily.name + ' trend';
    mkChart('cDaily', { type: 'line', data: { labels: d.daily.labels, datasets: [{ label: d.daily.name + ' (veh/h)', data: d.daily.avg, borderColor: C.teal, backgroundColor: 'rgba(13,148,136,.1)', fill: true }] },
      options: { plugins: { legend: { display: false } }, scales: { x: xAxis({ ticks: { maxTicksLimit: 8, maxRotation: 0 } }), y: yAxis() } } });

    mkChart('cDow', { type: 'bar', data: { labels: d.dow.labels, datasets: [{ label: 'Average volume (veh/h)', data: d.dow.avg, backgroundColor: d.dow.labels.map((_, i) => i >= 5 ? C.orange : C.purple) }] },
      options: { plugins: { legend: { display: false } }, scales: { x: xAxis(), y: yAxis() } } });

    mkChart('cMonth', { type: 'line', data: { labels: d.monthly.labels, datasets: [{ label: 'Monthly average (veh/h)', data: d.monthly.avg, borderColor: C.blue, pointRadius: 2 }] },
      options: { plugins: { legend: { display: false } }, scales: { x: xAxis({ ticks: { maxTicksLimit: 8, maxRotation: 0 } }), y: yAxis() } } });

    mkChart('cMoy', { type: 'bar', data: { labels: d.month_of_year.labels, datasets: [{ label: 'Average volume (veh/h)', data: d.month_of_year.avg, backgroundColor: C.teal }] },
      options: { plugins: { legend: { display: false } }, scales: { x: xAxis(), y: yAxis() } } });

    if (document.getElementById('cWeather')) {
      mkChart('cWeather', { type: 'bar', data: { labels: d.weather.labels, datasets: [{ label: 'Average volume (veh/h)', data: d.weather.avg, backgroundColor: C.blue }] },
        options: { indexAxis: 'y', plugins: { legend: { display: false }, tooltip: { callbacks: { afterLabel: c => fmt(d.weather.counts[c.dataIndex]) + ' records' } } },
          scales: { x: yAxis(), y: { grid: { display: false } } } } });
    }
    mkChart('cCong', { type: 'doughnut', data: { labels: d.congestion.labels, datasets: [{ data: d.congestion.counts, backgroundColor: d.congestion.colors, borderColor: '#fff', borderWidth: 3 }] },
      options: { cutout: '62%', plugins: { legend: { position: 'bottom' }, tooltip: { callbacks: { label: c => { const t = c.dataset.data.reduce((a, b) => a + b, 0); return ` ${c.label}: ${fmt(c.parsed)} h (${(100 * c.parsed / t).toFixed(1)}%)`; } } } } } });
  }

  form.addEventListener('submit', (e) => { e.preventDefault(); load(); });
  $('#resetBtn').addEventListener('click', () => {
    form.reset(); $('#a_start').value = ANA_DEFAULTS.start; $('#a_end').value = ANA_DEFAULTS.end; load();
  });
  load();
})();
