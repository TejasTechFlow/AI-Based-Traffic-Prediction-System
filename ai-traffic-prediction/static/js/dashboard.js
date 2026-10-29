(async function () {
  let d;
  try { d = await api('/api/dashboard'); }
  catch (e) { showError($('#dashError'), e.message); return; }

  $('#rangePill').innerHTML = `<i class="bi bi-calendar3"></i>Data: ${esc(d.dataset_range)}`;
  const L = d.latest_record;
  $('#kLatest').innerHTML = `${fmt(L.volume)}<small> veh/h</small>`;
  $('#kLatestNote').innerHTML = `${esc(L.time)} · ${catBadge(L.category)}`;
  if (d.next_hour) {
    $('#kPred').innerHTML = `${fmt(d.next_hour.volume)}<small> veh/h</small>`;
    $('#kPredNote').textContent = d.next_hour.label + ' · model forecast';
    $('#kCong').innerHTML = catBadge(d.next_hour.category);
    $('#kMeter').style.width = d.next_hour.intensity_pct + '%';
    $('#kMeter').style.background = C_COLOR(d.next_hour.category);
    $('#kCongNote').textContent = d.next_hour.assumption;
  } else {
    $('#kPredNote').textContent = 'Model not available';
  }
  $('#kAvg').innerHTML = `${fmt(d.avg_volume)}<small> veh/h</small>`;
  $('#kPeak').textContent = 'Peak hours: ' + d.peak_hours.join(', ');

  function C_COLOR(cat) { return window.CATEGORY_COLORS[cat] || '#94a3b8'; }
  const peakSet = new Set(d.peak_hours);

  mkChart('chLine', { type: 'line', data: { labels: d.line_last_week.labels, datasets: [
    { label: 'Recorded volume (veh/h)', data: d.line_last_week.values, borderColor: C.blue, backgroundColor: C.blueSoft, fill: true } ] },
    options: { plugins: { legend: { display: false } }, scales: { x: xAxis({ ticks: { maxTicksLimit: 8, maxRotation: 0 } }), y: yAxis() } } });

  mkChart('chDonut', { type: 'doughnut', data: { labels: d.congestion.labels, datasets: [{ data: d.congestion.counts,
    backgroundColor: d.congestion.colors, borderColor: '#fff', borderWidth: 3, hoverOffset: 6 }] },
    options: { cutout: '64%', plugins: { legend: { position: 'bottom' }, tooltip: { callbacks: {
      label: (c) => { const t = c.dataset.data.reduce((a, b) => a + b, 0); return ` ${c.label}: ${fmt(c.parsed)} hours (${(100 * c.parsed / t).toFixed(1)}%)`; } } } } } });

  mkChart('chHour', { type: 'bar', data: { labels: d.hourly.labels, datasets: [{ label: 'Average volume (veh/h)', data: d.hourly.avg,
    backgroundColor: d.hourly.labels.map(l => peakSet.has(l) ? C.orange : C.blue) }] },
    options: { plugins: { legend: { display: false } }, scales: { x: xAxis({ ticks: { maxRotation: 0, autoSkip: true, maxTicksLimit: 12 } }), y: yAxis() } } });

  if (d.hist_vs_pred) {
    mkChart('chHvP', { type: 'line', data: { labels: d.hist_vs_pred.labels, datasets: [
      { label: 'Actual', data: d.hist_vs_pred.actual, borderColor: C.blue },
      { label: 'Predicted', data: d.hist_vs_pred.predicted, borderColor: C.orange, borderDash: [5, 4] }] },
      options: { plugins: { legend: { position: 'top', align: 'end' } }, scales: { x: xAxis({ ticks: { maxTicksLimit: 7, maxRotation: 0 } }), y: yAxis() } } });
  }
  mkChart('chDaily', { type: 'line', data: { labels: d.daily_30.labels, datasets: [{ label: 'Daily average (veh/h)', data: d.daily_30.values, borderColor: C.teal, backgroundColor: 'rgba(13,148,136,.1)', fill: true }] },
    options: { plugins: { legend: { display: false } }, scales: { x: xAxis({ ticks: { maxTicksLimit: 8, maxRotation: 0 } }), y: yAxis() } } });
  mkChart('chWeek', { type: 'bar', data: { labels: d.weekly.labels, datasets: [{ label: 'Average volume (veh/h)', data: d.weekly.avg, backgroundColor: C.purple }] },
    options: { plugins: { legend: { display: false } }, scales: { x: xAxis(), y: yAxis() } } });
})();
