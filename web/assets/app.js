import { barChart, lineChart, strip } from './charts.js?v=202611152008';
import { STR, fmt } from './i18n.js?v=202611152008';

const C = { clean: '#3987e5', raw: '#d95926', naive: '#199e70', base: '#5f5e58' };
const TRACK_NAME = { '143_A_025': '143 A', '108_D_065': '108 D' };
const MONTH_DOY = [152, 182, 213, 244, 274, 305];
const RFI_RAMP = 'linear-gradient(90deg, rgba(255,214,102,0) 0%, #ffd666 18%, #fa8c32 50%, #eb4034 78%, #d61e6e 100%)';

const $ = s => document.querySelector(s);
const ddmm = d => `${d.slice(8, 10)}.${d.slice(5, 7)}`;
const sgn = v => (v > 0 ? '+' : v < 0 ? '−' : '') + Math.abs(v).toFixed(1);
const num = v => String(v);

const state = { lang: 'en', mode: 'compare', track: '143_A_025', date: null, base: 'dark', opacity: 1, swipe: 0.5, rfi: false,
  chartTrack: '143_A_025', pol: 'hv', classTrack: '143_A_025', classPol: 'hh' };
let APP, mapA, mapB;
const T = (k, v) => fmt(STR[state.lang][k] ?? k, v);

// ---------- мова ----------
function applyLang() {
  document.documentElement.lang = state.lang;
  document.querySelectorAll('[data-i18n]').forEach(e => { e.textContent = T(e.dataset.i18n); });
  $('#limitsList').innerHTML = STR[state.lang].limits.map(x => `<li>${x}</li>`).join('');
  $('#loading').textContent = T('loading');
  if (APP) { buildTrackSegs(); figures(); charts(); classes(); renderLegend(); }
}

// ---------- перемикачі ----------
function seg(root, items, value, onPick) {
  root.replaceChildren();
  for (const it of items) {
    const b = document.createElement('button'); b.dataset.v = it.v; b.textContent = it.label;
    if (it.sub) { const s = document.createElement('small'); s.textContent = it.sub; b.appendChild(s); }
    if (it.v === value) b.classList.add('on');
    b.addEventListener('click', () => { root.querySelectorAll('button').forEach(x => x.classList.toggle('on', x === b)); onPick(it.v); });
    root.appendChild(b);
  }
}
function bindSeg(root, onPick) {
  root.querySelectorAll('button').forEach(b => b.addEventListener('click', () => {
    root.querySelectorAll('button').forEach(x => x.classList.toggle('on', x === b)); onPick(b.dataset.v);
  }));
}
function buildTrackSegs() {
  const items = Object.keys(APP.tracks).map(t => ({ v: t, label: TRACK_NAME[t], sub: T(`trk_${t}`) }));
  seg($('#trackSeg'), items, state.track, v => { state.track = v; buildDates(); render(); });
  const short = items.map(i => ({ v: i.v, label: i.label }));
  seg($('#chartTrack'), short, state.chartTrack, v => { state.chartTrack = v; trackCharts(); });
  seg($('#classTrack'), short, state.classTrack, v => { state.classTrack = v; classes(); });
}

// ---------- карта ----------
function style() {
  return {
    version: 8,
    sources: {
      dark: { type: 'raster', tiles: ['https://server.arcgisonline.com/ArcGIS/rest/services/Elevation/World_Hillshade_Dark/MapServer/tile/{z}/{y}/{x}'], tileSize: 256, maxzoom: 16,
        attribution: '© Esri, USGS, NGA, NASA' },
      sat: { type: 'raster', tiles: ['https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}'], tileSize: 256, maxzoom: 18,
        attribution: '© Esri, Maxar, Earthstar Geographics' },
    },
    layers: [
      { id: 'bg', type: 'background', paint: { 'background-color': '#0b0b0c' } },
      { id: 'dark', type: 'raster', source: 'dark', paint: { 'raster-brightness-max': 0.55, 'raster-contrast': -0.2 } },
      { id: 'sat', type: 'raster', source: 'sat', layout: { visibility: 'none' }, paint: { 'raster-saturation': -0.4, 'raster-brightness-max': 0.75 } },
    ],
  };
}
const cache = new Map();
// Версія в адресі шару: після нового експорту браузер не покаже стару копію з кешу
// (раніше за адресою <дата>_raw.png лежав тонований знімок, і кеш на годину показував його без галочки).
const V = u => (u.includes('?') ? u : `${u}?v=${encodeURIComponent((APP && APP.generated) || '')}`);
function preload(url) {
  url = V(url);
  if (!cache.has(url)) cache.set(url, new Promise((ok, bad) => { const i = new Image(); i.onload = () => ok(url); i.onerror = bad; i.src = url; }));
  return cache.get(url);
}
// Кожен виклик render() отримує номер покоління; зображення, що догрузилось для застарілого покоління,
// відкидається. Інакше повільне завантаження «увімкненого» шару могло перезаписати пізніше вимкнення.
let GEN = 0, LOADING = 0;
async function setImage(map, id, url, opacity, gen, nearest = false) {
  url = V(url);
  const ld = $('#loading'); LOADING++; ld.hidden = false;
  try { await preload(url); } finally { if (--LOADING <= 0) { LOADING = 0; ld.hidden = true; } }
  if (gen !== GEN) return false;
  const src = map.getSource(id);
  if (src) src.updateImage({ url, coordinates: APP.bounds });
  else {
    map.addSource(id, { type: 'image', url, coordinates: APP.bounds });
    map.addLayer({ id, type: 'raster', source: id, paint: { 'raster-opacity': opacity, 'raster-fade-duration': 0, 'raster-resampling': nearest ? 'nearest' : 'linear' } });
  }
  map.setPaintProperty(id, 'raster-opacity', opacity);
  return true;
}
function drop(map, id) { if (map.getLayer(id)) map.removeLayer(id); if (map.getSource(id)) map.removeSource(id); }
const img = () => APP.tracks[state.track].images.find(x => x.date === state.date);

async function render() {
  const TR = APP.tracks[state.track];
  if (!TR.dates.includes(state.date)) state.date = TR.dates[Math.min(2, TR.dates.length - 1)];
  $('#dateNow').textContent = ddmm(state.date);
  document.querySelectorAll('#dates button').forEach(b => b.classList.toggle('on', b.dataset.v === state.date));
  const cmp = state.mode === 'compare';
  for (const id of ['#mapB', '#swipe', '#tagA', '#tagB']) $(id).style.display = cmp ? '' : 'none';
  $('#layerCtl').style.display = cmp ? '' : 'none';
  const I = img(), gen = ++GEN;
  renderLegend();
  if (cmp) {
    drop(mapA, 'chg');
    if (!state.rfi) drop(mapB, 'mask');
    await Promise.all([
      setImage(mapA, 'sar', state.rfi ? I.rawrfi : I.raw, state.opacity, gen),
      setImage(mapB, 'sar', I.clean, state.opacity, gen),
      state.rfi ? setImage(mapB, 'mask', I.mask, 1, gen, true) : null,
    ]);
    if (gen !== GEN) return;
    // фоном готуємо обидва варіанти поточної дати й сусідні дати, щоб перемикання було миттєвим
    [I.raw, I.rawrfi, I.mask].forEach(preload);
    const i = TR.dates.indexOf(state.date);
    [-1, 1].forEach(k => { const n = TR.images[i + k]; if (n) [state.rfi ? n.rawrfi : n.raw, n.clean, state.rfi ? n.mask : null].filter(Boolean).forEach(preload); });
  } else {
    drop(mapB, 'mask');
    await Promise.all([setImage(mapA, 'sar', I.clean, state.opacity * 0.55, gen), setImage(mapA, 'chg', state.mode === 'change' ? TR.change : APP.agree, 1, gen, true)]);
  }
}

function renderLegend() {
  const lg = $('#legend'); lg.replaceChildren();
  const block = (title, rampCss, ticks) => {
    const d = document.createElement('div');
    const t = document.createElement('div'); t.className = 'lg-t'; t.textContent = title; d.appendChild(t);
    const r = document.createElement('div'); r.className = 'ramp'; r.style.background = rampCss; d.appendChild(r);
    const k = document.createElement('div'); k.className = 'ticks';
    ticks.forEach(x => { const s = document.createElement('span'); s.textContent = x; k.appendChild(s); }); d.appendChild(k);
    lg.appendChild(d);
  };
  const row = (cls, text) => {
    const d = document.createElement('div'); d.className = 'lg-row';
    const s = document.createElement('span'); s.className = `sw ${cls}`; const p = document.createElement('span'); p.textContent = text;
    d.append(s, p); lg.appendChild(d);
  };
  const note = html => { const p = document.createElement('div'); p.className = 'lg-note'; p.innerHTML = html; lg.appendChild(p); };
  const I = img();
  $('#quality').innerHTML = I ? T('quality', { bad: num(I.bad), deg: num(I.degraded) }) : '';
  $('#quality').style.display = state.mode === 'compare' ? '' : 'none';
  if (state.mode === 'compare') {
    const gray = 'linear-gradient(90deg,#000,#fff)', gt = [`${APP.gray_range_db[0]}`, `${APP.gray_range_db[1]}`];
    block(T('lg_raw_plain'), gray, gt);
    if (state.rfi) block(T('lg_raw'), RFI_RAMP, ['×1', '×4', '×16']);
    block(T('lg_clean'), gray, gt);
    if (state.rfi) { row('deg', T('lg_deg')); row('bad', T('lg_bad')); }
  } else {
    const pal = APP.season_palette, lo = MONTH_DOY[0], hi = 334;
    const css = `linear-gradient(90deg, ${pal.map(([d, c]) => `${c} ${(100 * (d - lo) / (hi - lo)).toFixed(1)}%`).join(', ')})`;
    if (state.mode === 'change') {
      block(T('lg_change'), css, STR[state.lang].months);
      const c = APP.change[state.track];
      note(T('lg_change_n', { pct: `<b>${num(c['змінилось, %'])}</b>`, km: c['змінилось, км²'].toLocaleString('en'), n: APP.tracks[state.track].dates.length }));
    } else {
      block(T('lg_agree'), css, STR[state.lang].months);
      const x = APP.cross_track['рілля'];
      note(T('lg_agree_n', { lift: `<b>${num(x['підйом (в обох / очікувано)'].toFixed(1))}</b>`, a: num(x['дата першої зміни збігається ±12 діб, %']), b: num(x['випадково (перемішування), %']) }));
    }
  }
}

function buildDates() {
  const TR = APP.tracks[state.track], root = $('#dates'); root.replaceChildren();
  TR.images.forEach(I => {
    const b = document.createElement('button'); b.dataset.v = I.date; b.textContent = ddmm(I.date);
    const bar = document.createElement('i'); bar.style.width = `${Math.min(100, (I.bad + 0.5 * I.degraded) * 2.5)}%`; b.appendChild(bar);
    b.title = `${I.date}: ${I.bad} % / ${I.degraded} %`;
    b.addEventListener('click', () => { state.date = I.date; render(); });
    root.appendChild(b);
  });
}

// Шторка порівняння: лінія з кружком по центру карти.
// Тягнеться лише поки натиснута ліва кнопка: pointer capture тримає рух на лінії навіть за межами карти,
// а якщо подія руху прийшла без натиснутої кнопки (відпускання загубилось) — перетягування одразу завершується.
// Оновлення не частіше одного разу на кадр.
const SW = { w: 0, left: 0, raf: 0, pending: null, drag: false };
function measureSwipe() { const r = $('#mapWrap').getBoundingClientRect(); SW.w = r.width; SW.left = r.left; }
function setSwipe(frac) {
  if (!SW.w) measureSwipe();
  // 16 px від країв: смуга, за яку хапають лінію (32 px), завжди лишається видимою
  const m = SW.w > 64 ? 16 / SW.w : 0;
  state.swipe = Math.max(m, Math.min(1 - m, frac));
  const px = Math.round(state.swipe * SW.w);
  const s = $('#swipe');
  s.style.transform = `translate3d(${px}px,0,0)`;
  s.setAttribute('aria-valuenow', Math.round(state.swipe * 100));
  $('#mapB').style.clipPath = `inset(0 0 0 ${px}px)`;
}
function initSwipe() {
  const s = $('#swipe');
  const queue = x => {
    SW.pending = (x - SW.left) / SW.w;
    if (!SW.raf) SW.raf = requestAnimationFrame(() => { SW.raf = 0; if (SW.pending != null) setSwipe(SW.pending); SW.pending = null; });
  };
  const stop = () => { SW.drag = false; s.classList.remove('dragging'); };
  s.addEventListener('pointerdown', e => {
    if (e.button !== 0) return;
    e.preventDefault();
    measureSwipe();
    SW.drag = true; s.classList.add('dragging');
    s.setPointerCapture(e.pointerId);
    queue(e.clientX);
  });
  s.addEventListener('pointermove', e => {
    if (!SW.drag) return;
    if (e.buttons === 0) { stop(); return; }
    queue(e.clientX);
  });
  s.addEventListener('pointerup', stop);
  s.addEventListener('pointercancel', stop);
  s.addEventListener('lostpointercapture', stop);
  s.addEventListener('keydown', e => {
    const step = e.shiftKey ? 0.1 : 0.02;
    if (e.key === 'ArrowLeft') { setSwipe(state.swipe - step); e.preventDefault(); }
    if (e.key === 'ArrowRight') { setSwipe(state.swipe + step); e.preventDefault(); }
  });
  addEventListener('resize', () => { measureSwipe(); setSwipe(state.swipe); });
  addEventListener('scroll', measureSwipe, { passive: true });
  measureSwipe(); setSwipe(0.5);
}
function initMaps() {
  const bb = APP.bounds, b = [[Math.min(...bb.map(p => p[0])), Math.min(...bb.map(p => p[1]))], [Math.max(...bb.map(p => p[0])), Math.max(...bb.map(p => p[1]))]];
  const opts = { style: style(), bounds: b, fitBoundsOptions: { padding: 24 }, attributionControl: false, maxZoom: 14, minZoom: 6, dragRotate: false, pitchWithRotate: false };
  mapA = new maplibregl.Map({ container: 'mapA', ...opts });
  mapB = new maplibregl.Map({ container: 'mapB', ...opts, interactive: false });
  mapA.addControl(new maplibregl.NavigationControl({ showCompass: false }), 'bottom-right');
  mapA.addControl(new maplibregl.AttributionControl({ compact: true }), 'bottom-left');
  mapA.addControl(new maplibregl.ScaleControl({ unit: 'metric' }), 'bottom-left');
  mapA.on('move', () => mapB.jumpTo({ center: mapA.getCenter(), zoom: mapA.getZoom(), bearing: mapA.getBearing(), pitch: mapA.getPitch() }));
  let fitted = false;
  new ResizeObserver(() => {
    mapA.resize(); mapB.resize();
    if (!fitted && $('#mapWrap').clientWidth > 200) { fitted = true; mapA.fitBounds(b, { padding: 24, animate: false }); }
    measureSwipe(); setSwipe(state.swipe);
  }).observe($('#mapWrap'));
  return Promise.all([mapA, mapB].map(m => new Promise(ok => m.on('load', ok))));
}
function setBase(v) {
  state.base = v;
  for (const m of [mapA, mapB]) { m.setLayoutProperty('dark', 'visibility', v === 'dark' ? 'visible' : 'none'); m.setLayoutProperty('sat', 'visibility', v === 'sat' ? 'visible' : 'none'); }
}

// ---------- ключові числа ----------
function figures() {
  const L = APP.leak_143.tests, last = L.raw.by_ratio.length - 1;
  const f = APP.contrast.find(c => c.zone === 'рілля' && c.track === '108_D_065' && c.pol === 'HV');
  const x = APP.cross_track['рілля'], w = APP.water_iou['143_A_025'];
  const items = [
    [`${num(L.raw.by_ratio[last])} %`, `${num(L.aware.by_ratio[last])} %`, 'fig1'],
    [`${num(sgn(f.raw))} ${T('unit_db')}`, `${num(sgn(f.clean))} ${T('unit_db')}`, 'fig2'],
    [null, `${num(x['підйом (в обох / очікувано)'].toFixed(1))}×`, 'fig3'],
    [num(w['сир'].toFixed(2)), num(w['чист'].toFixed(2)), 'fig4'],
  ];
  $('#figs').innerHTML = items.map(([a, b, k]) =>
    `<div class="fig"><div class="v">${a ? `<s>${a}</s><em>→</em>` : ''}<span>${b}</span></div><p>${T(k)}</p></div>`).join('');
}

// ---------- графіки ----------
function charts() {
  const L = APP.leak_143;
  barChart($('#chLeak'), {
    categories: L.bins.map(b => T('leak_bin', { b })), unit: ' %', yMin: 0, yMax: 100, tickUnit: '%', height: 250, tableLabel: T('table'),
    series: [
      { name: T('leak_s1'), color: C.raw, values: L.tests.raw.by_ratio },
      { name: T('leak_s2'), color: C.naive, values: L.tests.naive.by_ratio },
      { name: T('leak_s3'), color: C.clean, values: L.tests.aware.by_ratio },
    ],
    note: T('leak_note', { raw: num(L.tests.raw.total), naive: num(L.tests.naive.total), aware: num(L.tests.aware.total) }),
  });
  trackCharts();
  const X = APP.cross_track, cls = ['рілля', 'дно водосховища', 'забудова'];
  barChart($('#chCross'), {
    categories: [T('c_crop'), T('c_bed'), T('c_urban')], unit: ' %', yMin: 0, tickUnit: '%', tableLabel: T('table'),
    series: [
      { name: T('cross_s1'), color: C.clean, values: cls.map(c => X[c]['в обох, %']) },
      { name: T('cross_s2'), color: C.base, values: cls.map(c => X[c]['очікувано при незалежності, %']) },
    ],
    note: T('cross_note', { a: num(X['рілля']['дата першої зміни збігається ±12 діб, %']), ar: num(X['рілля']['випадково (перемішування), %']),
      b: num(X['дно водосховища']['дата першої зміни збігається ±12 діб, %']), br: num(X['дно водосховища']['випадково (перемішування), %']) }),
  });
}

function trackCharts() {
  const t = state.chartTrack, q = APP.qc[t], P = state.pol, PU = P.toUpperCase();
  const keys = ['water_raw', 'water_clean', 'urban_raw', 'urban_clean'];
  const vals = keys.flatMap(k => q[`${P}_${k}`]).filter(v => v != null);
  lineChart($('#chProbe'), {
    x: q.dates.map(ddmm), xFull: q.dates, unit: ` ${T('unit_db')}`, tableLabel: T('table'), xLabel: T('x_date'),
    yMin: Math.min(-24, Math.floor(Math.min(...vals) - 1)), yMax: Math.ceil(Math.max(...vals) + 1),
    band: { from: -40, to: -20, label: T('probe_band') },
    series: [
      { name: `${T('s_water_raw')} ${PU}`, color: C.raw, values: q[`${P}_water_raw`] },
      { name: `${T('s_water_clean')} ${PU}`, color: C.clean, values: q[`${P}_water_clean`] },
      { name: `${T('s_urban_raw')} ${PU}`, color: C.raw, dash: true, values: q[`${P}_urban_raw`] },
      { name: `${T('s_urban_clean')} ${PU}`, color: C.clean, dash: true, values: q[`${P}_urban_clean`] },
    ],
  });
  const S = ['рілля', 'дно'].flatMap(z => ['HH', 'HV'].map(p => APP.contrast.find(c => c.zone === z && c.track === t && c.pol === p)));
  barChart($('#chShift'), {
    categories: S.map(c => `${T(c.zone === 'рілля' ? 'z_crop' : 'z_bed')}\n${c.pol}`), fmt: v => `${num(sgn(v))} ${T('unit_db')}`,
    showValues: true, valueLabel: v => num(sgn(v)), tableLabel: T('table'),
    series: [{ name: T('s_raw'), color: C.raw, values: S.map(c => c.raw) }, { name: T('s_clean'), color: C.clean, values: S.map(c => c.clean) }],
  });
  const ch = APP.change[t], when = ch['перша зміна, км²'];
  barChart($('#chWhen'), {
    categories: Object.keys(when).map(ddmm), unit: ` ${T('km2')}`, yMin: 0, tableLabel: T('table'),
    series: [{ name: T('when_s'), color: C.clean, values: Object.values(when) }],
    note: T('when_note', { km: ch['змінилось, км²'].toLocaleString('en'), pct: num(ch['змінилось, %']) }),
  });
  const B = ch['дно проти DSWx-S1'], bk = ['сухе весь сезон', 'вода весь сезон', 'перехід суша↔вода'];
  barChart($('#chBed'), {
    categories: [T('bed_c1'), T('bed_c2'), T('bed_c3')], unit: ' %', yMin: 0, tickUnit: '%', showValues: true, valueLabel: v => `${num(v)}%`, tableLabel: T('table'),
    series: [{ name: T('bed_s'), color: C.clean, values: bk.map(k => B[k]['змінилось, %']) }],
    note: T('bed_note', { p: num(B['дата зміни = перший перехід DSWx ±1 дата, %']) }),
  });
}

// ---------- класи ----------
function classes() {
  const t = state.classTrack, P = state.classPol, PU = P.toUpperCase(), K = APP.classes, dates = APP.tracks[t].dates;
  $('#classKey').innerHTML = `<span><i class="l" style="border-color:${C.raw}"></i>${T('key_raw')}</span>
    <span><i class="l" style="border-color:${C.clean}"></i>${T('key_clean')}</span><span><i class="b"></i>${T('key_corrupt')}</span>`;
  const grid = $('#classGrid'); grid.replaceChildren();
  for (const c of K.order) {
    const r = K[t][c];
    // власна шкала класу з однаковим запасом: видно часову динаміку, а підписи осі показують рівень
    const v = [...r[`${P}_raw`], ...r[`${P}_clean`]].filter(x => x != null);
    const yMin = Math.floor(Math.min(...v) - 1.5), yMax = Math.ceil(Math.max(...v) + 1.5);
    const card = document.createElement('article'); card.className = 'cls';
    card.innerHTML = `<header><h3><i style="background:${K.colors[c]}"></i>${T(`cls_${c}`)}</h3>
      <span class="meta">${T('cls_meta', { km: r.area_km2.toLocaleString('en'), pct: r.changed_pct == null ? '—' : num(r.changed_pct) })}</span></header>`;
    const ch = document.createElement('div'); ch.className = 'chart'; card.appendChild(ch);
    grid.appendChild(card);
    lineChart(ch, {
      x: dates.map(ddmm), xFull: dates, unit: ` ${T('unit_db')}`, yMin, yMax, height: 140, compact: true, noLegend: true, noTable: true,
      series: [{ name: `${T('key_raw')} ${PU}`, color: C.raw, values: r[`${P}_raw`] }, { name: `${T('key_clean')} ${PU}`, color: C.clean, values: r[`${P}_clean`] }],
    });
    strip(card, r[`${P}_corrupt`], dates, T('cls_strip'));
    const sr = r[`${P}_raw`].filter(v => v != null), sc = r[`${P}_clean`].filter(v => v != null);
    const sd = a => Math.sqrt(a.reduce((s, v) => s + (v - a.reduce((x, y) => x + y, 0) / a.length) ** 2, 0) / a.length);
    const d = document.createElement('div'); d.className = 'delta';
    d.textContent = `σ(t) ${PU}: ${num(sd(sr).toFixed(2))} → ${num(sd(sc).toFixed(2))} ${T('unit_db')}`;
    card.appendChild(d);
  }
}

// ---------- старт ----------
async function main() {
  applyLang();
  APP = await (await fetch('data/app.json')).json();
  $('#generated').textContent = APP.generated;
  buildTrackSegs();
  bindSeg($('#chartPol'), v => { state.pol = v; trackCharts(); });
  bindSeg($('#classPol'), v => { state.classPol = v; classes(); });
  bindSeg($('#modeSeg'), v => { state.mode = v; render(); });
  bindSeg($('#baseSeg'), setBase);
  $('#rfiToggle').addEventListener('change', e => { state.rfi = e.target.checked; render(); });
  $('#opacity').addEventListener('input', e => {
    state.opacity = e.target.value / 100;
    for (const m of [mapA, mapB]) if (m.getLayer('sar')) m.setPaintProperty('sar', 'raster-opacity', state.mode === 'compare' ? state.opacity : state.opacity * 0.55);
  });
  figures(); charts(); classes(); buildDates();
  await initMaps();
  initSwipe();
  state.date = APP.tracks[state.track].dates[2];
  render();
}
main().catch(e => { console.error(e); const l = $('#loading'); l.hidden = false; l.textContent = `${T('load_err')}: ${e.message}`; });
