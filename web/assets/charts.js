// Маленька бібліотека SVG-графіків: згруповані стовпці й лінії, підказка при наведенні, таблиця даних.
const NS = 'http://www.w3.org/2000/svg';
const tip = () => document.getElementById('tooltip');

function el(tag, attrs = {}, parent) {
  const e = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
  if (parent) parent.appendChild(e);
  return e;
}

function niceTicks(lo, hi, n = 5) {
  const span = hi - lo || 1, step0 = span / n, mag = 10 ** Math.floor(Math.log10(step0));
  const step = [1, 2, 2.5, 5, 10].map(m => m * mag).find(s => s >= step0);
  const out = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) out.push(+v.toFixed(6));
  return out;
}

function showTip(evt, head, rows) {
  const t = tip(); t.replaceChildren();
  const h = document.createElement('div'); h.className = 'tt-h'; h.textContent = head; t.appendChild(h);
  for (const r of rows) {
    const row = document.createElement('div'); row.className = 'tt-r';
    const i = document.createElement('i'); i.style.borderTopColor = r.color; if (r.dash) i.style.borderTopStyle = 'dashed';
    const b = document.createElement('b'); b.textContent = r.value;
    const s = document.createElement('span'); s.textContent = r.name;
    row.append(i, b, s); t.appendChild(row);
  }
  t.hidden = false;
  const pad = 14, w = t.offsetWidth, hgt = t.offsetHeight;
  let x = evt.clientX + pad, y = evt.clientY + pad;
  if (x + w > innerWidth - 8) x = evt.clientX - w - pad;
  if (y + hgt > innerHeight - 8) y = evt.clientY - hgt - pad;
  t.style.left = x + 'px'; t.style.top = y + 'px';
}
const hideTip = () => { tip().hidden = true; };

function legend(container, series, kind) {
  if (series.length < 2) return;
  const lg = document.createElement('div'); lg.className = 'chart-legend';
  for (const s of series) {
    const sp = document.createElement('span'); const i = document.createElement('i');
    if (kind === 'line') { i.className = 'line' + (s.dash ? ' dash' : ''); i.style.borderTopColor = s.color; }
    else i.style.background = s.color;
    const tx = document.createElement('span'); tx.textContent = s.name;
    sp.append(i, tx); lg.appendChild(sp);
  }
  container.appendChild(lg);
}

function table(container, head, rows, label = 'Data table') {
  const d = document.createElement('details'); d.className = 'table';
  const s = document.createElement('summary'); s.textContent = label; d.appendChild(s);
  const tb = document.createElement('table'); const tr = document.createElement('tr');
  for (const h of head) { const th = document.createElement('th'); th.textContent = h; tr.appendChild(th); }
  tb.appendChild(tr);
  for (const r of rows) {
    const row = document.createElement('tr');
    for (const c of r) { const td = document.createElement('td'); td.textContent = c; row.appendChild(td); }
    tb.appendChild(row);
  }
  d.appendChild(tb); container.appendChild(d);
}

function responsive(container, draw) {
  let last = 0;
  const ro = new ResizeObserver(() => {
    const w = Math.round(container.clientWidth);
    if (w && Math.abs(w - last) > 4) { last = w; draw(w); }
  });
  ro.observe(container);
  draw(container.clientWidth || 600);
  return ro;
}

/** Згруповані стовпці. cfg: {categories, series:[{name,color,values}], unit, fmt, yMin, yMax, height, note} */
export function barChart(container, cfg) {
  container.replaceChildren();
  legend(container, cfg.series, 'bar');
  const host = document.createElement('div'); container.appendChild(host);
  const fmt = cfg.fmt || (v => v == null ? '—' : `${v}${cfg.unit || ''}`);
  const all = cfg.series.flatMap(s => s.values).filter(v => v != null);
  const lo = Math.min(0, ...all), hi = Math.max(0, ...all), span = (hi - lo) || 1;
  // запас під підписи значень: зверху для додатних, знизу для від'ємних стовпців
  const yMin = cfg.yMin ?? (lo < 0 ? lo - span * 0.12 : 0), yMax = cfg.yMax ?? hi + span * 0.12;
  responsive(host, W => {
    host.replaceChildren();
    const H = cfg.height || 240, m = { l: 40, r: 8, t: 14, b: 38 };
    const svg = el('svg', { viewBox: `0 0 ${W} ${H}`, role: 'img', 'aria-label': cfg.label || '' }, host);
    const iw = W - m.l - m.r, ih = H - m.t - m.b;
    const y = v => m.t + ih * (1 - (v - yMin) / (yMax - yMin));
    const ticks = niceTicks(yMin, yMax, 4);
    const g = el('g', { class: 'axis' }, svg);
    for (const t of ticks) {
      el('line', { x1: m.l, x2: W - m.r, y1: y(t), y2: y(t), class: t === 0 ? 'zero' : 'grid-line' }, g);
      el('text', { x: m.l - 6, y: y(t) + 4, 'text-anchor': 'end' }, g).textContent = `${t}${cfg.tickUnit ?? ''}`;
    }
    const nC = cfg.categories.length, nS = cfg.series.length;
    const band = iw / nC, inner = Math.min(band * 0.78, nS * 34), bw = (inner - (nS - 1) * 2) / nS;
    cfg.categories.forEach((c, ci) => {
      const x0 = m.l + ci * band + (band - inner) / 2;
      const lab = el('text', { x: m.l + ci * band + band / 2, y: H - m.b + 16, 'text-anchor': 'middle', class: 'lbl-t' }, svg);
      String(c).split('\n').forEach((line, li) => { const ts = el('tspan', { x: m.l + ci * band + band / 2, dy: li ? 13 : 0 }, lab); ts.textContent = line; });
      const bars = [];
      cfg.series.forEach((s, si) => {
        const v = s.values[ci]; if (v == null) return;
        const x = x0 + si * (bw + 2), y0 = y(0), y1 = y(v), top = Math.min(y0, y1), h = Math.max(Math.abs(y1 - y0), 1);
        const r = Math.min(4, bw / 2, h);
        const up = v >= 0;
        const d = up
          ? `M${x},${y0} V${top + r} Q${x},${top} ${x + r},${top} H${x + bw - r} Q${x + bw},${top} ${x + bw},${top + r} V${y0} Z`
          : `M${x},${y0} V${top + h - r} Q${x},${top + h} ${x + r},${top + h} H${x + bw - r} Q${x + bw},${top + h} ${x + bw},${top + h - r} V${y0} Z`;
        bars.push(el('path', { d, fill: s.color, class: 'bar' }, svg));
        if (cfg.showValues) {
          const ty = up ? top - 4 : top + h + 12;
          el('text', { x: x + bw / 2, y: ty, 'text-anchor': 'middle', class: 'val' }, svg).textContent = cfg.valueLabel ? cfg.valueLabel(v) : v;
        }
      });
      const hit = el('rect', { x: m.l + ci * band, y: m.t, width: band, height: ih, class: 'hit', tabindex: 0 }, svg);
      const rows = () => cfg.series.map(s => ({ name: s.name, color: s.color, value: fmt(s.values[ci]) }));
      const on = e => { bars.forEach(b => b.classList.remove('dim')); svg.querySelectorAll('.bar').forEach(b => { if (!bars.includes(b)) b.classList.add('dim'); });
        showTip(e.clientX ? e : { clientX: hit.getBoundingClientRect().right, clientY: hit.getBoundingClientRect().top }, String(c).replace('\n', ' '), rows()); };
      const off = () => { svg.querySelectorAll('.bar').forEach(b => b.classList.remove('dim')); hideTip(); };
      hit.addEventListener('pointermove', on); hit.addEventListener('pointerleave', off);
      hit.addEventListener('focus', on); hit.addEventListener('blur', off);
    });
  });
  if (cfg.note) { const n = document.createElement('p'); n.className = 'chart-note'; n.innerHTML = cfg.note; container.appendChild(n); }
  table(container, ['', ...cfg.series.map(s => s.name)],
    cfg.categories.map((c, i) => [String(c).replace('\n', ' '), ...cfg.series.map(s => fmt(s.values[i]))]), cfg.tableLabel);
}

/** Лінії з перехрестям. cfg: {x:[labels], series:[{name,color,dash,values}], unit, yMin, yMax, band:{from,to,label}} */
export function lineChart(container, cfg) {
  container.replaceChildren();
  if (!cfg.noLegend) legend(container, cfg.series, 'line');
  const host = document.createElement('div'); container.appendChild(host);
  const fmt = v => v == null ? '—' : `${v.toFixed(1)}${cfg.unit || ''}`;
  const all = cfg.series.flatMap(s => s.values).filter(v => v != null);
  const yMin = cfg.yMin ?? Math.floor(Math.min(...all) - 1), yMax = cfg.yMax ?? Math.ceil(Math.max(...all) + 1);
  responsive(host, W => {
    host.replaceChildren();
    const H = cfg.height || 250, m = cfg.compact ? { l: 30, r: 6, t: 8, b: 20 } : { l: 40, r: 10, t: 12, b: 28 };
    const svg = el('svg', { viewBox: `0 0 ${W} ${H}`, role: 'img', 'aria-label': cfg.label || '' }, host);
    const iw = W - m.l - m.r, ih = H - m.t - m.b, n = cfg.x.length;
    const x = i => m.l + (n === 1 ? iw / 2 : iw * i / (n - 1));
    const y = v => m.t + ih * (1 - (v - yMin) / (yMax - yMin));
    if (cfg.band) {
      const a = y(Math.min(cfg.band.to, yMax)), b = y(Math.max(cfg.band.from, yMin));
      el('rect', { x: m.l, y: a, width: iw, height: b - a, class: 'band' }, svg);
      el('text', { x: m.l + 6, y: b - 6, class: 'band-lbl' }, svg).textContent = cfg.band.label;
    }
    const g = el('g', { class: 'axis' }, svg);
    for (const t of niceTicks(yMin, yMax, cfg.compact ? 3 : 5)) {
      if (cfg.compact && (y(t) > m.t + ih - 6 || y(t) < m.t + 4)) continue;
      el('line', { x1: m.l, x2: W - m.r, y1: y(t), y2: y(t), class: 'grid-line' }, g);
      el('text', { x: m.l - 6, y: y(t) + 4, 'text-anchor': 'end' }, g).textContent = t;
    }
    const every = Math.ceil(n / Math.max(2, Math.floor(iw / (cfg.compact ? 70 : 58))));
    cfg.x.forEach((lab, i) => { if (i % every === 0 || i === n - 1) el('text', { x: x(i), y: H - 8, 'text-anchor': 'middle' }, g).textContent = lab; });
    for (const s of cfg.series) {
      let d = '', pen = false;
      s.values.forEach((v, i) => { if (v == null) { pen = false; return; } d += `${pen ? 'L' : 'M'}${x(i)},${y(v)}`; pen = true; });
      el('path', { d, fill: 'none', stroke: s.color, 'stroke-width': 2, 'stroke-dasharray': s.dash ? '5 4' : 'none', 'stroke-linejoin': 'round' }, svg);
      const r = cfg.compact ? 2.5 : 4;
      s.values.forEach((v, i) => { if (v != null) el('circle', { cx: x(i), cy: y(v), r, fill: s.dash ? 'var(--surface-1)' : s.color, stroke: s.color, 'stroke-width': cfg.compact ? 1.5 : 2 }, svg); });
    }
    const xh = el('line', { y1: m.t, y2: m.t + ih, class: 'xhair', visibility: 'hidden' }, svg);
    const hit = el('rect', { x: m.l - 10, y: 0, width: iw + 20, height: H, class: 'hit', tabindex: 0 }, svg);
    const at = i => { xh.setAttribute('x1', x(i)); xh.setAttribute('x2', x(i)); xh.setAttribute('visibility', 'visible'); };
    hit.addEventListener('pointermove', e => {
      const r = svg.getBoundingClientRect(), px = (e.clientX - r.left) * W / r.width;
      const i = Math.max(0, Math.min(n - 1, Math.round((px - m.l) / (iw / Math.max(n - 1, 1)))));
      at(i); showTip(e, cfg.xFull ? cfg.xFull[i] : cfg.x[i], cfg.series.map(s => ({ name: s.name, color: s.color, dash: s.dash, value: fmt(s.values[i]) })));
    });
    hit.addEventListener('pointerleave', () => { xh.setAttribute('visibility', 'hidden'); hideTip(); });
  });
  if (!cfg.noTable) table(container, [cfg.xLabel || 'date', ...cfg.series.map(s => s.name)],
    cfg.x.map((lab, i) => [cfg.xFull ? cfg.xFull[i] : lab, ...cfg.series.map(s => fmt(s.values[i]))]), cfg.tableLabel);
}

/** Смуга-теплокарта: values 0..100 (%), колір від прозорого до червоного; підказка з датою і значенням. */
export function strip(container, values, labels, title) {
  const g = document.createElement('div'); g.className = 'strip'; g.style.gridTemplateColumns = `repeat(${values.length}, 1fr)`;
  values.forEach((v, i) => {
    const c = document.createElement('div');
    const a = v == null ? 0 : Math.min(1, Math.sqrt(v / 60));
    c.style.background = v == null ? 'var(--line)' : `rgba(235,64,52,${0.08 + 0.92 * a})`;
    c.addEventListener('pointermove', e => showTip(e, labels[i], [{ name: title, color: '#eb4034', value: v == null ? '—' : `${v} %` }]));
    c.addEventListener('pointerleave', hideTip);
    g.appendChild(c);
  });
  container.appendChild(g);
}
