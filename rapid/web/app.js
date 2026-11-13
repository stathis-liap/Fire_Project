// WILSON Rapid — field UI.  Vanilla JS over the C++ backend's JSON API.
'use strict';

const $ = (s) => document.querySelector(s);
const $$ = (s) => Array.from(document.querySelectorAll(s));

const state = {
  result: null,      // latest /api/result
  version: -1,
  view: 'base',      // 'base' | 'plan'
  mode: 'zones',     // 'zones' (most likely) | 'prob' (chance-of-fire heat map) | 'danger'
  probHour: 2,       // index (1-based) into result.frames for the heat map
  showWind: false,
  planNo: {},        // recommendation id → number shown in the plan
  pending: null,     // {lat, lon} pin before the fire is started
  tool: null,        // active drawing tool
  draft: [],         // points being drawn [lon, lat]
  startAgo: 0,
  startHours: 6,
  windDir: 0,
  busy: false,
};

// ── Formatting ───────────────────────────────────────────────────────────
const clock = (epoch) => new Date(epoch * 1000).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', hourCycle: 'h23' });
// Engine text is translated first, then its {t:epoch} tokens become local clock times.
const tokens = (s) => I18N.tr(s || '').replace(/\{t:(\d+)\}/g, (_, e) => clock(+e));
function inMinutes(epoch) {
  const m = Math.round((epoch * 1000 - Date.now()) / 60000);
  if (m <= 0) return t('now');
  if (m < 60) return I18N.tr(`${Math.max(5, Math.round(m / 5) * 5)} min`);
  const h = Math.floor(m / 60), r = Math.round((m % 60) / 5) * 5;
  return I18N.tr(`${h} h${r ? ' ' + r + ' min' : ''}`);
}
function inDur(m) {
  m = Math.round(m / 5) * 5;
  return I18N.tr(m < 60 ? `${Math.max(5, m)} min` : `${Math.floor(m / 60)} h${m % 60 ? ' ' + (m % 60) + ' min' : ''}`);
}
const esc = (s) => String(s ?? '').replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
const gradeText = new Proxy({ high: 'High', medium: 'Medium', low: 'Low' }, { get: (o, k) => t(o[k]) });

function toast(msg, err = false) {
  const t = $('#toast');
  t.textContent = msg;
  t.className = err ? 'err' : '';
  clearTimeout(toast._t);
  toast._t = setTimeout(() => t.classList.add('hidden'), err ? 6000 : 3500);
}

// ?debug mirrors console warnings/errors on screen (for field troubleshooting).
if (location.search.includes('debug')) {
  const box = document.createElement('pre');
  box.style.cssText = 'position:absolute;left:150px;top:80px;z-index:99;max-width:60%;font-size:12px;color:#fbb;background:#000c;white-space:pre-wrap;padding:6px';
  document.body.append(box);
  for (const k of ['warn', 'error']) {
    const orig = console[k].bind(console);
    console[k] = (...a) => { box.textContent += a.map(String).join(' ') + '\n'; orig(...a); };
  }
}
window.addEventListener('error', (e) => toast(t('Display error: {m}', { m: `${e.message} (line ${e.lineno})` }), true));
window.addEventListener('unhandledrejection', (e) => toast(t('Display error: {m}', { m: e.reason?.message || e.reason }), true));

// ── API ──────────────────────────────────────────────────────────────────
async function api(path, body) {
  const opt = body === undefined ? {} : { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) };
  const res = await fetch(path, opt);
  const j = await res.json().catch(() => ({ error: t('Server did not answer.') }));
  if (j.error) throw new Error(I18N.tr(j.error));
  return j;
}

async function busy(text, fn) {
  state.busy = true;
  $('#busy-text').textContent = t(text);
  $('#busy').classList.remove('hidden');
  const poll = setInterval(async () => {
    try { const s = await api('/api/status'); if (s.status && s.status !== 'ready' && s.status !== 'idle') $('#busy-text').textContent = I18N.tr(s.status); } catch (_) {}
  }, 700);
  try { return await fn(); }
  catch (e) { toast(e.message, true); }
  finally { clearInterval(poll); $('#busy').classList.add('hidden'); state.busy = false; }
}

// ── Map ──────────────────────────────────────────────────────────────────
const map = new maplibregl.Map({
  container: 'map',
  style: {
    version: 8,
    glyphs: 'https://fonts.openmaptiles.org/{fontstack}/{range}.pbf',
    sources: {
      sat: { type: 'raster', tileSize: 256, maxzoom: 18, attribution: 'Imagery © Esri, Maxar, Earthstar',
        tiles: ['https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}'] },
      labels: { type: 'raster', tileSize: 256, maxzoom: 18,
        tiles: ['https://server.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}'] },
      roads: { type: 'raster', tileSize: 256, maxzoom: 18,
        tiles: ['https://server.arcgisonline.com/ArcGIS/rest/services/Reference/World_Transportation/MapServer/tile/{z}/{y}/{x}'] },
    },
    layers: [
      { id: 'bg', type: 'background', paint: { 'background-color': '#1b1f24' } },
      { id: 'sat', type: 'raster', source: 'sat', paint: { 'raster-saturation': -0.25, 'raster-brightness-max': 0.85 } },
      { id: 'roads', type: 'raster', source: 'roads', paint: { 'raster-opacity': 0.55 } },
      { id: 'labels', type: 'raster', source: 'labels' },
    ],
  },
  center: [22.5, 38.6],
  zoom: 7,
  attributionControl: { compact: true },
  preserveDrawingBuffer: true,  // lets crews screenshot / print the map
});
map.on('error', (e) => console.warn('map:', e.error?.message || e));
window.wilson = { map, state };  // handle for debugging / automation
if (location.search.includes('debug')) {
  map.on('webglcontextlost', () => console.warn('webgl context lost'));
  let ev = {};
  for (const n of ['styledata', 'sourcedata', 'dataloading', 'render', 'idle', 'load']) map.on(n, () => (ev[n] = (ev[n] || 0) + 1));
  setTimeout(() => { const st = map.style; console.warn('events', JSON.stringify(ev), 'style._loaded', st?._loaded,
    'srcs', st && JSON.stringify(Object.fromEntries(Object.entries(st.sourceCaches || {}).map(([k, v]) => [k, v.loaded()]))),
    'images', st?.imageManager?.isLoaded?.(), 'glyphs', st?.glyphManager?.url); }, 11000);
  setTimeout(() => console.warn('map status: loaded', map.loaded(), 'style', map.isStyleLoaded(), 'ready', mapReady,
    'gl', map.painter?.context?.gl?.constructor?.name, 'size', map.getCanvas().width, map.getCanvas().height), 12000);
}
map.addControl(new maplibregl.NavigationControl({ showCompass: true }), 'bottom-right');
map.addControl(new maplibregl.GeolocateControl({ positionOptions: { enableHighAccuracy: true } }), 'bottom-right');
map.addControl(new maplibregl.ScaleControl({ unit: 'metric' }), 'bottom-left');

const EMPTY = { type: 'FeatureCollection', features: [] };
const svgImg = (svg, size) => new Promise((res) => {
  const img = new Image(size, size);
  img.onload = () => res(img);
  img.src = 'data:image/svg+xml;charset=utf-8,' + encodeURIComponent(svg);
});

map.on('load', async () => {
  // Plain operational symbology: lettered markers, no pictograms.
  const marker = (fill, letter) => [`<svg xmlns="http://www.w3.org/2000/svg" width="36" height="36" viewBox="0 0 36 36"><rect x="3" y="3" width="30" height="30" fill="${fill}" stroke="#fff" stroke-width="2.5"/><text x="18" y="24.5" font-family="Arial, sans-serif" font-weight="700" font-size="17" fill="#fff" text-anchor="middle">${letter}</text></svg>`, 36];
  const icons = {
    'arrow-head': ['<svg xmlns="http://www.w3.org/2000/svg" width="40" height="40" viewBox="0 0 40 40"><path d="M20 3 L35 35 L20 27 L5 35 Z" fill="#d63a1e" stroke="#fff" stroke-width="2.5" stroke-linejoin="miter"/></svg>', 40],
    'arrow-front': ['<svg xmlns="http://www.w3.org/2000/svg" width="28" height="28" viewBox="0 0 28 28"><path d="M14 3 L24 24 L14 19 L4 24 Z" fill="#e07b28" stroke="#fff" stroke-width="2" stroke-linejoin="miter"/></svg>', 28],
    truck: marker('#b3302c', 'E'),
    drop: marker('#2f6aa8', 'A'),
    evac: marker('#6f3d9e', 'X'),
    dozer: marker('#a8700f', 'D'),
    fire: ['<svg xmlns="http://www.w3.org/2000/svg" width="30" height="30" viewBox="0 0 30 30"><path d="M15 2 L28 15 L15 28 L2 15 Z" fill="#d63a1e" stroke="#fff" stroke-width="2.5"/><circle cx="15" cy="15" r="3" fill="#fff"/></svg>', 30],
    wind: ['<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24"><path d="M12 2 L18 12 L13.5 11 L13.5 22 L10.5 22 L10.5 11 L6 12 Z" fill="#b9d3ee" stroke="#0b1a2e" stroke-width="1.2"/></svg>', 24],
    house: ['<svg xmlns="http://www.w3.org/2000/svg" width="22" height="22" viewBox="0 0 22 22"><rect x="4" y="4" width="14" height="14" fill="#fff" stroke="#111" stroke-width="2"/></svg>', 22],
  };
  for (const [name, [svg, size]] of Object.entries(icons)) map.addImage(name, await svgImg(svg, size), { pixelRatio: 2 });

  const src = (id) => map.addSource(id, { type: 'geojson', data: EMPTY });
  ['iso', 'arrows', 'observed', 'recs', 'manual', 'recs-pts', 'manual-pts', 'places', 'draft', 'pin', 'wind'].forEach(src);
  map.addSource('zones', { type: 'image', url: 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII=',
    coordinates: [[0, 1], [1, 1], [1, 0], [0, 0]] });

  map.addLayer({ id: 'zones', type: 'raster', source: 'zones', paint: { 'raster-opacity': 0.85, 'raster-resampling': 'nearest', 'raster-fade-duration': 0 } });
  map.addLayer({ id: 'observed-fill', type: 'fill', source: 'observed', filter: ['==', '$type', 'Polygon'], paint: { 'fill-color': '#000', 'fill-opacity': 0.25 } });
  map.addLayer({ id: 'observed-line', type: 'line', source: 'observed', filter: ['==', '$type', 'Polygon'], paint: { 'line-color': '#fff', 'line-width': 2, 'line-dasharray': [2, 1] } });
  map.addLayer({ id: 'iso-halo', type: 'line', source: 'iso', paint: { 'line-color': '#000', 'line-width': 4, 'line-opacity': 0.55 } });
  map.addLayer({ id: 'iso', type: 'line', source: 'iso', paint: {
    'line-color': ['case', ['==', ['get', 'hours'], 0], '#ff3b1f', '#fff'],
    'line-width': ['case', ['==', ['get', 'hours'], 0], 3, 1.6] } });
  map.addLayer({ id: 'iso-label', type: 'symbol', source: 'iso', layout: {
    'symbol-placement': 'line', 'text-field': ['get', 'label'], 'text-font': ['Open Sans Bold'], 'text-size': 13,
    'symbol-spacing': 300, 'text-keep-upright': true }, paint: { 'text-color': '#fff', 'text-halo-color': '#000', 'text-halo-width': 2 } });
  map.addLayer({ id: 'arrows-line', type: 'line', source: 'arrows', filter: ['==', '$type', 'LineString'], layout: { 'line-cap': 'round', 'line-join': 'round' },
    paint: { 'line-color': ['case', ['==', ['get', 'kind'], 'head'], '#ff2a1a', '#ff8a1a'], 'line-width': ['case', ['==', ['get', 'kind'], 'head'], 6, 3.5], 'line-opacity': 0.95 } });
  map.addLayer({ id: 'arrows-head', type: 'symbol', source: 'arrows', filter: ['==', '$type', 'Point'], layout: {
    'icon-image': ['case', ['==', ['get', 'kind'], 'head'], 'arrow-head', 'arrow-front'], 'icon-rotate': ['get', 'bearing'],
    'icon-rotation-alignment': 'map', 'icon-allow-overlap': true } });
  map.addLayer({ id: 'wind', type: 'symbol', source: 'wind', layout: {
    'icon-image': 'wind', 'icon-rotate': ['get', 'to'], 'icon-rotation-alignment': 'map', 'icon-allow-overlap': true,
    'icon-size': ['interpolate', ['linear'], ['get', 'kmh'], 0, 0.55, 15, 0.85, 40, 1.4],
    'text-field': ['to-string', ['get', 'kmh']], 'text-font': ['Open Sans Bold'], 'text-size': 10, 'text-offset': [0, 1.3],
    'text-allow-overlap': false }, paint: { 'icon-opacity': 0.85, 'text-color': '#cfe5ff', 'text-halo-color': '#000', 'text-halo-width': 1.5 } });
  map.addLayer({ id: 'observed-pts', type: 'circle', source: 'observed', filter: ['==', '$type', 'Point'], paint: { 'circle-radius': 5, 'circle-color': '#ff7a00', 'circle-stroke-color': '#fff', 'circle-stroke-width': 1.5 } });

  // Recommendations & own actions
  const lineCol = ['match', ['get', 'type'], 'firebreak', '#f59e0b', 'air_drop', '#3b82f6', '#fff'];
  for (const id of ['recs', 'manual']) {
    map.addLayer({ id: `${id}-casing`, type: 'line', source: id, filter: ['==', '$type', 'LineString'], layout: { 'line-cap': 'round' },
      paint: { 'line-color': '#000', 'line-width': ['match', ['get', 'type'], 'air_drop', 13, 9], 'line-opacity': ['case', ['get', 'enabled'], 0.7, 0.25] } });
    map.addLayer({ id: `${id}-line`, type: 'line', source: id, filter: ['==', '$type', 'LineString'], layout: { 'line-cap': 'round' },
      paint: { 'line-color': lineCol, 'line-width': ['match', ['get', 'type'], 'air_drop', 9, 5], 'line-opacity': ['case', ['get', 'enabled'], 1, 0.35],
        'line-dasharray': ['literal', id === 'manual' ? [1, 1] : [1, 0]] } });
    map.addLayer({ id: `${id}-icon`, type: 'symbol', source: `${id}-pts`, layout: {
      'icon-image': ['match', ['get', 'type'], 'truck', 'truck', 'air_drop', 'drop', 'evacuate', 'evac', 'firebreak', 'dozer', 'truck'],
      'icon-allow-overlap': true, 'text-allow-overlap': true, 'text-field': ['get', 'label'], 'text-font': ['Open Sans Bold'],
      'text-size': 13, 'text-offset': [0, 1.4], 'text-anchor': 'top' },
      paint: { 'icon-opacity': ['case', ['get', 'enabled'], 1, 0.4], 'text-color': '#fff', 'text-halo-color': '#000', 'text-halo-width': 2.5 } });
  }
  map.addLayer({ id: 'places', type: 'symbol', source: 'places', layout: {
    'icon-image': 'house', 'icon-allow-overlap': true, 'text-field': ['get', 'label'], 'text-font': ['Open Sans Bold'], 'text-size': 13,
    'text-offset': [0, 1.2], 'text-anchor': 'top', 'text-allow-overlap': false },
    paint: { 'text-color': ['get', 'color'], 'text-halo-color': '#000', 'text-halo-width': 2 } });
  map.addLayer({ id: 'draft-line', type: 'line', source: 'draft', paint: { 'line-color': '#fff', 'line-width': 3, 'line-dasharray': [2, 1] } });
  map.addLayer({ id: 'draft-pts', type: 'circle', source: 'draft', filter: ['==', '$type', 'Point'], paint: { 'circle-radius': 7, 'circle-color': '#ff6a1a', 'circle-stroke-color': '#fff', 'circle-stroke-width': 2 } });
  map.addLayer({ id: 'pin', type: 'symbol', source: 'pin', layout: { 'icon-image': 'fire', 'icon-allow-overlap': true } });

  for (const l of ['recs-line', 'recs-icon']) {
    map.on('mouseenter', l, () => (map.getCanvas().style.cursor = 'pointer'));
    map.on('mouseleave', l, () => (map.getCanvas().style.cursor = ''));
  }
  mapReady = true;
  if (state.result) updateMap(state.result, true);
  else {
    const cur = await api('/api/result').catch(() => null);
    if (cur && cur.status === 'ready') render(cur, true);
  }
});
// Results are shown even if the map cannot start (no WebGL, tiles offline).
api('/api/result').then((cur) => { if (cur && cur.status === 'ready' && !state.result) render(cur, true); })
  .catch((e) => console.warn('initial load:', e.stack || e));

// ── Map clicks ───────────────────────────────────────────────────────────
map.on('click', async (e) => {
  if (state.busy) return;
  const ll = [e.lngLat.lng, e.lngLat.lat];
  if (state.tool) return addDraftPoint(ll);
  if (!state.result) return askStart(e.lngLat.lat, e.lngLat.lng);

  const hit = map.queryRenderedFeatures(e.point, { layers: ['recs-line', 'recs-icon', 'manual-line', 'manual-icon'] })[0];
  if (hit) return showAction(hit.properties.id, e.lngLat);
  showPoint(e.lngLat);
});

async function showPoint(lngLat) {
  let p;
  try { p = await api(`/api/point?lat=${lngLat.lat}&lon=${lngLat.lng}&plan=${state.view === 'plan' ? 1 : 0}`); }
  catch (e) { return toast(e.message, true); }
  let html;
  if (!p.inside) html = `<b>${t('Outside the forecast area.')}</b>`;
  else {
    let eta;
    if (p.burning_now) eta = `<div class="pop-eta">${t('Burning')}</div>`;
    else if (p.eta_epoch) eta = `<div class="pop-eta">${t('Expected arrival {t}', { t: clock(p.eta_epoch) })}</div><div>${t('In {d}. Earliest {t}', { d: inMinutes(p.eta_epoch), t: p.earliest_epoch ? clock(p.earliest_epoch) : 'n/a' })}</div>`;
    else eta = `<div class="pop-eta">${t(p.prob > 0 ? 'Unlikely' : 'Not reached')}</div><div class="muted">${t('within the forecast window')}</div>`;
    const dg = p.danger;
    const dhtml = dg && dg.level ? `<div class="pop-danger"><b class="dtx-${dg.level}">${t('Danger: {c}', { c: I18N.danger(dg.level) })}</b>.
      ${t('Potential spread {v} km/h towards the {dir}', { v: dg.spread_kmh, dir: esc(I18N.dir(dg.toward)) })}${dg.flame_m ? t(', flames ~{f} m', { f: dg.flame_m }) : ''}.
      ${dg.reaches ? (dg.reaches_min < 5 ? t('Inside / at the edge of {p}.', { p: `<b>${esc(dg.reaches)}</b>` }) :
        t('Would reach {p} in ~{d}.', { p: `<b>${esc(dg.reaches)}</b>`, d: inDur(dg.reaches_min) })) : t('No village within 6 h of fire travel.')}</div>` : '';
    html = `${eta}<div>${t('Probability of fire: {p}%', { p: p.prob })}</div>${dhtml}
      <div class="muted small">${t('{fuel}, slope {s}%, elevation {e} m', { fuel: esc(I18N.fuel(p.fuel)), s: p.slope_pct, e: p.elevation_m })}${p.flame_m ? t(', flame length ~{f} m', { f: p.flame_m }) : ''}</div>
      <div class="muted small">${p.lat.toFixed(5)}, ${p.lon.toFixed(5)}</div>
      <button class="btn" id="pop-protect">${t('Add as protected asset')}</button>`;
  }
  const pop = new maplibregl.Popup({ maxWidth: '300px' }).setLngLat(lngLat).setHTML(`<div class="pop">${html}</div>`).addTo(map);
  const b = document.getElementById('pop-protect');
  if (b) b.onclick = () => { pop.remove(); askName(lngLat); };
}

function showAction(id, lngLat) {
  const r = (state.result.recommendations || []).find((x) => x.id === id);
  if (!r) return;
  new maplibregl.Popup({ maxWidth: '320px' }).setLngLat(lngLat)
    .setHTML(`<div class="pop"><b>${esc(tokens(r.title))}</b><div class="muted small">${t('Start before {t}', { t: clock(r.act_before_epoch) })}</div><div class="small">${esc(tokens(r.detail))}</div></div>`)
    .addTo(map);
}

// ── Start a fire ─────────────────────────────────────────────────────────
function askStart(lat, lon) {
  state.pending = { lat, lon };
  map.getSource('pin').setData({ type: 'FeatureCollection', features: [{ type: 'Feature', geometry: { type: 'Point', coordinates: [lon, lat] }, properties: {} }] });
  $('#hint').classList.add('hidden');
  $('#dlg-start').classList.remove('hidden');
}

$('#start-ago').onclick = (e) => {
  const b = e.target.closest('button'); if (!b) return;
  $$('#start-ago button').forEach((x) => x.classList.toggle('on', x === b));
  state.startAgo = +b.dataset.ago;
};
$('#start-hours').onclick = (e) => {
  const b = e.target.closest('button'); if (!b) return;
  $$('#start-hours button').forEach((x) => x.classList.toggle('on', x === b));
  $('#start-hours-custom').value = '';
  state.startHours = +b.dataset.h;
};
const clampHours = (v) => Math.min(48, Math.max(0.5, Math.round(v * 2) / 2));
$('#start-hours-custom').oninput = (e) => {
  const v = parseFloat(e.target.value);
  if (!(v > 0)) return;
  $$('#start-hours button').forEach((x) => x.classList.remove('on'));
  state.startHours = clampHours(v);
};
$('#horizon-form').onsubmit = async (e) => {
  e.preventDefault();
  const v = parseFloat($('#horizon-input').value);
  if (!(v > 0)) return toast(t('Enter the forecast window in hours (0.5 to 48).'), true);
  const h = clampHours(v);
  const r = await busy(t('Forecasting the next {h} h', { h }), () => api('/api/refresh', { horizon_h: h }));
  if (r) render(r);
};
$('#show-wind').onchange = (e) => {
  state.showWind = e.target.checked;
  if (mapReady && state.result) updateMap(state.result, false);
};
$('#start-cancel').onclick = () => {
  $('#dlg-start').classList.add('hidden');
  if (!state.result) { $('#hint').classList.remove('hidden'); map.getSource('pin').setData(EMPTY); }
};
$('#start-go').onclick = async () => {
  $('#dlg-start').classList.add('hidden');
  const p = state.pending;
  const r = await busy('Loading weather, terrain and vegetation', () =>
    api('/api/fire', { lat: p.lat, lon: p.lon, started_min_ago: state.startAgo, horizon_h: state.startHours }));
  if (r) { state.view = 'base'; render(r, true); }
  else if (!state.result) $('#hint').classList.remove('hidden');
};

// ── Search: coordinates or place names ───────────────────────────────────
function parseCoords(s) {
  s = s.trim().toUpperCase().replace(/,/g, ' ').replace(/\s+/g, ' ');
  const dms = /(\d+(?:\.\d+)?)[°\s]+(\d+(?:\.\d+)?)?['′\s]*(\d+(?:\.\d+)?)?["″\s]*([NSEW])/g;
  const parts = [...s.matchAll(dms)];
  if (parts.length === 2) {
    const v = parts.map((m) => {
      let d = +m[1] + (+m[2] || 0) / 60 + (+m[3] || 0) / 3600;
      if (m[4] === 'S' || m[4] === 'W') d = -d;
      return { d, axis: m[4] === 'N' || m[4] === 'S' ? 'lat' : 'lon' };
    });
    const lat = v.find((x) => x.axis === 'lat'), lon = v.find((x) => x.axis === 'lon');
    if (lat && lon) return { lat: lat.d, lon: lon.d };
  }
  const nums = s.match(/-?\d+(?:\.\d+)?/g);
  if (nums && nums.length === 2 && /^[-\d.\s]+$/.test(s)) {
    let [a, b] = nums.map(Number);
    if (Math.abs(a) > 90 && Math.abs(b) <= 90) [a, b] = [b, a];
    if (Math.abs(a) <= 90 && Math.abs(b) <= 180) return { lat: a, lon: b };
  }
  return null;
}

$('#search').onsubmit = async (e) => {
  e.preventDefault();
  const q = $('#search-input').value;
  if (!q.trim()) return;
  let p = parseCoords(q);
  if (!p) {
    try {
      const r = await fetch(`https://nominatim.openstreetmap.org/search?format=json&limit=1&q=${encodeURIComponent(q)}`).then((x) => x.json());
      if (r[0]) p = { lat: +r[0].lat, lon: +r[0].lon };
    } catch (_) {}
  }
  if (!p) return toast(t('Location not found. Enter coordinates, e.g. 38.56, 21.94'), true);
  map.flyTo({ center: [p.lon, p.lat], zoom: Math.max(map.getZoom(), 13) });
  if (!state.result) askStart(p.lat, p.lon);
  else showPoint({ lat: p.lat, lng: p.lon });
};

// Parts of the screen that only make sense with an active incident.
const INCIDENT_UI = ['#left', '#dock', '#rt', '#legend', '#conf-pill', '#btn-new', '#btn-update', '#horizon-form', '#incident'];
document.body.classList.add('no-incident');

$('#btn-new').onclick = async () => {
  if (!confirm(t('Close the current incident and start a new one? The current forecast will be cleared.'))) return;
  await api('/api/reset', {}).catch(() => {});
  state.result = null;
  state.version = -1;
  clearMap();
  INCIDENT_UI.forEach((s) => $(s).classList.add('hidden'));
  $('#left-tab').classList.add('hidden');
  document.body.classList.add('no-incident');
  $('#hint').classList.remove('hidden');
};
$('#btn-update').onclick = async () => {
  const r = await busy('Updating forecast with current weather', () => api('/api/refresh', { refetch_weather: true }));
  if (r) render(r);
};

// ── Theme and language (remembered per browser) ──────────────────────────
function applyTheme(theme) {
  document.documentElement.dataset.theme = theme;
  try { localStorage.setItem('wilson-theme', theme); } catch (_) {}
  const b = $('#btn-theme');
  b.setAttribute('aria-checked', String(theme === 'dark'));
  b.setAttribute('aria-label', t('Dark theme'));
  b.title = t(theme === 'light' ? 'Switch to the dark theme' : 'Switch to the light theme');
  const meta = document.querySelector('meta[name="theme-color"]');
  if (meta) meta.content = theme === 'light' ? '#eef0f3' : '#0e1116';
}
function applyLanguage(lang) {
  I18N.set(lang);
  I18N.apply();
  $('#lang-switch').setAttribute('aria-checked', String(I18N.lang === 'el'));
  applyTheme(document.documentElement.dataset.theme || 'dark');
  if (typeof applyLayout === 'function' && state.result) applyLayout();
  if (state.result) render(state.result, false);
  if (state.tool) $('#draw-text').textContent = t(TOOL_TEXT[state.tool]);
}
$('#btn-theme').onclick = () => {
  const root = document.documentElement;
  root.classList.add('theme-anim');
  applyTheme(root.dataset.theme === 'light' ? 'dark' : 'light');
  clearTimeout(applyTheme._t);
  applyTheme._t = setTimeout(() => root.classList.remove('theme-anim'), 350);
};
$('#lang-switch').onclick = () => applyLanguage(I18N.lang === 'el' ? 'en' : 'el');
{
  // ?theme=light|dark and ?lang=el|en override the saved choice (bookmarks, wall screens).
  const q = new URLSearchParams(location.search);
  let theme = 'dark';
  try { theme = localStorage.getItem('wilson-theme') === 'light' ? 'light' : 'dark'; } catch (_) {}
  if (q.get('theme') === 'light' || q.get('theme') === 'dark') theme = q.get('theme');
  applyTheme(theme);
  if (q.get('lang') === 'el' || q.get('lang') === 'en') I18N.set(q.get('lang'));
}

// ── Layout: collapsible column, reference panels and dock ────────────────
const layout = (() => {
  try { return Object.assign({ left: 'normal', dock: 'open', wx: false, quality: false }, JSON.parse(localStorage.getItem('wilson-layout-v2') || '{}')); }
  catch (_) { return { left: 'normal', dock: 'open', wx: false, quality: false }; }
})();
let savedLayout = null;
try { savedLayout = localStorage.getItem('wilson-layout-v2'); } catch (_) {}
if (innerWidth <= 900 && !savedLayout) layout.left = 'hidden';
// Minimal window-control icons; the hover text (title) explains each one.
const WIN_ICONS = {
  min: '<path d="M3.5 12.5h9"/>',
  max: '<rect x="3.5" y="3.5" width="9" height="9" rx="1"/>',
  restore: '<rect x="3.5" y="6" width="6.5" height="6.5" rx="1"/><path d="M6 6V4.5a1 1 0 0 1 1-1h4.5a1 1 0 0 1 1 1V9a1 1 0 0 1-1 1H10"/>',
  close: '<path d="M4.5 4.5l7 7M11.5 4.5l-7 7"/>',
};
function winBtn(el, icon, label) {
  if (el.dataset.icon !== icon || !el.firstChild) {
    el.dataset.icon = icon;
    el.innerHTML = `<svg viewBox="0 0 16 16" aria-hidden="true">${WIN_ICONS[icon]}</svg>`;
  }
  el.title = t(label);
  el.setAttribute('aria-label', t(label));
}
function applyLayout() {
  try { localStorage.setItem('wilson-layout-v2', JSON.stringify(layout)); } catch (_) {}
  const b = document.body.classList;
  b.toggle('left-hidden', layout.left === 'hidden');
  b.toggle('left-wide', layout.left === 'wide');
  b.toggle('dock-min', layout.dock === 'min');
  winBtn($('#left-wide'), layout.left === 'wide' ? 'restore' : 'max',
    layout.left === 'wide' ? 'Narrow the column' : 'Widen the column');
  winBtn($('#left-hide'), 'min', 'Collapse the column to the left');
  winBtn($('#dock-hide'), layout.dock === 'min' ? 'max' : 'min',
    layout.dock === 'min' ? 'Expand the operations panel' : 'Minimize the operations panel');
  $$('[data-close-panel]').forEach((x) => winBtn(x, 'close', 'Close'));
  $('#left-tab').classList.toggle('hidden', !state.result || layout.left !== 'hidden');
  for (const k of ['wx', 'quality']) {
    $(`#rp-${k}`).classList.toggle('hidden', !layout[k]);
    $(`#rt-tabs [data-panel="${k}"]`).classList.toggle('on', layout[k]);
  }
  map.resize();
}
$('#left-hide').onclick = () => { layout.left = 'hidden'; applyLayout(); };
$('#left-tab').onclick = () => { layout.left = 'normal'; applyLayout(); };
$('#left-wide').onclick = () => { layout.left = layout.left === 'wide' ? 'normal' : 'wide'; applyLayout(); };
$('#dock-hide').onclick = () => {
  layout.dock = layout.dock === 'min' ? 'open' : 'min';
  document.body.classList.remove('dock-peek');
  applyLayout();
};
// A minimized dock opens while the cursor is over it and closes shortly after it leaves.
let peekTimer = null;
$('#dock').addEventListener('mouseenter', () => {
  clearTimeout(peekTimer);
  if (layout.dock === 'min') document.body.classList.add('dock-peek');
});
$('#dock').addEventListener('mouseleave', () => {
  clearTimeout(peekTimer);
  peekTimer = setTimeout(() => document.body.classList.remove('dock-peek'), 300);
});
$('#rt-tabs').onclick = (e) => {
  const b = e.target.closest('button'); if (!b) return;
  layout[b.dataset.panel] = !layout[b.dataset.panel];
  applyLayout();
};
$$('[data-close-panel]').forEach((b) => (b.onclick = () => { layout[b.dataset.closePanel] = false; applyLayout(); }));

function clearMap() {
  ['iso', 'arrows', 'observed', 'recs', 'manual', 'recs-pts', 'manual-pts', 'places', 'draft', 'pin', 'wind'].forEach((s) => map.getSource(s)?.setData(EMPTY));
  map.setLayoutProperty('zones', 'visibility', 'none');
}

// ── Toolbar & drawing tools ──────────────────────────────────────────────
const TOOL_TEXT = {
  perimeter: 'Burned area: click points around the perimeter, then Finish.', firebreak: 'Firebreak: click points along the line, then Finish.',
  air_drop: 'Air drop: click the start and end of the drop line.', truck: 'Engine: click the engine position.', protect: 'Protected asset: click its location.',
};
$('#toolbar').onclick = async (e) => {
  const b = e.target.closest('button.tool'); if (!b) return;
  const tool = b.dataset.tool;
  if (tool === 'wind') return $('#dlg-wind').classList.remove('hidden');
  setTool(state.tool === tool ? null : tool);
};

function setTool(tool) {
  state.tool = tool;
  state.draft = [];
  drawDraft();
  $$('#toolbar .tool').forEach((x) => x.classList.toggle('on', x.dataset.tool === tool));
  $('#draw-bar').classList.toggle('hidden', !tool);
  $('#draw-text').textContent = TOOL_TEXT[tool] ? t(TOOL_TEXT[tool]) : '';
  const single = tool === 'truck' || tool === 'protect';
  $('#draw-done').classList.toggle('hidden', single);
  $('#draw-undo').classList.toggle('hidden', single);
  map.getCanvas().style.cursor = tool ? 'crosshair' : '';
}

function addDraftPoint(ll) {
  if (state.tool === 'truck') { finishTool([ll]); return; }
  if (state.tool === 'protect') { const t = state.tool; setTool(null); askName({ lng: ll[0], lat: ll[1] }); return t; }
  state.draft.push(ll);
  drawDraft();
  if (state.tool === 'air_drop' && state.draft.length === 2) finishTool(state.draft);
}

function drawDraft() {
  const f = state.draft.map((c) => ({ type: 'Feature', geometry: { type: 'Point', coordinates: c }, properties: {} }));
  if (state.draft.length > 1) {
    const ring = state.tool === 'perimeter' ? [...state.draft, state.draft[0]] : state.draft;
    f.push({ type: 'Feature', geometry: { type: 'LineString', coordinates: ring }, properties: {} });
  }
  map.getSource('draft')?.setData({ type: 'FeatureCollection', features: f });
}

$('#draw-undo').onclick = () => { state.draft.pop(); drawDraft(); };
$('#draw-cancel').onclick = () => setTool(null);
$('#draw-done').onclick = () => finishTool(state.draft);

async function finishTool(pts) {
  const tool = state.tool;
  setTool(null);
  if (tool === 'perimeter') {
    if (pts.length < 3) return toast(t('A burned area needs at least 3 points.'), true);
    const r = await busy('Calibrating the model to the observed fire', () => api('/api/local/perimeter', { polygon: pts, source: 'drawn on map' }));
    if (r) {
      render(r);
      if (r.observation_result?.calibration) toast(r.observation_result.calibration.text);
    }
  } else if (['firebreak', 'air_drop', 'truck'].includes(tool)) {
    if (tool === 'firebreak' && pts.length < 2) return toast(t('A firebreak needs at least 2 points.'), true);
    if (!pts.length) return;
    const r = await busy('Simulating the action', () => api('/api/plan', { add: { type: tool, points: pts } }));
    if (r) { state.view = 'plan'; render(r); }
  }
}

// Wind dialog
$('#wind-dir').onclick = (e) => {
  const b = e.target.closest('button'); if (!b) return;
  $$('#wind-dir button').forEach((x) => x.classList.toggle('on', x === b));
  state.windDir = +b.dataset.d;
};
$('#wind-speed').oninput = (e) => ($('#wind-val').textContent = e.target.value);
$('#wind-go').onclick = async () => {
  $('#dlg-wind').classList.add('hidden');
  const body = { wind_kmh: +$('#wind-speed').value, wind_dir_deg: state.windDir, source: 'reported in app' };
  if ($('#wind-temp').value !== '') body.temp_c = +$('#wind-temp').value;
  if ($('#wind-rh').value !== '') body.rh = +$('#wind-rh').value;
  const r = await busy('Applying measured wind', () => api('/api/local/weather', body));
  if (r) { render(r); toast(t('Measured wind applied.')); }
};
$$('[data-close]').forEach((b) => (b.onclick = () => b.closest('.dlg').classList.add('hidden')));

// Name dialog (protect a place)
function askName(lngLat) {
  state.namePos = lngLat;
  $('#name-input').value = '';
  $('#dlg-name').classList.remove('hidden');
  setTimeout(() => $('#name-input').focus(), 50);
}
$('#name-go').onclick = async () => {
  $('#dlg-name').classList.add('hidden');
  const p = state.namePos;
  const r = await busy('Calculating time to impact', () =>
    api('/api/destination', { lat: p.lat, lon: p.lng, name: $('#name-input').value.trim() || t('Protected asset') }));
  if (r) render(r);
};
$('#name-input').onkeydown = (e) => { if (e.key === 'Enter') $('#name-go').click(); };


// ── Rendering ────────────────────────────────────────────────────────────
const urgencyClass = (epoch) => {
  if (!epoch) return 'u-3';
  const m = (epoch * 1000 - Date.now()) / 60000;
  return m < 60 ? 'u-0' : m < 120 ? 'u-1' : m < 180 ? 'u-2' : 'u-3';
};
const urgencyColor = { 'u-0': '#ff7b74', 'u-1': '#f2a25c', 'u-2': '#e9c75a', 'u-3': '#d6dbe2' };
const COMPASS = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE', 'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW'];
const compass = (deg) => t(COMPASS[Math.round((((deg % 360) + 360) % 360) / 22.5) % 16]);
const ACTION_KIND_EN = { firebreak: 'Firebreak', air_drop: 'Air drop', truck: 'Engine', evacuate: 'Evacuation' };
const ACTION_KIND = new Proxy(ACTION_KIND_EN, { get: (o, k) => (o[k] ? t(o[k]) : undefined) });
const arrowSvg = '<svg viewBox="0 0 40 40"><path d="M20 3 L35 35 L20 27 L5 35 Z" fill="#d63a1e" stroke="#e6e9ee" stroke-width="1.5"/></svg>';

let mapReady = false;

function render(r, fit = false) {
  if (!r || r.status !== 'ready') return;
  const first = !state.result;
  state.result = r;
  state.version = r.version;
  if (!r.plan) state.view = 'base';

  INCIDENT_UI.forEach((s) => $(s).classList.remove('hidden'));
  document.body.classList.remove('no-incident');
  $('#hint').classList.add('hidden');
  if (first) applyLayout();

  state.planNo = {};
  r.recommendations.filter((x) => x.enabled).forEach((x, i) => (state.planNo[x.id] = i + 1));
  const c = r.confidence;
  $('#conf-pill').className = `conf g-${c.grade}`;
  $('#conf-pill').innerHTML = `${t('Confidence')}<b>${gradeText[c.grade]} ${c.score}%</b>`;
  const ig = r.incident.ignition;
  $('#incident').innerHTML = `<b>${t('Incident {pos}', { pos: `${ig.lat.toFixed(4)}, ${ig.lon.toFixed(4)}` })}</b><br>
    <span class="muted">${t('Ignition {t0} · forecast to {t1}', { t0: clock(r.incident.t0_epoch), t1: clock(r.incident.end_epoch) })}</span>`;
  renderHeadline(r);
  renderResources(r);
  renderView(r);
  renderPlaces(r);
  renderActions(r);
  renderWeather(r);
  renderQuality(r);
  // Re-frame when the forecast now reaches beyond what is on screen.
  if (mapReady && !fit && !first && r.extent) {
    const v = map.getBounds(), e = r.extent;
    fit = e[0] < v.getWest() || e[1] < v.getSouth() || e[2] > v.getEast() || e[3] > v.getNorth();
    if (fit && state.lastExtent && state.lastExtent.join() === e.join()) fit = false;  // user panned away on purpose
  }
  state.lastExtent = r.extent;
  if (mapReady) updateMap(r, fit || first);
  else state.fitPending = true;
}

function legendHtml(danger) {
  if (danger)
    return `<div class="lg-title">${t('Danger class')}</div>` +
      [1, 2, 3, 4, 5].map((i) => `<div class="it"><i class="dg-${i}"></i>${I18N.danger(i)}</div>`).reverse().join('');
  return `<div class="lg-title">${t('Fire arrival')}</div>
    <div class="it"><i style="background:rgba(45,25,25,.85)"></i>${t('Burned')}</div><div class="it"><i style="background:#e11e1e"></i>${t('Within 1 h')}</div>
    <div class="it"><i style="background:#f57314"></i>${t('1 to 2 h')}</div><div class="it"><i style="background:#fab923"></i>${t('2 to 3 h')}</div>
    <div class="it"><i style="background:#fae85a"></i>${t('Later')}</div><div class="it"><i class="hatch"></i>${t('Possible (pessimistic)')}</div>`;
}

function updateMap(r, fit) {
  const L = (state.view === 'plan' && r.plan) ? r.plan : r.base;
  map.setLayoutProperty('zones', 'visibility', 'visible');
  const prob = state.mode === 'prob', danger = state.mode === 'danger';
  const h = Math.min(state.probHour, L.probability.length);
  const url = prob ? L.probability[h - 1] : danger ? r.danger.overlay : L.overlay.url;
  map.getSource('wind').setData(state.showWind ? windFeatures(r, prob ? h : 0) : EMPTY);
  map.getSource('zones').updateImage({ url, coordinates: L.overlay.coordinates });
  for (const id of ['iso', 'iso-halo', 'iso-label']) map.setLayoutProperty(id, 'visibility', prob || danger ? 'none' : 'visible');
  map.getSource('iso').setData(L.isochrones);
  map.getSource('arrows').setData(state.view === 'base' && !prob ? r.base.arrows : EMPTY);
  $('#legend').classList.toggle('hidden', prob);
  $('#legend').innerHTML = legendHtml(danger);
  map.getSource('observed').setData(r.observed);
  const ig = r.incident.ignition;
  map.getSource('pin').setData({ type: 'FeatureCollection', features: [{ type: 'Feature', geometry: { type: 'Point', coordinates: [ig.lon, ig.lat] }, properties: {} }] });
  const fc = (items, label) => ({ type: 'FeatureCollection', features: items.map((x) => ({
    type: 'Feature', geometry: x.geometry, properties: { id: x.id, type: x.type, enabled: x.enabled ?? true, label: label(x) } })) });
  const pts = (items, label) => ({ type: 'FeatureCollection', features: items.map((x) => ({
    type: 'Feature', geometry: { type: 'Point', coordinates: geomCenter(x.geometry) },
    properties: { id: x.id, type: x.type, enabled: x.enabled ?? true, label: label(x) } })) });
  const shown = r.recommendations.filter((x) => x.enabled || (x.simulated && x.selected));
  const lbl = (x) => x.enabled ? String(state.planNo[x.id] || '') : 'R';
  map.getSource('recs').setData(fc(shown.filter((x) => x.geometry.type !== 'Point'), lbl));
  map.getSource('recs-pts').setData(pts(shown, lbl));
  map.getSource('manual').setData(fc(r.manual.filter((x) => x.geometry.type !== 'Point'), () => t('OWN')));
  map.getSource('manual-pts').setData(pts(r.manual, () => t('OWN')));
  map.getSource('places').setData({ type: 'FeatureCollection', features: r.destinations.filter((d) => !d.boundary && (d.eta_epoch || d.user)).map((d) => {
    const dd = (state.view === 'plan' && d.with_plan) ? d.with_plan : d;
    const u = urgencyClass(dd.eta_epoch);
    return { type: 'Feature', geometry: { type: 'Point', coordinates: [d.lon, d.lat] },
      properties: { label: `${d.name}\n${dd.burning_now ? t('BURNING') : dd.eta_epoch ? clock(dd.eta_epoch) : t('not reached')}`, color: urgencyColor[u] } };
  }) });

  if (fit) {
    const g = r.grid;
    // Fit to where the fire may go, inside the part of the map left free by the panels.
    let b = r.extent || [ig.lon, ig.lat, ig.lon, ig.lat];
    if (b[2] - b[0] < 0.01) b = [b[0] - 0.01, b[1] - 0.01, b[2] + 0.01, b[3] + 0.01];
    const pad = 0.004;
    const left = $('#left').offsetParent && !document.body.classList.contains('left-hidden') ? $('#left').offsetWidth + 40 : 50;
    const bottom = ($('#dock').offsetParent ? $('#dock').offsetHeight : 0) + 40;
    const rt = $('#rt').offsetParent ? Math.min(innerWidth * 0.35, $('#rt').offsetWidth + 30) : 60;
    const pd = { top: 80, bottom, left: Math.min(left, innerWidth * 0.45), right: Math.min(rt, innerWidth * 0.3) };
    map.fitBounds([[Math.max(g.west, b[0] - pad), Math.max(g.south, b[1] - pad)], [Math.min(g.east, b[2] + pad), Math.min(g.north, b[3] + pad)]],
      { padding: pd, duration: 800, maxZoom: 15 });
  }
}

function renderHeadline(r) {
  const s = r.summary;
  const intensity = s.max_flame_m < 1.2 ? 0 : s.max_flame_m < 2.4 ? 1 : s.max_flame_m < 3.4 ? 2 : 3;
  const shift = r.weather.shift ? `<div class="alert">${esc(tokens(r.weather.shift.text))}</div>` : '';
  const grow = s.area_ha_end - s.area_ha_now;
  const lvl = r.danger.fire_area_level;
  $('#headline').innerHTML = `
    <h3>${t('Fire behaviour')}</h3>
    <div class="fb-dir">
      <div class="fb-arrow">${arrowSvg.replace('<svg', `<svg style="transform:rotate(${s.direction_deg}deg)"`)}</div>
      <div><div class="lbl">${t('Head of fire moving')}</div><div class="big">${t('Towards {dir} ({deg}°)', { dir: esc(I18N.dir(s.direction_text)), deg: Math.round(s.direction_deg) })}</div>
        <div class="muted">${t('Forecast window {t0} to {t1}', { t0: clock(r.incident.now_epoch), t1: clock(r.incident.end_epoch) })}</div></div>
    </div>
    <div class="stats">
      <div class="stat"><span>${t('Head spread')}</span><b>${s.head_speed_kmh}<small> km/h</small></b></div>
      <div class="stat"><span>${t('Flame length')}</span><b>${s.max_flame_m}<small> m</small></b></div>
      <div class="stat"><span>${t('Head reach')}</span><b>${s.head_reach_km ?? 0}<small> km</small></b></div>
      <div class="stat"><span>${t('Burned now')}</span><b>${s.area_ha_now.toLocaleString()}<small> ha</small></b></div>
      <div class="stat"><span>${t('New growth')}</span><b>+${Math.max(0, grow).toLocaleString()}<small> ha</small></b></div>
      <div class="stat"><span>${t('Area at end')}</span><b>${s.area_ha_end.toLocaleString()}<small> ha</small></b></div>
    </div>
    <div class="advice i${intensity}">${esc(tokens(s.attack_advice))}</div>
    <div class="danger-here" id="danger-chip" title="${t('Show the danger map')}">
      <span><i class="sw dg-${lvl}"></i>${t('Area danger class: {c}', { c: `<b>${I18N.danger(lvl)}</b>` })}</span><span class="lnk">${t('Danger map')}</span></div>${shift}`;
  $('#danger-chip').onclick = () => { state.mode = 'danger'; render(state.result, false); $('#view-switch').scrollIntoView({ behavior: 'smooth' }); };
}

function renderView(r) {
  $$('#mode-seg button').forEach((b) => b.classList.toggle('on', b.dataset.mode === state.mode));
  $$('#plan-seg button').forEach((b) => b.classList.toggle('on', b.dataset.view === state.view));
  const planBtn = $('#plan-seg [data-view="plan"]');
  planBtn.disabled = !r.plan;
  planBtn.title = t(r.plan ? 'Forecast with the applied actions' : 'No actions applied yet. Switch on an action or draw your own.');
  const prob = state.mode === 'prob';
  $('#prob-ctl').classList.toggle('hidden', !prob);
  $('#danger-ctl').classList.toggle('hidden', state.mode !== 'danger');
  if (state.mode === 'danger') renderDanger(r);
  const H = r.base.probability.length;
  if (document.activeElement !== $('#horizon-input')) $('#horizon-input').value = r.incident.horizon_h;
  $('#show-wind').checked = state.showWind;
  const sl = $('#prob-hour');
  sl.max = H;
  state.probHour = Math.min(Math.max(1, state.probHour), H);
  sl.value = state.probHour;
  probHourText();
  $$('#scen-n button').forEach((b) => b.classList.toggle('on', +b.dataset.n === r.settings.members));
  $('#scen-wide').checked = r.settings.wide;
  let txt = '';
  if (prob) txt = t('Share of the {n} scenarios in which fire reaches each cell.', { n: r.settings.members }) +
    (r.settings.wide ? t(' Wide uncertainty is on.') : '');
  else if (state.mode === 'danger') txt = '';
  else if (r.plan) txt = r.plan.saved_ha > 0 ? t('With the plan: about {ha} ha less burned by {t}.', { ha: r.plan.saved_ha, t: clock(r.incident.end_epoch) })
    : t('The plan does not materially reduce the burned area. See places at risk.');
  $('#plan-gain').textContent = txt;
}

function fmtLead(min) {
  const m = Math.round(min);
  if (m < 60) return I18N.tr(`${m} min`);
  const h = Math.floor(m / 60), r = m % 60;
  return I18N.tr(r ? `${h} h ${r} min` : `${h} h`);
}

function probHourText() {
  const r = state.result;
  if (!r) return;
  const e = (r.frames || [])[state.probHour - 1] ?? r.incident.now_epoch + state.probHour * 3600;
  $('#prob-hour-text').textContent = `+${fmtLead((e - r.incident.now_epoch) / 60)} (${clock(e)})`;
}

// Wind arrows for one frame of result.wind_field (0 = now, i = frames[i-1]).
function windFeatures(r, frame) {
  const w = r.wind_field;
  if (!w || !w.frames.length) return EMPTY;
  const f = w.frames[Math.min(frame, w.frames.length - 1)];
  return { type: 'FeatureCollection', features: w.points.map((p, i) => ({
    type: 'Feature', geometry: { type: 'Point', coordinates: p }, properties: { to: f.to[i], kmh: f.kmh[i] } })) };
}


function renderDanger(r) {
  const d = r.danger;
  const lvl = d.fire_area_level;
  const total = d.area_by_class.reduce((a, x) => a + x.ha, 0) || 1;
  const bar = d.area_by_class.map((x, i) => `<i class="dg-${i + 1}" style="width:${(100 * x.ha) / total}%"></i>`).join('');
  const zones = d.places.length ? d.places.map((p) => `<div class="dz" data-lat="${p.lat}" data-lon="${p.lon}">
      ${t('{name}: fire within {km} km to the {dir} reaches it in under 1 h', { name: `<b>${esc(p.name)}</b>`, km: `<b>${p.reach_km}</b>`, dir: esc(I18N.dir(p.from_dir)) })}
      <span class="muted">${t('({ha} ha, spread up to {v} km/h)', { ha: p.zone_ha, v: p.max_ros_kmh })}</span></div>`).join('')
    : `<div class="note">${t('No settlement can be reached within 1 h from its surroundings.')}</div>`;
  $('#danger-ctl').innerHTML = `
    <div class="note">${t('Fire area is in danger class {c}. Danger combines potential spread rate with travel time to settlements and marked assets. Peak at {t}.',
      { c: `<b class="dtx-${lvl}">${I18N.danger(lvl)}</b>`, t: clock(d.peak_epoch) })}</div>
    <div class="danger-bar">${bar}</div>
    <div class="danger-keys"><span>${I18N.danger(1)}</span><span>${I18N.danger(5)}</span></div>
    <h3>${t('1-hour exposure zones')}</h3>${zones}`;
  $$('#danger-ctl .dz').forEach((el) => (el.onclick = () => map.flyTo({ center: [+el.dataset.lon, +el.dataset.lat], zoom: 13 })));
}

const RES = ['aircraft', 'trucks', 'dozers'];
let resTimer = null;
function renderResources(r) {
  for (const k of RES) {
    const row = $(`.res-row[data-res="${k}"]`);
    const v = r.resources[k];
    if (!resTimer) row.querySelector('.res-n').textContent = v.available;
    const use = row.querySelector('.res-use');
    use.textContent = v.available ? t('{n} assigned', { n: v.used }) : '';
    use.classList.toggle('full', v.available > 0 && v.used >= v.available);
  }
  $('#plan-notes').innerHTML = (r.plan_notes || []).map((n) => n.startsWith('1 more')
    ? `<span>${t('Request: {s}', { s: esc(tokens(n)) })}</span>` : `<span class="reserve">${esc(tokens(n))}</span>`).join('');
}

$('#res-card').addEventListener('click', (e) => {
  const b = e.target.closest('.step'); if (!b || !state.result) return;
  const n = b.parentElement.querySelector('.res-n');
  n.textContent = Math.max(0, Math.min(50, +n.textContent + +b.dataset.d));
  clearTimeout(resTimer);
  resTimer = setTimeout(async () => {
    const body = {};
    for (const k of RES) body[k] = +$(`.res-row[data-res="${k}"] .res-n`).textContent;
    resTimer = null;
    const res = await busy('Re-planning for available resources', () => api('/api/resources', body));
    if (res) render(res);
  }, 700);
});

$('#mode-seg').onclick = (e) => {
  const b = e.target.closest('button'); if (!b || !state.result) return;
  state.mode = b.dataset.mode;
  render(state.result, false);
};
$('#plan-seg').onclick = (e) => {
  const b = e.target.closest('button'); if (!b || b.disabled || !state.result) return;
  state.view = b.dataset.view;
  render(state.result, false);
};
$('#prob-hour').oninput = (e) => {
  state.probHour = +e.target.value;
  probHourText();
  if (mapReady && state.result) updateMap(state.result, false);
};
$('#scen-n').onclick = async (e) => {
  const b = e.target.closest('button'); if (!b) return;
  const res = await busy(t('Running {n} scenarios', { n: b.dataset.n }), () => api('/api/settings', { members: +b.dataset.n }));
  if (res) render(res);
};
$('#scen-wide').onchange = async (e) => {
  const res = await busy('Re-running with wide uncertainty', () => api('/api/settings', { wide: e.target.checked }));
  if (res) render(res);
};

function renderPlaces(r) {
  const list = r.destinations.filter((d) => d.eta_epoch || d.user || d.prob >= 20);
  if (!list.length) {
    $('#places').innerHTML = `<div class="note">${t('No settlements or marked assets reached within the {h} h window. Click any point on the map for its expected arrival time.', { h: r.incident.horizon_h })}</div>`;
    return;
  }
  const rows = list.slice(0, 20).map((d) => {
    const dd = (state.view === 'plan' && d.with_plan) ? d.with_plan : d;
    const u = urgencyClass(dd.eta_epoch);
    let eta;
    if (dd.burning_now) eta = t('Burning');
    else if (dd.eta_epoch) eta = `${clock(dd.eta_epoch)}<small>${t('in {d}', { d: inMinutes(dd.eta_epoch) })}</small>`;
    else eta = t('Not expected');
    const delta = (state.view === 'plan' && d.with_plan?.delay_text) ? `<div class="plan-delta">${t('With plan: {s}', { s: esc(tokens(d.with_plan.delay_text)) })}</div>` : '';
    const kind = d.boundary ? t(d.kind === 'river' ? 'River' : 'Road') : d.kind === 'user' ? t('Marked asset') : I18N.kind(d.kind);
    return `<tr class="place" data-lat="${d.lat}" data-lon="${d.lon}">
      <td class="urg ${u}"></td>
      <td><div class="pl-name">${esc(d.name)}</div><div class="pl-kind">${esc(kind)}</div>${delta}</td>
      <td class="r pl-eta">${eta}</td>
      <td class="r">${dd.prob ?? d.prob}%</td>
      <td class="r t-${d.grade}">${gradeText[d.grade]}</td></tr>`;
  }).join('');
  $('#places').innerHTML = `<table class="places"><thead><tr><th class="urg"></th><th>${t('Place')}</th><th class="r">${t('Arrival')}</th>
    <th class="r">${t('Prob.')}</th><th class="r">${t('Conf.')}</th></tr></thead><tbody>${rows}</tbody></table>`;
  $$('#places .place').forEach((el) => (el.onclick = () => map.flyTo({ center: [+el.dataset.lon, +el.dataset.lat], zoom: 14 })));
}

const fmtLL = (c) => `${c[1].toFixed(5)}, ${c[0].toFixed(5)}`;
function coordText(g) {
  if (g.type === 'Point') return fmtLL(g.coordinates);
  const c = g.coordinates;
  return `${fmtLL(c[0])} → ${fmtLL(c[c.length - 1])}`;
}

function geomCenter(g) {
  if (g.type === 'Point') return g.coordinates;
  const c = g.coordinates;
  if (c.length === 2) return [(c[0][0] + c[1][0]) / 2, (c[0][1] + c[1][1]) / 2];
  return c[Math.floor(c.length / 2)];
}

const RES_LABEL = { aircraft: 'aircraft', trucks: 'engine', dozers: 'dozer' };

// status: 'applied' (switched on by the user), 'recommended' (the planner's
// choice for the available resources, not applied) or 'other'.
function actionCard(x, status, now) {
  const late = x.act_before_epoch - now < 15 * 60;
  const when = x.type === 'evacuate' ? t('Immediately') : late ? t('Immediately (before {t})', { t: clock(x.act_before_epoch) }) : t('Start before {t}', { t: clock(x.act_before_epoch) });
  let conf = '';
  if (!x.simulated) conf = `<span class="t-${x.grade}">${t('{p}% of forecasts', { p: x.confidence })}</span>`;
  else if (status !== 'other') conf = `<span class="t-${x.grade}">${t('Effective in {p}% of scenarios', { p: x.confidence })}</span>`;
  else conf = t('Requires an additional {r}', { r: t(RES_LABEL[x.resource] || 'resource') });
  const no = status === 'applied' && state.planNo[x.id] ? `${state.planNo[x.id]}. ` : '';
  const tag = !x.simulated ? ` <span class="tag">${t('Advisory')}</span>`
    : status === 'applied' ? ` <span class="tag you">${t('Applied')}</span>`
    : status === 'recommended' ? ` <span class="tag rec">${t('Recommended')}</span>` : '';
  const cls = status === 'applied' || !x.simulated ? '' : status === 'recommended' ? 'rec' : 'off other';
  return `<div class="action k-${x.type} ${cls}" data-id="${x.id}">
      <div class="ahead"><span class="kind">${ACTION_KIND[x.type] || x.type}${tag}</span>
        ${x.simulated ? `<label class="switch" title="${t(status === 'applied' ? 'Remove from the plan' : 'Apply this action')}"><input type="checkbox" data-key="${esc(x.key)}" ${status === 'applied' ? 'checked' : ''}><span></span></label>` : ''}</div>
      <div class="title">${no}${esc(tokens(x.title))}</div>
      ${status !== 'other' ? `<div class="when ${late ? 'late' : ''}">${when}</div>` : ''}
      <div class="detail" title="${esc(tokens(x.detail))}">${esc(tokens(x.detail))}</div>
      <div class="coords" title="${t('Click to copy')}">${coordText(x.geometry)}</div>
      <div class="effect">${x.effect ? esc(tokens(x.effect)) + ' ' : ''}${conf}</div>
    </div>`;
}

function renderActions(r) {
  const now = Date.now() / 1000;
  const applied = r.recommendations.filter((x) => !x.simulated || x.enabled);
  const recommended = r.recommendations.filter((x) => x.simulated && !x.enabled && x.selected);
  const others = r.recommendations.filter((x) => x.simulated && !x.enabled && !x.selected).sort((a, b) => b.gain - a.gain);
  const own = r.manual.map((m) => `<div class="action k-${m.type}" data-mid="${m.id}">
      <div class="ahead"><span class="kind">${ACTION_KIND[m.type]} <span class="tag you">${t('Own')}</span></span>
        <button class="lnk" data-remove="${m.id}">${t('Remove')}</button></div>
      <div class="title">${t('Own {k}', { k: (ACTION_KIND[m.type] || m.type).toLowerCase() })}</div>
      <div class="when">${t('Active from {t}', { t: clock(m.active_from_epoch) })}</div>
      <div class="coords" title="${t('Click to copy')}">${coordText(m.geometry)}</div></div>`);
  let html = applied.map((x) => actionCard(x, 'applied', now)).concat(own).join('');
  if (!html) html = `<div class="empty">${t('No actions applied.')} ${t(recommended.length ? 'Switch on a recommended action, or draw your own.' : 'Enter available resources for recommendations, or draw your own.')}</div>`;
  if (recommended.length) html += `<div class="sep">${t('Recommended actions')}</div>` + recommended.map((x) => actionCard(x, 'recommended', now)).join('');
  if (others.length) html += `<div class="sep">${t('Other options')}</div>` + others.map((x) => actionCard(x, 'other', now)).join('');
  $('#actions').innerHTML = html;
  const n = applied.filter((x) => x.simulated).length + r.manual.length;
  $('#dock-summary').textContent = t('{n} applied', { n }) + (r.plan && r.plan.saved_ha > 0 ? t(' (about {ha} ha saved)', { ha: r.plan.saved_ha }) : '') +
    t(', {r} recommended, {o} other options', { r: recommended.length, o: others.length });

  $$('#actions input[type=checkbox]').forEach((cb) => (cb.onchange = async () => {
    const res = await busy(cb.checked ? 'Applying action' : 'Removing action', () => api('/api/plan', { toggle: { key: cb.dataset.key, enabled: cb.checked } }));
    if (res) render(res);
  }));
  $$('#actions [data-remove]').forEach((b) => (b.onclick = async () => {
    const res = await busy('Removing action', () => api('/api/plan', { remove: b.dataset.remove }));
    if (res) render(res);
  }));
  $$('#actions .coords').forEach((el) => (el.onclick = () => {
    navigator.clipboard?.writeText(el.textContent).then(() => toast(t('Coordinates copied.')), () => {});
  }));
  $$('#actions .title').forEach((t) => (t.onclick = () => {
    const card = t.closest('.action');
    const x = r.recommendations.find((y) => y.id === card.dataset.id) || r.manual.find((y) => y.id === card.dataset.mid);
    if (x) map.flyTo({ center: geomCenter(x.geometry), zoom: 14 });
  }));
}

function renderWeather(r) {
  const w = r.weather.now;
  const rows = r.weather.hourly.slice(1, 13).map((h) =>
    `<tr><td>${clock(h.epoch)}</td><td>${compass(h.dir_from)}</td><td>${h.wind_kmh}</td><td>${h.temp_c}</td><td>${h.rh}</td></tr>`).join('');
  $('#weather-card').innerHTML = `
    <div class="wx-now">
      <div><span>${t('Wind')}</span><b>${w.wind_kmh} km/h</b></div>
      <div><span>${t('From')}</span><b>${esc(I18N.dir(w.dir_text))} (${w.dir_from}°)</b></div>
      <div><span>${t('Temperature')}</span><b>${w.temp_c} °C</b></div>
      <div><span>${t('Rel. humidity')}</span><b>${w.rh}%</b></div>
    </div>
    <div class="wx-src">${t('Source: {s}', { s: esc(tokens(w.source)) })}</div>
    ${r.weather.shift ? `<div class="alert" style="margin:0 0 10px">${esc(tokens(r.weather.shift.text))}</div>` : ''}
    <table class="grid"><thead><tr><th>${t('Time')}</th><th>${t('From')}</th><th>km/h</th><th>°C</th><th>${t('RH %')}</th></tr></thead><tbody>${rows}</tbody></table>`;
}

function renderQuality(r) {
  const c = r.confidence;
  const cal = r.calibration ? `<div class="note" style="margin-top:8px">${esc(tokens(r.calibration.text))}</div>` : '';
  $('#quality').innerHTML = `<div class="q-grade"><span class="muted">${t('Overall')}</span><b class="t-${c.grade}">${gradeText[c.grade]} (${c.score}%)</b></div>
    <ul class="reasons">${c.reasons.map((x) => `<li>${esc(tokens(x))}</li>`).join('')}</ul>${cal}`;
  const s = r.sources, tm = r.timing_ms;
  const warn = (r.warnings || []).map((w) => `<div class="warn">${esc(tokens(w))}</div>`).join('');
  $('#sources').innerHTML = `${warn}<dl class="kv">
    <dt>${t('Terrain')}</dt><dd>${esc(s.terrain)}</dd><dt>${t('Vegetation')}</dt><dd>${esc(s.fuel)}</dd>
    <dt>${t('Weather')}</dt><dd>${esc(s.weather)}</dd><dt>${t('Places')}</dt><dd>${esc(s.places)}</dd>
    ${s.local.length ? `<dt>${t('Field data')}</dt><dd>${t('{n} report(s)', { n: s.local.length })}</dd>` : ''}
    ${r.wind_field ? `<dt>${t('Surface wind')}</dt><dd>${esc(r.wind_field.solver)}</dd>` : ''}
    <dt>${t('Computed')}</dt><dd>${t('{ms} ms, {n} scenarios, {c} m cells', { ms: tm.recompute, n: tm.members, c: r.grid.cell_m })}</dd></dl>`;
}

applyLanguage(I18N.lang);

// ── Live updates (drop-in files, background data, other devices) ─────────
setInterval(async () => {
  if (state.busy || document.hidden) return;
  try {
    const s = await api('/api/status');
    if (s.version !== state.version && s.status === 'ready') render(await api('/api/result'));
  } catch (e) { console.warn(e); }
}, 3000);
// Keep "in X min" labels honest as time passes.
setInterval(() => { if (state.result && !state.busy) { renderPlaces(state.result); renderActions(state.result); } }, 60000);
