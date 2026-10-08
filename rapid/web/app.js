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
const tokens = (s) => (s || '').replace(/\{t:(\d+)\}/g, (_, e) => clock(+e));
function inMinutes(epoch) {
  const m = Math.round((epoch * 1000 - Date.now()) / 60000);
  if (m <= 0) return 'now';
  if (m < 60) return `${Math.max(5, Math.round(m / 5) * 5)} min`;
  const h = Math.floor(m / 60), r = Math.round((m % 60) / 5) * 5;
  return `${h} h${r ? ' ' + r + ' min' : ''}`;
}
function inDur(m) {
  m = Math.round(m / 5) * 5;
  return m < 60 ? `${Math.max(5, m)} min` : `${Math.floor(m / 60)} h${m % 60 ? ' ' + (m % 60) + ' min' : ''}`;
}
const esc = (s) => String(s ?? '').replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
const gradeText = { high: 'High', medium: 'Medium', low: 'Low' };

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
window.addEventListener('error', (e) => toast(`Display error: ${e.message} (line ${e.lineno})`, true));
window.addEventListener('unhandledrejection', (e) => toast(`Display error: ${e.reason?.message || e.reason}`, true));

// ── API ──────────────────────────────────────────────────────────────────
async function api(path, body) {
  const opt = body === undefined ? {} : { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) };
  const res = await fetch(path, opt);
  const j = await res.json().catch(() => ({ error: 'Server did not answer.' }));
  if (j.error) throw new Error(j.error);
  return j;
}

async function busy(text, fn) {
  state.busy = true;
  $('#busy-text').textContent = text;
  $('#busy').classList.remove('hidden');
  const poll = setInterval(async () => {
    try { const s = await api('/api/status'); if (s.status && s.status !== 'ready' && s.status !== 'idle') $('#busy-text').textContent = s.status; } catch (_) {}
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
map.addControl(new maplibregl.NavigationControl({ showCompass: true }), 'top-right');
map.addControl(new maplibregl.ScaleControl({ unit: 'metric' }), 'bottom-right');
map.addControl(new maplibregl.GeolocateControl({ positionOptions: { enableHighAccuracy: true } }), 'top-right');

const EMPTY = { type: 'FeatureCollection', features: [] };
const svgImg = (svg, size) => new Promise((res) => {
  const img = new Image(size, size);
  img.onload = () => res(img);
  img.src = 'data:image/svg+xml;charset=utf-8,' + encodeURIComponent(svg);
});

map.on('load', async () => {
  const icons = {
    'arrow-head': ['<svg xmlns="http://www.w3.org/2000/svg" width="48" height="48" viewBox="0 0 48 48"><path d="M24 3 L42 40 L24 31 L6 40 Z" fill="#ff2a1a" stroke="#fff" stroke-width="3" stroke-linejoin="round"/></svg>', 48],
    'arrow-front': ['<svg xmlns="http://www.w3.org/2000/svg" width="32" height="32" viewBox="0 0 32 32"><path d="M16 3 L28 27 L16 21 L4 27 Z" fill="#ff8a1a" stroke="#fff" stroke-width="2.5" stroke-linejoin="round"/></svg>', 32],
    truck: ['<svg xmlns="http://www.w3.org/2000/svg" width="44" height="44" viewBox="0 0 44 44"><circle cx="22" cy="22" r="20" fill="#dc2626" stroke="#fff" stroke-width="3"/><path d="M10 15h14v12H10zM24 19h6l4 4v4H24z" fill="#fff"/><circle cx="15" cy="29" r="3" fill="#fff"/><circle cx="29" cy="29" r="3" fill="#fff"/></svg>', 44],
    drop: ['<svg xmlns="http://www.w3.org/2000/svg" width="44" height="44" viewBox="0 0 44 44"><circle cx="22" cy="22" r="20" fill="#2563eb" stroke="#fff" stroke-width="3"/><path d="M22 9c4 6 8 10 8 15a8 8 0 0 1-16 0c0-5 4-9 8-15z" fill="#fff"/></svg>', 44],
    evac: ['<svg xmlns="http://www.w3.org/2000/svg" width="44" height="44" viewBox="0 0 44 44"><circle cx="22" cy="22" r="20" fill="#9333ea" stroke="#fff" stroke-width="3"/><path d="M22 10v15M22 30v3" stroke="#fff" stroke-width="5" stroke-linecap="round"/></svg>', 44],
    fire: ['<svg xmlns="http://www.w3.org/2000/svg" width="40" height="40" viewBox="0 0 32 32"><path d="M16 2c2 6 9 9 9 17a9 9 0 0 1-18 0c0-5 3-7 4-11 1 3 3 4 3 4s-1-6 2-10z" fill="#ff5a1a" stroke="#fff" stroke-width="2"/></svg>', 40],
    dozer: ['<svg xmlns="http://www.w3.org/2000/svg" width="44" height="44" viewBox="0 0 44 44"><circle cx="22" cy="22" r="20" fill="#d97706" stroke="#fff" stroke-width="3"/><path d="M9 26h20v5H9zM12 19h10v7H12zM29 17l6 4v10h-6" fill="#fff"/></svg>', 44],
    wind: ['<svg xmlns="http://www.w3.org/2000/svg" width="28" height="28" viewBox="0 0 28 28"><path d="M14 2 L21 14 L16 13 L16 26 L12 26 L12 13 L7 14 Z" fill="#9ecbff" stroke="#0b1a2e" stroke-width="1.5" stroke-linejoin="round"/></svg>', 28],
    house: ['<svg xmlns="http://www.w3.org/2000/svg" width="30" height="30" viewBox="0 0 30 30"><path d="M3 14 15 4l12 10v13H3z" fill="#fff" stroke="#111" stroke-width="2"/></svg>', 30],
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
      'text-size': 14, 'text-offset': [0, 1.5], 'text-anchor': 'top' },
      paint: { 'icon-opacity': ['case', ['get', 'enabled'], 1, 0.4], 'text-color': '#fff', 'text-halo-color': '#000', 'text-halo-width': 2.5 } });
  }
  map.addLayer({ id: 'places', type: 'symbol', source: 'places', layout: {
    'icon-image': 'house', 'icon-allow-overlap': true, 'text-field': ['get', 'label'], 'text-font': ['Open Sans Bold'], 'text-size': 13,
    'text-offset': [0, 1.2], 'text-anchor': 'top', 'text-allow-overlap': false },
    paint: { 'text-color': ['get', 'color'], 'text-halo-color': '#000', 'text-halo-width': 2 } });
  map.addLayer({ id: 'draft-line', type: 'line', source: 'draft', paint: { 'line-color': '#fff', 'line-width': 3, 'line-dasharray': [2, 1] } });
  map.addLayer({ id: 'draft-pts', type: 'circle', source: 'draft', filter: ['==', '$type', 'Point'], paint: { 'circle-radius': 7, 'circle-color': '#ff6a1a', 'circle-stroke-color': '#fff', 'circle-stroke-width': 2 } });
  map.addLayer({ id: 'pin', type: 'symbol', source: 'pin', layout: { 'icon-image': 'fire', 'icon-allow-overlap': true, 'icon-anchor': 'bottom' } });

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
api('/api/result').then((cur) => { if (cur && cur.status === 'ready' && !state.result) render(cur, true); }).catch(() => {});

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
  if (!p.inside) html = '<b>Outside the forecast area.</b>';
  else {
    let eta;
    if (p.burning_now) eta = '<div class="pop-eta">Already burning</div>';
    else if (p.eta_epoch) eta = `<div class="pop-eta">Fire ~${clock(p.eta_epoch)}</div><div>in ${inMinutes(p.eta_epoch)} · earliest ${p.earliest_epoch ? clock(p.earliest_epoch) : '—'}</div>`;
    else eta = `<div class="pop-eta">${p.prob > 0 ? 'Unlikely' : 'Not reached'}</div><div class="muted">within the forecast period</div>`;
    const dg = p.danger;
    const dhtml = dg && dg.level ? `<div class="pop-danger"><b class="dtx-${dg.level}">${DANGER_NAMES[dg.level]} danger</b> —
      fire here could spread ${dg.spread_kmh} km/h towards the ${esc(dg.toward)}${dg.flame_m ? `, flames ~${dg.flame_m} m` : ''}.
      ${dg.reaches ? (dg.reaches_min < 5 ? `Inside / at the edge of <b>${esc(dg.reaches)}</b>.` :
        `Would reach <b>${esc(dg.reaches)}</b> in ~${inDur(dg.reaches_min)}.`) : 'No village within 6 h of fire travel.'}</div>` : '';
    html = `${eta}<div>${p.prob}% chance of fire here</div>${dhtml}
      <div class="muted small">${esc(p.fuel)} · slope ${p.slope_pct}% · ${p.elevation_m} m${p.flame_m ? ` · flames ~${p.flame_m} m` : ''}</div>
      <button class="btn primary" id="pop-protect">Protect this place</button>`;
  }
  const pop = new maplibregl.Popup({ maxWidth: '300px' }).setLngLat(lngLat).setHTML(`<div class="pop">${html}</div>`).addTo(map);
  const b = document.getElementById('pop-protect');
  if (b) b.onclick = () => { pop.remove(); askName(lngLat); };
}

function showAction(id, lngLat) {
  const r = (state.result.recommendations || []).find((x) => x.id === id);
  if (!r) return;
  new maplibregl.Popup({ maxWidth: '320px' }).setLngLat(lngLat)
    .setHTML(`<div class="pop"><b>${esc(r.title)}</b><div class="muted small">Act before ${clock(r.act_before_epoch)}</div><div class="small">${esc(tokens(r.detail))}</div></div>`)
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
  if (!(v > 0)) return toast('Enter the forecast length in hours (0.5–48).', true);
  const h = clampHours(v);
  const r = await busy(`Forecasting the next ${h} h…`, () => api('/api/refresh', { horizon_h: h }));
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
  const r = await busy('Getting weather, terrain and vegetation…', () =>
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
  if (!p) return toast('Could not find that place. Try coordinates like 38.56, 21.94', true);
  map.flyTo({ center: [p.lon, p.lat], zoom: Math.max(map.getZoom(), 13) });
  if (!state.result) askStart(p.lat, p.lon);
  else showPoint({ lat: p.lat, lng: p.lon });
};

$('#btn-new').onclick = async () => {
  if (!confirm('Start over with a new fire? The current forecast will be cleared.')) return;
  await api('/api/reset', {}).catch(() => {});
  state.result = null;
  state.version = -1;
  clearMap();
  ['#panel', '#toolbar', '#legend', '#conf-pill', '#btn-new'].forEach((s) => $(s).classList.add('hidden'));
  $('#hint').classList.remove('hidden');
};

function clearMap() {
  ['iso', 'arrows', 'observed', 'recs', 'manual', 'recs-pts', 'manual-pts', 'places', 'draft', 'pin', 'wind'].forEach((s) => map.getSource(s)?.setData(EMPTY));
  map.setLayoutProperty('zones', 'visibility', 'none');
}

// ── Toolbar & drawing tools ──────────────────────────────────────────────
const TOOL_TEXT = {
  perimeter: 'Tap around the burned area', firebreak: 'Tap along the firebreak line',
  air_drop: 'Tap the drop line start and end (or one point)', truck: 'Tap where the truck is', protect: 'Tap the place to protect',
};
$('#toolbar').onclick = async (e) => {
  const b = e.target.closest('button'); if (!b) return;
  const tool = b.dataset.tool;
  if (tool === 'wind') return $('#dlg-wind').classList.remove('hidden');
  if (tool === 'refresh') {
    const r = await busy('Updating forecast…', () => api('/api/refresh', { refetch_weather: true }));
    if (r) render(r);
    return;
  }
  setTool(state.tool === tool ? null : tool);
};

function setTool(tool) {
  state.tool = tool;
  state.draft = [];
  drawDraft();
  $$('#toolbar button').forEach((x) => x.classList.toggle('on', x.dataset.tool === tool));
  $('#draw-bar').classList.toggle('hidden', !tool);
  $('#draw-text').textContent = TOOL_TEXT[tool] || '';
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
    if (pts.length < 3) return toast('Tap at least 3 points around the burned area.', true);
    const r = await busy('Matching the model to the real fire…', () => api('/api/local/perimeter', { polygon: pts, source: 'drawn on map' }));
    if (r) {
      render(r);
      if (r.observation_result?.calibration) toast(r.observation_result.calibration.text);
    }
  } else if (['firebreak', 'air_drop', 'truck'].includes(tool)) {
    if (tool === 'firebreak' && pts.length < 2) return toast('A firebreak needs at least 2 points.', true);
    if (!pts.length) return;
    const r = await busy('Testing your action…', () => api('/api/plan', { add: { type: tool, points: pts } }));
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
  const r = await busy('Updating with local wind…', () => api('/api/local/weather', body));
  if (r) { render(r); toast('Local wind applied.'); }
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
  const r = await busy('Calculating time to impact…', () =>
    api('/api/destination', { lat: p.lat, lon: p.lng, name: $('#name-input').value.trim() || 'Protected place' }));
  if (r) render(r);
};
$('#name-input').onkeydown = (e) => { if (e.key === 'Enter') $('#name-go').click(); };


// Mobile: tap the grip to expand the sheet
$('#panel-grip').onclick = () => $('#panel').classList.toggle('expanded');

// ── Rendering ────────────────────────────────────────────────────────────
const urgencyClass = (epoch) => {
  if (!epoch) return 'u-3';
  const m = (epoch * 1000 - Date.now()) / 60000;
  return m < 60 ? 'u-0' : m < 120 ? 'u-1' : m < 180 ? 'u-2' : 'u-3';
};
const urgencyColor = { 'u-0': '#ff6b6b', 'u-1': '#ffa45c', 'u-2': '#ffd34d', 'u-3': '#e5e7eb' };
const ACTION_ICON = {
  firebreak: '<svg viewBox="0 0 24 24"><path d="M3 20L21 4" stroke-dasharray="3 2"/></svg>',
  air_drop: '<svg viewBox="0 0 24 24"><path d="M12 3c3 5 6 8 6 11a6 6 0 0 1-12 0c0-3 3-6 6-11z"/></svg>',
  truck: '<svg viewBox="0 0 24 24"><path d="M2 7h11v9H2zM13 10h5l3 3v3h-8z"/><circle cx="6" cy="18" r="2"/><circle cx="17" cy="18" r="2"/></svg>',
  evacuate: '<svg viewBox="0 0 24 24"><path d="M12 4v10M12 19v1"/></svg>',
};
const arrowSvg = '<svg viewBox="0 0 48 48"><path d="M24 3 L42 40 L24 31 L6 40 Z" fill="#ff3b1f" stroke="#fff" stroke-width="2.5" stroke-linejoin="round"/></svg>';
const windSvg = (deg) => `<svg viewBox="0 0 48 48" style="transform:rotate(${deg + 180}deg)"><circle cx="24" cy="24" r="22" fill="#262a31"/><path d="M24 6 L34 30 L24 24 L14 30 Z" fill="#9ecbff"/></svg>`;

let mapReady = false;

function render(r, fit = false) {
  if (!r || r.status !== 'ready') return;
  const first = !state.result;
  state.result = r;
  state.version = r.version;
  if (!r.plan) state.view = 'base';

  ['#panel', '#toolbar', '#legend', '#conf-pill', '#btn-new'].forEach((s) => $(s).classList.remove('hidden'));
  $('#hint').classList.add('hidden');

  state.planNo = {};
  r.recommendations.filter((x) => x.enabled).forEach((x, i) => (state.planNo[x.id] = i + 1));
  const c = r.confidence;
  $('#conf-pill').innerHTML = `<span class="dot g-${c.grade}"></span><span class="long">${gradeText[c.grade]} confidence · </span>${c.score}%`;
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
  $('#legend').innerHTML = danger
    ? DANGER_NAMES.slice(1).map((n, i) => `<div><i class="dg-${i + 1}"></i>${n}</div>`).reverse().join('')
    : `<div><i style="background:rgba(45,25,25,.85)"></i>Burned</div><div><i style="background:#e11e1e"></i>Within 1 h</div>
       <div><i style="background:#f57314"></i>1–2 h</div><div><i style="background:#fab923"></i>2–3 h</div>
       <div><i style="background:#fae85a"></i>Later</div><div><i class="hatch"></i>Possible</div>`;
  map.getSource('observed').setData(r.observed);
  const ig = r.incident.ignition;
  map.getSource('pin').setData({ type: 'FeatureCollection', features: [{ type: 'Feature', geometry: { type: 'Point', coordinates: [ig.lon, ig.lat] }, properties: {} }] });
  const fc = (items, label) => ({ type: 'FeatureCollection', features: items.map((x) => ({
    type: 'Feature', geometry: x.geometry, properties: { id: x.id, type: x.type, enabled: x.enabled ?? true, label: label(x) } })) });
  const pts = (items, label) => ({ type: 'FeatureCollection', features: items.map((x) => ({
    type: 'Feature', geometry: { type: 'Point', coordinates: geomCenter(x.geometry) },
    properties: { id: x.id, type: x.type, enabled: x.enabled ?? true, label: label(x) } })) });
  const inPlan = r.recommendations.filter((x) => x.enabled);
  map.getSource('recs').setData(fc(inPlan.filter((x) => x.geometry.type !== 'Point'), (x) => String(state.planNo[x.id] || '')));
  map.getSource('recs-pts').setData(pts(inPlan, (x) => String(state.planNo[x.id] || '')));
  map.getSource('manual').setData(fc(r.manual.filter((x) => x.geometry.type !== 'Point'), () => 'yours'));
  map.getSource('manual-pts').setData(pts(r.manual, () => 'yours'));
  map.getSource('places').setData({ type: 'FeatureCollection', features: r.destinations.filter((d) => !d.boundary && (d.eta_epoch || d.user)).map((d) => {
    const dd = (state.view === 'plan' && d.with_plan) ? d.with_plan : d;
    const u = urgencyClass(dd.eta_epoch);
    return { type: 'Feature', geometry: { type: 'Point', coordinates: [d.lon, d.lat] },
      properties: { label: `${d.name}\n${dd.burning_now ? 'burning' : dd.eta_epoch ? clock(dd.eta_epoch) : 'safe'}`, color: urgencyColor[u] } };
  }) });

  if (fit) {
    const g = r.grid;
    // Fit to where the fire may go, not the whole data area.
    let b = r.extent || [ig.lon, ig.lat, ig.lon, ig.lat];
    if (b[2] - b[0] < 0.01) b = [b[0] - 0.01, b[1] - 0.01, b[2] + 0.01, b[3] + 0.01];
    const pad = 0.004, desktop = innerWidth > 820;
    map.fitBounds([[Math.max(g.west, b[0] - pad), Math.max(g.south, b[1] - pad)], [Math.min(g.east, b[2] + pad), Math.min(g.north, b[3] + pad)]],
      { padding: desktop ? { top: 90, bottom: 110, left: 60, right: 440 } : { top: 80, bottom: innerHeight * 0.5, left: 30, right: 30 }, duration: 800, maxZoom: 15 });
  }
}

function renderHeadline(r) {
  const s = r.summary;
  const intensity = s.max_flame_m < 1.2 ? 0 : s.max_flame_m < 2.4 ? 1 : s.max_flame_m < 3.4 ? 2 : 3;
  const shift = r.weather.shift ? `<div class="alert">⚠ ${esc(tokens(r.weather.shift.text))}</div>` : '';
  const grow = s.area_ha_end - s.area_ha_now;
  $('#headline').innerHTML = `
    <div class="dir">
      <div class="arrow">${arrowSvg.replace('<svg', `<svg style="transform:rotate(${s.direction_deg}deg)"`)}</div>
      <div><div class="muted small">FIRE IS MOVING</div><div class="big">Towards the ${esc(s.direction_text)}</div>
      <div class="muted">up to <b style="color:var(--text)">${s.head_speed_kmh} km/h</b></div></div>
    </div>
    <div class="stats">
      <div class="stat"><span class="muted small">Flames up to</span><b>${s.max_flame_m} m</b></div>
      <div class="stat"><span class="muted small">Burned now</span><b>${s.area_ha_now.toLocaleString()}<small> ha</small></b></div>
      <div class="stat"><span class="muted small">By ${clock(r.incident.end_epoch)}</span><b>+${Math.max(0, grow).toLocaleString()}<small> ha</small></b></div>
    </div>
    <div class="advice i${intensity}">${esc(s.attack_advice)}</div>
    <div class="danger-here dg-${r.danger.fire_area_level}" style="cursor:pointer" id="danger-chip">
      Area danger: ${DANGER_NAMES[r.danger.fire_area_level]} — tap to see the danger map</div>${shift}`;
  $('#danger-chip').onclick = () => { state.mode = 'danger'; render(state.result, false); $('#view-switch').scrollIntoView({ behavior: 'smooth' }); };
}

function renderView(r) {
  $$('#mode-seg button').forEach((b) => b.classList.toggle('on', b.dataset.mode === state.mode));
  $('#plan-seg').classList.toggle('hidden', !r.plan);
  $$('#plan-seg button').forEach((b) => b.classList.toggle('on', b.dataset.view === state.view));
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
  if (prob) txt = `Colours show in how many of the ${r.settings.members} scenarios the fire reaches each spot. ` +
    (r.settings.wide ? 'Wide range: inputs are treated as very uncertain.' : 'Use the wide range when wind or fuel are unclear.');
  else if (state.mode === 'danger') txt = '';
  else if (r.plan) txt = r.plan.saved_ha > 0 ? `With the plan about ${r.plan.saved_ha} ha less burns by ${clock(r.incident.end_epoch)}.`
    : 'The plan barely changes the burned area — see each place below.';
  $('#plan-gain').textContent = txt;
}

function fmtLead(min) {
  const m = Math.round(min);
  if (m < 60) return `${m} min`;
  const h = Math.floor(m / 60), r = m % 60;
  return r ? `${h} h ${r}` : `${h} h`;
}

function probHourText() {
  const r = state.result;
  if (!r) return;
  const e = (r.frames || [])[state.probHour - 1] ?? r.incident.now_epoch + state.probHour * 3600;
  $('#prob-hour-text').textContent = `+${fmtLead((e - r.incident.now_epoch) / 60)} · ${clock(e)}`;
}

// Wind arrows for one frame of result.wind_field (0 = now, i = frames[i-1]).
function windFeatures(r, frame) {
  const w = r.wind_field;
  if (!w || !w.frames.length) return EMPTY;
  const f = w.frames[Math.min(frame, w.frames.length - 1)];
  return { type: 'FeatureCollection', features: w.points.map((p, i) => ({
    type: 'Feature', geometry: { type: 'Point', coordinates: p }, properties: { to: f.to[i], kmh: f.kmh[i] } })) };
}

const DANGER_NAMES = ['None', 'Low', 'Moderate', 'High', 'Very high', 'Extreme'];
const DANGER_HELP = {
  1: 'fire spreads slowly here', 2: 'fire spreads moderately', 3: 'fast fire, or a place within reach',
  4: 'fast fire with places within reach', 5: 'very fast fire close to places',
};

function renderDanger(r) {
  const d = r.danger;
  const lvl = d.fire_area_level;
  const total = d.area_by_class.reduce((a, x) => a + x.ha, 0) || 1;
  const bar = d.area_by_class.map((x, i) => `<i class="dg-${i + 1}" style="width:${(100 * x.ha) / total}%"></i>`).join('');
  const zones = d.places.length ? d.places.map((p) => `<div class="dz" data-lat="${p.lat}" data-lon="${p.lon}">
      <b>${esc(p.name)}</b><br>Fire anywhere within <b>${p.reach_km} km</b> to the ${esc(p.from_dir)} would reach it in under 1 h
      <span class="muted">(${p.zone_ha} ha of ground, spreading up to ${p.max_ros_kmh} km/h)</span></div>`).join('')
    : '<div class="muted small">No village can be reached within 1 h from its surroundings.</div>';
  $('#danger-ctl').innerHTML = `
    <div class="danger-here dg-${lvl}">The fire is in a ${DANGER_NAMES[lvl].toUpperCase()} danger area</div>
    <div class="small muted" style="margin-top:8px">Danger combines how fast fire would spread from each spot with how quickly it
      would reach a village or other important place. It is worst around ${clock(d.peak_epoch)}.</div>
    <div class="danger-bar">${bar}</div>
    <div class="danger-keys"><span>Low</span><span>Extreme</span></div>
    <h3>1-hour danger zones around places</h3>${zones}`;
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
    use.textContent = v.available ? `${v.used} in plan` : '';
    use.classList.toggle('full', v.available > 0 && v.used >= v.available);
  }
  $('#plan-notes').innerHTML = (r.plan_notes || []).map((n) =>
    `<div class="${n.startsWith('1 more') ? 'ask' : ''}">${n.startsWith('1 more') ? 'Ask for: ' : ''}${esc(tokens(n))}</div>`).join('');
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
    const res = await busy('Planning with your resources…', () => api('/api/resources', body));
    if (res) render(res);
  }, 700);
});

$('#mode-seg').onclick = (e) => {
  const b = e.target.closest('button'); if (!b || !state.result) return;
  state.mode = b.dataset.mode;
  render(state.result, false);
};
$('#plan-seg').onclick = (e) => {
  const b = e.target.closest('button'); if (!b || !state.result) return;
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
  const res = await busy(`Running ${b.dataset.n} scenarios…`, () => api('/api/settings', { members: +b.dataset.n }));
  if (res) render(res);
};
$('#scen-wide').onchange = async (e) => {
  const res = await busy('Exploring a wider range of conditions…', () => api('/api/settings', { wide: e.target.checked }));
  if (res) render(res);
};

function renderPlaces(r) {
  const list = r.destinations.filter((d) => d.eta_epoch || d.user || d.prob >= 20);
  if (!list.length) {
    $('#places').innerHTML = `<div class="muted">No villages or marked places reached in the next ${r.incident.horizon_h} h.
      Tap the map anywhere to see when fire could get there.</div>`;
    return;
  }
  $('#places').innerHTML = list.slice(0, 15).map((d) => {
    const dd = (state.view === 'plan' && d.with_plan) ? d.with_plan : d;
    const u = urgencyClass(dd.eta_epoch);
    const icon = d.kind === 'road' ? '═' : d.kind === 'river' ? '≈' : d.user ? '★' : '⌂';
    let eta;
    if (dd.burning_now) eta = 'Burning<small>now</small>';
    else if (dd.eta_epoch) eta = `${inMinutes(dd.eta_epoch)}<small>~${clock(dd.eta_epoch)}</small>`;
    else eta = `Not expected<small>${dd.prob ?? d.prob}% chance</small>`;
    const delta = (state.view === 'plan' && d.with_plan?.delay_text) ? `<div class="plan-delta">With actions: ${esc(d.with_plan.delay_text)}</div>` : '';
    const what = d.boundary ? (d.kind === 'river' ? 'River reached' : 'Road reached') : esc(d.kind === 'user' ? 'Marked place' : d.kind);
    return `<div class="place" data-lat="${d.lat}" data-lon="${d.lon}">
      <div class="ico ${u}">${icon}</div>
      <div class="main"><div class="name">${esc(d.name)}</div>
        <div class="muted small">${what} · <span class="t-${d.grade}">${gradeText[d.grade]} confidence</span></div>
        <div class="bar" title="${dd.prob ?? d.prob}% chance fire reaches it"><i style="width:${dd.prob ?? d.prob}%;background:${urgencyColor[u]}"></i></div>${delta}</div>
      <div class="eta">${eta}</div></div>`;
  }).join('');
  $$('#places .place').forEach((el) => (el.onclick = () => map.flyTo({ center: [+el.dataset.lon, +el.dataset.lat], zoom: 14 })));
}

const fmtLL = (c) => `${c[1].toFixed(5)}, ${c[0].toFixed(5)}`;
function coordText(g) {
  if (g.type === 'Point') return `📍 ${fmtLL(g.coordinates)}`;
  const c = g.coordinates;
  return `📍 ${fmtLL(c[0])} → ${fmtLL(c[c.length - 1])}`;
}

function geomCenter(g) {
  if (g.type === 'Point') return g.coordinates;
  const c = g.coordinates;
  if (c.length === 2) return [(c[0][0] + c[1][0]) / 2, (c[0][1] + c[1][1]) / 2];
  return c[Math.floor(c.length / 2)];
}

const RES_LABEL = { aircraft: 'aircraft', trucks: 'fire truck', dozers: 'bulldozer' };

function actionCard(x, inPlan, now) {
  const late = x.act_before_epoch - now < 15 * 60;
  const when = x.type === 'evacuate' ? 'Now' : late ? `Now — before ${clock(x.act_before_epoch)}` : `Before ${clock(x.act_before_epoch)}`;
  let conf = '';
  if (!x.simulated) conf = `<span class="t-${x.grade}">${x.confidence}% of forecasts</span>`;
  else if (inPlan) conf = `<span class="t-${x.grade}">works in ${x.confidence}% of scenarios</span>`;
  else conf = `<span class="muted">needs a free ${RES_LABEL[x.resource] || 'resource'}</span>`;
  const no = inPlan ? `${state.planNo[x.id] || '•'}. ` : '';
  const tag = x.forced ? '<span class="tag you">added by you</span>' : '';
  return `<div class="action ${inPlan || !x.simulated ? '' : 'off'}" data-id="${x.id}">
      <div class="num k-${x.type}">${ACTION_ICON[x.type] || ''}</div>
      <div><div class="title">${no}${esc(x.title)}${tag}</div>
        ${inPlan || !x.simulated ? `<div class="when ${late ? 'late' : ''}">${when}</div>` : ''}
        <div class="detail">${esc(tokens(x.detail))}</div>
        <div class="coords" title="Tap to copy">${coordText(x.geometry)}</div>
        <div class="effect">${x.effect ? esc(tokens(x.effect)) + ' · ' : ''}${conf}</div></div>
      <div>${x.simulated ? `<label class="switch" title="${inPlan ? 'Remove from the plan' : 'Add to the plan'}"><input type="checkbox" data-key="${esc(x.key)}" ${inPlan ? 'checked' : ''}><span></span></label>` : ''}</div>
    </div>`;
}

function renderActions(r) {
  const now = Date.now() / 1000;
  const plan = r.recommendations.filter((x) => !x.simulated || x.enabled);
  const others = r.recommendations.filter((x) => x.simulated && !x.enabled).sort((a, b) => b.gain - a.gain);
  const own = r.manual.map((m) => `<div class="action" data-mid="${m.id}">
      <div class="num k-${m.type}">${ACTION_ICON[m.type]}</div>
      <div><div class="title">Your ${m.type === 'air_drop' ? 'water drop' : m.type}<span class="tag you">drawn by you</span></div>
        <div class="when">Active from ${clock(m.active_from_epoch)}</div></div>
      <div><button class="x" data-remove="${m.id}" title="Remove">×</button></div></div>`);
  let html = plan.map((x) => actionCard(x, true, now)).concat(own).join('');
  if (!html) html = '<div class="muted">No action is worth a resource right now — keep monitoring, or add resources above.</div>';
  if (others.length)
    html += `<details class="others"><summary>Other options (${others.length}) — need more resources or help less</summary>
      ${others.map((x) => actionCard(x, false, now)).join('')}</details>`;
  $('#actions').innerHTML = html;

  $$('#actions input[type=checkbox]').forEach((cb) => (cb.onchange = async () => {
    const res = await busy('Re-planning…', () => api('/api/plan', { toggle: { key: cb.dataset.key, enabled: cb.checked } }));
    if (res) render(res);
  }));
  $$('#actions [data-remove]').forEach((b) => (b.onclick = async () => {
    const res = await busy('Removing…', () => api('/api/plan', { remove: b.dataset.remove }));
    if (res) render(res);
  }));
  $$('#actions .coords').forEach((el) => (el.onclick = () => {
    navigator.clipboard?.writeText(el.textContent.replace('📍 ', '')).then(() => toast('Coordinates copied.'), () => {});
  }));
  $$('#actions .title').forEach((t) => (t.onclick = () => {
    const card = t.closest('.action');
    const x = r.recommendations.find((y) => y.id === card.dataset.id) || r.manual.find((y) => y.id === card.dataset.mid);
    if (x) map.flyTo({ center: geomCenter(x.geometry), zoom: 14 });
  }));
}

function renderWeather(r) {
  const w = r.weather.now;
  const next = r.weather.hourly.slice(1, 7).map((h) =>
    `<div style="text-align:center;flex:1"><div class="muted small">${clock(h.epoch)}</div>${windSvg(h.dir_from).replace('<svg', '<svg width="26" height="26"')}<div class="small">${h.wind_kmh}</div></div>`).join('');
  $('#weather-card').innerHTML = `<h2>Wind & weather</h2>
    <div class="wx">${windSvg(w.dir_from)}<div><b>${w.wind_kmh} km/h</b> from the ${esc(w.dir_text)}<br>
      <span class="muted">${w.temp_c} °C · ${w.rh}% humidity · ${esc(w.source)}</span></div></div>
    <div style="display:flex;gap:4px;margin-top:12px">${next}</div>`;
}

function renderQuality(r) {
  const c = r.confidence;
  const cal = r.calibration ? `<div class="small" style="margin-top:8px">${esc(r.calibration.text)}</div>` : '';
  $('#quality').innerHTML = `<div style="display:flex;align-items:center;gap:10px"><span class="dot g-${c.grade}" style="width:18px;height:18px"></span>
    <b style="font-size:20px">${gradeText[c.grade]} · ${c.score}%</b></div>
    <ul class="reasons">${c.reasons.map((x) => `<li>${esc(tokens(x))}</li>`).join('')}</ul>${cal}`;
  const s = r.sources, t = r.timing_ms;
  const warn = (r.warnings || []).map((w) => `<div style="color:var(--amber)">⚠ ${esc(w)}</div>`).join('');
  $('#sources').innerHTML = `${warn}Terrain: ${esc(s.terrain)}<br>Vegetation: ${esc(s.fuel)}<br>Weather: ${esc(s.weather)}<br>
    Places: ${esc(s.places)}${s.local.length ? `<br>Field data: ${s.local.length} report(s)` : ''}<br>
    ${r.wind_field ? `Wind on the ground: ${esc(r.wind_field.solver)}<br>` : ''}
    Forecast computed in ${t.recompute} ms (${t.members} scenarios, ${r.grid.cell_m} m cells).
    <div class="mobile-only" style="margin-top:12px"><button class="btn ghost" onclick="document.querySelector('#btn-new').click()">Start a new fire</button></div>`;
}

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
