// ── Configuration ──────────────────────────────────────────────────────────────
const API_BASE = '/api';
const MAPBOX_TOKEN = 'MAPBOX_TOKEN_PLACEHOLDER';

// AQI colour palette (EPA standard)
const AQI_COLORS = {
  'Good':                            '#22c55e',
  'Moderate':                        '#eab308',
  'Unhealthy for Sensitive Groups':  '#f97316',
  'Unhealthy':                       '#ef4444',
  'Very Unhealthy':                  '#a855f7',
  'Hazardous':                       '#ff0055',
  'Unknown':                         '#6b7280',
};

// ── Globe init ─────────────────────────────────────────────────────────────────
mapboxgl.accessToken = MAPBOX_TOKEN;

const map = new mapboxgl.Map({
  container: 'map',
  // Dark satellite base — one style load, no extra API calls
  style: 'mapbox://styles/mapbox/satellite-streets-v12',
  projection: 'globe',
  zoom: 2.2,
  center: [15, 15],
  pitch: 20,
  bearing: -10,
  antialias: true,
});

// ── Globe atmosphere & star field ──────────────────────────────────────────────
map.on('style.load', () => {
  // Rich atmosphere glow
  map.setFog({
    color: 'rgb(10, 15, 35)',
    'high-color': 'rgb(20, 40, 100)',
    'horizon-blend': 0.06,
    'space-color': 'rgb(4, 7, 20)',
    'star-intensity': 0.7,
  });

  // ── 3D Terrain ──
  map.addSource('mapbox-dem', {
    'type': 'raster-dem',
    'url': 'mapbox://mapbox.mapbox-terrain-dem-v1',
    'tileSize': 512,
    'maxzoom': 14
  });
  map.setTerrain({ 'source': 'mapbox-dem', 'exaggeration': 1.5 });

  // Load AQI data after style is ready
  loadMapData();
});

// ── Auto-rotate (stops on user interaction) ────────────────────────────────────
let rotating = true;
let animFrameId = null;

function startRotation() {
  if (!rotating) return;
  const bearing = map.getBearing();
  map.setBearing((bearing + 0.04) % 360);
  animFrameId = requestAnimationFrame(startRotation);
}

map.on('mousedown', () => { rotating = false; cancelAnimationFrame(animFrameId); });
map.on('touchstart', () => { rotating = false; cancelAnimationFrame(animFrameId); });

// Re-enable after 8 s of inactivity
let idleTimer;
['mouseup', 'touchend'].forEach(evt => {
  map.on(evt, () => {
    clearTimeout(idleTimer);
    idleTimer = setTimeout(() => { rotating = true; startRotation(); }, 8000);
  });
});

map.once('style.load', () => {
  setTimeout(() => startRotation(), 1200);
});

// ── Hover tooltip ──────────────────────────────────────────────────────────────
const tooltip = document.getElementById('map-tooltip');

function showTooltip(e, node) {
  const color = AQI_COLORS[node.aqi_category] || AQI_COLORS['Unknown'];
  tooltip.innerHTML = `
    <div class="tt-city">${node.city}, ${node.country}</div>
    <div class="tt-aqi">AQI <strong>${Math.round(node.aqi_value)}</strong></div>
    <div class="tt-cat" style="color:${color}">${node.aqi_category}</div>
  `;
  tooltip.classList.remove('hidden');
  moveTooltip(e);
}

function moveTooltip(e) {
  const x = e.originalEvent.clientX;
  const y = e.originalEvent.clientY;
  tooltip.style.left = (x + 14) + 'px';
  tooltip.style.top  = (y - 10) + 'px';
}

function hideTooltip() {
  tooltip.classList.add('hidden');
}

// ── Load & render AQI data ────────────────────────────────────────────────────
let mapNodesCache = [];

async function loadMapData() {
  try {
    const res = await fetch(`${API_BASE}/predict/map-global`);
    if (!res.ok) throw new Error('API degraded');
    const data = await res.json();
    mapNodesCache = data.nodes;

    const geojson = {
  type: 'FeatureCollection',
  features: data.nodes
    .filter(n => n.lat && n.lng)
    .map(n => ({
      type: 'Feature',
      geometry: { type: 'Point', coordinates: [n.lng, n.lat] },
      properties: {
        city: n.city, country: n.country,
        aqi_value: n.aqi_value || 0,
        aqi_category: n.aqi_category || 'Unknown',
        color: AQI_COLORS[n.aqi_category] || AQI_COLORS['Unknown'],
        cluster: n.cluster, // Add cluster property
      },
    })),
};

    // Plain GeoJSON source — no clustering
    map.addSource('aqi-points', { type: 'geojson', data: geojson });

    // ── 1. Heatmap (visible at low zoom, fades at zoom 6) ──
    map.addLayer({
      id: 'aqi-heat',
      type: 'heatmap',
      source: 'aqi-points',
      maxzoom: 8,
      paint: {
        'heatmap-weight': ['interpolate', ['linear'], ['get', 'aqi_value'], 0, 0, 500, 1],
        'heatmap-intensity': ['interpolate', ['linear'], ['zoom'], 0, 0.6, 8, 2],
        'heatmap-color': [
          'interpolate', ['linear'], ['heatmap-density'],
          0,   'rgba(0,0,0,0)',
          0.15,'rgba(34,197,94,0.6)',
          0.35,'rgba(234,179,8,0.75)',
          0.55,'rgba(249,115,22,0.85)',
          0.75,'rgba(239,68,68,0.9)',
          0.9, 'rgba(168,85,247,0.95)',
          1.0, 'rgba(255,0,85,1)',
        ],
        'heatmap-radius': ['interpolate', ['linear'], ['zoom'], 0, 12, 5, 28, 8, 45],
        'heatmap-opacity': ['interpolate', ['linear'], ['zoom'], 5, 1, 8, 0],
      },
    });

    // ── 2. Outer glow halo ──
    map.addLayer({
      id: 'aqi-glow',
      type: 'circle',
      source: 'aqi-points',
      paint: {
        'circle-color': ['get', 'color'],
        'circle-radius': ['interpolate', ['linear'], ['get', 'aqi_value'], 0, 8, 100, 18, 300, 26, 500, 34],
        'circle-blur': 0.85,
        'circle-opacity': 0.35,
      },
    });

    // ── 3. Solid core dot ──
    map.addLayer({
      id: 'aqi-dots',
      type: 'circle',
      source: 'aqi-points',
      paint: {
        'circle-color': ['get', 'color'],
        'circle-radius': ['interpolate', ['linear'], ['get', 'aqi_value'], 0, 3, 100, 6, 300, 9, 500, 12],
        'circle-stroke-width': 0.8,
        'circle-stroke-color': 'rgba(255,255,255,0.25)',
        'circle-opacity': 0.9,
      },
    });

    // ── Dot hover & click ──
    map.on('mouseenter', 'aqi-dots', e => {
      map.getCanvas().style.cursor = 'pointer';
      showTooltip(e, e.features[0].properties);
    });
    map.on('mousemove', 'aqi-dots', e => {
      showTooltip(e, e.features[0].properties); moveTooltip(e);
    });
    map.on('mouseleave', 'aqi-dots', () => {
      map.getCanvas().style.cursor = ''; hideTooltip();
    });
    map.on('click', 'aqi-dots', e => {
      rotating = false; cancelAnimationFrame(animFrameId);
      map.flyTo({
        center: e.features[0].geometry.coordinates, zoom: 5,
        pitch: 50, bearing: map.getBearing() + 15, duration: 1800, essential: true,
      });
      clearTimeout(idleTimer);
      idleTimer = setTimeout(() => { rotating = true; startRotation(); }, 10000);
    });

    // ── Stats ──
    const hazardous = data.nodes.filter(n => n.aqi_category === 'Hazardous').length;
    document.getElementById('total-stations').innerText = data.nodes.length.toLocaleString();
    document.getElementById('total-hazardous').innerText = hazardous.toLocaleString();

  } catch (err) {
    console.error('Failed to load map data:', err);
    document.getElementById('total-stations').innerText = 'ERROR';
  }
}

// ── Geocoding search ───────────────────────────────────────────────────────────
let searchTimeout = null;

async function geocodeFetch(query) {
  const url = `https://api.mapbox.com/geocoding/v5/mapbox.places/${encodeURIComponent(query)}.json`
    + `?access_token=${MAPBOX_TOKEN}&limit=5&types=place,region,country`;
  const res = await fetch(url);
  const data = await res.json();
  return data.features || [];
}

const searchInput  = document.getElementById('geo-search-input');
const searchDrop   = document.getElementById('geo-search-drop');

function renderDropdown(features) {
  searchDrop.innerHTML = '';
  if (!features.length) { searchDrop.classList.add('hidden'); return; }
  features.forEach(f => {
    const li = document.createElement('li');
    li.textContent = f.place_name;
    li.addEventListener('click', () => {
      searchInput.value = f.place_name;
      searchDrop.classList.add('hidden');
      rotating = false; cancelAnimationFrame(animFrameId);
      map.flyTo({ center: f.center, zoom: 5, pitch: 45, duration: 2200, essential: true });
      clearTimeout(idleTimer);
      idleTimer = setTimeout(() => { rotating = true; startRotation(); }, 12000);
    });
    searchDrop.appendChild(li);
  });
  searchDrop.classList.remove('hidden');
}

if (searchInput) {
  searchInput.addEventListener('input', () => {
    clearTimeout(searchTimeout);
    const q = searchInput.value.trim();
    if (q.length < 2) { searchDrop.classList.add('hidden'); return; }
    searchTimeout = setTimeout(async () => {
      const features = await geocodeFetch(q);
      renderDropdown(features);
    }, 300);
  });
  document.addEventListener('click', e => {
    if (!e.target.closest('.geo-search-wrap')) searchDrop.classList.add('hidden');
  });
}


// ── Live PM2.5 Prediction ──────────────────────────────────────────────────────
document.getElementById('predict-form').addEventListener('submit', async (e) => {
  e.preventDefault();

  const btn = document.getElementById('btn-predict');
  btn.innerText = 'Analyzing…';
  btn.style.opacity = '0.7';

  const payload = {
    PM10: parseFloat(document.getElementById('inp-pm10').value),
    SO2:  parseFloat(document.getElementById('inp-so2').value),
    NO2:  parseFloat(document.getElementById('inp-no2').value),
    CO:   parseFloat(document.getElementById('inp-co').value),
    O3:   parseFloat(document.getElementById('inp-o3').value),
    TEMP: parseFloat(document.getElementById('inp-temp').value),
    PRES: parseFloat(document.getElementById('inp-pres').value),
    DEWP: parseFloat(document.getElementById('inp-dewp').value),
    RAIN: parseFloat(document.getElementById('inp-rain').value),
    WSPM: parseFloat(document.getElementById('inp-wspm').value),
    wd_encoded: parseInt(document.getElementById('inp-wd').value),
  };

  try {
    const res = await fetch(`${API_BASE}/predict/pm25`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });

    if (!res.ok) throw new Error('API call failed');
    const data = await res.json();

    const resBox = document.getElementById('prediction-result');
    document.getElementById('res-pm25').innerText = `${data.pm25_predicted} µg/m³`;

    const cat = document.getElementById('res-category');
    cat.innerText = data.aqi_category;
    const col = AQI_COLORS[data.aqi_category] || '#fff';
    cat.style.color = col;
    cat.style.boxShadow = `0 0 12px ${col}55`;

    resBox.style.display = 'block';
    setTimeout(() => resBox.classList.remove('hidden'), 10);

  } catch (err) {
    alert('Prediction failed. Is the API connected?');
    console.error(err);
  } finally {
    btn.innerText = 'Initialize Prediction';
    btn.style.opacity = '1';
  }
});

// ── Outdoor Safety Recommendation ─────────────────────────────────────────────
document.getElementById('recommend-form').addEventListener('submit', async (e) => {
  e.preventDefault();

  const btn = document.getElementById('btn-recommend');
  btn.innerText = 'Scanning…';
  btn.style.opacity = '0.7';

  const station = document.getElementById('inp-station').value;

  try {
    const res = await fetch(`${API_BASE}/recommend/activity-window`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ station, top_n: 4 }),
    });

    if (!res.ok) throw new Error('API call failed');
    const data = await res.json();

    const list = document.getElementById('recommend-list');
    list.innerHTML = '';

    data.recommendations.forEach(rec => {
      const isSafe = rec.safety_score >= 0.7;
      const li = document.createElement('li');
      li.style.cssText = `
        background: ${isSafe ? 'rgba(74,222,128,0.05)' : 'rgba(250,204,21,0.05)'};
        padding: .75rem; border-radius: 8px; margin-bottom: .5rem;
        display:flex; flex-direction:column; gap:.4rem;
        border: 1px solid rgba(255,255,255,.08);
      `;
      li.innerHTML = `
        <div style="font-size:.95rem;font-weight:600;color:var(--text-primary)">
          ⏱️ ${rec.time_label}
        </div>
        <div style="font-size:.8rem;display:flex;justify-content:space-between;align-items:center;
                    color:${isSafe ? '#4ade80' : '#facc15'}">
          <span>${isSafe ? '🏃‍♂️ Optimal for Exercise' : '😷 Proceed with Caution'}</span>
          <span style="background:rgba(0,0,0,.3);padding:.2rem .5rem;border-radius:4px">
            ${(rec.safety_score * 100).toFixed(0)}% Air Purity
          </span>
        </div>
      `;
      list.appendChild(li);
    });

    const resBox = document.getElementById('recommend-result');
    resBox.style.display = 'block';
    setTimeout(() => resBox.classList.remove('hidden'), 10);

  } catch (err) {
    alert('Failed to fetch safe windows.');
    console.error(err);
  } finally {
    btn.innerText = 'Forecast Safe Outdoor Times';
    btn.style.opacity = '1';
  }
});

// ── Modal & Dock ───────────────────────────────────────────────────────────────
function closeAllModals(e) {
  document.querySelectorAll('.modal').forEach(m => m.classList.add('hidden'));
  document.querySelectorAll('.dock-btn').forEach(b => b.classList.remove('active'));
  document.getElementById('dock-map').classList.add('active');
}

function openModal(e, id) {
  closeAllModals(e);
  document.getElementById(id).classList.remove('hidden');
  if (e && e.currentTarget) e.currentTarget.classList.add('active');
  if (id === 'modal-timeseries'  && !window.tsLoaded)       loadTimeSeries();
  if (id === 'modal-experiments' && !window.expLoaded)      loadExperiments();
  if (id === 'modal-clusters'    && !window.clustersLoaded) loadClusters();
}

// ── Chart Loaders ──────────────────────────────────────────────────────────────
let tsChartInstance = null;

async function loadTimeSeries(station = 'Aotizhongxin') {
  try {
    const res  = await fetch(`${API_BASE}/metrics/time-series?station=${station}`);
    const data = await res.json();
    const ctx  = document.getElementById('timeSeriesChart').getContext('2d');
    if (tsChartInstance) tsChartInstance.destroy();
    tsChartInstance = new Chart(ctx, {
      type: 'line',
      data: {
        labels: data.dates,
        datasets: [{
          label: `PM2.5 (${data.station})`,
          data: data.values,
          borderColor: '#3b82f6',
          backgroundColor: 'rgba(59,130,246,0.15)',
          borderWidth: 2,
          fill: true,
          tension: 0.3,
          pointRadius: 0,
        }],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        color: '#f3f4f6',
        scales: {
          x: { ticks: { color: '#9ca3af' }, grid: { color: 'rgba(255,255,255,.05)' } },
          y: { ticks: { color: '#9ca3af' }, grid: { color: 'rgba(255,255,255,.05)' } },
        },
      },
    });
    window.tsLoaded = true;
  } catch (err) { console.error(err); }
}

async function loadExperiments() {
  try {
    const res  = await fetch(`${API_BASE}/metrics/experiments`);
    const data = await res.json();
    const grid = document.getElementById('experiment-grid');
    grid.innerHTML = '';
    data.experiments.forEach(exp => {
      const card = document.createElement('div');
      card.className = 'experiment-card';
      const metricsHtml = Object.entries(exp.metrics)
        .map(([k, v]) => `<li><span>${k}</span> <strong>${typeof v === 'number' ? v.toFixed(3) : v}</strong></li>`)
        .join('');
      card.innerHTML = `
        <h3>${exp.task}</h3>
        <p style="margin-bottom:1rem;color:#f3f4f6">Best Model: <strong>${exp.best_model}</strong></p>
        <ul>${metricsHtml}</ul>
      `;
      grid.appendChild(card);
    });
    window.expLoaded = true;
  } catch (err) { console.error(err); }
}

async function loadClusters() {
  try {
    const res  = await fetch(`${API_BASE}/metrics/projections`);
    const data = await res.json();

    const pcaData  = data.points.map(p => ({ x: p.pca_x,  y: p.pca_y,  aqi: p.aqi_category }));
    const tsneData = data.points.map(p => ({ x: p.tsne_x, y: p.tsne_y, aqi: p.aqi_category }));

    const scatterCfg = title => ({
      type: 'scatter',
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          title: { display: true, text: title, color: '#fff' },
          legend: { display: false },
        },
        scales: {
          x: { ticks: { display: false }, grid: { display: false } },
          y: { ticks: { display: false }, grid: { display: false } },
        },
      },
    });

    const pcaCtx = document.getElementById('pcaChart').getContext('2d');
    const pcaChart = new Chart(pcaCtx, scatterCfg('PCA Projection'));
    pcaChart.data = { datasets: [{ data: pcaData,  backgroundColor: '#8b5cf6', pointRadius: 3 }] };
    pcaChart.update();

    const tsneCtx = document.getElementById('tsneChart').getContext('2d');
    const tsneChart = new Chart(tsneCtx, scatterCfg('t-SNE Projection'));
    tsneChart.data = { datasets: [{ data: tsneData, backgroundColor: '#ec4899', pointRadius: 3 }] };
    tsneChart.update();

    window.clustersLoaded = true;
  } catch (err) { console.error(err); }
}
