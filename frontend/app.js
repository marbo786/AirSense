// Configuration
const API_BASE = '/api'; // Will be routed via NGINX in docker

// Initialize Map
const map = L.map('map', {
    center: [20, 0],
    zoom: 3,
    zoomControl: false // We will move it
});

// Move zoom control to bottom right so it doesn't collide with our sidebar
L.control.zoom({
    position: 'bottomright'
}).addTo(map);

// Dark theme map tiles (CartoDB Dark Matter)
L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
    attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>',
    subdomains: 'abcd',
    maxZoom: 20
}).addTo(map);

// Color palette mapping based on EPA standards
const colorMap = {
    "Good": "#22c55e", // Green
    "Moderate": "#eab308", // Yellow
    "Unhealthy for Sensitive Groups": "#f97316", // Orange
    "Unhealthy": "#ef4444", // Red
    "Very Unhealthy": "#a855f7", // Purple
    "Hazardous": "#881337", // Dark Maroon
    "Unknown": "#6b7280" // Gray
};

// Application State
let mapNodes = [];

// Fetch Map Data
async function loadMapData() {
    try {
        const response = await fetch(`${API_BASE}/predict/map-global`);
        if (!response.ok) throw new Error('API degraded');
        const data = await response.json();
        
        mapNodes = data.nodes;
        let hazardousCount = 0;

        // Render nodes
        mapNodes.forEach(node => {
            if (!node.lat || !node.lng) return;

            if (node.aqi_category === "Hazardous") hazardousCount++;

            const color = colorMap[node.aqi_category] || colorMap["Unknown"];
            
            // Define marker size based on AQI value (capped)
            const radius = Math.min(Math.max((node.aqi_value || 0) / 10, 4), 15);

            const marker = L.circleMarker([node.lat, node.lng], {
                radius: radius,
                fillColor: color,
                color: color,
                weight: 1,
                opacity: 0.8,
                fillOpacity: 0.6
            }).addTo(map);

            // Bind popup
            const popupHtml = `
                <div class="popup-content">
                    <h4>${node.city}, ${node.country}</h4>
                    <p>AQI: <strong>${Math.round(node.aqi_value)}</strong></p>
                    <p style="color: ${color}; font-weight: 600;">${node.aqi_category}</p>
                </div>
            `;
            marker.bindPopup(popupHtml);
        });

        // Update Stats
        document.getElementById('total-stations').innerText = mapNodes.length.toLocaleString();
        document.getElementById('total-hazardous').innerText = hazardousCount.toLocaleString();

    } catch (err) {
        console.error("Failed to load map data:", err);
        document.getElementById('total-stations').innerText = "ERROR";
    }
}

// Live Prediction Logic
document.getElementById('predict-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const btn = e.target.querySelector('button');
    const originalText = btn.innerText;
    btn.innerText = "Analyzing...";
    btn.style.opacity = 0.7;

    const payload = {
        "PM10": parseFloat(document.getElementById('inp-pm10').value),
        "SO2": parseFloat(document.getElementById('inp-so2').value),
        "NO2": parseFloat(document.getElementById('inp-no2').value),
        "CO": parseFloat(document.getElementById('inp-co').value),
        "O3": parseFloat(document.getElementById('inp-o3').value),
        "TEMP": parseFloat(document.getElementById('inp-temp').value),
        "PRES": parseFloat(document.getElementById('inp-pres').value),
        "DEWP": parseFloat(document.getElementById('inp-dewp').value),
        "RAIN": parseFloat(document.getElementById('inp-rain').value),
        "WSPM": parseFloat(document.getElementById('inp-wspm').value),
        "wd_encoded": parseInt(document.getElementById('inp-wd').value)
    };

    try {
        const response = await fetch(`${API_BASE}/predict/pm25`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        if (!response.ok) throw new Error("API call failed");
        
        const data = await response.json();
        
        // Show result
        const resBox = document.getElementById('prediction-result');
        const resPm = document.getElementById('res-pm25');
        const resCat = document.getElementById('res-category');

        resPm.innerText = `${data.pm25_predicted} µg/m³`;
        resCat.innerText = data.aqi_category;
        
        const color = colorMap[data.aqi_category] || '#fff';
        resCat.style.color = color;
        resExtColor = color.replace(')', ', 0.1)').replace('rgb', 'rgba');
        resCat.style.background = `rgba(255,255,255,0.1)`;

        resBox.style.display = 'block';
        setTimeout(() => {
            resBox.classList.remove('hidden');
        }, 10);

    } catch (err) {
        alert("Prediction failed. Is the API connected?");
        console.error(err);
    } finally {
        btn.innerText = originalText;
        btn.style.opacity = 1;
    }
});

// Activity Window Logic
document.getElementById('recommend-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const btn = e.target.querySelector('button');
    const originalText = btn.innerText;
    btn.innerText = "Scanning...";
    btn.style.opacity = 0.7;

    const stationName = document.getElementById('inp-station').value;

    try {
        const response = await fetch(`${API_BASE}/recommend/activity-window`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ "station": stationName, "top_n": 4 })
        });

        if (!response.ok) throw new Error("API call failed");
        const data = await response.json();
        
        const resBox = document.getElementById('recommend-result');
        const list = document.getElementById('recommend-list');
        list.innerHTML = "";

        data.recommendations.forEach(rec => {
            const isSafe = rec.safety_score >= 0.7;
            const styleClass = isSafe ? 'rec-safe' : 'rec-warn';
            const icon = isSafe ? '🏃‍♂️ Optimal for Exercise' : '😷 Proceed with Caution';
            const bgClass = isSafe ? 'rgba(74, 222, 128, 0.05)' : 'rgba(250, 204, 21, 0.05)';
            
            const li = document.createElement('li');
            li.style.background = bgClass;
            li.style.padding = '0.75rem';
            li.style.borderRadius = '8px';
            li.style.marginBottom = '0.5rem';
            li.style.display = 'flex';
            li.style.flexDirection = 'column';
            li.style.gap = '0.4rem';
            li.style.border = '1px solid rgba(255, 255, 255, 0.08)';

            li.innerHTML = `
                <div style="font-size:0.95rem; font-weight:600; color:var(--text-primary)">
                    ⏱️ Forecast: ${rec.time_label}
                </div>
                <div class="${styleClass}" style="font-size:0.8rem; display:flex; justify-content:space-between; align-items:center;">
                    <span>${icon}</span>
                    <span style="background:rgba(0,0,0,0.3); padding:0.2rem 0.5rem; border-radius:4px;">${(rec.safety_score*100).toFixed(0)}% Air Purity</span>
                </div>
            `;
            list.appendChild(li);
        });

        resBox.style.display = 'block';
        setTimeout(() => {
            resBox.classList.remove('hidden');
        }, 10);

    } catch (err) {
        alert("Failed to fetch safe windows.");
        console.error(err);
    } finally {
        btn.innerText = originalText;
        btn.style.opacity = 1;
    }
});

// ── Modal & Dock Logic ─────────────────────────────────────────────────────────

function closeAllModals(e) {
    document.querySelectorAll('.modal').forEach(m => m.classList.add('hidden'));
    document.querySelectorAll('.dock-btn').forEach(btn => btn.classList.remove('active'));
    if (!e || e.target.innerText.includes("Map")) {
        document.querySelector('.dock-btn').classList.add('active'); // Map btn fallback
    }
}

function openModal(e, id) {
    closeAllModals(e);
    document.getElementById(id).classList.remove('hidden');
    
    // Highlight correct dock button securely
    if (e && e.currentTarget) {
        e.currentTarget.classList.add('active');
    }
    
    // Load data based on modal
    if(id === 'modal-timeseries' && !window.tsLoaded) loadTimeSeries();
    if(id === 'modal-experiments' && !window.expLoaded) loadExperiments();
    if(id === 'modal-clusters' && !window.clustersLoaded) loadClusters();
}

// ── Data Loaders & Charts ──────────────────────────────────────────────────────

let tsChartInstance = null;

async function loadTimeSeries(station = 'Aotizhongxin') {
    try {
        const res = await fetch(`${API_BASE}/metrics/time-series?station=${station}`);
        const data = await res.json();
        
        const ctx = document.getElementById('timeSeriesChart').getContext('2d');
        if (tsChartInstance) tsChartInstance.destroy();
        
        tsChartInstance = new Chart(ctx, {
            type: 'line',
            data: {
                labels: data.dates,
                datasets: [{
                    label: `PM2.5 (${data.station})`,
                    data: data.values,
                    borderColor: '#3b82f6',
                    backgroundColor: 'rgba(59, 130, 246, 0.2)',
                    borderWidth: 2,
                    fill: true,
                    tension: 0.3,
                    pointRadius: 0
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                color: '#f3f4f6',
                scales: {
                    x: { ticks: { color: '#9ca3af' }, grid: { color: 'rgba(255,255,255,0.05)' } },
                    y: { ticks: { color: '#9ca3af' }, grid: { color: 'rgba(255,255,255,0.05)' } }
                }
            }
        });
        window.tsLoaded = true;
    } catch(e) { console.error(e); }
}

async function loadExperiments() {
    try {
        const res = await fetch(`${API_BASE}/metrics/experiments`);
        const data = await res.json();
        
        const grid = document.getElementById('experiment-grid');
        grid.innerHTML = '';
        
        data.experiments.forEach(exp => {
            const card = document.createElement('div');
            card.className = 'experiment-card';
            let metricsHtml = Object.entries(exp.metrics).map(([k,v]) => {
                const val = typeof v === 'number' ? v.toFixed(3) : v;
                return `<li><span>${k}</span> <strong>${val}</strong></li>`;
            }).join('');
            
            card.innerHTML = `
                <h3>${exp.task}</h3>
                <p style="margin-bottom:1rem;color:#f3f4f6;">Best Model: <strong>${exp.best_model}</strong></p>
                <ul>${metricsHtml}</ul>
            `;
            grid.appendChild(card);
        });
        window.expLoaded = true;
    } catch(e) { console.error(e); }
}

async function loadClusters() {
    try {
        const res = await fetch(`${API_BASE}/metrics/projections`);
        const data = await res.json();
        
        const pcaData = data.points.map(p => ({ x: p.pca_x, y: p.pca_y, aqi: p.aqi_category }));
        const tsneData = data.points.map(p => ({ x: p.tsne_x, y: p.tsne_y, aqi: p.aqi_category }));
        
        const scatterCfg = (title) => ({
            type: 'scatter',
            options: {
                responsive: true, maintainAspectRatio: false,
                plugins: { title: { display: true, text: title, color:'#fff' }, legend: {display:false} },
                scales: { x: { ticks: {display:false}, grid:{display:false} }, y: { ticks: {display:false}, grid:{display:false} } }
            }
        });

        // PCA
        const pcaCtx = document.getElementById('pcaChart').getContext('2d');
        const pcaChart = new Chart(pcaCtx, scatterCfg('PCA Projection'));
        pcaChart.data = { datasets: [{ data: pcaData, backgroundColor: '#8b5cf6', pointRadius: 3 }] };
        pcaChart.update();

        // t-SNE
        const tsneCtx = document.getElementById('tsneChart').getContext('2d');
        const tsneChart = new Chart(tsneCtx, scatterCfg('t-SNE Projection'));
        tsneChart.data = { datasets: [{ data: tsneData, backgroundColor: '#ec4899', pointRadius: 3 }] };
        tsneChart.update();

        window.clustersLoaded = true;
    } catch(e) { console.error(e); }
}

// Start
loadMapData();
