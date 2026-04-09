/**
 * IoT Predictive Maintenance — Real-time Dashboard
 * WebSocket-driven frontend with Chart.js visualizations.
 */

// ============================================================================
// STATE
// ============================================================================

const APP = {
    ws: null,
    config: null,
    connected: false,
    sensorHistory: {},
    timestamps: [],
    anomalyHistory: [],
    healthHistory: [],
    maxPoints: 120,
    sparklineCharts: {},
    anomalyChart: null,
    healthChart: null,
    modalChart: null,
    currentHealth: 100,
    currentAlert: 'Normal',
    currentFault: 'Healthy',
    sampleCount: 0,
    lastData: null,          // latest sensor_data message
    activeSensors: new Set(),  // which sensors are enabled
};

// ============================================================================
// INIT
// ============================================================================

document.addEventListener('DOMContentLoaded', () => {
    connectWebSocket();
    setupEventListeners();
});

function connectWebSocket() {
    const proto = location.protocol === 'https:' ? 'wss' : 'ws';
    const url = `${proto}://${location.host}/ws`;
    APP.ws = new WebSocket(url);
    APP.ws.onopen = () => { APP.connected = true; updateConnectionBadge(); };
    APP.ws.onclose = () => {
        APP.connected = false; updateConnectionBadge();
        setTimeout(connectWebSocket, 3000);
    };
    APP.ws.onerror = () => {};
    APP.ws.onmessage = (e) => handleMessage(JSON.parse(e.data));
}

function handleMessage(msg) {
    if (msg.type === 'init') { APP.config = msg.config; initDashboard(); }
    else if (msg.type === 'sensor_data') { updateDashboard(msg); }
    else if (msg.type === 'status') { handleStatusUpdate(msg); }
    else if (msg.type === 'server_status') { updateServerStatus(msg); }
}

// ============================================================================
// DASHBOARD INIT
// ============================================================================

function initDashboard() {
    const cfg = APP.config;
    cfg.feature_names.forEach(f => {
        APP.sensorHistory[f] = [];
        APP.activeSensors.add(f);
    });
    buildSensorToggles();
    buildSensorCards();
    buildAnomalyChart();
    buildHealthChart();
}

// ---- Sensor Toggles (sidebar) ----
function buildSensorToggles() {
    const container = document.getElementById('sensor-toggles');
    if (!container || !APP.config) return;
    container.innerHTML = '';
    APP.config.feature_names.forEach(feat => {
        const label = (APP.config.feature_labels[feat] || feat).replace(/^[^\s]+\s/, '');
        const color = APP.config.sensor_colors[feat] || '#6366f1';
        const el = document.createElement('label');
        el.className = 'sensor-toggle';
        el.innerHTML = `
            <input type="checkbox" checked data-feat="${feat}">
            <span class="toggle-dot" style="background:${color}"></span>
            <span>${label}</span>
        `;
        el.querySelector('input').addEventListener('change', (e) => {
            if (e.target.checked) APP.activeSensors.add(feat);
            else APP.activeSensors.delete(feat);
            rebuildSensorCards();
        });
        container.appendChild(el);
    });
}

// ---- Sensor Cards with Mini Gauge ----
function buildSensorCards() { rebuildSensorCards(); }

function rebuildSensorCards() {
    const container = document.getElementById('sensor-cards');
    if (!container || !APP.config) return;
    container.innerHTML = '';

    // Destroy old sparkline charts
    Object.keys(APP.sparklineCharts).forEach(k => {
        APP.sparklineCharts[k]?.destroy();
        delete APP.sparklineCharts[k];
    });

    const cfg = APP.config;
    const active = cfg.feature_names.filter(f => APP.activeSensors.has(f));

    if (!active.length) {
        container.innerHTML = '<div class="card" style="grid-column:span 3;text-align:center"><p class="info-text" style="padding:30px">No sensors selected</p></div>';
        return;
    }

    active.forEach(feat => {
        const label = cfg.feature_labels[feat] || feat;
        const unit = cfg.feature_units[feat] || '';
        const color = cfg.sensor_colors[feat] || '#6366f1';
        const miniR = 20;
        const miniCirc = 2 * Math.PI * miniR;

        const card = document.createElement('div');
        card.className = 'sensor-card';
        card.id = `sensor-${feat}`;
        card.innerHTML = `
            <div class="sensor-status" style="background:${color}"></div>
            <div class="sensor-row">
                <div class="mini-gauge" id="mini-gauge-${feat}">
                    <svg viewBox="0 0 48 48">
                        <circle class="mini-track" cx="24" cy="24" r="${miniR}" />
                        <circle class="mini-fill" cx="24" cy="24" r="${miniR}"
                            id="mini-fill-${feat}"
                            style="stroke:${color}; stroke-dasharray:${miniCirc}; stroke-dashoffset:0" />
                    </svg>
                    <div class="mini-value" id="mini-pct-${feat}" style="color:${color}">—</div>
                </div>
                <div class="sensor-info">
                    <div class="sensor-name">${label}</div>
                    <div class="sensor-value" id="val-${feat}" style="color:${color}">—</div>
                    <span class="sensor-unit">${unit}</span>
                </div>
            </div>
            <div class="sparkline-container">
                <canvas id="spark-${feat}"></canvas>
            </div>
            <div class="diagnose-hint">🔬 Click to diagnose</div>
        `;
        card.addEventListener('click', () => openLightbox(feat));
        container.appendChild(card);

        // Build sparkline
        const ctx = document.getElementById(`spark-${feat}`).getContext('2d');
        APP.sparklineCharts[feat] = new Chart(ctx, {
            type: 'line',
            data: {
                labels: APP.timestamps.slice(),
                datasets: [{
                    data: (APP.sensorHistory[feat] || []).slice(),
                    borderColor: color,
                    backgroundColor: hexToRgba(color, 0.1),
                    borderWidth: 1.5, fill: true, tension: 0.4, pointRadius: 0,
                }]
            },
            options: {
                responsive: true, maintainAspectRatio: false,
                plugins: { legend: { display: false }, tooltip: { enabled: false } },
                scales: { x: { display: false }, y: { display: false } },
                animation: { duration: 300 },
            }
        });
    });
}

function buildAnomalyChart() {
    const ctx = document.getElementById('anomaly-chart').getContext('2d');
    APP.anomalyChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [
                { label: 'Anomaly Score', data: [], borderColor: '#a855f7', backgroundColor: 'rgba(168,85,247,0.08)', borderWidth: 2, fill: true, tension: 0.3, pointRadius: 0 },
                { label: 'Threshold', data: [], borderColor: '#ef4444', borderWidth: 1.5, borderDash: [6, 4], fill: false, pointRadius: 0 }
            ]
        },
        options: chartOptions('Anomaly Score'),
    });
}

function buildHealthChart() {
    const ctx = document.getElementById('health-chart').getContext('2d');
    APP.healthChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{ label: 'Health Score', data: [], borderColor: '#06b6d4', backgroundColor: 'rgba(6,182,212,0.08)', borderWidth: 2, fill: true, tension: 0.3, pointRadius: 0 }]
        },
        options: chartOptions('Health %'),
    });
}

function chartOptions(yLabel) {
    return {
        responsive: true, maintainAspectRatio: false,
        interaction: { intersect: false, mode: 'index' },
        plugins: {
            legend: { display: true, labels: { color: '#94a3b8', font: { size: 11, family: 'Inter' }, boxWidth: 12 } },
            tooltip: { backgroundColor: 'rgba(10,14,26,0.9)', titleColor: '#f1f5f9', bodyColor: '#94a3b8', borderColor: 'rgba(255,255,255,0.1)', borderWidth: 1, cornerRadius: 8 }
        },
        scales: {
            x: { display: true, ticks: { color: '#64748b', font: { size: 10 }, maxTicksLimit: 8 }, grid: { color: 'rgba(255,255,255,0.03)' } },
            y: { display: true, title: { display: true, text: yLabel, color: '#64748b', font: { size: 11 } }, ticks: { color: '#64748b', font: { size: 10 } }, grid: { color: 'rgba(255,255,255,0.04)' } }
        },
        animation: { duration: 400 },
    };
}

// ============================================================================
// REAL-TIME UPDATES
// ============================================================================

function updateDashboard(data) {
    if (!APP.config) return;
    APP.lastData = data;

    const timeLabel = new Date(data.timestamp).toLocaleTimeString('en-US', { hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' });

    APP.timestamps.push(timeLabel);
    APP.anomalyHistory.push(data.anomaly_score);
    APP.healthHistory.push(data.health_score);
    if (APP.timestamps.length > APP.maxPoints) {
        APP.timestamps.shift(); APP.anomalyHistory.shift(); APP.healthHistory.shift();
    }

    APP.config.feature_names.forEach(feat => {
        const val = data.sensors[feat];
        APP.sensorHistory[feat].push(val);
        if (APP.sensorHistory[feat].length > APP.maxPoints) APP.sensorHistory[feat].shift();

        // Only update cards for active sensors
        if (!APP.activeSensors.has(feat)) return;

        const valEl = document.getElementById(`val-${feat}`);
        if (valEl) valEl.textContent = val !== null ? val.toFixed(2) : '—';

        // Update sparkline
        const chart = APP.sparklineCharts[feat];
        if (chart) {
            chart.data.labels = APP.timestamps.slice();
            chart.data.datasets[0].data = APP.sensorHistory[feat].slice();
            chart.update('none');
        }

        // Update mini gauge — shows per-sensor health %
        updateMiniGauge(feat, data.per_feature_errors, data.threshold);

        // Update status dot
        const statusDot = document.querySelector(`#sensor-${feat} .sensor-status`);
        if (statusDot && data.per_feature_errors) {
            const err = data.per_feature_errors[feat] || 0;
            const thr = data.threshold;
            if (err > thr) { statusDot.style.background = '#ef4444'; statusDot.style.boxShadow = '0 0 8px rgba(239,68,68,0.5)'; }
            else if (err > thr * 0.7) { statusDot.style.background = '#eab308'; statusDot.style.boxShadow = '0 0 8px rgba(234,179,8,0.4)'; }
            else { statusDot.style.background = '#22c55e'; statusDot.style.boxShadow = 'none'; }
        }
    });

    // Main charts
    APP.anomalyChart.data.labels = APP.timestamps.slice();
    APP.anomalyChart.data.datasets[0].data = APP.anomalyHistory.slice();
    APP.anomalyChart.data.datasets[1].data = Array(APP.timestamps.length).fill(data.threshold);
    APP.anomalyChart.update('none');

    APP.healthChart.data.labels = APP.timestamps.slice();
    APP.healthChart.data.datasets[0].data = APP.healthHistory.slice();
    APP.healthChart.update('none');

    updateHealthGauge(data.health_score);
    updateAlertBanner(data.alert_level, data.fault_type);
    updateMetrics(data);
    updateFaultBars(data.per_feature_errors, data.threshold);

    APP.currentHealth = data.health_score;
    APP.currentAlert = data.alert_level;
    APP.currentFault = data.fault_type;
    APP.sampleCount = data.sample_count;
}

function updateMiniGauge(feat, perFeatureErrors, threshold) {
    const err = (perFeatureErrors && perFeatureErrors[feat]) || 0;
    const k = 0.347;
    const normalized = err / (threshold + 1e-10);
    const health = Math.max(0, Math.min(100, 100 * Math.exp(-k * normalized)));

    const miniR = 20;
    const circ = 2 * Math.PI * miniR;
    const offset = circ * (1 - health / 100);

    const fill = document.getElementById(`mini-fill-${feat}`);
    const pct = document.getElementById(`mini-pct-${feat}`);
    if (!fill || !pct) return;

    fill.style.strokeDashoffset = offset;
    if (health >= 80) fill.style.stroke = '#22c55e';
    else if (health >= 50) fill.style.stroke = '#eab308';
    else fill.style.stroke = '#ef4444';

    pct.textContent = `${Math.round(health)}`;
    pct.style.color = health >= 80 ? '#22c55e' : health >= 50 ? '#eab308' : '#ef4444';
}

function updateHealthGauge(health) {
    const circumference = 2 * Math.PI * 72;
    const offset = circumference * (1 - health / 100);
    const fill = document.getElementById('gauge-fill');
    const numberEl = document.getElementById('gauge-number');
    if (fill) {
        fill.style.strokeDashoffset = offset;
        fill.style.strokeDasharray = circumference;
        fill.style.stroke = health >= 80 ? '#22c55e' : health >= 50 ? '#eab308' : '#ef4444';
    }
    if (numberEl) {
        numberEl.textContent = `${Math.round(health)}%`;
        numberEl.style.color = health >= 80 ? '#22c55e' : health >= 50 ? '#eab308' : '#ef4444';
    }
}

function updateAlertBanner(alertLevel, faultType) {
    const banner = document.getElementById('alert-banner');
    if (!banner) return;
    banner.className = 'alert-banner';
    if (alertLevel === 'Critical') { banner.classList.add('alert-critical'); banner.innerHTML = `🚨 <span>ALERT — ${faultType}</span>`; }
    else if (alertLevel === 'Warning') { banner.classList.add('alert-warning'); banner.innerHTML = `⚠️ <span>WARNING — ${faultType}</span>`; }
    else { banner.classList.add('alert-normal'); banner.innerHTML = `✅ <span>All Systems Nominal</span>`; }
}

function updateMetrics(data) {
    setText('metric-samples', data.sample_count);
    setText('metric-anomaly', data.anomaly_score.toFixed(5));
    setText('metric-fault', data.fault_type);
    const uptimeEl = document.getElementById('metric-uptime');
    if (uptimeEl && APP.healthHistory.length) {
        const pct = (APP.healthHistory.filter(h => h >= 80).length / APP.healthHistory.length * 100);
        uptimeEl.textContent = `${pct.toFixed(1)}%`;
    }
}

function updateFaultBars(perFeatureErrors, threshold) {
    const container = document.getElementById('fault-bars');
    if (!container || !APP.config) return;

    const entries = APP.config.feature_names
        .filter(f => APP.activeSensors.has(f))
        .map(f => ({
            feat: f,
            label: (APP.config.feature_labels[f] || f).replace(/^[^\s]+\s/, ''),
            error: perFeatureErrors[f] || 0,
        }))
        .sort((a, b) => b.error - a.error);

    const maxError = Math.max(threshold * 1.5, ...entries.map(e => e.error)) || 1;

    container.innerHTML = entries.map(e => {
        const pct = Math.min(100, (e.error / maxError) * 100);
        const threshPct = Math.min(100, (threshold / maxError) * 100);
        let color = e.error > threshold ? '#ef4444' : e.error > threshold * 0.7 ? '#eab308' : '#22c55e';
        return `
            <div class="fault-bar">
                <span class="fault-bar-label">${e.label}</span>
                <div class="fault-bar-track">
                    <div class="fault-bar-fill" style="width:${pct}%;background:${color}">${(e.error / threshold * 100).toFixed(0)}%</div>
                    <div class="fault-bar-threshold" style="left:${threshPct}%"></div>
                </div>
            </div>`;
    }).join('');
}

// ============================================================================
// LIGHTBOX MODAL
// ============================================================================

function openLightbox(feat) {
    const overlay = document.getElementById('modal-overlay');
    if (!overlay || !APP.config || !APP.lastData) return;

    const cfg = APP.config;
    const label = cfg.feature_labels[feat] || feat;
    const unit = cfg.feature_units[feat] || '';
    const color = cfg.sensor_colors[feat] || '#6366f1';
    const err = APP.lastData.per_feature_errors[feat] || 0;
    const thr = APP.lastData.threshold;
    const errPct = (err / (thr + 1e-10)) * 100;

    // Title
    setText('modal-title', label);

    // Status badge
    const badge = document.getElementById('modal-status-badge');
    if (err > thr) {
        badge.className = 'badge badge-red'; badge.innerHTML = '🔴 OVER THRESHOLD';
    } else if (err > thr * 0.7) {
        badge.className = 'badge badge-yellow'; badge.innerHTML = '🟡 APPROACHING';
    } else {
        badge.className = 'badge badge-green'; badge.innerHTML = '🟢 Healthy';
    }

    // Top metrics
    const history = APP.sensorHistory[feat] || [];
    const currentVal = history.length ? history[history.length - 1] : 0;
    setText('modal-current', `${currentVal.toFixed(2)} ${unit}`);
    setText('modal-error', `${err.toFixed(5)} (${errPct.toFixed(0)}%)`);
    setText('modal-min', history.length ? `${Math.min(...history).toFixed(2)} ${unit}` : '—');
    setText('modal-max', history.length ? `${Math.max(...history).toFixed(2)} ${unit}` : '—');

    // Alert message
    const alertEl = document.getElementById('modal-alert');
    if (err > thr) {
        alertEl.className = 'modal-alert alert-crit';
        alertEl.innerHTML = `🔴 <strong>Sensor OVER anomaly threshold.</strong> Error ${err.toFixed(5)} > threshold ${thr.toFixed(5)}. Primary contributor to current alert.`;
    } else if (err > thr * 0.7) {
        alertEl.className = 'modal-alert alert-warn';
        alertEl.innerHTML = `🟡 <strong>Approaching threshold.</strong> Error at ${errPct.toFixed(0)}% — monitor closely.`;
    } else {
        alertEl.className = 'modal-alert alert-ok';
        alertEl.innerHTML = `🟢 <strong>Sensor healthy.</strong> Error at ${errPct.toFixed(0)}% of threshold — normal operation.`;
    }

    // Rolling stats
    const recent = history.slice(-50);
    if (recent.length >= 10) {
        const mean = recent.reduce((a, b) => a + b, 0) / recent.length;
        const variance = recent.reduce((a, b) => a + (b - mean) ** 2, 0) / recent.length;
        const std = Math.sqrt(variance);
        const n = recent.length;
        const skew = recent.reduce((a, b) => a + ((b - mean) / (std + 1e-10)) ** 3, 0) / n;

        setText('modal-mean', mean.toFixed(2));
        setText('modal-std', std.toFixed(3));
        setText('modal-skew', skew.toFixed(2));

        // Drift rate (linear regression slope)
        if (recent.length >= 20) {
            const x = Array.from({ length: recent.length }, (_, i) => i);
            const xMean = (recent.length - 1) / 2;
            const yMean = mean;
            let num = 0, den = 0;
            for (let i = 0; i < recent.length; i++) {
                num += (x[i] - xMean) * (recent[i] - yMean);
                den += (x[i] - xMean) ** 2;
            }
            const slope = num / (den + 1e-10);
            const driftEl = document.getElementById('modal-drift');
            if (driftEl) {
                driftEl.textContent = `${slope >= 0 ? '+' : ''}${slope.toFixed(4)}/s`;
                driftEl.style.color = Math.abs(slope) > 0.01 ? '#eab308' : '#22c55e';
            }
        } else {
            setText('modal-drift', '—');
        }
    } else {
        setText('modal-mean', '—'); setText('modal-std', '—');
        setText('modal-skew', '—'); setText('modal-drift', '—');
    }

    // Cross-validation (temp sensors only)
    const cvPanel = document.getElementById('modal-crossval');
    if ((feat === 'temp_dht' || feat === 'temp_therm') && APP.sensorHistory.temp_dht && APP.sensorHistory.temp_therm) {
        cvPanel.style.display = 'block';
        const dht = APP.sensorHistory.temp_dht.slice(-50);
        const therm = APP.sensorHistory.temp_therm.slice(-50);
        const len = Math.min(dht.length, therm.length);
        if (len > 0) {
            const diffs = [];
            for (let i = 0; i < len; i++) diffs.push(Math.abs(dht[dht.length - len + i] - therm[therm.length - len + i]));
            const meanDiff = diffs.reduce((a, b) => a + b, 0) / diffs.length;
            const maxDiff = Math.max(...diffs);
            setText('modal-cv-mean', `${meanDiff.toFixed(2)} °C`);
            setText('modal-cv-max', `${maxDiff.toFixed(2)} °C`);
        }
    } else {
        cvPanel.style.display = 'none';
    }

    // Build modal chart
    buildModalChart(feat, color, unit);

    // Show modal
    overlay.classList.add('active');
}

function buildModalChart(feat, color, unit) {
    const canvas = document.getElementById('modal-chart');
    if (APP.modalChart) { APP.modalChart.destroy(); APP.modalChart = null; }

    const ctx = canvas.getContext('2d');
    const history = (APP.sensorHistory[feat] || []).slice();
    const labels = APP.timestamps.slice(-history.length);

    APP.modalChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels,
            datasets: [{
                label: feat,
                data: history,
                borderColor: color,
                backgroundColor: hexToRgba(color, 0.08),
                borderWidth: 2, fill: true, tension: 0.3, pointRadius: 0,
            }]
        },
        options: {
            responsive: true, maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: { backgroundColor: 'rgba(10,14,26,0.9)', titleColor: '#f1f5f9', bodyColor: '#94a3b8', borderColor: 'rgba(255,255,255,0.1)', borderWidth: 1, cornerRadius: 8 }
            },
            scales: {
                x: { display: true, ticks: { color: '#64748b', font: { size: 10 }, maxTicksLimit: 10 }, grid: { color: 'rgba(255,255,255,0.03)' } },
                y: { display: true, title: { display: true, text: unit, color: '#64748b' }, ticks: { color: '#64748b' }, grid: { color: 'rgba(255,255,255,0.04)' } }
            },
            animation: { duration: 400 },
        }
    });
}

function closeLightbox() {
    const overlay = document.getElementById('modal-overlay');
    if (overlay) overlay.classList.remove('active');
    if (APP.modalChart) { APP.modalChart.destroy(); APP.modalChart = null; }
}

// ============================================================================
// EVENT LISTENERS
// ============================================================================

function setupEventListeners() {
    // Data source tabs
    document.querySelectorAll('.source-tab').forEach(tab => {
        tab.addEventListener('click', () => {
            document.querySelectorAll('.source-tab').forEach(t => t.classList.remove('active'));
            tab.classList.add('active');
        });
    });

    // MQTT
    document.getElementById('btn-mqtt-connect')?.addEventListener('click', () => sendAction('mqtt_connect'));
    document.getElementById('btn-mqtt-disconnect')?.addEventListener('click', () => sendAction('mqtt_disconnect'));

    // Simulation
    document.getElementById('btn-sim-start')?.addEventListener('click', () => sendAction('sim_start'));
    document.getElementById('btn-sim-stop')?.addEventListener('click', () => sendAction('sim_stop'));

    // Actuator commands
    document.querySelectorAll('[data-cmd]').forEach(btn => {
        btn.addEventListener('click', () => sendAction('command', { cmd: btn.dataset.cmd }));
    });

    // Clear data
    document.getElementById('btn-clear')?.addEventListener('click', () => {
        sendAction('clear');
        clearAllLocalState();
    });

    // Modal close
    document.getElementById('modal-close')?.addEventListener('click', closeLightbox);
    document.getElementById('modal-overlay')?.addEventListener('click', (e) => {
        if (e.target.id === 'modal-overlay') closeLightbox();
    });
    document.addEventListener('keydown', (e) => { if (e.key === 'Escape') closeLightbox(); });

    // Status poll
    setInterval(() => {
        if (APP.ws && APP.ws.readyState === WebSocket.OPEN) sendAction('get_status');
    }, 5000);
}

function sendAction(action, extra = {}) {
    if (APP.ws && APP.ws.readyState === WebSocket.OPEN)
        APP.ws.send(JSON.stringify({ action, ...extra }));
}

function handleStatusUpdate(msg) {
    if (msg.action === 'mqtt_connect') updateSourceStatus(msg.connected ? 'badge-green' : 'badge-yellow', msg.connected ? 'MQTT Connected' : 'Connecting...');
    else if (msg.action === 'mqtt_disconnect') updateSourceStatus('badge-red', 'Disconnected');
    else if (msg.action === 'sim_start') updateSourceStatus('badge-blue', 'Simulating');
    else if (msg.action === 'sim_stop') updateSourceStatus('badge-red', 'Stopped');
}

function updateServerStatus(msg) {
    const el = document.getElementById('source-status');
    if (!el) return;
    if (msg.mqtt_connected) { el.className = 'badge badge-green'; el.innerHTML = '<span class="dot"></span> MQTT Live'; }
    else if (msg.sim_mode) { el.className = 'badge badge-blue'; el.innerHTML = '<span class="dot"></span> Simulating'; }
    else if (msg.streaming) { el.className = 'badge badge-yellow'; el.innerHTML = '<span class="dot"></span> Connecting...'; }
    else { el.className = 'badge badge-red'; el.innerHTML = '<span class="dot"></span> Disconnected'; }
    setText('metric-samples', msg.sample_count);
    const modelBadge = document.getElementById('model-badge');
    if (modelBadge) {
        if (msg.lstm_active) { modelBadge.className = 'badge badge-green'; modelBadge.innerHTML = '🧠 LSTM Active'; }
        else { modelBadge.className = 'badge badge-yellow'; modelBadge.innerHTML = '📊 Statistical'; }
    }
}

function updateConnectionBadge() {
    const el = document.getElementById('ws-status');
    if (!el) return;
    if (APP.connected) { el.className = 'badge badge-green'; el.innerHTML = '<span class="dot"></span> WebSocket OK'; }
    else { el.className = 'badge badge-red'; el.innerHTML = '<span class="dot"></span> WebSocket Down'; }
}

function updateSourceStatus(cls, txt) {
    const el = document.getElementById('source-status');
    if (!el) return;
    el.className = `badge ${cls}`;
    el.innerHTML = `<span class="dot"></span> ${txt}`;
}

// ============================================================================
// CLEAR ALL STATE
// ============================================================================

function clearAllLocalState() {
    // 1. Clear data arrays
    APP.timestamps = [];
    APP.anomalyHistory = [];
    APP.healthHistory = [];
    APP.lastData = null;
    APP.sampleCount = 0;
    APP.currentHealth = 100;
    APP.currentAlert = 'Normal';
    APP.currentFault = 'Healthy';
    if (APP.config) {
        APP.config.feature_names.forEach(f => APP.sensorHistory[f] = []);
    }

    // 2. Reset anomaly chart
    if (APP.anomalyChart) {
        APP.anomalyChart.data.labels = [];
        APP.anomalyChart.data.datasets[0].data = [];
        APP.anomalyChart.data.datasets[1].data = [];
        APP.anomalyChart.update();
    }

    // 3. Reset health chart
    if (APP.healthChart) {
        APP.healthChart.data.labels = [];
        APP.healthChart.data.datasets[0].data = [];
        APP.healthChart.update();
    }

    // 4. Reset health gauge to 100%
    updateHealthGauge(100);

    // 5. Reset alert banner
    const banner = document.getElementById('alert-banner');
    if (banner) {
        banner.className = 'alert-banner alert-normal';
        banner.innerHTML = '✅ <span>Data cleared — connect MQTT or start simulation</span>';
    }

    // 6. Reset metrics
    setText('metric-samples', '0');
    setText('metric-anomaly', '0');
    setText('metric-uptime', '100%');
    setText('metric-fault', 'Healthy');

    // 7. Reset fault bars
    const faultBars = document.getElementById('fault-bars');
    if (faultBars) {
        faultBars.innerHTML = '<p class="info-text" style="padding:20px;text-align:center">Waiting for sensor data...</p>';
    }

    // 8. Reset sensor cards — rebuild them so sparklines clear
    rebuildSensorCards();

    // 9. Close modal if open
    closeLightbox();
}

// ============================================================================
// UTILITIES
// ============================================================================

function setText(id, value) {
    const el = document.getElementById(id);
    if (el) el.textContent = value;
}

function hexToRgba(hex, alpha) {
    if (!hex || hex.charAt(0) !== '#') return `rgba(100,100,241,${alpha})`;
    const r = parseInt(hex.slice(1, 3), 16);
    const g = parseInt(hex.slice(3, 5), 16);
    const b = parseInt(hex.slice(5, 7), 16);
    return `rgba(${r},${g},${b},${alpha})`;
}
