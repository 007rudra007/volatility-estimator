/**
 * app.js - Mobile Dashboard Controller
 * ====================================
 * Connects directly to Supabase client-side, runs database queries,
 * cleans calculations data, and renders interactive charts with ApexCharts.
 */

let supabase = null;
let chartInstances = {};

// Cache DOM Elements
const alertBanner = document.getElementById('connection-alert');
const dashboardContent = document.getElementById('dashboard-content');
const selectTicker = document.getElementById('select-ticker');
const btnRefresh = document.getElementById('btn-refresh');
const btnSettings = document.getElementById('btn-settings');
const modalSettings = document.getElementById('modal-settings');
const btnCloseSettings = document.getElementById('btn-close-settings');
const btnSaveSettings = document.getElementById('btn-save-settings');
const btnClearSettings = document.getElementById('btn-clear-settings');
const inputUrl = document.getElementById('input-url');
const inputKey = document.getElementById('input-key');

// Format Helper Utilities
const formatPercent = (val) => val != null ? `${(val * 100).toFixed(1)}%` : '—';
const formatPrice = (val) => val != null ? `₹${Number(val).toLocaleString('en-IN', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}` : '—';
const formatNumber = (val) => val != null ? Number(val).toLocaleString('en-IN') : '—';
const formatDate = (dateStr) => {
    if (!dateStr) return '—';
    try {
        const d = new Date(dateStr);
        return d.toLocaleDateString('en-IN', { day: '2-digit', month: 'short', year: 'numeric' });
    } catch {
        return dateStr;
    }
};

// ==============================================================================
// 1. Connection & LocalStorage Management
// ==============================================================================

function initConnection() {
    const url = localStorage.getItem('sb_url');
    const key = localStorage.getItem('sb_key');
    const tabsNav = document.getElementById('tabs-navigation');

    if (url && key) {
        try {
            // Instantiate Supabase Client
            supabase = window.supabase.createClient(url, key);
            
            // Update UI status
            alertBanner.className = 'alert-banner success';
            alertBanner.innerHTML = '🟢 Connected to Supabase Database.';
            dashboardContent.classList.remove('hidden');
            if (tabsNav) tabsNav.classList.remove('hidden');
            
            // Populate inputs in config modal
            inputUrl.value = url;
            inputKey.value = key;
            
            // Fetch tickers
            loadTickers();
        } catch (err) {
            console.error(err);
            showConfigError('Failed to initialize Supabase. Check credentials.');
        }
    } else {
        showConfigWarning();
    }
}

function showConfigWarning() {
    const tabsNav = document.getElementById('tabs-navigation');
    alertBanner.className = 'alert-banner warning';
    alertBanner.innerHTML = '⚠️ Supabase connection not configured. Tap the ⚙️ icon to set up keys.';
    dashboardContent.classList.add('hidden');
    if (tabsNav) tabsNav.classList.add('hidden');
    selectTicker.disabled = true;
    btnRefresh.disabled = true;
}

function showConfigError(msg) {
    const tabsNav = document.getElementById('tabs-navigation');
    alertBanner.className = 'alert-banner warning';
    alertBanner.innerHTML = `❌ Connection Error: ${msg}. Tap ⚙️ to reconfigure.`;
    dashboardContent.classList.add('hidden');
    if (tabsNav) tabsNav.classList.add('hidden');
    selectTicker.disabled = true;
    btnRefresh.disabled = true;
}

// Settings Modal Events
btnSettings.addEventListener('click', () => modalSettings.classList.remove('hidden'));
btnCloseSettings.addEventListener('click', () => modalSettings.classList.add('hidden'));

btnSaveSettings.addEventListener('click', () => {
    const url = inputUrl.value.trim();
    const key = inputKey.value.trim();
    
    if (url && key) {
        localStorage.setItem('sb_url', url);
        localStorage.setItem('sb_key', key);
        modalSettings.classList.add('hidden');
        initConnection();
    } else {
        alert("Please enter both URL and Anon Key.");
    }
});

btnClearSettings.addEventListener('click', () => {
    localStorage.removeItem('sb_url');
    localStorage.removeItem('sb_key');
    inputUrl.value = '';
    inputKey.value = '';
    modalSettings.classList.add('hidden');
    initConnection();
});

// Refresh triggers
btnRefresh.addEventListener('click', () => {
    const ticker = selectTicker.value;
    if (ticker) loadTickerData(ticker);
});

selectTicker.addEventListener('change', (e) => {
    if (e.target.value) loadTickerData(e.target.value);
});


// ==============================================================================
// 2. Fetch Watchlist Tickers
// ==============================================================================

async function loadTickers() {
    try {
        selectTicker.disabled = true;
        selectTicker.innerHTML = '<option value="">Loading tickers...</option>';
        
        // Fetch last 500 rows to find unique tickers
        const { data, error } = await supabase
            .from('volatility_analysis')
            .select('ticker')
            .order('timestamp', { ascending: false })
            .limit(1000);
            
        if (error) throw error;
        
        if (!data || data.length === 0) {
            selectTicker.innerHTML = '<option value="">No data in database</option>';
            return;
        }
        
        // Find unique tickers
        const tickers = [...new Set(data.map(item => item.ticker))].sort();
        
        selectTicker.innerHTML = '';
        tickers.forEach(t => {
            const opt = document.createElement('option');
            opt.value = t;
            opt.textContent = t;
            selectTicker.appendChild(opt);
        });
        
        selectTicker.disabled = false;
        btnRefresh.disabled = false;
        
        // Auto-load first ticker
        if (tickers.length > 0) {
            selectTicker.value = tickers[0];
            loadTickerData(tickers[0]);
        }
    } catch (err) {
        console.error(err);
        showConfigError(`Database load failed: ${err.message}`);
    }
}


// ==============================================================================
// 3. Load Metrics & Charts Data
// ==============================================================================

async function loadTickerData(ticker) {
    console.log(`Loading metrics for: ${ticker}`);
    
    // Fetch Macro events in parallel since they are independent
    fetchMacroEvents();
    
    try {
        // A. Fetch latest 150 volatility metrics rows
        const { data: volData, error: volErr } = await supabase
            .from('volatility_analysis')
            .select('*')
            .eq('ticker', ticker)
            .order('timestamp', { ascending: false })
            .limit(150);
            
        if (volErr) throw volErr;
        if (!volData || volData.length === 0) {
            alert(`No historical volatility records found for ${ticker}`);
            return;
        }
        
        // Chronological order for charting
        const chronVolData = [...volData].reverse();
        
        // B. Fetch latest GEX key levels
        const { data: gexLevels, error: gexErr } = await supabase
            .from('gex_key_levels')
            .select('*')
            .eq('ticker', ticker)
            .order('timestamp', { ascending: false })
            .limit(1);
            
        if (gexErr) throw gexErr;
        const currentGex = gexLevels && gexLevels.length > 0 ? gexLevels[0] : null;
        
        // C. Fetch Strike Level GEX Profile
        let profileStrikes = [];
        // First find latest profile timestamp
        const { data: latestProfileTs, error: tsErr } = await supabase
            .from('gex_profiles')
            .select('timestamp')
            .eq('ticker', ticker)
            .order('timestamp', { ascending: false })
            .limit(1);
            
        if (!tsErr && latestProfileTs && latestProfileTs.length > 0) {
            const { data: strikes, error: strikesErr } = await supabase
                .from('gex_profiles')
                .select('*')
                .eq('ticker', ticker)
                .eq('timestamp', latestProfileTs[0].timestamp)
                .order('strike', { ascending: true });
            if (!strikesErr) profileStrikes = strikes;
        }

        // D. Fetch CVD Data
        const { data: cvdData, error: cvdErr } = await supabase
            .from('cvd_data')
            .select('*')
            .eq('ticker', ticker)
            .order('timestamp', { ascending: false })
            .limit(100);
            
        const chronCvdData = cvdData ? [...cvdData].reverse() : [];
        
        // E. Fetch Positioning Data (COT)
        const { data: cotData, error: cotErr } = await supabase
            .from('positioning_data')
            .select('*')
            .eq('ticker', ticker)
            .order('timestamp', { ascending: false })
            .limit(100);
            
        const chronCotData = cotData ? [...cotData].reverse() : [];
        
        // Update Stats UI Panel
        updateStatsPanel(chronVolData[chronVolData.length - 1], currentGex);
        
        // Render Charts
        renderPriceVolChart(chronVolData);
        renderGexProfileChart(profileStrikes, currentGex);
        renderCvdChart(chronCvdData);
        renderPositioningChart(chronCotData);
        
        // F. Fetch Intraday Predictions and Backtest Results
        fetchIntradayPredictions(ticker);
        fetchBacktestResults(ticker);
        
    } catch (err) {
        console.error(err);
        alert(`Error loading stock metrics: ${err.message}`);
    }
}


// ==============================================================================
// 4. Update Stats Dashboard Card Panel
// ==============================================================================

function updateStatsPanel(latestVol, latestGex) {
    if (!latestVol) return;
    
    // Ticker Close & change
    document.getElementById('val-spot').textContent = formatPrice(latestVol.close_price);
    const changeLabel = document.getElementById('val-change');
    changeLabel.textContent = `20d Vol: ${formatPercent(latestVol.vol_20d)}`;
    changeLabel.className = 'sub-label';

    // Volatility regime
    const valRegime = document.getElementById('val-regime');
    const regimeCard = document.getElementById('card-regime');
    const regimeStr = latestVol.regime || 'NEUTRAL';
    valRegime.textContent = regimeStr;
    document.getElementById('val-percentile').textContent = latestVol.percentile != null ? `${latestVol.percentile}th percentile` : '— percentile';
    
    // Regime colors styling classes
    regimeCard.className = 'stat-card';
    const cleanRegime = regimeStr.toLowerCase();
    if (cleanRegime.includes('expansion')) regimeCard.classList.add('regime-expansion');
    else if (cleanRegime.includes('elevated')) regimeCard.classList.add('regime-elevated');
    else if (cleanRegime.includes('neutral')) regimeCard.classList.add('regime-neutral');
    else if (cleanRegime.includes('compression') || cleanRegime.includes('low')) regimeCard.classList.add('regime-compression');

    // Gamma Flip
    if (latestGex) {
        document.getElementById('val-flip').textContent = formatPrice(latestGex.gamma_flip_price);
        const gexRegimeVal = document.getElementById('val-gex-regime');
        gexRegimeVal.textContent = latestGex.gex_regime || '—';
        if (latestGex.gex_regime && latestGex.gex_regime.includes('Positive')) {
            gexRegimeVal.style.color = '#10B981';
        } else {
            gexRegimeVal.style.color = '#EF4444';
        }
    } else {
        document.getElementById('val-flip').textContent = '—';
        document.getElementById('val-gex-regime').textContent = 'No GEX cached';
    }

    // Relative Volume (RVOL)
    const valRvol = document.getElementById('val-rvol');
    const rvolCard = document.getElementById('card-rvol');
    const rvol = latestVol.rvol;
    valRvol.textContent = rvol != null ? `${rvol.toFixed(2)}x` : '—';
    
    const volRegimeVal = document.getElementById('val-volume-regime');
    rvolCard.className = 'stat-card';
    if (rvol >= 1.5) {
        volRegimeVal.textContent = 'Volume Breakout';
        volRegimeVal.style.color = '#10B981';
    } else if (rvol <= 0.6) {
        volRegimeVal.textContent = 'Volume Compression';
        volRegimeVal.style.color = '#F59E0B';
    } else {
        volRegimeVal.textContent = 'Normal liquidity';
        volRegimeVal.style.color = '#9CA3AF';
    }
}


// ==============================================================================
// 5. Chart Render Helpers (ApexCharts)
// ==============================================================================

function cleanChartInstance(id) {
    if (chartInstances[id]) {
        chartInstances[id].destroy();
        chartInstances[id] = null;
    }
}

// Chart 1: Price and Volatility
function renderPriceVolChart(data) {
    const id = 'chart-price-vol';
    cleanChartInstance(id);
    
    const dates = data.map(d => formatDate(d.timestamp));
    const closePrices = data.map(d => d.close_price);
    const vol20 = data.map(d => d.vol_20d != null ? (d.vol_20d * 100).toFixed(2) : null);
    const ewma = data.map(d => d.ewma_vol != null ? (d.ewma_vol * 100).toFixed(2) : null);
    const sma = data.map(d => d.sma_vol != null ? (d.sma_vol * 100).toFixed(2) : null);

    const options = {
        series: [
            { name: 'Close Price', type: 'line', data: closePrices },
            { name: 'Vol 20d (Ann.)', type: 'line', data: vol20 },
            { name: 'EWMA Vol', type: 'line', data: ewma },
            { name: 'SMA Vol', type: 'line', data: sma }
        ],
        chart: {
            height: 280,
            type: 'line',
            toolbar: { show: false },
            background: 'transparent',
            foreColor: '#9CA3AF'
        },
        theme: { mode: 'dark' },
        stroke: {
            width: [2, 1.5, 1.5, 1.5],
            curve: 'smooth',
            dashArray: [0, 0, 4, 3]
        },
        colors: ['#F9FAFB', '#3B82F6', '#F59E0B', '#8B5CF6'], // white, blue, orange, purple
        xaxis: {
            categories: dates,
            tickAmount: 5,
            axisBorder: { show: false },
            axisTicks: { show: false }
        },
        yaxis: [
            {
                title: { text: 'Price (₹)', style: { color: '#F9FAFB' } },
                labels: { formatter: (v) => v ? v.toFixed(0) : '' }
            },
            {
                opposite: true,
                title: { text: 'Volatility (%)', style: { color: '#3B82F6' } },
                labels: { formatter: (v) => v ? `${v}%` : '' }
            }
        ],
        grid: { borderColor: '#374151', strokeDashArray: 3 },
        legend: { position: 'top', horizontalAlign: 'right' },
        tooltip: { theme: 'dark' }
    };

    chartInstances[id] = new ApexCharts(document.getElementById(id), options);
    chartInstances[id].render();
}

// Chart 2: Option GEX Profile
function renderGexProfileChart(strikes, gexKeyLevel) {
    const id = 'chart-gex';
    cleanChartInstance(id);
    
    if (!strikes || strikes.length === 0) {
        document.getElementById(id).innerHTML = '<div style="display:flex; align-items:center; justify-content:center; height:100%; color:var(--text-muted); font-size:13px;">No strike-level Option GEX data cached.</div>';
        return;
    }
    
    const strikePrices = strikes.map(s => s.strike);
    const netGex = strikes.map(s => s.net_gex != null ? (s.net_gex / 1e6).toFixed(2) : 0);
    const callGex = strikes.map(s => s.call_gex != null ? (s.call_gex / 1e6).toFixed(2) : 0);
    const putGex = strikes.map(s => s.put_gex != null ? (s.put_gex / 1e6).toFixed(2) : 0);

    const options = {
        series: [
            { name: 'Net GEX', type: 'line', data: netGex },
            { name: 'Call GEX', type: 'bar', data: callGex },
            { name: 'Put GEX', type: 'bar', data: putGex }
        ],
        chart: {
            height: 280,
            type: 'line',
            toolbar: { show: false },
            background: 'transparent',
            foreColor: '#9CA3AF'
        },
        theme: { mode: 'dark' },
        stroke: {
            width: [2, 0, 0],
            curve: 'smooth'
        },
        colors: ['#3B82F6', '#10B981', '#EF4444'], // Net=blue, Call=green, Put=red
        plotOptions: {
            bar: {
                borderRadius: 2,
                columnWidth: '80%'
            }
        },
        xaxis: {
            categories: strikePrices,
            tickAmount: 6,
            title: { text: 'Strike Price (₹)' },
            axisBorder: { show: false },
            axisTicks: { show: false }
        },
        yaxis: {
            title: { text: 'GEX Exposure Notional (M)' },
            labels: { formatter: (v) => `${v}M` }
        },
        grid: { borderColor: '#374151', strokeDashArray: 3 },
        legend: { position: 'top' },
        tooltip: { theme: 'dark' }
    };

    // Add vertical lines for spot and flip if present
    if (gexKeyLevel) {
        options.annotations = {
            xaxis: [
                {
                    x: gexKeyLevel.spot_price,
                    borderColor: '#F9FAFB',
                    borderWidth: 1.5,
                    strokeDashArray: 4,
                    label: {
                        borderColor: '#F9FAFB',
                        style: { color: '#090D1A', background: '#F9FAFB', fontSize: '9px', fontWeight: 600 },
                        text: `Spot: ₹${gexKeyLevel.spot_price}`
                    }
                },
                {
                    x: gexKeyLevel.gamma_flip_price,
                    borderColor: '#F59E0B',
                    borderWidth: 1.5,
                    strokeDashArray: 3,
                    label: {
                        borderColor: '#F59E0B',
                        style: { color: '#090D1A', background: '#F59E0B', fontSize: '9px', fontWeight: 600 },
                        text: `Flip: ₹${gexKeyLevel.gamma_flip_price}`
                    }
                }
            ]
        };
    }

    chartInstances[id] = new ApexCharts(document.getElementById(id), options);
    chartInstances[id].render();
}

// Chart 3: Cumulative Volume Delta (CVD)
function renderCvdChart(data) {
    const id = 'chart-cvd';
    cleanChartInstance(id);
    
    if (!data || data.length === 0) {
        document.getElementById(id).innerHTML = '<div style="display:flex; align-items:center; justify-content:center; height:100%; color:var(--text-muted); font-size:13px;">No CVD order flow cached.</div>';
        return;
    }
    
    const dates = data.map(d => formatDate(d.timestamp));
    const closePrices = data.map(d => d.close_price);
    const cvdVal = data.map(d => d.cvd);

    const options = {
        series: [
            { name: 'Close Price', type: 'line', data: closePrices },
            { name: 'CVD Index', type: 'line', data: cvdVal }
        ],
        chart: {
            height: 280,
            type: 'line',
            toolbar: { show: false },
            background: 'transparent',
            foreColor: '#9CA3AF'
        },
        theme: { mode: 'dark' },
        stroke: { width: [1.8, 2] },
        colors: ['#9CA3AF', '#8B5CF6'], // grey price, purple cvd
        xaxis: {
            categories: dates,
            tickAmount: 5,
            axisBorder: { show: false },
            axisTicks: { show: false }
        },
        yaxis: [
            {
                title: { text: 'Price (₹)', style: { color: '#9CA3AF' } },
                labels: { formatter: (v) => v ? v.toFixed(0) : '' }
            },
            {
                opposite: true,
                title: { text: 'CVD Notional', style: { color: '#8B5CF6' } },
                labels: { formatter: (v) => v ? (v / 1000).toFixed(0) + 'K' : '' }
            }
        ],
        grid: { borderColor: '#374151', strokeDashArray: 3 },
        tooltip: { theme: 'dark' }
    };

    // Find and add annotations for divergence signals
    const annotations = [];
    data.forEach((d, idx) => {
        if (d.divergence_signal === 1) { // Bullish Divergence
            annotations.push({
                x: formatDate(d.timestamp),
                y: d.close_price,
                borderColor: '#10B981',
                label: {
                    borderColor: '#10B981',
                    style: { color: '#FFF', background: '#10B981', fontSize: '9px', fontWeight: 600 },
                    text: 'BULLISH DIV'
                }
            });
        } else if (d.divergence_signal === -1) { // Bearish Divergence
            annotations.push({
                x: formatDate(d.timestamp),
                y: d.close_price,
                borderColor: '#EF4444',
                label: {
                    borderColor: '#EF4444',
                    style: { color: '#FFF', background: '#EF4444', fontSize: '9px', fontWeight: 600 },
                    text: 'BEARISH DIV'
                }
            });
        }
    });
    
    if (annotations.length > 0) {
        options.annotations = { points: annotations };
    }

    chartInstances[id] = new ApexCharts(document.getElementById(id), options);
    chartInstances[id].render();
}

// Chart 4: Institutional positioning (COT)
function renderPositioningChart(data) {
    const id = 'chart-positioning';
    cleanChartInstance(id);
    
    if (!data || data.length === 0) {
        document.getElementById(id).innerHTML = '<div style="display:flex; align-items:center; justify-content:center; height:100%; color:var(--text-muted); font-size:13px;">No futures positioning indices cached.</div>';
        return;
    }
    
    const dates = data.map(d => formatDate(d.timestamp));
    const speculator = data.map(d => d.speculator_net);
    const commercial = data.map(d => d.commercial_net);

    const options = {
        series: [
            { name: 'Speculators / FII Net', data: speculator },
            { name: 'Commercials / DII Net', data: commercial }
        ],
        chart: {
            height: 250,
            type: 'line',
            toolbar: { show: false },
            background: 'transparent',
            foreColor: '#9CA3AF'
        },
        theme: { mode: 'dark' },
        stroke: {
            width: [2, 1.5],
            curve: 'smooth',
            dashArray: [0, 4]
        },
        colors: ['#3B82F6', '#EF4444'], // Speculator=blue, Commercial=red
        xaxis: {
            categories: dates,
            tickAmount: 5,
            axisBorder: { show: false },
            axisTicks: { show: false }
        },
        yaxis: {
            title: { text: 'Net Position Contracts' },
            labels: { formatter: (v) => v ? (v / 1000).toFixed(0) + 'K' : '0' }
        },
        grid: { borderColor: '#374151', strokeDashArray: 3 },
        legend: { position: 'top' },
        tooltip: { theme: 'dark' }
    };

    chartInstances[id] = new ApexCharts(document.getElementById(id), options);
    chartInstances[id].render();
}


// ==============================================================================
// 6. Fetch & Render Macro Events Table
// ==============================================================================

async function fetchMacroEvents() {
    try {
        const tbody = document.querySelector('#table-events tbody');
        
        const { data: events, error } = await supabase
            .from('macro_events')
            .select('*')
            .order('date', { ascending: false })
            .limit(10);
            
        if (error) throw error;
        
        if (!events || events.length === 0) {
            tbody.innerHTML = '<tr><td colspan="3" class="text-center">No macro events found in database</td></tr>';
            return;
        }
        
        tbody.innerHTML = '';
        events.forEach(evt => {
            const tr = document.createElement('tr');
            tr.innerHTML = `
                <td><strong>${formatDate(evt.date)}</strong></td>
                <td><span class="badge">${evt.event_type}</span></td>
                <td>${evt.outcome || '—'}</td>
            `;
            tbody.appendChild(tr);
        });
    } catch (err) {
        console.error("Macro events fetch error:", err);
        const tbody = document.querySelector('#table-events tbody');
        tbody.innerHTML = '<tr><td colspan="3" class="text-center">Error reading macro events from Supabase</td></tr>';
    }
}


// ==============================================================================
// 7. Tab Toggling UI Logic
// ==============================================================================

const tabButtons = document.querySelectorAll('.tab-btn');
const tabPanes = document.querySelectorAll('.tab-pane');

tabButtons.forEach(btn => {
    btn.addEventListener('click', () => {
        const targetTab = btn.getAttribute('data-tab');
        
        tabButtons.forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        
        tabPanes.forEach(pane => {
            if (pane.id === targetTab) {
                pane.classList.remove('hidden-pane');
                pane.classList.add('active-pane');
            } else {
                pane.classList.remove('active-pane');
                pane.classList.add('hidden-pane');
            }
        });
        
        // Resize ApexCharts to recalculate coordinates when unhidden
        setTimeout(() => {
            window.dispatchEvent(new Event('resize'));
        }, 50);
    });
});


// ==============================================================================
// 8. Next-Day 15m Price Predictor Logic
// ==============================================================================

let currentPredictionsData = [];

async function fetchIntradayPredictions(ticker) {
    console.log(`Loading intraday predictions for: ${ticker}`);
    const chartId = 'chart-price-predictor';
    cleanChartInstance(chartId);
    
    try {
        // Find latest target_date
        const { data: latestDateRes, error: dateErr } = await supabase
            .from('intraday_predictions')
            .select('target_date')
            .eq('ticker', ticker)
            .order('target_date', { ascending: false })
            .limit(1);
            
        if (dateErr) throw dateErr;
        
        if (!latestDateRes || latestDateRes.length === 0) {
            document.getElementById(chartId).innerHTML = '<div style="display:flex; align-items:center; justify-content:center; height:100%; color:var(--text-muted); font-size:13px;">No intraday predictions found in database. Run pipeline to seed.</div>';
            return;
        }
        
        const targetDate = latestDateRes[0].target_date;
        
        // Fetch all 15m intervals for this date
        const { data: preds, error: fetchErr } = await supabase
            .from('intraday_predictions')
            .select('*')
            .eq('ticker', ticker)
            .eq('target_date', targetDate)
            .order('candle_time', { ascending: true });
            
        if (fetchErr) throw fetchErr;
        
        currentPredictionsData = preds;
        
        // Render predictor chart
        renderPredictorChart(preds, targetDate);
        
        // Configure slider range
        const slider = document.getElementById('slider-candle');
        slider.min = 0;
        slider.max = preds.length - 1;
        slider.value = 0;
        
        // Initialize slider labels
        document.getElementById('label-slider-start').textContent = preds[0].candle_time;
        document.getElementById('label-slider-end').textContent = preds[preds.length - 1].candle_time;
        
        updateSliderDetails(0);
        
    } catch (err) {
        console.error("Error fetching predictions:", err);
        document.getElementById(chartId).innerHTML = `<div style="display:flex; align-items:center; justify-content:center; height:100%; color:var(--text-muted); font-size:13px;">Error loading predictions: ${err.message}</div>`;
    }
}

function renderPredictorChart(preds, targetDate) {
    const id = 'chart-price-predictor';
    cleanChartInstance(id);
    
    const times = preds.map(p => p.candle_time);
    const predicted = preds.map(p => p.pred_price);
    const lowBound = preds.map(p => p.low_bound);
    const highBound = preds.map(p => p.high_bound);
    const actuals = preds.map(p => p.actual_price);
    
    const hasActuals = actuals.some(a => a != null);
    
    const series = [
        { name: 'Predicted Price', type: 'line', data: predicted },
        { name: 'Upper Band (90%)', type: 'line', data: highBound },
        { name: 'Lower Band (10%)', type: 'line', data: lowBound }
    ];
    
    if (hasActuals) {
        series.push({ name: 'Actual Price', type: 'line', data: actuals });
    }
    
    const formattedDate = formatDate(targetDate);
    
    const options = {
        series: series,
        chart: {
            height: 280,
            type: 'line',
            toolbar: { show: false },
            background: 'transparent',
            foreColor: '#9CA3AF'
        },
        title: {
            text: `Forecast Path for ${formattedDate}`,
            align: 'left',
            style: { fontSize: '12px', color: '#9CA3AF', fontFamily: 'var(--font-mono)' }
        },
        theme: { mode: 'dark' },
        stroke: {
            width: hasActuals ? [2.5, 1, 1, 2.5] : [2.5, 1, 1],
            curve: 'smooth',
            dashArray: hasActuals ? [0, 4, 4, 0] : [0, 4, 4]
        },
        colors: hasActuals ? ['#3B82F6', '#10B981', '#EF4444', '#F9FAFB'] : ['#3B82F6', '#10B981', '#EF4444'],
        xaxis: {
            categories: times,
            tickAmount: 5,
            axisBorder: { show: false },
            axisTicks: { show: false }
        },
        yaxis: {
            title: { text: 'Price (₹)', style: { color: '#F9FAFB' } },
            labels: { formatter: (v) => v ? v.toFixed(0) : '' }
        },
        grid: { borderColor: '#374151', strokeDashArray: 3 },
        legend: { position: 'top', horizontalAlign: 'right' },
        tooltip: { theme: 'dark' }
    };
    
    chartInstances[id] = new ApexCharts(document.getElementById(id), options);
    chartInstances[id].render();
}

function updateSliderDetails(index) {
    if (!currentPredictionsData || currentPredictionsData.length === 0 || index >= currentPredictionsData.length) {
        return;
    }
    
    const pred = currentPredictionsData[index];
    
    document.getElementById('label-slider-current').textContent = pred.candle_time;
    document.getElementById('pred-val-price').textContent = formatPrice(pred.pred_price);
    document.getElementById('pred-val-band').textContent = `${formatPrice(pred.low_bound)} - ${formatPrice(pred.high_bound)}`;
    
    const spotLabel = document.getElementById('val-spot');
    const spotValStr = spotLabel ? spotLabel.textContent.replace('₹', '').replace(/,/g, '') : null;
    const spotVal = spotValStr ? parseFloat(spotValStr) : null;
    
    if (spotVal && !isNaN(spotVal)) {
        const pctChg = ((pred.pred_price - spotVal) / spotVal) * 100;
        const sign = pctChg >= 0 ? '+' : '';
        const chgLabel = document.getElementById('pred-val-change');
        chgLabel.textContent = `${sign}${pctChg.toFixed(2)}%`;
        chgLabel.style.color = pctChg >= 0 ? '#10B981' : '#EF4444';
    } else {
        document.getElementById('pred-val-change').textContent = '—';
        document.getElementById('pred-val-change').style.color = 'var(--text)';
    }
}

document.getElementById('slider-candle').addEventListener('input', (e) => {
    updateSliderDetails(parseInt(e.target.value));
});


// ==============================================================================
// 9. Backtest Workspace UI Logic
// ==============================================================================

async function fetchBacktestResults(ticker) {
    console.log(`Loading backtesting results for: ${ticker}`);
    const chartId = 'chart-backtest-equity';
    cleanChartInstance(chartId);
    
    try {
        const { data: res, error } = await supabase
            .from('backtest_results')
            .select('*')
            .eq('ticker', ticker)
            .order('timestamp', { ascending: false })
            .limit(1);
            
        if (error) throw error;
        
        if (!res || res.length === 0) {
            document.getElementById(chartId).innerHTML = '<div style="display:flex; align-items:center; justify-content:center; height:100%; color:var(--text-muted); font-size:13px;">No backtest simulation found for this ticker. Run pipeline to seed.</div>';
            clearBacktestUI();
            return;
        }
        
        const bt = res[0];
        
        document.getElementById('bt-val-return').textContent = formatPercent(bt.cumulative_return);
        document.getElementById('bt-val-benchmark').textContent = formatPercent(bt.benchmark_return);
        
        const returnDiff = (bt.cumulative_return || 0) - (bt.benchmark_return || 0);
        const diffLabel = document.getElementById('bt-val-diff');
        diffLabel.textContent = `${returnDiff >= 0 ? '+' : ''}${formatPercent(returnDiff)} vs Asset`;
        diffLabel.style.color = returnDiff >= 0 ? '#10B981' : '#EF4444';
        
        document.getElementById('bt-val-sharpe').textContent = bt.sharpe_ratio != null ? bt.sharpe_ratio.toFixed(2) : '—';
        document.getElementById('bt-val-drawdown').textContent = formatPercent(bt.max_drawdown);
        document.getElementById('bt-val-winrate').textContent = formatPercent(bt.win_rate);
        document.getElementById('bt-val-trades').textContent = `${bt.trade_count || 0} trades`;
        
        let equityCurve = typeof bt.equity_curve === 'string' ? JSON.parse(bt.equity_curve) : bt.equity_curve;
        let tradeLogs = typeof bt.trade_logs === 'string' ? JSON.parse(bt.trade_logs) : bt.trade_logs;
        let cvdReturns = typeof bt.cvd_forward_returns === 'string' ? JSON.parse(bt.cvd_forward_returns) : bt.cvd_forward_returns;
        
        renderEquityChart(equityCurve);
        renderCvdTable(cvdReturns);
        renderTradeLogsTable(tradeLogs);
        
    } catch (err) {
        console.error("Error loading backtest:", err);
        document.getElementById(chartId).innerHTML = `<div style="display:flex; align-items:center; justify-content:center; height:100%; color:var(--text-muted); font-size:13px;">Error loading backtest: ${err.message}</div>`;
        clearBacktestUI();
    }
}

function clearBacktestUI() {
    document.getElementById('bt-val-return').textContent = '—';
    document.getElementById('bt-val-benchmark').textContent = '—';
    document.getElementById('bt-val-diff').textContent = '— vs Benchmark';
    document.getElementById('bt-val-diff').style.color = 'var(--text-muted)';
    document.getElementById('bt-val-sharpe').textContent = '—';
    document.getElementById('bt-val-drawdown').textContent = '—';
    document.getElementById('bt-val-winrate').textContent = '—';
    document.getElementById('bt-val-trades').textContent = '0 trades';
    
    document.querySelector('#table-bt-cvd tbody').innerHTML = '<tr><td colspan="5" class="text-center">No backtest data loaded</td></tr>';
    document.querySelector('#table-bt-trades tbody').innerHTML = '<tr><td colspan="6" class="text-center">No trades simulated</td></tr>';
}

function renderEquityChart(equityCurve) {
    const id = 'chart-backtest-equity';
    cleanChartInstance(id);
    
    if (!equityCurve || equityCurve.length === 0) {
        document.getElementById(id).innerHTML = '<div style="display:flex; align-items:center; justify-content:center; height:100%; color:var(--text-muted); font-size:13px;">No equity curve series data.</div>';
        return;
    }
    
    const chronCurve = [...equityCurve].sort((a, b) => new Date(a.date || a.Date || a.timestamp) - new Date(b.date || b.Date || b.timestamp));
    
    const dates = chronCurve.map(c => formatDate(c.date || c.Date || c.timestamp));
    const strat = chronCurve.map(c => c.strategy_return != null ? (c.strategy_return * 100).toFixed(2) : (c.Strategy_Return * 100).toFixed(2));
    const asset = chronCurve.map(c => c.benchmark_return != null ? (c.benchmark_return * 100).toFixed(2) : (c.Benchmark_Return * 100).toFixed(2));
    
    const options = {
        series: [
            { name: 'Quant Strategy', data: strat },
            { name: 'Benchmark Buy & Hold', data: asset }
        ],
        chart: {
            height: 280,
            type: 'line',
            toolbar: { show: false },
            background: 'transparent',
            foreColor: '#9CA3AF'
        },
        theme: { mode: 'dark' },
        stroke: { width: [2.5, 1.5], curve: 'smooth' },
        colors: ['#3B82F6', '#EF4444'],
        xaxis: {
            categories: dates,
            tickAmount: 5,
            axisBorder: { show: false },
            axisTicks: { show: false }
        },
        yaxis: {
            title: { text: 'Cumulative Return (%)', style: { color: '#F9FAFB' } },
            labels: { formatter: (v) => v ? `${v}%` : '0%' }
        },
        grid: { borderColor: '#374151', strokeDashArray: 3 },
        legend: { position: 'top' },
        tooltip: { theme: 'dark' }
    };
    
    chartInstances[id] = new ApexCharts(document.getElementById(id), options);
    chartInstances[id].render();
}

function renderCvdTable(cvdReturns) {
    const tbody = document.querySelector('#table-bt-cvd tbody');
    if (!cvdReturns || cvdReturns.length === 0) {
        tbody.innerHTML = '<tr><td colspan="5" class="text-center">No CVD stats stored</td></tr>';
        return;
    }
    
    tbody.innerHTML = '';
    cvdReturns.forEach(row => {
        const tr = document.createElement('tr');
        
        const horizon = row.horizon || row.Horizon || '—';
        const bullRet = row.bull_avg_return != null ? row.bull_avg_return : (row.Bull_Avg_Return != null ? row.Bull_Avg_Return : row.avg_return || row.Avg_Return);
        const bullWin = row.bull_win_rate != null ? row.bull_win_rate : (row.Bull_Win_Rate != null ? row.Bull_Win_Rate : row.win_rate || row.Win_Rate);
        const bearRet = row.bear_avg_return != null ? row.bear_avg_return : (row.Bear_Avg_Return != null ? row.Bear_Avg_Return : null);
        const bearWin = row.bear_win_rate != null ? row.bear_win_rate : (row.Bear_Win_Rate != null ? row.Bear_Win_Rate : null);
        
        tr.innerHTML = `
            <td><strong>${horizon}</strong></td>
            <td style="color: ${bullRet >= 0 ? '#10B981' : '#EF4444'}">${bullRet != null ? (bullRet * 100).toFixed(2) + '%' : '—'}</td>
            <td>${bullWin != null ? (bullWin * 100).toFixed(1) + '%' : '—'}</td>
            <td style="color: ${bearRet >= 0 ? '#10B981' : '#EF4444'}">${bearRet != null ? (bearRet * 100).toFixed(2) + '%' : '—'}</td>
            <td>${bearWin != null ? (bearWin * 100).toFixed(1) + '%' : '—'}</td>
        `;
        tbody.appendChild(tr);
    });
}

function renderTradeLogsTable(tradeLogs) {
    const tbody = document.querySelector('#table-bt-trades tbody');
    if (!tradeLogs || tradeLogs.length === 0) {
        tbody.innerHTML = '<tr><td colspan="6" class="text-center">No trades logged in simulation</td></tr>';
        return;
    }
    
    tbody.innerHTML = '';
    
    const chronTrades = [...tradeLogs].sort((a, b) => new Date(a.entry_date || a.Entry_Date) - new Date(b.entry_date || b.Entry_Date));
    
    chronTrades.forEach(t => {
        const tr = document.createElement('tr');
        
        const entryD = formatDate(t.entry_date || t.Entry_Date);
        const type = t.type || t.Type || 'Long';
        const entryP = t.entry_price != null ? t.entry_price : t.Entry_Price;
        const exitD = formatDate(t.exit_date || t.Exit_Date);
        const exitP = t.exit_price != null ? t.exit_price : t.Exit_Price;
        const ret = t.return != null ? t.return : (t.Return != null ? t.Return : 0);
        
        tr.innerHTML = `
            <td>${entryD}</td>
            <td><span class="badge" style="background-color: ${type.toLowerCase() === 'long' ? 'rgba(16, 185, 129, 0.1)' : 'rgba(239, 68, 68, 0.1)'}; color: ${type.toLowerCase() === 'long' ? '#10B981' : '#EF4444'}; border-color: ${type.toLowerCase() === 'long' ? 'rgba(16, 185, 129, 0.2)' : 'rgba(239, 68, 68, 0.2)'};">${type}</span></td>
            <td>₹${entryP ? entryP.toFixed(2) : '—'}</td>
            <td>${exitD}</td>
            <td>₹${exitP ? exitP.toFixed(2) : '—'}</td>
            <td style="color: ${ret >= 0 ? '#10B981' : '#EF4444'}; font-weight: 600;">${ret >= 0 ? '+' : ''}${(ret * 100).toFixed(2)}%</td>
        `;
        tbody.appendChild(tr);
    });
}


// Initialize application
document.addEventListener('DOMContentLoaded', initConnection);
