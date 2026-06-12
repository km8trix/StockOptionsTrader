// Analysis page: candlestick + volume + MACD + RSI panels (shared x-axis),
// SMA/Bollinger overlays, optional strategy BUY/SELL markers, provenance.

'use strict';

const SYMBOL_RE = /^[A-Za-z][A-Za-z.\-]{0,9}$/;

/* ==========================================================================
   Theme: pull the terminal palette out of CSS custom properties
   ========================================================================== */

function themeVar(name, fallback) {
    const v = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
    return v || fallback;
}

function plotlyTheme() {
    return {
        text: themeVar('--text', '#e6edf3'),
        muted: themeVar('--text-muted', '#8b949e'),
        grid: themeVar('--border', '#2d333b'),
        accent: themeVar('--accent', '#4493f8'),
        gain: themeVar('--gain', '#3fb950'),
        loss: themeVar('--loss', '#f85149'),
    };
}

/* ==========================================================================
   Form: populate strategy selector, validate inline, submit
   ========================================================================== */

async function loadStrategies() {
    const select = document.getElementById('strategySelect');
    try {
        const data = await fetchJSON('/api/strategies', { silent: true });
        (data.strategies || []).forEach((s) => {
            const opt = document.createElement('option');
            opt.value = s.id;
            opt.textContent = s.name;
            select.appendChild(opt);
        });
    } catch (_) {
        showToast('warning', 'Could not load strategy list — signal overlay unavailable');
    }
}

function validateForm() {
    const symbolEl = document.getElementById('analysisSymbol');
    const startEl = document.getElementById('startDate');
    const endEl = document.getElementById('endDate');
    let ok = true;

    const symbolOk = SYMBOL_RE.test(symbolEl.value.trim());
    symbolEl.classList.toggle('is-invalid', !symbolOk);
    ok = ok && symbolOk;

    const endOk = Boolean(endEl.value);
    endEl.classList.toggle('is-invalid', !endOk);
    ok = ok && endOk;

    const startOk = Boolean(startEl.value) &&
        (!endEl.value || startEl.value < endEl.value);
    startEl.classList.toggle('is-invalid', !startOk);
    ok = ok && startOk;

    return ok;
}

async function onAnalyze(event) {
    event.preventDefault();
    if (!validateForm()) return;

    const symbol = document.getElementById('analysisSymbol').value.trim().toUpperCase();
    const start = document.getElementById('startDate').value;
    const end = document.getElementById('endDate').value;
    const strategy = document.getElementById('strategySelect').value;

    const restoreBtn = btnLoading(document.getElementById('analyzeBtn'));
    const removeOverlay = showLoading(document.getElementById('chartPanel'));
    try {
        const params = new URLSearchParams({ start, end });
        if (strategy) params.set('strategy', strategy);
        const payload = await fetchJSON(`/api/analyze/${encodeURIComponent(symbol)}?${params}`);
        renderSnapshot(payload);
        renderChart(payload);
        renderProvenance(payload);
    } catch (_) {
        // fetchJSON already toasted; leave the previous chart in place.
    } finally {
        removeOverlay();
        restoreBtn();
    }
}

/* ==========================================================================
   Snapshot metric cards
   ========================================================================== */

function renderSnapshot(payload) {
    document.getElementById('snapshotRow').classList.remove('d-none');
    document.getElementById('snapPrice').textContent = fmtMoney(payload.current_price);
    document.getElementById('snapRsi').textContent = fmtNum(payload.current_rsi, 1);
    document.getElementById('snapMacd').textContent = fmtNum(payload.current_macd, 3);
    document.getElementById('snapSma20').textContent = fmtMoney(payload.current_sma_20);
    document.getElementById('snapSma50').textContent = fmtMoney(payload.current_sma_50);
}

/* ==========================================================================
   Provenance caption + provider-failure toast
   ========================================================================== */

function renderProvenance(payload) {
    const caption = document.getElementById('provenanceCaption');
    const ds = payload.data_source;
    if (!ds) {
        caption.textContent = '';
        return;
    }
    const served = ds.from_cache ? 'cache' : (ds.provider || 'cache');
    caption.textContent = `served by ${served}`;

    const failures = ds.failures || [];
    if (failures.length > 0) {
        const detail = failures
            .map((f) => `${f.provider}: ${f.error}`)
            .join('; ');
        showToast('warning', `Provider failures — ${detail}`);
    }
}

/* ==========================================================================
   Chart: 4 stacked panels on one shared x-axis
   ========================================================================== */

function col(rows, dates, key) {
    return dates.map((d) => rows[d][key] ?? null);
}

function renderChart(payload) {
    const rows = payload.data || {};
    const dates = Object.keys(rows).sort();
    if (dates.length === 0) return;

    const t = plotlyTheme();
    const open = col(rows, dates, 'open');
    const high = col(rows, dates, 'high');
    const low = col(rows, dates, 'low');
    const close = col(rows, dates, 'close');
    const volume = col(rows, dates, 'volume');

    const traces = [
        // --- Panel 1: price -------------------------------------------------
        {
            type: 'candlestick', name: payload.symbol,
            x: dates, open, high, low, close,
            increasing: { line: { color: t.gain, width: 1 }, fillcolor: t.gain },
            decreasing: { line: { color: t.loss, width: 1 }, fillcolor: t.loss },
            yaxis: 'y', showlegend: false,
        },
        // Bollinger band fill: upper first, lower fills to it.
        {
            type: 'scatter', name: 'BB upper', x: dates,
            y: col(rows, dates, 'bb_upper'),
            line: { width: 0 }, hoverinfo: 'skip',
            yaxis: 'y', showlegend: false,
        },
        {
            type: 'scatter', name: 'Bollinger 20·2σ', x: dates,
            y: col(rows, dates, 'bb_lower'),
            line: { width: 0 }, fill: 'tonexty',
            fillcolor: 'rgba(68, 147, 248, 0.08)',
            hoverinfo: 'skip', yaxis: 'y',
        },
        {
            type: 'scatter', name: 'SMA 20', x: dates,
            y: col(rows, dates, 'sma_20'),
            line: { color: t.accent, width: 1.25 }, yaxis: 'y',
        },
        {
            type: 'scatter', name: 'SMA 50', x: dates,
            y: col(rows, dates, 'sma_50'),
            line: { color: t.muted, width: 1.25, dash: 'dot' }, yaxis: 'y',
        },
        // --- Panel 2: volume ------------------------------------------------
        {
            type: 'bar', name: 'Volume', x: dates, y: volume,
            marker: {
                color: close.map((c, i) =>
                    c !== null && open[i] !== null && c < open[i]
                        ? 'rgba(248, 81, 73, 0.45)' : 'rgba(63, 185, 80, 0.45)'),
            },
            yaxis: 'y2', showlegend: false,
        },
        // --- Panel 3: MACD ----------------------------------------------
        {
            type: 'bar', name: 'MACD hist', x: dates,
            y: col(rows, dates, 'macd_hist'),
            marker: {
                color: col(rows, dates, 'macd_hist').map((v) =>
                    v !== null && v < 0
                        ? 'rgba(248, 81, 73, 0.45)' : 'rgba(63, 185, 80, 0.45)'),
            },
            yaxis: 'y3', showlegend: false,
        },
        {
            type: 'scatter', name: 'MACD', x: dates,
            y: col(rows, dates, 'macd'),
            line: { color: t.accent, width: 1.25 }, yaxis: 'y3',
        },
        {
            type: 'scatter', name: 'Signal', x: dates,
            y: col(rows, dates, 'signal'),
            line: { color: t.muted, width: 1.25 }, yaxis: 'y3',
        },
        // --- Panel 4: RSI -----------------------------------------------
        {
            type: 'scatter', name: 'RSI 14', x: dates,
            y: col(rows, dates, 'rsi'),
            line: { color: t.accent, width: 1.25 },
            yaxis: 'y4', showlegend: false,
        },
    ];

    // Strategy BUY/SELL markers on the price panel.
    const signals = payload.signals || [];
    const buys = signals.filter((s) => s.signal === 'BUY');
    const sells = signals.filter((s) => s.signal === 'SELL');
    if (buys.length > 0) {
        traces.push({
            type: 'scatter', mode: 'markers', name: 'BUY',
            x: buys.map((s) => s.date), y: buys.map((s) => s.price),
            marker: { symbol: 'triangle-up', size: 11, color: t.gain,
                      line: { color: '#0d1117', width: 1 } },
            yaxis: 'y',
        });
    }
    if (sells.length > 0) {
        traces.push({
            type: 'scatter', mode: 'markers', name: 'SELL',
            x: sells.map((s) => s.date), y: sells.map((s) => s.price),
            marker: { symbol: 'triangle-down', size: 11, color: t.loss,
                      line: { color: '#0d1117', width: 1 } },
            yaxis: 'y',
        });
    }

    const axisDefaults = {
        gridcolor: t.grid, zerolinecolor: t.grid,
        tickfont: { size: 10, color: t.muted },
        fixedrange: false,
    };

    const layout = {
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        font: { family: 'ui-monospace, SFMono-Regular, Menlo, monospace',
                size: 11, color: t.text },
        margin: { l: 56, r: 16, t: 8, b: 28 },
        height: 680,
        showlegend: true,
        legend: { orientation: 'h', y: 1.02, x: 0,
                  font: { size: 10, color: t.muted }, bgcolor: 'rgba(0,0,0,0)' },
        dragmode: 'pan',
        hovermode: 'x unified',
        hoverlabel: { bgcolor: '#1c232d', bordercolor: t.grid,
                      font: { size: 11, color: t.text } },
        bargap: 0.25,
        xaxis: {
            ...axisDefaults, rangeslider: { visible: false },
            type: 'category', tickmode: 'auto', nticks: 10, anchor: 'y4',
        },
        yaxis:  { ...axisDefaults, domain: [0.44, 1.00], title: { text: 'Price', font: { size: 10, color: t.muted } } },
        yaxis2: { ...axisDefaults, domain: [0.31, 0.42], title: { text: 'Vol', font: { size: 10, color: t.muted } } },
        yaxis3: { ...axisDefaults, domain: [0.16, 0.29], title: { text: 'MACD', font: { size: 10, color: t.muted } } },
        yaxis4: { ...axisDefaults, domain: [0.00, 0.14], range: [0, 100], title: { text: 'RSI', font: { size: 10, color: t.muted } } },
        shapes: [
            // RSI guide lines at 30 / 70.
            { type: 'line', xref: 'paper', x0: 0, x1: 1, yref: 'y4', y0: 70, y1: 70,
              line: { color: t.loss, width: 1, dash: 'dot' }, opacity: 0.5 },
            { type: 'line', xref: 'paper', x0: 0, x1: 1, yref: 'y4', y0: 30, y1: 30,
              line: { color: t.gain, width: 1, dash: 'dot' }, opacity: 0.5 },
        ],
    };

    const chartEl = document.getElementById('analysisChart');
    document.getElementById('chartEmpty').innerHTML = '';
    chartEl.classList.remove('d-none');
    document.getElementById('chartTitle').textContent =
        `${payload.symbol}${payload.strategy ? ` · ${payload.strategy} signals` : ''}`;

    Plotly.react(chartEl, traces, layout, {
        responsive: true, displaylogo: false,
        modeBarButtonsToRemove: ['lasso2d', 'select2d', 'autoScale2d'],
    });

    if (payload.strategy && Array.isArray(payload.signals) && signals.length === 0) {
        showToast('info', `No ${payload.strategy} signals in the charted window`);
    } else if (payload.strategy && payload.signals === null) {
        showToast('warning', 'Signal overlay failed — see server log');
    }
}

/* ==========================================================================
   Init
   ========================================================================== */

document.addEventListener('DOMContentLoaded', () => {
    document.getElementById('chartEmpty').innerHTML = emptyStateHTML(
        'bi-graph-up', 'No chart yet',
        'Pick a symbol and date range, then Analyze.');
    loadStrategies();
    document.getElementById('analysisForm').addEventListener('submit', onAnalyze);
});
