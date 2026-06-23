// Production Charts page: daily candlestick + volume via the vendored
// Lightweight Charts library, fed by /api/chart/<symbol>, with a polled live
// last-price line and your entry-price overlay for the configured live account.

let chart = null;
let candleSeries = null;
let volumeSeries = null;
let livePriceLine = null;     // current last-price marker on the candle series
let liveTimer = null;         // setInterval handle for the quote poll
let positionLines = [];       // entry-price overlay line(s) for this symbol
const LIVE_POLL_MS = 10000;   // poll the gated /api/live/quotes every 10s

function initChart() {
    const el = document.getElementById('chartContainer');
    chart = LightweightCharts.createChart(el, {
        autoSize: true,
        layout: { background: { color: 'transparent' }, textColor: '#8b949e' },
        grid: {
            vertLines: { color: 'rgba(255,255,255,0.05)' },
            horzLines: { color: 'rgba(255,255,255,0.05)' },
        },
        rightPriceScale: { borderColor: 'rgba(255,255,255,0.1)' },
        timeScale: { borderColor: 'rgba(255,255,255,0.1)' },
        crosshair: { mode: LightweightCharts.CrosshairMode.Normal },
    });
    candleSeries = chart.addCandlestickSeries({
        upColor: '#26a69a', downColor: '#ef5350', borderVisible: false,
        wickUpColor: '#26a69a', wickDownColor: '#ef5350',
    });
    volumeSeries = chart.addHistogramSeries({
        priceFormat: { type: 'volume' },
        priceScaleId: '',
        color: 'rgba(120,120,140,0.4)',
    });
    volumeSeries.priceScale().applyOptions({
        scaleMargins: { top: 0.8, bottom: 0 },
    });
}

async function loadChart(symbol) {
    const status = document.getElementById('chartStatus');
    status.textContent = `Loading ${symbol}…`;
    const payload = await fetchJSON(`/api/chart/${encodeURIComponent(symbol)}`);
    if (!payload) { status.textContent = ''; stopLive(); clearPositions(); return; }  // fetchJSON toasted
    if (!payload.candles || payload.candles.length === 0) {
        candleSeries.setData([]);
        volumeSeries.setData([]);
        status.textContent = `No data for ${symbol}.`;
        stopLive();
        clearPositions();
        return;
    }
    candleSeries.setData(payload.candles);
    // Volume is index-aligned with candles (same server loop), so color each
    // bar by that day's direction.
    const vol = (payload.volume || []).map((v, i) => {
        const c = payload.candles[i];
        const up = c && c.close >= c.open;
        return {
            time: v.time,
            value: v.value,
            color: up ? 'rgba(38,166,154,0.4)' : 'rgba(239,83,80,0.4)',
        };
    });
    volumeSeries.setData(vol);
    chart.timeScale().fitContent();
    const src = payload.data_source && payload.data_source.provider;
    status.textContent = `${payload.symbol} · ${payload.candles.length} daily bars`
        + (src ? ` · source: ${src}` : '');
    startLive(payload.symbol);
    loadPositions(payload.symbol);
}

function stopLive() {
    if (liveTimer !== null) { clearInterval(liveTimer); liveTimer = null; }
}

function clearPositions() {
    for (const line of positionLines) candleSeries.removePriceLine(line);
    positionLines = [];
}

// Draw your entry-price line(s) for the charted symbol from the configured
// live account (cost basis doesn't move intraday, so this is fetch-once per
// load, not polled). Fail-open: not-connected / flat / any error just leaves
// the chart clean — reload to retry, same contract as the live quote poll.
async function loadPositions(symbol) {
    clearPositions();
    const data = await fetchJSON(
        `/api/live/positions/${encodeURIComponent(symbol)}`, { silent: true });
    for (const pos of (data && data.positions) || []) {
        const price = pos.price_paid != null ? Number(pos.price_paid) : NaN;
        if (!Number.isFinite(price) || price <= 0) continue;
        const qty = pos.quantity != null ? Number(pos.quantity) : null;
        const isShort = pos.position_type === 'SHORT' || (qty != null && qty < 0);
        const qtyLabel = qty != null ? `${Math.abs(qty)} ` : '';
        positionLines.push(candleSeries.createPriceLine({
            price,
            color: isShort ? '#ef5350' : '#26a69a',
            lineWidth: 1,
            lineStyle: LightweightCharts.LineStyle.Dotted,
            axisLabelVisible: true,
            title: `${isShort ? 'short' : 'long'} ${qtyLabel}@ ${price.toFixed(2)}`,
        }));
    }
}

// Poll the EXISTING gated quote endpoint (/api/live/quotes, 409 when E*TRADE is
// not connected) and draw the last trade as a price line. Fail-open: any
// failure leaves the daily chart intact and stops the poll (reload to retry).
async function pollLive(symbol) {
    const live = document.getElementById('chartLive');
    const data = await fetchJSON(
        `/api/live/quotes?symbols=${encodeURIComponent(symbol)}`, { silent: true });
    const quote = data && data.quotes && data.quotes[symbol];
    const last = quote && quote.last != null ? Number(quote.last) : null;
    if (last === null) {
        stopLive();
        if (livePriceLine) { candleSeries.removePriceLine(livePriceLine); livePriceLine = null; }
        live.textContent = '○ live quote unavailable — showing daily close';
        return;
    }
    if (livePriceLine) candleSeries.removePriceLine(livePriceLine);
    livePriceLine = candleSeries.createPriceLine({
        price: last, color: '#4493f8', lineWidth: 1,
        lineStyle: LightweightCharts.LineStyle.Dashed,
        axisLabelVisible: true, title: 'last',
    });
    live.textContent = `● ${symbol} last $${last.toFixed(2)} · updated `
        + new Date().toLocaleTimeString();
}

function startLive(symbol) {
    stopLive();
    pollLive(symbol);
    liveTimer = setInterval(() => pollLive(symbol), LIVE_POLL_MS);
}

document.addEventListener('DOMContentLoaded', () => {
    if (typeof LightweightCharts === 'undefined') {
        document.getElementById('chartStatus').textContent =
            'Charting library failed to load.';
        return;
    }
    initChart();
    const symbolInput = document.getElementById('chartSymbol');
    document.getElementById('chartForm').addEventListener('submit', (e) => {
        e.preventDefault();
        const symbol = symbolInput.value.trim().toUpperCase();
        if (symbol) loadChart(symbol);
    });
    loadChart(symbolInput.value.trim().toUpperCase());
});
