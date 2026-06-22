// Production Charts page: daily candlestick + volume via the vendored
// Lightweight Charts library, fed by /api/chart/<symbol>. Live last-price and
// position overlays land in a follow-up; this is the read-only chart shell.

let chart = null;
let candleSeries = null;
let volumeSeries = null;

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
    if (!payload) { status.textContent = ''; return; }  // fetchJSON already toasted
    if (!payload.candles || payload.candles.length === 0) {
        candleSeries.setData([]);
        volumeSeries.setData([]);
        status.textContent = `No data for ${symbol}.`;
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
