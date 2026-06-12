// Backtest page: async job submission + progress polling, result charts
// (equity vs benchmark, drawdown), sortable trade table, saved history with
// two-way comparison, and Phase 5 desk mode (desk picker, desk header chip,
// trader's-notes timeline, walk-forward refit markers). Talks to:
//   POST /api/backtest/run            -> {job_id} (strategy OR desk payload)
//   GET  /api/backtest/status/<id>    -> JobManager record
//   GET  /api/backtests               -> saved history rows
//   GET  /api/backtest/<id>           -> saved detail (results blob)
//   GET  /api/floor/desks             -> desk registry (contract C1)

'use strict';

const SYMBOLS_RE = /^[A-Za-z][A-Za-z.\-]{0,9}$/;
const POLL_MS = 1000;

// Only a strict #rrggbb desk accent flows into inline styles.
const ACCENT_RE = /^#[0-9a-fA-F]{6}$/;
// Walk-forward refit markers: pod purple, same hue as the 'model' category.
const WF_COLOR = '#bc8cff';
// Trader's-notes categories, in filter-chip display order (contract C3).
const NOTE_CATEGORIES = ['signal', 'risk', 'allocation', 'model', 'info'];

let pollTimer = null;
let restoreRunBtn = null;
let currentTrades = [];
let tradeSort = { key: 'date', dir: 1 };
let hasResults = false;
let desksByKey = {};        // ready desks from /api/floor/desks, keyed by key
let currentNotes = [];      // trader_notes of the report on screen
let noteFilter = 'all';     // active category filter chip

/* ==========================================================================
   Theme helpers (same approach as analysis.js)
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

function baseLayout(t, height) {
    return {
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        font: { family: 'ui-monospace, SFMono-Regular, Menlo, monospace',
                size: 11, color: t.text },
        margin: { l: 64, r: 16, t: 8, b: 32 },
        height,
        showlegend: true,
        legend: { orientation: 'h', y: 1.04, x: 0,
                  font: { size: 10, color: t.muted }, bgcolor: 'rgba(0,0,0,0)' },
        hovermode: 'x unified',
        hoverlabel: { bgcolor: '#1c232d', bordercolor: t.grid,
                      font: { size: 11, color: t.text } },
        xaxis: { gridcolor: t.grid, zerolinecolor: t.grid,
                 tickfont: { size: 10, color: t.muted } },
        yaxis: { gridcolor: t.grid, zerolinecolor: t.grid,
                 tickfont: { size: 10, color: t.muted } },
    };
}

const PLOT_CONFIG = {
    responsive: true, displaylogo: false,
    modeBarButtonsToRemove: ['lasso2d', 'select2d', 'autoScale2d'],
};

/* ==========================================================================
   Form: strategy list, validation, submit
   ========================================================================== */

async function loadStrategies() {
    const select = document.getElementById('btStrategy');
    try {
        const data = await fetchJSON('/api/strategies', { silent: true });
        (data.strategies || []).forEach((s) => {
            const opt = document.createElement('option');
            opt.value = s.id;
            opt.textContent = s.name;
            select.appendChild(opt);
        });
    } catch (_) {
        showToast('error', 'Could not load strategy list — reload the page');
    }
}

/* ==========================================================================
   Desk mode (Phase 5): toggle, desk list, ?desk=<key> deep link
   ========================================================================== */

function currentMode() {
    return document.getElementById('modeDesk').checked ? 'desk' : 'strategy';
}

function applyMode() {
    const desk = currentMode() === 'desk';
    document.getElementById('strategyField').classList.toggle('d-none', desk);
    document.getElementById('deskField').classList.toggle('d-none', !desk);
}

/** Populate #btDesk with READY desks; returns them (empty array on error). */
async function loadDesks() {
    const select = document.getElementById('btDesk');
    const hint = document.getElementById('deskHint');
    let ready = [];
    try {
        const data = await fetchJSON('/api/floor/desks', { silent: true });
        ready = (data.desks || []).filter((d) => d.status === 'ready');
    } catch (_) {
        hint.textContent = 'Could not load desk list — reload the page.';
        document.getElementById('modeDesk').disabled = true;
        return [];
    }
    desksByKey = {};
    ready.forEach((d) => {
        desksByKey[d.key] = d;
        const opt = document.createElement('option');
        opt.value = d.key;
        opt.textContent = d.name;
        select.appendChild(opt);
    });
    if (ready.length === 0) {
        hint.textContent = 'No desks are ready yet — they activate in later phases.';
        document.getElementById('modeDesk').disabled = true;
    }
    return ready;
}

/** Wire the mode toggle and honor a /backtest?desk=<key> deep link. */
async function initDeskMode() {
    document.querySelectorAll('input[name="btMode"]').forEach((radio) => {
        radio.addEventListener('change', applyMode);
    });
    const ready = await loadDesks();
    const deskParam = new URLSearchParams(window.location.search).get('desk');
    if (!deskParam) return;
    if (ready.some((d) => d.key === deskParam)) {
        document.getElementById('modeDesk').checked = true;
        document.getElementById('btDesk').value = deskParam;
        applyMode();
    } else {
        showToast('warning',
            `Desk '${deskParam}' is not available for backtests yet`);
    }
}

function parseSymbols() {
    return document.getElementById('btSymbols').value
        .split(',').map((s) => s.trim().toUpperCase()).filter(Boolean);
}

function validateForm() {
    const symbolsEl = document.getElementById('btSymbols');
    const startEl = document.getElementById('startDate');
    const endEl = document.getElementById('endDate');
    const capitalEl = document.getElementById('btCapital');
    const sizeEl = document.getElementById('btPositionSize');
    let ok = true;

    const symbols = parseSymbols();
    const symbolsOk = symbols.length > 0 && symbols.every((s) => SYMBOLS_RE.test(s));
    symbolsEl.classList.toggle('is-invalid', !symbolsOk);
    ok = ok && symbolsOk;

    const endOk = Boolean(endEl.value);
    endEl.classList.toggle('is-invalid', !endOk);
    ok = ok && endOk;

    const startOk = Boolean(startEl.value) &&
        (!endEl.value || startEl.value < endEl.value);
    startEl.classList.toggle('is-invalid', !startOk);
    ok = ok && startOk;

    const capitalOk = Number(capitalEl.value) > 0;
    capitalEl.classList.toggle('is-invalid', !capitalOk);
    ok = ok && capitalOk;

    const size = Number(sizeEl.value);
    const sizeOk = size >= 0.1 && size <= 100;
    sizeEl.classList.toggle('is-invalid', !sizeOk);
    ok = ok && sizeOk;

    if (currentMode() === 'desk') {
        const deskEl = document.getElementById('btDesk');
        const deskOk = Boolean(deskEl.value);
        deskEl.classList.toggle('is-invalid', !deskOk);
        ok = ok && deskOk;
    }

    return ok;
}

async function onRun(event) {
    event.preventDefault();
    if (pollTimer !== null) return; // a job is already in flight
    if (!validateForm()) return;

    const payload = {
        symbols: parseSymbols().join(','),
        start_date: document.getElementById('startDate').value,
        end_date: document.getElementById('endDate').value,
        initial_capital: Number(document.getElementById('btCapital').value),
        position_size: Number(document.getElementById('btPositionSize').value) / 100,
    };
    // Exactly one of strategy/desk crosses the wire (contract C2 mirrors it).
    if (currentMode() === 'desk') {
        payload.desk = document.getElementById('btDesk').value;
    } else {
        payload.strategy = document.getElementById('btStrategy').value;
    }

    restoreRunBtn = btnLoading(document.getElementById('runBtn'));
    showProgress('submitting…', 0);
    try {
        const data = await fetchJSON('/api/backtest/run', {
            method: 'POST', body: JSON.stringify(payload),
        });
        pollJob(data.job_id);
    } catch (_) {
        resetRunState(); // fetchJSON already toasted
    }
}

/* ==========================================================================
   Progress polling
   ========================================================================== */

function showProgress(statusText, pct) {
    document.getElementById('progressCard').classList.remove('d-none');
    document.getElementById('progressStatus').textContent = statusText;
    document.getElementById('progressPct').textContent = `${pct.toFixed(0)}%`;
    document.getElementById('progressFill').style.width = `${pct}%`;
    document.getElementById('progressBar')
        .setAttribute('aria-valuenow', String(Math.round(pct)));
}

function hideProgress() {
    document.getElementById('progressCard').classList.add('d-none');
}

function resetRunState() {
    if (pollTimer !== null) { clearTimeout(pollTimer); pollTimer = null; }
    hideProgress();
    if (restoreRunBtn) { restoreRunBtn(); restoreRunBtn = null; }
}

function pollJob(jobId) {
    const tick = async () => {
        let job;
        try {
            job = await fetchJSON(`/api/backtest/status/${encodeURIComponent(jobId)}`,
                                  { silent: true });
        } catch (err) {
            resetRunState();
            showToast('error', `Lost track of backtest job: ${err.message}`);
            return;
        }

        if (job.status === 'done') {
            resetRunState();
            renderResults(job.result || {});
            loadSaved(); // the finished run was persisted server-side
            showToast('success', 'Backtest complete');
        } else if (job.status === 'error') {
            resetRunState();
            showToast('error', `Backtest failed: ${job.error || 'unknown error'}`, 0);
        } else {
            const pct = Number(job.progress) || 0;
            showProgress(job.status === 'pending' ? 'queued…' : 'simulating…', pct);
            pollTimer = setTimeout(tick, POLL_MS);
        }
    };
    pollTimer = setTimeout(tick, POLL_MS);
}

/* ==========================================================================
   Results rendering
   ========================================================================== */

function renderResults(report) {
    hasResults = true;
    document.getElementById('resultsEmpty').innerHTML = '';
    document.getElementById('compareSection').classList.add('d-none');
    document.getElementById('resultsSection').classList.remove('d-none');

    renderDeskChip(report.desk);
    renderMetrics(report.summary || {});
    renderPendingSignals(report.pending_signals || []);
    renderEquityChart(report);
    renderDrawdownChart(report.drawdown_series || []);
    renderTrades(report.trades || []);
    renderTraderNotes(report);
    renderProvenance(report.data_sources || {});
}

/* ==========================================================================
   Desk results (Phase 5): header chip + trader's-notes timeline
   ========================================================================== */

function renderDeskChip(desk) {
    const row = document.getElementById('deskChipRow');
    if (!desk || !desk.key) {
        row.classList.add('d-none');
        row.innerHTML = '';
        return;
    }
    const meta = desksByKey[desk.key] || {};
    const accent = ACCENT_RE.test(meta.accent || '') ? meta.accent : '';
    row.classList.remove('d-none');
    row.innerHTML =
        `<span class="desk-chip"${accent ? ` style="--desk-accent: ${accent};"` : ''}>` +
        '<i class="bi bi-building" aria-hidden="true"></i>' +
        `${escapeHTML(desk.name || desk.key)} desk</span>`;
}

function noteCategory(note) {
    return NOTE_CATEGORIES.includes(note.category) ? note.category : 'info';
}

function renderTraderNotes(report) {
    const card = document.getElementById('traderNotesCard');
    if (!report.desk) {
        // Strategy-mode runs have no desk rationale to show.
        card.classList.add('d-none');
        currentNotes = [];
        return;
    }
    card.classList.remove('d-none');
    currentNotes = Array.isArray(report.trader_notes) ? report.trader_notes : [];
    noteFilter = 'all';
    paintNoteFilters();
    paintNotes();
}

function paintNoteFilters() {
    const counts = {};
    currentNotes.forEach((n) => {
        const cat = noteCategory(n);
        counts[cat] = (counts[cat] || 0) + 1;
    });
    const chips = [['all', `All (${currentNotes.length})`]].concat(
        NOTE_CATEGORIES.filter((c) => counts[c])
            .map((c) => [c, `${c} (${counts[c]})`]));

    const wrap = document.getElementById('noteFilters');
    wrap.innerHTML = chips.map(([value, label]) =>
        '<button type="button" class="note-filter-chip' +
        `${value === 'all' ? '' : ` note-cat-${value}`}` +
        `${value === noteFilter ? ' active' : ''}"` +
        ` data-category="${value}" aria-pressed="${value === noteFilter}">` +
        `${escapeHTML(label)}</button>`).join('');
    wrap.querySelectorAll('button').forEach((btn) => {
        btn.addEventListener('click', () => {
            noteFilter = btn.dataset.category;
            paintNoteFilters();
            paintNotes();
        });
    });
}

function noteItem(note) {
    const category = noteCategory(note);
    const data = note.data && typeof note.data === 'object' ? note.data : null;
    const hasData = data !== null && Object.keys(data).length > 0;
    return (
        '<li class="note-entry">' +
        `<span class="note-ts num">${escapeHTML(note.timestamp ?? '—')}</span>` +
        `<span class="note-cat-chip note-cat-${category}">${escapeHTML(category)}</span>` +
        `<span class="note-msg">${escapeHTML(note.message ?? '')}</span>` +
        (hasData
            ? '<details class="note-data"><summary>data</summary>' +
              `<pre class="note-data-json">${escapeHTML(JSON.stringify(data, null, 2))}</pre>` +
              '</details>'
            : '') +
        '</li>'
    );
}

function paintNotes() {
    const list = document.getElementById('notesTimeline');
    const emptyEl = document.getElementById('notesEmpty');
    const visible = noteFilter === 'all'
        ? currentNotes
        : currentNotes.filter((n) => noteCategory(n) === noteFilter);

    if (visible.length === 0) {
        list.innerHTML = '';
        emptyEl.classList.remove('d-none');
        emptyEl.innerHTML = currentNotes.length === 0
            ? emptyStateHTML('bi-journal-text', 'No trader notes',
                'The desk logged nothing for this run.')
            : emptyStateHTML('bi-funnel', `No ${noteFilter} notes`,
                'Pick another category filter.');
        return;
    }
    emptyEl.classList.add('d-none');
    list.innerHTML = visible.map(noteItem).join('');
}

function setMetric(id, text, signedValue) {
    const el = document.getElementById(id);
    el.textContent = text;
    el.classList.remove('pnl-pos', 'pnl-neg', 'pnl-flat');
    if (signedValue !== undefined) el.classList.add(pnlClass(signedValue));
}

/**
 * Total return as a fraction (0.08 == +8%) from an engine summary.
 * summary.total_return is a DOLLAR P&L (realized + unrealized);
 * summary.total_return_pct is x100. Prefer the pct, fall back to
 * dollars / initial_capital, else null (renders as an em dash).
 */
function totalReturnFraction(summary) {
    if (summary.total_return_pct !== null &&
        summary.total_return_pct !== undefined) {
        return Number(summary.total_return_pct) / 100;
    }
    const cap = Number(summary.initial_capital);
    if (cap > 0 && summary.total_return !== null &&
        summary.total_return !== undefined) {
        return Number(summary.total_return) / cap;
    }
    return null;
}

function renderMetrics(summary) {
    // max_drawdown / win_rate arrive as percents (x100) and are formatted
    // directly; total return is converted to a fraction for fmtPct.
    const totalReturn = totalReturnFraction(summary);
    setMetric('mTotalReturn', fmtPct(totalReturn, { sign: true }),
              totalReturn);
    setMetric('mSharpe', fmtNum(summary.sharpe_ratio, 2));
    setMetric('mSortino', fmtNum(summary.sortino_ratio, 2));
    setMetric('mCalmar', fmtNum(summary.calmar_ratio, 2));
    setMetric('mMaxDD',
              summary.max_drawdown === null || summary.max_drawdown === undefined
                  ? '—' : `${Number(summary.max_drawdown).toFixed(2)}%`,
              summary.max_drawdown);
    setMetric('mWinRate',
              summary.win_rate === null || summary.win_rate === undefined
                  ? '—' : `${Number(summary.win_rate).toFixed(1)}%`);
}

function renderPendingSignals(pending) {
    const note = document.getElementById('pendingSignalsNote');
    if (pending.length === 0) {
        note.classList.add('d-none');
        note.innerHTML = '';
        return;
    }
    const detail = pending
        .map((p) => `${p.symbol} ${p.signal} (signal ${p.signal_date})`)
        .join(' · ');
    note.classList.remove('d-none');
    note.innerHTML =
        '<div class="note-line"><i class="bi bi-hourglass-split" aria-hidden="true"></i>' +
        `<span>Unfilled at end of simulation (no next bar): ${escapeHTML(detail)}</span></div>`;
}

function renderEquityChart(report) {
    const t = plotlyTheme();
    const history = report.portfolio_history || [];
    const traces = [{
        type: 'scatter', name: 'Portfolio',
        x: history.map((h) => h.timestamp),
        y: history.map((h) => h.portfolio_value),
        line: { color: t.accent, width: 1.5 },
    }];

    const caption = document.getElementById('equityProvenance');
    if (report.benchmark && Array.isArray(report.benchmark.equity_curve)) {
        traces.push({
            type: 'scatter',
            name: `${report.benchmark.symbol} buy & hold`,
            x: report.benchmark.equity_curve.map((p) => p.date),
            y: report.benchmark.equity_curve.map((p) => p.value),
            line: { color: t.muted, width: 1.25, dash: 'dot' },
        });
        caption.textContent = '';
    } else {
        caption.textContent = 'benchmark unavailable';
    }

    const layout = baseLayout(t, 320);
    layout.yaxis.title = { text: 'Value ($)', font: { size: 10, color: t.muted } };
    layout.yaxis.tickformat = ',.0f';
    addWalkForwardMarkers(report, history, traces, layout);
    Plotly.react(document.getElementById('equityChart'), traces, layout, PLOT_CONFIG);
}

/**
 * Walk-forward refit markers (desk mode, contract C3): a vertical dashed
 * line per report.walk_forward entry at its fit_date, plus one legend-bearing
 * marker trace whose hover text describes the training window. Skips cleanly
 * when the list is empty/absent (strategy runs) or there is no equity curve
 * to anchor the hover markers to.
 */
function addWalkForwardMarkers(report, history, traces, layout) {
    const refits = Array.isArray(report.walk_forward) ? report.walk_forward : [];
    if (refits.length === 0 || history.length === 0) return;

    layout.shapes = refits.map((wf) => ({
        type: 'line', xref: 'x', yref: 'paper',
        x0: wf.fit_date, x1: wf.fit_date, y0: 0, y1: 1,
        line: { color: WF_COLOR, width: 1, dash: 'dash' },
    }));

    const top = Math.max(...history.map((h) => h.portfolio_value));
    traces.push({
        type: 'scatter', name: 'Walk-forward refit', mode: 'markers',
        x: refits.map((wf) => wf.fit_date),
        y: refits.map(() => top),
        marker: { symbol: 'line-ns-open', size: 9, color: WF_COLOR,
                  line: { width: 1.5, color: WF_COLOR } },
        text: refits.map((wf) =>
            `Refit: trained ${wf.train_start} → ${wf.train_end} ` +
            `(${wf.n_samples} rows)`),
        hovertemplate: '%{text}<extra></extra>',
    });
}

function renderDrawdownChart(series) {
    const t = plotlyTheme();
    const traces = [{
        type: 'scatter', name: 'Drawdown',
        x: series.map((p) => p.date),
        y: series.map((p) => p.drawdown_pct),
        fill: 'tozeroy',
        fillcolor: 'rgba(248, 81, 73, 0.18)',
        line: { color: t.loss, width: 1.25 },
    }];
    const layout = baseLayout(t, 200);
    layout.showlegend = false;
    layout.yaxis.title = { text: 'DD (%)', font: { size: 10, color: t.muted } };
    Plotly.react(document.getElementById('drawdownChart'), traces, layout, PLOT_CONFIG);
}

function renderProvenance(dataSources) {
    const parts = [];
    const failures = [];
    Object.entries(dataSources).forEach(([symbol, info]) => {
        if (!info) return;
        parts.push(`${symbol}: ${info.from_cache ? 'cache' : (info.provider || 'cache')}`);
        (info.failures || []).forEach((f) =>
            failures.push(`${symbol} ${f.provider}: ${f.error}`));
    });
    if (parts.length > 0) {
        const caption = document.getElementById('equityProvenance');
        const benchNote = caption.textContent;
        caption.textContent =
            `served by ${parts.join(' · ')}${benchNote ? ` · ${benchNote}` : ''}`;
    }
    if (failures.length > 0) {
        showToast('warning', `Provider failures — ${failures.join('; ')}`);
    }
}

/* ==========================================================================
   Trades table (click-to-sort)
   ========================================================================== */

function tradeValue(trade) {
    return trade.cost !== undefined ? trade.cost : trade.proceeds;
}

function renderTrades(trades) {
    currentTrades = trades.map((trade) => ({ ...trade, value: tradeValue(trade) }));
    const emptyEl = document.getElementById('tradesEmpty');
    if (currentTrades.length === 0) {
        document.getElementById('tradesBody').innerHTML = '';
        emptyEl.classList.remove('d-none');
        emptyEl.innerHTML = emptyStateHTML('bi-receipt', 'No trades filled',
            'The strategy produced no fills over this window.');
        return;
    }
    emptyEl.classList.add('d-none');
    sortAndPaintTrades();
}

function sortAndPaintTrades() {
    const { key, dir } = tradeSort;
    const sorted = currentTrades.slice().sort((a, b) => {
        const av = a[key]; const bv = b[key];
        if (av === bv) return 0;
        if (av === undefined || av === null) return 1;
        if (bv === undefined || bv === null) return -1;
        return (av < bv ? -1 : 1) * dir;
    });

    document.getElementById('tradesBody').innerHTML = sorted.map((trade) => {
        const buy = trade.action === 'BUY';
        return '<tr>' +
            `<td class="num">${escapeHTML(trade.signal_date ?? '—')}</td>` +
            `<td class="num">${escapeHTML(trade.date ?? '—')}</td>` +
            `<td>${escapeHTML(trade.symbol ?? '')}</td>` +
            `<td class="${buy ? 'pnl-pos' : 'pnl-neg'}">${escapeHTML(trade.action ?? '')}</td>` +
            `<td class="num">${fmtNum(trade.quantity, 0)}</td>` +
            `<td class="num">${fmtMoney(trade.price)}</td>` +
            `<td class="num">${fmtMoney(trade.value)}</td>` +
            '</tr>';
    }).join('');

    document.querySelectorAll('#tradesTable th.sortable').forEach((th) => {
        const arrow = th.querySelector('.sort-arrow');
        const active = th.dataset.key === key;
        arrow.textContent = active ? (dir > 0 ? '▲' : '▼') : '';
        th.setAttribute('aria-sort',
            active ? (dir > 0 ? 'ascending' : 'descending') : 'none');
    });
}

function initTradeSorting() {
    document.querySelectorAll('#tradesTable th.sortable').forEach((th) => {
        th.addEventListener('click', () => {
            const key = th.dataset.key;
            tradeSort = {
                key,
                dir: tradeSort.key === key ? -tradeSort.dir : 1,
            };
            sortAndPaintTrades();
        });
    });
}

/* ==========================================================================
   Saved history + comparison
   ========================================================================== */

async function loadSaved() {
    const list = document.getElementById('savedList');
    const emptyEl = document.getElementById('savedEmpty');
    try {
        const data = await fetchJSON('/api/backtests', { silent: true });
        const rows = (data.backtests || []).slice(0, 15);
        if (rows.length === 0) {
            list.innerHTML = '';
            emptyEl.classList.remove('d-none');
            emptyEl.innerHTML = emptyStateHTML('bi-archive', 'No saved backtests',
                'Finished runs are saved here automatically.');
            updateCompareButton();
            return;
        }
        emptyEl.classList.add('d-none');
        list.innerHTML = rows.map(savedItem).join('');
        list.querySelectorAll('input[type="checkbox"]').forEach((box) => {
            box.addEventListener('change', updateCompareButton);
        });
        updateCompareButton();
    } catch (_) {
        emptyEl.classList.remove('d-none');
        emptyEl.innerHTML = emptyStateHTML('bi-exclamation-triangle',
            'Could not load history', 'Check the server log and refresh.');
    }
}

function savedItem(bt) {
    // bt.total_return is the DB column: a FRACTION of initial capital.
    const ret = bt.total_return;
    return (
        `<li class="d-flex align-items-center gap-2 px-3 py-2 border-bottom" style="border-color: var(--border) !important;">` +
        `<input class="form-check-input mt-0 flex-shrink-0" type="checkbox" value="${Number(bt.id)}" ` +
        `aria-label="Select ${escapeHTML(bt.name || `backtest ${bt.id}`)} for comparison">` +
        '<div class="flex-grow-1" style="min-width:0;">' +
        `<div class="text-truncate" style="font-size:12px;">${escapeHTML(bt.name || `#${bt.id}`)}</div>` +
        `<div class="provenance-caption">${escapeHTML(bt.strategy || '—')} · ${escapeHTML(bt.start_date || '?')} → ${escapeHTML(bt.end_date || '?')}</div>` +
        '</div>' +
        `<span class="num ${pnlClass(ret)}" style="font-size:12px;">${fmtPct(ret, { sign: true })}</span>` +
        '</li>'
    );
}

function selectedSavedIds() {
    return Array.from(
        document.querySelectorAll('#savedList input[type="checkbox"]:checked'),
        (box) => Number(box.value));
}

function updateCompareButton() {
    const btn = document.getElementById('compareBtn');
    const n = selectedSavedIds().length;
    btn.disabled = n !== 2;
    btn.innerHTML =
        '<i class="bi bi-layout-split" aria-hidden="true"></i> ' +
        (n === 2 ? 'Compare selected' : `Compare selected (pick 2, have ${n})`);
}

async function onCompare() {
    const ids = selectedSavedIds();
    if (ids.length !== 2) return;

    const restore = btnLoading(document.getElementById('compareBtn'));
    try {
        const [a, b] = await Promise.all(ids.map((id) =>
            fetchJSON(`/api/backtest/${id}`)));
        renderCompare(a.backtest, b.backtest);
    } catch (_) {
        // fetchJSON already toasted.
    } finally {
        restore();
        updateCompareButton();
    }
}

/**
 * Display percent (x100) for a compare-card metric. The engine summary
 * inside the results blob carries x100 percents; the flat DB columns store
 * FRACTIONS (see api_backtest._save_report and export_report's ':.2%'),
 * so the fallback is scaled by 100. Null when neither is present.
 */
function summaryPctOrDbFraction(summaryPct, dbFraction) {
    if (summaryPct !== null && summaryPct !== undefined) {
        return Number(summaryPct);
    }
    if (dbFraction !== null && dbFraction !== undefined) {
        return Number(dbFraction) * 100;
    }
    return null;
}

function compareCard(bt) {
    const results = bt.results || {};
    const summary = results.summary || {};
    const maxDD = summaryPctOrDbFraction(summary.max_drawdown, bt.max_drawdown);
    const winRate = summaryPctOrDbFraction(summary.win_rate, bt.win_rate);
    // Prefer the engine summary (total_return_pct, x100) when the results
    // blob has one; fall back to the DB column, a fraction.
    const ret = summary.total_return_pct !== null &&
                summary.total_return_pct !== undefined
        ? Number(summary.total_return_pct) / 100
        : bt.total_return;
    const rows = [
        ['Total return', fmtPct(ret, { sign: true }), ret],
        ['Sharpe', fmtNum(bt.sharpe_ratio, 2)],
        ['Sortino', fmtNum(summary.sortino_ratio, 2)],
        ['Calmar', fmtNum(summary.calmar_ratio, 2)],
        ['Max drawdown', maxDD === null || maxDD === undefined ? '—' : `${Number(maxDD).toFixed(2)}%`, maxDD],
        ['Win rate', winRate === null || winRate === undefined ? '—' : `${Number(winRate).toFixed(1)}%`],
    ];
    return (
        '<div class="col-md-6"><div class="card h-100">' +
        `<div class="card-header text-truncate">${escapeHTML(bt.name || `#${bt.id}`)}</div>` +
        '<div class="card-body"><div class="row g-2">' +
        rows.map(([label, text, signed]) =>
            '<div class="col-6 col-xl-4"><div class="metric-card">' +
            `<div class="metric-label">${label}</div>` +
            `<div class="metric-value num${signed !== undefined ? ` ${pnlClass(signed)}` : ''}" style="font-size:16px;">${text}</div>` +
            '</div></div>').join('') +
        '</div>' +
        `<div class="provenance-caption mt-2">${escapeHTML(bt.strategy || '—')} · ${escapeHTML(bt.start_date || '?')} → ${escapeHTML(bt.end_date || '?')}</div>` +
        '</div></div></div>'
    );
}

function renderCompare(a, b) {
    document.getElementById('resultsSection').classList.add('d-none');
    document.getElementById('resultsEmpty').innerHTML = '';
    const section = document.getElementById('compareSection');
    section.classList.remove('d-none');

    document.getElementById('compareCards').innerHTML =
        compareCard(a) + compareCard(b);

    const t = plotlyTheme();
    const traces = [];
    let missing = 0;
    [[a, t.accent, 'solid'], [b, t.gain, 'dash']].forEach(([bt, color, dash]) => {
        const history = (bt.results || {}).portfolio_history || [];
        if (history.length === 0) { missing += 1; return; }
        traces.push({
            type: 'scatter', name: bt.name || `#${bt.id}`,
            x: history.map((h) => h.timestamp),
            y: history.map((h) => h.portfolio_value),
            line: { color, width: 1.5, dash },
        });
    });

    const layout = baseLayout(t, 320);
    layout.yaxis.title = { text: 'Value ($)', font: { size: 10, color: t.muted } };
    layout.yaxis.tickformat = ',.0f';
    Plotly.react(document.getElementById('compareChart'), traces, layout, PLOT_CONFIG);

    if (missing > 0) {
        showToast('warning',
            `${missing === 2 ? 'Both' : 'One'} saved run${missing === 2 ? 's' : ''} ` +
            'predates equity-curve storage — curve omitted');
    }
    section.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function closeCompare() {
    document.getElementById('compareSection').classList.add('d-none');
    if (hasResults) {
        document.getElementById('resultsSection').classList.remove('d-none');
    } else {
        paintInitialEmptyState();
    }
}

/* ==========================================================================
   Init
   ========================================================================== */

function paintInitialEmptyState() {
    document.getElementById('resultsEmpty').innerHTML =
        '<div class="card"><div class="card-body p-0">' +
        emptyStateHTML('bi-clock-history', 'No backtest yet',
            'Configure parameters and run — progress streams in the background.') +
        '</div></div>';
}

document.addEventListener('DOMContentLoaded', () => {
    paintInitialEmptyState();
    loadStrategies();
    initDeskMode();
    loadSaved();
    initTradeSorting();
    document.getElementById('backtestForm').addEventListener('submit', onRun);
    document.getElementById('reloadSavedBtn').addEventListener('click', loadSaved);
    document.getElementById('compareBtn').addEventListener('click', onCompare);
    document.getElementById('closeCompareBtn').addEventListener('click', closeCompare);
});
