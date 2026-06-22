// Backtest page: async job submission + progress polling, result charts
// (equity vs benchmark, drawdown), sortable trade table, saved history with
// two-way comparison, and Phase 5 desk mode (desk picker, desk header chip,
// trader's-notes timeline, walk-forward refit markers). Phase 6 adds the
// Renaissance regime visualization (contract C5: translucent equity-chart
// bands + a current-regime chip), book filter chips on the trader's notes
// (contract C7: note.data.book), and per-model walk-forward marker colors
// (fits MAY carry 'model'). Phase 7 adds the Citadel pod visualization
// (contract C8: report.pod_history -> stacked pod-weight allocation chart
// with reallocation/probation/cut event markers + per-pod status cards) and
// pod filter chips on the trader's notes (contract C9: note.data.pod,
// composing with the category filter exactly like books). Phase 8 adds the
// Jane Street panels: a sortable option-structures table with expandable
// legs (contract C11: report.structures), a daily portfolio-Greeks chart
// (contract C12: report.greeks_series), an estimated-pricing disclaimer on
// janestreet reports (backtest option prices are SYNTHETIC — Black-Scholes
// on historical volatility; real chains arrive with E*TRADE in Phase 9),
// generalized book chips (contract C14 adds vrp/earnings/relative_value)
// and monospace structure tags on notes carrying data.structure_id, plus an
// additive 'instrument' line in the trades table (contract C13). Talks to:
//   POST /api/backtest/run            -> {job_id} (strategy OR desk payload)
//   GET  /api/backtest/status/<id>    -> JobManager record
//   GET  /api/backtests               -> saved history rows
//   GET  /api/backtest/<id>           -> saved detail (results blob)
//   GET  /api/floor/desks             -> desk registry (contract C1)

'use strict';

const SYMBOLS_RE = /^[A-Za-z][A-Za-z.\-]{0,9}$/;
const POLL_MS = 1000;

// The desk_model field on /api/backtest/run is accepted only for the
// model-selectable desks (foundation, twosigma); sending it for any other
// desk -> 400, so the Model picker is gated on membership in this set.
const MODEL_SELECTABLE_DESKS = new Set(['foundation', 'twosigma']);
// The historical default model id; selecting it == omitting desk_model.
const DEFAULT_DESK_MODEL = 'gbm';

// Only a strict #rrggbb desk accent flows into inline styles.
const ACCENT_RE = /^#[0-9a-fA-F]{6}$/;
// Walk-forward refit markers: pod purple, same hue as the 'model' category.
// Phase 6: fits MAY carry 'model' — those get a per-model hue instead.
const WF_COLOR = '#bc8cff';
const WF_MODEL_COLORS = {
    regime: '#58a6ff',
    stat_arb: '#d29922',
    pairs: '#bc8cff',
};
// Trader's-notes categories, in filter-chip display order (contract C3).
const NOTE_CATEGORIES = ['signal', 'risk', 'allocation', 'model', 'info'];
// Known note.data.book keys -> display label (contracts C7 + C14). Phase 8
// generalizes books: ANY non-blank data.book string now renders a chip (the
// key only ever reaches the DOM as escaped text/textContent), but ONLY the
// keys listed here additionally get a note-book-<key> color class — CSS
// suffixes stay whitelisted. Known keys also fix the filter-chip display
// order (declaration order), with unknown books appended first-seen.
const NOTE_BOOKS = {
    regime: 'Regime',
    mean_reversion: 'Mean Reversion',
    stat_arb: 'Stat Arb',
    pairs: 'Pairs',
    vrp: 'VRP',
    earnings: 'Earnings',
    relative_value: 'Relative Value',
};
// Citadel pods (contract C8): pod keys are dynamic backend strings, so
// unlike NOTE_BOOKS there is no whitelist — each pod is assigned the next
// hue from this fixed palette in first-seen order (cycling past the end),
// and pod text only ever reaches the DOM escaped/as textContent.
const POD_PALETTE = ['#58a6ff', '#3fb950', '#d29922', '#bc8cff',
                     '#39c5cf', '#ff7b72', '#7ee787', '#f0883e'];
// Pod lifecycle states (contract C8) -> status-badge label + CSS class.
const POD_STATUS_META = {
    active:    { label: 'ACTIVE',    cls: 'pod-status-active' },
    probation: { label: 'PROBATION', cls: 'pod-status-probation' },
    cut:       { label: 'CUT',       cls: 'pod-status-cut' },
};
// Pod-chart event markers: reallocation step-changes + status downgrades.
const POD_EVENT_META = {
    realloc:   { label: 'Reallocation',  color: '#bc8cff' },
    probation: { label: 'Pod probation', color: '#d29922' },
    cut:       { label: 'Pod cut',       color: '#f85149' },
};
// Jane Street structures (contract C11): whitelisted structure types ->
// display label. Unknown types render as escaped text with no extra class.
const STRUCTURE_TYPE_LABELS = {
    iron_condor: 'Iron condor',
    put_credit_spread: 'Put credit spread',
    call_credit_spread: 'Call credit spread',
};
// Structure lifecycle states (contract C11) -> status-badge label + CSS
// class (same anatomy as the pod status badges).
const STRUCTURE_STATUS_META = {
    open:    { label: 'OPEN',    cls: 'structure-status-open' },
    closed:  { label: 'CLOSED',  cls: 'structure-status-closed' },
    expired: { label: 'EXPIRED', cls: 'structure-status-expired' },
};
// close_reason chip labels (contract C11). Only these keys get a
// reason-<key> color class; unknown reasons render as escaped text only.
const CLOSE_REASON_LABELS = {
    profit_target: 'profit target',
    stop_loss: 'stop loss',
    time_exit: 'time exit',
    regime_flatten: 'regime flatten',
    expiry: 'expiry',
};
// Regime states (contract C5): equity-chart band tint (low alpha, behind
// the traces) + solid hue for the legend swatch and current-regime chip.
const REGIME_META = {
    mean_reverting: { label: 'Mean-reverting',
                      band: 'rgba(88, 166, 255, 0.10)', color: '#58a6ff' },
    trending:       { label: 'Trending',
                      band: 'rgba(63, 185, 80, 0.10)', color: '#3fb950' },
    high_vol:       { label: 'High-vol',
                      band: 'rgba(248, 81, 73, 0.10)', color: '#f85149' },
};

let pollTimer = null;
let restoreRunBtn = null;
let currentTrades = [];
let tradeSort = { key: 'date', dir: 1 };
let hasResults = false;
let desksByKey = {};        // ready desks from /api/floor/desks, keyed by key
let currentNotes = [];      // trader_notes of the report on screen
let noteFilter = 'all';     // active category filter chip
let bookFilter = 'all';     // active book filter chip (composes with above)
let podFilter = 'all';      // active pod filter chip (composes with above)
let podColors = new Map();  // pod key -> palette hue, first-seen order
let currentStructures = [];           // report.structures on screen (C11)
let structSort = { key: 'opened', dir: 1 };
let expandedStructures = new Set();   // structure _key -> legs row open

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
   Desk model picker (Phase A): /api/models -> #btDeskModel.
   desk_model is accepted on /api/backtest/run only for the model-selectable
   desks (foundation, twosigma), so the picker is shown only when one of those
   desks is the selected desk (see applyMode / the #btDesk change handler) and
   only then does desk_model cross the wire.
   ========================================================================== */

/** Populate #btDeskModel from /api/models, default-selecting gbm. Each option
 *  carries its description as a title tooltip; on the active option the
 *  description also surfaces under the select (mirroring the desk hint). The
 *  Model field is purely additive — failure leaves it empty + hidden, and a
 *  hidden/empty picker simply never sends desk_model. */
async function loadModels() {
    const select = document.getElementById('btDeskModel');
    if (!select) return;
    try {
        const data = await fetchJSON('/api/models', { silent: true });
        (data.models || []).forEach((m) => {
            const opt = document.createElement('option');
            opt.value = m.id;
            opt.textContent = m.name;
            if (m.description) opt.title = m.description;
            opt.dataset.description = m.description || '';
            select.appendChild(opt);
        });
        // Default to the historical default model (byte-identical to omitting).
        if (Array.from(select.options).some((o) => o.value === DEFAULT_DESK_MODEL)) {
            select.value = DEFAULT_DESK_MODEL;
        }
        select.addEventListener('change', paintDeskModelHint);
        paintDeskModelHint();
    } catch (_) {
        // The desk picker still gates on the model-selectable desk set; a
        // model-less picker just stays hidden and no desk_model is sent. The
        // page keeps working.
        showToast('error', 'Could not load model list — reload the page');
    }
}

/** Show the selected model's description under the picker (desk-hint style). */
function paintDeskModelHint() {
    const select = document.getElementById('btDeskModel');
    const hint = document.getElementById('deskModelHint');
    if (!select || !hint) return;
    const opt = select.selectedOptions[0];
    hint.textContent = (opt && opt.dataset.description) || '';
}

/** True when the Model picker is live: desk mode AND the selected desk is one
 *  of the model-selectable desks (foundation, twosigma). This is the EXACT
 *  gate for both visibility and whether desk_model is in the run payload. */
function deskModelActive() {
    return currentMode() === 'desk' &&
        MODEL_SELECTABLE_DESKS.has(document.getElementById('btDesk').value);
}

/** Show/hide + enable/disable the Model picker per deskModelActive(). Disabling
 *  the hidden control keeps it out of the tab order and makes the intent
 *  explicit; visibility is what gates the payload. */
function applyDeskModelVisibility() {
    const field = document.getElementById('deskModelField');
    const select = document.getElementById('btDeskModel');
    if (!field || !select) return;
    const active = deskModelActive();
    field.classList.toggle('d-none', !active);
    select.disabled = !active;
}

/* ==========================================================================
   Desk mode (Phase 5): toggle, desk list, ?desk=<key> deep link
   ========================================================================== */

function currentMode() {
    const checked = document.querySelector('input[name="btMode"]:checked');
    return checked ? checked.value : 'strategy';
}

function applyMode() {
    const mode = currentMode();
    document.getElementById('strategyField').classList.toggle('d-none', mode !== 'strategy');
    document.getElementById('deskField').classList.toggle('d-none', mode !== 'desk');
    document.getElementById('fundField').classList.toggle('d-none', mode !== 'fund');
    // Position size drives strategy/desk sizing only; a fund's desks size
    // themselves and the reweighter sets capital, so hide it in fund mode.
    document.getElementById('positionSizeField').classList.toggle('d-none', mode === 'fund');
    // Realistic fills are strategy/desk only (fund mode does not thread the
    // flag), so hide the checkbox in fund mode rather than leave an inert one.
    const realisticField = document.getElementById('realisticFillsField');
    if (realisticField) {
        realisticField.classList.toggle('d-none', mode === 'fund');
    }
    // The Model picker is limited to the model-selectable desks (foundation,
    // twosigma); refresh it on every mode change (the #btDesk change handler
    // covers desk-to-desk switches).
    applyDeskModelVisibility();
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
        document.getElementById('modeFund').disabled = true;
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
    buildFundDeskList(ready);
    if (ready.length === 0) {
        hint.textContent = 'No desks are ready yet — they activate in later phases.';
        document.getElementById('modeDesk').disabled = true;
        document.getElementById('modeFund').disabled = true;
    }
    return ready;
}

/** Render the fund desk checklist: one row per ready desk (checkbox + name +
 *  percent-weight input), all checked at equal weight by default. Names and
 *  keys reach the DOM only via textContent/dataset; accents pass ACCENT_RE
 *  before any inline style. */
function buildFundDeskList(ready) {
    const list = document.getElementById('fundDeskList');
    if (!list) return;
    list.textContent = '';
    if (ready.length === 0) return;
    const equalPct = Math.floor(100 / ready.length);
    ready.forEach((d) => {
        const accent = ACCENT_RE.test(d.accent || '') ? d.accent : '';
        const row = document.createElement('div');
        row.className = 'fund-desk-row d-flex align-items-center gap-2 mb-1';

        const cb = document.createElement('input');
        cb.type = 'checkbox';
        cb.className = 'form-check-input mt-0';
        cb.checked = true;
        cb.id = `fundDesk_${d.key}`;
        cb.dataset.fundDesk = d.key;

        const label = document.createElement('label');
        label.className = 'form-check-label flex-grow-1 text-truncate';
        label.htmlFor = cb.id;
        label.textContent = d.name;
        if (accent) {
            label.style.borderLeft = `3px solid ${accent}`;
            label.style.paddingLeft = '6px';
        }

        const weight = document.createElement('input');
        weight.type = 'number';
        weight.className = 'form-control form-control-sm num fund-weight';
        weight.style.width = '5rem';
        weight.min = '0';
        weight.max = '100';
        weight.step = '1';
        weight.value = String(equalPct);
        weight.dataset.fundWeight = d.key;
        weight.setAttribute('aria-label', `${d.name} weight (%)`);

        const pct = document.createElement('span');
        pct.className = 'provenance-caption';
        pct.textContent = '%';

        row.append(cb, label, weight, pct);
        list.appendChild(row);
    });
}

/** {desk_key: fraction} for the CHECKED desks (percent inputs -> fractions). */
function fundAllocations() {
    const out = {};
    document.querySelectorAll('#fundDeskList input[type="checkbox"]')
        .forEach((cb) => {
            if (!cb.checked) return;
            const key = cb.dataset.fundDesk;
            const wEl = document.querySelector(
                `#fundDeskList input[data-fund-weight="${key}"]`);
            const pct = Number(wEl && wEl.value);
            out[key] = Number.isFinite(pct) ? pct / 100 : 0;
        });
    return out;
}

/** Wire the mode toggle and honor a /backtest?desk=<key> deep link. */
async function initDeskMode() {
    // Sandbox workspace renders Strategy mode only (no Desk/Fund radios), so
    // skip all desk/fund wiring — listeners, the desk fetch, and the deep link.
    if (!document.getElementById('modeDesk')) return;
    document.querySelectorAll('input[name="btMode"]').forEach((radio) => {
        radio.addEventListener('change', applyMode);
    });
    // Desk-to-desk switches must toggle the model-selectable Model picker too.
    document.getElementById('btDesk')
        .addEventListener('change', applyDeskModelVisibility);
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

    if (currentMode() === 'fund') {
        ok = validateFundFields() && ok;
    }

    return ok;
}

/** Fund-mode field validation: at least one checked desk, each weight > 0,
 *  the checked weights sum to <= 100%, and valid reweight params. */
function validateFundFields() {
    const alloc = fundAllocations();
    const keys = Object.keys(alloc);
    const errEl = document.getElementById('fundDeskError');
    let fundOk = true;

    if (keys.length === 0) {
        errEl.textContent = 'Pick at least one desk.';
        fundOk = false;
    } else if (keys.some((k) => !(alloc[k] > 0))) {
        errEl.textContent = 'Each checked desk needs a weight greater than 0.';
        fundOk = false;
    } else {
        const sum = keys.reduce((s, k) => s + alloc[k], 0);
        if (sum > 1.0 + 1e-9) {
            errEl.textContent =
                `Weights sum to ${(sum * 100).toFixed(0)}% — must be ≤ 100%.`;
            fundOk = false;
        }
    }
    errEl.classList.toggle('d-none', fundOk);

    const reb = Number(document.getElementById('fundRebalance').value);
    const rebOk = Number.isInteger(reb) && reb >= 1;
    document.getElementById('fundRebalance').classList.toggle('is-invalid', !rebOk);

    const warm = Number(document.getElementById('fundWarmup').value);
    const warmOk = Number.isInteger(warm) && warm >= 0;
    document.getElementById('fundWarmup').classList.toggle('is-invalid', !warmOk);

    const tg = Number(document.getElementById('fundTargetGross').value);
    const tgOk = tg >= 1 && tg <= 100;
    document.getElementById('fundTargetGross').classList.toggle('is-invalid', !tgOk);

    return fundOk && rebOk && warmOk && tgOk;
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
    // Exactly one of strategy/desk/fund crosses the wire (the route mirrors it).
    if (currentMode() === 'desk') {
        payload.desk = document.getElementById('btDesk').value;
        // desk_model is for model-selectable desks (foundation, twosigma)
        // only: include it only when the picker is live. Sending it for any
        // other desk -> 400.
        if (deskModelActive()) {
            payload.desk_model = document.getElementById('btDeskModel').value;
        }
    } else if (currentMode() === 'fund') {
        payload.fund = fundAllocations();
        payload.rebalance_every = Number(document.getElementById('fundRebalance').value);
        payload.warmup = Number(document.getElementById('fundWarmup').value);
        payload.target_gross = Number(document.getElementById('fundTargetGross').value) / 100;
        delete payload.position_size; // the fund path sizes via desks + reweighter
    } else {
        payload.strategy = document.getElementById('btStrategy').value;
    }
    // Opt-in realistic fills (Step 6): strategy/desk modes only (the fund path
    // does not expose it yet).
    if (currentMode() !== 'fund') {
        const realistic = document.getElementById('realisticFills');
        if (realistic) payload.realistic_fills = realistic.checked;
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

    renderDeskHeader(report);
    renderSyntheticPricingNote(report);
    renderMetrics(report.summary || {});
    renderPendingSignals(report.pending_signals || []);
    renderEquityChart(report);
    renderDrawdownChart(report.drawdown_series || []);
    renderOosFolds(report);
    renderReweightChart(report);
    renderGreeksChart(report);
    renderPodAllocation(report);
    renderStructures(report);
    renderTrades(report.trades || []);
    renderTraderNotes(report);
    renderProvenance(report.data_sources || {});
}

/**
 * Estimated-pricing disclaimer (Phase 8): every janestreet report gets the
 * info callout shipped in backtest.html — backtest option prices are
 * SYNTHETIC (Black-Scholes from the underlying's history; free providers
 * carry no historical chains), so the desk's P&L validates LOGIC, not
 * executable prices. Real chain pricing arrives via E*TRADE in Phase 9.
 * The markup is static in the template; this only toggles visibility, so
 * every other desk and strategy mode render exactly as before.
 */
function renderSyntheticPricingNote(report) {
    const note = document.getElementById('syntheticPricingNote');
    const janestreet = Boolean(report.desk && report.desk.key === 'janestreet');
    note.classList.toggle('d-none', !janestreet);
}

/* ==========================================================================
   Desk results (Phase 5): header chip + trader's-notes timeline
   ========================================================================== */

function renderDeskHeader(report) {
    const row = document.getElementById('deskChipRow');
    const desk = report.desk;
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
        `${escapeHTML(desk.name || desk.key)} desk</span>` +
        regimeChipHTML(report);
}

/**
 * Regime entries with a whitelisted state (contract C5). Anything else —
 * key absent, [], or an unknown state — drops out, so foundation/strategy
 * runs render no regime UI at all.
 */
function regimeSeries(report) {
    return Array.isArray(report.regime_series)
        ? report.regime_series.filter(
            (r) => r && Object.prototype.hasOwnProperty.call(REGIME_META, r.state))
        : [];
}

/** 'Current regime' chip: the final entry's state + its max probability. */
function regimeChipHTML(report) {
    const series = regimeSeries(report);
    if (series.length === 0) return '';
    const last = series[series.length - 1];
    const meta = REGIME_META[last.state]; // whitelisted color -> inline style
    const probs = (last.probs && typeof last.probs === 'object')
        ? Object.values(last.probs).map(Number).filter(Number.isFinite)
        : [];
    const conf = probs.length > 0
        ? ` · ${(Math.max(...probs) * 100).toFixed(0)}%` : '';
    return `<span class="regime-chip" style="--regime-color: ${meta.color};">` +
        '<i class="bi bi-activity" aria-hidden="true"></i>' +
        `Current regime: ${escapeHTML(meta.label)}${conf}</span>`;
}

function noteCategory(note) {
    return NOTE_CATEGORIES.includes(note.category) ? note.category : 'info';
}

/** note.data.book (contracts C7/C14): any non-blank string, or null.
 *  Phase 8 generalized this beyond the renaissance whitelist so new desks'
 *  books (vrp/earnings/relative_value, and whatever comes next) get chips
 *  without frontend changes. Book keys only ever render as escaped
 *  text/textContent; bookClass() keeps CSS suffixes whitelisted. */
function noteBook(note) {
    const data = note.data;
    return (data && typeof data === 'object' &&
            typeof data.book === 'string' && data.book.trim() !== '')
        ? data.book : null;
}

/** Display label for a book key: known label, else underscores as spaces. */
function bookLabel(key) {
    return Object.prototype.hasOwnProperty.call(NOTE_BOOKS, key)
        ? NOTE_BOOKS[key] : String(key).replace(/_/g, ' ');
}

/** 'note-book-<key>' for KNOWN books only — arbitrary backend strings must
 *  never become CSS class suffixes. Unknown books get '' (default chip). */
function bookClass(key) {
    return Object.prototype.hasOwnProperty.call(NOTE_BOOKS, key)
        ? `note-book-${key}` : '';
}

/** note.data.structure_id (contract C14): non-blank string, or null. Only
 *  ever rendered as escaped text (a small monospace tag on the note). */
function noteStructureId(note) {
    const data = note.data;
    return (data && typeof data === 'object' &&
            typeof data.structure_id === 'string' &&
            data.structure_id.trim() !== '')
        ? data.structure_id : null;
}

/** note.data.pod (contract C9): any non-blank string, or null. Pod keys
 *  are NOT whitelisted — they only ever render as escaped text/textContent
 *  and never become CSS class suffixes (colors come from POD_PALETTE). */
function notePod(note) {
    const data = note.data;
    return (data && typeof data === 'object' &&
            typeof data.pod === 'string' && data.pod.trim() !== '')
        ? data.pod : null;
}

/** Display label for a pod key: underscores read as spaces. */
function podLabel(key) {
    return String(key).replace(/_/g, ' ');
}

/** Palette hue for a pod key; assigns the next color on first sight.
 *  A Map (not a plain object) so hostile keys like '__proto__' stay inert;
 *  the ACCENT_RE check keeps anything else out of inline styles. */
function podColorFor(key) {
    if (!podColors.has(key)) {
        podColors.set(key, POD_PALETTE[podColors.size % POD_PALETTE.length]);
    }
    const color = podColors.get(key);
    return ACCENT_RE.test(color) ? color : WF_COLOR;
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
    bookFilter = 'all';
    podFilter = 'all';
    paintNoteFilters();
    paintBookFilters();
    paintPodFilters();
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

/**
 * Second filter-chip row (contracts C7/C14): one chip per DISTINCT book
 * present in note.data.book, composing with the category filter. Hidden
 * entirely when no note is book-tagged (foundation runs stay byte-identical
 * on screen). Phase 8: book keys are no longer whitelisted, so the chips
 * are built with the DOM API (textContent/dataset) — never HTML
 * interpolation; known books keep their color class and lead the row in
 * NOTE_BOOKS declaration order, unknown books follow first-seen.
 */
function paintBookFilters() {
    const wrap = document.getElementById('noteBookFilters');
    wrap.textContent = '';
    const counts = new Map(); // insertion order == first appearance
    currentNotes.forEach((n) => {
        const book = noteBook(n);
        if (book) counts.set(book, (counts.get(book) || 0) + 1);
    });
    if (counts.size === 0) {
        wrap.classList.add('d-none');
        return;
    }
    wrap.classList.remove('d-none');
    const known = Object.keys(NOTE_BOOKS).filter((b) => counts.has(b));
    const present = known.concat(
        Array.from(counts.keys()).filter((b) => !known.includes(b)));
    const chips = [['all', `All books (${currentNotes.length})`]].concat(
        present.map((b) => [b, `${bookLabel(b)} (${counts.get(b)})`]));
    chips.forEach(([value, label]) => {
        const btn = document.createElement('button');
        btn.type = 'button';
        btn.className = 'note-filter-chip';
        if (value !== 'all' && bookClass(value)) {
            btn.classList.add(bookClass(value)); // whitelisted suffix only
        }
        if (value === bookFilter) btn.classList.add('active');
        btn.setAttribute('aria-pressed', String(value === bookFilter));
        btn.dataset.book = value;
        btn.textContent = label;
        btn.addEventListener('click', () => {
            bookFilter = btn.dataset.book;
            paintBookFilters();
            paintNotes();
        });
        wrap.appendChild(btn);
    });
}

/**
 * Third filter-chip row (contract C9): one chip per pod present in
 * note.data.pod, composing with the category (and book) filters. Hidden
 * entirely when no note is pod-tagged, so non-citadel runs render exactly
 * as before. Pod keys are arbitrary backend strings, so the chips are
 * built with the DOM API (textContent/dataset) — never HTML interpolation.
 */
function paintPodFilters() {
    const wrap = document.getElementById('notePodFilters');
    wrap.textContent = '';
    const counts = new Map(); // insertion order == first appearance
    currentNotes.forEach((n) => {
        const pod = notePod(n);
        if (pod) counts.set(pod, (counts.get(pod) || 0) + 1);
    });
    if (counts.size === 0) {
        wrap.classList.add('d-none');
        return;
    }
    wrap.classList.remove('d-none');
    const chips = [['all', `All pods (${currentNotes.length})`]].concat(
        Array.from(counts, ([pod, n]) => [pod, `${podLabel(pod)} (${n})`]));
    chips.forEach(([value, label]) => {
        const btn = document.createElement('button');
        btn.type = 'button';
        btn.className = 'note-filter-chip';
        if (value !== 'all') {
            btn.classList.add('note-pod-chip');
            btn.style.setProperty('--pod-color', podColorFor(value));
        }
        if (value === podFilter) btn.classList.add('active');
        btn.setAttribute('aria-pressed', String(value === podFilter));
        btn.dataset.pod = value;
        btn.textContent = label;
        btn.addEventListener('click', () => {
            podFilter = btn.dataset.pod;
            paintPodFilters();
            paintNotes();
        });
        wrap.appendChild(btn);
    });
}

function noteItem(note) {
    const category = noteCategory(note);
    const book = noteBook(note); // free-form key (escaped) or null
    const pod = notePod(note);   // free-form key (escaped) or null
    const structureId = noteStructureId(note); // escaped-text-only or null
    const data = note.data && typeof note.data === 'object' ? note.data : null;
    const hasData = data !== null && Object.keys(data).length > 0;
    return (
        '<li class="note-entry">' +
        `<span class="note-ts num">${escapeHTML(note.timestamp ?? '—')}</span>` +
        `<span class="note-cat-chip note-cat-${category}">${escapeHTML(category)}</span>` +
        (book
            // bookClass() yields a whitelisted suffix or '' — the key itself
            // only reaches the DOM escaped, inside the label.
            ? `<span class="note-cat-chip${bookClass(book) ? ` ${bookClass(book)}` : ''}">` +
              `${escapeHTML(bookLabel(book))}</span>`
            : '') +
        (structureId
            // Structure lifecycle tag (contract C14): small monospace id.
            ? `<span class="note-structure-tag num">${escapeHTML(structureId)}</span>`
            : '') +
        (pod
            // Fixed class + palette-only --pod-color; the key itself is
            // escaped text and never reaches a class or style.
            ? `<span class="note-cat-chip note-pod-chip" style="--pod-color: ${podColorFor(pod)};">` +
              `${escapeHTML(podLabel(pod))}</span>`
            : '') +
        `<span class="note-msg">${escapeHTML(note.message ?? '')}</span>` +
        (hasData
            ? '<details class="note-data"><summary>data</summary>' +
              `<pre class="note-data-json">${escapeHTML(JSON.stringify(data, null, 2))}</pre>` +
              '</details>'
            : '') +
        '</li>'
    );
}

function noteVisible(note) {
    return (noteFilter === 'all' || noteCategory(note) === noteFilter) &&
           (bookFilter === 'all' || noteBook(note) === bookFilter) &&
           (podFilter === 'all' || notePod(note) === podFilter);
}

/** Funnel empty state naming whichever filters are active. */
function filteredEmptyStateHTML() {
    const labels = [];
    if (noteFilter !== 'all') labels.push(noteFilter);
    if (bookFilter !== 'all') labels.push(bookLabel(bookFilter));
    if (podFilter !== 'all') labels.push(podLabel(podFilter));
    return emptyStateHTML('bi-funnel', `No ${labels.join(' · ')} notes`,
        bookFilter === 'all' && podFilter === 'all'
            ? 'Pick another category filter.'
            : 'Pick another filter combination.');
}

function paintNotes() {
    const list = document.getElementById('notesTimeline');
    const emptyEl = document.getElementById('notesEmpty');
    const visible = currentNotes.filter(noteVisible);

    if (visible.length === 0) {
        list.innerHTML = '';
        emptyEl.classList.remove('d-none');
        emptyEl.innerHTML = currentNotes.length === 0
            ? emptyStateHTML('bi-journal-text', 'No trader notes',
                'The desk logged nothing for this run.')
            : filteredEmptyStateHTML();
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
    // Research integrity (Phase 3): PSR / deflated Sharpe are probabilities in
    // [0,1] shown as percents; >50% leans real (green), <50% leans luck (red).
    renderProbabilityMetric('mPSR', summary.psr);
    renderProbabilityMetric('mDeflatedSharpe', summary.deflated_sharpe);
    const trialsEl = document.getElementById('mDeflatedTrials');
    const nTrials = Number(summary.n_trials);
    if (Number.isFinite(nTrials) && nTrials > 0) {
        trialsEl.textContent = `(${nTrials} trial${nTrials === 1 ? '' : 's'})`;
        // Be honest in the UI: n_trials is a proxy, not a true independent count.
        trialsEl.title = 'n_trials = walk-forward refit count — a conservative '
            + 'proxy for multiple testing. Overlapping train windows (and, in a '
            + 'fund, summing refits across desks) over-count independent trials, '
            + 'so the deflated Sharpe errs low.';
    } else {
        trialsEl.textContent = '';
        trialsEl.removeAttribute('title');
    }
}

/** A probability metric (PSR/DSR): em dash when null, else NN.N% colored by
 *  whether it clears the 50% line. */
function renderProbabilityMetric(id, value) {
    if (value === null || value === undefined) {
        setMetric(id, '—');
        return;
    }
    const p = Number(value);
    setMetric(id, `${(p * 100).toFixed(1)}%`, p - 0.5);
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
    addRegimeBands(report, traces, layout);
    addWalkForwardMarkers(report, history, traces, layout);
    Plotly.react(document.getElementById('equityChart'), traces, layout, PLOT_CONFIG);
}

/**
 * Regime background bands (contract C5): one translucent rect per run of
 * consecutive same-state dates, drawn on layer 'below' so the equity and
 * benchmark traces stay legible, plus one legend swatch per state present.
 * Skips cleanly when regime_series is absent/empty (foundation/strategy).
 */
function addRegimeBands(report, traces, layout) {
    const series = regimeSeries(report);
    if (series.length === 0) return;

    // Merge consecutive same-state entries; each band closes where the next
    // one opens so the covered span tiles without gaps.
    const bands = [];
    series.forEach((entry) => {
        const prev = bands[bands.length - 1];
        if (prev && prev.state === entry.state) {
            prev.end = entry.date;
        } else {
            if (prev) prev.end = entry.date;
            bands.push({ state: entry.state, start: entry.date, end: entry.date });
        }
    });

    layout.shapes = (layout.shapes || []).concat(bands.map((band) => ({
        type: 'rect', xref: 'x', yref: 'paper', layer: 'below',
        x0: band.start, x1: band.end, y0: 0, y1: 1,
        fillcolor: REGIME_META[band.state].band,
        line: { width: 0 },
    })));

    // Shapes never reach the legend — invisible marker traces carry one
    // swatch per state present, in REGIME_META display order.
    const present = new Set(series.map((entry) => entry.state));
    Object.keys(REGIME_META).filter((state) => present.has(state))
        .forEach((state) => {
            const meta = REGIME_META[state];
            traces.push({
                type: 'scatter', mode: 'markers', x: [null], y: [null],
                name: `${meta.label} regime`,
                marker: { symbol: 'square', size: 9, color: meta.band,
                          line: { width: 1, color: meta.color } },
                hoverinfo: 'skip', showlegend: true,
            });
        });
}

/** Marker color for one walk-forward fit: per-model hue, legacy fallback. */
function wfColor(wf) {
    return WF_MODEL_COLORS[wf.model] || WF_COLOR;
}

/**
 * Walk-forward refit markers (desk mode, contract C3): a vertical dashed
 * line per report.walk_forward entry at its fit_date, plus legend-bearing
 * marker traces whose hover text describes the training window. Phase 6:
 * fits MAY carry 'model' ('regime'|'stat_arb'|'pairs') — those are grouped
 * into one color-coded legend entry per model, while untagged fits keep the
 * legacy purple 'Walk-forward refit' entry (foundation back-compat). Skips
 * cleanly when the list is empty/absent (strategy runs) or there is no
 * equity curve to anchor the hover markers to.
 */
function addWalkForwardMarkers(report, history, traces, layout) {
    const refits = Array.isArray(report.walk_forward) ? report.walk_forward : [];
    if (refits.length === 0 || history.length === 0) return;

    layout.shapes = (layout.shapes || []).concat(refits.map((wf) => ({
        type: 'line', xref: 'x', yref: 'paper',
        x0: wf.fit_date, x1: wf.fit_date, y0: 0, y1: 1,
        line: { color: wfColor(wf), width: 1, dash: 'dash' },
    })));

    const top = Math.max(...history.map((h) => h.portfolio_value));
    const groups = []; // insertion-ordered: [{key, label, color, fits}]
    refits.forEach((wf) => {
        // Only whitelisted models get their own group; anything else
        // (absent or unknown) falls back to the legacy uncolored entry.
        const key = WF_MODEL_COLORS[wf.model] ? wf.model : '';
        let group = groups.find((g) => g.key === key);
        if (!group) {
            group = {
                key,
                color: key ? WF_MODEL_COLORS[key] : WF_COLOR,
                label: key ? `Refit · ${NOTE_BOOKS[key] || key}`
                           : 'Walk-forward refit',
                fits: [],
            };
            groups.push(group);
        }
        group.fits.push(wf);
    });

    groups.forEach((group) => {
        traces.push({
            type: 'scatter', name: group.label, mode: 'markers',
            x: group.fits.map((wf) => wf.fit_date),
            y: group.fits.map(() => top),
            marker: { symbol: 'line-ns-open', size: 9, color: group.color,
                      line: { width: 1.5, color: group.color } },
            text: group.fits.map((wf) =>
                `Refit${group.key ? ` [${group.key}]` : ''}: ` +
                `trained ${wf.train_start} → ${wf.train_end} ` +
                `(${wf.n_samples} rows)`),
            hovertemplate: '%{text}<extra></extra>',
        });
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

/* ==========================================================================
   Jane Street portfolio Greeks (Phase 8, contract C12): one entry per
   simulated day in report.greeks_series. AXIS CONVENTION (the backend's
   desks/options_pricing.py black_scholes_greeks conventions: per-share
   greeks with theta PER CALENDAR DAY and vega PER 1.00 VOL POINT, summed
   at desk level across open option legs x signed quantity x the 100 share
   multiplier): values are DESK-LEVEL DOLLAR AGGREGATES — delta is $ P&L
   per $1 underlying move, gamma is the delta change per $1 move, theta is
   $ per calendar day (POSITIVE for the desk's short-premium structures:
   time decay is collected, not paid), vega is $ per 1.0 vol-point
   (negative when net short vol). All four are exactly 0.0 on option-free
   days, so a flat line at zero reads as 'no options on'. One shared $
   y-axis: delta and vega carry the story and render as primary lines;
   gamma and theta (different magnitudes) start legend-toggled off
   ('legendonly' — click the legend to show them). A dotted zero-line
   anchors sign flips. Absent cleanly otherwise.
   ========================================================================== */

/** Contract-C12-shaped greeks_series entries; [] when absent/empty. */
function greeksSeries(report) {
    return Array.isArray(report.greeks_series)
        ? report.greeks_series.filter(
            (g) => g && typeof g.date === 'string')
        : [];
}

function renderGreeksChart(report) {
    const card = document.getElementById('greeksCard');
    const el = document.getElementById('greeksChart');
    const series = greeksSeries(report);
    if (series.length === 0) {
        card.classList.add('d-none');
        // Plotly.purge (not innerHTML='') so a later janestreet run can
        // re-plot into the same div without stale internal state.
        Plotly.purge(el);
        return;
    }
    card.classList.remove('d-none');

    const t = plotlyTheme();
    const dates = series.map((g) => g.date);
    // Delta=accent blue, vega=vol gold (the desk accent), gamma=pod purple,
    // theta=teal — fixed hues so legend-toggling stays color-stable.
    const metas = [
        { key: 'delta', label: 'Delta', color: t.accent, primary: true },
        { key: 'vega',  label: 'Vega',  color: '#d29922', primary: true },
        { key: 'gamma', label: 'Gamma', color: '#bc8cff', primary: false },
        { key: 'theta', label: 'Theta', color: '#39c5cf', primary: false },
    ];
    const traces = metas.map((m) => ({
        type: 'scatter', name: m.label, mode: 'lines',
        x: dates,
        y: series.map((g) => {
            const v = Number(g[m.key]);
            return Number.isFinite(v) ? v : 0;
        }),
        line: { color: m.color, width: m.primary ? 1.5 : 1.25,
                dash: m.primary ? 'solid' : 'dot' },
        visible: m.primary ? true : 'legendonly',
        hovertemplate: `${m.label}: %{y:,.2f}<extra></extra>`,
    }));

    const layout = baseLayout(t, 240);
    layout.yaxis.title = { text: 'Greeks ($)', font: { size: 10, color: t.muted } };
    // Zero-line reference: sign is the signal (short vega / long delta).
    layout.shapes = [{
        type: 'line', xref: 'paper', yref: 'y',
        x0: 0, x1: 1, y0: 0, y1: 0,
        line: { color: t.muted, width: 1, dash: 'dot' },
    }];
    Plotly.react(el, traces, layout, PLOT_CONFIG);
}

/* ==========================================================================
   Fund dynamic reweighter: desk capital weights over the rebalance schedule.
   report.reweight_log is [{date, day_number, weights:{desk:fraction},
   fallback, degraded_desks}], one entry per rebalance (fund-mode runs only).
   Rendered as a stacked-area of weights (x100) with dotted markers on
   rebalances that degenerated to equal weight. Absent cleanly otherwise.
   ========================================================================== */

/** Color for a desk key: its registry accent when known, else a palette hue. */
function deskColorFor(key) {
    const meta = desksByKey[key];
    if (meta && ACCENT_RE.test(meta.accent || '')) return meta.accent;
    return podColorFor(key); // Map-backed palette fallback (hostile-key safe)
}

/** Display label for a desk key: its registry name, else underscores->spaces. */
function deskLabel(key) {
    const meta = desksByKey[key];
    return (meta && meta.name) ? meta.name : String(key).replace(/_/g, ' ');
}

/** Contract-shaped reweight_log entries; [] when absent/empty/foreign. */
/**
 * Per-fold OOS significance (Phase 3 Step 4). Paints report.oos_folds as a
 * table — one row per walk-forward fold with its one-sided t-stat, p-value and
 * Bonferroni/BH verdicts — plus a summary line carrying the honest "heuristic,
 * not FWER" caveat. Hidden entirely for strategy/legacy runs (no key); shows an
 * N/A note in fund mode (netted multi-desk book). Mirrors the conditional
 * card-toggle pattern used by renderReweightChart.
 */
function renderOosFolds(report) {
    const card = document.getElementById('oosFoldsCard');
    const body = document.getElementById('oosFoldsBody');
    const caption = document.getElementById('oosFoldsCaption');
    const summary = document.getElementById('oosFoldsSummary');
    const data = report && report.oos_folds;
    if (!data || typeof data !== 'object') {
        card.classList.add('d-none');
        body.innerHTML = '';
        return;
    }
    card.classList.remove('d-none');

    if (data.available === false) {
        caption.textContent = '(fund mode)';
        body.innerHTML = '';
        summary.textContent = 'Not available for a fund: ' +
            (data.reason || 'per-fold OOS significance requires a single desk.');
        return;
    }

    const folds = Array.isArray(data.folds) ? data.folds : [];
    const alphaPct = Number.isFinite(Number(data.alpha))
        ? `${(Number(data.alpha) * 100).toFixed(0)}%` : '5%';
    caption.textContent = `(one-sided t-test · α=${alphaPct})`;

    if (folds.length === 0) {
        body.innerHTML =
            '<tr><td colspan="8" class="text-muted">No walk-forward folds.</td></tr>';
    } else {
        body.innerHTML = folds.map((f) => {
            const window = f.oos_end
                ? `${escapeHTML(f.oos_start)} → ${escapeHTML(f.oos_end)}`
                : `${escapeHTML(f.oos_start)} → end`;
            const mean = (f.mean_return === null || f.mean_return === undefined)
                ? '—' : `${(Number(f.mean_return) * 100).toFixed(3)}%`;
            const tstat = (f.tstat === null || f.tstat === undefined)
                ? '—' : Number(f.tstat).toFixed(2);
            const pval = (f.pvalue === null || f.pvalue === undefined)
                ? '—' : Number(f.pvalue).toFixed(4);
            return '<tr>' +
                `<td class="num">${escapeHTML(f.fit_date)}</td>` +
                `<td class="num">${window}</td>` +
                `<td class="text-end num">${Number(f.n_returns) || 0}</td>` +
                `<td class="text-end num">${mean}</td>` +
                `<td class="text-end num">${tstat}</td>` +
                `<td class="text-end num">${pval}</td>` +
                `<td>${oosBadge(f.significant_bonferroni)}</td>` +
                `<td>${oosBadge(f.significant_bh)}</td>` +
                '</tr>';
        }).join('');
    }

    const m = Number(data.n_testable_folds) || 0;
    const bonfAlpha = (data.bonferroni_alpha === null ||
                       data.bonferroni_alpha === undefined)
        ? '—' : Number(data.bonferroni_alpha).toFixed(4);
    summary.textContent =
        `${data.n_significant_bh || 0}/${m} folds significant (BH), ` +
        `${data.n_significant_bonferroni || 0}/${m} (Bonferroni, α/m=${bonfAlpha}). ` +
        `${data.caveat || ''}`;
}

function oosBadge(flag) {
    return flag
        ? '<span class="badge bg-success">✓</span>'
        : '<span class="badge bg-secondary">·</span>';
}

function reweightLog(report) {
    return Array.isArray(report.reweight_log)
        ? report.reweight_log.filter(
            (e) => e && typeof e.date === 'string' &&
                   e.weights && typeof e.weights === 'object')
        : [];
}

function renderReweightChart(report) {
    const card = document.getElementById('reweightCard');
    const el = document.getElementById('reweightChart');
    const log = reweightLog(report);
    if (log.length === 0) {
        card.classList.add('d-none');
        // Plotly.purge (not innerHTML='') so a later fund run can re-plot
        // into the same div without stale internal state.
        Plotly.purge(el);
        return;
    }
    card.classList.remove('d-none');

    // Desks in first-seen order across the log.
    const keys = [];
    log.forEach((e) => Object.keys(e.weights).forEach((k) => {
        if (!keys.includes(k)) keys.push(k);
    }));

    const t = plotlyTheme();
    const dates = log.map((e) => e.date);
    const traces = keys.map((key) => {
        const color = deskColorFor(key);
        const label = escapeHTML(deskLabel(key));
        return {
            type: 'scatter', name: label, mode: 'lines',
            x: dates,
            y: log.map((e) => {
                const w = Number(e.weights[key]);
                return Number.isFinite(w) ? w * 100 : 0;
            }),
            stackgroup: 'desks',
            line: { width: 0.75, color },
            fillcolor: hexToRGBA(color, 0.45),
            hovertemplate: `${label}: %{y:.1f}%<extra></extra>`,
        };
    });

    const layout = baseLayout(t, 260);
    layout.yaxis.title = { text: 'Weight (%)', font: { size: 10, color: t.muted } };
    layout.yaxis.range = [0, 100];
    addReweightFallbackMarkers(log, traces, layout);
    Plotly.react(el, traces, layout, PLOT_CONFIG);

    const fallbacks = log.filter((e) => e.fallback).length;
    document.getElementById('reweightCaption').textContent =
        `(${log.length} rebalance${log.length === 1 ? '' : 's'} · ` +
        'risk-parity weights' +
        (fallbacks > 0 ? ` · ${fallbacks} fell back to equal weight` : '') + ')';
}

/**
 * Dotted vertical markers on the rebalances that fell back to equal weight
 * (a desk's standalone curve was missing/flat/too short), plus one legend
 * entry whose hover names the degenerate desks. Same anatomy as the
 * walk-forward / pod-event markers.
 */
function addReweightFallbackMarkers(log, traces, layout) {
    const fallbacks = log.filter((e) => e.fallback);
    if (fallbacks.length === 0) return;
    layout.shapes = (layout.shapes || []).concat(fallbacks.map((e) => ({
        type: 'line', xref: 'x', yref: 'paper',
        x0: e.date, x1: e.date, y0: 0, y1: 1,
        line: { color: '#d29922', width: 1, dash: 'dot' },
    })));
    traces.push({
        type: 'scatter', name: 'Equal-weight fallback', mode: 'markers',
        x: fallbacks.map((e) => e.date),
        y: fallbacks.map(() => 100),
        marker: { symbol: 'line-ns-open', size: 9, color: '#d29922',
                  line: { width: 1.5, color: '#d29922' } },
        text: fallbacks.map((e) => {
            const desks = Array.isArray(e.degraded_desks)
                ? e.degraded_desks.map(deskLabel).join(', ') : '';
            return `Equal-weight fallback${desks ? `: ${desks} degenerate` : ''}`;
        }),
        hovertemplate: '%{text}<extra></extra>',
    });
}

/* ==========================================================================
   Citadel pods (Phase 7, contract C8): the capital-allocation story —
   per-pod status cards + a stacked pod-weight area chart with reallocation
   and probation/cut event markers. Absent cleanly for every other run.
   ========================================================================== */

/** Contract-C8-shaped pod_history entries; [] when absent/empty/foreign. */
function podHistory(report) {
    return Array.isArray(report.pod_history)
        ? report.pod_history.filter((entry) =>
            entry && typeof entry.date === 'string' &&
            entry.pods && typeof entry.pods === 'object')
        : [];
}

/** 'rgba(r, g, b, alpha)' from a strict #rrggbb palette color. */
function hexToRGBA(hex, alpha) {
    const n = parseInt(hex.slice(1), 16);
    return `rgba(${(n >> 16) & 255}, ${(n >> 8) & 255}, ${n & 255}, ${alpha})`;
}

function renderPodAllocation(report) {
    const card = document.getElementById('podCard');
    const cards = document.getElementById('podCards');
    const chart = document.getElementById('podAllocChart');
    podColors = new Map(); // fresh hue assignment per rendered report
    const history = podHistory(report);
    if (history.length === 0) {
        card.classList.add('d-none');
        cards.innerHTML = '';
        // Plotly.purge (not innerHTML='') so a later citadel run can
        // re-plot into the same div without stale internal state.
        Plotly.purge(chart);
        return;
    }
    card.classList.remove('d-none');

    // Pods in first-seen order; podColorFor pins each to a palette hue that
    // the cards, the chart bands, and the note tags all share.
    const keys = [];
    history.forEach((entry) => {
        Object.keys(entry.pods).forEach((key) => {
            if (!keys.includes(key)) keys.push(key);
        });
    });
    keys.forEach((key) => podColorFor(key));

    renderPodCards(cards, keys, history);
    renderPodChart(chart, keys, history);
}

/**
 * Per-pod summary cards (contract C8): name, final weight/NAV/status from
 * the LAST pod_history entry, max drawdown = the most negative
 * drawdown_pct (x100, <= 0) the pod printed across the run. Rendered in
 * final-weight-descending order — the book of capital, biggest pod first.
 */
function renderPodCards(container, keys, history) {
    const last = history[history.length - 1].pods;
    const rows = keys.map((key) => {
        const final = (last[key] && typeof last[key] === 'object') ? last[key] : {};
        let minDD = 0;
        history.forEach((entry) => {
            const pod = entry.pods[key];
            const dd = Number(pod && pod.drawdown_pct);
            if (Number.isFinite(dd) && dd < minDD) minDD = dd;
        });
        return { key, weight: Number(final.weight) || 0,
                 nav: final.nav, status: final.status, minDD };
    });
    rows.sort((a, b) => b.weight - a.weight);

    container.innerHTML = rows.map((row) => {
        const status = POD_STATUS_META[row.status] || { label: '—', cls: '' };
        return '<div class="col-6 col-md-4 col-xl-3">' +
            `<div class="metric-card pod-card" style="--pod-color: ${podColorFor(row.key)};">` +
            '<div class="d-flex justify-content-between align-items-baseline gap-2">' +
            `<div class="pod-name text-truncate">${escapeHTML(podLabel(row.key))}</div>` +
            `<span class="pod-status-badge ${status.cls}">${escapeHTML(status.label)}</span>` +
            '</div>' +
            `<div class="metric-value num">${fmtPct(row.weight)}</div>` +
            `<div class="pod-detail num">NAV ${fmtMoney(row.nav)} · ` +
            `max DD ${row.minDD.toFixed(1)}%</div>` +
            '</div></div>';
    }).join('');
}

/**
 * Stacked area of pod weights over time: one band per pod on a fixed
 * 0-100% axis. Weights arrive as fractions (contract C8) and are plotted
 * x100; vertical markers flag the days capital actually moved.
 */
function renderPodChart(el, keys, history) {
    const t = plotlyTheme();
    const dates = history.map((entry) => entry.date);
    const traces = keys.map((key) => {
        const color = podColorFor(key);
        const label = escapeHTML(podLabel(key));
        return {
            type: 'scatter', name: label, mode: 'lines',
            x: dates,
            y: history.map((entry) => {
                const pod = entry.pods[key];
                const w = Number(pod && pod.weight);
                return Number.isFinite(w) ? w * 100 : 0;
            }),
            stackgroup: 'pods',
            line: { width: 0.75, color },
            fillcolor: hexToRGBA(color, 0.45),
            hovertemplate: `${label}: %{y:.1f}%<extra></extra>`,
        };
    });

    const layout = baseLayout(t, 260);
    layout.yaxis.title = { text: 'Weight (%)', font: { size: 10, color: t.muted } };
    layout.yaxis.range = [0, 100];
    addPodEventMarkers(history, traces, layout);
    Plotly.react(el, traces, layout, PLOT_CONFIG);
}

/**
 * Pod lifecycle events derived from consecutive pod_history entries:
 * a weight step-change on any pod is one 'realloc' event for that date,
 * and a status downgrade to probation/cut is its own event per pod.
 */
function podEvents(history) {
    const events = [];
    for (let i = 1; i < history.length; i++) {
        const prev = history[i - 1].pods;
        const cur = history[i].pods;
        const date = history[i].date;
        const shifts = [];
        Object.keys(cur).forEach((key) => {
            const before = prev[key];
            const after = cur[key];
            if (!before || !after) return;
            const w0 = Number(before.weight);
            const w1 = Number(after.weight);
            if (Number.isFinite(w0) && Number.isFinite(w1) &&
                Math.abs(w1 - w0) > 1e-6) {
                shifts.push(`${podLabel(key)} ${(w0 * 100).toFixed(0)}%` +
                            `→${(w1 * 100).toFixed(0)}%`);
            }
            if (after.status !== before.status &&
                POD_EVENT_META[after.status]) {
                const dd = Number(after.drawdown_pct);
                events.push({
                    date,
                    kind: after.status,
                    text: `Pod ${podLabel(key)} → ${after.status.toUpperCase()}` +
                          (Number.isFinite(dd)
                              ? ` (drawdown ${dd.toFixed(1)}%)` : ''),
                });
            }
        });
        if (shifts.length > 0) {
            events.push({ date, kind: 'realloc',
                          text: `Reallocation: ${shifts.join(', ')}` });
        }
    }
    return events;
}

/**
 * Vertical event markers on the pod chart: a dashed line per reallocation
 * and a dotted line per probation/cut, plus one legend-bearing marker
 * trace per kind whose hover text narrates the event (same anatomy as the
 * walk-forward refit markers on the equity chart).
 */
function addPodEventMarkers(history, traces, layout) {
    const events = podEvents(history);
    if (events.length === 0) return;

    layout.shapes = (layout.shapes || []).concat(events.map((ev) => ({
        type: 'line', xref: 'x', yref: 'paper',
        x0: ev.date, x1: ev.date, y0: 0, y1: 1,
        line: { color: POD_EVENT_META[ev.kind].color, width: 1,
                dash: ev.kind === 'realloc' ? 'dash' : 'dot' },
    })));

    Object.keys(POD_EVENT_META).forEach((kind) => {
        const ofKind = events.filter((ev) => ev.kind === kind);
        if (ofKind.length === 0) return;
        const meta = POD_EVENT_META[kind];
        traces.push({
            type: 'scatter', name: meta.label, mode: 'markers',
            x: ofKind.map((ev) => ev.date),
            y: ofKind.map(() => 100),
            marker: { symbol: 'line-ns-open', size: 9, color: meta.color,
                      line: { width: 1.5, color: meta.color } },
            text: ofKind.map((ev) => escapeHTML(ev.text)),
            hovertemplate: '%{text}<extra></extra>',
        });
    });
}

/* ==========================================================================
   Jane Street structures (Phase 8, contract C11): a sortable table of
   defined-risk option structures — type, underlying, contracts, lifecycle
   dates, credit received, max loss (capped at entry by construction), P&L,
   status badge, close-reason chip — with an expandable per-row legs detail
   (full option-contract instrument strings). Every cell is built with the
   DOM API and textContent: structure ids, underlyings, and instrument
   strings are arbitrary backend text and never reach innerHTML. Absent
   cleanly for every non-janestreet run.
   ========================================================================== */

/** Contract-C11-shaped structures; [] when absent/empty/foreign. Each kept
 *  entry is stamped with a stable _key for the expanded-legs Set. */
function reportStructures(report) {
    if (!Array.isArray(report.structures)) return [];
    return report.structures
        .filter((s) => s && typeof s === 'object')
        .map((s, i) => ({
            ...s,
            _key: (typeof s.id === 'string' && s.id !== '')
                ? `id:${s.id}` : `idx:${i}`,
        }));
}

function renderStructures(report) {
    const card = document.getElementById('structuresCard');
    const body = document.getElementById('structuresBody');
    currentStructures = reportStructures(report);
    expandedStructures = new Set();
    if (currentStructures.length === 0) {
        card.classList.add('d-none');
        body.textContent = '';
        return;
    }
    card.classList.remove('d-none');
    structSort = { key: 'opened', dir: 1 };
    sortAndPaintStructures();
}

/** <td> with textContent (never markup) and optional classes. */
function structCell(text, cls) {
    const td = document.createElement('td');
    if (cls) td.className = cls;
    td.textContent = text;
    return td;
}

/** Display label for a structure type: whitelisted, else underscores read
 *  as spaces (escaped text only). */
function structureTypeLabel(type) {
    return STRUCTURE_TYPE_LABELS[type] ||
        String(type ?? '—').replace(/_/g, ' ');
}

function sortAndPaintStructures() {
    const { key, dir } = structSort;
    const sorted = currentStructures.slice().sort((a, b) => {
        const av = a[key]; const bv = b[key];
        if (av === bv) return 0;
        if (av === undefined || av === null) return 1;
        if (bv === undefined || bv === null) return -1;
        return (av < bv ? -1 : 1) * dir;
    });

    const body = document.getElementById('structuresBody');
    body.textContent = '';
    sorted.forEach((s) => {
        const legsRow = structureLegsRow(s);
        body.append(structureRow(s, legsRow), legsRow);
    });

    document.querySelectorAll('#structuresTable th.sortable').forEach((th) => {
        const arrow = th.querySelector('.sort-arrow');
        const active = th.dataset.key === key;
        arrow.textContent = active ? (dir > 0 ? '▲' : '▼') : '';
        th.setAttribute('aria-sort',
            active ? (dir > 0 ? 'ascending' : 'descending') : 'none');
    });
}

/** Main table row; the toggle button shows/hides the paired legs row. */
function structureRow(s, legsRow) {
    const tr = document.createElement('tr');
    tr.className = 'structure-row';

    // Legs expander (chevron) — only when there is leg detail to show.
    const tdToggle = document.createElement('td');
    tdToggle.className = 'structure-toggle-cell';
    if (Array.isArray(s.legs) && s.legs.length > 0) {
        const expanded = expandedStructures.has(s._key);
        const btn = document.createElement('button');
        btn.type = 'button';
        btn.className = `structure-toggle${expanded ? ' open' : ''}`;
        btn.setAttribute('aria-expanded', String(expanded));
        btn.setAttribute('aria-label',
            `Toggle legs for structure ${typeof s.id === 'string' ? s.id : ''}`);
        const icon = document.createElement('i');
        icon.className = 'bi bi-chevron-right';
        icon.setAttribute('aria-hidden', 'true');
        btn.appendChild(icon);
        btn.addEventListener('click', () => {
            const nowOpen = legsRow.classList.toggle('d-none') === false;
            btn.setAttribute('aria-expanded', String(nowOpen));
            btn.classList.toggle('open', nowOpen);
            if (nowOpen) expandedStructures.add(s._key);
            else expandedStructures.delete(s._key);
        });
        tdToggle.appendChild(btn);
    }
    tr.appendChild(tdToggle);

    tr.appendChild(structCell(structureTypeLabel(s.type)));
    tr.appendChild(structCell(String(s.underlying ?? '—')));
    tr.appendChild(structCell(fmtNum(s.contracts, 0), 'num'));
    tr.appendChild(structCell(s.opened ?? '—', 'num'));
    tr.appendChild(structCell(s.closed ?? '—', 'num'));
    tr.appendChild(structCell(fmtMoney(s.credit), 'num'));
    tr.appendChild(structCell(fmtMoney(s.max_loss), 'num'));

    // P&L: realized $ when closed/expired (contract C11), em dash while
    // open — colored by sign, like every other P&L in the app.
    const pnlOpen = s.pnl === null || s.pnl === undefined;
    tr.appendChild(structCell(
        pnlOpen ? '—' : fmtMoney(s.pnl, { sign: true }),
        `num ${pnlOpen ? 'pnl-flat' : pnlClass(s.pnl)}`));

    // Status badge: OPEN / CLOSED / EXPIRED (whitelisted classes only).
    const tdStatus = document.createElement('td');
    const statusMeta = STRUCTURE_STATUS_META[s.status] ||
        { label: '—', cls: '' };
    const badge = document.createElement('span');
    badge.className =
        `structure-status-badge${statusMeta.cls ? ` ${statusMeta.cls}` : ''}`;
    badge.textContent = statusMeta.label;
    tdStatus.appendChild(badge);
    tr.appendChild(tdStatus);

    // close_reason chip: whitelisted reasons get a color class; any other
    // non-blank string renders as a default chip (textContent only).
    const tdReason = document.createElement('td');
    if (typeof s.close_reason === 'string' && s.close_reason.trim() !== '') {
        const chip = document.createElement('span');
        chip.className = 'note-cat-chip';
        if (CLOSE_REASON_LABELS[s.close_reason]) {
            chip.classList.add(`reason-${s.close_reason}`);
        }
        chip.textContent = CLOSE_REASON_LABELS[s.close_reason] ||
            s.close_reason.replace(/_/g, ' ');
        tdReason.appendChild(chip);
    } else {
        tdReason.textContent = '—';
        tdReason.className = 'pnl-flat';
    }
    tr.appendChild(tdReason);

    return tr;
}

/** Hidden detail row: one line per leg — action, right/strike/expiry, and
 *  the full instrument contract string (textContent only). */
function structureLegsRow(s) {
    const tr = document.createElement('tr');
    tr.className = 'structure-legs-row';
    if (!expandedStructures.has(s._key)) tr.classList.add('d-none');

    const td = document.createElement('td');
    td.colSpan = 11;
    const legs = Array.isArray(s.legs) ? s.legs : [];
    if (legs.length === 0) {
        td.className = 'structure-legs-empty';
        td.textContent = 'No leg detail for this structure.';
    } else {
        const list = document.createElement('ul');
        list.className = 'structure-legs list-unstyled mb-0';
        legs.forEach((leg) => {
            if (!leg || typeof leg !== 'object') return;
            const li = document.createElement('li');
            li.className = 'structure-leg';

            const action = String(leg.action ?? '').toUpperCase();
            const actionEl = document.createElement('span');
            actionEl.className = 'structure-leg-action ' +
                (action === 'BUY' ? 'pnl-pos' : 'pnl-neg');
            actionEl.textContent = action || '—';

            const descEl = document.createElement('span');
            descEl.className = 'structure-leg-desc';
            descEl.textContent =
                `${String(leg.right ?? '').toUpperCase() || '—'} ` +
                `${fmtMoney(leg.strike)} · exp ${leg.expiry ?? '—'}`;

            const instrumentEl = document.createElement('span');
            instrumentEl.className = 'structure-leg-instrument';
            instrumentEl.textContent = String(leg.instrument ?? '');

            li.append(actionEl, descEl, instrumentEl);
            list.appendChild(li);
        });
        td.appendChild(list);
    }
    tr.appendChild(td);
    return tr;
}

function initStructureSorting() {
    document.querySelectorAll('#structuresTable th.sortable').forEach((th) => {
        th.addEventListener('click', () => {
            const key = th.dataset.key;
            structSort = {
                key,
                dir: structSort.key === key ? -structSort.dir : 1,
            };
            if (currentStructures.length > 0) sortAndPaintStructures();
        });
    });
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
        // Contract C13 (additive): 'instrument' is str(asset) — the bare
        // symbol for stocks, the full contract string for options. Only
        // option fills get the extra monospace line (stocks would just
        // repeat the Symbol cell).
        const instrument =
            typeof trade.instrument === 'string' ? trade.instrument : '';
        const showInstrument = instrument && instrument !== trade.symbol;
        return '<tr>' +
            `<td class="num">${escapeHTML(trade.signal_date ?? '—')}</td>` +
            `<td class="num">${escapeHTML(trade.date ?? '—')}</td>` +
            `<td>${escapeHTML(trade.symbol ?? '')}` +
            (showInstrument
                ? `<div class="trade-instrument">${escapeHTML(instrument)}</div>`
                : '') +
            '</td>' +
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
        // Research integrity (Phase 3): compare overfitting risk side by side.
        ['PSR', summary.psr === null || summary.psr === undefined
            ? '—' : `${(Number(summary.psr) * 100).toFixed(1)}%`,
            summary.psr === null || summary.psr === undefined
                ? undefined : Number(summary.psr) - 0.5],
        ['Defl. Sharpe', summary.deflated_sharpe === null
            || summary.deflated_sharpe === undefined
            ? '—' : `${(Number(summary.deflated_sharpe) * 100).toFixed(1)}%`,
            summary.deflated_sharpe === null
                || summary.deflated_sharpe === undefined
                ? undefined : Number(summary.deflated_sharpe) - 0.5],
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
    loadModels();
    initDeskMode();
    loadSaved();
    initTradeSorting();
    initStructureSorting();
    document.getElementById('backtestForm').addEventListener('submit', onRun);
    document.getElementById('reloadSavedBtn').addEventListener('click', loadSaved);
    document.getElementById('compareBtn').addEventListener('click', onCompare);
    document.getElementById('closeCompareBtn').addEventListener('click', closeCompare);
});
