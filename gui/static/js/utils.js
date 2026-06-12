// Shared utilities for the StockOptionsTrader terminal UI.
// Dependency-light vanilla JS. Everything here is page-agnostic.

'use strict';

/* ==========================================================================
   Toasts
   ========================================================================== */

/**
 * Show a toast in the top-right container.
 * @param {('success'|'error'|'warning'|'info')} level
 * @param {string} msg
 * @param {number} [timeoutMs=6000] auto-dismiss; 0 disables.
 */
function showToast(level, msg, timeoutMs = 6000) {
    const container = document.getElementById('toastContainer');
    if (!container) return;

    const icons = {
        success: 'bi-check-circle',
        error: 'bi-x-octagon',
        warning: 'bi-exclamation-triangle',
        info: 'bi-info-circle',
    };
    const lvl = icons[level] ? level : 'info';

    const toast = document.createElement('div');
    toast.className = `toast-app toast-${lvl}`;
    toast.setAttribute('role', lvl === 'error' ? 'alert' : 'status');

    const icon = document.createElement('i');
    icon.className = `bi ${icons[lvl]}`;
    icon.setAttribute('aria-hidden', 'true');

    const body = document.createElement('div');
    body.className = 'toast-msg';
    body.textContent = msg;

    const close = document.createElement('button');
    close.type = 'button';
    close.className = 'btn-close';
    close.setAttribute('aria-label', 'Dismiss');
    close.addEventListener('click', () => toast.remove());

    toast.append(icon, body, close);
    container.appendChild(toast);

    if (timeoutMs > 0) {
        setTimeout(() => toast.remove(), timeoutMs);
    }
}

/* ==========================================================================
   Number formatting — tabular, sign-aware
   ========================================================================== */

const _moneyFmt = new Intl.NumberFormat('en-US', {
    style: 'currency', currency: 'USD',
    minimumFractionDigits: 2, maximumFractionDigits: 2,
});

const _numFmt = new Intl.NumberFormat('en-US', {
    minimumFractionDigits: 0, maximumFractionDigits: 2,
});

/** "$1,234.56"; signed ("+$12.34") when opts.sign is true. NaN-safe. */
function fmtMoney(v, opts = {}) {
    if (v === null || v === undefined || Number.isNaN(Number(v))) return '—';
    const n = Number(v);
    const s = _moneyFmt.format(Math.abs(n));
    if (n < 0) return `-${s}`;
    return opts.sign && n > 0 ? `+${s}` : s;
}

/** Fraction in, percent out: fmtPct(0.0823) -> "8.23%". */
function fmtPct(v, opts = {}) {
    if (v === null || v === undefined || Number.isNaN(Number(v))) return '—';
    const n = Number(v) * 100;
    const s = `${Math.abs(n).toFixed(opts.digits ?? 2)}%`;
    if (n < 0) return `-${s}`;
    return opts.sign && n > 0 ? `+${s}` : s;
}

/** Plain tabular number, up to 2 decimals by default. */
function fmtNum(v, digits) {
    if (v === null || v === undefined || Number.isNaN(Number(v))) return '—';
    const n = Number(v);
    return digits === undefined ? _numFmt.format(n) : n.toFixed(digits);
}

/** Semantic class for a signed value: pnl-pos / pnl-neg / pnl-flat. */
function pnlClass(v) {
    const n = Number(v);
    if (Number.isNaN(n) || n === 0) return 'pnl-flat';
    return n > 0 ? 'pnl-pos' : 'pnl-neg';
}

/* ==========================================================================
   Loading states
   ========================================================================== */

/** Cover an element with a spinner overlay. Returns a remover function. */
function showLoading(el) {
    if (!el) return () => {};
    el.classList.add('pos-relative');
    const overlay = document.createElement('div');
    overlay.className = 'loading-overlay';
    overlay.innerHTML =
        '<div class="spinner-border spinner-border-sm" role="status">' +
        '<span class="visually-hidden">Loading…</span></div>';
    el.appendChild(overlay);
    return () => overlay.remove();
}

/** Disable a button and show a small spinner; returns a restore function. */
function btnLoading(btn) {
    if (!btn) return () => {};
    const original = btn.innerHTML;
    btn.disabled = true;
    btn.innerHTML =
        '<span class="spinner-border spinner-border-sm me-1" aria-hidden="true"></span>' +
        'Working…';
    return () => { btn.disabled = false; btn.innerHTML = original; };
}

/** HTML string of skeleton placeholder rows for a loading table. */
function skeletonRows(cols, rows = 3) {
    let html = '';
    for (let r = 0; r < rows; r++) {
        html += '<tr>';
        for (let c = 0; c < cols; c++) {
            html += '<td><span class="skeleton"></span></td>';
        }
        html += '</tr>';
    }
    return html;
}

/** HTML string for an empty-state block (icon + message + hint). */
function emptyStateHTML(icon, msg, hint = '') {
    return (
        '<div class="empty-state">' +
        `<i class="bi ${escapeHTML(icon)}" aria-hidden="true"></i>` +
        `<div class="empty-msg">${escapeHTML(msg)}</div>` +
        (hint ? `<div class="empty-hint">${escapeHTML(hint)}</div>` : '') +
        '</div>'
    );
}

/* ==========================================================================
   Fetch wrapper
   ========================================================================== */

/**
 * fetch() returning parsed JSON. On HTTP error or network failure, shows an
 * error toast (unless opts.silent) and throws.
 */
async function fetchJSON(url, opts = {}) {
    const { silent, ...fetchOpts } = opts;
    if (fetchOpts.body && !(fetchOpts.headers || {})['Content-Type']) {
        fetchOpts.headers = {
            'Content-Type': 'application/json', ...(fetchOpts.headers || {}),
        };
    }
    let response;
    try {
        response = await fetch(url, fetchOpts);
    } catch (err) {
        if (!silent) showToast('error', `Network error: ${err.message}`);
        throw err;
    }
    let data = null;
    try {
        data = await response.json();
    } catch (_) {
        // Non-JSON body; fall through with null.
    }
    if (!response.ok) {
        const msg = (data && data.error) || `Request failed (${response.status})`;
        if (!silent) showToast('error', msg);
        const error = new Error(msg);
        error.status = response.status;
        error.body = data;
        throw error;
    }
    return data;
}

/* ==========================================================================
   Misc
   ========================================================================== */

function escapeHTML(s) {
    return String(s)
        .replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;')
        .replaceAll('"', '&quot;').replaceAll("'", '&#39;');
}

function getDateString(date) {
    return date.toISOString().split('T')[0];
}

function setDefaultDates() {
    const endDate = new Date();
    const startDate = new Date();
    startDate.setFullYear(startDate.getFullYear() - 1);

    const startInput = document.getElementById('startDate');
    const endInput = document.getElementById('endDate');

    if (startInput && !startInput.value) startInput.value = getDateString(startDate);
    if (endInput && !endInput.value) endInput.value = getDateString(endDate);
}

/* ==========================================================================
   Status bar: ping /health and color the dot
   ========================================================================== */

async function _pingHealth() {
    const dot = document.getElementById('healthDot');
    const label = document.getElementById('healthLabel');
    if (!dot) return;
    try {
        const data = await fetchJSON('/health', { silent: true });
        const ok = data && data.status === 'ok';
        dot.className = `status-dot ${ok ? 'ok' : 'error'}`;
        if (label) label.textContent = ok ? 'online' : 'degraded';
    } catch (_) {
        dot.className = 'status-dot error';
        if (label) label.textContent = 'offline';
    }
}

document.addEventListener('DOMContentLoaded', () => {
    setDefaultDates();
    _pingHealth();
    setInterval(_pingHealth, 60000);
});
