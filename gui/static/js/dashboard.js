// Dashboard: paper-portfolio metrics, recent backtests, alerts.
// Read-only on load — never creates traders or triggers price fetches.

'use strict';

const DEFAULT_TRADER_ID = 'default';

/* ==========================================================================
   Portfolio metric cards
   ========================================================================== */

async function loadPortfolio() {
    const metricsEl = document.getElementById('portfolioMetrics');
    const emptyEl = document.getElementById('portfolioEmpty');
    try {
        const status = await fetchJSON(
            `/api/trader/${DEFAULT_TRADER_ID}/status`, { silent: true });
        renderPortfolio(status);
    } catch (err) {
        if (err.status === 404) {
            // No default trader yet — expected on a fresh install.
            metricsEl.classList.add('d-none');
            emptyEl.classList.remove('d-none');
            emptyEl.innerHTML =
                '<div class="card"><div class="card-body p-0">' +
                emptyStateHTML('bi-wallet2', 'No paper trading session',
                    'Start one from the Paper Trading page to see portfolio metrics here.') +
                '</div></div>';
        } else {
            showToast('error', 'Could not load portfolio status');
            setText('metricPortfolioValue', '—');
            setText('metricCash', '—');
            setText('metricUnrealized', '—');
            setText('metricPositions', '—');
        }
    }
}

function renderPortfolio(status) {
    const positions = status.positions || [];
    const pnl = Number(status.unrealized_pnl ?? 0);
    const value = Number(status.portfolio_value ?? 0);
    // Position-only cost basis (cash excluded), matching paper_trade.js
    // renderStatus and the per-position pnl_pct semantics.
    const costBasis = value - Number(status.cash ?? 0) - pnl;

    setText('metricPortfolioValue', fmtMoney(value));
    setText('metricCash', fmtMoney(status.cash));

    const pnlEl = document.getElementById('metricUnrealized');
    pnlEl.textContent = fmtMoney(pnl, { sign: true });
    pnlEl.classList.add(pnlClass(pnl));

    const pnlPctEl = document.getElementById('metricUnrealizedPct');
    if (costBasis > 0) {
        pnlPctEl.textContent = fmtPct(pnl / costBasis, { sign: true });
        pnlPctEl.classList.add(pnlClass(pnl));
    }

    setText('metricPositions', String(positions.length));
    const pending = Number(status.pending_orders ?? 0);
    setText('metricPendingOrders',
        pending > 0 ? `${pending} pending order${pending === 1 ? '' : 's'}` : '');
}

function setText(id, text) {
    const el = document.getElementById(id);
    if (el) el.textContent = text;
}

/* ==========================================================================
   Recent backtests
   ========================================================================== */

async function loadBacktests() {
    const tbody = document.getElementById('backtestRows');
    const emptyEl = document.getElementById('backtestEmpty');
    tbody.innerHTML = skeletonRows(5, 3);
    try {
        const data = await fetchJSON('/api/backtests', { silent: true });
        const backtests = (data.backtests || []).slice(0, 8);
        if (backtests.length === 0) {
            tbody.innerHTML = '';
            emptyEl.classList.remove('d-none');
            emptyEl.innerHTML = emptyStateHTML('bi-clock-history',
                'No saved backtests',
                'Run one from the Backtest page — results are saved automatically.');
            return;
        }
        emptyEl.classList.add('d-none');
        tbody.innerHTML = backtests.map(backtestRow).join('');
    } catch (_) {
        tbody.innerHTML = '';
        emptyEl.classList.remove('d-none');
        emptyEl.innerHTML = emptyStateHTML('bi-exclamation-triangle',
            'Could not load backtests', 'Check the server log and refresh.');
    }
}

function backtestRow(bt) {
    // bt.total_return is the DB column: a FRACTION of initial capital.
    const ret = bt.total_return;
    const period = `${bt.start_date || '?'} → ${bt.end_date || '?'}`;
    return (
        '<tr>' +
        `<td><a href="/backtest">${escapeHTML(bt.name || `#${bt.id}`)}</a></td>` +
        `<td>${escapeHTML(bt.strategy || '—')}</td>` +
        `<td class="num">${escapeHTML(period)}</td>` +
        `<td class="num ${pnlClass(ret)}">${fmtPct(ret, { sign: true })}</td>` +
        `<td class="num">${fmtNum(bt.sharpe_ratio, 2)}</td>` +
        '</tr>'
    );
}

/* ==========================================================================
   Alerts
   ========================================================================== */

async function loadAlerts() {
    const list = document.getElementById('alertList');
    const emptyEl = document.getElementById('alertEmpty');
    const badge = document.getElementById('alertUnreadBadge');
    try {
        const data = await fetchJSON('/api/alerts', { silent: true });
        const alerts = (data.alerts || []).slice().reverse().slice(0, 10);

        const unread = Number(data.unread_count || 0);
        badge.textContent = String(unread);
        badge.classList.toggle('d-none', unread === 0);

        if (alerts.length === 0) {
            list.innerHTML = '';
            emptyEl.classList.remove('d-none');
            emptyEl.innerHTML = emptyStateHTML('bi-bell-slash',
                'No alerts', 'Price targets and signal alerts will appear here.');
            return;
        }
        emptyEl.classList.add('d-none');
        list.innerHTML = alerts.map(alertItem).join('');

        list.querySelectorAll('[data-alert-id]').forEach((btn) => {
            btn.addEventListener('click', () => markAlertRead(btn.dataset.alertId));
        });
    } catch (_) {
        emptyEl.classList.remove('d-none');
        emptyEl.innerHTML = emptyStateHTML('bi-exclamation-triangle',
            'Could not load alerts', 'Check the server log and refresh.');
    }
}

function alertItem(alert) {
    const ts = String(alert.timestamp || '').slice(0, 16).replace('T', ' ');
    const unread = !alert.read;
    return (
        `<li class="d-flex align-items-start gap-2 px-3 py-2 border-bottom border-secondary-subtle${unread ? '' : ' opacity-50'}">` +
        '<div class="flex-grow-1">' +
        `<div style="font-size:13px;">${escapeHTML(alert.message || '')}</div>` +
        `<div class="provenance-caption">${escapeHTML(alert.symbol || '')} · ${escapeHTML(ts)}</div>` +
        '</div>' +
        (unread
            ? `<button type="button" class="btn btn-outline-secondary btn-sm" data-alert-id="${Number(alert.id)}" aria-label="Mark alert as read"><i class="bi bi-check2" aria-hidden="true"></i></button>`
            : '') +
        '</li>'
    );
}

async function markAlertRead(alertId) {
    try {
        await fetchJSON(`/api/alert/${alertId}/read`, { method: 'POST' });
        await loadAlerts();
    } catch (_) {
        // fetchJSON already toasted the failure.
    }
}

/* ==========================================================================
   Init
   ========================================================================== */

document.addEventListener('DOMContentLoaded', () => {
    loadPortfolio();
    loadBacktests();
    loadAlerts();
});
