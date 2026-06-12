// Paper trading page: account metrics, positions, pending orders with
// cancel, order entry with risk feedback, 15s auto-refresh (paused while
// the tab is hidden). Talks to:
//   POST /api/trader/create
//   GET  /api/trader/<id>/status
//   GET  /api/trader/<id>/orders
//   POST /api/trader/<id>/order
//   POST /api/trader/<id>/order/<order_id>/cancel
//   GET  /api/risk/settings

'use strict';

const TRADER_ID_KEY = 'sot.paperTraderId';
const TRADER_ID_RE = /^[A-Za-z0-9_\-]{1,32}$/;
const ORDER_SYMBOL_RE = /^[A-Za-z][A-Za-z.\-]{0,9}$/;
const REFRESH_MS = 15000;

let traderId = null;
let refreshTimer = null;
let lastStatus = null;     // most recent /status payload (for risk feedback)
let riskSettings = null;   // /api/risk/settings payload

/* ==========================================================================
   Session discovery / creation
   ========================================================================== */

function storedTraderId() {
    try {
        return window.localStorage.getItem(TRADER_ID_KEY) || 'default';
    } catch (_) {
        return 'default';
    }
}

function rememberTraderId(id) {
    try { window.localStorage.setItem(TRADER_ID_KEY, id); } catch (_) { /* no-op */ }
}

function showCreatePanel() {
    document.getElementById('tradingPanel').classList.add('d-none');
    document.getElementById('refreshControls').classList.add('d-none');
    document.getElementById('createTraderPanel').classList.remove('d-none');
}

function showTradingPanel() {
    document.getElementById('createTraderPanel').classList.add('d-none');
    document.getElementById('refreshControls').classList.remove('d-none');
    document.getElementById('tradingPanel').classList.remove('d-none');
}

async function discoverSession() {
    const candidate = storedTraderId();
    try {
        const status = await fetchJSON(
            `/api/trader/${encodeURIComponent(candidate)}/status`, { silent: true });
        traderId = candidate;
        showTradingPanel();
        renderStatus(status);
        loadOrders();
        startAutoRefresh();
    } catch (err) {
        if (err.status === 404) {
            showCreatePanel();
        } else {
            showToast('error', 'Could not reach the trading API — see server log');
            showCreatePanel();
        }
    }
}

async function onCreateTrader(event) {
    event.preventDefault();
    const idEl = document.getElementById('traderId');
    const capitalEl = document.getElementById('traderCapital');

    const idOk = TRADER_ID_RE.test(idEl.value.trim());
    idEl.classList.toggle('is-invalid', !idOk);
    const capitalOk = Number(capitalEl.value) > 0;
    capitalEl.classList.toggle('is-invalid', !capitalOk);
    if (!idOk || !capitalOk) return;

    const restore = btnLoading(document.getElementById('createTraderBtn'));
    try {
        const data = await fetchJSON('/api/trader/create', {
            method: 'POST',
            body: JSON.stringify({
                trader_id: idEl.value.trim(),
                mode: 'paper',
                initial_capital: Number(capitalEl.value),
            }),
        });
        traderId = data.trader_id;
        rememberTraderId(traderId);
        showToast('success', `Paper session "${traderId}" created`);
        showTradingPanel();
        await refreshAll();
        startAutoRefresh();
    } catch (_) {
        // fetchJSON already toasted.
    } finally {
        restore();
    }
}

/* ==========================================================================
   Status: account cards + positions table
   ========================================================================== */

async function refreshAll() {
    if (!traderId) return;
    try {
        const status = await fetchJSON(
            `/api/trader/${encodeURIComponent(traderId)}/status`, { silent: true });
        renderStatus(status);
    } catch (err) {
        if (err.status === 404) {
            // Server restarted: sessions are in-memory and this one is gone.
            stopAutoRefresh();
            traderId = null;
            showCreatePanel();
            showToast('warning', 'Paper session expired (server restart) — create a new one');
            return;
        }
        showToast('error', 'Could not refresh portfolio status');
    }
    loadOrders();
}

function renderStatus(status) {
    lastStatus = status;
    const positions = status.positions || [];
    const pnl = Number(status.unrealized_pnl ?? 0);
    const value = Number(status.portfolio_value ?? 0);
    const costBasis = value - Number(status.cash ?? 0) - pnl;

    document.getElementById('acctCash').textContent = fmtMoney(status.cash);
    document.getElementById('acctValue').textContent = fmtMoney(value);

    const pnlEl = document.getElementById('acctUnrealized');
    pnlEl.textContent = fmtMoney(pnl, { sign: true });
    pnlEl.classList.remove('pnl-pos', 'pnl-neg', 'pnl-flat');
    pnlEl.classList.add(pnlClass(pnl));

    const pnlPctEl = document.getElementById('acctUnrealizedPct');
    pnlPctEl.classList.remove('pnl-pos', 'pnl-neg', 'pnl-flat');
    if (costBasis > 0) {
        pnlPctEl.textContent = fmtPct(pnl / costBasis, { sign: true });
        pnlPctEl.classList.add(pnlClass(pnl));
    } else {
        pnlPctEl.textContent = '';
    }

    document.getElementById('acctPositions').textContent = String(positions.length);
    const ts = String(status.timestamp || '').slice(11, 19);
    document.getElementById('acctUpdated').textContent = ts ? `as of ${ts}` : '';

    renderPositions(positions);
    updateRiskFeedback();
}

function renderPositions(positions) {
    const body = document.getElementById('positionsBody');
    const emptyEl = document.getElementById('positionsEmpty');
    if (positions.length === 0) {
        body.innerHTML = '';
        emptyEl.classList.remove('d-none');
        emptyEl.innerHTML = emptyStateHTML('bi-inboxes', 'No open positions',
            'Fills land here once a pending order becomes marketable.');
        return;
    }
    emptyEl.classList.add('d-none');
    body.innerHTML = positions.map((pos) => {
        const pnl = Number(pos.pnl ?? 0);
        // pnl_pct arrives as a percent (e.g. 4.2), not a fraction.
        const pct = pos.pnl_pct === null || pos.pnl_pct === undefined
            ? '—' : `${pnl >= 0 && Number(pos.pnl_pct) > 0 ? '+' : ''}${Number(pos.pnl_pct).toFixed(2)}%`;
        return '<tr>' +
            `<td>${escapeHTML(pos.symbol)}</td>` +
            `<td class="num">${fmtNum(pos.quantity, 0)}</td>` +
            `<td class="num">${fmtMoney(pos.entry_price)}</td>` +
            `<td class="num">${fmtMoney(pos.current_price)}</td>` +
            `<td class="num ${pnlClass(pnl)}">${fmtMoney(pnl, { sign: true })}</td>` +
            `<td class="num ${pnlClass(pnl)}">${pct}</td>` +
            '</tr>';
    }).join('');
}

/* ==========================================================================
   Pending orders + cancel
   ========================================================================== */

async function loadOrders() {
    if (!traderId) return;
    const body = document.getElementById('ordersBody');
    const emptyEl = document.getElementById('ordersEmpty');
    try {
        const data = await fetchJSON(
            `/api/trader/${encodeURIComponent(traderId)}/orders`, { silent: true });
        const orders = data.orders || [];
        if (orders.length === 0) {
            body.innerHTML = '';
            emptyEl.classList.remove('d-none');
            emptyEl.innerHTML = emptyStateHTML('bi-hourglass', 'No pending orders',
                'Orders wait here until they become marketable.');
            return;
        }
        emptyEl.classList.add('d-none');
        body.innerHTML = orders.map(orderRow).join('');
        body.querySelectorAll('[data-order-id]').forEach((btn) => {
            btn.addEventListener('click', () => cancelOrder(btn.dataset.orderId, btn));
        });
    } catch (_) {
        emptyEl.classList.remove('d-none');
        emptyEl.innerHTML = emptyStateHTML('bi-exclamation-triangle',
            'Could not load pending orders', 'Check the server log and refresh.');
    }
}

function orderRow(order) {
    const buy = order.side === 'BUY';
    const limit = order.limit_price === null || order.limit_price === undefined
        ? 'MKT' : fmtMoney(order.limit_price);
    return '<tr>' +
        `<td class="num">${escapeHTML(order.order_id)}</td>` +
        `<td>${escapeHTML(order.symbol)}</td>` +
        `<td class="${buy ? 'pnl-pos' : 'pnl-neg'}">${escapeHTML(order.side)}</td>` +
        `<td class="num">${fmtNum(order.quantity, 0)}</td>` +
        `<td class="num">${limit}</td>` +
        '<td class="text-end">' +
        `<button type="button" class="btn btn-outline-secondary btn-sm" data-order-id="${escapeHTML(order.order_id)}" ` +
        `aria-label="Cancel order ${escapeHTML(order.order_id)}">` +
        '<i class="bi bi-x-lg" aria-hidden="true"></i> Cancel</button></td>' +
        '</tr>';
}

async function cancelOrder(orderId, btn) {
    const restore = btnLoading(btn);
    try {
        await fetchJSON(
            `/api/trader/${encodeURIComponent(traderId)}/order/${encodeURIComponent(orderId)}/cancel`,
            { method: 'POST' });
        showToast('success', `Order ${orderId} cancelled`);
        loadOrders();
    } catch (_) {
        restore(); // fetchJSON already toasted; row may simply be gone
        loadOrders();
    }
}

/* ==========================================================================
   Order entry + risk feedback
   ========================================================================== */

function validateOrderForm() {
    const symbolEl = document.getElementById('orderSymbol');
    const qtyEl = document.getElementById('orderQty');
    const limitEl = document.getElementById('orderLimit');
    let ok = true;

    const symbolOk = ORDER_SYMBOL_RE.test(symbolEl.value.trim());
    symbolEl.classList.toggle('is-invalid', !symbolOk);
    ok = ok && symbolOk;

    const qty = Number(qtyEl.value);
    const qtyOk = Number.isInteger(qty) && qty >= 1;
    qtyEl.classList.toggle('is-invalid', !qtyOk);
    ok = ok && qtyOk;

    const limitRaw = limitEl.value.trim();
    const limitOk = limitRaw === '' || Number(limitRaw) > 0;
    limitEl.classList.toggle('is-invalid', !limitOk);
    ok = ok && limitOk;

    return ok;
}

async function loadRiskSettings() {
    try {
        riskSettings = await fetchJSON('/api/risk/settings', { silent: true });
    } catch (_) {
        riskSettings = null; // feedback degrades silently; orders still work
    }
    updateRiskFeedback();
}

function updateRiskFeedback() {
    const el = document.getElementById('riskFeedback');
    if (!el) return;
    el.classList.remove('risk-breach');

    const qty = Number(document.getElementById('orderQty').value);
    const limitRaw = document.getElementById('orderLimit').value.trim();
    if (!Number.isFinite(qty) || qty <= 0) { el.textContent = ''; return; }

    if (limitRaw === '') {
        el.textContent = 'Market order — sized at next available close.';
        return;
    }

    const notional = qty * Number(limitRaw);
    let text = `Notional ${fmtMoney(notional)}`;
    if (riskSettings && lastStatus) {
        const maxFrac = Number(riskSettings.max_position_size);
        const value = Number(lastStatus.portfolio_value);
        if (Number.isFinite(maxFrac) && Number.isFinite(value) && value > 0) {
            const cap = maxFrac * value;
            text += ` · position cap ${fmtMoney(cap)} (${fmtPct(maxFrac, { digits: 0 })} of portfolio)`;
            if (notional > cap) {
                el.classList.add('risk-breach');
                text += ' — exceeds cap';
            }
        }
    }
    el.textContent = text;
}

async function onPlaceOrder(event) {
    event.preventDefault();
    if (!traderId) { showToast('warning', 'Create a paper session first'); return; }
    if (!validateOrderForm()) return;

    const limitRaw = document.getElementById('orderLimit').value.trim();
    const payload = {
        symbol: document.getElementById('orderSymbol').value.trim().toUpperCase(),
        action: document.getElementById('orderSide').value,
        quantity: Number(document.getElementById('orderQty').value),
        // API contract: price 0 == market order (limit_price=None server-side).
        price: limitRaw === '' ? 0 : Number(limitRaw),
    };

    const restore = btnLoading(document.getElementById('placeOrderBtn'));
    try {
        const data = await fetchJSON(
            `/api/trader/${encodeURIComponent(traderId)}/order`,
            { method: 'POST', body: JSON.stringify(payload) });
        showToast('success', `Order ${data.order_id} placed — ` +
            `${payload.action} ${payload.quantity} ${payload.symbol}` +
            (limitRaw === '' ? ' at market' : ` limit ${fmtMoney(payload.price)}`));
        await refreshAll();
    } catch (_) {
        // fetchJSON already toasted.
    } finally {
        restore();
    }
}

/* ==========================================================================
   Auto-refresh (15s, paused while the tab is hidden)
   ========================================================================== */

function startAutoRefresh() {
    stopAutoRefresh();
    if (!document.getElementById('autoRefreshToggle').checked) return;
    refreshTimer = setInterval(() => {
        if (!document.hidden) refreshAll();
    }, REFRESH_MS);
}

function stopAutoRefresh() {
    if (refreshTimer !== null) { clearInterval(refreshTimer); refreshTimer = null; }
}

function onVisibilityChange() {
    if (document.hidden) {
        stopAutoRefresh();
    } else if (traderId) {
        refreshAll();
        startAutoRefresh();
    }
}

/* ==========================================================================
   Init
   ========================================================================== */

document.addEventListener('DOMContentLoaded', () => {
    document.getElementById('createTraderForm')
        .addEventListener('submit', onCreateTrader);
    document.getElementById('orderForm').addEventListener('submit', onPlaceOrder);
    document.getElementById('refreshBtn').addEventListener('click', refreshAll);
    document.getElementById('autoRefreshToggle')
        .addEventListener('change', startAutoRefresh);
    document.addEventListener('visibilitychange', onVisibilityChange);
    ['orderQty', 'orderLimit', 'orderSide', 'orderSymbol'].forEach((id) => {
        document.getElementById(id).addEventListener('input', updateRiskFeedback);
    });

    loadRiskSettings();
    discoverSession();
});
