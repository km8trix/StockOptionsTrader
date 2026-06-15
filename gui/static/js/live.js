// Live Trading page (Phase 9): paints connection/auth state, kill switch,
// working orders, audit log and reconciliation from the /api/live/* routes
// (contract C20). Depends on utils.js (fetchJSON, showToast, escapeHTML,
// emptyStateHTML, btnLoading, fmtMoney, fmtNum).
//
// Design rules enforced here:
// * This is the ONLY page that polls live status — the global kill-switch
//   banner and the nav dot in base.html are updated from THIS page's poll
//   (other pages get the server-rendered banner state, no polling).
// * Auth is NEVER auto-started: page load only ever reads GET /status;
//   every OAuth step requires an explicit click.
// * Both kill-switch directions require a confirm dialog; engaging
//   requires a reason.

'use strict';

const LIVE_POLL_MS = 10000;
const AUDIT_PAGE_SIZE = 50;

// Last known status payload; null until the first poll resolves.
let liveStatus = null;
// Audit pager state.
let auditOffset = 0;
let auditLastCount = 0;
// Distinct event types seen so far (keeps the filter select stable).
const auditEventTypes = new Set();

/* ==========================================================================
   Small helpers
   ========================================================================== */

/** Only http(s) authorize URLs become link-outs; anything else is text. */
function safeHttpUrl(url) {
    try {
        const parsed = new URL(String(url));
        return parsed.protocol === 'https:' || parsed.protocol === 'http:'
            ? parsed.href : null;
    } catch (_) {
        return null;
    }
}

/** "3m 12s" from a second count; '—' when not a finite number. */
function fmtDuration(seconds) {
    const s = Number(seconds);
    if (!Number.isFinite(s) || s < 0) return '—';
    const m = Math.floor(s / 60);
    const rest = Math.round(s % 60);
    return m > 0 ? `${m}m ${rest}s` : `${rest}s`;
}

function fmtTimestamp(iso) {
    if (!iso) return '—';
    const d = new Date(iso);
    return Number.isNaN(d.getTime()) ? escapeHTML(String(iso))
        : escapeHTML(d.toLocaleString());
}

/* ==========================================================================
   Environment badge + nav dot + global banner
   ========================================================================== */

function renderEnvBadge(env, available) {
    const badge = document.getElementById('envBadge');
    if (!badge) return;
    if (!available) {
        badge.className = 'env-badge env-unknown ms-auto';
        badge.textContent = 'UNAVAILABLE';
    } else if (env === 'production') {
        badge.className = 'env-badge env-production ms-auto';
        badge.textContent = 'PRODUCTION';
    } else if (env === 'sandbox') {
        badge.className = 'env-badge env-sandbox ms-auto';
        badge.textContent = 'SANDBOX';
    } else {
        badge.className = 'env-badge env-unknown ms-auto';
        badge.textContent = 'ENV UNKNOWN';
    }
}

function renderNavDot(state, available) {
    const dot = document.getElementById('navLiveDot');
    if (!dot) return;
    let cls = 'off';
    if (!available) cls = 'error';
    else if (state === 'connected') cls = 'ok';
    else if (state === 'pending_verifier' || state === 'expired') cls = 'warn';
    dot.className = `nav-live-dot ${cls}`;
}

function renderKillBanner(engaged) {
    const banner = document.getElementById('killSwitchBanner');
    if (!banner) return;
    banner.classList.toggle('d-none', !engaged);
}

/* ==========================================================================
   Connection panel (auth state machine, contract C16)
   ========================================================================== */

const AUTH_BADGES = {
    unconfigured: ['auth-muted', 'unconfigured'],
    disconnected: ['auth-muted', 'disconnected'],
    pending_verifier: ['auth-warn', 'pending verifier'],
    connected: ['auth-ok', 'connected'],
    expired: ['auth-warn', 'expired'],
};

function renderAuthBadge(state, available) {
    const badge = document.getElementById('authStateBadge');
    if (!badge) return;
    const [cls, label] = available
        ? (AUTH_BADGES[state] || ['auth-unknown', state || 'unknown'])
        : ['auth-err', 'unavailable'];
    badge.className = `auth-state-badge ${cls}`;
    badge.textContent = label;
}

function tokenTimesHTML(auth) {
    return (
        '<dl class="token-times">' +
        '<dt>Token issued</dt>' +
        `<dd class="num">${fmtTimestamp(auth.token_issued_at)}</dd>` +
        '<dt>Last renewed</dt>' +
        `<dd class="num">${fmtTimestamp(auth.renewed_at)}</dd>` +
        '</dl>'
    );
}

function connectionHTML(auth) {
    const state = auth.state;
    if (state === 'unconfigured') {
        return emptyStateHTML('bi-key', 'E*TRADE is not configured',
            'Add ETRADE_CONSUMER_KEY and ETRADE_CONSUMER_SECRET to .env, ' +
            'then restart. Keys are never displayed here.');
    }
    if (state === 'disconnected') {
        return (
            '<p class="conn-line">No active E*TRADE session. Starting the ' +
            'OAuth flow opens an E*TRADE authorization page in a new tab.</p>' +
            '<button type="button" class="btn btn-primary" id="btnConnect">' +
            '<i class="bi bi-plug-fill" aria-hidden="true"></i> ' +
            'Connect to E*TRADE</button>'
        );
    }
    if (state === 'pending_verifier') {
        const url = safeHttpUrl(auth.authorize_url);
        const link = url
            ? `<a class="btn btn-outline-secondary" href="${escapeHTML(url)}"
                  target="_blank" rel="noopener noreferrer">
                  <i class="bi bi-box-arrow-up-right" aria-hidden="true"></i>
                  Open E*TRADE authorization</a>`
            : '<p class="conn-line conn-warn">No authorize URL was returned — ' +
              'restart the connection.</p>';
        return (
            '<p class="conn-line">Authorize this app on E*TRADE, then paste ' +
            'the verification code below.</p>' +
            `<div class="d-flex flex-wrap gap-2 mb-2">${link}</div>` +
            '<div class="d-flex flex-wrap align-items-end gap-2">' +
            '<div>' +
            '<label class="form-label" for="verifierCode">Verifier code</label>' +
            '<input type="text" class="form-control verifier-input" ' +
            'id="verifierCode" maxlength="32" autocomplete="off" ' +
            'placeholder="e.g. A1B2C">' +
            '</div>' +
            '<button type="button" class="btn btn-primary" id="btnSubmitVerifier">' +
            'Submit code</button>' +
            '</div>'
        );
    }
    if (state === 'connected') {
        return (
            '<p class="conn-line conn-ok"><i class="bi bi-check-circle-fill" ' +
            'aria-hidden="true"></i> Authenticated with E*TRADE. Tokens expire ' +
            'daily at midnight ET — renew before placing orders.</p>' +
            tokenTimesHTML(auth) +
            '<div class="d-flex flex-wrap gap-2">' +
            '<button type="button" class="btn btn-outline-secondary" id="btnRenew">' +
            '<i class="bi bi-arrow-clockwise" aria-hidden="true"></i> Renew token</button>' +
            '<button type="button" class="btn btn-outline-secondary" id="btnDisconnect">' +
            '<i class="bi bi-plug" aria-hidden="true"></i> Disconnect</button>' +
            '</div>'
        );
    }
    if (state === 'expired') {
        return (
            '<p class="conn-line conn-warn"><i class="bi ' +
            'bi-exclamation-triangle-fill" aria-hidden="true"></i> ' +
            'The E*TRADE token has EXPIRED — live requests will be rejected ' +
            'until you re-authenticate.</p>' +
            tokenTimesHTML(auth) +
            '<button type="button" class="btn btn-primary" id="btnConnect">' +
            '<i class="bi bi-plug-fill" aria-hidden="true"></i> ' +
            'Re-authenticate with E*TRADE</button>'
        );
    }
    return emptyStateHTML('bi-question-circle',
        `Unknown auth state: ${state || '(none)'}`,
        'Check the server log.');
}

// Key of the last connection-panel paint. The 10s status poll calls
// renderConnection on every tick; blindly rebuilding the panel would wipe
// the verifier <input> mid-entry (the operator is away at E*TRADE for well
// over one poll period), drop focus, and could swallow a Submit click when
// the button node is replaced between mousedown and mouseup. So the panel
// repaints ONLY when something it displays actually changed.
let lastConnectionKey = null;

function connectionRenderKey(auth) {
    return JSON.stringify([
        auth.state ?? null,
        auth.authorize_url ?? null,
        auth.token_issued_at ?? null,
        auth.renewed_at ?? null,
    ]);
}

function renderConnection(auth) {
    const body = document.getElementById('connectionBody');
    if (!body) return;
    const key = connectionRenderKey(auth);
    if (key === lastConnectionKey) return; // nothing visible changed

    // Belt and braces: if a repaint IS needed while a verifier entry is in
    // progress (e.g. the authorize_url changed), carry the typed code and
    // focus across the rebuild rather than destroying them.
    const oldInput = document.getElementById('verifierCode');
    const savedCode = oldInput ? oldInput.value : null;
    const hadFocus = oldInput !== null && document.activeElement === oldInput;

    body.innerHTML = connectionHTML(auth);
    lastConnectionKey = key;

    // Wire the buttons the current state rendered. Every auth transition
    // is an explicit user action — nothing here fires automatically.
    const btnConnect = document.getElementById('btnConnect');
    if (btnConnect) btnConnect.addEventListener('click', startAuth);
    const btnVerifier = document.getElementById('btnSubmitVerifier');
    if (btnVerifier) btnVerifier.addEventListener('click', submitVerifier);
    const verifierInput = document.getElementById('verifierCode');
    if (verifierInput) {
        if (savedCode) verifierInput.value = savedCode;
        if (hadFocus) verifierInput.focus();
        verifierInput.addEventListener('keydown', (ev) => {
            if (ev.key === 'Enter') submitVerifier();
        });
    }
    const btnRenew = document.getElementById('btnRenew');
    if (btnRenew) btnRenew.addEventListener('click', renewToken);
    const btnDisconnect = document.getElementById('btnDisconnect');
    if (btnDisconnect) btnDisconnect.addEventListener('click', disconnect);
}

function renderConnectionUnavailable(reason) {
    const body = document.getElementById('connectionBody');
    if (!body) return;
    lastConnectionKey = null; // next good status must repaint the panel
    body.innerHTML = emptyStateHTML('bi-cloud-slash',
        'Live trading unavailable',
        reason || 'The live-trading backend has not been configured.');
}

async function startAuth() {
    const restore = btnLoading(document.getElementById('btnConnect'));
    try {
        const data = await fetchJSON('/api/live/auth/start', { method: 'POST' });
        showToast('info', 'Authorization started — open the E*TRADE page and ' +
            'paste the verifier code.');
        applyStatusPatch(data.auth);
        const url = safeHttpUrl(data.authorize_url);
        if (url) window.open(url, '_blank', 'noopener');
    } catch (_) {
        // fetchJSON already toasted the error.
    } finally {
        restore();
    }
}

async function submitVerifier() {
    const input = document.getElementById('verifierCode');
    const code = (input ? input.value : '').trim();
    if (!code) {
        if (input) input.classList.add('is-invalid');
        showToast('warning', 'Enter the verifier code from E*TRADE first.');
        return;
    }
    const restore = btnLoading(document.getElementById('btnSubmitVerifier'));
    try {
        const data = await fetchJSON('/api/live/auth/verifier', {
            method: 'POST', body: JSON.stringify({ code }),
        });
        showToast('success', 'E*TRADE session established.');
        applyStatusPatch(data.auth);
    } catch (_) {
        // toasted by fetchJSON
    } finally {
        restore();
    }
}

async function renewToken() {
    const restore = btnLoading(document.getElementById('btnRenew'));
    try {
        const data = await fetchJSON('/api/live/auth/renew', { method: 'POST' });
        showToast(data.renewed ? 'success' : 'warning',
            data.renewed ? 'Token renewed.' : 'Token renewal did not succeed.');
        applyStatusPatch(data.auth);
    } catch (_) {
        // toasted by fetchJSON
    } finally {
        restore();
    }
}

async function disconnect() {
    if (!window.confirm('Disconnect from E*TRADE? You will need to run the ' +
        'full OAuth flow to reconnect.')) return;
    const restore = btnLoading(document.getElementById('btnDisconnect'));
    try {
        const data = await fetchJSON('/api/live/auth/disconnect',
            { method: 'POST' });
        showToast('info', 'Disconnected from E*TRADE.');
        applyStatusPatch(data.auth);
    } catch (_) {
        // toasted by fetchJSON
    } finally {
        restore();
    }
}

/** Re-render auth-driven chrome after an action returns a fresh C16 status
    (cheaper than waiting for the next poll, and never racy: the poll will
    confirm). */
function applyStatusPatch(auth) {
    if (!auth) { refreshStatus(); return; }
    const wasConnected = isConnected();
    liveStatus = { ...(liveStatus || {}), auth, env: auth.env };
    renderEnvBadge(auth.env, true);
    renderAuthBadge(auth.state, true);
    renderNavDot(auth.state, true);
    renderConnection(auth);
    if (isConnected() && (!wasConnected || !accountsLoaded)) {
        loadAccounts();
    } else if (!isConnected()) {
        accountsLoaded = false;
        renderAccountBody('expired');
    }
    syncOrderTicketGate();
}

/* ==========================================================================
   Kill switch (contract C17 via POST /api/live/killswitch)
   ========================================================================== */

function renderKillSwitch(kill) {
    const state = document.getElementById('ksState');
    const label = document.getElementById('ksStateLabel');
    const detail = document.getElementById('ksStateDetail');
    const btn = document.getElementById('btnKillSwitch');
    if (!state || !label || !detail || !btn) return;

    const engaged = !!(kill && kill.engaged === true);
    const known = !!(kill && typeof kill.engaged === 'boolean');

    if (!known) {
        state.className = 'ks-state ks-unknown';
        state.firstElementChild.className = 'bi bi-question-circle';
        label.textContent = 'UNKNOWN';
        detail.textContent = 'Kill-switch state could not be read.';
        btn.disabled = true;
        return;
    }
    if (engaged) {
        state.className = 'ks-state ks-engaged';
        state.firstElementChild.className = 'bi bi-sign-stop-fill';
        label.textContent = 'ENGAGED';
        detail.textContent =
            'Live order placement is BLOCKED. Disengage to resume trading.';
        btn.className = 'btn btn-outline-secondary btn-kill';
        btn.textContent = 'Disengage kill switch';
    } else {
        state.className = 'ks-state ks-off';
        state.firstElementChild.className = 'bi bi-shield-check';
        label.textContent = 'NOT ENGAGED';
        detail.textContent =
            'Live order placement is allowed (subject to auth + risk checks).';
        btn.className = 'btn btn-danger btn-kill';
        btn.textContent = 'Engage kill switch';
    }
    btn.disabled = false;
    renderKillBanner(engaged);
}

function openKillSwitchModal() {
    const engaged = !!(liveStatus && liveStatus.kill_switch &&
        liveStatus.kill_switch.engaged);
    const engaging = !engaged;

    document.getElementById('ksModalTitle').textContent = engaging
        ? 'Engage kill switch?' : 'Disengage kill switch?';
    document.getElementById('ksModalBody').textContent = engaging
        ? 'This immediately blocks ALL live order placement (preview and ' +
          'place) until the switch is disengaged. Cancels and quotes stay ' +
          'available.'
        : 'This re-enables live order placement. Confirm only if the reason ' +
          'for the halt is resolved.';
    const reasonGroup = document.getElementById('ksReasonGroup');
    const reasonInput = document.getElementById('ksReason');
    reasonGroup.classList.toggle('d-none', !engaging);
    reasonInput.value = '';
    reasonInput.classList.remove('is-invalid');

    const confirmBtn = document.getElementById('ksModalConfirm');
    confirmBtn.className = engaging ? 'btn btn-danger' : 'btn btn-primary';
    confirmBtn.textContent = engaging ? 'Engage — halt trading'
        : 'Disengage — resume trading';
    confirmBtn.dataset.engaging = engaging ? '1' : '0';

    bootstrap.Modal.getOrCreateInstance(
        document.getElementById('ksConfirmModal')).show();
}

async function confirmKillSwitch() {
    const confirmBtn = document.getElementById('ksModalConfirm');
    const engaging = confirmBtn.dataset.engaging === '1';
    const reasonInput = document.getElementById('ksReason');
    const reason = (reasonInput.value || '').trim();
    if (engaging && !reason) {
        reasonInput.classList.add('is-invalid');
        return;
    }
    const restore = btnLoading(confirmBtn);
    try {
        const data = await fetchJSON('/api/live/killswitch', {
            method: 'POST',
            body: JSON.stringify({ engaged: engaging, reason }),
        });
        bootstrap.Modal.getOrCreateInstance(
            document.getElementById('ksConfirmModal')).hide();
        showToast(engaging ? 'warning' : 'success', engaging
            ? 'Kill switch ENGAGED — live order placement blocked.'
            : 'Kill switch disengaged.');
        liveStatus = {
            ...(liveStatus || {}),
            kill_switch: { engaged: !!data.engaged },
        };
        renderKillSwitch(liveStatus.kill_switch);
        syncOrderTicketGate(); // a flip enables/blocks the Place button
        loadAudit(); // every flip is audit-logged — show it immediately
    } catch (_) {
        // toasted by fetchJSON
    } finally {
        restore();
    }
}

/* ==========================================================================
   Working orders (patient executor)
   ========================================================================== */

function workingOrderRow(order) {
    const stepList = Array.isArray(order.steps) ? order.steps : null;
    const steps = stepList ? stepList.length : Number(order.steps);
    // Current working limit: explicit field, else the executor's latest
    // repricing step (ExecutionReport steps: [{'ts','limit'}]).
    const limit = order.limit_price ??
        (stepList && stepList.length
            ? stepList[stepList.length - 1].limit : null);
    const elapsed = order.started_at
        ? (Date.now() - new Date(order.started_at).getTime()) / 1000
        : order.elapsed_seconds;
    let remaining = order.remaining_seconds;
    if (remaining === undefined && order.expires_at) {
        remaining = (new Date(order.expires_at).getTime() - Date.now()) / 1000;
    }
    const side = String(order.side || '').toUpperCase();
    return (
        '<tr>' +
        `<td class="num">${escapeHTML(order.instrument ?? order.symbol ?? '—')}</td>` +
        `<td><span class="${side === 'BUY' ? 'pnl-pos' : 'pnl-neg'}">${escapeHTML(side || '—')}</span></td>` +
        `<td class="num">${fmtNum(order.quantity, 0)}</td>` +
        `<td class="num">${fmtMoney(limit)}</td>` +
        `<td class="num">${Number.isFinite(steps) ? steps : '—'}</td>` +
        `<td class="num">${fmtDuration(elapsed)}</td>` +
        `<td class="num">${fmtDuration(remaining)}</td>` +
        '<td class="num">' +
        '<button type="button" class="btn btn-outline-secondary btn-sm js-cancel-order" ' +
        `data-order-id="${escapeHTML(String(order.order_id ?? ''))}">Cancel</button>` +
        '</td></tr>'
    );
}

function renderWorkingOrders(orders) {
    const body = document.getElementById('workingOrdersBody');
    const empty = document.getElementById('workingOrdersEmpty');
    const table = document.getElementById('workingOrdersTable');
    if (!body || !empty || !table) return;
    if (!orders || orders.length === 0) {
        body.innerHTML = '';
        table.classList.add('d-none');
        empty.innerHTML = emptyStateHTML('bi-hourglass',
            'No working orders',
            'Patient-executor orders (limit-at-mid, edge-decay cancel) ' +
            'appear here while they work.');
        return;
    }
    table.classList.remove('d-none');
    empty.innerHTML = '';
    body.innerHTML = orders.map(workingOrderRow).join('');
}

async function refreshWorkingOrders() {
    try {
        const data = await fetchJSON('/api/live/orders', { silent: true });
        renderWorkingOrders(data.orders || []);
    } catch (_) {
        renderWorkingOrders([]);
    }
}

async function cancelWorkingOrder(orderId, btn, accountKey) {
    if (!window.confirm(`Cancel order ${orderId}?`)) return;
    const restore = btnLoading(btn);
    // A GUI-placed order is cancelled through the shared client by passing
    // its account_id_key; a patient-executor working order omits it and the
    // route falls back to the live broker session.
    const body = accountKey ? JSON.stringify({ account_id_key: accountKey })
        : undefined;
    try {
        await fetchJSON(
            `/api/live/orders/${encodeURIComponent(orderId)}/cancel`,
            { method: 'POST', body });
        showToast('info', `Order ${orderId} cancelled.`);
        if (accountKey) removePlacedOrder(orderId);
        refreshWorkingOrders();
        loadAudit();
    } catch (_) {
        restore();
    }
}

// --- GUI-placed orders this session (cancellable via the shared client) ---
const _placedOrders = [];   // [{ orderId, accountKey }]

function recordPlacedOrder(orderId, accountKey) {
    if (orderId === undefined || orderId === null || !accountKey) return;
    if (!_placedOrders.some((o) => String(o.orderId) === String(orderId))) {
        _placedOrders.push({ orderId, accountKey });
    }
    renderPlacedOrders();
}

function removePlacedOrder(orderId) {
    const i = _placedOrders.findIndex(
        (o) => String(o.orderId) === String(orderId));
    if (i !== -1) _placedOrders.splice(i, 1);
    renderPlacedOrders();
}

function renderPlacedOrders() {
    const wrap = document.getElementById('placedOrdersWrap');
    const list = document.getElementById('placedOrdersList');
    if (!wrap || !list) return;
    list.replaceChildren();
    if (_placedOrders.length === 0) {
        wrap.classList.add('d-none');
        return;
    }
    wrap.classList.remove('d-none');
    for (const { orderId, accountKey } of _placedOrders) {
        const li = document.createElement('li');
        li.className = 'placed-orders-item';
        const label = document.createElement('span');
        label.className = 'num';
        label.textContent = `Order ${orderId}`;   // textContent: no injection
        const btn = document.createElement('button');
        btn.type = 'button';
        btn.className = 'btn btn-sm btn-outline-danger';
        btn.textContent = 'Cancel';
        btn.addEventListener('click',
            () => cancelWorkingOrder(orderId, btn, accountKey));
        li.append(label, btn);
        list.append(li);
    }
}

/* ==========================================================================
   Audit log (contract C18 via GET /api/live/audit)
   ========================================================================== */

function renderVerifyChip(verify) {
    const chip = document.getElementById('auditVerify');
    if (!chip) return;
    if (!verify) {
        chip.className = 'verify-chip verify-unknown';
        chip.innerHTML = '<i class="bi bi-shield" aria-hidden="true"></i> chain unverified';
    } else if (verify.ok) {
        chip.className = 'verify-chip verify-ok';
        chip.innerHTML = '<i class="bi bi-shield-check" aria-hidden="true"></i> hash chain OK';
    } else {
        chip.className = 'verify-chip verify-bad';
        chip.innerHTML = '<i class="bi bi-shield-x" aria-hidden="true"></i> ' +
            `CHAIN BROKEN at seq ${escapeHTML(String(verify.first_bad_seq ?? '?'))}`;
    }
}

function auditRow(entry) {
    let payload = '';
    try {
        payload = JSON.stringify(entry.payload ?? {}, null, 2);
    } catch (_) {
        payload = String(entry.payload);
    }
    return (
        '<tr>' +
        `<td class="num">${fmtNum(entry.seq, 0)}</td>` +
        `<td class="num">${fmtTimestamp(entry.ts)}</td>` +
        `<td>${escapeHTML(entry.env ?? '—')}</td>` +
        `<td>${escapeHTML(entry.actor ?? '—')}</td>` +
        `<td><span class="audit-event">${escapeHTML(entry.event_type ?? '—')}</span></td>` +
        '<td class="audit-payload"><details><summary>payload</summary>' +
        `<pre class="note-data-json">${escapeHTML(payload)}</pre>` +
        '</details></td></tr>'
    );
}

function updateAuditFilterOptions(entries) {
    const select = document.getElementById('auditEventType');
    if (!select) return;
    const current = select.value;
    (entries || []).forEach((e) => {
        if (e.event_type) auditEventTypes.add(String(e.event_type));
    });
    const known = Array.from(auditEventTypes).sort();
    select.innerHTML = '<option value="">All event types</option>' +
        known.map((t) =>
            `<option value="${escapeHTML(t)}">${escapeHTML(t)}</option>`,
        ).join('');
    select.value = current;
}

async function loadAudit(clamped) {
    const body = document.getElementById('auditBody');
    const empty = document.getElementById('auditEmpty');
    const info = document.getElementById('auditPageInfo');
    if (!body || !empty) return;
    const eventType = (document.getElementById('auditEventType') || {}).value || '';

    let data;
    try {
        const params = new URLSearchParams({
            limit: String(AUDIT_PAGE_SIZE),
            offset: String(auditOffset),
            verify: '1',
        });
        if (eventType) params.set('event_type', eventType);
        data = await fetchJSON(`/api/live/audit?${params}`, { silent: true });
    } catch (err) {
        body.innerHTML = '';
        renderVerifyChip(null);
        empty.innerHTML = emptyStateHTML('bi-journal-x', 'Audit log unavailable',
            err.message || 'The audit backend has not been configured.');
        if (info) info.textContent = '—';
        return;
    }

    const entries = data.entries || [];

    // Pager edge: when the total count is an exact multiple of the page
    // size, 'Older' can step onto an empty page (e.g. 'entries 101–100').
    // Clamp the offset back one page and re-fetch once instead of painting
    // a blank table; the single-retry guard keeps this loop-free.
    if (entries.length === 0 && auditOffset > 0 && clamped !== true) {
        auditOffset = Math.max(0, auditOffset - AUDIT_PAGE_SIZE);
        return loadAudit(true);
    }

    auditLastCount = entries.length;
    renderVerifyChip(data.verify);
    updateAuditFilterOptions(entries);

    if (entries.length === 0) {
        body.innerHTML = '';
        empty.innerHTML = auditOffset === 0
            ? emptyStateHTML('bi-journal', 'No audit entries yet',
                'Auth changes, kill-switch flips, orders and reconciliations ' +
                'are recorded here append-only.')
            : emptyStateHTML('bi-journal', 'End of audit log',
                'No entries beyond this point.');
    } else {
        empty.innerHTML = '';
        body.innerHTML = entries.map(auditRow).join('');
    }
    if (info) {
        info.textContent = (entries.length === 0
            ? 'no entries'
            : `entries ${auditOffset + 1}–${auditOffset + entries.length}`) +
            (eventType ? ` · ${eventType}` : '');
    }
    const prev = document.getElementById('auditPrev');
    const next = document.getElementById('auditNext');
    if (prev) prev.disabled = auditOffset === 0;
    if (next) next.disabled = auditLastCount < AUDIT_PAGE_SIZE;
}

/* ==========================================================================
   Reconciliation (contract C19 via POST /api/live/reconcile)
   ========================================================================== */

function mismatchRow(m) {
    return (
        '<tr>' +
        `<td>${escapeHTML(m.kind ?? '—')}</td>` +
        `<td>${escapeHTML(m.symbol ?? '—')}</td>` +
        `<td class="num">${fmtNum(m.local)}</td>` +
        `<td class="num">${fmtNum(m.broker)}</td>` +
        '</tr>'
    );
}

function renderReconciliation(result) {
    const body = document.getElementById('reconBody');
    if (!body) return;
    if (!result) {
        body.innerHTML = emptyStateHTML('bi-check2-square',
            'No reconciliation yet',
            'Compares the local book against E*TRADE positions and cash. ' +
            'Mismatches engage the kill switch.');
        return;
    }
    const checked = `<div class="provenance-caption mt-2">checked ` +
        `${fmtTimestamp(result.checked_at)}</div>`;
    if (result.ok) {
        body.innerHTML =
            '<div class="recon-result recon-ok">' +
            '<i class="bi bi-check-circle-fill" aria-hidden="true"></i> ' +
            'In sync — local book matches the broker.</div>' + checked;
        return;
    }
    const mismatches = result.mismatches || [];
    body.innerHTML =
        '<div class="recon-result recon-bad">' +
        '<i class="bi bi-x-octagon-fill" aria-hidden="true"></i> ' +
        `${mismatches.length} mismatch(es) found` +
        (result.kill_switch_engaged
            ? ' — <strong>kill switch engaged</strong>' : '') +
        '</div>' +
        '<div class="table-responsive mt-2"><table class="table">' +
        '<thead><tr><th>Kind</th><th>Symbol</th>' +
        '<th class="num">Local</th><th class="num">Broker</th></tr></thead>' +
        `<tbody>${mismatches.map(mismatchRow).join('')}</tbody>` +
        '</table></div>' + checked;
}

async function runReconcile() {
    const restore = btnLoading(document.getElementById('btnReconcile'));
    try {
        const result = await fetchJSON('/api/live/reconcile', { method: 'POST' });
        renderReconciliation(result);
        if (result.ok) {
            showToast('success', 'Reconciliation OK — local book matches broker.');
        } else {
            showToast('error', 'Reconciliation mismatch — kill switch engaged.');
        }
        refreshStatus();  // kill switch / status may have flipped
        loadAudit();
        loadParity();  // reconciliation follows trading — refresh fill parity
    } catch (_) {
        // toasted by fetchJSON (covers 409 no-live-session and 503s)
    } finally {
        restore();
    }
}

/* ==========================================================================
   Account panel + quotes + order ticket (live trading surface)

   Render rules (mirror the connection panel / scheduler protections):
   * The account <select>, the quote input and EVERY order-ticket field
     belong to the operator — the 10s status poll NEVER rebuilds them. The
     account select rebuilds only when the set of accounts actually changes
     (render-keyed); balances/positions repaint on explicit selection.
   * A fresh PREVIEW is required before PLACE; ANY ticket-field edit voids
     the preview so a stale preview can never be placed.
   ========================================================================== */

let accountsLoaded = false;        // accounts fetched at least once this connection
let lastAccountsKey = null;        // render-key of the account <select>
let selectedAccountKey = null;     // currently-selected accountIdKey
let knownAccounts = [];            // last accounts payload (for the ticket label)
let currentOrderRef = null;        // fresh preview's order_ref (null = stale/none)

/** Is the live session connected right now? */
function isConnected() {
    return !!(liveStatus && liveStatus.auth && liveStatus.auth.state === 'connected');
}

function killSwitchEngaged() {
    return !!(liveStatus && liveStatus.kill_switch && liveStatus.kill_switch.engaged);
}

/* -------------------- Account panel -------------------- */

function accountOptionLabel(acct) {
    const parts = [];
    if (acct.accountType) parts.push(acct.accountType);
    if (acct.accountDesc) parts.push(acct.accountDesc);
    if (acct.accountId_masked) parts.push(acct.accountId_masked);
    return parts.join(' · ') || (acct.accountIdKey || 'account');
}

function accountsRenderKey(accounts) {
    return JSON.stringify(accounts.map((a) => [
        a.accountIdKey ?? null, a.accountType ?? null,
        a.accountDesc ?? null, a.accountId_masked ?? null,
    ]));
}

function renderAccountSelect(accounts) {
    const select = document.getElementById('accountSelect');
    if (!select) return;
    const key = accountsRenderKey(accounts);
    if (key === lastAccountsKey) return; // nothing changed — keep selection
    lastAccountsKey = key;

    select.innerHTML = '';
    accounts.forEach((acct) => {
        const opt = document.createElement('option');
        opt.value = acct.accountIdKey || '';
        opt.textContent = accountOptionLabel(acct); // textContent only
        select.appendChild(opt);
    });
    select.classList.toggle('d-none', accounts.length === 0);

    // Preserve a still-present selection; otherwise pick the first account.
    if (selectedAccountKey &&
        accounts.some((a) => a.accountIdKey === selectedAccountKey)) {
        select.value = selectedAccountKey;
    } else if (accounts.length) {
        selectedAccountKey = accounts[0].accountIdKey;
        select.value = selectedAccountKey;
    } else {
        selectedAccountKey = null;
    }
    updateTicketAccountLabel(accounts);
}

function updateTicketAccountLabel(accounts) {
    const label = document.getElementById('ticketAccountLabel');
    if (!label) return;
    const acct = (accounts || []).find(
        (a) => a.accountIdKey === selectedAccountKey);
    label.textContent = acct ? accountOptionLabel(acct) : '—';
}

function renderAccountBody(state) {
    // state: 'loading' | 'empty' | 'expired'
    const body = document.getElementById('accountBody');
    if (!body) return;
    if (state === 'expired') {
        body.innerHTML = emptyStateHTML('bi-plug',
            'Session not connected',
            'Connect to E*TRADE to load accounts, balances and positions.');
    } else if (state === 'loading') {
        body.innerHTML = emptyStateHTML('bi-hourglass', 'Loading accounts…',
            'Reading your E*TRADE accounts.');
    } else {
        body.innerHTML = emptyStateHTML('bi-wallet2', 'No accounts',
            'E*TRADE returned no accounts for this session.');
    }
}

/** Metric chips + positions table for the selected account. */
function renderAccountDetail(balances, positions) {
    const body = document.getElementById('accountBody');
    if (!body) return;
    const netValue = pickBalance(balances,
        ['netAccountValue', 'netValue', 'accountValue']);
    const cash = pickBalance(balances,
        ['cashAvailableForInvestment', 'cashBalance',
            'netCash', 'cashAvailable']);

    const chips =
        '<div class="metric-chips mb-2">' +
        metricChip('Net account value', fmtMoney(netValue)) +
        metricChip('Cash available', fmtMoney(cash)) +
        '</div>';

    const rows = (positions || []).map(positionRow).join('');
    const table = (positions && positions.length)
        ? '<div class="table-responsive"><table class="table">' +
          '<thead><tr><th>Symbol</th><th class="num">Qty</th>' +
          '<th class="num">Market value</th><th class="num">P&amp;L</th>' +
          '</tr></thead>' + `<tbody>${rows}</tbody></table></div>`
        : emptyStateHTML('bi-inboxes', 'No open positions',
            'This account holds no positions right now.');
    body.innerHTML = chips + table;
}

/** First finite numeric value among candidate balance keys (nested-safe). */
function pickBalance(balances, keys) {
    if (!balances || typeof balances !== 'object') return null;
    for (const key of keys) {
        const v = deepFind(balances, key);
        if (Number.isFinite(Number(v))) return Number(v);
    }
    return null;
}

function deepFind(obj, key, depth = 0) {
    if (!obj || typeof obj !== 'object' || depth > 4) return undefined;
    if (Object.prototype.hasOwnProperty.call(obj, key)) return obj[key];
    for (const val of Object.values(obj)) {
        if (val && typeof val === 'object') {
            const found = deepFind(val, key, depth + 1);
            if (found !== undefined) return found;
        }
    }
    return undefined;
}

function metricChip(label, value) {
    return '<div class="metric-chip">' +
        `<div class="metric-chip-label">${escapeHTML(label)}</div>` +
        `<div class="metric-chip-value num">${escapeHTML(value)}</div></div>`;
}

function positionRow(pos) {
    const symbol = pos.symbolDescription || pos.symbol ||
        (pos.Product && pos.Product.symbol) || '—';
    const qty = pos.quantity ?? pos.qty;
    const mv = pos.marketValue ?? pos.totalMarketValue ??
        (pos.Complete && pos.Complete.marketValue);
    const pnl = pos.totalGainPct !== undefined ? null
        : (pos.totalGain ?? pos.totalGainLoss ?? pos.daysGain);
    const pnlCell = (pnl === undefined || pnl === null || pnl === '')
        ? '<td class="num">—</td>'
        : `<td class="num ${pnlClass(Number(pnl))}">${fmtMoney(pnl)}</td>`;
    return '<tr>' +
        `<td class="num">${escapeHTML(String(symbol))}</td>` +
        `<td class="num">${fmtNum(qty, 0)}</td>` +
        `<td class="num">${fmtMoney(mv)}</td>` +
        pnlCell + '</tr>';
}

async function loadAccounts() {
    if (!isConnected()) {
        accountsLoaded = false;
        lastAccountsKey = null;
        renderAccountBody('expired');
        const select = document.getElementById('accountSelect');
        if (select) select.classList.add('d-none');
        syncOrderTicketGate();
        return;
    }
    let data;
    try {
        data = await fetchJSON('/api/live/accounts', { silent: true });
    } catch (err) {
        if (err.status === 401) renderAccountBody('expired');
        else renderAccountBody('empty');
        return;
    }
    accountsLoaded = true;
    const accounts = data.accounts || [];
    knownAccounts = accounts;
    renderAccountSelect(accounts);
    if (accounts.length === 0) {
        renderAccountBody('empty');
    } else {
        loadAccountDetail(selectedAccountKey);
    }
    syncOrderTicketGate();
}

async function loadAccountDetail(idKey) {
    if (!idKey) { renderAccountBody('empty'); return; }
    renderAccountBody('loading');
    let balances = {};
    let positions = [];
    try {
        const b = await fetchJSON(
            `/api/live/accounts/${encodeURIComponent(idKey)}/balances`,
            { silent: true });
        balances = b.balances || {};
    } catch (err) {
        if (err.status === 401) { renderAccountBody('expired'); return; }
    }
    try {
        const p = await fetchJSON(
            `/api/live/accounts/${encodeURIComponent(idKey)}/portfolio`,
            { silent: true });
        positions = p.positions || [];
    } catch (_) {
        // A balances-only render is still useful; leave positions empty.
    }
    renderAccountDetail(balances, positions);
}

/* -------------------- Quotes -------------------- */

function quoteRow(symbol, q) {
    return '<tr>' +
        `<td class="num">${escapeHTML(symbol)}</td>` +
        `<td class="num">${fmtMoney(q && q.bid)}</td>` +
        `<td class="num">${fmtMoney(q && q.ask)}</td>` +
        `<td class="num">${fmtMoney(q && q.last)}</td></tr>`;
}

function renderQuotes(quotes, requested) {
    const table = document.getElementById('quoteTable');
    const body = document.getElementById('quoteBody');
    const empty = document.getElementById('quoteEmpty');
    if (!table || !body || !empty) return;
    const syms = requested && requested.length
        ? requested : Object.keys(quotes || {});
    if (!syms.length) {
        table.classList.add('d-none');
        body.innerHTML = '';
        empty.innerHTML = emptyStateHTML('bi-graph-up',
            'No quotes', 'Enter one or more symbols and look them up.');
        return;
    }
    empty.innerHTML = '';
    table.classList.remove('d-none');
    body.innerHTML = syms.map((s) => quoteRow(s, (quotes || {})[s])).join('');
}

async function lookupQuotes() {
    const input = document.getElementById('quoteSymbols');
    const raw = (input ? input.value : '').trim();
    if (!raw) {
        showToast('warning', 'Enter at least one symbol.');
        return;
    }
    const symbols = raw.split(',').map((s) => s.trim()).filter(Boolean);
    if (symbols.length > 25) {
        showToast('warning', 'Up to 25 symbols at a time.');
        return;
    }
    const restore = btnLoading(document.getElementById('btnQuotes'));
    try {
        const params = new URLSearchParams({ symbols: symbols.join(',') });
        const data = await fetchJSON(`/api/live/quotes?${params}`);
        renderQuotes(data.quotes || {}, data.requested || symbols);
    } catch (_) {
        // toasted by fetchJSON
    } finally {
        restore();
    }
}

/* -------------------- Order ticket -------------------- */

const TICKET_KINDS = ['equity', 'option', 'spread'];

/** Read the current ticket kind ('equity' fallback). */
function ticketKind() {
    const sel = document.getElementById('ticketKind');
    const v = sel ? sel.value : 'equity';
    return TICKET_KINDS.includes(v) ? v : 'equity';
}

/** Show only the active kind's pane and the right spread sub-pane. */
function renderTicketPanes() {
    const kind = ticketKind();
    [['paneEquity', 'equity'], ['paneOption', 'option'],
     ['paneSpread', 'spread']].forEach(([id, k]) => {
        const el = document.getElementById(id);
        if (el) el.classList.toggle('d-none', k !== kind);
    });
    const preset = (document.getElementById('spPreset') || {}).value
        || 'vertical';
    const vert = document.getElementById('spVertical');
    const condor = document.getElementById('spCondor');
    if (vert) vert.classList.toggle('d-none', preset !== 'vertical');
    if (condor) condor.classList.toggle('d-none', preset !== 'iron_condor');
}

function numVal(id) {
    const el = document.getElementById(id);
    if (!el) return null;
    const raw = (el.value || '').trim();
    if (raw === '') return null;
    const n = Number(raw);
    return Number.isFinite(n) ? n : NaN;
}

function strVal(id) {
    const el = document.getElementById(id);
    return el ? (el.value || '').trim() : '';
}

/** Build the POST /order/preview body from the ticket, or null + a toast. */
function buildPreviewBody() {
    const kind = ticketKind();
    const account_id_key = selectedAccountKey;
    if (!account_id_key) {
        showToast('warning', 'Select an account first.');
        return null;
    }
    const body = { account_id_key, kind };
    if (kind === 'equity') {
        body.symbol = strVal('eqSymbol').toUpperCase();
        body.side = strVal('eqSide').toUpperCase();
        body.quantity = numVal('eqQty');
        const limit = numVal('eqLimit');
        if (limit !== null) body.limit_price = limit;
        if (!body.symbol || !(body.quantity > 0)) {
            showToast('warning', 'Symbol and a positive quantity are required.');
            return null;
        }
    } else if (kind === 'option') {
        body.symbol = strVal('optSymbol').toUpperCase();
        body.call_put = strVal('optCallPut').toUpperCase();
        body.strike = numVal('optStrike');
        body.expiry = strVal('optExpiry');
        body.action = strVal('optAction').toUpperCase();
        body.quantity = numVal('optQty');
        const limit = numVal('optLimit');
        if (limit !== null) body.limit_price = limit;
        if (!body.symbol || !(body.strike > 0) || !body.expiry ||
            !(body.quantity > 0)) {
            showToast('warning',
                'Underlying, strike, expiry and quantity are required.');
            return null;
        }
    } else {
        const legs = buildSpreadLegs();
        if (!legs) return null; // toasted inside
        body.legs = legs;
        const net = numVal('spNetCredit');
        if (net === null || !Number.isFinite(net) || net === 0) {
            showToast('warning',
                'Net credit/debit must be a non-zero number.');
            return null;
        }
        body.net_price = net;
    }
    return body;
}

/** Vertical (2) or iron-condor (4) legs from the spread pane, or null. */
function buildSpreadLegs() {
    const symbol = strVal('spSymbol').toUpperCase();
    const expiry = strVal('spExpiry');
    const preset = (document.getElementById('spPreset') || {}).value
        || 'vertical';
    if (!symbol || !expiry) {
        showToast('warning', 'Underlying and expiry are required.');
        return null;
    }
    if (preset === 'vertical') {
        const right = strVal('spVType').toUpperCase();
        const shortK = numVal('spVShort');
        const longK = numVal('spVLong');
        const qty = numVal('spVQty');
        if (!(shortK > 0) || !(longK > 0) || !(qty > 0)) {
            showToast('warning',
                'Both strikes and a positive quantity are required.');
            return null;
        }
        return [
            { symbol, call_put: right, strike: shortK, expiry,
              action: 'SELL_OPEN', quantity: qty },
            { symbol, call_put: right, strike: longK, expiry,
              action: 'BUY_OPEN', quantity: qty },
        ];
    }
    // Iron condor: short put + long put + short call + long call.
    const sp = numVal('spCShortPut');
    const lp = numVal('spCLongPut');
    const sc = numVal('spCShortCall');
    const lc = numVal('spCLongCall');
    const qty = numVal('spCQty');
    if (!(sp > 0) || !(lp > 0) || !(sc > 0) || !(lc > 0) || !(qty > 0)) {
        showToast('warning',
            'All four strikes and a positive quantity are required.');
        return null;
    }
    return [
        { symbol, call_put: 'PUT', strike: sp, expiry,
          action: 'SELL_OPEN', quantity: qty },
        { symbol, call_put: 'PUT', strike: lp, expiry,
          action: 'BUY_OPEN', quantity: qty },
        { symbol, call_put: 'CALL', strike: sc, expiry,
          action: 'SELL_OPEN', quantity: qty },
        { symbol, call_put: 'CALL', strike: lc, expiry,
          action: 'BUY_OPEN', quantity: qty },
    ];
}

/** Render the preview summary and the order_ref state. */
function renderPreviewResult(preview) {
    const el = document.getElementById('previewResult');
    if (!el) return;
    const ids = (preview && preview.previewIds) || [];
    let text;
    try {
        text = JSON.stringify(preview || {}, null, 2);
    } catch (_) {
        text = String(preview);
    }
    el.innerHTML =
        '<div class="preview-ok"><i class="bi bi-check-circle-fill" ' +
        'aria-hidden="true"></i> Previewed — review the estimate, then ' +
        'Place.</div>' +
        `<dl class="token-times"><dt>Preview IDs</dt>` +
        `<dd class="num">${escapeHTML(ids.join(', ') || '—')}</dd></dl>` +
        `<pre class="note-data-json">${escapeHTML(text)}</pre>`;
}

function clearPreviewResult() {
    const el = document.getElementById('previewResult');
    if (el) el.innerHTML = '';
}

/** A fresh preview enables Place; voiding it (any edit) disables Place. */
function setOrderRef(ref) {
    currentOrderRef = ref || null;
    syncOrderTicketGate();
}

/** Single source of truth for Place-button enablement + the locked state. */
function syncOrderTicketGate() {
    const form = document.getElementById('orderTicketForm');
    const locked = document.getElementById('ticketDisconnected');
    const placeBtn = document.getElementById('btnPlace');
    const killNote = document.getElementById('ticketKillNote');
    if (!form || !locked || !placeBtn) return;

    const ready = isConnected() && accountsLoaded && !!selectedAccountKey;
    form.classList.toggle('d-none', !ready);
    locked.classList.toggle('d-none', ready);

    const engaged = killSwitchEngaged();
    if (killNote) killNote.classList.toggle('d-none', !engaged);

    // Place requires a fresh preview AND no kill switch.
    const canPlace = ready && !!currentOrderRef && !engaged;
    placeBtn.disabled = !canPlace;
    placeBtn.title = engaged
        ? 'Kill switch engaged — order placement is blocked.'
        : (currentOrderRef ? 'Place the previewed order.'
            : 'Preview an order first.');

    const previewBtn = document.getElementById('btnPreview');
    if (previewBtn) previewBtn.disabled = !ready || engaged;
}

/** ANY ticket edit voids the current preview (a stale preview is never
    placeable — mirrors the verifier-input / scheduler-input protection). */
function voidPreviewOnEdit() {
    if (currentOrderRef !== null) {
        currentOrderRef = null;
        clearPreviewResult();
    }
    syncOrderTicketGate();
}

async function previewOrder() {
    const body = buildPreviewBody();
    if (!body) return;
    const restore = btnLoading(document.getElementById('btnPreview'));
    try {
        const data = await fetchJSON('/api/live/order/preview', {
            method: 'POST', body: JSON.stringify(body),
        });
        renderPreviewResult(data.preview || {});
        setOrderRef(data.order_ref);
        showToast('success', 'Order previewed — review, then Place.');
        loadAudit(); // preview is audit-logged
    } catch (_) {
        setOrderRef(null);
        // toasted by fetchJSON (covers 409 kill-switch, 401, 422)
    } finally {
        restore();
    }
}

function openPlaceModal() {
    if (!currentOrderRef || killSwitchEngaged()) return;
    const summary = document.getElementById('placeModalSummary');
    if (summary) {
        const preview = document.getElementById('previewResult');
        summary.textContent = preview
            ? preview.textContent.slice(0, 2000) : '';
    }
    bootstrap.Modal.getOrCreateInstance(
        document.getElementById('placeConfirmModal')).show();
}

async function confirmPlaceOrder() {
    if (!currentOrderRef) return;
    const ref = currentOrderRef;
    const restore = btnLoading(document.getElementById('placeModalConfirm'));
    try {
        const data = await fetchJSON('/api/live/order/place', {
            method: 'POST', body: JSON.stringify({ order_ref: ref }),
        });
        bootstrap.Modal.getOrCreateInstance(
            document.getElementById('placeConfirmModal')).hide();
        const order = data.order || {};
        showToast('success',
            `Order placed — id ${order.orderId ?? '(pending)'}.`);
        // Track it so it can be cancelled (via the shared client) from the
        // ticket — GUI-placed orders are not patient-executor working orders.
        if (order.orderId !== undefined && order.orderId !== null) {
            recordPlacedOrder(order.orderId, selectedAccountKey);
        }
        // The ref is single-use and now consumed: void it locally too.
        setOrderRef(null);
        clearPreviewResult();
        refreshWorkingOrders();
        loadAudit();
        if (selectedAccountKey) loadAccountDetail(selectedAccountKey);
    } catch (err) {
        // Kill switch / circuit breaker (409), reauthorize (401), reject
        // (422) all toasted by fetchJSON. Keep the order UN-placed; the ref
        // is consumed server-side on success only, but a 404 (expired) or a
        // place failure means the operator must re-preview.
        if (err.status === 404 || err.status === 409) {
            bootstrap.Modal.getOrCreateInstance(
                document.getElementById('placeConfirmModal')).hide();
            setOrderRef(null);
        }
        restore();
    }
}

/* ==========================================================================
   Scheduler (market-hours evaluate_once loop via /api/live/scheduler)

   Render rules:
   * EVERY dynamic field is painted with textContent on pre-existing nodes —
     no innerHTML rebuilds — so the 10s poll can never wipe an in-progress
     input or swallow a click (the same failure class the connection
     panel's render-keying fix guards against; this card avoids it by
     construction).
   * The interval <input> belongs to the OPERATOR while they are editing:
     the poll only fills it while it is untouched and unfocused.
   ========================================================================== */

let schedIntervalDirty = false;

/** Plain-text timestamp (textContent — no HTML escaping wanted). */
function fmtTimestampText(iso) {
    if (!iso) return '—';
    const d = new Date(iso);
    return Number.isNaN(d.getTime()) ? String(iso) : d.toLocaleString();
}

const SCHED_STATES = {
    running: ['ks-state ks-off', 'bi bi-play-circle-fill', 'RUNNING'],
    paused: ['ks-state ks-engaged', 'bi bi-pause-circle-fill', 'PAUSED'],
    stopped: ['ks-state ks-unknown', 'bi bi-stop-circle', 'STOPPED'],
    unknown: ['ks-state ks-unknown', 'bi bi-question-circle', 'UNKNOWN'],
};

function paintSchedulerState(kind, detail) {
    const state = document.getElementById('schedState');
    const label = document.getElementById('schedStateLabel');
    const detailEl = document.getElementById('schedStateDetail');
    if (!state || !label || !detailEl) return;
    const [cls, icon, text] = SCHED_STATES[kind] || SCHED_STATES.unknown;
    state.className = cls;
    state.firstElementChild.className = icon;
    label.textContent = text;
    detailEl.textContent = detail;
}

function renderScheduler(data) {
    const running = !!(data && data.running === true);
    const pausedReason = data ? data.paused_reason : null;

    if (running) {
        paintSchedulerState('running',
            'Evaluating automatically during NYSE market hours. ' +
            'Kill switch and circuit breaker are checked every cycle.');
    } else if (pausedReason) {
        paintSchedulerState('paused',
            `Paused: ${pausedReason}. Resolve the halt, then start ` +
            'manually — the scheduler never resumes on its own.');
    } else {
        paintSchedulerState('stopped',
            'Not running. Start to evaluate automatically during market ' +
            'hours.');
    }

    const fields = {
        schedInterval: Number.isFinite(Number(data.interval_minutes))
            ? `${data.interval_minutes} min` : '—',
        schedLastRun: fmtTimestampText(data.last_run),
        schedNextRun: running ? fmtTimestampText(data.next_run_estimate) : '—',
        schedRunsToday: Number.isFinite(Number(data.runs_today))
            ? String(data.runs_today) : '—',
        schedErrors: Number.isFinite(Number(data.consecutive_errors))
            ? String(data.consecutive_errors) : '—',
    };
    Object.entries(fields).forEach(([id, value]) => {
        const el = document.getElementById(id);
        if (el) el.textContent = value;
    });

    const btnStart = document.getElementById('btnSchedStart');
    const btnStop = document.getElementById('btnSchedStop');
    if (btnStart) btnStart.disabled = running;
    if (btnStop) btnStop.disabled = !running;

    // The interval input is the operator's while focused/edited; the poll
    // only keeps the placeholder fresh and fills an untouched field.
    const input = document.getElementById('schedIntervalInput');
    if (input && data.interval_minutes != null) {
        input.placeholder = String(data.interval_minutes);
        if (!schedIntervalDirty && document.activeElement !== input) {
            input.value = String(data.interval_minutes);
        }
    }
}

function renderSchedulerUnavailable(reason) {
    paintSchedulerState('unknown',
        reason || 'The scheduler backend has not been configured.');
    const btnStart = document.getElementById('btnSchedStart');
    const btnStop = document.getElementById('btnSchedStop');
    if (btnStart) btnStart.disabled = true;
    if (btnStop) btnStop.disabled = true;
}

async function refreshScheduler() {
    try {
        const data = await fetchJSON('/api/live/scheduler', { silent: true });
        renderScheduler(data);
    } catch (err) {
        renderSchedulerUnavailable(
            (err.body && (err.body.reason || err.body.error)) || err.message);
    }
}

/** The interval to send with 'start': null = omit (server keeps current),
    undefined = invalid input (caller aborts). */
function schedIntervalToSend() {
    const input = document.getElementById('schedIntervalInput');
    if (!input) return null;
    const raw = (input.value || '').trim();
    if (!raw) { input.classList.remove('is-invalid'); return null; }
    const n = Number(raw);
    if (!Number.isInteger(n) || n < 1 || n > 240) {
        input.classList.add('is-invalid');
        return undefined;
    }
    input.classList.remove('is-invalid');
    return n;
}

async function startScheduler() {
    const interval = schedIntervalToSend();
    if (interval === undefined) {
        showToast('warning', 'Interval must be a whole number of minutes ' +
            'between 1 and 240.');
        return;
    }
    const label = interval === null ? 'the configured interval'
        : `every ${interval} minutes`;
    if (!window.confirm('Start the market-hours scheduler? It will run ' +
        `evaluate_once() ${label} during NYSE hours until stopped, ` +
        'and pause itself on any kill-switch or circuit-breaker halt.')) {
        return;
    }
    const body = { action: 'start' };
    if (interval !== null) body.interval_minutes = interval;
    const restore = btnLoading(document.getElementById('btnSchedStart'));
    try {
        const data = await fetchJSON('/api/live/scheduler', {
            method: 'POST', body: JSON.stringify(body),
        });
        showToast('success', 'Scheduler started.');
        schedIntervalDirty = false;
        renderScheduler(data);
        loadAudit(); // start is audit-logged — show it immediately
    } catch (_) {
        // toasted by fetchJSON
    } finally {
        restore();
    }
}

async function stopScheduler() {
    if (!window.confirm('Stop the scheduler? Automatic evaluations halt ' +
        'until it is started again. Working orders are not affected.')) {
        return;
    }
    const restore = btnLoading(document.getElementById('btnSchedStop'));
    try {
        const data = await fetchJSON('/api/live/scheduler', {
            method: 'POST', body: JSON.stringify({ action: 'stop' }),
        });
        showToast('info', 'Scheduler stopped.');
        renderScheduler(data);
        loadAudit(); // stop is audit-logged — show it immediately
    } catch (_) {
        // toasted by fetchJSON
    } finally {
        restore();
    }
}

/* ==========================================================================
   Token keep-alive card + expiry countdown + morning reconnect prompt
   (Piece 2). The keep-alive loop is renew-only and NEVER trades.
   ========================================================================== */

let kaIntervalDirty = false;
let lastKeepalive = null;     // last /api/live/keepalive payload (for the banner)
let reauthNotified = false;   // de-dupe the browser notification per re-auth event

function paintKaState(kind, detail) {
    const state = document.getElementById('kaState');
    const label = document.getElementById('kaStateLabel');
    const detailEl = document.getElementById('kaStateDetail');
    if (!state || !label || !detailEl) return;
    const [cls, icon, text] = SCHED_STATES[kind] || SCHED_STATES.unknown;
    state.className = cls;
    state.firstElementChild.className = icon;
    label.textContent = text;
    detailEl.textContent = detail;
}

function renderKeepAlive(data) {
    const running = !!(data && data.running === true);
    const pausedReason = data ? data.paused_reason : null;
    lastKeepalive = data || null;

    if (running) {
        paintKaState('running',
            'Renewing automatically during NYSE market hours — clears the ' +
            "~2h idle timeout. It never trades.");
    } else if (pausedReason === 'token_expired') {
        paintKaState('paused',
            'Paused: the daily token expired. Reconnect to resume keep-alive.');
    } else if (pausedReason) {
        paintKaState('paused',
            `Paused: ${pausedReason}. Start again after resolving.`);
    } else {
        paintKaState('stopped',
            'Not running. Connecting starts it automatically; you can also ' +
            'start it here.');
    }

    const fields = {
        kaInterval: Number.isFinite(Number(data.interval_minutes))
            ? `${data.interval_minutes} min` : '—',
        kaLastRenew: fmtTimestampText(data.last_renew),
        kaNextRun: running ? fmtTimestampText(data.next_run_estimate) : '—',
        kaRenewsToday: Number.isFinite(Number(data.renews_today))
            ? String(data.renews_today) : '—',
        kaFailures: Number.isFinite(Number(data.consecutive_failures))
            ? String(data.consecutive_failures) : '—',
    };
    Object.entries(fields).forEach(([id, value]) => {
        const el = document.getElementById(id);
        if (el) el.textContent = value;
    });

    const btnStart = document.getElementById('btnKaStart');
    const btnStop = document.getElementById('btnKaStop');
    if (btnStart) btnStart.disabled = running;
    if (btnStop) btnStop.disabled = !running;

    const input = document.getElementById('kaIntervalInput');
    if (input && data.interval_minutes != null) {
        input.placeholder = String(data.interval_minutes);
        if (!kaIntervalDirty && document.activeElement !== input) {
            input.value = String(data.interval_minutes);
        }
    }
    updateReconnectBanner();
}

function renderKeepAliveUnavailable(reason) {
    paintKaState('unknown',
        reason || 'The keep-alive backend has not been configured.');
    lastKeepalive = null;
    ['kaInterval', 'kaLastRenew', 'kaNextRun', 'kaRenewsToday', 'kaFailures']
        .forEach((id) => {
            const el = document.getElementById(id);
            if (el) el.textContent = '—';
        });
    const btnStart = document.getElementById('btnKaStart');
    const btnStop = document.getElementById('btnKaStop');
    if (btnStart) btnStart.disabled = true;
    if (btnStop) btnStop.disabled = true;
    updateReconnectBanner();
}

async function refreshKeepAlive() {
    try {
        const data = await fetchJSON('/api/live/keepalive', { silent: true });
        renderKeepAlive(data);
    } catch (err) {
        renderKeepAliveUnavailable(
            (err.body && (err.body.reason || err.body.error)) || err.message);
    }
}

/** The interval to send with 'start': null = omit, undefined = invalid. */
function kaIntervalToSend() {
    const input = document.getElementById('kaIntervalInput');
    if (!input) return null;
    const raw = (input.value || '').trim();
    if (!raw) { input.classList.remove('is-invalid'); return null; }
    const n = Number(raw);
    if (!Number.isInteger(n) || n < 1 || n > 60) {
        input.classList.add('is-invalid');
        return undefined;
    }
    input.classList.remove('is-invalid');
    return n;
}

async function startKeepAlive() {
    const interval = kaIntervalToSend();
    if (interval === undefined) {
        showToast('warning', 'Interval must be a whole number of minutes ' +
            'between 1 and 60.');
        return;
    }
    const body = { action: 'start' };
    if (interval !== null) body.interval_minutes = interval;
    const restore = btnLoading(document.getElementById('btnKaStart'));
    try {
        const data = await fetchJSON('/api/live/keepalive', {
            method: 'POST', body: JSON.stringify(body),
        });
        showToast('success', 'Token keep-alive started.');
        kaIntervalDirty = false;
        renderKeepAlive(data);
    } catch (_) {
        // toasted by fetchJSON
    } finally {
        restore();
    }
}

async function stopKeepAlive() {
    const restore = btnLoading(document.getElementById('btnKaStop'));
    try {
        const data = await fetchJSON('/api/live/keepalive', {
            method: 'POST', body: JSON.stringify({ action: 'stop' }),
        });
        showToast('info', 'Token keep-alive stopped.');
        renderKeepAlive(data);
    } catch (_) {
        // toasted by fetchJSON
    } finally {
        restore();
    }
}

/* -------------------- Token-expiry countdown -------------------- */

/** Milliseconds until the next midnight in America/New_York — when the
    E*TRADE token hard-expires. Computed from the ET wall clock so it is
    DST-correct without any timezone library. */
function msUntilMidnightET() {
    const fmt = new Intl.DateTimeFormat('en-US', {
        timeZone: 'America/New_York', hour12: false,
        hour: '2-digit', minute: '2-digit', second: '2-digit',
    });
    const parts = Object.fromEntries(
        fmt.formatToParts(new Date()).map((p) => [p.type, p.value]));
    let hh = Number(parts.hour);
    if (hh === 24) hh = 0; // some engines emit '24' at midnight
    const elapsed = hh * 3600 + Number(parts.minute) * 60 + Number(parts.second);
    return (86400 - elapsed) * 1000;
}

function fmtCountdown(ms) {
    const s = Math.max(0, Math.floor(ms / 1000));
    const h = Math.floor(s / 3600);
    const m = Math.floor((s % 3600) / 60);
    const sec = s % 60;
    return `${h}h ${String(m).padStart(2, '0')}m ${String(sec).padStart(2, '0')}s`;
}

/** Repaint the midnight-ET expiry countdown from the last known auth state
    (driven by a 1s ticker, independent of the 10s status poll). */
function tickExpiryCountdown() {
    const el = document.getElementById('kaExpiry');
    if (!el) return;
    const auth = liveStatus && liveStatus.auth;
    const state = auth && auth.state;
    if (state === 'connected') {
        const ms = msUntilMidnightET();
        el.textContent = fmtCountdown(ms);
        // Under 30 minutes: flag it (the morning re-auth is near).
        el.className = 'ka-expiry-value num'
            + (ms <= 30 * 60 * 1000 ? ' expiring' : '');
    } else if (state === 'expired') {
        el.textContent = 'EXPIRED — reconnect';
        el.className = 'ka-expiry-value expired';
    } else {
        el.textContent = '—';
        el.className = 'ka-expiry-value num';
    }
}

/* -------------------- Morning reconnect prompt -------------------- */

/** Re-auth is due when the token has expired, or the keep-alive loop paused
    itself for the daily expiry. (A never-connected 'disconnected' state is
    NOT a re-auth prompt — the connection card already handles first connect.) */
function reauthNeeded() {
    const auth = liveStatus && liveStatus.auth;
    const expired = auth && auth.state === 'expired';
    const kaPaused = lastKeepalive
        && lastKeepalive.paused_reason === 'token_expired';
    return !!(expired || kaPaused);
}

function updateReconnectBanner() {
    const banner = document.getElementById('reconnectBanner');
    if (!banner) return;
    const needed = reauthNeeded();
    banner.classList.toggle('d-none', !needed);
    maybeNotifyReauth(needed);
}

/** Fire a browser notification once per re-auth event, only if the operator
    opted in (granted permission via "Enable alerts"). Best-effort. */
function maybeNotifyReauth(needed) {
    if (!needed) { reauthNotified = false; return; }
    if (reauthNotified) return;
    reauthNotified = true;
    if (typeof Notification !== 'undefined'
            && Notification.permission === 'granted') {
        try {
            new Notification('E*TRADE re-authentication needed', {
                body: 'The daily token has expired. Reconnect to resume '
                    + 'live access.',
            });
        } catch (_) { /* notifications are best-effort */ }
    }
}

function enableReauthAlerts() {
    if (typeof Notification === 'undefined') {
        showToast('warning', 'This browser does not support notifications.');
        return;
    }
    Notification.requestPermission().then((perm) => {
        if (perm === 'granted') {
            showToast('success', 'Re-auth alerts enabled for this browser.');
        } else {
            showToast('info', 'Notifications not granted — the on-page '
                + 'banner still appears.');
        }
    });
}

function reconnectNow() {
    const card = document.getElementById('connectionBody');
    if (card) card.scrollIntoView({ behavior: 'smooth', block: 'center' });
    const btn = document.getElementById('btnConnect');
    if (btn) btn.click(); // the connection card renders Connect/Re-authenticate
}

/* ==========================================================================
   Status poll — the ONLY live-status poll in the app
   ========================================================================== */

async function refreshStatus() {
    const wasConnected = isConnected();
    let data;
    try {
        data = await fetchJSON('/api/live/status', { silent: true });
    } catch (err) {
        liveStatus = null;
        renderEnvBadge(null, false);
        renderAuthBadge(null, false);
        renderNavDot(null, false);
        renderConnectionUnavailable(
            (err.body && (err.body.reason || err.body.error)) || err.message);
        renderKillSwitch(null);
        accountsLoaded = false;
        renderAccountBody('expired');
        syncOrderTicketGate();
        updateReconnectBanner();
        return;
    }
    liveStatus = data;
    renderEnvBadge(data.env, true);
    renderAuthBadge(data.auth && data.auth.state, true);
    renderNavDot(data.auth && data.auth.state, true);
    renderConnection(data.auth || {});
    renderKillSwitch(data.kill_switch);
    if (data.reconciliation) renderReconciliation(data.reconciliation);

    // Drive the account panel + order-ticket gate from the connection state.
    // Load accounts once on (re)connect; the poll never rebuilds operator
    // inputs, only flips the kill-switch-driven Place gate.
    const nowConnected = isConnected();
    if (nowConnected && (!wasConnected || !accountsLoaded)) {
        loadAccounts();
    } else if (!nowConnected && wasConnected) {
        accountsLoaded = false;
        renderAccountBody('expired');
    }
    syncOrderTicketGate();
    updateReconnectBanner();
}

/* ==========================================================================
   Bootstrap
   ========================================================================== */

// ==================== EXECUTION PARITY (Phase 3 Step 5) ====================

function fmtParityNum(value, digits) {
    if (value === null || value === undefined || Number.isNaN(Number(value))) {
        return '—';
    }
    return Number(value).toFixed(digits);
}

async function loadParity() {
    const summary = document.getElementById('paritySummary');
    const table = document.getElementById('parityTable');
    const body = document.getElementById('parityBody');
    const empty = document.getElementById('parityEmpty');
    if (!summary || !body || !table || !empty) return;

    let data;
    try {
        data = await fetchJSON('/api/live/parity', { silent: true });
    } catch (err) {
        table.classList.add('d-none');
        body.innerHTML = '';
        summary.textContent = '';
        empty.innerHTML = emptyStateHTML('bi-rulers', 'Parity unavailable',
            err.message || 'The audit backend has not been configured.');
        return;
    }
    renderParity(data);
}

function renderParity(data) {
    const summary = document.getElementById('paritySummary');
    const table = document.getElementById('parityTable');
    const body = document.getElementById('parityBody');
    const empty = document.getElementById('parityEmpty');

    const fills = Array.isArray(data.fills) ? data.fills : [];
    if (fills.length === 0) {
        table.classList.add('d-none');
        body.innerHTML = '';
        summary.textContent = '';
        empty.innerHTML = emptyStateHTML('bi-rulers', 'No live fills yet',
            'Execution parity appears once the live session has recorded ' +
            'filled orders in the audit log.');
        return;
    }

    empty.innerHTML = '';
    table.classList.remove('d-none');
    body.innerHTML = fills.map((f) => {
        const driftClass = Number(f.slippage_drift) > 0 ? 'pnl-neg' : 'pnl-pos';
        const recovered = f.arrival_mid_recovered
            ? ' <span class="text-muted" title="arrival mid recovered from'
              + ' shortfall (legacy row)">~</span>'
            : '';
        return '<tr>' +
            `<td class="num">${escapeHTML(String(f.seq ?? '—'))}</td>` +
            `<td>${escapeHTML(f.symbol ?? '—')}</td>` +
            `<td>${escapeHTML(f.action ?? '—')}</td>` +
            `<td>${escapeHTML(f.asset_type ?? '—')}</td>` +
            `<td class="num">${f.quantity ?? '—'}</td>` +
            `<td class="num">${fmtParityNum(f.arrival_mid, 2)}${recovered}</td>` +
            `<td class="num">${fmtParityNum(f.realized_shortfall, 4)}</td>` +
            `<td class="num">${fmtParityNum(f.expected_shortfall, 4)}</td>` +
            `<td class="num ${driftClass}">` +
            `${fmtParityNum(f.slippage_drift, 4)}</td>` +
            `<td class="num ${driftClass}">` +
            `${fmtParityNum(f.slippage_drift_total, 2)}</td>` +
            '</tr>';
    }).join('');

    const s = data.summary || {};
    const commNote = data.commission_available
        ? `actual commission $${fmtParityNum(s.total_actual_commission, 2)}`
        : 'actual commission N/A (broker does not surface it yet)';
    summary.textContent =
        `${data.n_fills} fill${data.n_fills === 1 ? '' : 's'} ` +
        `(${data.n_skipped} skipped) · total slippage drift ` +
        `$${fmtParityNum(s.total_slippage_drift, 2)} · ` +
        `modeled commission $${fmtParityNum(s.total_modeled_commission, 2)} · ` +
        `${commNote}. Drift > 0 = live worse than the model.`;
}

document.addEventListener('DOMContentLoaded', () => {
    // Initial paint + the page's poll. Status, working orders and the
    // scheduler refresh every LIVE_POLL_MS; the audit log loads once and
    // after actions (it is paged and append-only — no need to poll it).
    refreshStatus();
    refreshWorkingOrders();
    refreshScheduler();
    refreshKeepAlive();
    tickExpiryCountdown();
    loadAudit();
    loadParity();
    setInterval(refreshStatus, LIVE_POLL_MS);
    setInterval(refreshWorkingOrders, LIVE_POLL_MS);
    setInterval(refreshScheduler, LIVE_POLL_MS);
    setInterval(refreshKeepAlive, LIVE_POLL_MS);
    setInterval(tickExpiryCountdown, 1000); // smooth 1s expiry countdown

    const ksBtn = document.getElementById('btnKillSwitch');
    if (ksBtn) ksBtn.addEventListener('click', openKillSwitchModal);
    const ksConfirm = document.getElementById('ksModalConfirm');
    if (ksConfirm) ksConfirm.addEventListener('click', confirmKillSwitch);
    const ksReason = document.getElementById('ksReason');
    if (ksReason) {
        ksReason.addEventListener('keydown', (ev) => {
            if (ev.key === 'Enter') confirmKillSwitch();
        });
    }

    const btnReconcile = document.getElementById('btnReconcile');
    if (btnReconcile) btnReconcile.addEventListener('click', runReconcile);

    // Account selection: load that account's balances + positions.
    const accountSelect = document.getElementById('accountSelect');
    if (accountSelect) {
        accountSelect.addEventListener('change', () => {
            selectedAccountKey = accountSelect.value || null;
            updateTicketAccountLabel(knownAccounts);
            loadAccountDetail(selectedAccountKey);
            // Switching accounts voids any standing preview (it was built
            // for the old account_id_key).
            voidPreviewOnEdit();
        });
    }

    // Quotes.
    const btnQuotes = document.getElementById('btnQuotes');
    if (btnQuotes) btnQuotes.addEventListener('click', lookupQuotes);
    const quoteInput = document.getElementById('quoteSymbols');
    if (quoteInput) {
        quoteInput.addEventListener('keydown', (ev) => {
            if (ev.key === 'Enter') { ev.preventDefault(); lookupQuotes(); }
        });
    }

    // Order ticket.
    const ticketKindSel = document.getElementById('ticketKind');
    if (ticketKindSel) {
        ticketKindSel.addEventListener('change', () => {
            renderTicketPanes();
            voidPreviewOnEdit();
        });
    }
    const spPreset = document.getElementById('spPreset');
    if (spPreset) {
        spPreset.addEventListener('change', () => {
            renderTicketPanes();
            voidPreviewOnEdit();
        });
    }
    // EVERY ticket field voids the standing preview on edit.
    document.querySelectorAll('.ticket-input').forEach((el) => {
        const ev = el.tagName === 'SELECT' ? 'change' : 'input';
        el.addEventListener(ev, voidPreviewOnEdit);
    });
    const btnPreview = document.getElementById('btnPreview');
    if (btnPreview) btnPreview.addEventListener('click', previewOrder);
    const btnPlace = document.getElementById('btnPlace');
    if (btnPlace) btnPlace.addEventListener('click', openPlaceModal);
    const placeConfirm = document.getElementById('placeModalConfirm');
    if (placeConfirm) placeConfirm.addEventListener('click', confirmPlaceOrder);
    renderTicketPanes();
    syncOrderTicketGate();

    const btnSchedStart = document.getElementById('btnSchedStart');
    if (btnSchedStart) btnSchedStart.addEventListener('click', startScheduler);
    const btnSchedStop = document.getElementById('btnSchedStop');
    if (btnSchedStop) btnSchedStop.addEventListener('click', stopScheduler);
    const schedInput = document.getElementById('schedIntervalInput');
    if (schedInput) {
        // Any keystroke hands the field to the operator until the next
        // explicit start (mirrors the verifier-input protection).
        schedInput.addEventListener('input', () => {
            schedIntervalDirty = true;
            schedInput.classList.remove('is-invalid');
        });
    }

    // Token keep-alive card + morning reconnect prompt (Piece 2).
    const btnKaStart = document.getElementById('btnKaStart');
    if (btnKaStart) btnKaStart.addEventListener('click', startKeepAlive);
    const btnKaStop = document.getElementById('btnKaStop');
    if (btnKaStop) btnKaStop.addEventListener('click', stopKeepAlive);
    const kaInput = document.getElementById('kaIntervalInput');
    if (kaInput) {
        kaInput.addEventListener('input', () => {
            kaIntervalDirty = true;
            kaInput.classList.remove('is-invalid');
        });
    }
    const btnReconnectNow = document.getElementById('btnReconnectNow');
    if (btnReconnectNow) btnReconnectNow.addEventListener('click', reconnectNow);
    const btnEnableAlerts = document.getElementById('btnEnableAlerts');
    if (btnEnableAlerts) {
        btnEnableAlerts.addEventListener('click', enableReauthAlerts);
    }

    const auditFilter = document.getElementById('auditEventType');
    if (auditFilter) {
        auditFilter.addEventListener('change', () => {
            auditOffset = 0;
            loadAudit();
        });
    }
    const auditPrev = document.getElementById('auditPrev');
    if (auditPrev) auditPrev.addEventListener('click', () => {
        auditOffset = Math.max(0, auditOffset - AUDIT_PAGE_SIZE);
        loadAudit();
    });
    const auditNext = document.getElementById('auditNext');
    if (auditNext) auditNext.addEventListener('click', () => {
        auditOffset += AUDIT_PAGE_SIZE;
        loadAudit();
    });

    // Working-order cancel buttons (delegated — rows re-render on poll).
    const ordersBody = document.getElementById('workingOrdersBody');
    if (ordersBody) {
        ordersBody.addEventListener('click', (ev) => {
            const btn = ev.target.closest('.js-cancel-order');
            if (btn) cancelWorkingOrder(btn.dataset.orderId, btn);
        });
    }
});
