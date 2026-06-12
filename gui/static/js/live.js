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
    liveStatus = { ...(liveStatus || {}), auth, env: auth.env };
    renderEnvBadge(auth.env, true);
    renderAuthBadge(auth.state, true);
    renderNavDot(auth.state, true);
    renderConnection(auth);
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

async function cancelWorkingOrder(orderId, btn) {
    if (!window.confirm(`Cancel working order ${orderId}?`)) return;
    const restore = btnLoading(btn);
    try {
        await fetchJSON(
            `/api/live/orders/${encodeURIComponent(orderId)}/cancel`,
            { method: 'POST' });
        showToast('info', `Order ${orderId} cancelled.`);
        refreshWorkingOrders();
        loadAudit();
    } catch (_) {
        restore();
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
    } catch (_) {
        // toasted by fetchJSON (covers 409 no-live-session and 503s)
    } finally {
        restore();
    }
}

/* ==========================================================================
   Status poll — the ONLY live-status poll in the app
   ========================================================================== */

async function refreshStatus() {
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
        return;
    }
    liveStatus = data;
    renderEnvBadge(data.env, true);
    renderAuthBadge(data.auth && data.auth.state, true);
    renderNavDot(data.auth && data.auth.state, true);
    renderConnection(data.auth || {});
    renderKillSwitch(data.kill_switch);
    if (data.reconciliation) renderReconciliation(data.reconciliation);
}

/* ==========================================================================
   Bootstrap
   ========================================================================== */

document.addEventListener('DOMContentLoaded', () => {
    // Initial paint + the page's poll. Status and working orders refresh
    // every LIVE_POLL_MS; the audit log loads once and after actions (it
    // is paged and append-only — no need to poll it).
    refreshStatus();
    refreshWorkingOrders();
    loadAudit();
    setInterval(refreshStatus, LIVE_POLL_MS);
    setInterval(refreshWorkingOrders, LIVE_POLL_MS);

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
