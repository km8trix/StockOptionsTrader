// Trading Floor page: renders desk cards from GET /api/floor/desks
// (Phase 5 desk registry, contract C1). Ready desks deep-link into
// /backtest?desk=<key>; planned desks show the phase they activate in.
// Depends on utils.js (fetchJSON, escapeHTML, emptyStateHTML).

'use strict';

// Only a strict #rrggbb accent flows into the inline style; anything else
// falls back to the theme accent via the --desk-accent custom property.
const ACCENT_RE = /^#[0-9a-fA-F]{6}$/;

const DESK_ICONS = {
    foundation: 'bi-building',
    renaissance: 'bi-diagram-3',
    citadel: 'bi-grid-1x2',
    janestreet: 'bi-sliders',
};

function deskIcon(key) {
    return DESK_ICONS[key] || 'bi-briefcase';
}

function deskAccentStyle(accent) {
    return ACCENT_RE.test(accent || '') ? ` style="--desk-accent: ${accent};"` : '';
}

function deskStatusBadge(desk) {
    if (desk.status === 'ready') {
        return '<span class="badge-ready">Ready</span>';
    }
    const phase = Number(desk.activates_in_phase);
    return `<span class="badge-phase">${
        Number.isFinite(phase) && phase > 0
            ? `Activates in Phase ${phase}` : 'Planned'}</span>`;
}

function deskCard(desk) {
    const ready = desk.status === 'ready';
    return (
        '<div class="col-md-6 col-xl-4">' +
        `<div class="card desk-card h-100"${deskAccentStyle(desk.accent)}>` +
        '<div class="card-header d-flex justify-content-between align-items-center">' +
        `<span><i class="bi ${deskIcon(desk.key)}" aria-hidden="true"></i> ` +
        `${escapeHTML(desk.name ?? desk.key ?? '')}</span>` +
        deskStatusBadge(desk) +
        '</div>' +
        '<div class="card-body d-flex flex-column">' +
        `<div class="desk-firm-tag">${escapeHTML(desk.firm_inspiration ?? '')}</div>` +
        `<p class="desk-desc flex-grow-1">${escapeHTML(desk.description ?? '')}</p>` +
        (ready
            ? '<a class="btn btn-outline-secondary btn-sm align-self-start" ' +
              `href="/backtest?desk=${encodeURIComponent(desk.key)}">` +
              '<i class="bi bi-play-fill" aria-hidden="true"></i> Run a desk backtest</a>'
            : '') +
        '</div></div></div>'
    );
}

function paintGridMessage(grid, icon, msg, hint) {
    grid.innerHTML =
        '<div class="col-12"><div class="card"><div class="card-body p-0">' +
        emptyStateHTML(icon, msg, hint) +
        '</div></div></div>';
}

async function loadDesks() {
    const grid = document.getElementById('deskGrid');
    let desks;
    try {
        const data = await fetchJSON('/api/floor/desks', { silent: true });
        desks = data.desks || [];
    } catch (err) {
        paintGridMessage(grid, 'bi-exclamation-triangle', 'Could not load desks',
            err.message || 'Check the server log and refresh.');
        return;
    }
    if (desks.length === 0) {
        paintGridMessage(grid, 'bi-building', 'No desks registered',
            'Desks appear here as their phases activate.');
        return;
    }
    grid.innerHTML = desks.map(deskCard).join('');
}

document.addEventListener('DOMContentLoaded', loadDesks);
