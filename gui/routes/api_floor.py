"""
API routes for the Trading Floor page.

GET /api/floor/desks proxies :func:`desks.registry.list_desks` — the Phase 5
shared contract (C1). Each entry carries exactly:
``key / name / firm_inspiration / description / status /
activates_in_phase / accent``.
"""
from __future__ import annotations

import logging

from flask import Blueprint, jsonify

logger = logging.getLogger(__name__)

# The desks package is built in the parallel Phase 5 backend task. The GUI
# must keep importing (and every other page keep working) if it is absent or
# broken, so the import is guarded — mirroring the LiveEtradeBroker pattern
# in api_trading — and failures surface as a loud 503 on the endpoint.
DESK_REGISTRY_IMPORT_ERROR: str | None
try:
    from desks.registry import list_desks
    DESK_REGISTRY_IMPORT_ERROR = None
except Exception as e:  # noqa: BLE001 - any import failure disables the floor API
    list_desks = None
    DESK_REGISTRY_IMPORT_ERROR = f'{type(e).__name__}: {e}'
    logger.error(
        'Failed to import desks.registry — Trading Floor desks API is '
        'unavailable: %s', DESK_REGISTRY_IMPORT_ERROR,
    )

floor_bp = Blueprint('floor', __name__, url_prefix='/api')


@floor_bp.route('/floor/desks', methods=['GET'])
def floor_desks():
    """All registered desks (contract C1) for the Trading Floor page."""
    if list_desks is None:
        return jsonify({
            'error': 'Desk framework unavailable',
            'reason': DESK_REGISTRY_IMPORT_ERROR,
        }), 503
    return jsonify({'desks': list_desks()})
