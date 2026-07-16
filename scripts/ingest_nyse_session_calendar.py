"""Acquire and archive official NYSE session-close source pages.

The command downloads only the official URLs frozen in the session-close
calendar.  Raw HTML is published create-only under its SHA-256 identity.  A
new source receipt is built and validated from a same-directory staging file
before the active ``receipt.json`` is atomically replaced.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import hashlib
import os
from pathlib import Path
import tempfile
from typing import Any, Callable, Mapping
from urllib.parse import urlsplit

import data.session_close_calendar as calendar_evidence


MAX_HTML_BYTES = 8 * 1024 * 1024
DEFAULT_TIMEOUT_SECONDS = 60.0
OFFICIAL_HOSTS = frozenset({"ir.theice.com", "www.nyse.com"})
REQUEST_HEADERS = {
    "Accept": "text/html,application/xhtml+xml",
    "User-Agent": "StockOptionsTrader-research-evidence/1.0",
}


class NyseCalendarAcquisitionError(RuntimeError):
    """Official calendar-source acquisition or publication failed."""


def _canonical_utc(value: str | None) -> str:
    if value is None:
        return (
            datetime.now(timezone.utc)
            .isoformat(timespec="seconds")
            .replace("+00:00", "Z")
        )
    if not isinstance(value, str) or not value.endswith("Z"):
        raise NyseCalendarAcquisitionError("created_at_utc must be canonical UTC")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise NyseCalendarAcquisitionError(
            "created_at_utc must be canonical UTC"
        ) from exc
    canonical = parsed.astimezone(timezone.utc).isoformat(
        timespec="microseconds" if parsed.microsecond else "seconds"
    ).replace("+00:00", "Z")
    if canonical != value:
        raise NyseCalendarAcquisitionError("created_at_utc must be canonical UTC")
    return value


def _official_url(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise NyseCalendarAcquisitionError(f"{label} must be a non-empty URL")
    parsed = urlsplit(value)
    if (
        parsed.scheme != "https"
        or parsed.hostname not in OFFICIAL_HOSTS
        or parsed.username is not None
        or parsed.password is not None
        or parsed.port not in {None, 443}
        or parsed.fragment
    ):
        raise NyseCalendarAcquisitionError(
            f"{label} must be an official ICE/NYSE HTTPS URL"
        )
    return value


def _header(headers: Any, name: str) -> str | None:
    if not isinstance(headers, Mapping):
        return None
    value = headers.get(name)
    if value is None:
        value = headers.get(name.lower())
    if value is None:
        return None
    if not isinstance(value, str) or value != value.strip():
        raise NyseCalendarAcquisitionError(
            f"HTTP {name} header must be a trimmed string"
        )
    return value


def _http_date_utc(headers: Any, name: str) -> str | None:
    value = _header(headers, name)
    if value is None:
        return None
    try:
        parsed = parsedate_to_datetime(value)
    except (TypeError, ValueError) as exc:
        raise NyseCalendarAcquisitionError(
            f"HTTP {name} header is not an RFC date"
        ) from exc
    if parsed.tzinfo is None:
        raise NyseCalendarAcquisitionError(
            f"HTTP {name} header does not identify a timezone"
        )
    return parsed.astimezone(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


def _download_source(
    source: Mapping[str, Any],
    *,
    http_get: Callable[..., Any],
    timeout_seconds: float,
    fetched_at_utc: str | None,
) -> tuple[bytes, dict[str, Any]]:
    source_id = source.get("source_id")
    if not isinstance(source_id, str) or not source_id.strip():
        raise NyseCalendarAcquisitionError("calendar source_id is missing")
    requested_url = _official_url(source.get("url"), f"{source_id} requested URL")
    response = http_get(
        requested_url,
        headers=dict(REQUEST_HEADERS),
        timeout=timeout_seconds,
        allow_redirects=True,
    )
    raise_for_status = getattr(response, "raise_for_status", None)
    if callable(raise_for_status):
        raise_for_status()
    status_code = getattr(response, "status_code", None)
    if type(status_code) is not int or status_code != 200:
        raise NyseCalendarAcquisitionError(
            f"{source_id} returned non-qualifying HTTP status {status_code!r}"
        )
    final_url = _official_url(
        getattr(response, "url", None), f"{source_id} final URL"
    )
    requested_parts = urlsplit(requested_url)
    final_parts = urlsplit(final_url)
    if (
        final_parts.hostname != requested_parts.hostname
        or final_parts.path.rstrip("/") != requested_parts.path.rstrip("/")
    ):
        raise NyseCalendarAcquisitionError(
            f"{source_id} final URL is outside its frozen host/path"
        )
    history = getattr(response, "history", ())
    if not isinstance(history, (list, tuple)):
        raise NyseCalendarAcquisitionError(f"{source_id} redirect history is malformed")
    for index, redirect in enumerate(history):
        _official_url(
            getattr(redirect, "url", None),
            f"{source_id} redirect {index} URL",
        )
    headers = getattr(response, "headers", None)
    content_type = _header(headers, "Content-Type")
    if content_type is None or content_type.split(";", 1)[0].strip().lower() not in {
        "text/html",
        "application/xhtml+xml",
    }:
        raise NyseCalendarAcquisitionError(
            f"{source_id} did not return an HTML content type"
        )
    raw = getattr(response, "content", None)
    if not isinstance(raw, bytes) or not raw:
        raise NyseCalendarAcquisitionError(f"{source_id} returned no HTML bytes")
    if len(raw) > MAX_HTML_BYTES:
        raise NyseCalendarAcquisitionError(
            f"{source_id} HTML exceeds {MAX_HTML_BYTES} bytes"
        )
    # Parse before any local publication.  The receipt builder performs the
    # authoritative calendar-to-source comparison.
    calendar_evidence.normalized_html_text(raw)
    calendar_evidence.extract_early_close_dates(raw)
    return raw, {
        "retrieved_at_utc": _canonical_utc(fetched_at_utc),
        "http": {
            "status_code": status_code,
            "date_utc": _http_date_utc(headers, "Date"),
            "content_type": content_type,
            "etag": _header(headers, "ETag"),
            "last_modified_utc": _http_date_utc(headers, "Last-Modified"),
        },
    }


def _create_only(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as stream:
            temporary = stream.name
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != raw:
                raise NyseCalendarAcquisitionError(
                    "content-addressed NYSE evidence collision"
                )
    finally:
        if temporary is not None:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass


def _publish_receipt(
    document: Mapping[str, Any],
    *,
    calendar_path: Path,
    receipt_path: Path,
    prior_receipt: bytes | None,
) -> dict[str, Any]:
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (calendar_evidence.canonical_json(document) + "\n").encode("utf-8")
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=receipt_path.parent,
            prefix=f".{receipt_path.name}.",
            suffix=".json",
            delete=False,
        ) as stream:
            temporary = stream.name
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        staged_path = Path(temporary)
        staged_evidence = calendar_evidence.load_session_close_calendar_evidence(
            calendar_path=calendar_path,
            receipt_path=staged_path,
        )
        current = receipt_path.read_bytes() if receipt_path.exists() else None
        if current != prior_receipt:
            raise NyseCalendarAcquisitionError(
                "active NYSE source receipt changed during acquisition"
            )
        os.replace(temporary, receipt_path)
        temporary = None
        active_evidence = calendar_evidence.load_session_close_calendar_evidence(
            calendar_path=calendar_path,
            receipt_path=receipt_path,
        )
        if active_evidence != staged_evidence:
            raise NyseCalendarAcquisitionError(
                "promoted NYSE source receipt did not revalidate identically"
            )
        return active_evidence
    finally:
        if temporary is not None:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass


def acquire_nyse_session_calendar_sources(
    *,
    calendar_path: str | os.PathLike[str] | None = None,
    receipt_path: str | os.PathLike[str] | None = None,
    http_get: Callable[..., Any] | None = None,
    created_at_utc: str | None = None,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """Acquire every frozen official source and publish validated evidence."""
    if isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, (int, float)):
        raise NyseCalendarAcquisitionError("timeout_seconds must be positive")
    timeout = float(timeout_seconds)
    if not 0 < timeout <= 600:
        raise NyseCalendarAcquisitionError("timeout_seconds must be in (0, 600]")
    calendar_file = Path(
        calendar_path or calendar_evidence.DEFAULT_SESSION_CLOSE_CALENDAR
    ).resolve()
    receipt_file = Path(
        receipt_path or calendar_evidence.DEFAULT_SOURCE_RECEIPT
    ).resolve()
    fixed_timestamp = (
        _canonical_utc(created_at_utc) if created_at_utc is not None else None
    )
    calendar = calendar_evidence.load_session_close_calendar(calendar_file)
    payload = calendar.get("payload")
    sources = payload.get("sources") if isinstance(payload, Mapping) else None
    if not isinstance(sources, list) or not sources:
        raise NyseCalendarAcquisitionError("calendar has no official sources")

    if http_get is None:
        import requests

        http_get = requests.get
    documents: dict[str, bytes] = {}
    metadata: dict[str, dict[str, Any]] = {}
    for source in sources:
        if not isinstance(source, Mapping):
            raise NyseCalendarAcquisitionError("calendar source is malformed")
        source_id = source.get("source_id")
        if not isinstance(source_id, str) or source_id in documents:
            raise NyseCalendarAcquisitionError(
                "calendar source IDs must be non-empty and unique"
            )
        raw, response_metadata = _download_source(
            source,
            http_get=http_get,
            timeout_seconds=timeout,
            fetched_at_utc=fixed_timestamp,
        )
        documents[source_id] = raw
        metadata[source_id] = response_metadata

    created = fixed_timestamp or _canonical_utc(None)
    try:
        document = calendar_evidence.build_session_close_source_receipt(
            calendar_file,
            documents,
            metadata,
            created_at_utc=created,
        )
    except calendar_evidence.SessionCloseCalendarError as exc:
        raise NyseCalendarAcquisitionError(
            "official calendar source evidence is not candidate-grade"
        ) from exc
    prior_receipt = receipt_file.read_bytes() if receipt_file.exists() else None
    raw_dir = receipt_file.parent / "raw"
    archives: dict[str, str] = {}
    for source_id, raw in documents.items():
        digest = hashlib.sha256(raw).hexdigest()
        archive = raw_dir / f"{digest}.html"
        _create_only(archive, raw)
        archives[source_id] = archive.relative_to(receipt_file.parent).as_posix()
    receipt_archive = (
        receipt_file.parent
        / "receipts"
        / f"{document['artifact_hash']}.json"
    )
    _create_only(
        receipt_archive,
        (calendar_evidence.canonical_json(document) + "\n").encode("utf-8"),
    )
    evidence = _publish_receipt(
        document,
        calendar_path=calendar_file,
        receipt_path=receipt_file,
        prior_receipt=prior_receipt,
    )
    return {
        "receipt": document,
        "evidence": evidence,
        "raw_archives": dict(sorted(archives.items())),
        "receipt_archive": receipt_archive.relative_to(
            receipt_file.parent
        ).as_posix(),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--calendar",
        default=str(calendar_evidence.DEFAULT_SESSION_CLOSE_CALENDAR),
    )
    parser.add_argument(
        "--receipt",
        default=str(calendar_evidence.DEFAULT_SOURCE_RECEIPT),
    )
    parser.add_argument("--timeout-seconds", type=float, default=DEFAULT_TIMEOUT_SECONDS)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = acquire_nyse_session_calendar_sources(
        calendar_path=args.calendar,
        receipt_path=args.receipt,
        timeout_seconds=args.timeout_seconds,
    )
    receipt = result["receipt"]
    print(
        calendar_evidence.canonical_json(
            {
                "calendar_artifact_hash": receipt["payload"][
                    "calendar_artifact_hash"
                ],
                "raw_archives": result["raw_archives"],
                "receipt_archive": result["receipt_archive"],
                "source_count": len(receipt["payload"]["sources"]),
                "source_receipt_artifact_hash": receipt["artifact_hash"],
            }
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - operator entry point
    raise SystemExit(main())
