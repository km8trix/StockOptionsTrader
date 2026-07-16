#!/usr/bin/env python
"""Acquire immutable official Sharadar SEP field-semantics evidence.

This operator command reads ``NASDAQ_DATA_LINK_API_KEY`` directly from the
process environment.  It does not source ``.env`` or any shell file.  The key
is added only to the HTTPS transport request and is never printed, placed in a
canonical request, written to an artifact, or included in an error message.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import math
import os
import re
import sys
from typing import Any, Callable

from data.sharadar_semantics_evidence import (
    INDICATORS_URL,
    MAX_RAW_RESPONSE_BYTES,
    SharadarSemanticsEvidenceError,
    publish_sharadar_semantics_receipt,
)


API_KEY_ENV = "NASDAQ_DATA_LINK_API_KEY"
DEFAULT_CANDIDATE_ID = "pead-vq-source-qualification-v3"
DEFAULT_WAREHOUSE_DIR = "pit_warehouse"
_CREDENTIAL = re.compile(r"^[A-Za-z0-9._~-]{8,512}$")


class SharadarSemanticsAcquisitionError(ValueError):
    """The credential, transport, or provider response was not safe to use."""


def _credential(environ: Mapping[str, str]) -> str:
    value = environ.get(API_KEY_ENV)
    if not isinstance(value, str) or _CREDENTIAL.fullmatch(value) is None:
        raise SharadarSemanticsAcquisitionError(
            f"{API_KEY_ENV} is missing or is not canonical"
        )
    return value


def _captured_at(clock: Callable[[], datetime]) -> str:
    value = clock()
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise SharadarSemanticsAcquisitionError("clock did not return an aware timestamp")
    return (
        value.astimezone(timezone.utc)
        .isoformat(timespec="microseconds" if value.microsecond else "seconds")
        .replace("+00:00", "Z")
    )


def fetch_sharadar_semantics(
    *,
    credential: str,
    get: Callable[..., Any],
    timeout: float,
) -> bytes:
    """Return the exact decoded HTTP entity bytes without exposing transport state."""
    if not isinstance(credential, str) or _CREDENTIAL.fullmatch(credential) is None:
        raise SharadarSemanticsAcquisitionError("credential is not canonical")
    if (
        isinstance(timeout, bool)
        or not isinstance(timeout, (int, float))
        or not math.isfinite(timeout)
        or not 0 < timeout <= 600
    ):
        raise SharadarSemanticsAcquisitionError(
            "timeout must be finite and in the interval (0, 600]"
        )
    params = {
        "qopts.per_page": "10000",
        "table": "SEP",
        "api_key": credential,
    }
    response: Any = None
    try:
        try:
            response = get(
                INDICATORS_URL,
                params=params,
                timeout=float(timeout),
                stream=True,
                allow_redirects=False,
            )
        except Exception:
            # Request exceptions commonly render the complete URL, including
            # query parameters.  Never chain or interpolate that exception.
            raise SharadarSemanticsAcquisitionError(
                "SHARADAR/INDICATORS network request failed"
            ) from None
        if type(getattr(response, "status_code", None)) is not int:
            raise SharadarSemanticsAcquisitionError(
                "SHARADAR/INDICATORS returned an invalid HTTP status"
            )
        if response.status_code != 200:
            raise SharadarSemanticsAcquisitionError(
                "SHARADAR/INDICATORS did not return HTTP 200"
            )
        iterator = getattr(response, "iter_content", None)
        if not callable(iterator):
            raise SharadarSemanticsAcquisitionError(
                "SHARADAR/INDICATORS response is not streamable"
            )
        chunks: list[bytes] = []
        size = 0
        try:
            stream = iterator(chunk_size=64 * 1024)
            for chunk in stream:
                if not chunk:
                    continue
                if not isinstance(chunk, bytes):
                    raise SharadarSemanticsAcquisitionError(
                        "SHARADAR/INDICATORS returned non-byte content"
                    )
                size += len(chunk)
                if size > MAX_RAW_RESPONSE_BYTES:
                    raise SharadarSemanticsAcquisitionError(
                        "SHARADAR/INDICATORS response exceeds the evidence size limit"
                    )
                chunks.append(chunk)
        except SharadarSemanticsAcquisitionError:
            raise
        except Exception:
            raise SharadarSemanticsAcquisitionError(
                "SHARADAR/INDICATORS response stream failed"
            ) from None
        raw = b"".join(chunks)
        if not raw:
            raise SharadarSemanticsAcquisitionError(
                "SHARADAR/INDICATORS returned an empty response"
            )
        if credential.encode("ascii") in raw:
            raise SharadarSemanticsAcquisitionError(
                "SHARADAR/INDICATORS response reflected request credentials"
            )
        return raw
    finally:
        close = getattr(response, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                pass


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m scripts.acquire_sharadar_semantics",
        description=__doc__.splitlines()[0],
    )
    parser.add_argument("--candidate-id", default=DEFAULT_CANDIDATE_ID)
    parser.add_argument("--warehouse-dir", default=DEFAULT_WAREHOUSE_DIR)
    parser.add_argument("--timeout", type=float, default=60.0)
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    get: Callable[..., Any] | None = None,
    clock: Callable[[], datetime] | None = None,
    environ: Mapping[str, str] | None = None,
) -> int:
    args = build_parser().parse_args(argv)
    environment = os.environ if environ is None else environ
    now = (lambda: datetime.now(timezone.utc)) if clock is None else clock
    try:
        credential = _credential(environment)
        if credential in args.candidate_id or credential in args.warehouse_dir:
            raise SharadarSemanticsAcquisitionError(
                "request credentials cannot be reused in artifact arguments"
            )
        if get is None:  # pragma: no cover - operator network path
            import requests

            get = requests.get
        raw = fetch_sharadar_semantics(
            credential=credential,
            get=get,
            timeout=args.timeout,
        )
        # Capture time means the exact response bytes are completely received,
        # not merely that the request was about to begin.
        captured_at_utc = _captured_at(now)
        receipt, receipt_path = publish_sharadar_semantics_receipt(
            args.warehouse_dir,
            raw,
            candidate_id=args.candidate_id,
            captured_at_utc=captured_at_utc,
        )
    except SharadarSemanticsAcquisitionError as exc:
        # These messages are closed strings and never contain provider details.
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    except SharadarSemanticsEvidenceError:
        print(
            "ERROR: provider bytes or local evidence failed the closed semantics contract",
            file=sys.stderr,
        )
        return 1
    except (OSError, TypeError, ValueError):
        print("ERROR: semantics evidence could not be published safely", file=sys.stderr)
        return 1

    print(f"artifact_hash={receipt['artifact_hash']}")
    print(f"receipt_path={receipt_path}")
    print(f"raw_sha256={receipt['payload']['raw_artifact']['sha256']}")
    print(f"captured_at_utc={receipt['payload']['captured_at_utc']}")
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through main()
    raise SystemExit(main())
