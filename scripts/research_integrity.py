#!/usr/bin/env python
"""Operator CLI for the immutable research-integrity ledger.

JSON-backed commands accept a specification object containing the keyword
arguments for the corresponding ``ResearchProtocol``, ``TrialRegistration``,
or ``TrialOutcome`` ``create`` method.  JSON is parsed strictly: duplicate
keys, non-finite numbers, non-object roots, and unknown constructor fields are
rejected.  Use ``-`` as a JSON path to read that document from standard input.

The ledger derives every trial count itself.  This CLI deliberately exposes no
option for entering or overriding a trial count. Every mutating command also
requires an explicit canonical timestamp. That timestamp is part of the
content hash, so retrying the same command remains idempotent instead of
silently creating new content with the retry wall clock.

Fixed holdouts open against realized artifact bytes. Prospective holdouts open
against the commitment hash emitted by ``freeze-protocol`` and accept their
realized snapshot only when ``decide-holdout`` permanently finalizes them.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any, Callable, Mapping, Sequence

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.research_integrity import (  # noqa: E402
    ResearchIntegrityLedger,
    ResearchProtocol,
    TrialOutcome,
    TrialRegistration,
    prospective_data_commitment_hash,
)


MAX_JSON_BYTES = 8 * 1024 * 1024


class CliUsageError(ValueError):
    """Raised instead of printing argparse's non-JSON error surface."""


class JsonArgumentParser(argparse.ArgumentParser):
    """Argument parser whose failures are handled by :func:`main`."""

    def error(self, message: str) -> None:
        raise CliUsageError(message)


def _emit(payload: Mapping[str, Any], *, stream=None) -> None:
    destination = stream if stream is not None else sys.stdout
    print(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False),
        file=destination,
    )


def _strict_json_object(path_value: str, *, document_name: str) -> dict[str, Any]:
    if path_value == "-":
        raw = sys.stdin.read(MAX_JSON_BYTES + 1)
    else:
        path = Path(path_value)
        if path.stat().st_size > MAX_JSON_BYTES:
            raise ValueError(f"{document_name} exceeds the {MAX_JSON_BYTES}-byte limit")
        raw = path.read_text(encoding="utf-8")
    if len(raw.encode("utf-8")) > MAX_JSON_BYTES:
        raise ValueError(f"{document_name} exceeds the {MAX_JSON_BYTES}-byte limit")

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{document_name} contains duplicate key {key!r}")
            result[key] = value
        return result

    def reject_constant(token: str) -> None:
        raise ValueError(f"{document_name} contains invalid JSON number {token}")

    try:
        document = json.loads(
            raw,
            object_pairs_hook=unique_object,
            parse_constant=reject_constant,
        )
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"{document_name} is invalid JSON at line {exc.lineno}, column {exc.colno}"
        ) from exc
    if not isinstance(document, dict):
        raise TypeError(f"{document_name} must be a JSON object")
    return document


def _ledger(args: argparse.Namespace) -> ResearchIntegrityLedger:
    return ResearchIntegrityLedger(args.ledger_root, program_id=args.program_id)


def _require_explicit_timestamp(
    specification: Mapping[str, Any], field_name: str, *, document_name: str
) -> None:
    if field_name not in specification or specification[field_name] is None:
        raise ValueError(
            f"{document_name} must include explicit {field_name}; "
            "content timestamps may not default to the retry wall clock"
        )


def _with_head(
    ledger: ResearchIntegrityLedger, operation: str, payload: Mapping[str, Any]
) -> dict[str, Any]:
    summary = ledger.verify()
    return {
        "ok": True,
        "operation": operation,
        **payload,
        "head_hash": summary["head_hash"],
    }


def command_freeze_protocol(args: argparse.Namespace) -> dict[str, Any]:
    specification = _strict_json_object(args.json, document_name="protocol specification")
    _require_explicit_timestamp(
        specification, "frozen_at", document_name="protocol specification"
    )
    protocol = ResearchProtocol.create(**specification)
    ledger = _ledger(args)
    frozen = ledger.freeze_protocol(protocol)
    prospective_commitments = {
        holdout_id: prospective_data_commitment_hash(holdout)
        for holdout_id, holdout in frozen.payload["holdouts"].items()
        if holdout.get("mode") == "prospective"
    }
    return _with_head(
        ledger,
        "freeze_protocol",
        {
            "protocol_hash": frozen.protocol_hash,
            "protocol_id": frozen.payload["protocol_id"],
            "version": frozen.payload["version"],
            "prospective_data_commitments": prospective_commitments,
        },
    )


def command_register_trial(args: argparse.Namespace) -> dict[str, Any]:
    specification = _strict_json_object(args.json, document_name="trial specification")
    _require_explicit_timestamp(
        specification, "registered_at", document_name="trial specification"
    )
    trial = TrialRegistration.create(**specification)
    ledger = _ledger(args)
    registered = ledger.register_trial(trial)
    return _with_head(
        ledger,
        "register_trial",
        {
            "trial_hash": registered.trial_hash,
            "trial_id": registered.payload["trial_id"],
            "program_trial_count": ledger.trial_count,
            "protocol_trial_count": ledger.protocol_trial_count(
                registered.payload["protocol_hash"]
            ),
        },
    )


def command_record_trial_outcome(args: argparse.Namespace) -> dict[str, Any]:
    specification = _strict_json_object(
        args.json, document_name="trial outcome specification"
    )
    _require_explicit_timestamp(
        specification, "recorded_at", document_name="trial outcome specification"
    )
    outcome = TrialOutcome.create(**specification)
    ledger = _ledger(args)
    recorded = ledger.record_trial_outcome(outcome)
    return _with_head(
        ledger,
        "record_trial_outcome",
        {
            "outcome_hash": recorded.record_hash,
            "trial_hash": recorded.payload["trial_hash"],
            "status": recorded.payload["status"],
        },
    )


def command_open_holdout(args: argparse.Namespace) -> dict[str, Any]:
    ledger = _ledger(args)
    opening = ledger.open_holdout(
        protocol_hash=args.protocol_hash,
        holdout_id=args.holdout_id,
        trial_hash=args.trial_hash,
        data_artifact_hash=args.data_artifact_hash,
        data_commitment_hash=args.data_commitment_hash,
        actor=args.actor,
        opened_at=args.opened_at,
    )
    return _with_head(
        ledger,
        "open_holdout",
        {
            "opening_hash": opening.opening_hash,
            "protocol_hash": opening.payload["protocol_hash"],
            "holdout_id": opening.payload["holdout_id"],
            "program_trial_count_at_open": opening.payload[
                "program_trial_count_at_open"
            ],
            "protocol_trial_count_at_open": opening.payload[
                "protocol_trial_count_at_open"
            ],
            "data_binding_mode": opening.payload.get(
                "data_binding_mode", "fixed_snapshot"),
            "data_artifact_hash": opening.payload.get("data_artifact_hash"),
            "data_commitment_hash": opening.payload.get("data_commitment_hash"),
        },
    )


def command_decide_holdout(args: argparse.Namespace) -> dict[str, Any]:
    result_summary = _strict_json_object(
        args.result_summary_json, document_name="holdout result summary"
    )
    ledger = _ledger(args)
    decision = ledger.record_holdout_decision(
        opening_hash=args.opening_hash,
        decision=args.decision,
        result_summary=result_summary,
        result_artifact_hash=args.result_artifact_hash,
        actor=args.actor,
        decided_at=args.decided_at,
        realized_data_artifact_hash=args.realized_data_artifact_hash,
    )
    return _with_head(
        ledger,
        "decide_holdout",
        {
            "decision_hash": decision.decision_hash,
            "opening_hash": decision.payload["opening_hash"],
            "decision": decision.payload["decision"],
            "realized_data_artifact_hash": decision.payload.get(
                "realized_data_artifact_hash"),
        },
    )


def command_verify(args: argparse.Namespace) -> dict[str, Any]:
    summary = _ledger(args).verify(expected_head_hash=args.expected_head_hash)
    return {"ok": True, "operation": "verify", **summary}


def _add_ledger_identity(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--ledger-root",
        required=True,
        help="directory containing one append-only research ledger",
    )
    parser.add_argument(
        "--program-id",
        required=True,
        help="stable program identity permanently bound to the ledger root",
    )


def parser() -> argparse.ArgumentParser:
    root = JsonArgumentParser(description=__doc__)
    _add_ledger_identity(root)
    commands = root.add_subparsers(dest="command", required=True)

    freeze = commands.add_parser(
        "freeze-protocol", help="freeze a preregistration specification"
    )
    freeze.add_argument("--json", required=True, help="strict JSON specification path or -")
    freeze.set_defaults(handler=command_freeze_protocol)

    register = commands.add_parser(
        "register-trial", help="append one attempted trial before running it"
    )
    register.add_argument("--json", required=True, help="strict JSON specification path or -")
    register.set_defaults(handler=command_register_trial)

    outcome = commands.add_parser(
        "record-trial-outcome", help="append the trial's sole terminal outcome"
    )
    outcome.add_argument("--json", required=True, help="strict JSON specification path or -")
    outcome.set_defaults(handler=command_record_trial_outcome)

    opening = commands.add_parser(
        "open-holdout", help="permanently consume one preregistered holdout"
    )
    opening.add_argument("--protocol-hash", required=True)
    opening.add_argument("--holdout-id", required=True)
    opening.add_argument("--trial-hash", required=True)
    data_binding = opening.add_mutually_exclusive_group()
    data_binding.add_argument(
        "--data-artifact-hash",
        help="realized fixed-snapshot digest (never valid for prospective opening)",
    )
    data_binding.add_argument(
        "--data-commitment-hash",
        help="frozen prospective acquisition-commitment digest",
    )
    opening.add_argument("--actor", required=True)
    opening.add_argument(
        "--opened-at", required=True, help="explicit canonical UTC timestamp ending in Z"
    )
    opening.set_defaults(handler=command_open_holdout)

    decide = commands.add_parser(
        "decide-holdout", help="append the holdout's sole permanent decision"
    )
    decide.add_argument("--opening-hash", required=True)
    decide.add_argument("--decision", required=True, choices=("pass", "fail", "invalid"))
    decide.add_argument("--result-summary-json", required=True)
    decide.add_argument("--result-artifact-hash", required=True)
    decide.add_argument(
        "--realized-data-artifact-hash",
        default=None,
        help="required final snapshot digest for a prospective holdout",
    )
    decide.add_argument("--actor", required=True)
    decide.add_argument(
        "--decided-at", required=True, help="explicit canonical UTC timestamp ending in Z"
    )
    decide.set_defaults(handler=command_decide_holdout)

    verify = commands.add_parser(
        "verify", help="verify the hash chain, references, counts, and optional anchor"
    )
    verify.add_argument("--expected-head-hash", default=None)
    verify.set_defaults(handler=command_verify)
    return root


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = parser().parse_args(argv)
        handler: Callable[[argparse.Namespace], dict[str, Any]] = args.handler
        _emit(handler(args))
        return 0
    except KeyboardInterrupt:
        _emit(
            {"ok": False, "error_type": "KeyboardInterrupt", "error": "interrupted"},
            stream=sys.stderr,
        )
        return 130
    except Exception as exc:  # one concise, traceback-free operator failure surface
        _emit(
            {"ok": False, "error_type": type(exc).__name__, "error": str(exc)},
            stream=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
