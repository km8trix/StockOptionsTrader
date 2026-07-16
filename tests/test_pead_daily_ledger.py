from __future__ import annotations

from decimal import Decimal, ROUND_HALF_EVEN, localcontext

from analysis.pead_daily_ledger import _cohort_daily_rows


def _row(
    *,
    ticker: str,
    m_ticker: str,
    rank: int,
    leg: str,
    target: str,
    quantity: str,
    accruals: list[dict] | None = None,
) -> dict:
    return {
        "cohort_id": "pooled:2020-01-01:2",
        "formation_date": "2020-01-01",
        "entry_date": "2020-01-02",
        "exit_date": "2020-01-06",
        "horizon_sessions": 2,
        "ticker": ticker,
        "m_ticker": m_ticker,
        "permaticker": 1000 + rank,
        "source_event_key": {
            "m_ticker": m_ticker,
            "per_end_date": "2019-12-31",
            "per_type": "Q",
        },
        "rank": rank,
        "leg": leg,
        "signal": f"{rank / 100:.18f}",
        "signed_target_notional": target,
        "signed_split_normalized_share_equivalent_quantity": quantity,
        "entry_fee": "0.003000000000000000",
        "exit_fee": "0.003000000000000000",
        "distribution_accruals": accruals or [],
    }


def test_daily_accounting_keeps_distribution_receivable_out_of_settled_cash():
    rows = [
        _row(
            ticker="LONG",
            m_ticker="LONG",
            rank=1,
            leg="long",
            target="1.000000000000000000",
            quantity="0.100000000000000000000000",
        ),
        _row(
            ticker="SHORT",
            m_ticker="SHORT",
            rank=2,
            leg="short",
            target="-1.000000000000000000",
            quantity="-0.050000000000000000000000",
            accruals=[
                {
                    "date": "2020-01-03",
                    "amount_per_split_normalized_share": "0.200000000000000000",
                    "signed_accrual_pnl": "-0.010000000000000000",
                    "action_key": {"action": "dividend", "name": "A"},
                },
                {
                    "date": "2020-01-03",
                    "amount_per_split_normalized_share": "0.300000000000000000",
                    "signed_accrual_pnl": "-0.015000000000000000",
                    "action_key": {"action": "dividend", "name": "B"},
                },
            ],
        ),
    ]
    prices = {
        ("LONG", "2020-01-02"): Decimal("10"),
        ("LONG", "2020-01-03"): Decimal("11"),
        ("LONG", "2020-01-06"): Decimal("12"),
        ("SHORT", "2020-01-02"): Decimal("20"),
        ("SHORT", "2020-01-03"): Decimal("19"),
        ("SHORT", "2020-01-06"): Decimal("18"),
    }
    summaries = [
        {
            "cohort_id": "pooled:2020-01-01:2",
            "names_per_leg": 1,
            "terminal_nav": "1.263000000000000000",
        }
    ]
    with localcontext() as context:
        context.prec = 50
        context.rounding = ROUND_HALF_EVEN
        constituents, cohorts = _cohort_daily_rows(
            rows,
            summaries,
            ["2020-01-02", "2020-01-03", "2020-01-06"],
            prices,
        )

    assert len(constituents) == 6
    assert len(cohorts) == 3
    assert cohorts[0]["settled_cash"] == "0.994000000000000000"
    assert cohorts[1]["settled_cash"] == "0.994000000000000000"
    assert cohorts[1]["candidate_distribution_receivable"] == (
        "-0.025000000000000000"
    )
    assert cohorts[-1]["settled_cash"] == "1.288000000000000000"
    assert cohorts[-1]["candidate_distribution_receivable"] == (
        "-0.025000000000000000"
    )
    assert cohorts[-1]["market_value"] == "0.000000000000000000"
    assert cohorts[-1]["nav"] == "1.263000000000000000"
    assert cohorts[-1]["open_position_count"] == 0
    short_dividend = next(
        row
        for row in constituents
        if row["ticker"] == "SHORT" and row["session_date"] == "2020-01-03"
    )
    assert short_dividend["candidate_distribution_accrual"] == (
        "-0.025000000000000000"
    )
    assert short_dividend["applied_distribution_action_keys"] == [
        {"action": "dividend", "name": "A"},
        {"action": "dividend", "name": "B"},
    ]


def test_daily_path_uses_entry_mark_and_flat_exit_orders():
    rows = [
        _row(
            ticker="LONG",
            m_ticker="LONG",
            rank=1,
            leg="long",
            target="1.000000000000000000",
            quantity="0.100000000000000000000000",
        ),
        _row(
            ticker="SHORT",
            m_ticker="SHORT",
            rank=2,
            leg="short",
            target="-1.000000000000000000",
            quantity="-0.050000000000000000000000",
        ),
    ]
    prices = {
        ("LONG", session): value
        for session, value in zip(
            ["2020-01-02", "2020-01-03", "2020-01-06"],
            map(Decimal, ("10", "11", "12")),
            strict=True,
        )
    } | {
        ("SHORT", session): value
        for session, value in zip(
            ["2020-01-02", "2020-01-03", "2020-01-06"],
            map(Decimal, ("20", "19", "18")),
            strict=True,
        )
    }
    with localcontext() as context:
        context.prec = 50
        constituents, cohorts = _cohort_daily_rows(
            rows,
            [
                {
                    "cohort_id": rows[0]["cohort_id"],
                    "names_per_leg": 1,
                    "terminal_nav": "1.288000000000000000",
                }
            ],
            ["2020-01-02", "2020-01-03", "2020-01-06"],
            prices,
        )

    long_rows = [row for row in constituents if row["ticker"] == "LONG"]
    assert [row["checkpoint"] for row in long_rows] == [
        "entry_close",
        "mark_close",
        "exit_close",
    ]
    assert [row["order"] for row in long_rows] == [
        "0.100000000000000000000000",
        "0.000000000000000000000000",
        "-0.100000000000000000000000",
    ]
    assert long_rows[-1]["position"] == "0.000000000000000000000000"
    assert cohorts[-1]["cumulative_fees"] == "0.012000000000000000"


def test_daily_accounting_uses_exact_target_and_entry_close_not_serialized_quantity():
    rows = [
        _row(
            ticker="LONG",
            m_ticker="LONG",
            rank=1,
            leg="long",
            target="1.000000000000000000",
            quantity="0.333333333333333333333333",
        ),
        _row(
            ticker="SHORT",
            m_ticker="SHORT",
            rank=2,
            leg="short",
            target="-1.000000000000000000",
            quantity="-0.142857142857142857142857",
        ),
    ]
    prices = {
        ("LONG", "2020-01-02"): Decimal("3"),
        ("LONG", "2020-01-03"): Decimal("6"),
        ("LONG", "2020-01-06"): Decimal("6"),
        ("SHORT", "2020-01-02"): Decimal("7"),
        ("SHORT", "2020-01-03"): Decimal("7"),
        ("SHORT", "2020-01-06"): Decimal("7"),
    }

    with localcontext() as context:
        context.prec = 50
        constituents, cohorts = _cohort_daily_rows(
            rows,
            [
                {
                    "cohort_id": rows[0]["cohort_id"],
                    "names_per_leg": 1,
                    "terminal_nav": "1.988000000000000000",
                }
            ],
            ["2020-01-02", "2020-01-03", "2020-01-06"],
            prices,
        )

    long_mark = next(
        row
        for row in constituents
        if row["ticker"] == "LONG" and row["session_date"] == "2020-01-03"
    )
    assert long_mark["price_pnl"] == "1.000000000000000000"
    assert long_mark["net_pnl_contribution"] == "1.000000000000000000"
    assert cohorts[-1]["nav"] == "1.988000000000000000"
