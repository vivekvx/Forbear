"""State machine tests.

Two things matter here: that the transition table is enforced exactly, and that
concurrency cannot produce a second successful transition out of a state that
has already been left. The second is what protects attempt counting.
"""

from __future__ import annotations

import asyncio

import pytest

from forbear.core import audit
from forbear.core.state_machine import (
    ALLOWED_TRANSITIONS,
    InvalidTransitionError,
    MissingSkipReasonError,
    transition,
)
from forbear.models.models import RecordStatus
from tests.conftest import insert_record

pytestmark = pytest.mark.asyncio

ALL_STATUSES = list(RecordStatus)

VALID_PAIRS = [
    (source, target)
    for source, targets in ALLOWED_TRANSITIONS.items()
    for target in sorted(targets, key=lambda status: status.value)
]

INVALID_PAIRS = [
    (source, target)
    for source in ALL_STATUSES
    for target in ALL_STATUSES
    if target not in ALLOWED_TRANSITIONS[source]
]


@pytest.mark.parametrize(
    ("source", "target"),
    VALID_PAIRS,
    ids=[f"{s.value}->{t.value}" for s, t in VALID_PAIRS],
)
async def test_valid_transition_succeeds(conn, source, target):
    record_id = await insert_record(conn, status=source.value, suffix="valid")
    reason = "not_worth_chasing" if target is RecordStatus.SKIPPED else None

    updated = await transition(conn, record_id, target, reason=reason)

    assert updated.status is target
    stored = await conn.fetchval(
        "SELECT status FROM at_risk_records WHERE id = $1", record_id
    )
    assert stored == target.value


@pytest.mark.parametrize(
    ("source", "target"),
    INVALID_PAIRS,
    ids=[f"{s.value}->{t.value}" for s, t in INVALID_PAIRS],
)
async def test_invalid_transition_raises(conn, source, target):
    record_id = await insert_record(conn, status=source.value, suffix="invalid")

    with pytest.raises(InvalidTransitionError) as excinfo:
        await transition(conn, record_id, target, reason="reason_code")

    assert excinfo.value.current is source
    assert excinfo.value.attempted is target

    unchanged = await conn.fetchval(
        "SELECT status FROM at_risk_records WHERE id = $1", record_id
    )
    assert unchanged == source.value


async def test_skip_without_reason_is_refused(conn):
    record_id = await insert_record(conn, suffix="noreason")

    with pytest.raises(MissingSkipReasonError):
        await transition(conn, record_id, RecordStatus.SKIPPED)

    unchanged = await conn.fetchval(
        "SELECT status FROM at_risk_records WHERE id = $1", record_id
    )
    assert unchanged == "open"


async def test_skip_records_reason_and_audit_entry(conn):
    record_id = await insert_record(conn, suffix="skip")

    updated = await transition(
        conn, record_id, RecordStatus.SKIPPED, reason="churn_risk_exceeds_invoice"
    )

    assert updated.skip_reason == "churn_risk_exceeds_invoice"
    entries = await conn.fetch(
        """
        SELECT action, details FROM audit_log
        WHERE entity_type = 'at_risk_record' AND entity_id = $1
        ORDER BY id
        """,
        str(record_id),
    )
    assert len(entries) == 1
    assert entries[0]["action"] == "transition:open->skipped"
    assert await audit.verify_chain(conn, "at_risk_record", record_id) is None


async def test_transition_outside_transaction_is_refused(db_pool):
    async with db_pool.acquire() as connection:
        async with connection.transaction():
            record_id = await insert_record(connection, suffix="notx")
        try:
            with pytest.raises(RuntimeError):
                await transition(connection, record_id, RecordStatus.SCHEDULED)
        finally:
            async with connection.transaction():
                await connection.execute(
                    "DELETE FROM at_risk_records WHERE id = $1", record_id
                )


async def test_concurrent_transitions_elect_exactly_one_winner(clean_db):
    """50 tasks race open -> scheduled. One wins, 49 must fail.

    Each task uses its own connection and transaction, so the only thing
    serialising them is the SELECT FOR UPDATE inside transition().
    """
    pool = clean_db
    async with pool.acquire() as connection:
        async with connection.transaction():
            record_id = await insert_record(connection, suffix="race")

    async def attempt_transition():
        async with pool.acquire() as connection:
            async with connection.transaction():
                return await transition(
                    connection, record_id, RecordStatus.SCHEDULED
                )

    results = await asyncio.gather(
        *(attempt_transition() for _ in range(50)), return_exceptions=True
    )

    winners = [r for r in results if not isinstance(r, BaseException)]
    losers = [r for r in results if isinstance(r, BaseException)]

    assert len(winners) == 1, f"expected 1 winner, got {len(winners)}"
    assert len(losers) == 49
    assert all(isinstance(error, InvalidTransitionError) for error in losers)
    assert all(error.current is RecordStatus.SCHEDULED for error in losers)

    async with pool.acquire() as connection:
        status = await connection.fetchval(
            "SELECT status FROM at_risk_records WHERE id = $1", record_id
        )
        entry_count = await connection.fetchval(
            """
            SELECT count(*) FROM audit_log
            WHERE entity_type = 'at_risk_record' AND entity_id = $1
            """,
            str(record_id),
        )
        chain_intact = await audit.verify_chain(
            connection, "at_risk_record", record_id
        )

    assert status == "scheduled"
    # Exactly one transition happened, so exactly one entry may exist.
    assert entry_count == 1
    assert chain_intact is None
