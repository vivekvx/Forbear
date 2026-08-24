"""Lifecycle of an at-risk record.

The transition table below is the whole specification. A status change that is
not in it cannot happen: transition() re-reads the record under SELECT FOR
UPDATE and refuses anything the table does not permit, so two callers racing on
the same record cannot both act on a stale status.

Every transition, including a skip, writes an audit entry in the same
transaction as the status change. There is no code path that moves a record
without leaving a trace.
"""

from __future__ import annotations

from typing import Optional

from forbear.core.audit import append_entry, server_now
from forbear.models.models import (
    AtRiskRecord,
    RecordStatus,
    at_risk_record_from_row,
)

ENTITY_TYPE = "at_risk_record"

# status -> statuses reachable from it. Terminal statuses map to an empty set.
#
# open -> recovered and scheduled -> recovered exist for one reason: the
# customer can pay out of band at any moment, and a payment.captured webhook
# then arrives for a record Forbear has not acted on yet. Without those edges
# that real event is unrepresentable, and ingestion would have to either drop
# the recovery or fake a scheduled/in_flight history that never happened.
ALLOWED_TRANSITIONS: dict[RecordStatus, frozenset[RecordStatus]] = {
    RecordStatus.OPEN: frozenset(
        {
            RecordStatus.SCHEDULED,
            RecordStatus.SKIPPED,
            RecordStatus.ABANDONED,
            RecordStatus.RECOVERED,
        }
    ),
    RecordStatus.SCHEDULED: frozenset(
        {
            RecordStatus.IN_FLIGHT,
            RecordStatus.ABANDONED,
            RecordStatus.RECOVERED,
        }
    ),
    RecordStatus.IN_FLIGHT: frozenset(
        {
            RecordStatus.RECOVERED,
            RecordStatus.OPEN,  # retry eligible
            RecordStatus.ABANDONED,
        }
    ),
    RecordStatus.RECOVERED: frozenset(),
    RecordStatus.ABANDONED: frozenset(),
    RecordStatus.SKIPPED: frozenset(),
}


class InvalidTransitionError(Exception):
    """A status change the transition table does not permit."""

    def __init__(
        self,
        current: RecordStatus,
        attempted: RecordStatus,
        record_id: Optional[int] = None,
    ) -> None:
        self.current = current
        self.attempted = attempted
        self.record_id = record_id
        where = f" (record {record_id})" if record_id is not None else ""
        super().__init__(
            f"cannot transition from {current.value} to {attempted.value}{where}"
        )


class MissingSkipReasonError(Exception):
    """A skip was requested without the reason code it must carry."""


def is_allowed(current: RecordStatus, new_status: RecordStatus) -> bool:
    return new_status in ALLOWED_TRANSITIONS[current]


async def transition(
    conn,
    record: AtRiskRecord | int,
    new_status: RecordStatus,
    reason: Optional[str] = None,
) -> AtRiskRecord:
    """Move a record to new_status, or raise.

    Must run inside a transaction: the row lock taken here has to be held until
    the status change and its audit entry commit together.

    Accepts a record or a bare id. The in-memory status of a passed record is
    never trusted; the status read under the lock is the only one that counts.
    """
    if not conn.is_in_transaction():
        raise RuntimeError("transition must run inside a transaction")

    record_id = record.id if isinstance(record, AtRiskRecord) else int(record)
    new_status = RecordStatus(new_status)

    if new_status is RecordStatus.SKIPPED and not reason:
        raise MissingSkipReasonError(
            f"skipping record {record_id} requires a reason code"
        )

    row = await conn.fetchrow(
        """
        SELECT id, status
        FROM at_risk_records
        WHERE id = $1
        FOR UPDATE
        """,
        record_id,
    )
    if row is None:
        raise LookupError(f"at_risk_record {record_id} does not exist")

    current = RecordStatus(row["status"])
    if not is_allowed(current, new_status):
        raise InvalidTransitionError(current, new_status, record_id)

    now = await server_now(conn)
    skip_reason = reason if new_status is RecordStatus.SKIPPED else None

    updated = await conn.fetchrow(
        """
        UPDATE at_risk_records
        SET status = $2::record_status,
            skip_reason = $3,
            updated_at = $4
        WHERE id = $1
        RETURNING id, subscription_id, customer_id, invoice_id, amount,
                  failure_code, failure_class, status, uplift_score,
                  whittle_index, skip_reason, created_at, updated_at
        """,
        record_id,
        new_status.value,
        skip_reason,
        now,
    )

    await append_entry(
        conn,
        ENTITY_TYPE,
        record_id,
        f"transition:{current.value}->{new_status.value}",
        {"from": current.value, "to": new_status.value, "reason": reason},
    )

    return at_risk_record_from_row(updated)
