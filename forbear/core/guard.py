"""The last gate before any outbound action.

Invariant 2: the allocator plans, the guard permits, and they share no code.
This module imports only forbear.config.limits and forbear.models.models. It
does not import anything from forbear.services, and it never calls the
allocator. If the allocator is wrong, the only thing between that bug and a
customer's bank account is the code in this file, so it re-derives every
constraint from the database rather than believing anything it was handed.

Two consequences of that, both deliberate:

  * _server_now() below duplicates one line of forbear.core.audit. Importing it
    would make the planning path and the permission path share a clock, and one
    line is a cheap price for not sharing.
  * Every check re-reads its own state. The at_risk_record argument supplies an
    id and nothing else.

Checks run in a fixed order and stop at the first failure, so a blocked verdict
names the first rule violated, not all of them.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional

from forbear.config.limits import (
    ATTEMPT_COOLDOWN,
    IST,
    MAX_ATTEMPTS,
    NOTIFICATION_VALIDITY,
    NPCI_DEBIT_WINDOWS_IST,
)
from forbear.models.models import (
    AtRiskRecord,
    AttemptOutcome,
    ContactPurpose,
    MandateStatus,
    ProposedAction,
)

RULE_RECORD_EXISTS = "record_exists"
RULE_MANDATE_NOT_REVOKED = "mandate_not_revoked"
RULE_MANDATE_NOT_EXPIRED = "mandate_not_expired"
RULE_ATTEMPT_CAP = "attempt_cap_not_exceeded"
RULE_COOLDOWN = "cooldown_satisfied"
RULE_EXECUTION_WINDOW = "execution_window_legal"
RULE_NOTIFICATION = "notification_sent"
RULE_DUPLICATE_ATTEMPT = "duplicate_attempt"


@dataclass(frozen=True)
class GuardVerdict:
    """The guard's answer.

    details is stored verbatim in attempts.guard_verdict, so every value in it
    must be JSON-serialisable.
    """

    allowed: bool
    rule_name: Optional[str]
    details: dict[str, Any]


def _pass(**details: Any) -> GuardVerdict:
    return GuardVerdict(allowed=True, rule_name=None, details=details)


def _block(rule_name: str, **details: Any) -> GuardVerdict:
    return GuardVerdict(allowed=False, rule_name=rule_name, details=details)


async def _server_now(conn) -> datetime:
    """The guard's own clock read.

    Not imported from forbear.core.audit: see the module docstring. Tests patch
    this to pin the instant every rule is evaluated against.
    """
    return await conn.fetchval("SELECT now()")


async def _mandate_status(conn, record_id: int) -> Optional[str]:
    return await conn.fetchval(
        """
        SELECT s.mandate_status
        FROM at_risk_records r
        JOIN subscriptions s ON s.id = r.subscription_id
        WHERE r.id = $1
        """,
        record_id,
    )


async def _check_mandate_not_revoked(
    conn, record_id: int, action: ProposedAction, now: datetime
) -> GuardVerdict:
    status = await _mandate_status(conn, record_id)
    if status is None:
        return _block(RULE_MANDATE_NOT_REVOKED, reason="mandate_not_found")
    if status == MandateStatus.REVOKED.value:
        return _block(RULE_MANDATE_NOT_REVOKED, mandate_status=status)
    return _pass(mandate_status=status)


async def _check_mandate_not_expired(
    conn, record_id: int, action: ProposedAction, now: datetime
) -> GuardVerdict:
    status = await _mandate_status(conn, record_id)
    if status is None:
        return _block(RULE_MANDATE_NOT_EXPIRED, reason="mandate_not_found")
    if status == MandateStatus.EXPIRED.value:
        return _block(RULE_MANDATE_NOT_EXPIRED, mandate_status=status)
    return _pass(mandate_status=status)


async def _check_attempt_cap_not_exceeded(
    conn, record_id: int, action: ProposedAction, now: datetime
) -> GuardVerdict:
    """Checks the attempts already consumed and the number being proposed.

    Counting alone would let an allocator bug slip attempt 9 through on a
    record that has no attempt rows yet.

    FOR UPDATE locks the rows this count is derived from, so two guards cannot
    both read "3 used" for the same record. It cannot stop a phantom insert
    arriving from elsewhere; UNIQUE (at_risk_record_id, attempt_number) is what
    stops that, and the two together are what make the cap hold.
    """
    rows = await conn.fetch(
        "SELECT id FROM attempts WHERE at_risk_record_id = $1 FOR UPDATE",
        record_id,
    )
    used = len(rows)

    if used >= MAX_ATTEMPTS:
        return _block(
            RULE_ATTEMPT_CAP,
            attempts_used=used,
            max_attempts=MAX_ATTEMPTS,
            proposed_attempt_number=action.attempt_number,
            reason="cap_already_consumed",
        )
    if action.attempt_number > MAX_ATTEMPTS:
        return _block(
            RULE_ATTEMPT_CAP,
            attempts_used=used,
            max_attempts=MAX_ATTEMPTS,
            proposed_attempt_number=action.attempt_number,
            reason="proposed_attempt_number_above_cap",
        )
    return _pass(attempts_used=used, max_attempts=MAX_ATTEMPTS)


async def _check_cooldown_satisfied(
    conn, record_id: int, action: ProposedAction, now: datetime
) -> GuardVerdict:
    last_executed = await conn.fetchval(
        """
        SELECT max(executed_at)
        FROM attempts
        WHERE at_risk_record_id = $1 AND executed_at IS NOT NULL
        """,
        record_id,
    )
    if last_executed is None:
        return _pass(last_executed_at=None)

    ready_at = last_executed + ATTEMPT_COOLDOWN
    if not ready_at < now:
        return _block(
            RULE_COOLDOWN,
            last_executed_at=last_executed.isoformat(),
            ready_at=ready_at.isoformat(),
            server_now=now.isoformat(),
            cooldown_seconds=int(ATTEMPT_COOLDOWN.total_seconds()),
        )
    return _pass(last_executed_at=last_executed.isoformat())


def _window_label(start: int, end: int) -> str:
    return f"{start // 60:02d}:{start % 60:02d}-{end // 60:02d}:{end % 60:02d}"


async def _check_execution_window_legal(
    conn, record_id: int, action: ProposedAction, now: datetime
) -> GuardVerdict:
    """now is already a database timestamp; only the fixed offset is applied."""
    ist_now = now.astimezone(IST)
    minute_of_day = ist_now.hour * 60 + ist_now.minute

    for start, end in NPCI_DEBIT_WINDOWS_IST:
        if start <= minute_of_day < end:
            return _pass(
                ist_time=ist_now.strftime("%H:%M"),
                window=_window_label(start, end),
            )

    return _block(
        RULE_EXECUTION_WINDOW,
        ist_time=ist_now.strftime("%H:%M"),
        server_now=now.isoformat(),
        allowed_windows=[_window_label(s, e) for s, e in NPCI_DEBIT_WINDOWS_IST],
    )


async def _check_notification_sent(
    conn, record_id: int, action: ProposedAction, now: datetime
) -> GuardVerdict:
    """A pre-debit notification for this subscription, still inside its window.

    Scoped to the subscription, not the customer: a notification about one
    mandate authorises a debit on that mandate and no other.
    """
    subscription_id = await conn.fetchval(
        "SELECT subscription_id FROM at_risk_records WHERE id = $1", record_id
    )
    if subscription_id is None:
        return _block(RULE_NOTIFICATION, reason="subscription_not_found")

    sent_at = await conn.fetchval(
        """
        SELECT max(sent_at)
        FROM contacts
        WHERE subscription_id = $1 AND purpose = $2::contact_purpose
        """,
        subscription_id,
        ContactPurpose.PRE_DEBIT_NOTIFICATION.value,
    )
    if sent_at is None:
        return _block(
            RULE_NOTIFICATION,
            subscription_id=subscription_id,
            reason="no_notification_found",
        )

    valid_from = now - NOTIFICATION_VALIDITY
    if sent_at < valid_from:
        return _block(
            RULE_NOTIFICATION,
            subscription_id=subscription_id,
            sent_at=sent_at.isoformat(),
            valid_from=valid_from.isoformat(),
            reason="notification_expired",
        )
    if sent_at > now:
        # Not sent yet. Bad data, but fail closed rather than honour it.
        return _block(
            RULE_NOTIFICATION,
            subscription_id=subscription_id,
            sent_at=sent_at.isoformat(),
            server_now=now.isoformat(),
            reason="notification_in_the_future",
        )
    return _pass(subscription_id=subscription_id, sent_at=sent_at.isoformat())


async def _check_duplicate_attempt(
    conn, record_id: int, action: ProposedAction, now: datetime
) -> GuardVerdict:
    existing = await conn.fetchval(
        """
        SELECT id
        FROM attempts
        WHERE at_risk_record_id = $1
          AND attempt_number = $2
          AND outcome = $3::attempt_outcome
        """,
        record_id,
        action.attempt_number,
        AttemptOutcome.PENDING.value,
    )
    if existing is not None:
        return _block(
            RULE_DUPLICATE_ATTEMPT,
            attempt_number=action.attempt_number,
            existing_attempt_id=existing,
        )
    return _pass(attempt_number=action.attempt_number)


# Order matters: the most absolute constraints first, so a revoked mandate is
# reported as a revoked mandate rather than as a stale notification.
CHECKS: tuple[tuple[str, Any], ...] = (
    (RULE_MANDATE_NOT_REVOKED, _check_mandate_not_revoked),
    (RULE_MANDATE_NOT_EXPIRED, _check_mandate_not_expired),
    (RULE_ATTEMPT_CAP, _check_attempt_cap_not_exceeded),
    (RULE_COOLDOWN, _check_cooldown_satisfied),
    (RULE_EXECUTION_WINDOW, _check_execution_window_legal),
    (RULE_NOTIFICATION, _check_notification_sent),
    (RULE_DUPLICATE_ATTEMPT, _check_duplicate_attempt),
)


async def guard_check(
    conn, at_risk_record: AtRiskRecord | int, proposed_action: ProposedAction
) -> GuardVerdict:
    """Permit or refuse one proposed action.

    at_risk_record supplies an id and nothing more; every fact is re-read from
    the database. Returns the first blocking verdict, or an allowing verdict
    listing the checks that passed.
    """
    record_id = (
        at_risk_record.id
        if isinstance(at_risk_record, AtRiskRecord)
        else int(at_risk_record)
    )

    now = await _server_now(conn)

    exists = await conn.fetchval(
        "SELECT 1 FROM at_risk_records WHERE id = $1", record_id
    )
    if exists is None:
        return _block(RULE_RECORD_EXISTS, at_risk_record_id=record_id)

    passed: list[str] = []
    for rule_name, check in CHECKS:
        verdict = await check(conn, record_id, proposed_action, now)
        if not verdict.allowed:
            return verdict
        passed.append(rule_name)

    return _pass(
        at_risk_record_id=record_id,
        attempt_number=proposed_action.attempt_number,
        action_kind=proposed_action.kind.value,
        server_now=now.isoformat(),
        checks_passed=passed,
    )
