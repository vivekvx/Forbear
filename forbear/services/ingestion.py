"""Razorpay events -> Forbear records.

Every at_risk_record in the system is born here, so the rules are conservative:

  * Nothing is dropped silently. A code the classifier does not know still
    produces a record, with failure_class NULL and an audit entry naming the
    code. A record on the exception list is visible; a discarded event is not.
  * Every write is keyed so a replay collides instead of duplicating. Customers
    and subscriptions upsert on Razorpay's external id, records on invoice_id.
  * Status changes go through the state machine, never through UPDATE. An event
    that would require an illegal transition is recorded and refused, not
    forced.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from forbear.core.audit import append_entry
from forbear.core.state_machine import transition
from forbear.models.models import MandateStatus, RecordStatus
from forbear.services.classifier import (
    MAPPING_VERSION,
    UnknownFailureCode,
    classify,
)

logger = logging.getLogger(__name__)

ENTITY_TYPE = "at_risk_record"

DEFAULT_BILLING_CYCLE_DAYS = 30

_PERIOD_DAYS = {
    "daily": 1,
    "weekly": 7,
    "monthly": 30,
    "quarterly": 90,
    "yearly": 365,
}

# Razorpay subscription status -> our mandate status. A halted subscription
# still has a valid mandate; Razorpay has simply stopped using it, which is the
# whole reason Forbear exists.
_MANDATE_STATUS_BY_SUBSCRIPTION_STATUS = {
    "created": MandateStatus.ACTIVE,
    "authenticated": MandateStatus.ACTIVE,
    "active": MandateStatus.ACTIVE,
    "pending": MandateStatus.ACTIVE,
    "halted": MandateStatus.ACTIVE,
    "paused": MandateStatus.PAUSED,
    "cancelled": MandateStatus.REVOKED,
    "expired": MandateStatus.EXPIRED,
    "completed": MandateStatus.EXPIRED,
}

RECOVERABLE_STATUSES = (
    RecordStatus.OPEN,
    RecordStatus.SCHEDULED,
    RecordStatus.IN_FLIGHT,
)


class MalformedPayload(Exception):
    """The event does not carry enough to act on."""


def _entity(event: dict, name: str) -> dict:
    payload = event.get("payload") or {}
    section = payload.get(name) or {}
    return section.get("entity") or {}


def _subscription_external_id(event: dict) -> str:
    value = (
        _entity(event, "subscription").get("id")
        or _entity(event, "payment").get("subscription_id")
        or _entity(event, "invoice").get("subscription_id")
    )
    if not value:
        raise MalformedPayload("event carries no subscription id")
    return value


def _customer_external_id(event: dict) -> str:
    value = (
        _entity(event, "subscription").get("customer_id")
        or _entity(event, "payment").get("customer_id")
        or _entity(event, "invoice").get("customer_id")
    )
    if not value:
        raise MalformedPayload("event carries no customer id")
    return value


def _invoice_id(event: dict) -> Optional[str]:
    return _entity(event, "invoice").get("id") or _entity(event, "payment").get(
        "invoice_id"
    )


def _amount(event: dict) -> Optional[int]:
    for value in (
        _entity(event, "payment").get("amount"),
        _entity(event, "invoice").get("amount"),
        _entity(event, "subscription").get("plan_amount"),
    ):
        if value:
            return int(value)
    return None


def _failure_fields(event: dict) -> tuple[Optional[str], Optional[str]]:
    payment = _entity(event, "payment")
    nested = payment.get("error") or {}
    code = payment.get("error_code") or nested.get("code")
    reason = payment.get("error_reason") or nested.get("reason")
    return code, reason


def _plan_amount(event: dict) -> Optional[int]:
    """Only an amount the subscription itself states.

    A payment amount is what was attempted, which is not necessarily what the
    plan costs.
    """
    value = _entity(event, "subscription").get("plan_amount")
    return int(value) if value else None


def _billing_cycle_days(event: dict) -> Optional[int]:
    subscription = _entity(event, "subscription")
    explicit = subscription.get("billing_cycle_days")
    if explicit:
        return int(explicit)

    plan = _entity(event, "plan") or subscription.get("plan") or {}
    period = str(plan.get("period") or subscription.get("period") or "").lower()
    if period not in _PERIOD_DAYS:
        return None
    interval = int(plan.get("interval") or subscription.get("interval") or 1)
    return _PERIOD_DAYS[period] * max(interval, 1)


def _mandate_status(event: dict) -> Optional[str]:
    subscription = _entity(event, "subscription")

    explicit = subscription.get("mandate_status")
    if explicit:
        return MandateStatus(explicit).value

    mapped = _MANDATE_STATUS_BY_SUBSCRIPTION_STATUS.get(
        str(subscription.get("status") or "").lower()
    )
    # None leaves whatever is already stored alone. An unrecognised Razorpay
    # status must never downgrade a mandate we know to be revoked.
    return mapped.value if mapped else None


async def _upsert_customer(conn, external_id: str) -> int:
    # DO UPDATE rather than DO NOTHING: a no-op update still RETURNINGs the id,
    # which DO NOTHING does not do on conflict.
    return await conn.fetchval(
        """
        INSERT INTO customers (external_id)
        VALUES ($1)
        ON CONFLICT (external_id)
        DO UPDATE SET external_id = EXCLUDED.external_id
        RETURNING id
        """,
        external_id,
    )


async def _upsert_subscription(
    conn,
    *,
    customer_id: int,
    external_id: str,
    plan_amount: Optional[int],
    fallback_plan_amount: int,
    billing_cycle_days: Optional[int],
    mandate_status: Optional[str],
) -> int:
    """Conflict key is Razorpay's external_id, never our primary key.

    Every nullable field is COALESCEd against what is already stored, so an
    event that omits a field leaves it as it was instead of overwriting it.
    """
    return await conn.fetchval(
        """
        INSERT INTO subscriptions
            (customer_id, external_id, plan_amount, billing_cycle_days,
             mandate_status)
        VALUES (
            $1,
            $2,
            COALESCE($3::bigint, $4::bigint),
            COALESCE($5::integer, $6::integer),
            COALESCE($7::mandate_status, 'active')
        )
        ON CONFLICT (external_id) DO UPDATE
        SET plan_amount = COALESCE($3::bigint, subscriptions.plan_amount),
            billing_cycle_days = COALESCE(
                $5::integer, subscriptions.billing_cycle_days
            ),
            mandate_status = COALESCE(
                $7::mandate_status, subscriptions.mandate_status
            )
        RETURNING id
        """,
        customer_id,
        external_id,
        plan_amount,
        fallback_plan_amount,
        billing_cycle_days,
        DEFAULT_BILLING_CYCLE_DAYS,
        mandate_status,
    )


async def _sync_subscription(conn, event: dict) -> tuple[int, int]:
    """Upsert the customer and subscription an event refers to."""
    amount = _amount(event)
    if amount is None:
        raise MalformedPayload("event carries no amount")

    customer_id = await _upsert_customer(conn, _customer_external_id(event))
    subscription_id = await _upsert_subscription(
        conn,
        customer_id=customer_id,
        external_id=_subscription_external_id(event),
        plan_amount=_plan_amount(event),
        fallback_plan_amount=amount,
        billing_cycle_days=_billing_cycle_days(event),
        mandate_status=_mandate_status(event),
    )
    return customer_id, subscription_id


def _classify(
    failure_code: str, failure_reason: Optional[str]
) -> tuple[Optional[str], Optional[dict[str, Any]]]:
    """Returns (class value or None, details of the failure to classify)."""
    try:
        return classify(failure_code, failure_reason).value, None
    except UnknownFailureCode as error:
        return None, {
            "failure_code": error.failure_code,
            "failure_reason": error.failure_reason,
            "mapping_version": error.mapping_version,
        }


def _failure_code_of(event: dict) -> tuple[str, Optional[str]]:
    code, reason = _failure_fields(event)
    # failure_code is NOT NULL. An event with no code at all still describes a
    # failure, so it gets a placeholder the classifier will reject, routing the
    # record to the exception list rather than to a guessed class.
    return code or reason or "UNKNOWN", reason


async def _create_at_risk_record(
    conn, event: dict, *, event_type: str
) -> tuple[Optional[int], bool]:
    """Create the record for this event. Returns (id, created)."""
    customer_id, subscription_id = await _sync_subscription(conn, event)

    amount = _amount(event)
    subscription_external_id = _subscription_external_id(event)
    # A halted subscription can arrive with no invoice reference at all. The
    # placeholder is derived, not random, so a replay collides on it instead of
    # creating a second record; a later event carrying the real invoice makes
    # its own record.
    invoice_id = _invoice_id(event) or f"{subscription_external_id}:{event_type}"

    failure_code, failure_reason = _failure_code_of(event)
    failure_class, unknown = _classify(failure_code, failure_reason)

    record_id = await conn.fetchval(
        """
        INSERT INTO at_risk_records
            (subscription_id, customer_id, invoice_id, amount, failure_code,
             failure_class, status)
        VALUES ($1, $2, $3, $4, $5, $6::failure_class, 'open')
        ON CONFLICT (invoice_id) DO NOTHING
        RETURNING id
        """,
        subscription_id,
        customer_id,
        invoice_id,
        amount,
        failure_code,
        failure_class,
    )

    if record_id is None:
        existing = await conn.fetchval(
            "SELECT id FROM at_risk_records WHERE invoice_id = $1", invoice_id
        )
        return existing, False

    await append_entry(
        conn,
        ENTITY_TYPE,
        record_id,
        "at_risk_record_created",
        {
            "event": event_type,
            "invoice_id": invoice_id,
            "amount": amount,
            "failure_code": failure_code,
            "failure_class": failure_class,
            "mapping_version": MAPPING_VERSION,
        },
    )

    if unknown is not None:
        # Loud, attributable, and still recoverable by a human. The alternative
        # is a default class, which is a guess wearing a fact's clothes.
        logger.warning(
            "unknown failure code %r on record %s", failure_code, record_id
        )
        await append_entry(
            conn, ENTITY_TYPE, record_id, "unknown_failure_code", unknown
        )

    return record_id, True


async def _find_record_for_event(conn, event: dict, *, only_recoverable: bool):
    """Locate the record an event refers to, locked for update."""
    invoice_id = _invoice_id(event)
    if invoice_id:
        row = await conn.fetchrow(
            "SELECT id, status FROM at_risk_records WHERE invoice_id = $1 "
            "FOR UPDATE",
            invoice_id,
        )
        if row is not None:
            return row

    try:
        subscription_external_id = _subscription_external_id(event)
    except MalformedPayload:
        return None

    statuses = [
        status.value
        for status in (RECOVERABLE_STATUSES if only_recoverable else RecordStatus)
    ]
    return await conn.fetchrow(
        """
        SELECT r.id, r.status
        FROM at_risk_records r
        JOIN subscriptions s ON s.id = r.subscription_id
        WHERE s.external_id = $1
          AND r.status = ANY($2::record_status[])
        ORDER BY r.id DESC
        LIMIT 1
        FOR UPDATE OF r
        """,
        subscription_external_id,
        statuses,
    )


async def handle_subscription_pending(conn, event: dict) -> Optional[int]:
    """Razorpay is retrying. This is where most records originate."""
    record_id, _created = await _create_at_risk_record(
        conn, event, event_type="subscription.pending"
    )
    return record_id


async def handle_subscription_halted(conn, event: dict) -> Optional[int]:
    """Razorpay has given up. Forbear takes over from here."""
    row = await _find_record_for_event(conn, event, only_recoverable=False)

    if row is None:
        # Halted without ever passing through pending. Still ours to work.
        record_id, _created = await _create_at_risk_record(
            conn, event, event_type="subscription.halted"
        )
        return record_id

    await _sync_subscription(conn, event)

    record_id = row["id"]
    status = RecordStatus(row["status"])

    if status is RecordStatus.OPEN:
        return record_id
    if status is RecordStatus.IN_FLIGHT:
        await transition(
            conn,
            record_id,
            RecordStatus.OPEN,
            reason="webhook:subscription.halted",
        )
        return record_id

    # scheduled, or already terminal. Reopening is not a legal move from here,
    # so record that the event arrived rather than forcing the status.
    await append_entry(
        conn,
        ENTITY_TYPE,
        record_id,
        "halted_without_reopen",
        {"event": "subscription.halted", "status": status.value},
    )
    return record_id


async def handle_payment_captured(conn, event: dict) -> Optional[int]:
    """Money arrived. Possibly ours, possibly the customer paying out of band."""
    row = await _find_record_for_event(conn, event, only_recoverable=True)
    if row is None:
        return None

    record_id = row["id"]
    status = RecordStatus(row["status"])
    if status not in RECOVERABLE_STATUSES:
        return record_id

    event_type = event.get("event") or "payment.captured"
    await transition(
        conn, record_id, RecordStatus.RECOVERED, reason=f"webhook:{event_type}"
    )
    return record_id


async def handle_subscription_charged(conn, event: dict) -> Optional[int]:
    """A successful charge on the subscription closes the record the same way."""
    return await handle_payment_captured(conn, event)


async def handle_payment_failed(conn, event: dict) -> Optional[int]:
    """A new decline, or a fresh reason for one already known."""
    row = await _find_record_for_event(conn, event, only_recoverable=False)

    if row is None:
        record_id, _created = await _create_at_risk_record(
            conn, event, event_type="payment.failed"
        )
        return record_id

    await _sync_subscription(conn, event)

    record_id = row["id"]
    failure_code, failure_reason = _failure_code_of(event)
    failure_class, unknown = _classify(failure_code, failure_reason)

    await conn.execute(
        """
        UPDATE at_risk_records
        SET failure_code = $2,
            failure_class = $3::failure_class,
            updated_at = now()
        WHERE id = $1
        """,
        record_id,
        failure_code,
        failure_class,
    )
    await append_entry(
        conn,
        ENTITY_TYPE,
        record_id,
        "failure_reclassified",
        {
            "event": "payment.failed",
            "failure_code": failure_code,
            "failure_class": failure_class,
            "mapping_version": MAPPING_VERSION,
        },
    )

    if unknown is not None:
        logger.warning(
            "unknown failure code %r on record %s", failure_code, record_id
        )
        await append_entry(
            conn, ENTITY_TYPE, record_id, "unknown_failure_code", unknown
        )

    return record_id


# Events with no entry here are stored and ignored. Razorpay adds event types
# without asking, and an unknown one is not an error.
HANDLERS = {
    "subscription.pending": handle_subscription_pending,
    "subscription.halted": handle_subscription_halted,
    "subscription.charged": handle_subscription_charged,
    "payment.captured": handle_payment_captured,
    "payment.failed": handle_payment_failed,
}
