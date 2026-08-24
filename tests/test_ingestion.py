"""Ingestion tests.

The handlers are exercised directly, against a real database, inside the
rolled-back transaction fixture. What matters is what ends up in the tables:
one record per invoice however many times an event arrives, a NULL class rather
than a guessed one, and an audit trail for both.
"""

from __future__ import annotations

import json

import pytest

from forbear.core import audit
from forbear.core.state_machine import transition
from forbear.models.models import FailureClass, RecordStatus
from forbear.services.ingestion import (
    MalformedPayload,
    handle_payment_captured,
    handle_payment_failed,
    handle_subscription_charged,
    handle_subscription_halted,
    handle_subscription_pending,
)

pytestmark = pytest.mark.asyncio

ENTITY = "at_risk_record"


def event(
    event_type,
    *,
    subscription_id="sub_test_1",
    customer_id="cust_test_1",
    invoice_id="inv_test_1",
    amount=49900,
    error_code="INSUFFICIENT_FUNDS",
    error_reason=None,
    status="pending",
    mandate_status=None,
    plan_amount=None,
    include_payment=True,
):
    """A Razorpay event, shaped the way Razorpay shapes them."""
    subscription = {
        "id": subscription_id,
        "customer_id": customer_id,
        "status": status,
    }
    if mandate_status is not None:
        subscription["mandate_status"] = mandate_status
    if plan_amount is not None:
        subscription["plan_amount"] = plan_amount

    payload = {"subscription": {"entity": subscription}}
    if include_payment:
        payload["payment"] = {
            "entity": {
                "id": "pay_test_1",
                "subscription_id": subscription_id,
                "customer_id": customer_id,
                "invoice_id": invoice_id,
                "amount": amount,
                "error_code": error_code,
                "error_reason": error_reason,
            }
        }

    return {"entity": "event", "event": event_type, "payload": payload}


async def fetch_record(conn, record_id):
    return await conn.fetchrow(
        "SELECT * FROM at_risk_records WHERE id = $1", record_id
    )


async def audit_actions(conn, record_id):
    rows = await conn.fetch(
        """
        SELECT action FROM audit_log
        WHERE entity_type = $1 AND entity_id = $2
        ORDER BY id
        """,
        ENTITY,
        str(record_id),
    )
    return [row["action"] for row in rows]


async def test_pending_creates_customer_subscription_and_record(conn):
    record_id = await handle_subscription_pending(
        conn, event("subscription.pending")
    )

    record = await fetch_record(conn, record_id)
    assert record["status"] == "open"
    assert record["invoice_id"] == "inv_test_1"
    assert record["amount"] == 49900
    assert record["failure_code"] == "INSUFFICIENT_FUNDS"
    assert record["failure_class"] == FailureClass.TIME_DEPENDENT.value

    customer = await conn.fetchrow(
        "SELECT * FROM customers WHERE id = $1", record["customer_id"]
    )
    subscription = await conn.fetchrow(
        "SELECT * FROM subscriptions WHERE id = $1", record["subscription_id"]
    )
    assert customer["external_id"] == "cust_test_1"
    assert subscription["external_id"] == "sub_test_1"
    assert subscription["mandate_status"] == "active"

    assert await audit_actions(conn, record_id) == ["at_risk_record_created"]
    assert await audit.verify_chain(conn, ENTITY, record_id) is None


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        ("INSUFFICIENT_FUNDS", "time_dependent"),
        ("GATEWAY_ERROR", "transient"),
        ("MANDATE_EXPIRED", "reauth_required"),
        ("CUSTOMER_DISPUTED", "terminal"),
    ],
)
async def test_failure_class_comes_from_the_classifier(conn, code, expected):
    record_id = await handle_subscription_pending(
        conn, event("subscription.pending", error_code=code)
    )

    record = await fetch_record(conn, record_id)
    assert record["failure_class"] == expected


async def test_unknown_code_creates_record_with_null_class_and_audits_it(conn):
    record_id = await handle_subscription_pending(
        conn, event("subscription.pending", error_code="CARD_ON_FIRE")
    )

    record = await fetch_record(conn, record_id)
    assert record is not None, "an unknown code must not drop the record"
    assert record["failure_class"] is None
    assert record["failure_code"] == "CARD_ON_FIRE"

    assert await audit_actions(conn, record_id) == [
        "at_risk_record_created",
        "unknown_failure_code",
    ]
    entry = await conn.fetchrow(
        """
        SELECT details FROM audit_log
        WHERE entity_type = $1 AND entity_id = $2
          AND action = 'unknown_failure_code'
        """,
        ENTITY,
        str(record_id),
    )
    details = json.loads(entry["details"])
    assert details["failure_code"] == "CARD_ON_FIRE"
    assert details["mapping_version"]


async def test_event_with_no_failure_code_lands_on_the_exception_list(conn):
    record_id = await handle_subscription_pending(
        conn, event("subscription.pending", error_code=None)
    )

    record = await fetch_record(conn, record_id)
    assert record["failure_code"] == "UNKNOWN"
    assert record["failure_class"] is None
    assert "unknown_failure_code" in await audit_actions(conn, record_id)


async def test_duplicate_pending_does_not_create_a_second_record(conn):
    first = await handle_subscription_pending(conn, event("subscription.pending"))
    second = await handle_subscription_pending(conn, event("subscription.pending"))

    assert first == second
    count = await conn.fetchval(
        "SELECT count(*) FROM at_risk_records WHERE invoice_id = $1", "inv_test_1"
    )
    assert count == 1
    # And the replay adds no second creation entry.
    assert await audit_actions(conn, first) == ["at_risk_record_created"]


async def test_duplicate_pending_does_not_duplicate_customer_or_subscription(conn):
    await handle_subscription_pending(conn, event("subscription.pending"))
    await handle_subscription_pending(conn, event("subscription.pending"))

    assert await conn.fetchval("SELECT count(*) FROM customers") == 1
    assert await conn.fetchval("SELECT count(*) FROM subscriptions") == 1


async def test_halted_without_prior_pending_creates_the_record(conn):
    record_id = await handle_subscription_halted(
        conn, event("subscription.halted", status="halted")
    )

    record = await fetch_record(conn, record_id)
    assert record is not None
    assert record["status"] == "open"
    assert "at_risk_record_created" in await audit_actions(conn, record_id)


async def test_halted_without_any_invoice_reference_still_creates_a_record(conn):
    """No payment entity at all: the invoice id is derived so replays collide."""
    halted = dict(
        status="halted",
        include_payment=False,
        plan_amount=49900,
    )
    first = await handle_subscription_halted(
        conn, event("subscription.halted", **halted)
    )
    second = await handle_subscription_halted(
        conn, event("subscription.halted", **halted)
    )

    assert first is not None
    assert first == second
    assert await conn.fetchval("SELECT count(*) FROM at_risk_records") == 1


async def test_halted_after_pending_reuses_the_record(conn):
    pending_id = await handle_subscription_pending(
        conn, event("subscription.pending")
    )
    halted_id = await handle_subscription_halted(
        conn, event("subscription.halted", status="halted")
    )

    assert halted_id == pending_id
    assert await conn.fetchval("SELECT count(*) FROM at_risk_records") == 1


async def test_halted_updates_mandate_status_when_the_payload_carries_it(conn):
    record_id = await handle_subscription_pending(
        conn, event("subscription.pending")
    )
    await handle_subscription_halted(
        conn, event("subscription.halted", status="cancelled")
    )

    record = await fetch_record(conn, record_id)
    mandate_status = await conn.fetchval(
        "SELECT mandate_status FROM subscriptions WHERE id = $1",
        record["subscription_id"],
    )
    assert mandate_status == "revoked"


async def test_halted_reopens_an_in_flight_record(conn):
    record_id = await handle_subscription_pending(
        conn, event("subscription.pending")
    )
    await transition(conn, record_id, RecordStatus.SCHEDULED)
    await transition(conn, record_id, RecordStatus.IN_FLIGHT)

    await handle_subscription_halted(
        conn, event("subscription.halted", status="halted")
    )

    record = await fetch_record(conn, record_id)
    assert record["status"] == "open"


async def test_halted_on_a_scheduled_record_records_rather_than_forces(conn):
    """scheduled -> open is not a legal move, so the event is noted instead."""
    record_id = await handle_subscription_pending(
        conn, event("subscription.pending")
    )
    await transition(conn, record_id, RecordStatus.SCHEDULED)

    await handle_subscription_halted(
        conn, event("subscription.halted", status="halted")
    )

    record = await fetch_record(conn, record_id)
    assert record["status"] == "scheduled"
    assert "halted_without_reopen" in await audit_actions(conn, record_id)


async def test_payment_captured_recovers_an_in_flight_record(conn):
    record_id = await handle_subscription_pending(
        conn, event("subscription.pending")
    )
    await transition(conn, record_id, RecordStatus.SCHEDULED)
    await transition(conn, record_id, RecordStatus.IN_FLIGHT)

    await handle_payment_captured(conn, event("payment.captured"))

    record = await fetch_record(conn, record_id)
    assert record["status"] == "recovered"
    assert "transition:in_flight->recovered" in await audit_actions(conn, record_id)


@pytest.mark.parametrize("starting_status", ["open", "scheduled"])
async def test_payment_captured_recovers_a_record_paid_out_of_band(
    conn, starting_status
):
    """The customer paid before Forbear got to it."""
    record_id = await handle_subscription_pending(
        conn, event("subscription.pending")
    )
    if starting_status == "scheduled":
        await transition(conn, record_id, RecordStatus.SCHEDULED)

    await handle_payment_captured(conn, event("payment.captured"))

    record = await fetch_record(conn, record_id)
    assert record["status"] == "recovered"


async def test_payment_captured_leaves_a_terminal_record_alone(conn):
    record_id = await handle_subscription_pending(
        conn, event("subscription.pending")
    )
    await transition(conn, record_id, RecordStatus.SKIPPED, reason="not_worth_it")

    await handle_payment_captured(conn, event("payment.captured"))

    record = await fetch_record(conn, record_id)
    assert record["status"] == "skipped"


async def test_payment_captured_for_an_unknown_record_is_a_no_op(conn):
    result = await handle_payment_captured(conn, event("payment.captured"))

    assert result is None
    assert await conn.fetchval("SELECT count(*) FROM at_risk_records") == 0


async def test_subscription_charged_recovers_the_same_way(conn):
    record_id = await handle_subscription_pending(
        conn, event("subscription.pending")
    )

    await handle_subscription_charged(conn, event("subscription.charged"))

    record = await fetch_record(conn, record_id)
    assert record["status"] == "recovered"


async def test_payment_failed_creates_a_record_when_none_exists(conn):
    record_id = await handle_payment_failed(
        conn, event("payment.failed", error_code="GATEWAY_ERROR")
    )

    record = await fetch_record(conn, record_id)
    assert record["status"] == "open"
    assert record["failure_class"] == "transient"


async def test_payment_failed_reclassifies_an_existing_record(conn):
    record_id = await handle_subscription_pending(
        conn, event("subscription.pending", error_code="INSUFFICIENT_FUNDS")
    )

    await handle_payment_failed(
        conn, event("payment.failed", error_code="MANDATE_REVOKED")
    )

    record = await fetch_record(conn, record_id)
    assert record["failure_code"] == "MANDATE_REVOKED"
    assert record["failure_class"] == "terminal"
    assert "failure_reclassified" in await audit_actions(conn, record_id)


async def test_payment_failed_reclassifying_to_an_unknown_code_nulls_the_class(
    conn,
):
    record_id = await handle_subscription_pending(
        conn, event("subscription.pending")
    )

    await handle_payment_failed(
        conn, event("payment.failed", error_code="BRAND_NEW_CODE")
    )

    record = await fetch_record(conn, record_id)
    assert record["failure_class"] is None
    assert "unknown_failure_code" in await audit_actions(conn, record_id)


async def test_event_without_a_subscription_id_is_refused(conn):
    broken = {"event": "subscription.pending", "payload": {}}

    with pytest.raises(MalformedPayload):
        await handle_subscription_pending(conn, broken)


async def test_audit_chain_stays_intact_across_a_record_lifecycle(conn):
    record_id = await handle_subscription_pending(
        conn, event("subscription.pending", error_code="MYSTERY_CODE")
    )
    await handle_payment_failed(
        conn, event("payment.failed", error_code="GATEWAY_ERROR")
    )
    await handle_payment_captured(conn, event("payment.captured"))

    assert await audit.verify_chain(conn, ENTITY, record_id) is None
