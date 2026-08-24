"""Webhook endpoint tests.

These go through the real ASGI app against a real database, because the things
worth testing here only exist end to end: that the signature is checked against
the exact bytes received, that a redelivery collides on the unique index, and
that a broken handler still answers 200.
"""

from __future__ import annotations

import hashlib
import hmac
import json

import httpx
import pytest
import pytest_asyncio

from forbear.api.webhooks import (
    EVENT_ID_HEADER,
    SECRET_ENV,
    SIGNATURE_HEADER,
    create_app,
)
from forbear.services import ingestion

pytestmark = pytest.mark.asyncio

SECRET = "test_webhook_secret"
URL = "/webhooks/razorpay"


def sign(body: bytes, secret: str = SECRET) -> str:
    return hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()


def pending_event(*, subscription_id="sub_test_1", invoice_id="inv_test_1"):
    return {
        "entity": "event",
        "event": "subscription.pending",
        "payload": {
            "subscription": {
                "entity": {
                    "id": subscription_id,
                    "customer_id": "cust_test_1",
                    "status": "pending",
                }
            },
            "payment": {
                "entity": {
                    "id": "pay_test_1",
                    "subscription_id": subscription_id,
                    "invoice_id": invoice_id,
                    "amount": 49900,
                    "error_code": "INSUFFICIENT_FUNDS",
                }
            },
        },
    }


@pytest_asyncio.fixture
async def client(clean_db, monkeypatch):
    monkeypatch.setenv(SECRET_ENV, SECRET)
    app = create_app(clean_db)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://forbear.test"
    ) as http_client:
        yield http_client


async def post(client, event, *, event_id="evt_test_1", secret=SECRET, body=None):
    """Sign and send. body overrides what is sent without re-signing it."""
    signed_body = json.dumps(event).encode()
    sent_body = signed_body if body is None else body
    return await client.post(
        URL,
        content=sent_body,
        headers={
            SIGNATURE_HEADER: sign(signed_body, secret),
            EVENT_ID_HEADER: event_id,
            "Content-Type": "application/json",
        },
    )


async def counts(pool):
    async with pool.acquire() as conn:
        return {
            "events": await conn.fetchval("SELECT count(*) FROM webhook_events"),
            "records": await conn.fetchval("SELECT count(*) FROM at_risk_records"),
        }


async def test_valid_signature_stores_event_and_creates_record(client, clean_db):
    response = await post(client, pending_event())

    assert response.status_code == 200
    assert response.json()["status"] == "processed"
    assert await counts(clean_db) == {"events": 1, "records": 1}

    async with clean_db.acquire() as conn:
        stored = await conn.fetchrow("SELECT * FROM webhook_events")
        record = await conn.fetchrow("SELECT * FROM at_risk_records")
    assert stored["event_id"] == "evt_test_1"
    assert stored["event_type"] == "subscription.pending"
    assert json.loads(stored["payload"])["event"] == "subscription.pending"
    assert record["invoice_id"] == "inv_test_1"
    assert record["failure_class"] == "time_dependent"


async def test_invalid_signature_is_rejected_and_stores_nothing(client, clean_db):
    body = json.dumps(pending_event()).encode()
    response = await client.post(
        URL,
        content=body,
        headers={SIGNATURE_HEADER: "0" * 64, EVENT_ID_HEADER: "evt_test_1"},
    )

    assert response.status_code == 401
    assert await counts(clean_db) == {"events": 0, "records": 0}


async def test_missing_signature_header_is_rejected(client, clean_db):
    response = await client.post(URL, content=json.dumps(pending_event()).encode())

    assert response.status_code == 401
    assert await counts(clean_db) == {"events": 0, "records": 0}


async def test_signature_from_a_different_secret_is_rejected(client, clean_db):
    response = await post(client, pending_event(), secret="not_the_secret")

    assert response.status_code == 401
    assert await counts(clean_db) == {"events": 0, "records": 0}


async def test_body_altered_after_signing_is_rejected(client, clean_db):
    """The signature covers the raw bytes, so a one-field edit invalidates it."""
    tampered = json.dumps(pending_event(invoice_id="inv_attacker")).encode()

    response = await post(client, pending_event(), body=tampered)

    assert response.status_code == 401
    assert await counts(clean_db) == {"events": 0, "records": 0}


async def test_duplicate_event_id_is_processed_once(client, clean_db):
    first = await post(client, pending_event())
    second = await post(client, pending_event())

    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json()["status"] == "processed"
    assert second.json()["status"] == "duplicate"
    assert await counts(clean_db) == {"events": 1, "records": 1}


async def test_replay_with_a_new_event_id_still_makes_one_record(client, clean_db):
    """Razorpay reissuing the same invoice under a new event id.

    The unique index on event_id does not catch this one; the unique index on
    invoice_id does.
    """
    await post(client, pending_event(), event_id="evt_test_1")
    await post(client, pending_event(), event_id="evt_test_2")

    assert await counts(clean_db) == {"events": 2, "records": 1}


async def test_unknown_event_type_is_stored_and_ignored(client, clean_db):
    event = {
        "entity": "event",
        "event": "subscription.some_new_thing",
        "payload": {},
    }

    response = await post(client, event)

    assert response.status_code == 200
    assert response.json()["status"] == "ignored"
    assert await counts(clean_db) == {"events": 1, "records": 0}


async def test_handler_failure_still_returns_200_and_keeps_the_payload(
    client, clean_db, monkeypatch
):
    """A 500 would make Razorpay retry into the same broken handler."""

    async def exploding_handler(conn, event):
        raise RuntimeError("handler is broken")

    monkeypatch.setitem(
        ingestion.HANDLERS, "subscription.pending", exploding_handler
    )

    response = await post(client, pending_event())

    assert response.status_code == 200
    assert response.json()["status"] == "error_logged"
    # The event survives for investigation; only the handler's work is undone.
    assert await counts(clean_db) == {"events": 1, "records": 0}


async def test_malformed_payload_in_a_signed_event_does_not_500(client, clean_db):
    """A signed event we cannot act on is logged, not retried."""
    event = {"entity": "event", "event": "subscription.pending", "payload": {}}

    response = await post(client, event)

    assert response.status_code == 200
    assert response.json()["status"] == "error_logged"
    assert await counts(clean_db) == {"events": 1, "records": 0}


async def test_signed_body_that_is_not_json_returns_200_without_storing(
    client, clean_db
):
    body = b"this is not json"
    response = await client.post(
        URL,
        content=body,
        headers={SIGNATURE_HEADER: sign(body), EVENT_ID_HEADER: "evt_test_1"},
    )

    assert response.status_code == 200
    assert response.json()["status"] == "unparseable"
    assert await counts(clean_db) == {"events": 0, "records": 0}


async def test_event_id_falls_back_to_the_body_hash(client, clean_db):
    """No event id header: identical redeliveries must still collide."""
    body = json.dumps(pending_event()).encode()
    headers = {SIGNATURE_HEADER: sign(body), "Content-Type": "application/json"}

    first = await client.post(URL, content=body, headers=headers)
    second = await client.post(URL, content=body, headers=headers)

    assert first.json()["status"] == "processed"
    assert second.json()["status"] == "duplicate"
    assert first.json()["event_id"] == hashlib.sha256(body).hexdigest()
    assert await counts(clean_db) == {"events": 1, "records": 1}


async def test_captured_after_pending_recovers_the_record(client, clean_db):
    """Two events, one record, end to end through the endpoint."""
    await post(client, pending_event(), event_id="evt_pending")

    captured = {
        "entity": "event",
        "event": "payment.captured",
        "payload": {
            "payment": {
                "entity": {
                    "id": "pay_test_2",
                    "subscription_id": "sub_test_1",
                    "invoice_id": "inv_test_1",
                    "amount": 49900,
                }
            }
        },
    }
    response = await post(client, captured, event_id="evt_captured")

    assert response.json()["status"] == "processed"
    async with clean_db.acquire() as conn:
        status = await conn.fetchval(
            "SELECT status FROM at_risk_records WHERE invoice_id = $1", "inv_test_1"
        )
    assert status == "recovered"
