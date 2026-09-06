"""Round-trips the emitter through Forbear's real receiver and ingestion path.

Same rule as test_webhooks.py: real ASGI app, real database. The emitter is
the thing under test here, not the receiver -- if a test fails, the emitter
built or signed something wrong, and the receiver stays exactly as strict as
it is in production.
"""

from __future__ import annotations

import httpx
import pytest
import pytest_asyncio

from forbear.api.webhooks import SECRET_ENV, create_app
from forbear.emitter.emitter import WebhookEmitter, emit_scenario
from forbear.emitter.payload_templates import (
    build_subscription_charged,
    build_subscription_pending,
)
from forbear.emitter.scenarios import (
    batch_scenario,
    duplicate_delivery,
    insufficient_funds_then_halts,
    insufficient_funds_then_recovers,
    revoked_mandate,
    tampered_signature,
)

SECRET = "emitter_test_secret"


@pytest_asyncio.fixture
async def emitter(clean_db, monkeypatch):
    monkeypatch.setenv(SECRET_ENV, SECRET)
    app = create_app(clean_db)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://forbear.test"
    ) as client:
        yield WebhookEmitter(secret=SECRET, client=client)


async def record_status(pool, invoice_id):
    async with pool.acquire() as conn:
        return await conn.fetchval(
            "SELECT status FROM at_risk_records WHERE invoice_id = $1", invoice_id
        )


async def counts(pool):
    async with pool.acquire() as conn:
        return {
            "events": await conn.fetchval("SELECT count(*) FROM webhook_events"),
            "records": await conn.fetchval("SELECT count(*) FROM at_risk_records"),
        }


@pytest.mark.asyncio
async def test_emitted_signature_round_trips_through_the_real_receiver(
    emitter, clean_db
):
    """Sign here, verify there: the whole point of the emitter."""
    payload = build_subscription_pending(
        subscription_id="sub_rt",
        customer_id="cust_rt",
        invoice_id="inv_rt",
        amount=49900,
        error_code="INSUFFICIENT_FUNDS",
    )

    result = await emitter.emit(payload, event_id="evt_rt")

    assert result.status_code == 200
    assert result.response_json["status"] == "processed"
    assert await record_status(clean_db, "inv_rt") == "open"


@pytest.mark.asyncio
async def test_tampered_signature_is_rejected_with_401(emitter, clean_db):
    scenario = tampered_signature(suffix="tamper1")

    [result] = await emit_scenario(emitter, scenario)

    assert result.status_code == 401
    assert await counts(clean_db) == {"events": 0, "records": 0}


@pytest.mark.asyncio
async def test_duplicate_delivery_is_deduped_to_one_record(emitter, clean_db):
    scenario = duplicate_delivery(suffix="dup1")

    results = await emit_scenario(emitter, scenario)

    assert [r.status_code for r in results] == [200, 200]
    assert results[0].response_json["status"] == "processed"
    assert results[1].response_json["status"] == "duplicate"
    assert await counts(clean_db) == {"events": 1, "records": 1}


@pytest.mark.asyncio
async def test_insufficient_funds_then_recovers_ends_recovered(emitter, clean_db):
    scenario = insufficient_funds_then_recovers(suffix="rec1")

    results = await emit_scenario(emitter, scenario)

    assert all(r.status_code == 200 for r in results)
    assert await record_status(clean_db, "inv_rec1") == "recovered"


@pytest.mark.asyncio
async def test_insufficient_funds_then_halts_stays_open_for_forbear(
    emitter, clean_db
):
    scenario = insufficient_funds_then_halts(suffix="halt1")

    results = await emit_scenario(emitter, scenario)

    assert all(r.status_code == 200 for r in results)
    assert await record_status(clean_db, "inv_halt1") == "open"


@pytest.mark.asyncio
async def test_revoked_mandate_is_classified_terminal_and_never_reopened(
    emitter, clean_db
):
    scenario = revoked_mandate(suffix="rev1")

    [result] = await emit_scenario(emitter, scenario)

    assert result.status_code == 200
    async with clean_db.acquire() as conn:
        failure_class = await conn.fetchval(
            "SELECT failure_class FROM at_risk_records WHERE invoice_id = $1",
            "inv_rev1",
        )
    assert failure_class == "terminal"


@pytest.mark.asyncio
async def test_batch_scenario_populates_a_full_worklist(emitter, clean_db):
    scenario = batch_scenario(6)

    results = await emit_scenario(emitter, scenario)

    assert all(r.status_code == 200 for r in results)
    async with clean_db.acquire() as conn:
        records = await conn.fetchval("SELECT count(*) FROM at_risk_records")
    assert records == 6


def test_subscription_pending_matches_razorpays_documented_structure():
    payload = build_subscription_pending(
        subscription_id="sub_shape",
        customer_id="cust_shape",
        invoice_id="inv_shape",
        amount=49900,
        error_code="INSUFFICIENT_FUNDS",
    )

    assert payload["entity"] == "event"
    assert payload["event"] == "subscription.pending"

    payment_entity = payload["payload"]["payment"]["entity"]
    assert payment_entity["id"].startswith("pay_")
    assert payment_entity["error_code"] == "INSUFFICIENT_FUNDS"
    assert "error_description" in payment_entity
    assert "error_reason" in payment_entity

    subscription_entity = payload["payload"]["subscription"]["entity"]
    assert subscription_entity["id"] == "sub_shape"
    assert subscription_entity["customer_id"] == "cust_shape"
    assert subscription_entity["status"] == "pending"


def test_subscription_charged_carries_payment_subscription_and_invoice():
    payload = build_subscription_charged(
        subscription_id="sub_shape2",
        customer_id="cust_shape2",
        invoice_id="inv_shape2",
        amount=49900,
    )

    assert payload["event"] == "subscription.charged"
    assert payload["payload"]["payment"]["entity"]["status"] == "captured"
    assert payload["payload"]["subscription"]["entity"]["status"] == "active"
    assert payload["payload"]["invoice"]["entity"]["status"] == "paid"
