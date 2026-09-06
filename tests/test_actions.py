"""Action endpoint tests: state transition, audit trail, idempotency, override."""

from __future__ import annotations

import uuid

import httpx
import pytest
from fastapi import FastAPI

from forbear.api import actions
from tests.conftest import insert_scenario

pytestmark = pytest.mark.asyncio


def make_app(pool) -> FastAPI:
    app = FastAPI()
    app.state.pool = pool
    app.include_router(actions.router)
    return app


@pytest.fixture
def client(clean_db):
    app = make_app(clean_db)
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://forbear.test")


async def _mark_scheduled_chase(conn, record_id):
    await conn.execute(
        """
        UPDATE at_risk_records
        SET worklist_bucket = 'chase', worklist_action = 'send_payment_link'
        WHERE id = $1
        """,
        record_id,
    )


async def _mark_leave_alone(conn, record_id, cost_paise=598800):
    await conn.execute(
        """
        UPDATE at_risk_records
        SET worklist_bucket = 'leave_alone', worklist_cost_paise = $2
        WHERE id = $1
        """,
        record_id,
        cost_paise,
    )


async def test_action_writes_audit_entry(clean_db, client):
    async with clean_db.acquire() as conn:
        scenario = await insert_scenario(conn, status="open", suffix="act1")
        await _mark_scheduled_chase(conn, scenario["record_id"])

    async with client as http_client:
        response = await http_client.post(
            f"/actions/{scenario['record_id']}",
            json={"action": "send_payment_link", "idempotency_key": str(uuid.uuid4())},
        )

    assert response.status_code == 200
    assert response.json()["status"] == "sent"

    async with clean_db.acquire() as conn:
        entries = await conn.fetch(
            "SELECT action FROM audit_log WHERE entity_id = $1",
            str(scenario["record_id"]),
        )
    actions_logged = [row["action"] for row in entries]
    assert "merchant_action:send_payment_link" in actions_logged


async def test_idempotent_repeat_produces_one_effect(clean_db, client):
    async with clean_db.acquire() as conn:
        scenario = await insert_scenario(conn, status="open", suffix="act2")
        await _mark_scheduled_chase(conn, scenario["record_id"])

    key = str(uuid.uuid4())
    async with client as http_client:
        first = await http_client.post(
            f"/actions/{scenario['record_id']}",
            json={"action": "send_payment_link", "idempotency_key": key},
        )
        second = await http_client.post(
            f"/actions/{scenario['record_id']}",
            json={"action": "send_payment_link", "idempotency_key": key},
        )

    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json() == second.json()

    async with clean_db.acquire() as conn:
        action_rows = await conn.fetchval(
            "SELECT count(*) FROM merchant_actions WHERE idempotency_key = $1", key
        )
        audit_rows = await conn.fetchval(
            """
            SELECT count(*) FROM audit_log
            WHERE entity_id = $1 AND action = 'merchant_action:send_payment_link'
            """,
            str(scenario["record_id"]),
        )
    assert action_rows == 1
    assert audit_rows == 1


async def test_override_on_leave_alone_requires_confirmation(clean_db, client):
    async with clean_db.acquire() as conn:
        scenario = await insert_scenario(conn, status="open", suffix="act3")
        await _mark_leave_alone(conn, scenario["record_id"])

    async with client as http_client:
        warned = await http_client.post(
            f"/actions/{scenario['record_id']}",
            json={"action": "send_payment_link", "idempotency_key": str(uuid.uuid4())},
        )

    assert warned.status_code == 200
    body = warned.json()
    assert body["status"] == "confirmation_required"
    assert body["estimated_cost_rupees"] == 5988
    assert "₹" in body["warning"]

    async with clean_db.acquire() as conn:
        status = await conn.fetchval(
            "SELECT status FROM at_risk_records WHERE id = $1", scenario["record_id"]
        )
    assert status == "open"  # unconfirmed: nothing changed


async def test_override_confirmed_reopens_and_logs(clean_db, client):
    async with clean_db.acquire() as conn:
        scenario = await insert_scenario(conn, status="open", suffix="act4")
        await _mark_leave_alone(conn, scenario["record_id"])

    async with client as http_client:
        response = await http_client.post(
            f"/actions/{scenario['record_id']}",
            json={
                "action": "send_payment_link",
                "idempotency_key": str(uuid.uuid4()),
                "confirm": True,
            },
        )

    assert response.status_code == 200
    assert response.json()["status"] == "sent"
    assert response.json()["override"] is True

    async with clean_db.acquire() as conn:
        status = await conn.fetchval(
            "SELECT status FROM at_risk_records WHERE id = $1", scenario["record_id"]
        )
        override_logged = await conn.fetchval(
            """
            SELECT count(*) FROM audit_log
            WHERE entity_id = $1 AND action = 'merchant_override_confirmed'
            """,
            str(scenario["record_id"]),
        )
    assert status == "open"
    assert override_logged == 1
