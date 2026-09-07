"""Tests for GET /worklist/protected - the "saved by leaving alone" panel.

A save is a precise thing: left alone (do-not-disturb, not a terminal skip),
would have churned if chased, stays and pays if not. Counting every leave_alone
record as a "save" would be dishonest, so most of this file is about the cases
that must NOT count.
"""

from __future__ import annotations

import httpx
import pytest
from fastapi import FastAPI

from forbear.api import worklist
from forbear.services.allocator import SKIP_NEGATIVE_NET_VALUE, SKIP_TERMINAL_FAILURE_CLASS
from tests.conftest import insert_scenario

pytestmark = pytest.mark.asyncio


def make_app(pool) -> FastAPI:
    app = FastAPI()
    app.state.pool = pool
    app.include_router(worklist.router)
    return app


@pytest.fixture
def client(clean_db):
    app = make_app(clean_db)
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://forbear.test")


async def _leave_alone(
    conn,
    *,
    suffix: str,
    skip_reason: str = SKIP_NEGATIVE_NET_VALUE,
    ltv_paise: int = 598_800,
) -> int:
    scenario = await insert_scenario(conn, status="open", suffix=suffix)
    record_id = scenario["record_id"]
    await conn.execute(
        """
        UPDATE at_risk_records
        SET worklist_bucket = 'leave_alone', worklist_action = 'none',
            worklist_reason = 'because', worklist_cost_paise = $2,
            worklist_skip_reason = $3, worklist_decided_at = now()
        WHERE id = $1
        """,
        record_id,
        ltv_paise,
        skip_reason,
    )
    return record_id


async def _ground_truth(
    conn,
    record_id: int,
    *,
    would_churn_if_contacted: bool,
    would_pay_without_contact: bool,
    ltv_paise: int = 598_800,
) -> None:
    await conn.execute(
        """
        INSERT INTO demo_ground_truth
            (at_risk_record_id, would_churn_if_contacted,
             would_pay_without_contact, remaining_ltv_paise)
        VALUES ($1, $2, $3, $4)
        """,
        record_id,
        would_churn_if_contacted,
        would_pay_without_contact,
        ltv_paise,
    )


async def test_do_not_disturb_save_appears(clean_db, client):
    async with clean_db.acquire() as conn:
        record_id = await _leave_alone(conn, suffix="save")
        await _ground_truth(
            conn,
            record_id,
            would_churn_if_contacted=True,
            would_pay_without_contact=True,
        )

    async with client as http_client:
        response = await http_client.get("/worklist/protected")

    body = response.json()
    assert body["available"] is True
    assert [row["record_id"] for row in body["saves"]] == [record_id]
    assert body["customers_protected"] == 1


async def test_terminal_skip_is_not_a_save(clean_db, client):
    async with clean_db.acquire() as conn:
        record_id = await _leave_alone(
            conn, suffix="terminal", skip_reason=SKIP_TERMINAL_FAILURE_CLASS
        )
        # Even if ground truth says they'd have churned and would pay without
        # contact, a dead mandate could never have been chased in the first
        # place - there is nothing this decision protected.
        await _ground_truth(
            conn,
            record_id,
            would_churn_if_contacted=True,
            would_pay_without_contact=True,
        )

    async with client as http_client:
        response = await http_client.get("/worklist/protected")

    body = response.json()
    assert body["available"] is True
    assert body["saves"] == []
    assert body["customers_protected"] == 0


async def test_leave_alone_that_would_not_churn_is_not_a_save(clean_db, client):
    async with clean_db.acquire() as conn:
        record_id = await _leave_alone(conn, suffix="no_risk")
        # Left alone protected nothing: this customer was never going to
        # churn even if chased, so there is no value the skip preserved.
        await _ground_truth(
            conn,
            record_id,
            would_churn_if_contacted=False,
            would_pay_without_contact=True,
        )

    async with client as http_client:
        response = await http_client.get("/worklist/protected")

    body = response.json()
    assert body["saves"] == []


async def test_leave_alone_that_would_not_have_paid_either_way_is_not_a_save(
    clean_db, client
):
    async with clean_db.acquire() as conn:
        record_id = await _leave_alone(conn, suffix="lost_cause")
        # Would have churned if chased, but would not have paid even if left
        # alone: chasing was never going to buy anything here, so leaving
        # them alone protected nothing.
        await _ground_truth(
            conn,
            record_id,
            would_churn_if_contacted=True,
            would_pay_without_contact=False,
        )

    async with client as http_client:
        response = await http_client.get("/worklist/protected")

    body = response.json()
    assert body["saves"] == []


async def test_total_protected_value_equals_sum_of_remaining_ltv(clean_db, client):
    async with clean_db.acquire() as conn:
        first = await _leave_alone(conn, suffix="a", ltv_paise=598_800)
        await _ground_truth(
            conn,
            first,
            would_churn_if_contacted=True,
            would_pay_without_contact=True,
            ltv_paise=598_800,
        )
        second = await _leave_alone(conn, suffix="b", ltv_paise=1_198_800)
        await _ground_truth(
            conn,
            second,
            would_churn_if_contacted=True,
            would_pay_without_contact=True,
            ltv_paise=1_198_800,
        )

    async with client as http_client:
        response = await http_client.get("/worklist/protected")

    body = response.json()
    assert body["customers_protected"] == 2
    assert body["total_protected_rupees"] == round(598_800 / 100) + round(
        1_198_800 / 100
    )


async def test_real_mode_with_no_ground_truth_returns_demo_only_response(
    clean_db, client
):
    async with clean_db.acquire() as conn:
        # A leave_alone decision with no demo_ground_truth row at all - what
        # every real production record looks like.
        await _leave_alone(conn, suffix="production")

    async with client as http_client:
        response = await http_client.get("/worklist/protected")

    assert response.status_code == 200
    body = response.json()
    assert body["available"] is False
    assert "demo mode" in body["message"].lower()
    assert body["saves"] == []
