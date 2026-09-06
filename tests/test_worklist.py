"""Worklist endpoint tests.

The endpoint is a read of precomputed decisions. test_worklist_read_never_
scores_or_allocates is the proof of the latency principle: it patches the
allocator and the uplift model so either would raise if called, then confirms
the endpoint still answers correctly.
"""

from __future__ import annotations

from datetime import datetime

import httpx
import pytest
from fastapi import FastAPI

from forbear.api import actions, worklist
from tests.conftest import insert_scenario

pytestmark = pytest.mark.asyncio


def make_app(pool) -> FastAPI:
    app = FastAPI()
    app.state.pool = pool
    app.include_router(worklist.router)
    app.include_router(actions.router)
    return app


@pytest.fixture
def client(clean_db):
    app = make_app(clean_db)
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://forbear.test")


async def _decide(conn, record_id, *, bucket, action="none", reason="because",
                   scheduled_at=None, cost_paise=None, status=None):
    if status is not None:
        await conn.execute(
            """
            UPDATE at_risk_records
            SET status = $2::record_status,
                skip_reason = CASE WHEN $2 = 'skipped' THEN 'seeded_for_test' END
            WHERE id = $1
            """,
            record_id,
            status,
        )
    await conn.execute(
        """
        UPDATE at_risk_records
        SET worklist_bucket = $2, worklist_action = $3, worklist_reason = $4,
            worklist_scheduled_at = $5, worklist_cost_paise = $6,
            worklist_decided_at = now()
        WHERE id = $1
        """,
        record_id,
        bucket,
        action,
        reason,
        datetime.fromisoformat(scheduled_at) if scheduled_at else None,
        cost_paise,
    )


async def test_worklist_buckets_have_the_right_records(clean_db, client):
    async with clean_db.acquire() as conn:
        chase = await insert_scenario(conn, status="open", suffix="chase")
        wait = await insert_scenario(conn, status="open", suffix="wait")
        leave = await insert_scenario(conn, status="open", suffix="leave")

        await _decide(
            conn, chase["record_id"], bucket="chase", action="retry",
            reason="Worth a retry.", scheduled_at="2026-09-10T10:00:00+00:00",
        )
        await _decide(
            conn, wait["record_id"], bucket="wait",
            reason="Likely to self-recover.",
            scheduled_at="2026-09-12T00:00:00+00:00",
        )
        await _decide(
            conn, leave["record_id"], bucket="leave_alone",
            reason="Contacting risks a ₹5,988 subscription.",
            cost_paise=598800,
        )

    async with client as http_client:
        response = await http_client.get("/worklist?date=today")

    assert response.status_code == 200
    body = response.json()

    assert [row["record_id"] for row in body["chase"]] == [chase["record_id"]]
    assert body["chase"][0]["action"] == "retry"

    assert [row["record_id"] for row in body["wait"]] == [wait["record_id"]]

    assert [row["record_id"] for row in body["leave_alone"]] == [leave["record_id"]]
    assert body["leave_alone"][0]["override_cost_rupees"] == 5988


async def test_records_with_no_decision_yet_are_not_shown(clean_db, client):
    async with clean_db.acquire() as conn:
        await insert_scenario(conn, status="open", suffix="undecided")

    async with client as http_client:
        response = await http_client.get("/worklist?date=today")

    body = response.json()
    assert body["chase"] == []
    assert body["wait"] == []
    assert body["leave_alone"] == []


async def test_fear_number_is_present_and_positive(clean_db, client):
    async with clean_db.acquire() as conn:
        chase = await insert_scenario(conn, status="open", suffix="fear_chase")
        leave = await insert_scenario(conn, status="open", suffix="fear_leave")
        await _decide(
            conn, chase["record_id"], bucket="chase", action="send_payment_link",
            reason="Reach out.", scheduled_at="2026-09-10T10:00:00+00:00",
        )
        await _decide(
            conn, leave["record_id"], bucket="leave_alone",
            reason="Contacting risks a subscription.",
            cost_paise=598800,
        )

    async with client as http_client:
        response = await http_client.get("/worklist?date=today")

    fear = response.json()["fear_number"]
    assert fear["at_risk_rupees"] > 0
    assert fear["chase_everything_cost_rupees"] > 0
    assert "at risk" in fear["sentence"]


async def test_worklist_read_never_scores_or_allocates(clean_db, client, monkeypatch):
    def _boom(*args, **kwargs):
        raise AssertionError("worklist read must not call the allocator or a model")

    monkeypatch.setattr("forbear.services.allocator.allocate", _boom)
    monkeypatch.setattr(
        "forbear.scoring.uplift.UpliftModel.__init__",
        lambda self, *a, **k: _boom(),
    )

    async with clean_db.acquire() as conn:
        record = await insert_scenario(conn, status="open", suffix="proof")
        await _decide(
            conn, record["record_id"], bucket="chase", action="retry",
            reason="ok", scheduled_at="2026-09-10T10:00:00+00:00",
        )

    async with client as http_client:
        response = await http_client.get("/worklist?date=today")

    assert response.status_code == 200
    assert len(response.json()["chase"]) == 1
