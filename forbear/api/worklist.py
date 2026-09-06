"""The merchant's daily to-do list: a read of decisions made ahead of time.

Everything here was decided in forbear.services.decisioning, triggered from
the webhook path, well before anyone opens this screen. This module runs no
model and no allocator - it is SQL and arithmetic only, which is what keeps it
under the sub-200ms budget the worklist promises. A record with no decision
yet (the background job has not reached it) simply does not appear; there is
no code path here that computes one on demand.
"""

from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Query, Request

router = APIRouter()


def _rupees(paise: Optional[int]) -> int:
    return round((paise or 0) / 100)


async def _chase_rows(conn) -> list[dict[str, Any]]:
    rows = await conn.fetch(
        """
        SELECT r.id, c.external_id AS customer_name, r.amount,
               r.worklist_action, r.worklist_reason, r.worklist_scheduled_at
        FROM at_risk_records r
        JOIN customers c ON c.id = r.customer_id
        WHERE r.status = 'open' AND r.worklist_bucket = 'chase'
        ORDER BY r.worklist_scheduled_at ASC
        """
    )
    return [
        {
            "record_id": row["id"],
            "customer_name": row["customer_name"],
            "amount_rupees": _rupees(row["amount"]),
            "action": row["worklist_action"],
            "reason": row["worklist_reason"],
            "scheduled_at": row["worklist_scheduled_at"],
        }
        for row in rows
    ]


async def _wait_rows(conn) -> list[dict[str, Any]]:
    rows = await conn.fetch(
        """
        SELECT r.id, c.external_id AS customer_name, r.amount,
               r.worklist_reason, r.worklist_scheduled_at
        FROM at_risk_records r
        JOIN customers c ON c.id = r.customer_id
        WHERE r.status = 'open' AND r.worklist_bucket = 'wait'
        ORDER BY r.worklist_scheduled_at ASC
        """
    )
    return [
        {
            "record_id": row["id"],
            "customer_name": row["customer_name"],
            "amount_rupees": _rupees(row["amount"]),
            "reason": row["worklist_reason"],
            "expected_at": row["worklist_scheduled_at"],
        }
        for row in rows
    ]


async def _leave_alone_rows(conn) -> list[dict[str, Any]]:
    rows = await conn.fetch(
        """
        SELECT r.id, c.external_id AS customer_name, r.amount,
               r.worklist_reason, r.worklist_cost_paise
        FROM at_risk_records r
        JOIN customers c ON c.id = r.customer_id
        WHERE r.status = 'open' AND r.worklist_bucket = 'leave_alone'
        ORDER BY r.worklist_cost_paise DESC NULLS LAST
        """
    )
    return [
        {
            "record_id": row["id"],
            "customer_name": row["customer_name"],
            "amount_rupees": _rupees(row["amount"]),
            "reason": row["worklist_reason"],
            "override_cost_rupees": _rupees(row["worklist_cost_paise"]),
        }
        for row in rows
    ]


async def _fear_number(conn, chase: list, wait: list, leave_alone: list) -> dict[str, Any]:
    at_risk_rupees = sum(row["amount_rupees"] for row in chase + wait)
    chase_everything_cost_rupees = sum(
        row["override_cost_rupees"] for row in leave_alone
    )
    return {
        "at_risk_rupees": at_risk_rupees,
        "chase_everything_cost_rupees": chase_everything_cost_rupees,
        "sentence": (
            f"You have ₹{at_risk_rupees:,} at risk. Chasing all of it the usual "
            f"way would cost you ₹{chase_everything_cost_rupees:,} in churned "
            f"customers. Here's what to do instead."
        ),
    }


@router.get("/worklist")
async def get_worklist(request: Request, date: str = Query("today")) -> dict[str, Any]:
    if date != "today":
        raise HTTPException(400, "only date=today is supported")

    pool = request.app.state.pool
    async with pool.acquire() as conn:
        chase = await _chase_rows(conn)
        wait = await _wait_rows(conn)
        leave_alone = await _leave_alone_rows(conn)
        fear_number = await _fear_number(conn, chase, wait, leave_alone)

    return {
        "fear_number": fear_number,
        "chase": chase,
        "wait": wait,
        "leave_alone": leave_alone,
    }
