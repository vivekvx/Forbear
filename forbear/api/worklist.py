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

from forbear.services.allocator import SKIP_NEGATIVE_NET_VALUE

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


DEMO_ONLY_MESSAGE = (
    "Available in demo mode only. This view needs outcome data - what would "
    "have happened if a customer had been contacted - that only exists in "
    "Forbear's simulator, never in real production data."
)


async def _protected_saves(conn) -> list[dict[str, Any]]:
    """The precise intersection: left alone, would have churned, would have paid.

    Reads only rows already decided by forbear.services.decisioning (leave_alone,
    the do-not-disturb reason specifically - not a terminal skip, which protects
    nobody because nothing could have been done regardless) joined against the
    demo-only ground truth. No scoring or allocation happens here.
    """
    rows = await conn.fetch(
        """
        SELECT r.id, c.external_id AS customer_name, r.amount,
               g.remaining_ltv_paise
        FROM at_risk_records r
        JOIN customers c ON c.id = r.customer_id
        JOIN demo_ground_truth g ON g.at_risk_record_id = r.id
        WHERE r.status = 'open'
          AND r.worklist_bucket = 'leave_alone'
          AND r.worklist_skip_reason = $1
          AND g.would_churn_if_contacted = TRUE
          AND g.would_pay_without_contact = TRUE
        ORDER BY g.remaining_ltv_paise DESC
        """,
        SKIP_NEGATIVE_NET_VALUE,
    )
    return [
        {
            "record_id": row["id"],
            "customer_name": row["customer_name"],
            "value_protected_rupees": _rupees(row["remaining_ltv_paise"]),
            "invoice_amount_rupees": _rupees(row["amount"]),
            "reason": (
                f"Would have cancelled a ₹{_rupees(row['remaining_ltv_paise']):,}"
                "/year subscription if chased."
            ),
        }
        for row in rows
    ]


@router.get("/worklist/protected")
async def get_protected(request: Request) -> dict[str, Any]:
    pool = request.app.state.pool
    async with pool.acquire() as conn:
        # Ground truth exists only when this database has been seeded by the
        # demo path (forbear.services.demo_seed). A real webhook never writes
        # demo_ground_truth, so an empty table is production, not an empty
        # demo run - the two are not distinguishable any other way, and that
        # is the point: no flag to get out of sync, just data that either
        # exists or doesn't.
        has_ground_truth = await conn.fetchval(
            "SELECT EXISTS (SELECT 1 FROM demo_ground_truth)"
        )
        if not has_ground_truth:
            return {
                "available": False,
                "message": DEMO_ONLY_MESSAGE,
                "customers_protected": 0,
                "total_protected_rupees": 0,
                "saves": [],
            }

        saves = await _protected_saves(conn)

    total_protected_rupees = sum(row["value_protected_rupees"] for row in saves)
    chase_everything_recovered_rupees = sum(
        row["invoice_amount_rupees"] for row in saves
    )

    return {
        "available": True,
        "customers_protected": len(saves),
        "total_protected_rupees": total_protected_rupees,
        "chase_everything_recovered_rupees": chase_everything_recovered_rupees,
        "sentence": (
            f"You protected ₹{total_protected_rupees:,} in subscriptions by "
            f"leaving {len(saves)} customer"
            f"{'s' if len(saves) != 1 else ''} alone. A chase-everything "
            f"approach would have recovered ₹{chase_everything_recovered_rupees:,} "
            f"from them and lost ₹{total_protected_rupees:,}."
        ),
        "saves": saves,
    }
