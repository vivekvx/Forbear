"""What happens when a merchant taps a button on the worklist.

Two things make this module careful rather than clever:

  * Idempotency. A merchant double-taps because the first tap felt slow, and
    the customer must receive exactly one payment link. Every request carries
    an idempotency key; an advisory transaction lock on that key (the same
    pattern forbear.core.audit uses for the audit chain) serialises concurrent
    requests so a race can't send twice before either has recorded that it
    sent once.
  * The override path. A leave_alone record was skipped as a decision, not an
    oversight, so acting on it needs a second, explicit confirmation, and the
    warning has to state what the override is estimated to cost.

Nothing here touches the charge path. Sending a payment link or a retry
nudge is a contact, not a debit; the guard and the executor's attempt-cap
machinery are for the allocator's charge attempts, and this module never
schedules one. For the demo, "sending" is a stub - it stands in for the real
emitter/executor call a production build would make.
"""

from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from forbear.core.audit import append_entry
from forbear.core.state_machine import ENTITY_TYPE

router = APIRouter()

VALID_ACTIONS = frozenset({"send_payment_link", "retry", "update_card"})


class ActionRequest(BaseModel):
    action: str
    idempotency_key: str
    confirm: bool = False


def _send(action: str, record_id: int) -> dict[str, Any]:
    """Stand-in for the real outbound call. Always succeeds, for the demo."""
    return {"provider": "stub", "action": action, "record_id": record_id}


async def _lock_idempotency_key(conn, key: str) -> None:
    await conn.execute("SELECT pg_advisory_xact_lock(hashtext($1))", f"forbear.action.{key}")


@router.post("/actions/{record_id}")
async def post_action(record_id: int, request: Request, body: ActionRequest) -> dict[str, Any]:
    if body.action not in VALID_ACTIONS:
        raise HTTPException(400, f"unknown action {body.action!r}")

    pool = request.app.state.pool
    async with pool.acquire() as conn:
        async with conn.transaction():
            await _lock_idempotency_key(conn, body.idempotency_key)

            existing = await conn.fetchrow(
                "SELECT result FROM merchant_actions WHERE idempotency_key = $1",
                body.idempotency_key,
            )
            if existing is not None:
                return json.loads(existing["result"])

            record = await conn.fetchrow(
                """
                SELECT id, status, worklist_bucket, worklist_cost_paise
                FROM at_risk_records
                WHERE id = $1
                FOR UPDATE
                """,
                record_id,
            )
            if record is None:
                raise HTTPException(404, f"no such record {record_id}")

            is_leave_alone = record["worklist_bucket"] == "leave_alone"

            if is_leave_alone and not body.confirm:
                cost_rupees = round((record["worklist_cost_paise"] or 0) / 100)
                # Not persisted: nothing irreversible has happened, so a
                # repeated warning request needs no idempotency guard of its
                # own - it's a read in every sense that matters.
                return {
                    "status": "confirmation_required",
                    "warning": (
                        f"This will likely cost you ₹{cost_rupees:,} in churn "
                        f"risk. Confirm to proceed."
                    ),
                    "estimated_cost_rupees": cost_rupees,
                }

            if is_leave_alone and body.confirm:
                # The record was never actually transitioned to a terminal
                # status by this preview (see decisioning.py: the worklist is
                # a preview, not the real allocation cycle), so there is
                # nothing to reopen in the state machine - only the override
                # itself needs to be on the record.
                await append_entry(
                    conn,
                    ENTITY_TYPE,
                    record_id,
                    "merchant_override_confirmed",
                    {
                        "action": body.action,
                        "estimated_cost_paise": record["worklist_cost_paise"],
                        "idempotency_key": body.idempotency_key,
                    },
                )

            outbound = _send(body.action, record_id)

            await append_entry(
                conn,
                ENTITY_TYPE,
                record_id,
                f"merchant_action:{body.action}",
                {
                    "idempotency_key": body.idempotency_key,
                    "override": is_leave_alone,
                    "outbound": outbound,
                },
            )

            result = {
                "status": "sent",
                "action": body.action,
                "record_id": record_id,
                "override": is_leave_alone,
            }

            await conn.execute(
                """
                INSERT INTO merchant_actions
                    (idempotency_key, at_risk_record_id, action_type,
                     is_override, result)
                VALUES ($1, $2, $3, $4, $5::jsonb)
                """,
                body.idempotency_key,
                record_id,
                body.action,
                is_leave_alone,
                json.dumps(result),
            )

            return result
