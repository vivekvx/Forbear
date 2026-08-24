"""Razorpay webhook intake.

Order of operations is the whole design:

  1. Read the raw bytes. Nothing is parsed before the signature is checked,
     because parsing attacker-controlled JSON is work done on an
     unauthenticated request.
  2. Verify HMAC-SHA256 over those exact bytes. Invalid, or absent, is 401.
  3. Insert into webhook_events on the event id. A conflict means Razorpay has
     redelivered something already handled, and the answer is 200 with no work.
  4. Dispatch, and swallow handler failures into a log line.

Step 4 is deliberate. A 500 makes Razorpay retry into the same broken handler,
turning one failure into a stream of them. A logged error against a stored
payload can be replayed by hand once the bug is fixed, which is why the event
row is committed before the handler runs rather than sharing its transaction.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
from typing import Any, Optional

from fastapi import APIRouter, FastAPI, Request
from fastapi.responses import JSONResponse

from forbear.services import ingestion

logger = logging.getLogger(__name__)

router = APIRouter()

SIGNATURE_HEADER = "X-Razorpay-Signature"
EVENT_ID_HEADER = "X-Razorpay-Event-Id"
SECRET_ENV = "RAZORPAY_WEBHOOK_SECRET"


def _secret() -> bytes:
    secret = os.environ.get(SECRET_ENV)
    if not secret:
        # Loud on purpose. A server that cannot verify signatures must not
        # quietly accept, and must not quietly reject either.
        raise RuntimeError(f"{SECRET_ENV} is not set; cannot verify webhooks")
    return secret.encode("utf-8")


def signature_is_valid(raw_body: bytes, signature: Optional[str]) -> bool:
    """HMAC-SHA256 of the raw body, compared in constant time."""
    if not signature:
        return False
    expected = hmac.new(_secret(), raw_body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, signature)


def _event_id(request: Request, event: dict, raw_body: bytes) -> str:
    """Razorpay's event id, or a deterministic stand-in.

    The hash fallback keeps replay detection working when the header is absent:
    an identical redelivery hashes identically and collides.
    """
    header = request.headers.get(EVENT_ID_HEADER)
    if header:
        return header

    payload_id = event.get("id")
    if isinstance(payload_id, str) and payload_id:
        return payload_id

    return hashlib.sha256(raw_body).hexdigest()


@router.post("/webhooks/razorpay")
async def razorpay_webhook(request: Request) -> Any:
    raw_body = await request.body()

    if not signature_is_valid(raw_body, request.headers.get(SIGNATURE_HEADER)):
        logger.warning("rejected webhook with invalid or missing signature")
        return JSONResponse({"status": "invalid_signature"}, status_code=401)

    # Signature checked; only now is the body worth parsing.
    try:
        event = json.loads(raw_body)
    except json.JSONDecodeError:
        logger.exception("signed webhook body is not valid JSON")
        return {"status": "unparseable"}

    if not isinstance(event, dict):
        logger.error("signed webhook body is not a JSON object")
        return {"status": "unparseable"}

    event_id = _event_id(request, event, raw_body)
    event_type = event.get("event") or "unknown"

    pool = request.app.state.pool
    async with pool.acquire() as conn:
        async with conn.transaction():
            stored_id = await conn.fetchval(
                """
                INSERT INTO webhook_events (event_id, event_type, payload)
                VALUES ($1, $2, $3::jsonb)
                ON CONFLICT (event_id) DO NOTHING
                RETURNING id
                """,
                event_id,
                event_type,
                raw_body.decode("utf-8"),
            )

        # No row inserted means the unique index caught a redelivery.
        if stored_id is None:
            return {"status": "duplicate", "event_id": event_id}

        handler = ingestion.HANDLERS.get(event_type)
        if handler is None:
            # Razorpay adds event types without asking. Storing and ignoring
            # one is correct; rejecting it would make them retry forever.
            return {"status": "ignored", "event_id": event_id}

        try:
            async with conn.transaction():
                await handler(conn, event)
        except Exception:
            # The event row is already committed, so the payload survives for
            # investigation and manual replay. Only the handler's work rolls
            # back.
            logger.exception(
                "handler for %s failed on event %s", event_type, event_id
            )
            return {"status": "error_logged", "event_id": event_id}

    return {"status": "processed", "event_id": event_id}


def create_app(pool) -> FastAPI:
    app = FastAPI(title="Forbear")
    app.state.pool = pool
    app.include_router(router)
    return app
