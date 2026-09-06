"""Sends signed, Razorpay-shaped webhook events at Forbear's real receiver.

The one rule that matters: raw_body is serialized exactly once, signed, and
sent unchanged. A second json.dumps() of the same dict is not guaranteed to
produce identical bytes, so signing one serialization and sending another is
the most common way to build a webhook sender whose signatures the receiver
silently rejects. Every path here threads the same bytes object through
build -> sign -> send.

The emitter takes an httpx.AsyncClient rather than owning one, so the same
code drives both a real running server (client base_url pointed at it) and
Forbear's real ASGI app in-process for tests (client built on
httpx.ASGITransport) -- the receiver under test is identical either way.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Optional

import httpx

from forbear.emitter.signer import sign_payload

SIGNATURE_HEADER = "X-Razorpay-Signature"
EVENT_ID_HEADER = "X-Razorpay-Event-Id"
WEBHOOK_PATH = "/webhooks/razorpay"


def serialize(payload: dict) -> bytes:
    """The one and only serialization of a payload. Call this once per event."""
    return json.dumps(payload).encode("utf-8")


@dataclass(frozen=True)
class EmittedEvent:
    """One send: the exact bytes sent, the signature sent, and the response."""

    event_type: str
    event_id: str
    raw_body: bytes
    signature: str
    status_code: int
    response_json: Any


@dataclass(frozen=True)
class ScenarioStep:
    """One event in a scenario. Overrides exist only to build negative tests."""

    payload: dict
    event_id: str
    signature_override: Optional[str] = None
    raw_body_override: Optional[bytes] = None


@dataclass(frozen=True)
class Scenario:
    name: str
    steps: list


class WebhookEmitter:
    """Signs and sends payloads at one Forbear webhook endpoint."""

    def __init__(self, *, secret: str, client: httpx.AsyncClient):
        self.secret = secret
        self.client = client

    async def emit(
        self,
        payload: dict,
        *,
        event_id: str,
        signature_override: Optional[str] = None,
        raw_body_override: Optional[bytes] = None,
    ) -> EmittedEvent:
        """Build once, sign once, send unchanged.

        signature_override / raw_body_override exist only for the
        tampered_signature scenario; every other caller signs exactly the
        bytes it sends.
        """
        raw_body = (
            raw_body_override if raw_body_override is not None else serialize(payload)
        )
        signature = (
            signature_override
            if signature_override is not None
            else sign_payload(raw_body, self.secret)
        )

        response = await self.client.post(
            WEBHOOK_PATH,
            content=raw_body,
            headers={
                SIGNATURE_HEADER: signature,
                EVENT_ID_HEADER: event_id,
                "Content-Type": "application/json",
            },
        )
        try:
            response_json = response.json()
        except ValueError:
            response_json = None

        return EmittedEvent(
            event_type=payload.get("event", "unknown"),
            event_id=event_id,
            raw_body=raw_body,
            signature=signature,
            status_code=response.status_code,
            response_json=response_json,
        )


async def emit_scenario(emitter: WebhookEmitter, scenario: Scenario) -> list:
    """Send a scenario's steps in order, over the same emitter."""
    results = []
    for step in scenario.steps:
        results.append(
            await emitter.emit(
                step.payload,
                event_id=step.event_id,
                signature_override=step.signature_override,
                raw_body_override=step.raw_body_override,
            )
        )
    return results
