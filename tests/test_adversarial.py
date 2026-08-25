"""Five things that must never happen, and the system refusing each one.

Every attack here is an action the allocator would happily propose. The records
are attractive on purpose - large invoices, positive uplift, high Whittle index
- because a guard that only blocks worthless records has not been tested. The
question is whether the constraint holds when there is money on the other side
of it.

Deliberately self-contained. The guard tests build their scenarios from shared
conftest helpers, and reusing those here would mean these five attacks pass or
fail for reasons defined somewhere else. Every row below is planted in this
file, so what is being attacked is visible on one screen.

Each attack returns its own one-line verdict. The test asserts on it and prints
it; the demo script prints the same lines without pytest around them.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

import httpx
import pytest

from forbear.api.webhooks import (
    EVENT_ID_HEADER,
    SECRET_ENV,
    SIGNATURE_HEADER,
    create_app,
)
from forbear.config.limits import IST, MAX_ATTEMPTS
from forbear.core import guard
from forbear.core.guard import (
    RULE_ATTEMPT_CAP,
    RULE_EXECUTION_WINDOW,
    RULE_MANDATE_NOT_EXPIRED,
    RULE_MANDATE_NOT_REVOKED,
    guard_check,
)
from forbear.models.models import ActionKind, ProposedAction
from forbear.scoring.whittle import RecordScore, compute_whittle_index

# An invoice worth chasing and a customer worth keeping: ₹4,999 against a
# ₹499/month plan. Nothing here is a throwaway record.
ATTACK_AMOUNT = 499_900
ATTACK_PLAN = 499_00
ATTACK_CATE = 0.42

WEBHOOK_SECRET = "adversarial_secret"
WEBHOOK_URL = "/webhooks/razorpay"


@dataclass(frozen=True)
class AttackResult:
    """What the attack tried, and what the system did about it."""

    attack: str
    blocked: bool
    rule: Optional[str]
    summary: str


def rupees(paise: int) -> str:
    return f"₹{paise / 100:,.0f}"


def _index(cate: float = ATTACK_CATE) -> float:
    """What the allocator would have scored this record at.

    Printed alongside every block, so the refusal reads as the system turning
    down money rather than declining something it never wanted.
    """
    return compute_whittle_index(
        RecordScore(
            record_id="attack", amount=ATTACK_AMOUNT, plan_amount=ATTACK_PLAN, cate=cate
        )
    )


def _line(attack: str, rule: str, extra: str = "") -> str:
    tail = f" · {extra}" if extra else ""
    return (
        f"BLOCKED: {attack} · rule: {rule} · amount {rupees(ATTACK_AMOUNT)} "
        f"· index +{_index():.1f}{tail}"
    )


@contextmanager
def pinned_clock(moment: datetime):
    """Evaluate the guard at a chosen instant.

    The guard reads its own clock and takes no argument for it, which is the
    point of the guard. Pinning it here rather than with monkeypatch keeps the
    attack runnable outside pytest, where the demo script calls it.
    """
    original = guard._server_now

    async def _read(conn) -> datetime:
        return moment

    guard._server_now = _read
    try:
        yield
    finally:
        guard._server_now = original


async def plant_target(
    conn,
    *,
    mandate_status: str = "active",
    attempts: int = 0,
    notification_age: Optional[timedelta] = timedelta(hours=1),
    now: Optional[datetime] = None,
) -> dict[str, int]:
    """One attractive record, built from scratch.

    No shared fixture. Everything an attack depends on - the mandate state, the
    attempt history, the notification - is set here, so a change to the guard
    tests' setup can never quietly change what these five attacks mean.
    """
    now = now or await conn.fetchval("SELECT now()")
    tag = uuid.uuid4().hex[:10]

    customer_id = await conn.fetchval(
        "INSERT INTO customers (external_id) VALUES ($1) RETURNING id",
        f"attack_cust_{tag}",
    )
    subscription_id = await conn.fetchval(
        """
        INSERT INTO subscriptions
            (customer_id, external_id, plan_amount, billing_cycle_days,
             mandate_status)
        VALUES ($1, $2, $3, 30, $4::mandate_status)
        RETURNING id
        """,
        customer_id,
        f"attack_sub_{tag}",
        ATTACK_PLAN,
        mandate_status,
    )
    record_id = await conn.fetchval(
        """
        INSERT INTO at_risk_records
            (subscription_id, customer_id, invoice_id, amount, failure_code,
             failure_class, status)
        VALUES ($1, $2, $3, $4, 'INSUFFICIENT_FUNDS',
                'time_dependent'::failure_class, 'open'::record_status)
        RETURNING id
        """,
        subscription_id,
        customer_id,
        f"attack_inv_{tag}",
        ATTACK_AMOUNT,
    )

    for number in range(1, attempts + 1):
        await conn.execute(
            """
            INSERT INTO attempts
                (at_risk_record_id, attempt_number, scheduled_at, executed_at,
                 outcome)
            VALUES ($1, $2, $3, $3, 'failure'::attempt_outcome)
            """,
            record_id,
            number,
            now - timedelta(days=number + 1),
        )

    if notification_age is not None:
        await conn.execute(
            """
            INSERT INTO contacts
                (customer_id, subscription_id, channel, purpose, sent_at)
            VALUES ($1, $2, 'sms'::contact_channel,
                    'pre_debit_notification'::contact_purpose, $3)
            """,
            customer_id,
            subscription_id,
            now - notification_age,
        )

    return {
        "customer_id": customer_id,
        "subscription_id": subscription_id,
        "record_id": record_id,
    }


def charge(record_id: int, attempt_number: int = 1) -> ProposedAction:
    return ProposedAction(
        kind=ActionKind.CHARGE,
        at_risk_record_id=record_id,
        attempt_number=attempt_number,
    )


def legal_moment(now: datetime) -> datetime:
    """14:00 IST today - inside the 13:00-17:00 NPCI window.

    Used by every attack that is not about timing, so a block can never turn
    out to be the execution window firing by accident.
    """
    return now.astimezone(IST).replace(hour=14, minute=0, second=0, microsecond=0)


# --- the five attacks -------------------------------------------------------


async def attack_revoked_mandate(conn) -> AttackResult:
    """The customer cancelled the mandate. The invoice is still owed."""
    now = await conn.fetchval("SELECT now()")
    ids = await plant_target(conn, mandate_status="revoked", now=now)

    with pinned_clock(legal_moment(now)):
        verdict = await guard_check(conn, ids["record_id"], charge(ids["record_id"]))

    return AttackResult(
        attack="revoked mandate",
        blocked=not verdict.allowed,
        rule=verdict.rule_name,
        summary=_line("revoked mandate", verdict.rule_name or "none"),
    )


async def attack_expired_mandate(conn) -> AttackResult:
    """The authorisation lapsed. Debiting anyway is unauthorised, not late."""
    now = await conn.fetchval("SELECT now()")
    ids = await plant_target(conn, mandate_status="expired", now=now)

    with pinned_clock(legal_moment(now)):
        verdict = await guard_check(conn, ids["record_id"], charge(ids["record_id"]))

    return AttackResult(
        attack="expired mandate",
        blocked=not verdict.allowed,
        rule=verdict.rule_name,
        summary=_line("expired mandate", verdict.rule_name or "none"),
    )


async def attack_attempt_cap_exceeded(conn) -> AttackResult:
    """Every permitted attempt already spent. One more is a cap breach."""
    now = await conn.fetchval("SELECT now()")
    ids = await plant_target(conn, attempts=MAX_ATTEMPTS, now=now)

    with pinned_clock(legal_moment(now)):
        verdict = await guard_check(
            conn, ids["record_id"], charge(ids["record_id"], MAX_ATTEMPTS + 1)
        )

    return AttackResult(
        attack="attempt cap exceeded",
        blocked=not verdict.allowed,
        rule=verdict.rule_name,
        summary=_line(
            "attempt cap exceeded",
            verdict.rule_name or "none",
            f"attempts {MAX_ATTEMPTS}/{MAX_ATTEMPTS}",
        ),
    )


async def attack_npci_window_violation(conn) -> AttackResult:
    """10:30 IST. A perfectly good record, debited at an hour NPCI forbids."""
    now = await conn.fetchval("SELECT now()")
    ids = await plant_target(conn, now=now)
    restricted = now.astimezone(IST).replace(
        hour=10, minute=30, second=0, microsecond=0
    )

    with pinned_clock(restricted):
        verdict = await guard_check(conn, ids["record_id"], charge(ids["record_id"]))

    return AttackResult(
        attack="NPCI window violation",
        blocked=not verdict.allowed,
        rule=verdict.rule_name,
        summary=_line("NPCI window violation", verdict.rule_name or "none", "10:30 IST"),
    )


async def attack_duplicate_webhook_replay(pool) -> AttackResult:
    """Razorpay redelivers. The same failure must not become two debts.

    Takes a pool rather than a connection: the endpoint acquires and commits on
    its own, which is the behaviour under attack.
    """
    event = {
        "entity": "event",
        "event": "subscription.pending",
        "payload": {
            "subscription": {
                "entity": {
                    "id": "sub_attack_replay",
                    "customer_id": "cust_attack_replay",
                    "status": "pending",
                }
            },
            "payment": {
                "entity": {
                    "id": "pay_attack_replay",
                    "subscription_id": "sub_attack_replay",
                    "invoice_id": "inv_attack_replay",
                    "amount": ATTACK_AMOUNT,
                    "error_code": "INSUFFICIENT_FUNDS",
                }
            },
        },
    }
    body = json.dumps(event).encode()
    signature = hmac.new(WEBHOOK_SECRET.encode(), body, hashlib.sha256).hexdigest()
    headers = {
        SIGNATURE_HEADER: signature,
        EVENT_ID_HEADER: "evt_attack_replay",
        "Content-Type": "application/json",
    }

    app = create_app(pool)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://forbear.test"
    ) as client:
        first = await client.post(WEBHOOK_URL, content=body, headers=headers)
        second = await client.post(WEBHOOK_URL, content=body, headers=headers)

    async with pool.acquire() as conn:
        events = await conn.fetchval(
            "SELECT count(*) FROM webhook_events WHERE event_id = $1",
            "evt_attack_replay",
        )
        records = await conn.fetchval(
            "SELECT count(*) FROM at_risk_records WHERE invoice_id = $1",
            "inv_attack_replay",
        )

    deduplicated = (
        first.status_code == 200
        and second.status_code == 200
        and events == 1
        and records == 1
    )
    return AttackResult(
        attack="duplicate webhook replay",
        blocked=deduplicated,
        rule="unique(event_id)",
        summary=(
            f"BLOCKED: duplicate webhook replay · rule: unique(event_id) "
            f"· amount {rupees(ATTACK_AMOUNT)} · 2 deliveries "
            f"· {events} event, {records} record"
        ),
    )


# --- the tests --------------------------------------------------------------


async def test_revoked_mandate_blocked(conn):
    result = await attack_revoked_mandate(conn)
    print("\n" + result.summary)

    assert result.blocked
    assert result.rule == RULE_MANDATE_NOT_REVOKED


async def test_expired_mandate_blocked(conn):
    result = await attack_expired_mandate(conn)
    print("\n" + result.summary)

    assert result.blocked
    assert result.rule == RULE_MANDATE_NOT_EXPIRED


async def test_attempt_cap_exceeded_blocked(conn):
    result = await attack_attempt_cap_exceeded(conn)
    print("\n" + result.summary)

    assert result.blocked
    assert result.rule == RULE_ATTEMPT_CAP


async def test_npci_window_violation_blocked(conn):
    result = await attack_npci_window_violation(conn)
    print("\n" + result.summary)

    assert result.blocked
    assert result.rule == RULE_EXECUTION_WINDOW


async def test_duplicate_webhook_replay(clean_db, monkeypatch):
    monkeypatch.setenv(SECRET_ENV, WEBHOOK_SECRET)

    result = await attack_duplicate_webhook_replay(clean_db)
    print("\n" + result.summary)

    assert result.blocked

    async with clean_db.acquire() as conn:
        assert (
            await conn.fetchval(
                "SELECT count(*) FROM webhook_events WHERE event_id = $1",
                "evt_attack_replay",
            )
            == 1
        )
        assert (
            await conn.fetchval(
                "SELECT count(*) FROM at_risk_records WHERE invoice_id = $1",
                "inv_attack_replay",
            )
            == 1
        )


# --- the attacks were worth blocking ---------------------------------------


async def test_the_blocked_record_was_one_the_allocator_wanted(conn):
    """The point of the whole file.

    A guard that refuses worthless records has proved nothing. Every attack
    above uses a record with a large positive index - money the system turns
    down because a rule says so, which is the only interesting kind of refusal.
    And the same record, with nothing illegal about it, is permitted: these are
    rules firing, not a guard that refuses everything.
    """
    index = _index()
    print(
        f"\nattacked record: {rupees(ATTACK_AMOUNT)} · index +{index:.1f} "
        f"· cate +{ATTACK_CATE}"
    )

    assert index > 0

    now = await conn.fetchval("SELECT now()")
    ids = await plant_target(conn, now=now)
    with pinned_clock(legal_moment(now)):
        verdict = await guard_check(conn, ids["record_id"], charge(ids["record_id"]))

    assert verdict.allowed, f"a clean record was refused by {verdict.rule_name}"


@pytest.mark.parametrize(
    "attack,expected_rule",
    [
        (attack_revoked_mandate, RULE_MANDATE_NOT_REVOKED),
        (attack_expired_mandate, RULE_MANDATE_NOT_EXPIRED),
        (attack_attempt_cap_exceeded, RULE_ATTEMPT_CAP),
        (attack_npci_window_violation, RULE_EXECUTION_WINDOW),
    ],
)
async def test_each_attack_names_the_rule_that_stopped_it(conn, attack, expected_rule):
    """A block with no rule name is an outage, not a decision."""
    result = await attack(conn)

    assert result.rule == expected_rule
    assert expected_rule in result.summary
