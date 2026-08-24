"""Guard tests.

Each rule gets a scenario that violates exactly that rule, so a blocked verdict
proves the rule fired and not something upstream of it. The clock is pinned for
every test, because a guard whose answer depends on when the suite runs is not
testable, and the execution-window rule would make it so.
"""

from __future__ import annotations

import ast
import json
import pathlib
from datetime import datetime, timedelta, timezone

import pytest

from forbear.config.limits import IST, MAX_ATTEMPTS
from forbear.core import guard
from forbear.core.guard import (
    RULE_ATTEMPT_CAP,
    RULE_COOLDOWN,
    RULE_DUPLICATE_ATTEMPT,
    RULE_EXECUTION_WINDOW,
    RULE_MANDATE_NOT_EXPIRED,
    RULE_MANDATE_NOT_REVOKED,
    RULE_NOTIFICATION,
    RULE_RECORD_EXISTS,
    guard_check,
)
from forbear.models.models import (
    ActionKind,
    AtRiskRecord,
    FailureClass,
    ProposedAction,
    RecordStatus,
)
from tests.conftest import insert_attempt, insert_notification, insert_scenario

# No module-level asyncio mark: pytest.ini runs in auto mode, and marking the
# two synchronous tests below as asyncio would warn.
REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent

# 14:00 IST sits inside the 13:00-17:00 window, so every scenario that is not
# specifically about timing is evaluated from here.
DEFAULT_HOUR_IST = 14


def at_ist(hour: int, minute: int = 0) -> datetime:
    """The UTC instant whose IST wall clock is hour:minute on a fixed date."""
    return datetime(2026, 8, 24, hour, minute, tzinfo=IST).astimezone(timezone.utc)


def freeze(monkeypatch, moment: datetime) -> datetime:
    """Pin the guard's clock. Every rule then evaluates against one instant."""

    async def _frozen(conn):
        return moment

    monkeypatch.setattr(guard, "_server_now", _frozen)
    return moment


def charge(record_id: int, attempt_number: int = 2) -> ProposedAction:
    return ProposedAction(
        kind=ActionKind.CHARGE,
        at_risk_record_id=record_id,
        attempt_number=attempt_number,
    )


async def build_valid_scenario(conn, now: datetime, *, mandate_status="active"):
    """A record that passes every check, so each test can break exactly one.

    One attempt executed 48h ago: inside the cap, outside the cooldown, and a
    real prior-attempt history rather than the empty case.
    """
    ids = await insert_scenario(conn, mandate_status=mandate_status)
    await insert_attempt(
        conn,
        ids["record_id"],
        1,
        outcome="failure",
        executed_at=now - timedelta(hours=48),
    )
    await insert_notification(
        conn,
        ids["customer_id"],
        ids["subscription_id"],
        now - timedelta(hours=1),
    )
    return ids


async def test_valid_record_passes_every_check(conn, monkeypatch):
    now = freeze(monkeypatch, at_ist(DEFAULT_HOUR_IST))
    ids = await build_valid_scenario(conn, now)

    verdict = await guard_check(conn, ids["record_id"], charge(ids["record_id"]))

    assert verdict.allowed is True
    assert verdict.rule_name is None
    assert verdict.details["checks_passed"] == [name for name, _ in guard.CHECKS]


async def test_blocked_when_mandate_revoked(conn, monkeypatch):
    now = freeze(monkeypatch, at_ist(DEFAULT_HOUR_IST))
    ids = await build_valid_scenario(conn, now, mandate_status="revoked")

    verdict = await guard_check(conn, ids["record_id"], charge(ids["record_id"]))

    assert verdict.allowed is False
    assert verdict.rule_name == RULE_MANDATE_NOT_REVOKED
    assert verdict.details["mandate_status"] == "revoked"


async def test_blocked_when_mandate_expired(conn, monkeypatch):
    now = freeze(monkeypatch, at_ist(DEFAULT_HOUR_IST))
    ids = await build_valid_scenario(conn, now, mandate_status="expired")

    verdict = await guard_check(conn, ids["record_id"], charge(ids["record_id"]))

    assert verdict.allowed is False
    assert verdict.rule_name == RULE_MANDATE_NOT_EXPIRED
    assert verdict.details["mandate_status"] == "expired"


@pytest.mark.parametrize("mandate_status", ["active", "paused"])
async def test_other_mandate_states_pass_the_mandate_rules(
    conn, monkeypatch, mandate_status
):
    now = freeze(monkeypatch, at_ist(DEFAULT_HOUR_IST))
    ids = await build_valid_scenario(conn, now, mandate_status=mandate_status)

    verdict = await guard_check(conn, ids["record_id"], charge(ids["record_id"]))

    assert verdict.allowed is True


async def test_blocked_when_attempt_cap_already_consumed(conn, monkeypatch):
    now = freeze(monkeypatch, at_ist(DEFAULT_HOUR_IST))
    ids = await build_valid_scenario(conn, now)
    for number in range(2, MAX_ATTEMPTS + 1):
        await insert_attempt(
            conn,
            ids["record_id"],
            number,
            outcome="failure",
            executed_at=now - timedelta(hours=48),
        )

    verdict = await guard_check(
        conn, ids["record_id"], charge(ids["record_id"], attempt_number=MAX_ATTEMPTS)
    )

    assert verdict.allowed is False
    assert verdict.rule_name == RULE_ATTEMPT_CAP
    assert verdict.details["attempts_used"] == MAX_ATTEMPTS
    assert verdict.details["reason"] == "cap_already_consumed"


async def test_blocked_when_allocator_proposes_an_attempt_past_the_cap(
    conn, monkeypatch
):
    """The allocator bug: a record with attempts to spare, but a bad number.

    Counting rows alone would wave this through, and the attempt would consume
    a slot the cap does not have.
    """
    now = freeze(monkeypatch, at_ist(DEFAULT_HOUR_IST))
    ids = await build_valid_scenario(conn, now)

    verdict = await guard_check(
        conn,
        ids["record_id"],
        charge(ids["record_id"], attempt_number=MAX_ATTEMPTS + 5),
    )

    assert verdict.allowed is False
    assert verdict.rule_name == RULE_ATTEMPT_CAP
    assert verdict.details["attempts_used"] < MAX_ATTEMPTS
    assert verdict.details["reason"] == "proposed_attempt_number_above_cap"


async def test_blocked_when_cooldown_has_not_elapsed(conn, monkeypatch):
    now = freeze(monkeypatch, at_ist(DEFAULT_HOUR_IST))
    ids = await insert_scenario(conn)
    await insert_attempt(
        conn,
        ids["record_id"],
        1,
        outcome="failure",
        executed_at=now - timedelta(hours=1),
    )
    await insert_notification(
        conn, ids["customer_id"], ids["subscription_id"], now - timedelta(hours=1)
    )

    verdict = await guard_check(conn, ids["record_id"], charge(ids["record_id"]))

    assert verdict.allowed is False
    assert verdict.rule_name == RULE_COOLDOWN
    assert verdict.details["server_now"] == now.isoformat()


async def test_cooldown_passes_with_no_prior_attempt(conn, monkeypatch):
    now = freeze(monkeypatch, at_ist(DEFAULT_HOUR_IST))
    ids = await insert_scenario(conn)
    await insert_notification(
        conn, ids["customer_id"], ids["subscription_id"], now - timedelta(hours=1)
    )

    verdict = await guard_check(
        conn, ids["record_id"], charge(ids["record_id"], attempt_number=1)
    )

    assert verdict.allowed is True


@pytest.mark.parametrize(
    ("hour", "minute", "allowed"),
    [
        (0, 0, True),      # window opens at midnight
        (9, 59, True),
        (10, 0, False),    # 10:00 is the exclusive end of the morning window
        (10, 30, False),   # the hour named in the spec
        (12, 59, False),
        (13, 0, True),     # afternoon window opens
        (14, 0, True),
        (16, 59, True),
        (17, 0, False),    # and closes
        (21, 29, False),
        (21, 30, True),    # evening window opens
        (23, 59, True),
    ],
)
async def test_execution_window(conn, monkeypatch, hour, minute, allowed):
    now = freeze(monkeypatch, at_ist(hour, minute))
    ids = await build_valid_scenario(conn, now)

    verdict = await guard_check(conn, ids["record_id"], charge(ids["record_id"]))

    assert verdict.allowed is allowed
    if not allowed:
        assert verdict.rule_name == RULE_EXECUTION_WINDOW
        assert verdict.details["ist_time"] == f"{hour:02d}:{minute:02d}"


async def test_execution_window_uses_ist_not_utc(conn, monkeypatch):
    """05:00 UTC is 10:30 IST: legal by the server's clock, illegal by NPCI's."""
    now = freeze(monkeypatch, datetime(2026, 8, 24, 5, 0, tzinfo=timezone.utc))
    ids = await build_valid_scenario(conn, now)

    verdict = await guard_check(conn, ids["record_id"], charge(ids["record_id"]))

    assert verdict.allowed is False
    assert verdict.rule_name == RULE_EXECUTION_WINDOW
    assert verdict.details["ist_time"] == "10:30"


async def test_blocked_when_no_notification_exists(conn, monkeypatch):
    now = freeze(monkeypatch, at_ist(DEFAULT_HOUR_IST))
    ids = await insert_scenario(conn)
    await insert_attempt(
        conn,
        ids["record_id"],
        1,
        outcome="failure",
        executed_at=now - timedelta(hours=48),
    )

    verdict = await guard_check(conn, ids["record_id"], charge(ids["record_id"]))

    assert verdict.allowed is False
    assert verdict.rule_name == RULE_NOTIFICATION
    assert verdict.details["reason"] == "no_notification_found"


async def test_blocked_when_notification_is_older_than_24h(conn, monkeypatch):
    now = freeze(monkeypatch, at_ist(DEFAULT_HOUR_IST))
    ids = await insert_scenario(conn)
    await insert_notification(
        conn,
        ids["customer_id"],
        ids["subscription_id"],
        now - timedelta(hours=24, minutes=1),
    )

    verdict = await guard_check(
        conn, ids["record_id"], charge(ids["record_id"], attempt_number=1)
    )

    assert verdict.allowed is False
    assert verdict.rule_name == RULE_NOTIFICATION
    assert verdict.details["reason"] == "notification_expired"


async def test_notification_exactly_24h_old_still_counts(conn, monkeypatch):
    """The rule is sent_at >= now - 24h, so the boundary is inclusive."""
    now = freeze(monkeypatch, at_ist(DEFAULT_HOUR_IST))
    ids = await insert_scenario(conn)
    await insert_notification(
        conn, ids["customer_id"], ids["subscription_id"], now - timedelta(hours=24)
    )

    verdict = await guard_check(
        conn, ids["record_id"], charge(ids["record_id"], attempt_number=1)
    )

    assert verdict.allowed is True


async def test_notification_for_another_subscription_does_not_authorise(
    conn, monkeypatch
):
    """Same customer, different mandate. It authorises nothing here."""
    now = freeze(monkeypatch, at_ist(DEFAULT_HOUR_IST))
    ids = await insert_scenario(conn)
    other = await insert_scenario(conn)
    await insert_notification(
        conn,
        ids["customer_id"],
        other["subscription_id"],
        now - timedelta(hours=1),
    )

    verdict = await guard_check(
        conn, ids["record_id"], charge(ids["record_id"], attempt_number=1)
    )

    assert verdict.allowed is False
    assert verdict.rule_name == RULE_NOTIFICATION
    assert verdict.details["reason"] == "no_notification_found"


async def test_dunning_contact_is_not_a_notification(conn, monkeypatch):
    now = freeze(monkeypatch, at_ist(DEFAULT_HOUR_IST))
    ids = await insert_scenario(conn)
    await insert_notification(
        conn,
        ids["customer_id"],
        ids["subscription_id"],
        now - timedelta(hours=1),
        purpose="dunning",
        channel="email",
    )

    verdict = await guard_check(
        conn, ids["record_id"], charge(ids["record_id"], attempt_number=1)
    )

    assert verdict.allowed is False
    assert verdict.rule_name == RULE_NOTIFICATION
    assert verdict.details["reason"] == "no_notification_found"


async def test_notification_dated_in_the_future_is_refused(conn, monkeypatch):
    now = freeze(monkeypatch, at_ist(DEFAULT_HOUR_IST))
    ids = await insert_scenario(conn)
    await insert_notification(
        conn, ids["customer_id"], ids["subscription_id"], now + timedelta(hours=1)
    )

    verdict = await guard_check(
        conn, ids["record_id"], charge(ids["record_id"], attempt_number=1)
    )

    assert verdict.allowed is False
    assert verdict.rule_name == RULE_NOTIFICATION
    assert verdict.details["reason"] == "notification_in_the_future"


async def test_blocked_when_a_pending_attempt_already_exists(conn, monkeypatch):
    now = freeze(monkeypatch, at_ist(DEFAULT_HOUR_IST))
    ids = await build_valid_scenario(conn, now)
    await insert_attempt(
        conn, ids["record_id"], 2, outcome="pending", scheduled_at=now
    )

    verdict = await guard_check(
        conn, ids["record_id"], charge(ids["record_id"], attempt_number=2)
    )

    assert verdict.allowed is False
    assert verdict.rule_name == RULE_DUPLICATE_ATTEMPT
    assert verdict.details["attempt_number"] == 2


async def test_settled_attempt_with_the_same_number_is_not_a_duplicate(
    conn, monkeypatch
):
    """Only a pending row means an action is already in flight."""
    now = freeze(monkeypatch, at_ist(DEFAULT_HOUR_IST))
    ids = await build_valid_scenario(conn, now)

    verdict = await guard_check(
        conn, ids["record_id"], charge(ids["record_id"], attempt_number=1)
    )

    assert verdict.allowed is True


async def test_missing_record_is_refused(conn, monkeypatch):
    freeze(monkeypatch, at_ist(DEFAULT_HOUR_IST))

    verdict = await guard_check(conn, 9_999_999, charge(9_999_999))

    assert verdict.allowed is False
    assert verdict.rule_name == RULE_RECORD_EXISTS


async def test_first_failing_rule_wins(conn, monkeypatch):
    """A record violating several rules reports the earliest one."""
    now = freeze(monkeypatch, at_ist(DEFAULT_HOUR_IST))
    ids = await insert_scenario(conn, mandate_status="revoked")
    for number in range(1, MAX_ATTEMPTS + 1):
        await insert_attempt(
            conn,
            ids["record_id"],
            number,
            outcome="failure",
            executed_at=now - timedelta(minutes=5),
        )

    verdict = await guard_check(conn, ids["record_id"], charge(ids["record_id"]))

    assert verdict.rule_name == RULE_MANDATE_NOT_REVOKED


async def test_verdict_details_survive_json_round_trip(conn, monkeypatch):
    """details is written to attempts.guard_verdict as JSONB."""
    now = freeze(monkeypatch, at_ist(10, 30))
    blocked_ids = await build_valid_scenario(conn, now)
    blocked = await guard_check(
        conn, blocked_ids["record_id"], charge(blocked_ids["record_id"])
    )

    freeze(monkeypatch, at_ist(DEFAULT_HOUR_IST))
    allowed_ids = await build_valid_scenario(conn, now)
    allowed = await guard_check(
        conn, allowed_ids["record_id"], charge(allowed_ids["record_id"])
    )

    assert blocked.allowed is False
    assert allowed.allowed is True
    for verdict in (blocked, allowed):
        assert json.loads(json.dumps(verdict.details)) == verdict.details


async def test_guard_ignores_a_stale_in_memory_record(conn, monkeypatch):
    """The record handed in claims an active mandate; the database says revoked."""
    now = freeze(monkeypatch, at_ist(DEFAULT_HOUR_IST))
    ids = await build_valid_scenario(conn, now, mandate_status="revoked")
    stale = AtRiskRecord(
        id=ids["record_id"],
        subscription_id=ids["subscription_id"],
        customer_id=ids["customer_id"],
        invoice_id="inv_stale",
        amount=49900,
        failure_code="INSUFFICIENT_FUNDS",
        failure_class=FailureClass.TIME_DEPENDENT,
        status=RecordStatus.SCHEDULED,
        created_at=now,
        updated_at=now,
    )

    verdict = await guard_check(conn, stale, charge(ids["record_id"]))

    assert verdict.allowed is False
    assert verdict.rule_name == RULE_MANDATE_NOT_REVOKED


def test_rule_order_is_the_documented_order():
    assert [name for name, _ in guard.CHECKS] == [
        RULE_MANDATE_NOT_REVOKED,
        RULE_MANDATE_NOT_EXPIRED,
        RULE_ATTEMPT_CAP,
        RULE_COOLDOWN,
        RULE_EXECUTION_WINDOW,
        RULE_NOTIFICATION,
        RULE_DUPLICATE_ATTEMPT,
    ]


def test_guard_shares_no_code_with_the_planning_path():
    """Invariant 2, enforced structurally rather than by memory.

    The guard may import models and config. Importing services, or the audit
    helpers the allocator uses, would make a bug in shared code invisible to
    exactly the layer meant to catch it.
    """
    tree = ast.parse((REPO_ROOT / "forbear" / "core" / "guard.py").read_text())

    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module or "")

    project_imports = {name for name in imported if name.startswith("forbear")}
    assert project_imports == {"forbear.config.limits", "forbear.models.models"}
