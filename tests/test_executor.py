"""Executor tests.

The executor is the last thing between a plan and someone's bank account, so
these tests care less about the happy path than about what happens when the
guard says no, when the outbound call fails, and when the plan is stale. One
property is worth stating outright: nothing here ever consults ground truth.
The simulator is passed in, and every test below could swap it for a function
that always returns False without the executor noticing.
"""

from __future__ import annotations

import ast
import json
import pathlib
from datetime import datetime, timedelta

import pytest

from forbear.config.limits import IST
from forbear.core.audit import server_now
from forbear.core.guard import GuardVerdict
from forbear.models.models import ActionKind, AttemptOutcome, RecordStatus
from forbear.services.allocator import AllocationPlan, ScheduledAction
from forbear.services.executor import (
    ACTION_BLOCKED,
    ACTION_FAILURE,
    ACTION_NOT_ATTEMPTED,
    ACTION_SUCCESS,
    OutboundResult,
    ServerClock,
    VirtualClock,
    allow_everything,
    execute_plan,
)
from tests.test_allocator import audit_actions, audit_details, scenario, status_of

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent


def plan_for(*actions: ScheduledAction) -> AllocationPlan:
    return AllocationPlan(scheduled=list(actions))


def charge(record_id: int, at: datetime) -> ScheduledAction:
    return ScheduledAction(record_id, ActionKind.CHARGE, at)


def always(success: bool):
    """An outbound stub. The executor cannot tell this from Razorpay."""

    def execute_fn(record, action):
        return OutboundResult(success=success, detail={"stub": True})

    return execute_fn


def blocking_guard(rule: str = "test_rule"):
    def guard_fn(conn, record_id, action):
        return GuardVerdict(
            allowed=False, rule_name=rule, details={"why": "test", "record": record_id}
        )

    return guard_fn


async def attempts_for(conn, record_id: int) -> list[dict]:
    rows = await conn.fetch(
        """
        SELECT attempt_number, outcome, scheduled_at, executed_at, guard_verdict
        FROM attempts WHERE at_risk_record_id = $1 ORDER BY attempt_number
        """,
        record_id,
    )
    return [dict(row) for row in rows]


# --- a. the happy path ------------------------------------------------------


async def test_a_successful_attempt_recovers_the_record(conn):
    ids = await scenario(conn)
    now = await server_now(conn)

    report = await execute_plan(
        conn,
        plan_for(charge(ids["record_id"], now + timedelta(hours=1))),
        allow_everything,
        always(True),
        VirtualClock(now),
    )

    status, _ = await status_of(conn, ids["record_id"])
    assert status == RecordStatus.RECOVERED.value

    rows = await attempts_for(conn, ids["record_id"])
    assert len(rows) == 1
    assert rows[0]["outcome"] == AttemptOutcome.SUCCESS.value
    assert rows[0]["attempt_number"] == 1
    assert rows[0]["executed_at"] is not None

    assert report.succeeded == 1
    assert report.amount_recovered == 49900
    assert ACTION_SUCCESS in await audit_actions(conn, ids["record_id"])


async def test_the_success_audit_entry_records_the_outbound_detail(conn):
    """Whatever the gateway said is what the chain should carry."""
    ids = await scenario(conn)
    now = await server_now(conn)

    await execute_plan(
        conn,
        plan_for(charge(ids["record_id"], now)),
        allow_everything,
        always(True),
        VirtualClock(now),
    )

    details = await audit_details(conn, ids["record_id"], ACTION_SUCCESS)
    assert details["outbound"] == {"stub": True}
    assert details["attempt_number"] == 1
    assert details["amount"] == 49900


# --- b. the guard says no ---------------------------------------------------


async def test_a_blocked_action_stores_the_verdict_and_charges_nobody(conn):
    """The attempt row exists because the guard was asked and refused, and that
    refusal is a fact about this record someone may later have to explain."""
    ids = await scenario(conn)
    now = await server_now(conn)
    calls: list = []

    def never_called(record, action):
        calls.append(record.id)
        return True

    report = await execute_plan(
        conn,
        plan_for(charge(ids["record_id"], now)),
        blocking_guard("mandate_not_revoked"),
        never_called,
        VirtualClock(now),
    )

    assert calls == [], "the outbound call ran despite a blocking verdict"

    rows = await attempts_for(conn, ids["record_id"])
    assert rows[0]["outcome"] == AttemptOutcome.BLOCKED_BY_GUARD.value
    verdict = rows[0]["guard_verdict"]
    verdict = json.loads(verdict) if isinstance(verdict, str) else verdict
    assert verdict["allowed"] is False
    assert verdict["rule_name"] == "mandate_not_revoked"

    assert report.blocked_by_guard == 1
    assert report.blocked_by_rule == {"mandate_not_revoked": 1}
    assert report.amount_blocked == 49900
    assert ACTION_BLOCKED in await audit_actions(conn, ids["record_id"])


async def test_a_blocked_record_keeps_its_plan(conn):
    """A block is the guard refusing this action at this instant, not a verdict
    on the record. Moving it to skipped would be the executor deciding
    something, which is not its job."""
    ids = await scenario(conn)
    now = await server_now(conn)

    await execute_plan(
        conn,
        plan_for(charge(ids["record_id"], now)),
        blocking_guard(),
        always(True),
        VirtualClock(now),
    )

    status, _ = await status_of(conn, ids["record_id"])
    assert status == RecordStatus.OPEN.value


async def test_the_real_guard_blocks_a_revoked_mandate(conn):
    """Not a stub: the guard the production wiring passes, refusing for real."""
    from forbear.core.guard import RULE_MANDATE_NOT_REVOKED, guard_check

    ids = await scenario(conn, mandate_status="revoked")
    now = await server_now(conn)

    report = await execute_plan(
        conn,
        plan_for(charge(ids["record_id"], now)),
        guard_check,
        always(True),
        VirtualClock(now),
    )

    assert report.blocked_by_guard == 1
    assert report.blocked_by_rule == {RULE_MANDATE_NOT_REVOKED: 1}


# --- c. the outbound call fails --------------------------------------------


async def test_a_failed_attempt_returns_the_record_to_open(conn):
    """The attempt is spent, the record is not: the next cycle gets to decide
    whether it is still worth chasing."""
    ids = await scenario(conn)
    now = await server_now(conn)

    report = await execute_plan(
        conn,
        plan_for(charge(ids["record_id"], now)),
        allow_everything,
        always(False),
        VirtualClock(now),
    )

    status, skip_reason = await status_of(conn, ids["record_id"])
    assert status == RecordStatus.OPEN.value
    assert skip_reason is None

    rows = await attempts_for(conn, ids["record_id"])
    assert rows[0]["outcome"] == AttemptOutcome.FAILURE.value
    assert report.failed == 1
    assert report.amount_recovered == 0
    assert ACTION_FAILURE in await audit_actions(conn, ids["record_id"])


async def test_a_second_attempt_numbers_itself_correctly(conn):
    """Attempt numbers come off the highest already used, not off a count - a
    gap must not cause a number to be handed out twice."""
    ids = await scenario(conn)
    now = await server_now(conn)

    await execute_plan(
        conn,
        plan_for(
            charge(ids["record_id"], now),
            charge(ids["record_id"], now + timedelta(days=2)),
        ),
        allow_everything,
        always(False),
        VirtualClock(now),
    )

    rows = await attempts_for(conn, ids["record_id"])
    assert [row["attempt_number"] for row in rows] == [1, 2]


async def test_a_recovered_record_drops_the_rest_of_its_plan(conn):
    """The plan was written before any of this ran. Firing the third attempt at
    someone who paid on the first would be the system charging them again."""
    ids = await scenario(conn)
    now = await server_now(conn)

    report = await execute_plan(
        conn,
        plan_for(
            charge(ids["record_id"], now),
            charge(ids["record_id"], now + timedelta(days=1)),
            charge(ids["record_id"], now + timedelta(days=2)),
        ),
        allow_everything,
        always(True),
        VirtualClock(now),
    )

    assert report.succeeded == 1
    assert report.not_attempted == 2
    assert len(await attempts_for(conn, ids["record_id"])) == 1
    assert ACTION_NOT_ATTEMPTED in await audit_actions(conn, ids["record_id"])


# --- d. the virtual clock ---------------------------------------------------


async def test_the_guard_sees_the_simulated_instant_not_the_wall_clock(conn):
    """The whole reason the clock is injectable. A record scheduled for 14:00
    IST must be evaluated at 14:00 IST, or a simulation of next month is really
    a simulation of the moment the test happened to run.
    """
    ids = await scenario(conn)
    now = await server_now(conn)
    slot = (
        (now + timedelta(days=3))
        .astimezone(IST)
        .replace(hour=14, minute=0, second=0, microsecond=0)
    )
    seen: list[datetime] = []

    clock = VirtualClock(now)

    async def clock_reading_guard(conn, record_id, action):
        seen.append(await clock.now(conn))
        return GuardVerdict(allowed=True, rule_name=None, details={})

    await execute_plan(
        conn,
        plan_for(charge(ids["record_id"], slot)),
        clock_reading_guard,
        always(True),
        clock,
    )

    assert len(seen) == 1
    observed = seen[0].astimezone(IST)
    assert observed == slot
    assert observed.hour == 14
    assert observed - now.astimezone(IST) > timedelta(days=2)


async def test_the_attempt_row_records_the_simulated_execution_time(conn):
    ids = await scenario(conn)
    now = await server_now(conn)
    slot = now + timedelta(days=5)

    await execute_plan(
        conn,
        plan_for(charge(ids["record_id"], slot)),
        allow_everything,
        always(True),
        VirtualClock(now),
    )

    rows = await attempts_for(conn, ids["record_id"])
    assert rows[0]["executed_at"] == slot
    assert rows[0]["scheduled_at"] == slot


async def test_the_clock_only_moves_forward(conn):
    """A plan replayed backwards would let a later attempt be evaluated before
    an earlier one, and the cooldown rule would read as satisfied when it is
    not."""
    now = await server_now(conn)
    clock = VirtualClock(now)

    await clock.advance_to(now + timedelta(days=4))
    await clock.advance_to(now - timedelta(days=10))

    assert await clock.now(conn) == now + timedelta(days=4)


async def test_actions_execute_in_slot_order_not_plan_order(conn):
    """Plans are ranked by value; time is not. Executing a value-ordered plan
    verbatim drags the clock past the slots of everything below the top
    record."""
    first = await scenario(conn)
    second = await scenario(conn)
    now = await server_now(conn)
    order: list[int] = []

    def recording(record, action):
        order.append(record.id)
        return False

    await execute_plan(
        conn,
        plan_for(
            charge(first["record_id"], now + timedelta(days=9)),
            charge(second["record_id"], now + timedelta(days=1)),
        ),
        allow_everything,
        recording,
        VirtualClock(now),
    )

    assert order == [second["record_id"], first["record_id"]]


async def test_the_server_clock_ignores_instructions_to_move(conn):
    """Production. Invariant 3: the database's clock is not negotiable."""
    clock = ServerClock()
    before = await clock.now(conn)

    await clock.advance_to(before + timedelta(days=365))

    assert await clock.now(conn) - before < timedelta(seconds=5)


# --- e. the report adds up --------------------------------------------------


async def test_report_counts_reconcile(conn):
    recovered = await scenario(conn)
    failed = await scenario(conn)
    blocked = await scenario(conn)
    now = await server_now(conn)

    def per_record(record, action):
        return record.id == recovered["record_id"]

    def guard_fn(conn, record_id, action):
        if record_id == blocked["record_id"]:
            return GuardVerdict(allowed=False, rule_name="cap", details={})
        return GuardVerdict(allowed=True, rule_name=None, details={})

    report = await execute_plan(
        conn,
        plan_for(
            charge(recovered["record_id"], now),
            charge(failed["record_id"], now + timedelta(hours=1)),
            charge(blocked["record_id"], now + timedelta(hours=2)),
        ),
        guard_fn,
        per_record,
        VirtualClock(now),
    )

    assert report.attempted == 3
    assert (
        report.attempted
        == report.succeeded + report.failed + report.blocked_by_guard
    )
    assert (report.succeeded, report.failed, report.blocked_by_guard) == (1, 1, 1)
    assert len(report.details) == 3


async def test_an_empty_plan_executes_nothing(conn):
    report = await execute_plan(
        conn, AllocationPlan(), allow_everything, always(True), None
    )

    assert report.attempted == 0
    assert report.details == []


# --- refusals ---------------------------------------------------------------


async def test_a_plan_for_a_missing_record_is_refused(conn):
    now = await server_now(conn)

    with pytest.raises(LookupError):
        await execute_plan(
            conn,
            plan_for(charge(999_999_999, now)),
            allow_everything,
            always(True),
            VirtualClock(now),
        )


async def test_an_execute_fn_returning_nonsense_is_refused(conn):
    """A stub returning None would otherwise read as a failed charge, and the
    record would be quietly marked unrecoverable."""
    ids = await scenario(conn)
    now = await server_now(conn)

    with pytest.raises(TypeError):
        await execute_plan(
            conn,
            plan_for(charge(ids["record_id"], now)),
            allow_everything,
            lambda record, action: None,
            VirtualClock(now),
        )


async def test_execute_plan_refuses_to_run_outside_a_transaction(db_pool):
    async with db_pool.acquire() as connection:
        with pytest.raises(RuntimeError, match="transaction"):
            await execute_plan(
                connection, AllocationPlan(), allow_everything, always(True), None
            )


async def test_a_bare_boolean_is_an_acceptable_outbound_result(conn):
    """A simulator should not have to build a result object to say no."""
    ids = await scenario(conn)
    now = await server_now(conn)

    report = await execute_plan(
        conn,
        plan_for(charge(ids["record_id"], now)),
        allow_everything,
        lambda record, action: True,
        VirtualClock(now),
    )

    assert report.succeeded == 1


# --- the boundary -----------------------------------------------------------


def test_the_executor_never_imports_the_generator():
    """execute_fn is injected precisely so this stays true: the code that moves
    money has no path to the answer key."""
    tree = ast.parse((REPO_ROOT / "forbear" / "services" / "executor.py").read_text())

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            assert not (node.module or "").startswith("forbear.generator")
        elif isinstance(node, ast.Import):
            assert not any(
                alias.name.startswith("forbear.generator") for alias in node.names
            )
