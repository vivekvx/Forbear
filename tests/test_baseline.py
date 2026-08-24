"""Fixed-schedule baseline tests.

The baseline has to be reproduced faithfully rather than favourably. Every
temptation here runs one way - drop the records that were never going to
recover, skip the ones with a dead mandate, space the retries a little more
sensibly - and every one of them would flatter Forbear by making its
counterfactual worse than the real thing. So these tests mostly assert that the
baseline does the blunt thing: three attempts, fixed offsets, no scoring, no
ranking, no budget, no opinion about the customer.
"""

from __future__ import annotations

from datetime import timedelta

import pytest

from forbear.models.models import ActionKind, RecordStatus
from forbear.services.allocator import (
    SKIP_TERMINAL_FAILURE_CLASS,
    SKIP_UNCLASSIFIED_FAILURE_CODE,
)
from forbear.services.baseline import (
    ACTION_SCHEDULE,
    FIXED_OFFSETS,
    allocate_fixed_schedule,
)
from tests.conftest import insert_scenario
from tests.test_allocator import audit_actions, scenario, status_of


async def test_every_non_terminal_record_is_scheduled_without_scoring(conn):
    """No scores are passed in at all - the signature takes bare ids, because a
    policy with no use for a score should not be able to accept one."""
    ids = [(await scenario(conn))["record_id"] for _ in range(4)]

    plan = await allocate_fixed_schedule(conn, ids)

    assert {action.record_id for action in plan.scheduled} == set(ids)
    assert plan.skipped == []
    assert plan.summary["scheduled_count"] == 4


async def test_three_attempts_per_record_at_24_48_and_72_hours(conn):
    ids = await scenario(conn)
    failed_at = await conn.fetchval(
        "SELECT created_at FROM at_risk_records WHERE id = $1", ids["record_id"]
    )

    plan = await allocate_fixed_schedule(conn, [ids["record_id"]])

    slots = [action.scheduled_at for action in plan.scheduled]
    assert slots == [
        failed_at + timedelta(hours=24),
        failed_at + timedelta(hours=48),
        failed_at + timedelta(hours=72),
    ]
    assert all(action.action_kind is ActionKind.CHARGE for action in plan.scheduled)
    assert plan.summary["attempts_scheduled"] == 3


def test_the_offsets_are_the_published_schedule():
    """T+1/T+2/T+3, stated once so a test cannot quietly disagree with it."""
    assert FIXED_OFFSETS == (
        timedelta(hours=24),
        timedelta(hours=48),
        timedelta(hours=72),
    )


async def test_attempts_are_spent_three_at_a_time_on_every_record(conn):
    """The comparison that matters: the baseline burns its budget uniformly,
    including on records where all three attempts were always going to fail."""
    ids = [(await scenario(conn))["record_id"] for _ in range(5)]

    plan = await allocate_fixed_schedule(conn, ids)

    assert len(plan.scheduled) == 15
    assert plan.summary["attempts_scheduled"] == 15


async def test_only_terminal_classes_are_skipped(conn):
    """A record that will certainly not recover still gets its three attempts,
    as long as the rails would accept them. That is the policy."""
    terminal = await scenario(conn, failure_class="terminal")
    reauth = await scenario(conn, failure_class="reauth_required")
    transient = await scenario(conn, failure_class="transient")
    time_dependent = await scenario(conn, failure_class="time_dependent")

    plan = await allocate_fixed_schedule(
        conn,
        [
            terminal["record_id"],
            reauth["record_id"],
            transient["record_id"],
            time_dependent["record_id"],
        ],
    )

    assert [skip.record_id for skip in plan.skipped] == [terminal["record_id"]]
    assert plan.skipped[0].skip_reason == SKIP_TERMINAL_FAILURE_CLASS
    # reauth_required is retried on purpose: Razorpay retries those too, and
    # dropping them here would quietly improve the baseline past what it does.
    assert reauth["record_id"] in {action.record_id for action in plan.scheduled}


async def test_a_dead_mandate_is_not_a_baseline_skip(conn):
    """The fixed schedule has no view of the mandate. It fires and the rails
    refuse - which is exactly the wasted attempt the comparison should show."""
    ids = await scenario(conn, mandate_status="revoked")

    plan = await allocate_fixed_schedule(conn, [ids["record_id"]])

    assert len(plan.scheduled) == 3
    assert plan.skipped == []


async def test_an_unclassified_record_is_skipped(conn):
    """The one rule the baseline does share: a code with no mapping is on the
    exception list, and nothing may treat it as recoverable."""
    ids = await insert_scenario(conn)
    await conn.execute(
        "UPDATE at_risk_records SET failure_class = NULL WHERE id = $1",
        ids["record_id"],
    )

    plan = await allocate_fixed_schedule(conn, [ids["record_id"]])

    assert plan.skipped[0].skip_reason == SKIP_UNCLASSIFIED_FAILURE_CODE


async def test_the_plan_shape_matches_the_allocator(conn):
    """The measurement harness reads both policies through one code path, so a
    difference in the comparison can never be a difference in the plumbing."""
    ids = await scenario(conn)

    plan = await allocate_fixed_schedule(conn, [ids["record_id"]])

    assert set(plan.summary) >= {
        "considered",
        "scheduled_count",
        "skipped_count",
        "scheduled_amount",
        "skipped_amount",
        "skips_by_reason",
        "batch_budget",
    }
    assert plan.summary["policy"] == "fixed_t1_t2_t3"
    action = plan.scheduled[0]
    assert (action.record_id, action.action_kind) == (
        ids["record_id"],
        ActionKind.CHARGE,
    )


async def test_records_move_to_scheduled_and_leave_a_trail(conn):
    ids = await scenario(conn)

    await allocate_fixed_schedule(conn, [ids["record_id"]])

    status, skip_reason = await status_of(conn, ids["record_id"])
    assert status == RecordStatus.SCHEDULED.value
    assert skip_reason is None

    actions = await audit_actions(conn, ids["record_id"])
    assert ACTION_SCHEDULE in actions
    assert "transition:open->scheduled" in actions


async def test_an_empty_batch_is_a_valid_cycle(conn):
    plan = await allocate_fixed_schedule(conn, [])

    assert plan.scheduled == []
    assert plan.summary["considered"] == 0


async def test_a_record_that_does_not_exist_is_refused(conn):
    with pytest.raises(LookupError):
        await allocate_fixed_schedule(conn, [999_999_999])


async def test_the_baseline_refuses_to_run_outside_a_transaction(db_pool):
    async with db_pool.acquire() as connection:
        with pytest.raises(RuntimeError, match="transaction"):
            await allocate_fixed_schedule(connection, [1])
