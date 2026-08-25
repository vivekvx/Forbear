"""Allocator tests.

Two kinds of assertion here. The first checks the plan: what got scheduled, in
what order, and what did not. The second checks the trail - that every decision
moved the record's status and left an audit entry carrying the numbers behind
it. The second kind matters more. A plan with a bug produces a bad cycle; a
decision with no reviewable reason produces a system nobody can be held to.

Every test runs against a real database inside a rolled-back transaction, which
is also what the allocator requires: it refuses to run outside one.
"""

from __future__ import annotations

import ast
import json
import pathlib
from datetime import timedelta

import pytest

from forbear.config.limits import IST, MAX_ATTEMPTS, NPCI_DEBIT_WINDOWS_IST
from forbear.core.audit import server_now
from forbear.models.models import ActionKind, RecordStatus
from forbear.services.allocator import (
    ACTION_SCHEDULE,
    ACTION_SKIP,
    SKIP_ATTEMPT_BUDGET_EXHAUSTED,
    SKIP_BATCH_BUDGET_EXHAUSTED,
    SKIP_MANDATE_STATE_INVALID,
    SKIP_NEGATIVE_NET_VALUE,
    SKIP_TERMINAL_FAILURE_CLASS,
    SKIP_UNCLASSIFIED_FAILURE_CODE,
    AllocationConfig,
    ScoredRecord,
    allocate,
)
from forbear.services.unconstrained_baseline import allocate_unconstrained
from tests.conftest import insert_attempt, insert_notification, insert_scenario

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent


async def scenario(
    conn,
    *,
    failure_class: str = "time_dependent",
    mandate_status: str = "active",
    amount: int | None = None,
) -> dict[str, int]:
    """A record the allocator can consider, with its class and mandate set."""
    ids = await insert_scenario(conn, mandate_status=mandate_status)
    await conn.execute(
        """
        UPDATE at_risk_records
        SET failure_class = $2::failure_class,
            amount = COALESCE($3, amount)
        WHERE id = $1
        """,
        ids["record_id"],
        failure_class,
        amount,
    )
    return ids


async def unclassified_scenario(conn) -> dict[str, int]:
    ids = await insert_scenario(conn)
    await conn.execute(
        "UPDATE at_risk_records SET failure_class = NULL WHERE id = $1",
        ids["record_id"],
    )
    return ids


def scored(record_id: int, index: float, cate: float = 0.3) -> ScoredRecord:
    return ScoredRecord(record_id=record_id, cate=cate, whittle_index=index)


async def status_of(conn, record_id: int) -> tuple[str, str | None]:
    row = await conn.fetchrow(
        "SELECT status, skip_reason FROM at_risk_records WHERE id = $1", record_id
    )
    return row["status"], row["skip_reason"]


async def audit_actions(conn, record_id: int) -> list[str]:
    rows = await conn.fetch(
        """
        SELECT action FROM audit_log
        WHERE entity_type = 'at_risk_record' AND entity_id = $1
        ORDER BY id
        """,
        str(record_id),
    )
    return [row["action"] for row in rows]


async def audit_details(conn, record_id: int, action: str) -> dict:
    raw = await conn.fetchval(
        """
        SELECT details FROM audit_log
        WHERE entity_type = 'at_risk_record' AND entity_id = $1 AND action = $2
        ORDER BY id DESC LIMIT 1
        """,
        str(record_id),
        action,
    )
    assert raw is not None, f"no {action} audit entry for record {record_id}"
    return json.loads(raw) if isinstance(raw, str) else raw


# --- a. positive index is scheduled ----------------------------------------


async def test_a_positive_index_record_is_scheduled(conn):
    ids = await scenario(conn)

    plan = await allocate(conn, [scored(ids["record_id"], index=4.2)])

    assert [action.record_id for action in plan.scheduled] == [ids["record_id"]]
    assert plan.scheduled[0].action_kind is ActionKind.CHARGE
    assert plan.skipped == []


async def test_the_scheduled_slot_lands_inside_an_npci_window(conn):
    """The allocator considers the windows when choosing a slot. It does not
    enforce them - the guard does that at execution - but a plan whose slots
    are all illegal is a plan that recovers nothing."""
    ids = await scenario(conn)

    plan = await allocate(conn, [scored(ids["record_id"], index=4.2)])

    slot_ist = plan.scheduled[0].scheduled_at.astimezone(IST)
    minute_of_day = slot_ist.hour * 60 + slot_ist.minute
    assert any(
        start <= minute_of_day < end for start, end in NPCI_DEBIT_WINDOWS_IST
    ), f"scheduled outside every NPCI window: {slot_ist:%H:%M}"


async def test_without_a_notification_nothing_is_scheduled_inside_24_hours(conn):
    """The executor has to send a pre-debit notification first, and the
    customer is owed the notice period. Scheduling sooner would produce a plan
    the guard refuses."""
    ids = await scenario(conn)
    now = await server_now(conn)

    plan = await allocate(conn, [scored(ids["record_id"], index=4.2)])

    assert plan.scheduled[0].scheduled_at >= now + timedelta(hours=24)


async def test_a_matured_notification_allows_a_sooner_slot(conn):
    """Notice given yesterday: the customer has had their day, so the wait is
    already served and the debit can go in the next legal window."""
    ids = await scenario(conn)
    now = await server_now(conn)
    await insert_notification(
        conn, ids["customer_id"], ids["subscription_id"], now - timedelta(hours=25)
    )

    plan = await allocate(conn, [scored(ids["record_id"], index=4.2)])

    assert plan.scheduled[0].scheduled_at < now + timedelta(hours=24)
    details = await audit_details(conn, ids["record_id"], ACTION_SCHEDULE)
    assert details["chosen_for"]["notification_usable"] is True


async def test_a_fresh_notification_does_not_buy_a_sooner_slot(conn):
    """The inverted-rule case, from the planning side.

    A notification sent an hour ago is not notice yet. The allocator has to
    wait for it to mature, or it would plan a debit the guard refuses - which
    is exactly what happened while the guard's bound was the wrong way round.
    """
    ids = await scenario(conn)
    now = await server_now(conn)
    sent_at = now - timedelta(hours=1)
    await insert_notification(
        conn, ids["customer_id"], ids["subscription_id"], sent_at
    )

    plan = await allocate(conn, [scored(ids["record_id"], index=4.2)])

    assert plan.scheduled[0].scheduled_at >= sent_at + timedelta(hours=24)


async def test_a_prior_success_hour_is_preferred(conn):
    """The customer paid at this hour before; the money is there then."""
    ids = await scenario(conn)
    now = await server_now(conn)
    # 14:00 IST sits inside the 13:00-17:00 window and is not the earliest slot
    # available, so choosing it can only be the history talking.
    paid_at = (now - timedelta(days=40)).astimezone(IST).replace(hour=14, minute=0)
    await insert_attempt(
        conn, ids["record_id"], 1, outcome="success", executed_at=paid_at
    )

    plan = await allocate(conn, [scored(ids["record_id"], index=4.2)])

    assert plan.scheduled[0].scheduled_at.astimezone(IST).hour == 14
    details = await audit_details(conn, ids["record_id"], ACTION_SCHEDULE)
    assert details["chosen_for"]["matched_prior_success_hour"] is True


# --- b. negative index is skipped, with the numbers ------------------------


async def test_a_negative_index_record_is_skipped_with_its_numbers(conn):
    """The most important output in the system. A reviewer has to be able to
    see the trade that was made and say which term was wrong."""
    ids = await scenario(conn)

    plan = await allocate(conn, [scored(ids["record_id"], index=-3.5, cate=-0.22)])

    assert plan.scheduled == []
    skip = plan.skipped[0]
    assert skip.skip_reason == SKIP_NEGATIVE_NET_VALUE
    assert skip.details["cate"] == pytest.approx(-0.22)
    assert skip.details["whittle_index"] == pytest.approx(-3.5)
    assert skip.details["amount"] == 49900
    assert skip.details["ltv_at_risk"] > 0
    assert skip.details["threshold"] == 0.0


async def test_the_skip_audit_entry_carries_the_quantitative_reason(conn):
    """"negative_net_value" on its own is not a reason, it is a label."""
    ids = await scenario(conn)

    await allocate(conn, [scored(ids["record_id"], index=-3.5, cate=-0.22)])

    details = await audit_details(conn, ids["record_id"], ACTION_SKIP)
    assert details["skip_reason"] == SKIP_NEGATIVE_NET_VALUE
    for field in ("cate", "amount", "ltv_at_risk", "whittle_index", "threshold"):
        assert field in details, f"skip audit entry is missing {field}"


# --- c. terminal records, regardless of index ------------------------------


@pytest.mark.parametrize("failure_class", ["terminal", "reauth_required"])
async def test_a_terminal_record_is_skipped_whatever_its_index(conn, failure_class):
    """Skipped for the class, not the score: "we did not chase this because the
    index was low" would be a false account of a record that could not have
    been charged at all."""
    ids = await scenario(conn, failure_class=failure_class)

    plan = await allocate(conn, [scored(ids["record_id"], index=99.0)])

    assert plan.scheduled == []
    assert plan.skipped[0].skip_reason == SKIP_TERMINAL_FAILURE_CLASS
    assert plan.skipped[0].details["failure_class"] == failure_class


@pytest.mark.parametrize("mandate_status", ["revoked", "expired"])
async def test_a_dead_mandate_is_skipped_whatever_its_index(conn, mandate_status):
    ids = await scenario(conn, mandate_status=mandate_status)

    plan = await allocate(conn, [scored(ids["record_id"], index=99.0)])

    assert plan.skipped[0].skip_reason == SKIP_MANDATE_STATE_INVALID
    assert plan.skipped[0].details["mandate_status"] == mandate_status


async def test_an_unclassified_code_is_never_treated_as_recoverable(conn):
    """NULL failure_class means the exception list, not a default."""
    ids = await unclassified_scenario(conn)

    plan = await allocate(conn, [scored(ids["record_id"], index=99.0)])

    assert plan.skipped[0].skip_reason == SKIP_UNCLASSIFIED_FAILURE_CODE


# --- d. attempt cap, regardless of index -----------------------------------


async def test_a_record_at_the_attempt_cap_is_skipped_whatever_its_index(conn):
    """Invariant 5: the cap is a hard limit, not a preference a high score can
    outbid."""
    ids = await scenario(conn)
    now = await server_now(conn)
    for number in range(1, MAX_ATTEMPTS + 1):
        await insert_attempt(
            conn,
            ids["record_id"],
            number,
            outcome="failure",
            executed_at=now - timedelta(days=number),
        )

    plan = await allocate(conn, [scored(ids["record_id"], index=99.0)])

    assert plan.scheduled == []
    skip = plan.skipped[0]
    assert skip.skip_reason == SKIP_ATTEMPT_BUDGET_EXHAUSTED
    assert skip.details["attempts_so_far"] == MAX_ATTEMPTS
    assert skip.details["max_attempts"] == MAX_ATTEMPTS


async def test_a_record_one_below_the_cap_is_still_eligible(conn):
    """Guards the boundary in the direction that costs money to get wrong."""
    ids = await scenario(conn)
    now = await server_now(conn)
    for number in range(1, MAX_ATTEMPTS):
        await insert_attempt(
            conn,
            ids["record_id"],
            number,
            outcome="failure",
            executed_at=now - timedelta(days=number),
        )

    plan = await allocate(conn, [scored(ids["record_id"], index=4.0)])

    assert len(plan.scheduled) == 1


# --- e. ranking ------------------------------------------------------------


async def test_records_are_scheduled_in_whittle_index_order(conn):
    indices = [1.5, 9.9, 0.2, 4.4, 7.1]
    ids = [(await scenario(conn))["record_id"] for _ in indices]
    records = [scored(record_id, index) for record_id, index in zip(ids, indices)]

    plan = await allocate(conn, records)

    by_id = {record.record_id: record.whittle_index for record in records}
    ordered = [by_id[action.record_id] for action in plan.scheduled]
    assert ordered == sorted(ordered, reverse=True)
    assert ordered[0] == max(indices)


# --- f. batch budget -------------------------------------------------------


async def test_a_batch_budget_of_three_schedules_three_and_skips_two(conn):
    indices = [1.0, 2.0, 3.0, 4.0, 5.0]
    ids = [(await scenario(conn))["record_id"] for _ in indices]
    records = [scored(record_id, index) for record_id, index in zip(ids, indices)]

    plan = await allocate(conn, records, AllocationConfig(batch_budget=3))

    assert len(plan.scheduled) == 3
    assert len(plan.skipped) == 2
    assert {skip.skip_reason for skip in plan.skipped} == {SKIP_BATCH_BUDGET_EXHAUSTED}

    # The budget goes to the top of the ranking, and the two left behind are
    # the two worth least - not the two that happened to be last in the list.
    by_id = {record.record_id: record.whittle_index for record in records}
    scheduled_indices = sorted(by_id[a.record_id] for a in plan.scheduled)
    skipped_indices = sorted(by_id[s.record_id] for s in plan.skipped)
    assert scheduled_indices == [3.0, 4.0, 5.0]
    assert skipped_indices == [1.0, 2.0]


async def test_a_budget_skip_still_records_what_it_gave_up(conn):
    """These records were worth chasing. The entry has to say so, or a reviewer
    cannot tell a budget skip from a value skip."""
    ids = [(await scenario(conn))["record_id"] for _ in range(2)]
    records = [scored(ids[0], 5.0), scored(ids[1], 1.0)]

    plan = await allocate(conn, records, AllocationConfig(batch_budget=1))

    skip = plan.skipped[0]
    assert skip.details["batch_budget"] == 1
    assert skip.details["whittle_index"] == pytest.approx(1.0)
    assert skip.details["ltv_at_risk"] > 0


async def test_no_budget_means_no_ceiling(conn):
    ids = [(await scenario(conn))["record_id"] for _ in range(4)]
    records = [scored(record_id, 3.0) for record_id in ids]

    plan = await allocate(conn, records)

    assert len(plan.scheduled) == 4


# --- g. every decision leaves a trail --------------------------------------


async def test_every_skip_has_an_audit_entry(conn):
    terminal = await scenario(conn, failure_class="terminal")
    negative = await scenario(conn)
    capped = await scenario(conn)
    now = await server_now(conn)
    for number in range(1, MAX_ATTEMPTS + 1):
        await insert_attempt(
            conn, capped["record_id"], number, executed_at=now - timedelta(days=number)
        )

    plan = await allocate(
        conn,
        [
            scored(terminal["record_id"], 5.0),
            scored(negative["record_id"], -1.0),
            scored(capped["record_id"], 5.0),
        ],
    )

    assert len(plan.skipped) == 3
    for skip in plan.skipped:
        actions = await audit_actions(conn, skip.record_id)
        assert ACTION_SKIP in actions
        assert "transition:open->skipped" in actions


async def test_every_scheduled_record_has_an_audit_entry(conn):
    ids = await scenario(conn)

    await allocate(conn, [scored(ids["record_id"], 4.0)])

    actions = await audit_actions(conn, ids["record_id"])
    assert ACTION_SCHEDULE in actions
    assert "transition:open->scheduled" in actions


async def test_the_schedule_audit_entry_records_the_score_it_acted_on(conn):
    ids = await scenario(conn)

    await allocate(conn, [scored(ids["record_id"], 4.25, cate=0.31)])

    details = await audit_details(conn, ids["record_id"], ACTION_SCHEDULE)
    assert details["whittle_index"] == pytest.approx(4.25)
    assert details["cate"] == pytest.approx(0.31)
    assert details["action_kind"] == ActionKind.CHARGE.value
    assert details["scheduled_at"]


# --- h. state transitions --------------------------------------------------


async def test_scheduled_records_end_in_scheduled_status(conn):
    ids = [(await scenario(conn))["record_id"] for _ in range(3)]

    plan = await allocate(conn, [scored(record_id, 3.0) for record_id in ids])

    for action in plan.scheduled:
        status, skip_reason = await status_of(conn, action.record_id)
        assert status == RecordStatus.SCHEDULED.value
        assert skip_reason is None


async def test_skipped_records_end_in_skipped_status_with_a_reason(conn):
    terminal = await scenario(conn, failure_class="terminal")
    negative = await scenario(conn)

    plan = await allocate(
        conn,
        [scored(terminal["record_id"], 5.0), scored(negative["record_id"], -2.0)],
    )

    for skip in plan.skipped:
        status, skip_reason = await status_of(conn, skip.record_id)
        assert status == RecordStatus.SKIPPED.value
        assert skip_reason == skip.skip_reason


async def test_the_persisted_scores_match_the_decision(conn):
    """The columns exist so a later reader can see what the record was scored
    at when it was acted on, without reconstructing the model."""
    ids = await scenario(conn)

    await allocate(conn, [scored(ids["record_id"], 4.25, cate=0.31)])

    row = await conn.fetchrow(
        "SELECT uplift_score, whittle_index FROM at_risk_records WHERE id = $1",
        ids["record_id"],
    )
    assert row["whittle_index"] == pytest.approx(4.25)
    assert row["uplift_score"] == pytest.approx(0.31)


# --- summary and refusals --------------------------------------------------


async def test_the_summary_counts_and_amounts_add_up(conn):
    good = await scenario(conn)
    bad = await scenario(conn)

    plan = await allocate(
        conn, [scored(good["record_id"], 5.0), scored(bad["record_id"], -5.0)]
    )

    summary = plan.summary
    assert summary["considered"] == 2
    assert summary["scheduled_count"] == 1
    assert summary["skipped_count"] == 1
    assert summary["scheduled_amount"] == 49900
    assert summary["skipped_amount"] == 49900
    assert summary["skips_by_reason"] == {SKIP_NEGATIVE_NET_VALUE: 1}


async def test_an_empty_batch_is_a_valid_cycle(conn):
    plan = await allocate(conn, [])

    assert plan.scheduled == []
    assert plan.skipped == []
    assert plan.summary["considered"] == 0


async def test_a_score_for_a_record_that_does_not_exist_is_refused(conn):
    """Scoring something the database has never heard of means the two sides
    disagree about what the portfolio is. Guessing is not an option."""
    with pytest.raises(LookupError):
        await allocate(conn, [scored(999_999_999, 4.0)])


async def test_allocate_refuses_to_run_outside_a_transaction(db_pool):
    """A half-committed cycle leaves records open that the next run would score
    against attempt counts that no longer describe reality."""
    async with db_pool.acquire() as connection:
        with pytest.raises(RuntimeError, match="transaction"):
            await allocate(connection, [scored(1, 4.0)])


# --- the unconstrained upper bound never touches anything ------------------


async def test_the_unconstrained_baseline_writes_nothing(conn):
    """It is a measuring stick, not a policy. If it could leave a trace, it
    could be promoted into the execution path by someone in a hurry."""
    ids = await scenario(conn)

    plan = await allocate_unconstrained(conn, [scored(ids["record_id"], -9.0)])

    status, skip_reason = await status_of(conn, ids["record_id"])
    assert status == RecordStatus.OPEN.value
    assert skip_reason is None
    assert await audit_actions(conn, ids["record_id"]) == []
    assert plan.summary["executable"] is False


async def test_the_unconstrained_baseline_chases_value_destroying_records(conn):
    """And says so. The gap between this plan and Forbear's is part compliance
    and part deliberate restraint; conflating them would overstate what the
    rules cost."""
    ids = await scenario(conn)

    plan = await allocate_unconstrained(conn, [scored(ids["record_id"], -9.0)])

    assert len(plan.scheduled) == 1
    assert plan.summary["negative_value_scheduled"] == 1
    assert plan.summary["negative_value_ltv_at_risk"] > 0


async def test_the_unconstrained_baseline_still_refuses_a_dead_mandate(conn):
    """The rails refusing is not a constraint we chose to impose."""
    ids = await scenario(conn, mandate_status="revoked")

    plan = await allocate_unconstrained(conn, [scored(ids["record_id"], 9.0)])

    assert plan.scheduled == []
    assert plan.skipped[0].skip_reason == SKIP_MANDATE_STATE_INVALID


async def test_the_unconstrained_baseline_ignores_the_attempt_cap(conn):
    ids = await scenario(conn)
    now = await server_now(conn)
    for number in range(1, MAX_ATTEMPTS + 1):
        await insert_attempt(
            conn, ids["record_id"], number, executed_at=now - timedelta(days=number)
        )

    plan = await allocate_unconstrained(conn, [scored(ids["record_id"], 4.0)])

    assert len(plan.scheduled) == 1


# --- the boundary ----------------------------------------------------------


def test_the_allocator_does_not_import_the_guard():
    """Invariant 2, enforced structurally. The allocator plans and the guard
    permits; two independent implementations of the same constraint catch each
    other's mistakes, and one shared helper catches nothing.
    """
    tree = ast.parse((REPO_ROOT / "forbear" / "services" / "allocator.py").read_text())

    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module or "")

    assert "forbear.core.guard" not in imported
    assert not any(name.endswith("guard") for name in imported)


def test_no_allocation_module_imports_the_generator():
    """Same rule as scoring: the answer key does not exist at decision time."""
    for name in ("allocator", "baseline", "unconstrained_baseline"):
        tree = ast.parse(
            (REPO_ROOT / "forbear" / "services" / f"{name}.py").read_text()
        )
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert not (node.module or "").startswith("forbear.generator")
