"""Turning a plan into outbound calls, one guarded action at a time.

The executor is the only place in Forbear where something irreversible
happens. Everything upstream produces opinions - scores, indices, plans - and
everything here produces money movement and messages to real people. So this
module is deliberately dull: take an action, ask the guard, do what it says,
record what happened, move on.

Two things are injected rather than imported, and both matter.

guard_fn is injected because the executor is not allowed to decide what is
permissible. It asks. A caller could pass a permissive stub - the unconstrained
upper bound does exactly that, on purpose - and that this is possible in a
measurement harness is not a hole, because the production wiring passes the
real guard and the audit entry records which verdict was acted on.

execute_fn is injected because "make the outbound call" means Razorpay in
production and a simulator in a harness, and the executor must not know the
difference. It is also what keeps invariant 1 intact from the other direction:
this module has no import path to the generator, so the ground truth cannot
reach the code that moves money even by accident.

ORDER OF OPERATIONS: the guard is asked before the attempt row is written, not
after. Writing the row first would put a pending attempt in front of the
guard's own duplicate-attempt rule, which then refuses the action that row was
created for - the executor would block every action it ever proposed. The
attempt is still consumed at execution and never at planning, which is the
property that keeps the cap honest.
"""

from __future__ import annotations

import inspect
import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, NamedTuple, Optional, Protocol

from forbear.core.audit import append_entry, server_now
from forbear.core.state_machine import ENTITY_TYPE, transition
from forbear.models.models import (
    ActionKind,
    AtRiskRecord,
    AttemptOutcome,
    ProposedAction,
    RecordStatus,
    at_risk_record_from_row,
)

ACTION_SUCCESS = "execution:success"
ACTION_FAILURE = "execution:failure"
ACTION_BLOCKED = "execution:blocked"
ACTION_NOT_ATTEMPTED = "execution:not_attempted"

# Statuses from which no attempt can be made. A record that recovered on its
# second attempt must not have its third fired at it, and the plan does not
# know that yet - it was written before any of this happened.
SETTLED_STATUSES = frozenset(
    {RecordStatus.RECOVERED, RecordStatus.ABANDONED, RecordStatus.SKIPPED}
)


class Clock(Protocol):
    """What the executor needs from time.

    Two implementations: one that reads the database and one that pretends. A
    simulation that could not move time forward would have to run for a month
    to observe a month, and a production path that could would be a way to lie
    to the guard about when it is.
    """

    async def now(self, conn) -> datetime: ...

    async def advance_to(self, moment: datetime) -> None: ...


class ServerClock:
    """Production. Invariant 3: the database is the only clock that counts."""

    async def now(self, conn) -> datetime:
        return await server_now(conn)

    async def advance_to(self, moment: datetime) -> None:
        """Real time does not take instruction. Deliberately a no-op."""


class VirtualClock:
    """Simulation. Moves forward only, so a plan cannot be replayed backwards.

    The guard reads its own clock, which is the point of the guard - so a
    caller who wants the guard evaluated against simulated time has to wire
    that itself. This clock governs what the executor writes down; making the
    guard agree with it is a separate and deliberate act.
    """

    def __init__(self, start: datetime) -> None:
        self._now = start

    async def now(self, conn) -> datetime:
        return self._now

    async def advance_to(self, moment: datetime) -> None:
        if moment > self._now:
            self._now = moment


class OutboundResult(NamedTuple):
    """What the injected execute_fn reports back.

    detail is stored in the audit entry verbatim: in production the gateway's
    response id, in simulation whatever the simulator wants a reader to know.
    """

    success: bool
    detail: dict[str, Any] = {}


class RecordOutcome(NamedTuple):
    record_id: int
    attempt_number: Optional[int]
    outcome: str
    scheduled_at: datetime
    executed_at: Optional[datetime]
    blocked_rule: Optional[str] = None
    amount: int = 0


@dataclass
class ExecutionReport:
    """What one pass over a plan did.

    attempted counts every action taken to the guard, which is why blocked
    actions are in it: the guard refusing is an outcome of trying, not a reason
    to pretend nothing happened. Actions dropped because the record had already
    settled are counted separately - nothing was tried there.
    """

    attempted: int = 0
    succeeded: int = 0
    failed: int = 0
    blocked_by_guard: int = 0
    not_attempted: int = 0
    blocked_by_rule: dict[str, int] = field(default_factory=dict)
    amount_recovered: int = 0
    amount_blocked: int = 0
    details: list[RecordOutcome] = field(default_factory=list)


async def _load_record(conn, record_id: int) -> Optional[AtRiskRecord]:
    row = await conn.fetchrow(
        """
        SELECT id, subscription_id, customer_id, invoice_id, amount,
               failure_code, failure_class, status, uplift_score, whittle_index,
               skip_reason, created_at, updated_at
        FROM at_risk_records
        WHERE id = $1
        """,
        record_id,
    )
    return at_risk_record_from_row(row) if row is not None else None


async def _next_attempt_number(conn, record_id: int) -> int:
    """One past the highest number on the record.

    Counting rows would repeat a number after a gap, and UNIQUE
    (at_risk_record_id, attempt_number) would then reject the insert - the
    constraint doing its job, but for a reason nobody would enjoy debugging.
    """
    highest = await conn.fetchval(
        "SELECT max(attempt_number) FROM attempts WHERE at_risk_record_id = $1",
        record_id,
    )
    return (highest or 0) + 1


async def _write_attempt(
    conn,
    record_id: int,
    attempt_number: int,
    scheduled_at: datetime,
    executed_at: datetime,
    outcome: AttemptOutcome,
    verdict_details: Optional[dict[str, Any]],
) -> int:
    return await conn.fetchval(
        """
        INSERT INTO attempts
            (at_risk_record_id, attempt_number, scheduled_at, executed_at,
             outcome, guard_verdict)
        VALUES ($1, $2, $3, $4, $5::attempt_outcome, $6::jsonb)
        RETURNING id
        """,
        record_id,
        attempt_number,
        scheduled_at,
        executed_at,
        outcome.value,
        json.dumps(verdict_details, default=str) if verdict_details else None,
    )


async def _call(execute_fn: Callable, record: AtRiskRecord, action) -> OutboundResult:
    """Run the injected outbound call, sync or async, and normalise the answer.

    A bare bool is accepted because a simulator has nothing useful to add and
    should not have to build a result object in order to say "no".
    """
    result = execute_fn(record, action)
    if inspect.isawaitable(result):
        result = await result

    if isinstance(result, OutboundResult):
        return result
    if isinstance(result, bool):
        return OutboundResult(success=result, detail={})
    raise TypeError(
        f"execute_fn must return OutboundResult or bool, got {type(result).__name__}"
    )


async def _ensure_schedulable(conn, record: AtRiskRecord) -> None:
    """Put the record into the one status an attempt can start from.

    A record whose previous attempt failed is back in open, and open cannot go
    straight to in_flight - the transition table routes it through scheduled,
    because "we are about to try this again" is a real intermediate fact and
    the audit chain should say so.
    """
    if record.status is RecordStatus.OPEN:
        await transition(conn, record.id, RecordStatus.SCHEDULED)


async def execute_plan(
    conn,
    plan,
    guard_fn: Callable,
    execute_fn: Callable,
    clock: Optional[Clock] = None,
) -> ExecutionReport:
    """Work through a plan's scheduled actions in order. Must be transactional.

    Actions run in slot order, not in the order the plan happens to list them.
    A plan is written ranked by value; time does not care about the ranking,
    and in production a scheduler fires each action when its moment arrives. A
    simulated clock only moves forward, so replaying a value-ordered plan
    verbatim would drag the clock past the slots of everything ranked below the
    first record - and every later action would be evaluated at the wrong time.

    A record that recovers partway through has its remaining actions dropped
    rather than fired.
    """
    if not conn.is_in_transaction():
        raise RuntimeError("execute_plan must run inside a transaction")

    clock = clock or ServerClock()
    report = ExecutionReport()

    chronological = sorted(
        plan.scheduled, key=lambda action: (action.scheduled_at, action.record_id)
    )

    for action in chronological:
        await clock.advance_to(action.scheduled_at)
        now = await clock.now(conn)

        record = await _load_record(conn, action.record_id)
        if record is None:
            raise LookupError(f"planned action for missing record {action.record_id}")

        # The plan was written before any of these attempts ran. A record that
        # has since recovered is done, and firing at it would be the system
        # charging someone who has already paid.
        if record.status in SETTLED_STATUSES:
            report.not_attempted += 1
            report.details.append(
                RecordOutcome(
                    record_id=record.id,
                    attempt_number=None,
                    outcome=ACTION_NOT_ATTEMPTED,
                    scheduled_at=action.scheduled_at,
                    executed_at=None,
                    amount=record.amount,
                )
            )
            await append_entry(
                conn,
                ENTITY_TYPE,
                record.id,
                ACTION_NOT_ATTEMPTED,
                {
                    "scheduled_at": action.scheduled_at.isoformat(),
                    "status": record.status.value,
                    "detail": "record already settled; remaining plan dropped",
                },
            )
            continue

        attempt_number = await _next_attempt_number(conn, record.id)
        proposed = ProposedAction(
            kind=action.action_kind,
            at_risk_record_id=record.id,
            attempt_number=attempt_number,
        )

        # Ask before doing. The guard re-reads every fact from the database and
        # owes this module nothing; its verdict is the only thing that
        # authorises what comes next.
        verdict = guard_fn(conn, record.id, proposed)
        if inspect.isawaitable(verdict):
            verdict = await verdict

        report.attempted += 1

        if not verdict.allowed:
            await _write_attempt(
                conn,
                record.id,
                attempt_number,
                action.scheduled_at,
                now,
                AttemptOutcome.BLOCKED_BY_GUARD,
                {
                    "allowed": False,
                    "rule_name": verdict.rule_name,
                    "details": verdict.details,
                },
            )
            await append_entry(
                conn,
                ENTITY_TYPE,
                record.id,
                ACTION_BLOCKED,
                {
                    "attempt_number": attempt_number,
                    "rule_name": verdict.rule_name,
                    "verdict": verdict.details,
                    "scheduled_at": action.scheduled_at.isoformat(),
                    "amount": record.amount,
                },
            )

            report.blocked_by_guard += 1
            report.amount_blocked += record.amount
            rule = verdict.rule_name or "unknown"
            report.blocked_by_rule[rule] = report.blocked_by_rule.get(rule, 0) + 1
            report.details.append(
                RecordOutcome(
                    record_id=record.id,
                    attempt_number=attempt_number,
                    outcome=AttemptOutcome.BLOCKED_BY_GUARD.value,
                    scheduled_at=action.scheduled_at,
                    executed_at=now,
                    blocked_rule=rule,
                    amount=record.amount,
                )
            )
            # The record keeps its plan. A block is the guard refusing this
            # action at this instant, not a verdict on the record.
            continue

        await _ensure_schedulable(conn, record)
        await transition(conn, record.id, RecordStatus.IN_FLIGHT)

        result = await _call(execute_fn, record, action)

        outcome = AttemptOutcome.SUCCESS if result.success else AttemptOutcome.FAILURE
        await _write_attempt(
            conn,
            record.id,
            attempt_number,
            action.scheduled_at,
            now,
            outcome,
            {"allowed": True, "rule_name": None, "details": verdict.details},
        )

        if result.success:
            await transition(conn, record.id, RecordStatus.RECOVERED)
            report.succeeded += 1
            report.amount_recovered += record.amount
        else:
            # Back to open: the attempt is spent but the record is not, and the
            # next cycle gets to decide whether it is still worth chasing.
            await transition(conn, record.id, RecordStatus.OPEN)
            report.failed += 1

        await append_entry(
            conn,
            ENTITY_TYPE,
            record.id,
            ACTION_SUCCESS if result.success else ACTION_FAILURE,
            {
                "attempt_number": attempt_number,
                "action_kind": action.action_kind.value,
                "scheduled_at": action.scheduled_at.isoformat(),
                "executed_at": now.isoformat(),
                "amount": record.amount,
                "outbound": result.detail,
            },
        )
        report.details.append(
            RecordOutcome(
                record_id=record.id,
                attempt_number=attempt_number,
                outcome=outcome.value,
                scheduled_at=action.scheduled_at,
                executed_at=now,
                amount=record.amount,
            )
        )

    return report


def allow_everything(conn, record_id: int, action: ProposedAction):
    """A guard-shaped stub that permits anything. NOT for production.

    It exists so the unconstrained upper bound can be computed at all: the
    number that says what the constraints cost is only meaningful if something
    can actually run without them. Named to be obvious in a stack trace and in
    a diff.
    """
    from forbear.core.guard import GuardVerdict  # local: nothing else needs it

    return GuardVerdict(
        allowed=True,
        rule_name=None,
        details={
            "stub": "allow_everything",
            "at_risk_record_id": record_id,
            "attempt_number": action.attempt_number,
            "action_kind": (
                action.kind.value
                if isinstance(action.kind, ActionKind)
                else str(action.kind)
            ),
        },
    )
