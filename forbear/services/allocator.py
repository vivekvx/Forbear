"""What to attempt, when, and what to leave alone.

The allocator turns a scored portfolio into a plan. It spends a shared attempt
budget on the records where an attempt changes the outcome, and declines to
spend on the rest - including records that would probably recover, because
"probably recovers anyway" is the definition of an attempt that buys nothing.

Nothing here executes. The plan is a list of intentions; the executor acts on
them and the guard re-validates every one immediately before it does. That
split is invariant 2, and it is why this module does not import the guard: two
independent implementations of the same constraint catch each other's bugs,
one shared helper catches nothing. The allocator therefore *considers*
constraints when picking a time slot - an unschedulable plan is a useless plan
- and *enforces* none of them. If a slot chosen here turns out to be illegal at
execution time, the guard blocks it, and that block is the system working
rather than failing.

The most important output of this module is the skip list. A skip is a
decision, and an audit entry recording only "negative_net_value" tells a
reviewer nothing they could check. Every skip written here carries the numbers
the decision was made from: the estimated effect, the money at stake, the
lifetime value at risk, the index, and the threshold it failed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta, timezone
from typing import Any, Iterable, NamedTuple, Optional

from forbear.config.limits import (
    IST,
    MAX_ATTEMPTS,
    NOTIFICATION_VALIDITY,
    NPCI_DEBIT_WINDOWS_IST,
)
from forbear.core.audit import append_entry, server_now
from forbear.core.state_machine import ENTITY_TYPE, transition
from forbear.models.models import ActionKind, FailureClass, MandateStatus, RecordStatus

# Skip reason codes. These are written to at_risk_records.skip_reason and read
# by anyone asking why the system did nothing, so they are a stable vocabulary
# rather than free text.
SKIP_TERMINAL_FAILURE_CLASS = "terminal_failure_class"
SKIP_MANDATE_STATE_INVALID = "mandate_state_invalid"
SKIP_UNCLASSIFIED_FAILURE_CODE = "unclassified_failure_code"
SKIP_NEGATIVE_NET_VALUE = "negative_net_value"
SKIP_ATTEMPT_BUDGET_EXHAUSTED = "attempt_budget_exhausted"
SKIP_BATCH_BUDGET_EXHAUSTED = "batch_budget_exhausted"
SKIP_NO_LEGAL_SLOT = "no_legal_slot"

ACTION_SCHEDULE = "allocation:schedule"
ACTION_SKIP = "allocation:skip"

# Failure classes an attempt cannot fix. Retrying a revoked mandate is not a
# long shot, it is a category error: no amount of waiting makes the debit legal.
UNRECOVERABLE_CLASSES = frozenset({FailureClass.TERMINAL, FailureClass.REAUTH_REQUIRED})

# Mandate states that cannot authorise a debit.
DEAD_MANDATE_STATES = frozenset({MandateStatus.REVOKED, MandateStatus.EXPIRED})

# Horizon used when reporting the lifetime value a skip was protecting. Matches
# the Whittle index's default so the number in the audit entry is the number the
# decision was actually made from.
LTV_HORIZON_MONTHS = 12


class ScheduledAction(NamedTuple):
    """(record_id, action_kind, scheduled_at) - a tuple, as the harness reads it."""

    record_id: int
    action_kind: ActionKind
    scheduled_at: datetime


class SkippedRecord(NamedTuple):
    """(record_id, skip_reason, details) - details is the reviewable part."""

    record_id: int
    skip_reason: str
    details: dict[str, Any]


@dataclass
class AllocationPlan:
    """One cycle's decisions. Every considered record appears exactly once."""

    scheduled: list[ScheduledAction] = field(default_factory=list)
    skipped: list[SkippedRecord] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ScoredRecord:
    """What the scoring modules hand over: an id and two numbers.

    Deliberately thin. Amounts, failure classes, mandate states and attempt
    counts are read from the database inside allocate() rather than accepted
    from the caller - they are money and lifecycle state, and a stale copy
    passed in from a scoring run half an hour ago is exactly how a system
    spends an attempt it no longer has.
    """

    record_id: int
    cate: float
    whittle_index: float


@dataclass(frozen=True)
class AllocationConfig:
    """Knobs the merchant sets per cycle.

    batch_budget is the ceiling on total attempts this run. None means no
    ceiling, which is the default: the per-record cap (MAX_ATTEMPTS) still
    applies and is the limit that actually protects the customer. The batch
    budget protects the merchant's retry volume, which is a business decision
    rather than a regulatory one.
    """

    batch_budget: Optional[int] = None
    horizon_days: int = 7
    # How far ahead to schedule when no valid pre-debit notification exists
    # yet. The executor has to send one and the customer has to have a chance
    # to see it, so the next day is the earliest honest slot.
    notification_lead: timedelta = NOTIFICATION_VALIDITY
    minimum_index: float = 0.0


@dataclass(frozen=True)
class _RecordFacts:
    """The database's view of a record, read once per cycle."""

    record_id: int
    customer_id: int
    subscription_id: int
    amount: int
    plan_amount: int
    failure_class: Optional[FailureClass]
    mandate_status: MandateStatus
    status: RecordStatus
    attempts_so_far: int
    last_notification_at: Optional[datetime]


async def _load_facts(conn, record_ids: list[int]) -> dict[int, _RecordFacts]:
    """One query for the whole batch. Explicit SQL: this touches attempt counts.

    The attempt count comes from the attempts table rather than a column,
    because a counter column can drift from the rows it counts, and the rows
    are what the regulatory cap is defined over.
    """
    rows = await conn.fetch(
        """
        SELECT r.id,
               r.customer_id,
               r.subscription_id,
               r.amount,
               r.failure_class,
               r.status,
               s.plan_amount,
               s.mandate_status,
               COALESCE(counted.attempts, 0) AS attempts_so_far,
               notified.last_notification_at
        FROM unnest($1::bigint[]) AS wanted(record_id)
        JOIN at_risk_records r ON r.id = wanted.record_id
        JOIN subscriptions s ON s.id = r.subscription_id
        LEFT JOIN LATERAL (
            SELECT count(*) AS attempts
            FROM attempts a
            WHERE a.at_risk_record_id = r.id
        ) counted ON TRUE
        LEFT JOIN LATERAL (
            SELECT max(c.sent_at) AS last_notification_at
            FROM contacts c
            WHERE c.subscription_id = r.subscription_id
              AND c.purpose = 'pre_debit_notification'
        ) notified ON TRUE
        """,
        record_ids,
    )
    return {
        row["id"]: _RecordFacts(
            record_id=row["id"],
            customer_id=row["customer_id"],
            subscription_id=row["subscription_id"],
            amount=row["amount"],
            plan_amount=row["plan_amount"],
            failure_class=(
                FailureClass(row["failure_class"])
                if row["failure_class"] is not None
                else None
            ),
            mandate_status=MandateStatus(row["mandate_status"]),
            status=RecordStatus(row["status"]),
            attempts_so_far=row["attempts_so_far"],
            last_notification_at=row["last_notification_at"],
        )
        for row in rows
    }


async def _success_patterns(
    conn, customer_ids: list[int]
) -> dict[int, tuple[frozenset[int], Optional[int]]]:
    """When each customer's debits have actually worked before.

    Two signals come out of the same history: the IST hours their money was
    there, and the day of month those successes cluster on - a salary day,
    inferred from behaviour rather than declared. A customer with no successful
    debit has neither, and gets scheduled on earliness alone.
    """
    # Joined against unnest rather than filtered with `= ANY($1)`. With ten
    # thousand ids in the array and no statistics on a freshly loaded table,
    # the planner reads ANY() as a filter it can satisfy by nested loop, and
    # the query stops finishing: a cycle that took seconds at five hundred
    # records ran for over an hour at ten thousand. As a join, the ids are a
    # relation with a known cardinality and the plan is a hash join at any
    # size.
    rows = await conn.fetch(
        """
        SELECT r.customer_id, a.executed_at
        FROM unnest($1::bigint[]) AS wanted(customer_id)
        JOIN at_risk_records r ON r.customer_id = wanted.customer_id
        JOIN attempts a ON a.at_risk_record_id = r.id
        WHERE a.outcome = 'success'
          AND a.executed_at IS NOT NULL
        """,
        customer_ids,
    )

    hours: dict[int, set[int]] = {}
    days: dict[int, list[int]] = {}
    for row in rows:
        moment = row["executed_at"].astimezone(IST)
        hours.setdefault(row["customer_id"], set()).add(moment.hour)
        days.setdefault(row["customer_id"], []).append(moment.day)

    patterns: dict[int, tuple[frozenset[int], Optional[int]]] = {}
    for customer_id in set(hours) | set(days):
        observed = days.get(customer_id, [])
        # The most common day, ties broken by the earliest. A mode over two or
        # three observations is a hint, not an estimate, and it is used only to
        # order otherwise-equal slots.
        salary_day = (
            min(observed, key=lambda day: (-observed.count(day), day))
            if observed
            else None
        )
        patterns[customer_id] = (frozenset(hours.get(customer_id, ())), salary_day)
    return patterns


def _day_distance(day: int, salary_day: Optional[int]) -> int:
    """Days between a candidate slot and the inferred salary day, wrapping.

    Wrapping matters: the 29th is two days from the 1st, not twenty-eight.
    """
    if salary_day is None:
        return 0
    raw = abs(day - salary_day)
    return min(raw, 30 - raw)


def _slot_candidates(
    earliest: datetime, horizon_days: int
) -> Iterable[tuple[datetime, datetime]]:
    """Every whole hour inside an NPCI debit window, out to the horizon.

    Yields (utc_instant, ist_instant). The windows are read from config rather
    than restated here, so the allocator and the guard work from the same
    published boundaries even though they never share code.
    """
    first_day = earliest.astimezone(IST).date()
    for day_offset in range(horizon_days + 1):
        day: date = first_day + timedelta(days=day_offset)
        for window_start, window_end in NPCI_DEBIT_WINDOWS_IST:
            for minute in range(window_start, window_end, 60):
                slot_ist = datetime.combine(
                    day, time(minute // 60, minute % 60), tzinfo=IST
                )
                yield slot_ist.astimezone(timezone.utc), slot_ist


def _choose_slot(
    earliest: datetime,
    deadline: Optional[datetime],
    success_hours: frozenset[int],
    salary_day: Optional[int],
    horizon_days: int,
) -> Optional[datetime]:
    """The best legal slot, or None if the record cannot be scheduled at all.

    Priority, in order: an hour this customer has actually paid at before, then
    proximity to their inferred salary day, then the earliest such slot. The
    ordering encodes what the money is doing - a debit at an hour the balance
    has historically been there beats a debit two days sooner into an empty
    account.

    deadline is the far edge of an existing notification's validity. Scheduling
    past it would produce a plan the guard refuses on the notification rule.
    """
    best: Optional[datetime] = None
    best_key: Optional[tuple[int, int, datetime]] = None

    for slot_utc, slot_ist in _slot_candidates(earliest, horizon_days):
        if slot_utc < earliest:
            continue
        if deadline is not None and slot_utc > deadline:
            continue

        key = (
            0 if slot_ist.hour in success_hours else 1,
            _day_distance(slot_ist.day, salary_day),
            slot_utc,
        )
        if best_key is None or key < best_key:
            best_key, best = key, slot_utc

    return best


def ltv_at_risk(plan_amount: int, remaining_months: int = LTV_HORIZON_MONTHS) -> int:
    """The subscription behind the invoice, over the standard horizon.

    Reported in skip details so a reviewer can see what was being protected,
    not merely that something was.
    """
    return plan_amount * remaining_months


async def _record_skip(
    conn,
    plan: AllocationPlan,
    facts: _RecordFacts,
    reason: str,
    details: dict[str, Any],
) -> None:
    """Move a record to skipped and write the reasoning behind it.

    Two audit entries end up on the chain: the state machine's own transition
    entry, which records that the status changed, and this one, which records
    why. The second is the one a reviewer actually needs - a status change with
    a reason code is a fact, and the numbers underneath it are the argument.
    """
    await transition(conn, facts.record_id, RecordStatus.SKIPPED, reason=reason)
    await append_entry(
        conn,
        ENTITY_TYPE,
        facts.record_id,
        ACTION_SKIP,
        {"skip_reason": reason, **details},
    )
    plan.skipped.append(SkippedRecord(facts.record_id, reason, details))


async def _record_schedule(
    conn,
    plan: AllocationPlan,
    facts: _RecordFacts,
    scored: ScoredRecord,
    slot: datetime,
    reasoning: dict[str, Any],
) -> None:
    """Move a record to scheduled and record what was decided and why.

    No attempts row is written here. The attempt is what the executor consumes
    against the cap, and creating it now would mean the cap was spent by a plan
    rather than by an action - exactly the accounting error the attempts table
    exists to prevent.
    """
    await transition(conn, facts.record_id, RecordStatus.SCHEDULED)
    await conn.execute(
        """
        UPDATE at_risk_records
        SET uplift_score = $2, whittle_index = $3
        WHERE id = $1
        """,
        facts.record_id,
        scored.cate,
        scored.whittle_index,
    )
    await append_entry(
        conn,
        ENTITY_TYPE,
        facts.record_id,
        ACTION_SCHEDULE,
        {
            "action_kind": ActionKind.CHARGE.value,
            "scheduled_at": slot.isoformat(),
            "cate": scored.cate,
            "whittle_index": scored.whittle_index,
            "amount": facts.amount,
            "attempts_so_far": facts.attempts_so_far,
            **reasoning,
        },
    )
    plan.scheduled.append(ScheduledAction(facts.record_id, ActionKind.CHARGE, slot))


async def allocate(
    conn,
    records_with_scores: list[ScoredRecord],
    config: Optional[AllocationConfig] = None,
) -> AllocationPlan:
    """Plan one cycle. Must run inside a transaction.

    The transaction requirement is not incidental: a cycle that scheduled half
    its records and then failed would leave the other half open, and the next
    run would score them against attempt counts that no longer describe
    reality. All of it commits or none of it does.

    The filters run in a fixed order, and the order is the explanation. A
    revoked mandate is skipped for the mandate, not for its index, because "we
    did not chase this because the index was low" would be a false account of a
    record that could not have been charged at all.
    """
    if not conn.is_in_transaction():
        raise RuntimeError("allocate must run inside a transaction")

    config = config or AllocationConfig()
    plan = AllocationPlan()
    scheduled_amount = 0

    if not records_with_scores:
        plan.summary = _summarise(plan, config, considered=0, scheduled_amount=0)
        return plan

    scores = {scored.record_id: scored for scored in records_with_scores}
    facts_by_id = await _load_facts(conn, list(scores))

    missing = sorted(set(scores) - set(facts_by_id))
    if missing:
        raise LookupError(f"scored records that do not exist: {missing}")

    patterns = await _success_patterns(
        conn, sorted({facts.customer_id for facts in facts_by_id.values()})
    )
    now = await server_now(conn)

    eligible: list[tuple[ScoredRecord, _RecordFacts]] = []

    for record_id, scored in scores.items():
        facts = facts_by_id[record_id]

        # (a) Nothing an attempt can fix.
        if facts.failure_class is None:
            await _record_skip(
                conn,
                plan,
                facts,
                SKIP_UNCLASSIFIED_FAILURE_CODE,
                {
                    "amount": facts.amount,
                    "cate": scored.cate,
                    "whittle_index": scored.whittle_index,
                    "detail": "no classifier mapping; on the exception list",
                },
            )
            continue

        if facts.failure_class in UNRECOVERABLE_CLASSES:
            await _record_skip(
                conn,
                plan,
                facts,
                SKIP_TERMINAL_FAILURE_CLASS,
                {
                    "failure_class": facts.failure_class.value,
                    "amount": facts.amount,
                    "ltv_at_risk": ltv_at_risk(facts.plan_amount),
                    "cate": scored.cate,
                    "whittle_index": scored.whittle_index,
                    "detail": "no attempt can recover this failure class",
                },
            )
            continue

        if facts.mandate_status in DEAD_MANDATE_STATES:
            await _record_skip(
                conn,
                plan,
                facts,
                SKIP_MANDATE_STATE_INVALID,
                {
                    "mandate_status": facts.mandate_status.value,
                    "amount": facts.amount,
                    "ltv_at_risk": ltv_at_risk(facts.plan_amount),
                    "cate": scored.cate,
                    "whittle_index": scored.whittle_index,
                    "detail": "mandate cannot authorise a debit",
                },
            )
            continue

        # (b) Chasing costs more than it recovers. The numbers below are the
        # whole point of the entry: a reviewer must be able to see the trade
        # that was made, disagree with it, and say which term was wrong.
        if scored.whittle_index < config.minimum_index:
            await _record_skip(
                conn,
                plan,
                facts,
                SKIP_NEGATIVE_NET_VALUE,
                {
                    "cate": scored.cate,
                    "amount": facts.amount,
                    "ltv_at_risk": ltv_at_risk(facts.plan_amount),
                    "whittle_index": scored.whittle_index,
                    "threshold": config.minimum_index,
                    "detail": (
                        "contacting this customer is estimated to destroy more "
                        "value than it recovers"
                    ),
                },
            )
            continue

        # (c) The cap is a hard limit (invariant 5), counted off attempt rows
        # rather than trusted from anywhere.
        if facts.attempts_so_far >= MAX_ATTEMPTS:
            await _record_skip(
                conn,
                plan,
                facts,
                SKIP_ATTEMPT_BUDGET_EXHAUSTED,
                {
                    "attempts_so_far": facts.attempts_so_far,
                    "max_attempts": MAX_ATTEMPTS,
                    "amount": facts.amount,
                    "ltv_at_risk": ltv_at_risk(facts.plan_amount),
                    "cate": scored.cate,
                    "whittle_index": scored.whittle_index,
                },
            )
            continue

        eligible.append((scored, facts))

    # (d) Highest value per attempt first. Ties broken by record id so a cycle
    # is reproducible rather than dependent on dictionary order.
    eligible.sort(key=lambda pair: (-pair[0].whittle_index, pair[0].record_id))

    # (e, f) Greedy walk down the ranking until the budget runs out.
    for scored, facts in eligible:
        over_budget = (
            config.batch_budget is not None
            and len(plan.scheduled) >= config.batch_budget
        )
        if over_budget:
            await _record_skip(
                conn,
                plan,
                facts,
                SKIP_BATCH_BUDGET_EXHAUSTED,
                {
                    "batch_budget": config.batch_budget,
                    "scheduled_before_this": len(plan.scheduled),
                    "cate": scored.cate,
                    "amount": facts.amount,
                    "ltv_at_risk": ltv_at_risk(facts.plan_amount),
                    "whittle_index": scored.whittle_index,
                    "detail": (
                        "eligible and positive-value, but the cycle's attempt "
                        "ceiling was already spent on higher-index records"
                    ),
                },
            )
            continue

        success_hours, salary_day = patterns.get(facts.customer_id, (frozenset(), None))

        # A notification already sent authorises a debit only until it expires;
        # without one, the executor has to send one first and the customer needs
        # the day the regulation gives them.
        notification_age = (
            now - facts.last_notification_at
            if facts.last_notification_at is not None
            else None
        )
        has_live_notification = (
            notification_age is not None
            and timedelta(0) <= notification_age <= NOTIFICATION_VALIDITY
        )
        if has_live_notification:
            earliest = now
            deadline = facts.last_notification_at + NOTIFICATION_VALIDITY
        else:
            earliest = now + config.notification_lead
            deadline = None

        slot = _choose_slot(
            earliest=earliest,
            deadline=deadline,
            success_hours=success_hours,
            salary_day=salary_day,
            horizon_days=config.horizon_days,
        )

        if slot is None:
            # Every legal window inside the horizon is unreachable - normally a
            # notification about to expire. Left for the next cycle rather than
            # scheduled into a slot the guard would refuse.
            await _record_skip(
                conn,
                plan,
                facts,
                SKIP_NO_LEGAL_SLOT,
                {
                    "earliest_considered": earliest.isoformat(),
                    "deadline": deadline.isoformat() if deadline else None,
                    "horizon_days": config.horizon_days,
                    "cate": scored.cate,
                    "amount": facts.amount,
                    "ltv_at_risk": ltv_at_risk(facts.plan_amount),
                    "whittle_index": scored.whittle_index,
                },
            )
            continue

        await _record_schedule(
            conn,
            plan,
            facts,
            scored,
            slot,
            {
                "chosen_for": {
                    "matched_prior_success_hour": slot.astimezone(IST).hour
                    in success_hours,
                    "inferred_salary_day": salary_day,
                    "notification_live": has_live_notification,
                }
            },
        )
        scheduled_amount += facts.amount

    plan.summary = _summarise(
        plan,
        config,
        considered=len(records_with_scores),
        scheduled_amount=scheduled_amount,
    )
    return plan


def _summarise(
    plan: AllocationPlan,
    config: AllocationConfig,
    considered: int,
    scheduled_amount: int,
) -> dict[str, Any]:
    """Counts and amounts, for the cycle report and the measurement harness.

    skipped_amount is the number that makes the thesis checkable: it is the
    money Forbear deliberately did not chase, and it should be large enough
    that someone asks about it.
    """
    skips_by_reason: dict[str, int] = {}
    skipped_amount = 0
    for skip in plan.skipped:
        skips_by_reason[skip.skip_reason] = skips_by_reason.get(skip.skip_reason, 0) + 1
        skipped_amount += int(skip.details.get("amount", 0))

    return {
        "considered": considered,
        "scheduled_count": len(plan.scheduled),
        "skipped_count": len(plan.skipped),
        "scheduled_amount": scheduled_amount,
        "skipped_amount": skipped_amount,
        "skips_by_reason": skips_by_reason,
        "batch_budget": config.batch_budget,
    }
