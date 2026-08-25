"""Three policies, one book of customers, one comparison table.

This is where the claim gets tested. Forbear says it recovers less money than
chasing everyone and is worth more anyway, because the money it declines to
chase is cheaper than the customers it declines to lose. That is a falsifiable
statement, and the only way to find out is to run the same customers through
three policies and count.

The three worlds are separate copies of the same book, inserted independently,
so each policy allocates against records nobody else has touched. Same
customers, same segments, same luck: the noise draws are computed once and
shared, so a difference in the table is a difference in policy rather than a
difference in dice.

WHAT MAKES THIS HONEST, AND WHAT DOES NOT
-----------------------------------------
Honest: treatment was randomised when the training outcomes were generated, so
the uplift model learned from an experiment rather than from a policy's own
choices. The executor never sees ground truth - the simulator is injected, and
what it returns is the only thing the executor knows.

Less than honest, and stated rather than hidden: the model is fitted on the
same batch it then scores. Out-of-sample scoring would shrink the estimates
further and is what a production pipeline would do. In-sample is what the
specified sequence asks for, and it flatters the model's precision - though not
the direction of the result, which is driven by segment structure the model
gets roughly right either way.

The simulation seam is the guard's clock. Attempts execute at simulated
instants weeks ahead of wall time, and the guard reads its own clock on
purpose, so the harness swaps that read for the length of a run. The swap is
simulation-only and lives here rather than in the guard, where a
caller-supplied time would be a way to talk the NPCI windows into anything.
"""

from __future__ import annotations

import contextlib
import statistics
import time
import uuid
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any, Callable, Optional, Sequence

import numpy as np

from forbear.config.limits import NOTIFICATION_MIN_LEAD
from forbear.core import guard
from forbear.core.audit import server_now
from forbear.generator.batch_generator import FailedDebit, generate_batch
from forbear.generator.customer_profiles import CustomerProfile, generate_profiles
from forbear.generator.outcome_simulator import (
    LOST_CAUSE_RECOVERY_RATE,
    SURE_THING_FAILURE_RATE,
    simulate_outcomes,
)
from forbear.generator.treatment_assignment import assign_treatment
from forbear.scoring.uplift import (
    FeatureRow,
    UpliftEvaluation,
    UpliftModel,
    build_feature_matrix,
)
from forbear.scoring.whittle import RecordScore, compute_indices
from forbear.services import baseline, unconstrained_baseline
from forbear.services.allocator import (
    AllocationConfig,
    ScheduledAction,
    ScoredRecord,
    allocate,
)
from forbear.services.classifier import classify
from forbear.services.unconstrained_baseline import UNCHARGEABLE_CLASSES
from forbear.services.executor import (
    OutboundResult,
    VirtualClock,
    allow_everything,
    execute_plan,
)

STRATEGY_FIXED = "fixed_schedule"
STRATEGY_FORBEAR = "forbear"
STRATEGY_FORBEAR_CONSTRAINED = "forbear_constrained"
STRATEGY_CLASSIFIER_ONLY = "classifier_only"
STRATEGY_UNCONSTRAINED = "unconstrained"
STRATEGIES = (
    STRATEGY_FIXED,
    STRATEGY_FORBEAR,
    STRATEGY_FORBEAR_CONSTRAINED,
    STRATEGY_CLASSIFIER_ONLY,
    STRATEGY_UNCONSTRAINED,
)

# The three policies that plan through Forbear's own allocator, and therefore
# pay Forbear's compliance cost: notice periods, NPCI windows, cooldowns, the
# attempt cap. The two baselines do not, which is the conservative direction -
# see the note at the guard dispatch in run_comparison.
FORBEAR_FAMILY = (
    STRATEGY_FORBEAR,
    STRATEGY_FORBEAR_CONSTRAINED,
    STRATEGY_CLASSIFIER_ONLY,
)

# How far ahead of each slot the simulated sender posts the pre-debit notice.
# An hour clear of the regulatory minimum, matching what the allocator plans
# for, so a slot is not refused over the boundary the guard compares strictly.
NOTIFICATION_STUB_LEAD = NOTIFICATION_MIN_LEAD + timedelta(hours=1)

# Hours a debit has historically succeeded at, all inside NPCI windows. Real
# books have this pattern; the allocator reads it back out of attempt history
# to work out when a customer's balance is actually there.
HISTORY_HOURS = (9, 14, 22)


@dataclass(frozen=True)
class HarnessConfig:
    treatment_rate: float = 0.5
    batch_budget: Optional[int] = None
    # Share of the book forbear_constrained is allowed to attempt. The plain
    # forbear strategy runs with no batch ceiling at all, which means its
    # Whittle index only ever decides sign - everything above the threshold is
    # scheduled, so the ranking never has to choose between two records it
    # wants. This ratio makes the constraint bind, which is the only condition
    # under which an index-based allocator is doing anything a filter could
    # not.
    constrained_budget_ratio: float = 0.3
    # Long enough to reach the next salary day. A horizon shorter than a
    # billing cycle would stop the allocator waiting for payday, which is the
    # behaviour under test.
    horizon_days: int = 35
    remaining_months: int = 12
    recovery_survival_rate: float = 0.85
    attempt_cost_paise: int = 200
    contact_sensitivity: Optional[float] = None
    # How often the unconstrained policy retries. It ignores the cap, so it
    # retries until the money lands or the horizon ends. Weekly rather than
    # daily because the measurement is whether the money came back, not which
    # day it arrived, and daily retries cost a great deal of simulation for an
    # answer that moves by rounding.
    unconstrained_retry_days: int = 7
    # Historical invoices per customer, each with a successful debit on their
    # salary day. This is what a real book looks like after a year, and it is
    # what the allocator infers a salary day from - with no history it has
    # nothing to time an attempt against.
    history_invoices_per_customer: int = 2


@dataclass
class StrategyResult:
    records_processed: int = 0
    amount_at_risk: int = 0
    amount_recovered: int = 0
    recovery_rate: float = 0.0
    attempts_consumed: int = 0
    recovered_per_attempt: float = 0.0
    records_skipped: int = 0
    records_blocked: int = 0
    ltv_at_risk_from_skips: int = 0
    churned_count: int = 0
    churn_rate: float = 0.0
    ltv_lost_to_churn: int = 0
    net_value: int = 0
    # Wall time for this policy's planning and execution, so a scale run can
    # report throughput per strategy rather than dividing one total three ways
    # and calling the result a measurement.
    elapsed_seconds: float = 0.0
    blocked_by_rule: dict[str, int] = field(default_factory=dict)
    skips_by_reason: dict[str, int] = field(default_factory=dict)


class ComparisonResult(dict):
    """Strategy name -> StrategyResult, with the model evaluation attached.

    A dict subclass rather than a wrapper, because every caller indexes this by
    strategy name and none of them should have to change. The uplift evaluation
    is a property of the comparison rather than of any one policy, so it rides
    alongside instead of being copied into five identical rows.
    """

    uplift: Optional[UpliftEvaluation] = None


@dataclass(frozen=True)
class _Noise:
    """One customer's luck, drawn once and shared by all three policies."""

    sure_thing_fails: bool
    lost_cause_pays: bool


@dataclass
class _World:
    """One strategy's private copy of the book."""

    record_ids: list[int]
    customer_of_record: dict[int, str]
    record_of_customer: dict[str, int]
    subscription_of_record: dict[int, int]
    customer_row_of_record: dict[int, int]


async def one_transaction_record_limit(conn) -> int:
    """How large a book this database can compare inside one transaction.

    Every audit append takes a transaction-scoped advisory lock on its entity,
    which is what stops two writers forking the same hash chain. The locks are
    held until the transaction ends, and a comparison runs three worlds inside
    one - so the lock table needs roughly three entries per record, and a
    default PostgreSQL holds about six thousand in total.

    Production is not subject to this: each decision commits on its own and
    releases its lock immediately. The ceiling belongs to measuring an entire
    cycle atomically, which is a property of the harness and not of the system
    it measures. Callers use this to say so plainly rather than discovering it
    twenty minutes into a run as "out of shared memory".
    """
    per_transaction = int(await conn.fetchval("SHOW max_locks_per_transaction"))
    connections = int(await conn.fetchval("SHOW max_connections"))
    prepared = int(await conn.fetchval("SHOW max_prepared_transactions"))

    # 0.8 leaves room for the ordinary relation and row locks taken alongside.
    capacity = per_transaction * (connections + prepared) * 0.8
    return int(capacity // len(STRATEGIES))


@contextlib.contextmanager
def simulated_guard_clock(clock: VirtualClock):
    """Point the guard's clock at the simulation, for the length of a run.

    The guard reads time itself and takes no argument for it, which is the
    correct design: a rule that can be told what time it is can be told the
    wrong time, and the NPCI windows are exactly what someone in a hurry would
    want to argue their way past.

    Simulation needs the guard evaluated against the simulated instant, so the
    harness swaps that read and puts it back. This is the only place in the
    codebase that does it, nothing in the API layer imports it, and it restores
    on the way out even if the run raises.
    """
    original = guard._server_now

    async def _clock_read(conn) -> datetime:
        return await clock.now(conn)

    guard._server_now = _clock_read
    try:
        yield guard.guard_check
    finally:
        guard._server_now = original


async def _insert_world(
    conn,
    strategy: str,
    profiles: list[CustomerProfile],
    batch: list[FailedDebit],
    now: datetime,
    config: HarnessConfig,
) -> _World:
    """Insert one strategy's copy of the book, history included.

    Bulk inserts keyed on external id rather than row by row: three worlds of a
    few thousand rows each is enough round trips to dominate the run otherwise.
    Nothing here trusts insertion order - every mapping is rebuilt from the
    external ids that came back.
    """
    # A token unique to this call. The sweep runs the whole comparison once per
    # churn rate against one connection, so external ids that were only unique
    # per strategy would collide on the second rate - and the collision would
    # surface as a UNIQUE violation halfway through a twenty-minute sweep.
    run = uuid.uuid4().hex[:8]

    def external(*parts: str) -> str:
        return ":".join((run, strategy, *parts))

    customer_external = {p.customer_id: external("cust", p.customer_id) for p in profiles}
    customer_rows = await conn.fetch(
        """
        INSERT INTO customers (external_id)
        SELECT ext FROM unnest($1::text[]) AS t(ext)
        RETURNING id, external_id
        """,
        [customer_external[p.customer_id] for p in profiles],
    )
    id_by_external = {row["external_id"]: row["id"] for row in customer_rows}
    customer_id_by_external = {
        customer: id_by_external[ext] for customer, ext in customer_external.items()
    }

    subscription_rows = await conn.fetch(
        """
        INSERT INTO subscriptions
            (customer_id, external_id, plan_amount, billing_cycle_days,
             mandate_status)
        SELECT customer_id, ext, plan_amount, 30, status::mandate_status
        FROM unnest($1::bigint[], $2::text[], $3::bigint[], $4::text[])
             AS t(customer_id, ext, plan_amount, status)
        RETURNING id, external_id
        """,
        [customer_id_by_external[p.customer_id] for p in profiles],
        [external("sub", p.customer_id) for p in profiles],
        [p.plan_amount for p in profiles],
        [p.mandate_status.value for p in profiles],
    )
    subscription_row_id = {row["external_id"]: row["id"] for row in subscription_rows}
    subscription_id_by_external = {
        p.customer_id: subscription_row_id[external("sub", p.customer_id)]
        for p in profiles
    }

    # The at-risk records this cycle actually decides about.
    record_rows = await conn.fetch(
        """
        INSERT INTO at_risk_records
            (subscription_id, customer_id, invoice_id, amount, failure_code,
             failure_class, status)
        SELECT subscription_id, customer_id, invoice_id, amount, failure_code,
               failure_class::failure_class, 'open'::record_status
        FROM unnest($1::bigint[], $2::bigint[], $3::text[], $4::bigint[],
                    $5::text[], $6::text[])
             AS t(subscription_id, customer_id, invoice_id, amount,
                  failure_code, failure_class)
        RETURNING id, invoice_id
        """,
        [subscription_id_by_external[r.customer_id] for r in batch],
        [customer_id_by_external[r.customer_id] for r in batch],
        [external("inv", r.customer_id) for r in batch],
        [r.amount for r in batch],
        [r.failure_code for r in batch],
        [classify(r.failure_code).value for r in batch],
    )
    record_row_id = {row["invoice_id"]: row["id"] for row in record_rows}
    record_of_customer = {
        r.customer_id: record_row_id[external("inv", r.customer_id)] for r in batch
    }

    # Statistics before anything plans against these rows. A bulk load inside
    # an open transaction leaves the planner with no idea how big these tables
    # are, and it will cheerfully pick a nested loop over ten thousand
    # customers. ANALYZE costs a second here and is the difference between a
    # cycle that finishes and one that does not.
    await conn.execute("ANALYZE customers, subscriptions, at_risk_records")

    await _insert_history(
        conn,
        external,
        profiles,
        customer_id_by_external,
        subscription_id_by_external,
        now,
        config,
    )

    return _World(
        record_ids=[record_of_customer[r.customer_id] for r in batch],
        customer_of_record={
            record_id: customer for customer, record_id in record_of_customer.items()
        },
        record_of_customer=record_of_customer,
        subscription_of_record={
            record_of_customer[r.customer_id]: subscription_id_by_external[
                r.customer_id
            ]
            for r in batch
        },
        customer_row_of_record={
            record_of_customer[r.customer_id]: customer_id_by_external[r.customer_id]
            for r in batch
        },
    )


async def _insert_history(
    conn,
    external: Callable[..., str],
    profiles: list[CustomerProfile],
    customer_id_by_external: dict[str, int],
    subscription_id_by_external: dict[str, int],
    now: datetime,
    config: HarnessConfig,
) -> None:
    """Settled invoices from earlier cycles, each paid on the customer's day.

    Without this the allocator has no observed history to infer a salary day
    from, and every record gets scheduled at the earliest legal slot - which for
    a time-dependent failure is precisely the wrong answer, because the money is
    not there yet. A book with no past is not a harder problem, it is a
    different one, and it is not the one Forbear is built for.

    These live on their own at_risk_records, so they add history without
    consuming any of the current record's attempt cap.
    """
    if config.history_invoices_per_customer <= 0:
        return

    invoice_ids: list[str] = []
    subscription_ids: list[int] = []
    customer_ids: list[int] = []
    plan_amounts: list[int] = []
    paid_at: list[datetime] = []

    for profile in profiles:
        hour = HISTORY_HOURS[profile.salary_day % len(HISTORY_HOURS)]
        for cycle in range(1, config.history_invoices_per_customer + 1):
            settled = (now - timedelta(days=30 * cycle)).replace(
                hour=hour, minute=0, second=0, microsecond=0
            )
            # Land it on the customer's salary day, and never in the future.
            settled += timedelta(days=profile.salary_day - settled.day)
            while settled >= now - timedelta(days=2):
                settled -= timedelta(days=30)

            invoice_ids.append(external("hist", profile.customer_id, str(cycle)))
            subscription_ids.append(subscription_id_by_external[profile.customer_id])
            customer_ids.append(customer_id_by_external[profile.customer_id])
            plan_amounts.append(profile.plan_amount)
            paid_at.append(settled)

    history_rows = await conn.fetch(
        """
        INSERT INTO at_risk_records
            (subscription_id, customer_id, invoice_id, amount, failure_code,
             failure_class, status)
        SELECT subscription_id, customer_id, invoice_id, amount,
               'INSUFFICIENT_FUNDS', 'time_dependent'::failure_class,
               'recovered'::record_status
        FROM unnest($1::bigint[], $2::bigint[], $3::text[], $4::bigint[])
             AS t(subscription_id, customer_id, invoice_id, amount)
        RETURNING id, invoice_id
        """,
        subscription_ids,
        customer_ids,
        invoice_ids,
        plan_amounts,
    )
    history_id_by_invoice = {row["invoice_id"]: row["id"] for row in history_rows}

    await conn.execute(
        """
        INSERT INTO attempts
            (at_risk_record_id, attempt_number, scheduled_at, executed_at,
             outcome)
        SELECT record_id, 1, moment, moment, 'success'::attempt_outcome
        FROM unnest($1::bigint[], $2::timestamptz[]) AS t(record_id, moment)
        """,
        [history_id_by_invoice[invoice] for invoice in invoice_ids],
        paid_at,
    )

    # The attempt history is what the allocator infers salary days from, and it
    # is the largest table in the world it just built.
    await conn.execute("ANALYZE at_risk_records, attempts")


def _feature_rows(
    profiles: list[CustomerProfile],
    batch: list[FailedDebit],
    rng: np.random.RandomState,
) -> list[FeatureRow]:
    """The observable half of each record. No segment, no counterfactual."""
    profile_by_customer = {profile.customer_id: profile for profile in profiles}
    rows = []
    for record in batch:
        profile = profile_by_customer[record.customer_id]
        contacted_before = rng.random_sample() < 0.4
        rows.append(
            FeatureRow(
                plan_amount=record.amount,
                subscription_age_months=profile.subscription_age_months,
                failure_code=record.failure_code,
                hour_of_failure=record.timestamp.hour,
                day_of_month=record.timestamp.day,
                attempts_so_far=int(rng.randint(0, 3)),
                days_since_last_contact=(
                    int(rng.randint(1, 61)) if contacted_before else None
                ),
            )
        )
    return rows


def _make_execute_fn(
    world: _World,
    truth_by_customer: dict[str, dict[str, Any]],
    noise: dict[str, _Noise],
    contacted: set[str],
    unchargeable: set[str],
) -> Callable:
    """The injected outbound call: what would have happened, in this world.

    An attempt succeeds if the customer would pay under contact and the money
    has arrived by the day it fires. That single rule is where the timing
    advantage lives: a debit two days after the failure and a debit on payday
    are the same action against different balances.

    Nothing about the segment reaches the returned detail. The executor writes
    that detail into the audit log, and the audit log is not a place to leak an
    answer key.
    """

    def execute_fn(record, action) -> OutboundResult:
        customer = world.customer_of_record[record.id]
        truth = truth_by_customer[customer]
        contacted.add(customer)

        # A dead mandate or a terminal decline is refused by the rails, whoever
        # the customer is and however willing they are. A policy that fires at
        # these anyway spends the attempt and gets nothing, which is precisely
        # what the fixed schedule does and what the comparison should show.
        if customer in unchargeable:
            return OutboundResult(success=False, detail={"simulated": True})

        slot_date = action.scheduled_at.date()
        pays = bool(truth["would_pay_with_contact"]) and (
            truth["would_pay_with_date"] <= slot_date
        )

        segment = truth["segment"]
        if segment == "sure_thing" and noise[customer].sure_thing_fails:
            pays = False
        elif segment == "lost_cause" and noise[customer].lost_cause_pays:
            pays = True

        return OutboundResult(success=pays, detail={"simulated": True})

    return execute_fn


async def _send_notifications(conn, world: _World, plan, lead: timedelta) -> None:
    """The executor's pre-debit notification step, stubbed for simulation.

    In production the executor sends this and the guard checks it landed. Here
    the notice is written straight to the contacts table, dated far enough
    before each slot to satisfy the regulatory lead time, which is what a
    compliant sender would have produced. Skipping it would make every attempt
    fail the guard's notification rule, and the comparison would be measuring
    the stub rather than the policy.

    The lead used to be two hours, which the guard accepted while its bound was
    reversed. Once the rule was corrected that stub blocked every attempt in
    the Forbear family - the harness was not compliant and nothing had been
    able to say so.
    """
    if not plan.scheduled:
        return

    await conn.execute(
        """
        INSERT INTO contacts
            (customer_id, subscription_id, channel, purpose, sent_at)
        SELECT customer_id, subscription_id, 'sms'::contact_channel,
               'pre_debit_notification'::contact_purpose, sent_at
        FROM unnest($1::bigint[], $2::bigint[], $3::timestamptz[])
             AS c(customer_id, subscription_id, sent_at)
        """,
        [world.customer_row_of_record[action.record_id] for action in plan.scheduled],
        [world.subscription_of_record[action.record_id] for action in plan.scheduled],
        [action.scheduled_at - lead for action in plan.scheduled],
    )


def _expand_unconstrained(plan, config: HarnessConfig, now: datetime):
    """Turn one action per record into a retry series across the horizon.

    The unconstrained policy ignores the attempt cap, and a policy that can
    retry forever does: it keeps trying until the money lands. Modelling it as
    a single attempt would understate the upper bound and make compliance look
    free, which is the one direction this number must not be wrong in.
    """
    expanded = []
    for action in plan.scheduled:
        day = 0
        while day <= config.horizon_days:
            expanded.append(
                ScheduledAction(
                    action.record_id, action.action_kind, now + timedelta(days=day)
                )
            )
            day += config.unconstrained_retry_days

    # Ordered by slot so every record's first attempt happens before any
    # record's second: one clock runs across the whole plan.
    expanded.sort(key=lambda item: (item.scheduled_at, item.record_id))
    plan.scheduled = expanded
    return plan


async def run_comparison(
    conn,
    seed: int,
    n_records: int,
    config: Optional[HarnessConfig] = None,
) -> ComparisonResult:
    """Run all three policies over the same book and return the comparison.

    Takes a connection because everything downstream of scoring is database
    work - state transitions, attempt rows, the audit chain - and a harness
    that faked those would be measuring a different system to the one that
    ships.
    """
    if not conn.is_in_transaction():
        raise RuntimeError("run_comparison must run inside a transaction")

    config = config or HarnessConfig()
    now = await server_now(conn)
    billing_date: date = now.date()

    # (a-b) Generate the book and randomise who gets contacted.
    profiles = generate_profiles(
        n_records, seed=seed, contact_sensitivity=config.contact_sensitivity
    )
    batch = generate_batch(profiles, billing_date, seed=seed)
    assignments = assign_treatment(batch, config.treatment_rate, seed=seed)
    observed = simulate_outcomes(batch, assignments, seed=seed)

    truth_by_customer = {record.customer_id: record.ground_truth for record in batch}
    plan_amount_by_customer = {
        profile.customer_id: profile.plan_amount for profile in profiles
    }
    # Customers no debit can succeed against, whatever the policy tries.
    unchargeable = {
        record.customer_id
        for record in batch
        if classify(record.failure_code) in UNCHARGEABLE_CLASSES
    }

    # One draw of luck, shared by every policy: a difference in the table has to
    # be a difference in policy.
    noise_rng = np.random.RandomState(seed + 977)
    noise = {
        record.customer_id: _Noise(
            sure_thing_fails=bool(noise_rng.random_sample() < SURE_THING_FAILURE_RATE),
            lost_cause_pays=bool(noise_rng.random_sample() < LOST_CAUSE_RECOVERY_RATE),
        )
        for record in batch
    }

    # (c-d) Fit on outcomes observed under random assignment, then score.
    feature_rng = np.random.RandomState(seed)
    X = build_feature_matrix(_feature_rows(profiles, batch, feature_rng))
    treatment = np.array([assignments[record.customer_id] for record in batch])
    outcome = np.array(
        [
            int(
                observed[record.customer_id].recovered
                and not observed[record.customer_id].churned
            )
            for record in batch
        ]
    )
    model = UpliftModel(seed=seed)
    # Held out before anything is scored, so the number reported next to the
    # table is the model's performance on records it never saw. The in-sample
    # figure runs several times higher and is kept only to show the gap.
    evaluation = model.fit_and_evaluate(X, treatment, outcome)
    cate = model.predict_cate(X)

    results = ComparisonResult()
    results.uplift = evaluation

    for strategy in STRATEGIES:
        started = time.perf_counter()
        world = await _insert_world(conn, strategy, profiles, batch, now, config)

        # (e) Index the records in this world's own id space.
        scores = compute_indices(
            [
                RecordScore(
                    record_id=str(world.record_of_customer[record.customer_id]),
                    amount=record.amount,
                    plan_amount=plan_amount_by_customer[record.customer_id],
                    cate=float(value),
                )
                for record, value in zip(batch, cate)
            ],
            remaining_months=config.remaining_months,
            recovery_survival_rate=config.recovery_survival_rate,
            attempt_cost_paise=config.attempt_cost_paise,
        )
        scored_records = [
            ScoredRecord(
                record_id=int(score.record_id),
                cate=score.cate,
                whittle_index=float(score.whittle_index),
            )
            for score in scores
        ]

        # (f) Plan.
        if strategy == STRATEGY_FIXED:
            plan = await baseline.allocate_fixed_schedule(conn, world.record_ids)
        elif strategy in FORBEAR_FAMILY:
            if strategy == STRATEGY_FORBEAR_CONSTRAINED:
                budget = int(round(config.constrained_budget_ratio * n_records))
                minimum_index = 0.0
            elif strategy == STRATEGY_CLASSIFIER_ONLY:
                # The ablation. Same allocator, same compliance, same
                # everything - except that no record is ever refused for what
                # the model thinks of it. A threshold of -inf disables the
                # value test, and with no batch ceiling the ranking never has
                # to break a tie, so the uplift estimate influences no decision
                # this policy makes. What is left is the classifier: terminal
                # failure classes and dead mandates still get skipped.
                #
                # If this scores what forbear scores, the model and the index
                # earned nothing and the honest thing is to say so.
                budget = None
                minimum_index = float("-inf")
            else:
                budget = config.batch_budget
                minimum_index = 0.0

            plan = await allocate(
                conn,
                scored_records,
                AllocationConfig(
                    batch_budget=budget,
                    horizon_days=config.horizon_days,
                    minimum_index=minimum_index,
                ),
            )
        else:
            plan = await unconstrained_baseline.allocate_unconstrained(
                conn, scored_records
            )
            plan = _expand_unconstrained(plan, config, now)

        # (g) Execute against the simulator on a virtual clock.
        contacted: set[str] = set()
        execute_fn = _make_execute_fn(
            world, truth_by_customer, noise, contacted, unchargeable
        )
        clock = VirtualClock(now)

        if strategy in FORBEAR_FAMILY:
            # Only the Forbear family runs through Forbear's guard. That is the
            # conservative direction for the claim being tested: the policy
            # under test pays the compliance cost - notice periods, NPCI
            # windows, cooldowns, the cap - and the two policies it is measured
            # against do not. Putting the baseline through this guard instead
            # would measure our own plumbing: Razorpay's fixed schedule fires
            # at whatever hour the invoice failed, so most of its attempts land
            # outside an NPCI window, and it would score zero for reasons that
            # have nothing to do with its policy.
            await _send_notifications(conn, world, plan, NOTIFICATION_STUB_LEAD)
            with simulated_guard_clock(clock) as guard_fn:
                report = await execute_plan(conn, plan, guard_fn, execute_fn, clock)
        else:
            report = await execute_plan(conn, plan, allow_everything, execute_fn, clock)

        # (h-i) Count what the contact cost, and net it off.
        results[strategy] = _score_strategy(
            plan=plan,
            report=report,
            batch=batch,
            contacted=contacted,
            truth_by_customer=truth_by_customer,
            plan_amount_by_customer=plan_amount_by_customer,
            config=config,
            elapsed_seconds=time.perf_counter() - started,
        )

    return results


def _score_strategy(
    plan,
    report,
    batch: list[FailedDebit],
    contacted: set[str],
    truth_by_customer: dict[str, dict[str, Any]],
    plan_amount_by_customer: dict[str, int],
    config: HarnessConfig,
    elapsed_seconds: float = 0.0,
) -> StrategyResult:
    """Turn one policy's run into the row it occupies in the table.

    Churn is counted off contact that actually happened. A guard-blocked action
    never reached the customer, so it cannot have driven them away - the one
    way the compliance layer earns something on the value side rather than only
    costing on the recovery side.
    """
    churned = [
        customer
        for customer in contacted
        if truth_by_customer[customer]["would_churn_if_contacted"]
    ]
    ltv_lost = sum(
        plan_amount_by_customer[customer] * config.remaining_months
        for customer in churned
    )

    amount_at_risk = sum(record.amount for record in batch)
    records_processed = len(batch)

    return StrategyResult(
        records_processed=records_processed,
        amount_at_risk=amount_at_risk,
        amount_recovered=report.amount_recovered,
        recovery_rate=(
            report.succeeded / records_processed if records_processed else 0.0
        ),
        attempts_consumed=report.attempted,
        recovered_per_attempt=(
            report.succeeded / report.attempted if report.attempted else 0.0
        ),
        records_skipped=len(plan.skipped),
        records_blocked=report.blocked_by_guard,
        ltv_at_risk_from_skips=sum(
            int(skip.details.get("ltv_at_risk", 0)) for skip in plan.skipped
        ),
        churned_count=len(churned),
        churn_rate=len(churned) / records_processed if records_processed else 0.0,
        ltv_lost_to_churn=ltv_lost,
        net_value=report.amount_recovered - ltv_lost,
        blocked_by_rule=dict(report.blocked_by_rule),
        skips_by_reason=dict(plan.summary.get("skips_by_reason", {})),
        elapsed_seconds=elapsed_seconds,
    )


@dataclass(frozen=True)
class MetricSummary:
    """One metric across seeds. The spread is the point, not the mean."""

    mean: float
    stdev: float
    minimum: float
    maximum: float
    values: tuple[float, ...]

    @property
    def positive_share(self) -> float:
        return sum(1 for value in self.values if value > 0) / len(self.values)


# strategy -> metric name -> summary across seeds.
MultiSeedResult = dict[str, dict[str, MetricSummary]]

# The metrics worth carrying into a headline table. Everything else is
# diagnostic and can be read off a single run.
SUMMARISED_METRICS = (
    "amount_recovered",
    "recovery_rate",
    "attempts_consumed",
    "recovered_per_attempt",
    "records_skipped",
    "churned_count",
    "ltv_lost_to_churn",
    "net_value",
)


async def run_multi_seed(
    conn,
    seeds: Sequence[int],
    n_records: int,
    config: Optional[HarnessConfig] = None,
) -> tuple[MultiSeedResult, MetricSummary]:
    """The comparison, repeated, so the table can report a spread.

    One seed is one draw of a synthetic world, and quoting it as though it were
    the system's performance is how a number ends up 40% above the mean of its
    own distribution. Two of the first twelve seeds tried here came out
    net-negative while the headline seed was comfortably positive.

    Each seed runs inside a savepoint that is rolled back, so ten comparisons
    do not accumulate ten books' worth of rows and advisory locks in one
    transaction.

    Returns the per-strategy summaries and, separately, the held-out Qini
    across seeds - a model diagnostic rather than a policy metric.
    """
    if not conn.is_in_transaction():
        raise RuntimeError("run_multi_seed must run inside a transaction")
    if len(seeds) < 2:
        raise ValueError("a spread needs at least two seeds")

    collected: dict[str, dict[str, list[float]]] = {
        strategy: {metric: [] for metric in SUMMARISED_METRICS}
        for strategy in STRATEGIES
    }
    qini_values: list[float] = []

    for seed in seeds:
        savepoint = conn.transaction()
        await savepoint.start()
        try:
            result = await run_comparison(
                conn, seed=seed, n_records=n_records, config=config
            )
            for strategy in STRATEGIES:
                metrics = result[strategy]
                for metric in SUMMARISED_METRICS:
                    collected[strategy][metric].append(float(getattr(metrics, metric)))
            if result.uplift is not None:
                qini_values.append(result.uplift.held_out_qini)
        finally:
            await savepoint.rollback()

    def summarise(values: list[float]) -> MetricSummary:
        return MetricSummary(
            mean=statistics.mean(values),
            stdev=statistics.stdev(values),
            minimum=min(values),
            maximum=max(values),
            values=tuple(values),
        )

    summaries: MultiSeedResult = {
        strategy: {
            metric: summarise(values) for metric, values in metrics.items()
        }
        for strategy, metrics in collected.items()
    }
    return summaries, summarise(qini_values)


def format_multi_seed(
    summaries: MultiSeedResult, qini: MetricSummary, seeds: Sequence[int]
) -> str:
    """Mean plus or minus one standard deviation, per strategy."""
    rows = [
        ("amount recovered", "amount_recovered", "rupees"),
        ("recovery rate", "recovery_rate", "rate"),
        ("attempts consumed", "attempts_consumed", "count"),
        ("recovered per attempt", "recovered_per_attempt", "ratio"),
        ("records skipped", "records_skipped", "count"),
        ("customers churned", "churned_count", "count"),
        ("ltv lost to churn", "ltv_lost_to_churn", "rupees"),
        ("NET VALUE", "net_value", "rupees"),
    ]

    def render(summary: MetricSummary, kind: str) -> str:
        if kind == "rupees":
            return f"{summary.mean / 100:,.0f} ± {summary.stdev / 100:,.0f}"
        if kind == "rate":
            return f"{summary.mean:.1%} ± {summary.stdev:.1%}"
        if kind == "ratio":
            return f"{summary.mean:.3f} ± {summary.stdev:.3f}"
        return f"{summary.mean:,.0f} ± {summary.stdev:,.0f}"

    label_width = max(len(label) for label, _, _ in rows) + 2
    columns = list(STRATEGIES)
    column_width = 24

    header = f"metric (mean ± σ, {len(seeds)} seeds)".ljust(label_width) + "".join(
        name.rjust(column_width) for name in columns
    )
    lines = [header, "-" * len(header)]
    for label, metric, kind in rows:
        if label == "NET VALUE":
            lines.append("-" * len(header))
        lines.append(
            label.ljust(label_width)
            + "".join(
                render(summaries[name][metric], kind).rjust(column_width)
                for name in columns
            )
        )

    lines.append("")
    for name in columns:
        net = summaries[name]["net_value"]
        lines.append(
            f"  {name:22s} net value positive in "
            f"{net.positive_share * len(seeds):.0f}/{len(seeds)} seeds  "
            f"(min {net.minimum / 100:,.0f}, max {net.maximum / 100:,.0f})"
        )
    lines.append("")
    lines.append(
        f"  held-out Qini across seeds: {qini.mean:.4f} ± {qini.stdev:.4f} "
        f"(min {qini.minimum:.4f}, max {qini.maximum:.4f})"
    )

    return "\n".join(lines)


def rupees(paise: int) -> str:
    return f"{paise / 100:,.0f}"


def format_comparison(result: ComparisonResult) -> str:
    """The table. Recovery rate near the top, net value at the bottom, because
    that is the argument: the policy that recovers most is not the one worth
    most."""
    rows = [
        ("records", lambda r: f"{r.records_processed:,}"),
        ("amount at risk", lambda r: rupees(r.amount_at_risk)),
        ("amount recovered", lambda r: rupees(r.amount_recovered)),
        ("recovery rate", lambda r: f"{r.recovery_rate:.1%}"),
        ("attempts consumed", lambda r: f"{r.attempts_consumed:,}"),
        ("recovered per attempt", lambda r: f"{r.recovered_per_attempt:.3f}"),
        ("records skipped", lambda r: f"{r.records_skipped:,}"),
        ("records blocked", lambda r: f"{r.records_blocked:,}"),
        ("ltv at risk from skips", lambda r: rupees(r.ltv_at_risk_from_skips)),
        ("customers churned", lambda r: f"{r.churned_count:,}"),
        ("churn rate", lambda r: f"{r.churn_rate:.1%}"),
        ("ltv lost to churn", lambda r: rupees(r.ltv_lost_to_churn)),
        ("NET VALUE", lambda r: rupees(r.net_value)),
    ]

    label_width = max(len(label) for label, _ in rows) + 2
    columns = [name for name in STRATEGIES if name in result]
    column_width = 20

    header = "metric (rupees)".ljust(label_width) + "".join(
        name.rjust(column_width) for name in columns
    )
    lines = [header, "-" * len(header)]
    for label, render in rows:
        if label == "NET VALUE":
            lines.append("-" * len(header))
        lines.append(
            label.ljust(label_width)
            + "".join(render(result[name]).rjust(column_width) for name in columns)
        )

    evaluation = getattr(result, "uplift", None)
    if evaluation is not None:
        lines.append("")
        lines.append(
            f"uplift model: held-out Qini {evaluation.held_out_qini:.4f} "
            f"(in-sample {evaluation.in_sample_qini:.4f}, "
            f"overstates by {evaluation.overstatement:.1f}x; "
            f"train {evaluation.n_train:,} / test {evaluation.n_test:,})"
        )

    return "\n".join(lines)
