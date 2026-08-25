"""Server-sent events for one Forbear cycle, decision by decision.

The screen this feeds asks one question: what is the system deciding right now,
and why. So the stream is ordered the way the allocator spends its budget -
highest index first - and every record produces exactly one decision event
whether it was chased or refused. The refusals are the point. A stream that
emitted only the attempts would be showing the easy half of the story.

Records execute one at a time rather than as a batch, so the guard verdict and
the outcome attached to each event are the real ones for that record, taken at
the moment it was decided. That is slower than executing the whole plan at once
and it is the only way the screen is not a replay.

The whole cycle runs inside one transaction that is rolled back at the end.
This is a demo surface: it should be able to run twice with the same seed and
show the same thing, which it cannot do if the first run leaves recovered
records behind.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import asdict
from datetime import timedelta
from typing import Any, AsyncIterator, Optional

import numpy as np
from fastapi import APIRouter, Query, Request
from fastapi.responses import StreamingResponse

from forbear.core.audit import server_now
from forbear.generator.batch_generator import generate_batch
from forbear.generator.customer_profiles import generate_profiles
from forbear.generator.outcome_simulator import (
    LOST_CAUSE_RECOVERY_RATE,
    SURE_THING_FAILURE_RATE,
    simulate_outcomes,
)
from forbear.generator.treatment_assignment import assign_treatment
from forbear.scoring.uplift import UpliftModel, build_feature_matrix
from forbear.scoring.whittle import RecordScore, compute_indices
from forbear.services.allocator import (
    AllocationConfig,
    AllocationPlan,
    ScoredRecord,
    allocate,
)
from forbear.services.classifier import classify
from forbear.services.executor import VirtualClock, execute_plan
from forbear.services.harness import (
    STRATEGY_FORBEAR,
    HarnessConfig,
    _feature_rows,
    _insert_world,
    _make_execute_fn,
    _Noise,
    _send_notifications,
    run_comparison,
    simulated_guard_clock,
)
from forbear.services.unconstrained_baseline import UNCHARGEABLE_CLASSES

router = APIRouter()

# A pause between decisions. Without it the whole cycle arrives in one frame and
# the screen shows a finished list rather than a system working. This is
# presentation, not throttling, and it is the only number in the backend chosen
# for how it looks.
DECISION_INTERVAL_SECONDS = 0.05

MAX_RECORDS = 1000


def sse(event: str, payload: dict[str, Any]) -> str:
    """One SSE frame. Every value must survive json.dumps, or the stream dies
    mid-run with nothing on screen to explain why - hence default=str."""
    return f"event: {event}\ndata: {json.dumps(payload, default=str)}\n\n"


async def _latest_hash(conn, record_id: int) -> Optional[str]:
    """The head of this record's audit chain, so the screen can show that a
    decision left a trace rather than merely claiming it did."""
    return await conn.fetchval(
        """
        SELECT hash FROM audit_log
        WHERE entity_type = 'at_risk_record' AND entity_id = $1
        ORDER BY id DESC LIMIT 1
        """,
        str(record_id),
    )


async def _run_forbear_stream(
    pool, seed: int, n_records: int, demo_mode: bool
) -> AsyncIterator[str]:
    """Score, allocate, then execute one record at a time, emitting as it goes."""
    config = HarnessConfig()

    async with pool.acquire() as conn:
        transaction = conn.transaction()
        await transaction.start()
        try:
            now = await server_now(conn)

            profiles = generate_profiles(n_records, seed=seed)
            batch = generate_batch(profiles, now.date(), seed=seed)
            assignments = assign_treatment(batch, config.treatment_rate, seed=seed)
            observed = simulate_outcomes(batch, assignments, seed=seed)

            truth_by_customer = {
                record.customer_id: record.ground_truth for record in batch
            }
            plan_amount_by_customer = {
                profile.customer_id: profile.plan_amount for profile in profiles
            }
            unchargeable = {
                record.customer_id
                for record in batch
                if classify(record.failure_code) in UNCHARGEABLE_CLASSES
            }

            noise_rng = np.random.RandomState(seed + 977)
            noise = {
                record.customer_id: _Noise(
                    sure_thing_fails=bool(
                        noise_rng.random_sample() < SURE_THING_FAILURE_RATE
                    ),
                    lost_cause_pays=bool(
                        noise_rng.random_sample() < LOST_CAUSE_RECOVERY_RATE
                    ),
                )
                for record in batch
            }

            feature_rng = np.random.RandomState(seed)
            X = build_feature_matrix(_feature_rows(profiles, batch, feature_rng))
            treatment = np.array([assignments[r.customer_id] for r in batch])
            outcome = np.array(
                [
                    int(
                        observed[r.customer_id].recovered
                        and not observed[r.customer_id].churned
                    )
                    for r in batch
                ]
            )
            model = UpliftModel(seed=seed).fit(X, treatment, outcome)
            cate = model.predict_cate(X)
            recovery_probability = model.predict_recovery_probability(X)

            world = await _insert_world(
                conn, STRATEGY_FORBEAR, profiles, batch, now, config
            )

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

            plan = await allocate(
                conn,
                scored_records,
                AllocationConfig(
                    batch_budget=config.batch_budget,
                    horizon_days=config.horizon_days,
                ),
            )
            await _send_notifications(conn, world, plan, timedelta(hours=2))

            by_record = {
                world.record_of_customer[record.customer_id]: record for record in batch
            }
            probability_by_record = {
                world.record_of_customer[record.customer_id]: float(value)
                for record, value in zip(batch, recovery_probability)
            }
            skip_by_record = {skip.record_id: skip for skip in plan.skipped}
            action_by_record = {action.record_id: action for action in plan.scheduled}

            total_at_risk = sum(record.amount for record in batch)
            contacted: set[str] = set()
            execute_fn = _make_execute_fn(
                world, truth_by_customer, noise, contacted, unchargeable
            )
            clock = VirtualClock(now)

            recovered_so_far = 0
            recovered_count = 0
            attempts_consumed = 0
            skipped_count = 0
            ltv_protected = 0

            yield sse(
                "start",
                {
                    "seed": seed,
                    "n_records": n_records,
                    "total_at_risk": total_at_risk,
                    "demo_mode": demo_mode,
                },
            )

            # Decisions in the order the budget was spent: highest value per
            # attempt first, each skip in the position it was refused.
            ordered = sorted(
                scored_records, key=lambda s: (-s.whittle_index, s.record_id)
            )

            with simulated_guard_clock(clock) as guard_fn:
                for scored in ordered:
                    record_id = scored.record_id
                    failed = by_record[record_id]

                    event: dict[str, Any] = {
                        "record_id": record_id,
                        "amount": failed.amount,
                        "failure_code": failed.failure_code,
                        "failure_class": classify(failed.failure_code).value,
                        "cate": round(scored.cate, 4),
                        "whittle_index": round(scored.whittle_index, 3),
                        "recovery_probability": round(
                            probability_by_record[record_id], 4
                        ),
                    }
                    # The segment is the answer key. It exists in a simulation
                    # and would not exist in production, so it is echoed only
                    # when the caller asks for demo mode - and nothing
                    # downstream of the allocator has ever seen it.
                    if demo_mode:
                        event["segment"] = failed.ground_truth["segment"]

                    if record_id in skip_by_record:
                        skip = skip_by_record[record_id]
                        ltv = int(skip.details.get("ltv_at_risk", 0))
                        skipped_count += 1
                        ltv_protected += ltv
                        event.update(
                            {
                                "action": "skipped",
                                "skip_reason": skip.skip_reason,
                                "skip_details": skip.details,
                                "ltv_at_risk": ltv,
                            }
                        )
                    elif record_id in action_by_record:
                        action = action_by_record[record_id]
                        verdicts: list[dict[str, Any]] = []

                        async def capturing_guard(conn, rid, proposed, _v=verdicts):
                            verdict = await guard_fn(conn, rid, proposed)
                            _v.append(
                                {
                                    "allowed": verdict.allowed,
                                    "rule_name": verdict.rule_name,
                                    "details": verdict.details,
                                }
                            )
                            return verdict

                        report = await execute_plan(
                            conn,
                            AllocationPlan(scheduled=[action]),
                            capturing_guard,
                            execute_fn,
                            clock,
                        )
                        attempts_consumed += report.attempted
                        if report.succeeded:
                            recovered_so_far += failed.amount
                            recovered_count += 1

                        event.update(
                            {
                                "action": "scheduled",
                                "scheduled_at": action.scheduled_at.isoformat(),
                                "guard_verdict": verdicts[0] if verdicts else None,
                                "outcome": (
                                    "recovered"
                                    if report.succeeded
                                    else "blocked"
                                    if report.blocked_by_guard
                                    else "failed"
                                ),
                                "audit_hash": await _latest_hash(conn, record_id),
                            }
                        )
                    else:
                        continue

                    yield sse("decision", event)

                    churned = sum(
                        1
                        for customer in contacted
                        if truth_by_customer[customer]["would_churn_if_contacted"]
                    )
                    yield sse(
                        "counter",
                        {
                            "total_at_risk": total_at_risk,
                            "recovered_so_far": recovered_so_far,
                            "recovery_rate": (
                                recovered_count / len(batch) if batch else 0.0
                            ),
                            "skipped_count": skipped_count,
                            "ltv_protected": ltv_protected,
                            "attempts_consumed": attempts_consumed,
                            "recovered_per_attempt": (
                                recovered_count / attempts_consumed
                                if attempts_consumed
                                else 0.0
                            ),
                            "churned_count": churned,
                        },
                    )
                    await asyncio.sleep(DECISION_INTERVAL_SECONDS)
        finally:
            await transaction.rollback()

    # The comparison builds three worlds of its own, so it runs after the live
    # cycle has been rolled back rather than inside it.
    async with pool.acquire() as conn:
        transaction = conn.transaction()
        await transaction.start()
        try:
            comparison = await run_comparison(conn, seed=seed, n_records=n_records)
            yield sse(
                "summary",
                {
                    "seed": seed,
                    "n_records": n_records,
                    "strategies": {
                        name: asdict(metrics) for name, metrics in comparison.items()
                    },
                },
            )
        finally:
            await transaction.rollback()


@router.get("/stream/run")
async def stream_run(
    request: Request,
    seed: int = Query(42, ge=0),
    n: int = Query(200, ge=1, le=MAX_RECORDS),
    demo_mode: bool = Query(True),
) -> StreamingResponse:
    """One cycle, streamed as it is decided.

    demo_mode defaults to true because this endpoint exists to be watched. It
    controls exactly one thing - whether the segment label is echoed back - and
    never what the system does.
    """
    pool = request.app.state.pool

    async def body() -> AsyncIterator[str]:
        try:
            async for frame in _run_forbear_stream(pool, seed, n, demo_mode):
                yield frame
        except Exception as error:  # the screen should say so, not just stop
            yield sse("error", {"detail": str(error), "type": type(error).__name__})

    return StreamingResponse(
        body(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            # Proxies buffer event streams into uselessness without this.
            "X-Accel-Buffering": "no",
        },
    )
