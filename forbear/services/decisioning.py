"""Turns one freshly-ingested record into a worklist entry.

This is the background half of the worklist's latency contract: everything a
merchant sees is decided here, once, right after ingestion, so the worklist
endpoint itself is a plain read.

ONE DECISION PATH
------------------
This module scores with the same loaded uplift model and plans with the same
forbear.services.allocator.allocate() the measurement harness uses. There is
no separate heuristic here: if the worklist and the harness ever disagreed
about what to do with a record, the project's central claim - that Forbear's
policy is worth running - would not be a claim about anything real.

allocate() is the real production cycle: with commit=True (its default, used
by the harness and the real cycle) it transitions a record to scheduled or
skipped and writes its own audit entries. Existing behaviour (see
tests/test_emitter.py's "...stays_open_for_forbear") depends on ingestion
leaving a record OPEN until the real cycle actually runs, so this module calls
allocate() with commit=False: the identical plan - the scheduled action or the
skip reason, with the real numbers behind it - comes back, but allocate()
itself writes no transition and no audit entry for it. The record stays
exactly where ingestion left it; only the merchant's view of what would happen
to it is precomputed, from the real allocator, and persisted once - the
worklist_* columns and a single audit entry - by this module.

Nothing here fits a model. The uplift model is fitted once, offline, by
scripts/train_model.py, and loaded from disk the first time this module needs
it - not per webhook, not per worklist read.
"""

from __future__ import annotations

import functools
import logging
from datetime import timedelta
from pathlib import Path
from typing import Any, Optional

from forbear.core.audit import append_entry, server_now
from forbear.models.models import FailureClass, MandateStatus
from forbear.scoring.uplift import FeatureRow, UpliftModel, build_feature_matrix
from forbear.scoring.whittle import RecordScore, compute_whittle_index
from forbear.services.allocator import (
    AllocationConfig,
    SKIP_ATTEMPT_BUDGET_EXHAUSTED,
    SKIP_BATCH_BUDGET_EXHAUSTED,
    SKIP_MANDATE_STATE_INVALID,
    SKIP_NEGATIVE_NET_VALUE,
    SKIP_NO_LEGAL_SLOT,
    SKIP_TERMINAL_FAILURE_CLASS,
    SKIP_UNCLASSIFIED_FAILURE_CODE,
    ScoredRecord,
    allocate,
)

logger = logging.getLogger(__name__)

ENTITY_TYPE = "at_risk_record"

# Fixed, versioned artifact fitted by scripts/train_model.py. Not refit here:
# scoring one freshly-arrived record has no batch to fit against, and fitting
# per webhook would be slow and, on gradient boosting, non-deterministic
# across runs in a way an audit entry cannot explain.
DEFAULT_MODEL_PATH = Path(__file__).resolve().parent.parent.parent / "models" / "uplift_model.pkl"

# A time-dependent failure the allocator would skip on value grounds, but
# whose CATE the model still called positive, is a customer likely to sort
# themselves out. There is no attempt history yet to infer an actual salary
# day from (that inference needs a successful attempt, which a fresh record
# has none of), so this is a stated display estimate for the "wait" bucket's
# expected date - never an input to the allocation decision itself.
_ASSUMED_SELF_RECOVERY_DAYS = 5

_PLAIN_REASONS = {
    SKIP_TERMINAL_FAILURE_CLASS: "This failure can't be fixed by contacting the customer.",
    SKIP_MANDATE_STATE_INVALID: "The customer's payment authorisation is no longer valid.",
    SKIP_UNCLASSIFIED_FAILURE_CODE: "This failure isn't recognised yet; a human should look at it.",
    SKIP_ATTEMPT_BUDGET_EXHAUSTED: "We've already tried the maximum number of times allowed.",
    SKIP_NEGATIVE_NET_VALUE: "Contacting this customer is estimated to destroy more value than it recovers.",
    SKIP_NO_LEGAL_SLOT: "There's no compliant time slot to reach this customer right now.",
    SKIP_BATCH_BUDGET_EXHAUSTED: "Today's contact budget is already spent on higher-priority customers.",
}


class ModelNotAvailable(Exception):
    """The persisted model artifact hasn't been trained yet.

    Raised rather than falling back to a guess: a worklist decision made
    without the real model would be exactly the heuristic this module exists
    to not have.
    """


@functools.lru_cache(maxsize=1)
def _get_model(path: Optional[Path] = None) -> UpliftModel:
    """Load the persisted model once and cache it for the process lifetime."""
    model_path = path or DEFAULT_MODEL_PATH
    if not model_path.exists():
        raise ModelNotAvailable(
            f"no model at {model_path}; run scripts/train_model.py first"
        )
    return UpliftModel.load_model(model_path)


def _rupees(paise: int) -> int:
    return round(paise / 100)


def _cost_sentence(base: str, cost_paise: Optional[int]) -> str:
    if not cost_paise:
        return base
    return f"{base} Contacting risks a ₹{_rupees(cost_paise):,} subscription."


def _leave_alone_reason(skip_reason: str, details: dict[str, Any]) -> str:
    base = _PLAIN_REASONS.get(skip_reason, "We're leaving this one alone for now.")
    cost = details.get("ltv_at_risk") or details.get("amount")
    return _cost_sentence(base, cost)


async def _facts(conn, record_id: int):
    return await conn.fetchrow(
        """
        SELECT r.id, r.amount, r.failure_code, r.failure_class, r.created_at,
               r.customer_id,
               s.plan_amount, s.mandate_status, s.created_at AS subscription_created_at,
               COALESCE(counted.attempts, 0) AS attempts_so_far,
               contact.last_contact_at
        FROM at_risk_records r
        JOIN subscriptions s ON s.id = r.subscription_id
        LEFT JOIN LATERAL (
            SELECT count(*) AS attempts FROM attempts a
            WHERE a.at_risk_record_id = r.id
        ) counted ON TRUE
        LEFT JOIN LATERAL (
            SELECT max(c.sent_at) AS last_contact_at FROM contacts c
            WHERE c.customer_id = r.customer_id
        ) contact ON TRUE
        WHERE r.id = $1 AND r.status = 'open'
        """,
        record_id,
    )


def _build_feature_row(row) -> FeatureRow:
    created_at = row["created_at"]
    subscription_age_months = max(
        0,
        (row["created_at"].year - row["subscription_created_at"].year) * 12
        + (row["created_at"].month - row["subscription_created_at"].month),
    )
    days_since_last_contact = (
        (row["created_at"] - row["last_contact_at"]).days
        if row["last_contact_at"] is not None
        else None
    )
    return FeatureRow(
        plan_amount=row["plan_amount"],
        subscription_age_months=subscription_age_months,
        failure_code=row["failure_code"],
        hour_of_failure=created_at.hour,
        day_of_month=created_at.day,
        attempts_so_far=row["attempts_so_far"],
        days_since_last_contact=days_since_last_contact,
    )


async def decide_record(conn, record_id: int) -> Optional[str]:
    """Preview a worklist decision for one record. Never transitions status.

    Returns the bucket assigned, or None if the record wasn't eligible to be
    previewed (already moved on by the time this ran, or does not exist).
    """
    if not conn.is_in_transaction():
        raise RuntimeError("decide_record must run inside a transaction")

    row = await _facts(conn, record_id)
    if row is None:
        return None

    failure_class = (
        FailureClass(row["failure_class"]) if row["failure_class"] else None
    )
    now = await server_now(conn)

    if failure_class is None:
        # Nothing to score: the classifier has no mapping, so there is no
        # encoding for the model to run on. allocate() below still produces
        # the authoritative skip - unclassified is checked before the score
        # is ever consulted - the placeholder just satisfies its signature.
        cate = 0.0
        recovery_probability = None
        whittle_index = 0.0
    else:
        model = _get_model()
        feature_row = _build_feature_row(row)
        X = build_feature_matrix([feature_row])
        cate = float(model.predict_cate(X)[0])
        recovery_probability = float(model.predict_recovery_probability(X)[0])
        whittle_index = compute_whittle_index(
            RecordScore(
                record_id=str(record_id),
                amount=row["amount"],
                plan_amount=row["plan_amount"],
                cate=cate,
            )
        )

    scored = ScoredRecord(record_id=record_id, cate=cate, whittle_index=whittle_index)
    plan = await allocate(conn, [scored], AllocationConfig(), commit=False)

    bucket: str
    action: str
    reason: str
    scheduled_at = None
    cost_paise = None
    # Set only for a real leave_alone (see below); chase and wait never carry
    # a skip reason, since nothing was skipped to produce them.
    worklist_skip_reason: Optional[str] = None

    if plan.scheduled:
        scheduled = plan.scheduled[0]
        bucket = "chase"
        action = "retry" if failure_class is FailureClass.TRANSIENT else "send_payment_link"
        reason = (
            "A quick retry should clear this."
            if failure_class is FailureClass.TRANSIENT
            else "Reaching out now is worth it - the customer usually responds."
        )
        scheduled_at = scheduled.scheduled_at
    else:
        skip = plan.skipped[0]
        details = skip.details
        if skip.skip_reason == SKIP_NEGATIVE_NET_VALUE and failure_class is FailureClass.TIME_DEPENDENT and cate > 0:
            bucket, action = "wait", "none"
            reason = "Likely to pay on their own, probably around salary day."
            scheduled_at = now + timedelta(days=_ASSUMED_SELF_RECOVERY_DAYS)
        else:
            bucket, action = "leave_alone", "none"
            cost_paise = details.get("ltv_at_risk") or details.get("amount")
            reason = _leave_alone_reason(skip.skip_reason, details)
            worklist_skip_reason = skip.skip_reason

    await conn.execute(
        """
        UPDATE at_risk_records
        SET worklist_bucket = $2,
            worklist_action = $3,
            worklist_reason = $4,
            worklist_scheduled_at = $5,
            worklist_cost_paise = $6,
            worklist_decided_at = $7,
            worklist_skip_reason = $8
        WHERE id = $1
        """,
        record_id,
        bucket,
        action,
        reason,
        scheduled_at,
        cost_paise,
        now,
        worklist_skip_reason,
    )
    await append_entry(
        conn,
        ENTITY_TYPE,
        record_id,
        "worklist_decision",
        {
            "bucket": bucket,
            "action": action,
            "reason": reason,
            "scheduled_at": scheduled_at.isoformat() if scheduled_at else None,
            "cost_paise": cost_paise,
            "cate": cate,
            "recovery_probability": recovery_probability,
            "whittle_index": whittle_index,
            "skip_reason": None if plan.scheduled else plan.skipped[0].skip_reason,
        },
    )
    return bucket


async def decide_record_in_background(pool, record_id: int) -> None:
    """Entry point for a FastAPI BackgroundTask: owns its own connection.

    Swallows its own failures into a log line, matching webhooks.py's rule
    that a handler failure must never turn into a retry storm - here there is
    no caller to retry, but the same "log and move on" discipline applies so
    one bad record can't take the background worker down.
    """
    try:
        async with pool.acquire() as conn:
            async with conn.transaction():
                await decide_record(conn, record_id)
    except Exception:
        logger.exception("worklist decisioning failed for record %s", record_id)
