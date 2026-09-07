"""Populates a demo book with real records plus their answer key.

Nothing here is part of the production ingestion path. A real webhook has no
counterfactual to report, so real records never get a demo_ground_truth row -
that emptiness is how GET /worklist/protected tells a demo run apart from
production and hides the protected-customers panel rather than fabricate one
(see forbear.api.worklist and schema.sql's comment on demo_ground_truth).

Decisions themselves are not computed here. Each seeded record is handed to
forbear.services.decisioning.decide_record - the same real model, the same
real allocator, the one the webhook path uses - so the worklist and the
protected panel read the same decision a live record would have gotten.
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone

from forbear.generator.batch_generator import generate_batch
from forbear.generator.customer_profiles import generate_profiles
from forbear.services.allocator import ltv_at_risk
from forbear.services.classifier import classify
from forbear.services.decisioning import decide_record


async def _insert_customer_and_subscription(
    conn, profile, billing_date: date
) -> tuple[int, int]:
    customer_id = await conn.fetchval(
        "INSERT INTO customers (external_id) VALUES ($1) RETURNING id",
        profile.customer_id,
    )
    # Backdated by the profile's own tenure. Left at the INSERT default (now)
    # this would tell the model every synthetic subscription is brand new
    # regardless of segment - decide_record's feature extraction reads tenure
    # from created_at, not from the profile, exactly like a real subscription
    # row would - and do_not_disturb is disproportionately long-tenured (see
    # customer_profiles.SEGMENT_TENURE_RATE), so a zeroed-out tenure feature
    # would make that segment structurally invisible to the model here even
    # though it is not invisible to it in general (predict_cate correctly
    # separates it given real features).
    created_at = datetime.combine(
        billing_date, time.min, tzinfo=timezone.utc
    ) - timedelta(days=30 * profile.subscription_age_months)
    subscription_id = await conn.fetchval(
        """
        INSERT INTO subscriptions
            (customer_id, external_id, plan_amount, billing_cycle_days,
             mandate_status, created_at)
        VALUES ($1, $2, $3, 30, $4::mandate_status, $5)
        RETURNING id
        """,
        customer_id,
        f"sub_{profile.customer_id}",
        profile.plan_amount,
        profile.mandate_status.value,
        created_at,
    )
    return customer_id, subscription_id


async def seed_demo_batch(conn, n: int, seed: int = 0) -> list[int]:
    """Insert n synthetic at-risk records, decide each, and record ground truth.

    Runs record-by-record in its own transaction per record (matching how
    ingestion actually calls decide_record for one freshly-arrived record at a
    time) rather than one transaction for the whole batch, so a demo run's
    shape matches production's as closely as a batch call can.

    Returns the inserted record ids.
    """
    billing_date = date.today()
    profiles = generate_profiles(n, seed=seed)
    batch = generate_batch(profiles, billing_date=billing_date, seed=seed)

    record_ids: list[int] = []
    for profile, failed in zip(profiles, batch):
        async with conn.transaction():
            customer_id, subscription_id = await _insert_customer_and_subscription(
                conn, profile, billing_date
            )
            record_id = await conn.fetchval(
                """
                INSERT INTO at_risk_records
                    (subscription_id, customer_id, invoice_id, amount,
                     failure_code, failure_class, status, created_at)
                VALUES ($1, $2, $3, $4, $5, $6, 'open', $7)
                RETURNING id
                """,
                subscription_id,
                customer_id,
                f"inv_{profile.customer_id}",
                failed.amount,
                failed.failure_code,
                classify(failed.failure_code).value,
                failed.timestamp,
            )
            await decide_record(conn, record_id)

            truth = failed.ground_truth
            await conn.execute(
                """
                INSERT INTO demo_ground_truth
                    (at_risk_record_id, would_churn_if_contacted,
                     would_pay_without_contact, remaining_ltv_paise)
                VALUES ($1, $2, $3, $4)
                """,
                record_id,
                truth["would_churn_if_contacted"],
                truth["would_pay_without_contact"],
                ltv_at_risk(profile.plan_amount),
            )
        record_ids.append(record_id)

    return record_ids
