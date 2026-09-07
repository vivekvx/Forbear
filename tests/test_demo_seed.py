"""forbear.services.demo_seed: the only thing that ever writes demo_ground_truth.

Minimal sanity check that seeding a demo batch produces real decisions (via
the real decisioning path) plus a ground-truth row for every record, so
GET /worklist/protected has real data to read in demo mode.
"""

from __future__ import annotations

import pytest

from forbear.services.demo_seed import seed_demo_batch

pytestmark = pytest.mark.asyncio


async def test_seed_demo_batch_decides_every_record_and_writes_ground_truth(
    clean_db,
):
    async with clean_db.acquire() as conn:
        record_ids = await seed_demo_batch(conn, n=20, seed=1)

        assert len(record_ids) == 20

        decided = await conn.fetchval(
            "SELECT count(*) FROM at_risk_records WHERE worklist_bucket IS NOT NULL"
        )
        assert decided == 20

        # Ingestion leaves the record open - decide_record previews, it never
        # transitions - matching the same invariant the webhook path relies on.
        still_open = await conn.fetchval(
            "SELECT count(*) FROM at_risk_records WHERE status = 'open'"
        )
        assert still_open == 20

        truth_rows = await conn.fetchval("SELECT count(*) FROM demo_ground_truth")
        assert truth_rows == 20


async def test_seed_demo_batch_produces_at_least_one_real_save(clean_db):
    """do_not_disturb is 15% of the mix; a big enough batch should yield saves."""
    async with clean_db.acquire() as conn:
        await seed_demo_batch(conn, n=80, seed=7)

        from forbear.services.allocator import SKIP_NEGATIVE_NET_VALUE

        saves = await conn.fetchval(
            """
            SELECT count(*)
            FROM at_risk_records r
            JOIN demo_ground_truth g ON g.at_risk_record_id = r.id
            WHERE r.worklist_bucket = 'leave_alone'
              AND r.worklist_skip_reason = $1
              AND g.would_churn_if_contacted = TRUE
              AND g.would_pay_without_contact = TRUE
            """,
            SKIP_NEGATIVE_NET_VALUE,
        )
        assert saves > 0
