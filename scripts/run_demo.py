#!/usr/bin/env python3
"""Everything Forbear claims, in one command, from nothing.

Four sections, in the order the argument is made: here are six things that
must never happen and the system refusing each one; here is what the policy is
worth against the platform's own schedule; here is the churn cost at which the
answer flips; and here is all of it again at twenty times the size.

The script builds its own database and drops it on the way out. That is the
whole reason it exists in this form - a demo with setup steps is a demo that
fails on recording day, because the step everyone forgets is the one that was
obvious in the room where it was written.

    python scripts/run_demo.py                  # the full run
    python scripts/run_demo.py --skip-scale     # rehearsal, minus the slow part

Nothing here is interactive and nothing prompts. It either prints four blocks
or it fails loudly.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import pathlib
import sys
import time
import uuid

import asyncpg

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from forbear.api.webhooks import SECRET_ENV  # noqa: E402
from forbear.services.harness import (  # noqa: E402
    STRATEGIES,
    STRATEGY_FORBEAR,
    format_comparison,
    one_transaction_record_limit,
    run_comparison,
)
from forbear.services.sensitivity import (  # noqa: E402
    find_crossover,
    format_sweep,
    run_sweep,
)

# The attacks live in the test file rather than here, so the narrative has one
# definition: what the suite asserts and what the demo prints cannot drift.
from tests.test_adversarial import (  # noqa: E402
    WEBHOOK_SECRET,
    attack_attempt_cap_exceeded,
    attack_duplicate_webhook_replay,
    attack_expired_mandate,
    attack_insufficient_notification_lead,
    attack_npci_window_violation,
    attack_revoked_mandate,
)

SCHEMA_PATH = REPO_ROOT / "schema.sql"
ADMIN_DSN = os.environ.get("FORBEAR_ADMIN_DSN", "postgres:///postgres")

# The nine-point sweep. Dense where the crossover sits, sparse at the ends.
SWEEP_RATES = (0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 1.0)

# The four attacks that run against a connection. The webhook replay needs a
# pool of its own, because the endpoint acquires and commits for itself.
CONNECTION_ATTACKS = (
    attack_revoked_mandate,
    attack_expired_mandate,
    attack_attempt_cap_exceeded,
    attack_npci_window_violation,
    attack_insufficient_notification_lead,
)

# Five run against a connection plus the webhook replay, which needs a pool.
TOTAL_ATTACKS = len(CONNECTION_ATTACKS) + 1

TRUNCATE_ALL = """
    TRUNCATE attempts, contacts, audit_log, at_risk_records,
             subscriptions, customers, webhook_events
    RESTART IDENTITY CASCADE
"""


def banner(title: str) -> None:
    print(f"\n\n=== {title} ===\n", flush=True)


async def adversarial_suite(pool) -> int:
    """Six illegal inputs, six refusals. Returns the number blocked."""
    banner("ADVERSARIAL SUITE")

    blocked = 0
    async with pool.acquire() as conn:
        transaction = conn.transaction()
        await transaction.start()
        try:
            for attack in CONNECTION_ATTACKS:
                result = await attack(conn)
                print(result.summary, flush=True)
                blocked += bool(result.blocked)
        finally:
            # These attacks plant rows on purpose, and none of them belong to
            # the comparison that runs next.
            await transaction.rollback()

    # The replay attack goes through the real endpoint, which commits.
    os.environ[SECRET_ENV] = WEBHOOK_SECRET
    replay = await attack_duplicate_webhook_replay(pool)
    print(replay.summary, flush=True)
    blocked += bool(replay.blocked)

    async with pool.acquire() as conn:
        await conn.execute(TRUNCATE_ALL)

    print(f"\n{blocked}/{TOTAL_ATTACKS} blocked.", flush=True)
    return blocked


async def comparison(pool, seed: int, n_records: int, title: str) -> dict:
    banner(title)

    async with pool.acquire() as conn:
        transaction = conn.transaction()
        await transaction.start()
        try:
            started = time.perf_counter()
            result = await run_comparison(conn, seed=seed, n_records=n_records)
            elapsed = time.perf_counter() - started

            print(format_comparison(result), flush=True)
            print(f"\nthroughput ({n_records:,} records)", flush=True)
            for name in STRATEGIES:
                metrics = result[name]
                print(
                    f"  {name:20s} "
                    f"{metrics.records_processed / metrics.elapsed_seconds:8.0f} rec/s"
                    f"  ({metrics.elapsed_seconds:6.1f}s)",
                    flush=True,
                )
            print(f"  {'whole comparison':20s} {elapsed:8.1f}s total", flush=True)
            return result
        finally:
            await transaction.rollback()


async def sweep(pool, seed: int, n_records: int, csv_path: str) -> None:
    banner("SENSITIVITY SWEEP")

    async with pool.acquire() as conn:
        transaction = conn.transaction()
        await transaction.start()
        try:
            points = await run_sweep(
                conn,
                seed=seed,
                n_records=n_records,
                churn_rates=SWEEP_RATES,
                csv_path=csv_path,
            )
        finally:
            await transaction.rollback()

    print(format_sweep(points), flush=True)

    crossover = find_crossover(points)
    if crossover is None:
        print("\nno crossover in the sampled range.", flush=True)
    else:
        print(
            f"\nCROSSOVER: {crossover:.2f} dunning churn per contact.\n"
            f"Below it, chase everything. At or above it, be selective.",
            flush=True,
        )
    print(f"sweep written to {csv_path}", flush=True)


async def main(args: argparse.Namespace) -> int:
    database = f"forbear_demo_{uuid.uuid4().hex[:10]}"

    admin = await asyncpg.connect(ADMIN_DSN)
    try:
        await admin.execute(f'CREATE DATABASE "{database}"')
    finally:
        await admin.close()

    pool = await asyncpg.create_pool(f"postgres:///{database}", min_size=2, max_size=8)
    try:
        async with pool.acquire() as conn:
            await conn.execute(SCHEMA_PATH.read_text())

        blocked = await adversarial_suite(pool)
        await comparison(pool, args.seed, args.n, f"COMPARISON (n={args.n})")
        await sweep(pool, args.seed, args.n, args.csv)

        scale = None
        if args.skip_scale:
            print("\n\n=== SCALE CHECK skipped (--skip-scale) ===", flush=True)
        else:
            async with pool.acquire() as conn:
                limit = await one_transaction_record_limit(conn)

            if args.scale_n > limit:
                # Said out loud rather than discovered as an out-of-memory
                # error twenty minutes in. The ceiling is this harness holding
                # one transaction across the whole book; production commits per
                # decision and never accumulates the locks.
                banner(f"SCALE CHECK (n={args.scale_n:,}) NOT RUN")
                print(
                    f"This PostgreSQL can compare at most {limit:,} records in "
                    f"one transaction.\nEvery audit append holds a "
                    f"transaction-scoped advisory lock, and a comparison runs\n"
                    f"three worlds at once. Raise max_locks_per_transaction and "
                    f"restart, or\nrun with --scale-n {limit:,}.",
                    flush=True,
                )
            else:
                scale = await comparison(
                    pool, args.seed, args.scale_n, f"SCALE CHECK (n={args.scale_n:,})"
                )

        banner("SUMMARY")
        print(f"adversarial: {blocked}/{TOTAL_ATTACKS} illegal actions refused", flush=True)
        if scale is not None:
            forbear = scale[STRATEGY_FORBEAR]
            print(
                f"at {args.scale_n:,} records, forbear net value "
                f"{forbear.net_value / 100:,.0f} rupees "
                f"({forbear.records_skipped:,} records deliberately not chased)",
                flush=True,
            )
        return 0 if blocked == TOTAL_ATTACKS else 1
    finally:
        await pool.close()
        admin = await asyncpg.connect(ADMIN_DSN)
        try:
            await admin.execute(f'DROP DATABASE IF EXISTS "{database}" WITH (FORCE)')
        finally:
            await admin.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the whole Forbear demo.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n", type=int, default=500, help="records in the comparison")
    parser.add_argument("--scale-n", type=int, default=10_000)
    parser.add_argument(
        "--skip-scale",
        action="store_true",
        help="skip the 10,000-record run; for rehearsing the first three blocks",
    )
    parser.add_argument("--csv", default="sweep.csv")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main(parse_args())))
