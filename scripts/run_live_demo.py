#!/usr/bin/env python3
"""One-command live-pipe demo: real signed webhooks into Forbear's real server.

Unlike scripts/run_demo.py, this does not spin up its own app or database --
it needs a Forbear server already running, because the whole point is proving
the real HTTP path (signature verification, replay detection, classification,
state transitions, audit logging) end to end, not exercising it in-process.

    RAZORPAY_WEBHOOK_SECRET=demo_secret \\
        uvicorn forbear.api.main:app --reload &
    RAZORPAY_WEBHOOK_SECRET=demo_secret python scripts/run_live_demo.py

The two RAZORPAY_WEBHOOK_SECRET values must be identical -- one signs, the
other verifies, and a mismatch is what a stolen or misconfigured secret looks
like in production too.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import pathlib
import sys

import asyncpg
import httpx

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from forbear.api.webhooks import SECRET_ENV  # noqa: E402
from forbear.emitter.emitter import WebhookEmitter, emit_scenario  # noqa: E402
from forbear.emitter.scenarios import batch_scenario  # noqa: E402

DSN = os.environ.get("FORBEAR_DSN", "postgres:///forbear")
SERVER_URL = os.environ.get("FORBEAR_SERVER_URL", "http://127.0.0.1:8000")

TRUNCATE_ALL = """
    TRUNCATE attempts, contacts, audit_log, at_risk_records,
             subscriptions, customers, webhook_events
    RESTART IDENTITY CASCADE
"""


async def main(args: argparse.Namespace) -> int:
    secret = os.environ.get(SECRET_ENV)
    if not secret:
        print(
            f"{SECRET_ENV} is not set. Export the same value the running "
            "server was started with -- the emitter signs with it, the "
            "receiver verifies with it, and they must agree.",
            file=sys.stderr,
        )
        return 1

    pool = await asyncpg.create_pool(DSN, min_size=1, max_size=4)
    try:
        async with pool.acquire() as conn:
            await conn.execute(TRUNCATE_ALL)

        scenario = batch_scenario(args.n)

        async with httpx.AsyncClient(base_url=SERVER_URL, timeout=10.0) as client:
            emitter = WebhookEmitter(secret=secret, client=client)
            try:
                results = await emit_scenario(emitter, scenario)
            except httpx.ConnectError:
                print(
                    f"Could not reach {SERVER_URL}. Start the server first:\n"
                    f"  {SECRET_ENV}={secret} uvicorn forbear.api.main:app --reload",
                    file=sys.stderr,
                )
                return 1

        statuses: dict[str, int] = {}
        for result in results:
            status = (
                result.response_json.get("status")
                if result.response_json
                else f"http_{result.status_code}"
            )
            statuses[status] = statuses.get(status, 0) + 1

        async with pool.acquire() as conn:
            at_risk = await conn.fetchval("SELECT count(*) FROM at_risk_records")
            recovered = await conn.fetchval(
                "SELECT count(*) FROM at_risk_records WHERE status = 'recovered'"
            )
            still_working = await conn.fetchval(
                "SELECT count(*) FROM at_risk_records "
                "WHERE status IN ('open', 'scheduled', 'in_flight')"
            )
            terminal = await conn.fetchval(
                "SELECT count(*) FROM at_risk_records WHERE failure_class = 'terminal'"
            )
            decisions = await conn.fetchval("SELECT count(*) FROM audit_log")

        print(f"\n=== LIVE-PIPE DEMO: {scenario.name} ===\n")
        print(f"events emitted:        {len(results)}")
        for status, count in sorted(statuses.items()):
            print(f"  {status:16s}  {count}")
        print(f"\nat_risk_records created: {at_risk}")
        print(f"  recovered:             {recovered}")
        print(f"  still open/working:    {still_working}")
        print(f"  terminal (never chased): {terminal}")
        print(f"\naudit entries (decisions made): {decisions}")
        return 0
    finally:
        await pool.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the live-pipe webhook demo.")
    parser.add_argument("--n", type=int, default=12, help="mixed lifecycles to emit")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main(parse_args())))
