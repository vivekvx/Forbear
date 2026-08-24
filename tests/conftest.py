"""Test fixtures. Every test runs against a real PostgreSQL database.

The state machine and the audit chain are mostly claims about what PostgreSQL
does under concurrency (row locks, advisory locks, unique constraints). A fake
database would only test the fake.
"""

from __future__ import annotations

import os
import pathlib
import uuid

import asyncpg
import pytest_asyncio

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
SCHEMA_PATH = REPO_ROOT / "schema.sql"

ADMIN_DSN = os.environ.get("FORBEAR_ADMIN_DSN", "postgres:///postgres")


def _test_dsn(db_name: str) -> str:
    return os.environ.get("FORBEAR_TEST_DSN", f"postgres:///{db_name}")


@pytest_asyncio.fixture(scope="session")
async def db_pool():
    """Create a throwaway database, load schema.sql, drop it afterwards."""
    db_name = f"forbear_test_{uuid.uuid4().hex[:12]}"

    admin = await asyncpg.connect(ADMIN_DSN)
    try:
        await admin.execute(f'CREATE DATABASE "{db_name}"')
    finally:
        await admin.close()

    pool = await asyncpg.create_pool(_test_dsn(db_name), min_size=2, max_size=60)
    try:
        async with pool.acquire() as conn:
            await conn.execute(SCHEMA_PATH.read_text())
        yield pool
    finally:
        await pool.close()
        admin = await asyncpg.connect(ADMIN_DSN)
        try:
            await admin.execute(f'DROP DATABASE IF EXISTS "{db_name}" WITH (FORCE)')
        finally:
            await admin.close()


@pytest_asyncio.fixture
async def conn(db_pool):
    """A connection whose work is rolled back at the end of the test."""
    async with db_pool.acquire() as connection:
        transaction = connection.transaction()
        await transaction.start()
        try:
            yield connection
        finally:
            await transaction.rollback()


TRUNCATE_ALL = """
    TRUNCATE attempts, contacts, audit_log, at_risk_records,
             subscriptions, customers, webhook_events
    RESTART IDENTITY CASCADE
"""


@pytest_asyncio.fixture
async def clean_db(db_pool):
    """Truncate everything. For tests that need real commits, not a rollback.

    Truncates on the way out as well: these tests commit, and committed rows
    would otherwise be visible to every later test that counts rows.
    """
    async with db_pool.acquire() as connection:
        await connection.execute(TRUNCATE_ALL)
    try:
        yield db_pool
    finally:
        async with db_pool.acquire() as connection:
            await connection.execute(TRUNCATE_ALL)


async def insert_scenario(
    conn,
    *,
    status: str = "open",
    mandate_status: str = "active",
    suffix: str | None = None,
) -> dict[str, int]:
    """Insert a customer, subscription and at-risk record.

    Returns all three ids, because the guard reads across all three tables.
    """
    if suffix is None:
        suffix = uuid.uuid4().hex[:10]

    customer_id = await conn.fetchval(
        "INSERT INTO customers (external_id) VALUES ($1) RETURNING id",
        f"cust_test_{suffix}",
    )
    subscription_id = await conn.fetchval(
        """
        INSERT INTO subscriptions
            (customer_id, external_id, plan_amount, billing_cycle_days,
             mandate_status)
        VALUES ($1, $2, 49900, 30, $3::mandate_status)
        RETURNING id
        """,
        customer_id,
        f"sub_test_{suffix}",
        mandate_status,
    )
    record_id = await conn.fetchval(
        """
        INSERT INTO at_risk_records
            (subscription_id, customer_id, invoice_id, amount, failure_code,
             failure_class, status, skip_reason)
        VALUES ($1, $2, $3, 49900, 'INSUFFICIENT_FUNDS', 'time_dependent',
                $4::record_status, $5)
        RETURNING id
        """,
        subscription_id,
        customer_id,
        f"inv_test_{suffix}",
        status,
        "seeded_for_test" if status == "skipped" else None,
    )
    return {
        "customer_id": customer_id,
        "subscription_id": subscription_id,
        "record_id": record_id,
    }


async def insert_record(conn, *, status: str = "open", suffix: str = "1") -> int:
    """Insert a customer, subscription and at-risk record. Returns record id."""
    scenario = await insert_scenario(conn, status=status, suffix=suffix)
    return scenario["record_id"]


async def insert_attempt(
    conn,
    record_id: int,
    attempt_number: int,
    *,
    outcome: str = "failure",
    scheduled_at=None,
    executed_at=None,
) -> int:
    """Insert one attempt row. executed_at is required unless outcome=pending."""
    return await conn.fetchval(
        """
        INSERT INTO attempts
            (at_risk_record_id, attempt_number, scheduled_at, executed_at,
             outcome)
        VALUES ($1, $2, $3, $4, $5::attempt_outcome)
        RETURNING id
        """,
        record_id,
        attempt_number,
        scheduled_at or executed_at,
        executed_at,
        outcome,
    )


async def insert_notification(
    conn,
    customer_id: int,
    subscription_id: int | None,
    sent_at,
    *,
    purpose: str = "pre_debit_notification",
    channel: str = "sms",
) -> int:
    return await conn.fetchval(
        """
        INSERT INTO contacts
            (customer_id, subscription_id, channel, purpose, sent_at)
        VALUES ($1, $2, $3::contact_channel, $4::contact_purpose, $5)
        RETURNING id
        """,
        customer_id,
        subscription_id,
        channel,
        purpose,
        sent_at,
    )
