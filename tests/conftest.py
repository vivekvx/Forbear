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


@pytest_asyncio.fixture
async def clean_db(db_pool):
    """Truncate everything. For tests that need real commits, not a rollback."""
    async with db_pool.acquire() as connection:
        await connection.execute(
            """
            TRUNCATE attempts, contacts, audit_log, at_risk_records,
                     subscriptions, customers, webhook_events
            RESTART IDENTITY CASCADE
            """
        )
    yield db_pool


async def insert_record(conn, *, status: str = "open", suffix: str = "1") -> int:
    """Insert a customer, subscription and at-risk record. Returns record id."""
    customer_id = await conn.fetchval(
        "INSERT INTO customers (external_id) VALUES ($1) RETURNING id",
        f"cust_test_{suffix}",
    )
    subscription_id = await conn.fetchval(
        """
        INSERT INTO subscriptions
            (customer_id, external_id, plan_amount, billing_cycle_days,
             mandate_status)
        VALUES ($1, $2, 49900, 30, 'active')
        RETURNING id
        """,
        customer_id,
        f"sub_test_{suffix}",
    )
    return await conn.fetchval(
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
