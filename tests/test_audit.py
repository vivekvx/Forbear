"""Audit chain tests.

The chain's only job is to make tampering visible, so the tests that matter are
the ones that tamper.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone

import pytest

from forbear.core.audit import (
    GENESIS_HASH,
    AuditError,
    append_entry,
    compute_hash,
    verify_chain,
)
from tests.conftest import insert_record

pytestmark = pytest.mark.asyncio

ENTITY = "at_risk_record"


async def _chain_of(conn, entity_id, length=3):
    entries = []
    for index in range(length):
        entries.append(
            await append_entry(
                conn, ENTITY, entity_id, f"action_{index}", {"step": index}
            )
        )
    return entries


async def test_first_entry_links_to_genesis(conn):
    record_id = await insert_record(conn, suffix="genesis")

    entry = await append_entry(
        conn, ENTITY, record_id, "opened", {"source": "webhook"}
    )

    assert entry.prev_hash == GENESIS_HASH
    assert len(entry.hash) == 64
    assert entry.details == {"source": "webhook"}


async def test_entries_link_to_their_predecessor(conn):
    record_id = await insert_record(conn, suffix="link")

    entries = await _chain_of(conn, record_id, length=4)

    for previous, current in zip(entries, entries[1:]):
        assert current.prev_hash == previous.hash
    assert await verify_chain(conn, ENTITY, record_id) is None


async def test_hash_covers_entry_content(conn):
    record_id = await insert_record(conn, suffix="content")

    entry = await append_entry(conn, ENTITY, record_id, "scored", {"uplift": 0.42})

    expected = compute_hash(
        entry.prev_hash,
        ENTITY,
        str(record_id),
        "scored",
        {"uplift": 0.42},
        entry.created_at,
    )
    assert entry.hash == expected


async def test_details_key_order_does_not_change_hash():
    timestamp = datetime(2026, 1, 1, tzinfo=timezone.utc)

    first = compute_hash(
        GENESIS_HASH, ENTITY, "1", "scored", {"a": 1, "b": 2}, timestamp
    )
    second = compute_hash(
        GENESIS_HASH, ENTITY, "1", "scored", {"b": 2, "a": 1}, timestamp
    )

    assert first == second


async def test_chains_are_independent_per_entity(conn):
    first_id = await insert_record(conn, suffix="ind1")
    second_id = await insert_record(conn, suffix="ind2")

    await _chain_of(conn, first_id, length=2)
    other = await append_entry(conn, ENTITY, second_id, "opened", {})

    assert other.prev_hash == GENESIS_HASH
    assert await verify_chain(conn, ENTITY, first_id) is None
    assert await verify_chain(conn, ENTITY, second_id) is None


async def test_verify_passes_on_empty_chain(conn):
    record_id = await insert_record(conn, suffix="empty")

    assert await verify_chain(conn, ENTITY, record_id) is None


async def test_verify_catches_corrupted_details(conn):
    record_id = await insert_record(conn, suffix="corrupt")
    entries = await _chain_of(conn, record_id, length=5)

    # Rewrite the payload of entry index 2 while leaving its hash untouched:
    # what someone editing the table directly would do.
    await conn.execute(
        "UPDATE audit_log SET details = $2::jsonb WHERE id = $1",
        entries[2].id,
        json.dumps({"step": 99}),
    )

    assert await verify_chain(conn, ENTITY, record_id) == 2


async def test_verify_catches_corrupted_action(conn):
    record_id = await insert_record(conn, suffix="action")
    entries = await _chain_of(conn, record_id, length=3)

    await conn.execute(
        "UPDATE audit_log SET action = 'rewritten' WHERE id = $1", entries[0].id
    )

    assert await verify_chain(conn, ENTITY, record_id) == 0


async def test_verify_catches_deleted_entry(conn):
    record_id = await insert_record(conn, suffix="deleted")
    entries = await _chain_of(conn, record_id, length=4)

    # Removing an entry orphans the link of the one that followed it.
    await conn.execute("DELETE FROM audit_log WHERE id = $1", entries[1].id)

    assert await verify_chain(conn, ENTITY, record_id) == 1


async def test_verify_catches_severed_link(conn):
    record_id = await insert_record(conn, suffix="severed")
    entries = await _chain_of(conn, record_id, length=3)

    await conn.execute(
        "UPDATE audit_log SET prev_hash = $2 WHERE id = $1",
        entries[2].id,
        "f" * 64,
    )

    assert await verify_chain(conn, ENTITY, record_id) == 2


async def test_append_outside_transaction_is_refused(db_pool):
    async with db_pool.acquire() as connection:
        with pytest.raises(AuditError):
            await append_entry(connection, ENTITY, 1, "opened", {})


async def test_concurrent_appends_do_not_fork_the_chain(clean_db):
    """Twenty concurrent appends to one entity must produce one linear chain."""
    pool = clean_db
    async with pool.acquire() as connection:
        async with connection.transaction():
            record_id = await insert_record(connection, suffix="concurrent")

    async def append_one(index):
        async with pool.acquire() as connection:
            async with connection.transaction():
                return await append_entry(
                    connection, ENTITY, record_id, "scored", {"n": index}
                )

    await asyncio.gather(*(append_one(index) for index in range(20)))

    async with pool.acquire() as connection:
        count = await connection.fetchval(
            "SELECT count(*) FROM audit_log WHERE entity_id = $1", str(record_id)
        )
        broken = await verify_chain(connection, ENTITY, record_id)

    assert count == 20
    assert broken is None
