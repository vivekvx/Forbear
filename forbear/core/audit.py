"""Tamper-evident audit chain.

Every decision Forbear makes, including every deliberate skip, appends an entry
here. Entries are hash-linked per entity: an entry's hash covers the previous
entry's hash, so altering or deleting any entry breaks every link after it.

Timestamps come from the database (server_now), never from Python.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from typing import Any, Optional

from forbear.models.models import AuditEntry

GENESIS_HASH = "0" * 64


class AuditError(Exception):
    """Raised when the audit chain cannot be appended to safely."""


async def server_now(conn) -> datetime:
    """The only sanctioned clock. Invariant 3: never Python datetime.now()."""
    return await conn.fetchval("SELECT now()")


def canonical_details(details: Optional[dict[str, Any]]) -> str:
    """Stable JSON encoding so a hash does not depend on key order."""
    return json.dumps(
        details or {},
        sort_keys=True,
        separators=(",", ":"),
        default=str,
        ensure_ascii=False,
    )


def compute_hash(
    prev_hash: str,
    entity_type: str,
    entity_id: str,
    action: str,
    details: Optional[dict[str, Any]],
    timestamp: datetime,
) -> str:
    """SHA-256 over the previous hash plus this entry's full content.

    Fields are length-prefixed before concatenation so no two different field
    sets can produce the same byte string (entity_type "ab" + id "c" must not
    collide with "a" + "bc").
    """
    parts = [
        prev_hash,
        entity_type,
        entity_id,
        action,
        canonical_details(details),
        timestamp.isoformat(),
    ]
    payload = "".join(f"{len(part)}:{part}" for part in parts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _lock_key(entity_type: str, entity_id: str) -> str:
    return f"forbear.audit.{entity_type}.{entity_id}"


async def append_entry(
    conn,
    entity_type: str,
    entity_id: str | int,
    action: str,
    details: Optional[dict[str, Any]] = None,
) -> AuditEntry:
    """Append one entry to an entity's chain.

    Must run inside a transaction. Takes a transaction-scoped advisory lock on
    the entity so two concurrent appends cannot read the same prev_hash and
    fork the chain. A row lock would not cover the first entry for an entity,
    where there is no row to lock.
    """
    if not conn.is_in_transaction():
        raise AuditError("append_entry must run inside a transaction")

    entity_id = str(entity_id)
    await conn.execute(
        "SELECT pg_advisory_xact_lock(hashtext($1))",
        _lock_key(entity_type, entity_id),
    )

    prev_hash = await conn.fetchval(
        """
        SELECT hash FROM audit_log
        WHERE entity_type = $1 AND entity_id = $2
        ORDER BY id DESC
        LIMIT 1
        """,
        entity_type,
        entity_id,
    )
    prev_hash = prev_hash or GENESIS_HASH

    created_at = await server_now(conn)
    entry_hash = compute_hash(
        prev_hash, entity_type, entity_id, action, details, created_at
    )

    row = await conn.fetchrow(
        """
        INSERT INTO audit_log
            (entity_type, entity_id, action, details, prev_hash, hash, created_at)
        VALUES ($1, $2, $3, $4::jsonb, $5, $6, $7)
        RETURNING id, entity_type, entity_id, action, details, prev_hash, hash,
                  created_at
        """,
        entity_type,
        entity_id,
        action,
        canonical_details(details),
        prev_hash,
        entry_hash,
        created_at,
    )
    return _row_to_entry(row)


async def verify_chain(
    conn, entity_type: str, entity_id: str | int
) -> Optional[int]:
    """Walk an entity's chain in order.

    Returns the 0-based index of the first entry whose stored hash does not
    match its recomputed hash, or whose prev_hash does not match the preceding
    entry's hash. Returns None if the chain is intact; an empty chain is
    intact.
    """
    rows = await conn.fetch(
        """
        SELECT id, entity_type, entity_id, action, details, prev_hash, hash,
               created_at
        FROM audit_log
        WHERE entity_type = $1 AND entity_id = $2
        ORDER BY id ASC
        """,
        entity_type,
        str(entity_id),
    )

    expected_prev = GENESIS_HASH
    for index, row in enumerate(rows):
        if row["prev_hash"] != expected_prev:
            return index

        recomputed = compute_hash(
            row["prev_hash"],
            row["entity_type"],
            row["entity_id"],
            row["action"],
            _decode_details(row["details"]),
            row["created_at"],
        )
        if recomputed != row["hash"]:
            return index

        expected_prev = row["hash"]

    return None


def _decode_details(details) -> dict[str, Any]:
    if details is None:
        return {}
    if isinstance(details, str):
        return json.loads(details)
    return details


def _row_to_entry(row) -> AuditEntry:
    return AuditEntry(
        id=row["id"],
        entity_type=row["entity_type"],
        entity_id=row["entity_id"],
        action=row["action"],
        details=_decode_details(row["details"]),
        prev_hash=row["prev_hash"],
        hash=row["hash"],
        created_at=row["created_at"],
    )
