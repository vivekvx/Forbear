"""Decisioning is the write side of the worklist's latency contract.

Where test_worklist.py proves the read never scores or allocates, this file
proves the opposite: decide_record (the background job that runs right after
ingestion) DOES call the real uplift model and the real allocator - there is
no separate scoring path for the merchant view.
"""

from __future__ import annotations

import inspect

import pytest

import forbear.services.decisioning as decisioning
from forbear.services import allocator
from tests.conftest import insert_scenario

pytestmark = pytest.mark.asyncio


async def _set_failure(conn, record_id: int, code: str, cls: str) -> None:
    await conn.execute(
        """
        UPDATE at_risk_records SET failure_code = $2, failure_class = $3::failure_class
        WHERE id = $1
        """,
        record_id,
        code,
        cls,
    )


class _FakeModel:
    def __init__(self):
        self.cate_calls = []
        self.recovery_calls = []

    def predict_cate(self, X):
        self.cate_calls.append(X)
        return [0.6] * len(X)

    def predict_recovery_probability(self, X):
        self.recovery_calls.append(X)
        return [0.5] * len(X)


async def test_decide_record_calls_the_real_model_and_allocator(conn, monkeypatch):
    """The opposite of test_worklist's proof: ingestion-time decisioning must
    reach the loaded model and forbear.services.allocator.allocate()."""
    scenario = await insert_scenario(conn, status="open", suffix="wired")
    await _set_failure(conn, scenario["record_id"], "GATEWAY_ERROR", "transient")

    fake_model = _FakeModel()
    monkeypatch.setattr(decisioning, "_get_model", lambda: fake_model)

    allocate_calls = []
    real_allocate = allocator.allocate

    async def _spy_allocate(conn_, records, config=None, commit=True):
        allocate_calls.append(records)
        return await real_allocate(conn_, records, config, commit=commit)

    monkeypatch.setattr(decisioning, "allocate", _spy_allocate)

    bucket = await decisioning.decide_record(conn, scenario["record_id"])

    assert fake_model.cate_calls, "the loaded model's predict_cate was never called"
    assert fake_model.recovery_calls, "predict_recovery_probability was never called"
    assert allocate_calls, "forbear.services.allocator.allocate was never called"
    assert allocate_calls[0][0].record_id == scenario["record_id"]
    assert bucket == "chase"  # positive CATE, transient failure -> scheduled


async def test_decisioning_leaves_no_trace_of_a_heuristic_table(conn):
    """The heuristic CATE table must not exist anywhere in the module."""
    source = inspect.getsource(decisioning)
    assert "_HEURISTIC_CATE" not in source
    assert not hasattr(decisioning, "_HEURISTIC_CATE")


async def test_ingestion_still_leaves_the_record_open(conn):
    """allocate() transitions a record; the preview must roll that back."""
    scenario = await insert_scenario(conn, status="open", suffix="stays_open")
    await _set_failure(conn, scenario["record_id"], "GATEWAY_ERROR", "transient")

    await decisioning.decide_record(conn, scenario["record_id"])

    row = await conn.fetchrow(
        "SELECT status, worklist_bucket FROM at_risk_records WHERE id = $1",
        scenario["record_id"],
    )
    assert row["status"] == "open"
    assert row["worklist_bucket"] == "chase"


async def test_worklist_bucket_matches_a_direct_call_to_the_real_allocator(conn):
    """End to end: the persisted bucket must match what allocate() itself
    would do, scored by the same loaded model, for the same record."""
    scenario = await insert_scenario(conn, status="open", suffix="e2e")
    await _set_failure(conn, scenario["record_id"], "GATEWAY_ERROR", "transient")
    record_id = scenario["record_id"]

    await decisioning.decide_record(conn, record_id)
    row = await conn.fetchrow(
        "SELECT worklist_bucket, worklist_scheduled_at FROM at_risk_records WHERE id = $1",
        record_id,
    )

    # Re-derive independently: same model, same features, a fresh allocate()
    # call in preview mode (commit=False), since the record cannot be
    # transitioned twice.
    model = decisioning._get_model()
    facts_row = await decisioning._facts(conn, record_id)
    feature_row = decisioning._build_feature_row(facts_row)
    from forbear.scoring.uplift import build_feature_matrix
    from forbear.scoring.whittle import RecordScore, compute_whittle_index
    from forbear.services.allocator import AllocationConfig, ScoredRecord, allocate

    X = build_feature_matrix([feature_row])
    cate = float(model.predict_cate(X)[0])
    whittle_index = compute_whittle_index(
        RecordScore(
            record_id=str(record_id),
            amount=facts_row["amount"],
            plan_amount=facts_row["plan_amount"],
            cate=cate,
        )
    )
    scored = ScoredRecord(record_id=record_id, cate=cate, whittle_index=whittle_index)
    plan = await allocate(conn, [scored], AllocationConfig(), commit=False)

    if plan.scheduled:
        assert row["worklist_bucket"] == "chase"
        assert row["worklist_scheduled_at"] == plan.scheduled[0].scheduled_at
    else:
        assert row["worklist_bucket"] in ("wait", "leave_alone")
