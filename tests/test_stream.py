"""SSE endpoint tests.

The frontend has no tests - it is a demo surface, and a screen that renders is
its own proof. The stream underneath it is different: it is the only place the
whole system runs end to end inside one request, and a malformed frame there
shows up as a blank page with nothing to debug.

So this checks the contract the screen depends on: frames arrive, each one is
valid JSON, decisions carry what a row needs to render, and refusals carry the
numbers behind them. Small n, because the point is the shape of the stream and
not the size of the book.
"""

from __future__ import annotations

import json

import httpx
import pytest
import pytest_asyncio
from fastapi import FastAPI

from forbear.api import stream

N_RECORDS = 30
SEED = 42


def parse_sse(text: str) -> list[tuple[str, dict]]:
    """Split a raw SSE body into (event, payload) pairs.

    Deliberately hand-rolled rather than taken from a client library: the test
    should fail if the frame format drifts, and a tolerant parser would quietly
    accept a stream no EventSource could read.
    """
    events: list[tuple[str, dict]] = []
    for block in text.split("\n\n"):
        block = block.strip()
        if not block:
            continue
        name = None
        data = None
        for line in block.split("\n"):
            if line.startswith("event: "):
                name = line[len("event: ") :]
            elif line.startswith("data: "):
                data = line[len("data: ") :]
        assert name is not None, f"frame with no event type: {block!r}"
        assert data is not None, f"frame with no data: {block!r}"
        events.append((name, json.loads(data)))
    return events


@pytest_asyncio.fixture
async def client(db_pool):
    """The stream router alone, wired to the test database.

    Not the full application: create_app() opens its own pool against a real
    database on startup, and a test that did that would be measuring whatever
    happened to be on the developer's machine.
    """
    app = FastAPI()
    app.include_router(stream.router)
    app.state.pool = db_pool

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


async def collect(client, **params) -> list[tuple[str, dict]]:
    query = {"seed": SEED, "n": N_RECORDS, **params}
    response = await client.get("/stream/run", params=query, timeout=180.0)
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    return parse_sse(response.text)


@pytest.fixture(scope="module")
def events_holder() -> dict:
    """One stream, read by every assertion: it fits the model and runs four
    allocation passes, which is not something to repeat per test."""
    return {}


async def get_events(client, events_holder: dict):
    if "events" not in events_holder:
        events_holder["events"] = await collect(client)
    return events_holder["events"]


# --- the contract the screen depends on ------------------------------------


async def test_the_stream_delivers_more_than_ten_parseable_events(client, events_holder):
    events = await get_events(client, events_holder)

    assert len(events) >= 10
    for name, payload in events:
        assert isinstance(payload, dict), f"{name} payload is not an object"


async def test_no_event_type_is_a_surprise(client, events_holder):
    """A frame the screen has no listener for is a silent hole in the demo."""
    events = await get_events(client, events_holder)

    assert {name for name, _ in events} <= {"start", "decision", "counter", "summary"}


async def test_every_record_produces_exactly_one_decision(client, events_holder):
    events = await get_events(client, events_holder)
    decisions = [payload for name, payload in events if name == "decision"]

    assert len(decisions) == N_RECORDS
    assert len({d["record_id"] for d in decisions}) == N_RECORDS


async def test_a_decision_carries_what_a_row_needs_to_render(client, events_holder):
    events = await get_events(client, events_holder)
    decisions = [payload for name, payload in events if name == "decision"]

    for decision in decisions:
        for field in (
            "record_id",
            "amount",
            "failure_code",
            "failure_class",
            "cate",
            "whittle_index",
            "action",
        ):
            assert field in decision, f"decision is missing {field}"
        assert decision["action"] in {"scheduled", "skipped"}
        assert isinstance(decision["amount"], int)


async def test_refusals_are_streamed_with_their_numbers(client, events_holder):
    """The refusal list is the strongest thing on the screen, and a refusal
    without its arithmetic is a label."""
    events = await get_events(client, events_holder)
    skips = [
        payload
        for name, payload in events
        if name == "decision" and payload["action"] == "skipped"
    ]

    assert skips, "nothing was refused; the demo has nothing to show"
    for skip in skips:
        assert skip["skip_reason"]
        assert "ltv_at_risk" in skip
        assert isinstance(skip["skip_details"], dict)

    value_skips = [s for s in skips if s["skip_reason"] == "negative_net_value"]
    assert value_skips, "no record was refused on value grounds"
    for skip in value_skips:
        details = skip["skip_details"]
        for field in ("cate", "amount", "ltv_at_risk", "whittle_index", "threshold"):
            assert field in details, f"value skip is missing {field}"


async def test_attempted_records_carry_a_verdict_and_an_outcome(client, events_holder):
    events = await get_events(client, events_holder)
    scheduled = [
        payload
        for name, payload in events
        if name == "decision" and payload["action"] == "scheduled"
    ]

    assert scheduled
    for decision in scheduled:
        assert decision["guard_verdict"] is not None
        assert "allowed" in decision["guard_verdict"]
        assert decision["outcome"] in {"recovered", "failed", "blocked"}
        assert decision["scheduled_at"]
        # Every decision leaves a trace, and the screen shows the head of it.
        assert decision["audit_hash"]


async def test_a_counter_follows_every_decision(client, events_holder):
    events = await get_events(client, events_holder)
    counters = [payload for name, payload in events if name == "counter"]

    assert len(counters) == N_RECORDS
    for counter in counters:
        for field in (
            "total_at_risk",
            "recovered_so_far",
            "recovery_rate",
            "skipped_count",
            "ltv_protected",
            "attempts_consumed",
            "recovered_per_attempt",
            "churned_count",
        ):
            assert field in counter, f"counter is missing {field}"


async def test_the_counters_only_move_forward(client, events_holder):
    """They read as running totals. A counter that went down would make the
    screen look broken even when the arithmetic underneath was right."""
    events = await get_events(client, events_holder)
    counters = [payload for name, payload in events if name == "counter"]

    for field in (
        "recovered_so_far",
        "ltv_protected",
        "skipped_count",
        "attempts_consumed",
    ):
        values = [counter[field] for counter in counters]
        assert values == sorted(values), f"{field} moved backwards"


async def test_the_summary_closes_the_stream_with_all_three_strategies(
    client, events_holder
):
    events = await get_events(client, events_holder)

    assert events[-1][0] == "summary", "the summary is not the last frame"
    summary = events[-1][1]
    assert set(summary["strategies"]) == {
        "fixed_schedule",
        "forbear",
        "unconstrained",
    }
    for metrics in summary["strategies"].values():
        for field in (
            "amount_at_risk",
            "amount_recovered",
            "recovery_rate",
            "attempts_consumed",
            "recovered_per_attempt",
            "records_skipped",
            "churned_count",
            "ltv_lost_to_churn",
            "net_value",
        ):
            assert field in metrics, f"summary is missing {field}"


# --- the answer key ---------------------------------------------------------


async def test_demo_mode_off_withholds_the_segment(client):
    """The segment does not exist in production. It is echoed for the demo and
    never used to decide anything, and the flag is what makes that checkable.

    N_RECORDS rather than a handful: a book too small for one treatment arm to
    contain both outcomes is one the uplift model refuses to fit, and the
    stream then correctly carries an error instead of decisions.
    """
    events = await collect(client, n=N_RECORDS, demo_mode=False)
    decisions = [payload for name, payload in events if name == "decision"]

    assert decisions, "no decisions to check for a leaked segment"
    assert all("segment" not in decision for decision in decisions)


async def test_demo_mode_on_includes_the_segment(client):
    events = await collect(client, n=N_RECORDS, demo_mode=True)
    decisions = [payload for name, payload in events if name == "decision"]

    assert decisions
    assert all("segment" in decision for decision in decisions)
    assert {d["segment"] for d in decisions} <= {
        "sure_thing",
        "persuadable",
        "lost_cause",
        "do_not_disturb",
    }


# --- refusals ---------------------------------------------------------------


async def test_the_run_leaves_the_database_as_it_found_it(client, db_pool):
    """The cycle is rolled back so the demo can be run twice with the same seed
    and show the same thing."""
    async with db_pool.acquire() as conn:
        before = await conn.fetchval("SELECT count(*) FROM at_risk_records")

    await collect(client, n=N_RECORDS)

    async with db_pool.acquire() as conn:
        after = await conn.fetchval("SELECT count(*) FROM at_risk_records")

    assert after == before


@pytest.mark.parametrize("n", [0, 5000])
async def test_an_out_of_range_batch_size_is_refused(client, n):
    response = await client.get("/stream/run", params={"seed": 1, "n": n})

    assert response.status_code == 422
