"""Measurement harness tests.

One assertion in this file is different from every other assertion in the
suite. `test_forbear_beats_the_fixed_schedule_on_net_value` is not checking
that the harness works - it is checking that the thesis is true on this book.
If it fails, the correct response is to find out which of the model, the
generator, the index or the value formula is wrong. Loosening the comparison
would turn the one falsifiable claim in the project into decoration.

The run is expensive by test standards - three policies, a fitted model, and
several thousand rows per world - and that is the cost of measuring the system
rather than a mock of it.
"""

from __future__ import annotations

import pytest

from forbear.services.allocator import (
    SKIP_NEGATIVE_NET_VALUE,
    SKIP_TERMINAL_FAILURE_CLASS,
)
from forbear.services.harness import (
    STRATEGY_FIXED,
    STRATEGY_FORBEAR,
    STRATEGY_UNCONSTRAINED,
    HarnessConfig,
    format_comparison,
    run_comparison,
)

SEED = 42
RECORDS = 500


@pytest.fixture(scope="module")
def comparison_holder() -> dict:
    """Somewhere to keep the one expensive run so every test can read it.

    A module-scoped fixture cannot take the function-scoped database
    connection, so the first test that needs the comparison computes it and the
    rest read it back. Slightly ugly, and much cheaper than fitting the model
    once per assertion.
    """
    return {}


async def get_comparison(conn, comparison_holder: dict):
    if "result" not in comparison_holder:
        comparison_holder["result"] = await run_comparison(
            conn, seed=SEED, n_records=RECORDS
        )
    return comparison_holder["result"]


# --- a, b, g. the run and the table ----------------------------------------


async def test_all_three_strategies_produce_results(conn, comparison_holder):
    result = await get_comparison(conn, comparison_holder)

    assert set(result) == {STRATEGY_FIXED, STRATEGY_FORBEAR, STRATEGY_UNCONSTRAINED}
    for metrics in result.values():
        assert metrics.records_processed == RECORDS
        assert metrics.amount_at_risk > 0


async def test_the_comparison_table_prints(conn, comparison_holder):
    """The table is the deliverable; a run that cannot be read is not one."""
    result = await get_comparison(conn, comparison_holder)

    table = format_comparison(result)
    print("\n" + table)

    assert "NET VALUE" in table
    assert STRATEGY_FORBEAR in table


# --- c. the baseline does not choose ---------------------------------------


async def test_the_fixed_schedule_skips_only_terminal_records(conn, comparison_holder):
    """No scoring means no discretion: everything the rails would accept gets
    its three attempts, including the records that were never going to pay."""
    result = await get_comparison(conn, comparison_holder)
    fixed = result[STRATEGY_FIXED]

    assert set(fixed.skips_by_reason) <= {SKIP_TERMINAL_FAILURE_CLASS}
    assert fixed.attempts_consumed > result[STRATEGY_FORBEAR].attempts_consumed


# --- d. Forbear does choose, and says why ----------------------------------


async def test_forbear_skips_records_on_net_value(conn, comparison_holder):
    """The skip the whole system exists to make."""
    result = await get_comparison(conn, comparison_holder)
    forbear = result[STRATEGY_FORBEAR]

    assert forbear.skips_by_reason.get(SKIP_NEGATIVE_NET_VALUE, 0) >= 1
    assert forbear.ltv_at_risk_from_skips > 0


# --- e. it still recovers money --------------------------------------------


async def test_forbear_recovers_something(conn, comparison_holder):
    """Restraint is only interesting from a policy that also collects. A system
    that skipped everything would score beautifully on churn and be useless."""
    result = await get_comparison(conn, comparison_holder)
    forbear = result[STRATEGY_FORBEAR]

    assert forbear.recovery_rate > 0
    assert forbear.amount_recovered > 0


async def test_forbear_spends_its_attempts_better(conn, comparison_holder):
    """Not the thesis, but the operational claim: a comparable outcome for a
    fraction of the outbound volume."""
    result = await get_comparison(conn, comparison_holder)

    assert (
        result[STRATEGY_FORBEAR].recovered_per_attempt
        > result[STRATEGY_FIXED].recovered_per_attempt
    )


# --- f. the thesis ---------------------------------------------------------


async def test_forbear_beats_the_fixed_schedule_on_net_value(conn, comparison_holder):
    """The claim, stated as an assertion.

    Forbear recovers no more money than the fixed schedule and is worth more
    anyway, because it does not spend the customer to collect the invoice. If
    this fails, something upstream is wrong - the uplift model, the generator,
    the index, or the value formula - and the fix belongs there, not here.
    """
    result = await get_comparison(conn, comparison_holder)
    forbear = result[STRATEGY_FORBEAR]
    fixed = result[STRATEGY_FIXED]

    print(
        f"\nnet value: forbear {forbear.net_value / 100:,.0f} "
        f"vs fixed_schedule {fixed.net_value / 100:,.0f} rupees"
    )
    assert forbear.net_value >= fixed.net_value


async def test_chasing_everyone_costs_more_churn_than_it_recovers(
    conn, comparison_holder
):
    """Why the thesis holds: the aggressive policies contact every
    do_not_disturb in the book, and the subscriptions they lose are worth more
    than the invoices they collect."""
    result = await get_comparison(conn, comparison_holder)

    assert result[STRATEGY_FIXED].churned_count > result[STRATEGY_FORBEAR].churned_count
    assert (
        result[STRATEGY_UNCONSTRAINED].churned_count
        > result[STRATEGY_FORBEAR].churned_count
    )


async def test_the_unconstrained_bound_recovers_the_most_money(conn, comparison_holder):
    """It should. It ignores every rule and retries until the money lands - the
    gap between that recovery and Forbear's is what the constraints and the
    restraint cost together."""
    result = await get_comparison(conn, comparison_holder)

    assert (
        result[STRATEGY_UNCONSTRAINED].amount_recovered
        >= result[STRATEGY_FORBEAR].amount_recovered
    )


# --- shape and reproducibility ---------------------------------------------


async def test_every_metric_the_comparison_promises_is_present(conn, comparison_holder):
    result = await get_comparison(conn, comparison_holder)

    for metrics in result.values():
        for field in (
            "records_processed",
            "amount_at_risk",
            "amount_recovered",
            "recovery_rate",
            "attempts_consumed",
            "recovered_per_attempt",
            "records_skipped",
            "records_blocked",
            "ltv_at_risk_from_skips",
            "churned_count",
            "churn_rate",
            "net_value",
        ):
            assert hasattr(metrics, field), f"comparison is missing {field}"

        assert metrics.net_value == metrics.amount_recovered - metrics.ltv_lost_to_churn


async def test_the_same_seed_produces_the_same_comparison(conn):
    """A headline number that moves between runs is not a measurement."""
    first = await run_comparison(conn, seed=7, n_records=120)
    second = await run_comparison(conn, seed=7, n_records=120)

    for strategy in first:
        assert first[strategy].net_value == second[strategy].net_value
        assert first[strategy].amount_recovered == second[strategy].amount_recovered
        assert first[strategy].churned_count == second[strategy].churned_count


async def test_the_harness_refuses_to_run_outside_a_transaction(db_pool):
    """It inserts three worlds; half of one committed would poison every later
    run against the same database."""
    async with db_pool.acquire() as connection:
        with pytest.raises(RuntimeError, match="transaction"):
            await run_comparison(connection, seed=1, n_records=10)


async def test_a_batch_budget_caps_forbear_without_capping_the_baseline(conn):
    """The merchant's volume ceiling reaches the policy that respects it."""
    result = await run_comparison(
        conn, seed=5, n_records=120, config=HarnessConfig(batch_budget=10)
    )

    assert result[STRATEGY_FORBEAR].attempts_consumed <= 10
    assert result[STRATEGY_FIXED].attempts_consumed > 10
