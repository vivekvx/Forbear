"""Sensitivity sweep tests.

The sweep answers a question the rest of the system can only assume: how bad
does dunning churn have to be before restraint pays. These tests check the two
ends of that curve, where the answer is known in advance from the structure of
the problem rather than from the model.

At a churn rate of zero, contact costs nothing, so the policy that contacts
everyone should win - and if Forbear won there anyway, the comparison would be
measuring something other than churn. At a rate of one, every contacted
do_not_disturb cancels, and a policy that cannot tell them apart is buying
invoices with subscriptions.

Both ends are assertions about the world, not about the implementation. The
crossover between them is the finding.
"""

from __future__ import annotations

import csv

import pytest

from forbear.services.harness import STRATEGY_FORBEAR, STRATEGY_UNCONSTRAINED
from forbear.services.sensitivity import (
    CSV_COLUMNS,
    DEFAULT_CHURN_RATES,
    find_crossover,
    format_sweep,
    run_sweep,
    write_sweep_csv,
)

SEED = 42
RECORDS = 500
RATES = (0.0, 0.5, 1.0)


@pytest.fixture(scope="module")
def sweep_holder() -> dict:
    """One sweep, read by every assertion. Three full comparisons is the most
    expensive thing in the suite; running it per test would be indefensible."""
    return {}


async def get_sweep(conn, sweep_holder: dict):
    if "points" not in sweep_holder:
        sweep_holder["points"] = await run_sweep(
            conn, seed=SEED, n_records=RECORDS, churn_rates=RATES
        )
    return sweep_holder["points"]


def at(points, rate: float):
    for churn_rate, result in points:
        if churn_rate == rate:
            return result
    raise AssertionError(f"sweep has no point at churn rate {rate}")


# --- a. the sweep runs ------------------------------------------------------


async def test_the_sweep_returns_one_comparison_per_rate(conn, sweep_holder):
    points = await get_sweep(conn, sweep_holder)

    assert [rate for rate, _ in points] == list(RATES)
    for _, result in points:
        assert set(result) >= {STRATEGY_FORBEAR, STRATEGY_UNCONSTRAINED}


# --- b. no churn cost, no reason for restraint ------------------------------


async def test_with_no_churn_cost_chasing_everyone_wins(conn, sweep_holder):
    """The honest half of the finding.

    At zero churn there is no such thing as a do_not_disturb: contact is free,
    every extra attempt is upside, and a policy that declines to make them is
    simply leaving money on the table. A sweep where Forbear won here too would
    mean the comparison was rewarding restraint for its own sake.
    """
    points = await get_sweep(conn, sweep_holder)
    result = at(points, 0.0)
    forbear = result[STRATEGY_FORBEAR]
    unconstrained = result[STRATEGY_UNCONSTRAINED]

    print(
        f"\nchurn 0.0 -> forbear {forbear.net_value / 100:,.0f} vs "
        f"unconstrained {unconstrained.net_value / 100:,.0f} rupees"
    )

    assert forbear.churned_count == 0
    assert unconstrained.churned_count == 0
    # Nothing is lost to churn on either side, so net value is recovery alone.
    assert forbear.net_value == forbear.amount_recovered
    assert unconstrained.net_value >= forbear.net_value


# --- c. maximum churn cost, restraint wins ---------------------------------


async def test_with_certain_churn_selective_chasing_wins(conn, sweep_holder):
    """Every contacted do_not_disturb cancels. A policy that cannot tell them
    apart pays for each invoice with a subscription worth twelve of them."""
    points = await get_sweep(conn, sweep_holder)
    result = at(points, 1.0)
    forbear = result[STRATEGY_FORBEAR]
    unconstrained = result[STRATEGY_UNCONSTRAINED]

    print(
        f"churn 1.0 -> forbear {forbear.net_value / 100:,.0f} vs "
        f"unconstrained {unconstrained.net_value / 100:,.0f} rupees"
    )

    assert unconstrained.churned_count > forbear.churned_count
    assert forbear.net_value > unconstrained.net_value


async def test_churn_cost_rises_with_the_churn_rate(conn, sweep_holder):
    """The sweep is varying what it claims to vary."""
    points = await get_sweep(conn, sweep_holder)

    losses = [result[STRATEGY_UNCONSTRAINED].ltv_lost_to_churn for _, result in points]
    assert losses == sorted(losses)
    assert losses[-1] > losses[0]


# --- d. the crossover -------------------------------------------------------


async def test_the_crossover_is_found_and_printed(conn, sweep_holder):
    """The number this whole project exists to produce.

    Reported as a sampled rate rather than an interpolated root: three points
    can say "between here and there", and pretending to more precision than the
    sweep has would be the easiest lie in the codebase.
    """
    points = await get_sweep(conn, sweep_holder)

    print("\n" + format_sweep(points))

    crossover = find_crossover(points)
    assert crossover is not None, "forbear never overtook unconstrained"
    assert 0.0 < crossover <= 1.0
    print(
        f"\ncrossover at churn rate {crossover:.2f}: below it, chase everything; "
        f"at or above it, be selective"
    )


def test_a_sweep_that_never_crosses_reports_none():
    """Not a failure. If the ordering never changes, saying so is the result."""

    class Fake:
        def __init__(self, net_value):
            self.net_value = net_value

    points = [
        (0.0, {STRATEGY_FORBEAR: Fake(10), STRATEGY_UNCONSTRAINED: Fake(99)}),
        (1.0, {STRATEGY_FORBEAR: Fake(20), STRATEGY_UNCONSTRAINED: Fake(99)}),
    ]

    assert find_crossover(points) is None
    assert "none in the sampled range" in format_sweep(points)


# --- the CSV ----------------------------------------------------------------


async def test_the_sweep_writes_a_readable_csv(conn, sweep_holder, tmp_path):
    points = await get_sweep(conn, sweep_holder)
    destination = write_sweep_csv(points, tmp_path / "sweep.csv")

    with destination.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert list(rows[0]) == list(CSV_COLUMNS)
    assert len(rows) == len(RATES) * 3
    assert {row["strategy"] for row in rows} == {
        "fixed_schedule",
        "forbear",
        "unconstrained",
    }
    assert {float(row["churn_rate"]) for row in rows} == set(RATES)


async def test_run_sweep_writes_the_csv_when_asked(conn, tmp_path):
    destination = tmp_path / "nested" / "sweep.csv"

    await run_sweep(
        conn, seed=3, n_records=80, churn_rates=(0.0, 1.0), csv_path=destination
    )

    assert destination.exists()
    assert destination.read_text(encoding="utf-8").startswith("churn_rate,strategy")


# --- refusals and defaults --------------------------------------------------


def test_the_default_rates_span_the_whole_range():
    """A sweep stopping short of 1.0 could not show the crossover at all for a
    book where dunning churn is severe."""
    assert DEFAULT_CHURN_RATES[0] == 0.0
    assert DEFAULT_CHURN_RATES[-1] == 1.0
    assert list(DEFAULT_CHURN_RATES) == sorted(DEFAULT_CHURN_RATES)


async def test_a_rate_outside_zero_to_one_is_refused(conn):
    with pytest.raises(ValueError):
        await run_sweep(conn, seed=1, n_records=10, churn_rates=(0.5, 1.5))


def test_an_empty_sweep_is_reported_rather_than_crashing():
    assert find_crossover([]) is None
    assert "no points" in format_sweep([])
