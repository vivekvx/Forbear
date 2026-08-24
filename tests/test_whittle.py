"""Whittle index tests.

The index is arithmetic, so these tests are about the arithmetic meaning the
right thing rather than about numerical accuracy. Three properties matter to
the allocator: the sign is a decision (negative means never spend here), the
ordering is a budget (descending is the order attempts are handed out in), and
the magnitude is denominated in attempts (an index of 3.0 means three attempts'
worth of value returned for one spent).

No model here, no database, no clock. A test that needed any of those would be
testing something other than the index.
"""

from __future__ import annotations

import pytest

from forbear.scoring.whittle import (
    DEFAULT_ATTEMPT_COST_PAISE,
    DEFAULT_RECOVERY_SURVIVAL_RATE,
    DEFAULT_REMAINING_MONTHS,
    RecordScore,
    compute_indices,
    compute_whittle_index,
)

# A mid-tier plan whose invoice is one month's charge - the ordinary case.
PLAN = 499_00


def record(
    cate: float,
    amount: int = PLAN,
    plan_amount: int = PLAN,
    record_id: str = "cust_000001",
) -> RecordScore:
    return RecordScore(
        record_id=record_id, amount=amount, plan_amount=plan_amount, cate=cate
    )


# --- a, b. the sign is the decision ----------------------------------------


def test_positive_cate_gives_a_positive_index():
    """Contact helps: the invoice and the subscription behind it are both worth
    something, and the record belongs in the budget."""
    assert compute_whittle_index(record(cate=0.4)) > 0


def test_negative_cate_gives_a_negative_index():
    """Contact destroys more than it creates. The allocator must never spend an
    attempt here, and the guard re-checks that it did not."""
    assert compute_whittle_index(record(cate=-0.4)) < 0


def test_zero_cate_gives_a_zero_index():
    """A customer who pays or does not regardless is worth exactly nothing to
    chase - not a small positive amount."""
    assert compute_whittle_index(record(cate=0.0)) == 0.0


@pytest.mark.parametrize("cate", [0.9, 0.4, 0.05, 0.0, -0.05, -0.4, -0.9])
def test_the_index_moves_monotonically_with_the_estimated_effect(cate):
    """Whatever else the formula does, a stronger effect in either direction
    must not reverse the ranking: the allocator's whole strategy is to sort by
    this number."""
    stronger = compute_whittle_index(record(cate=cate + 0.01))
    weaker = compute_whittle_index(record(cate=cate))

    assert stronger > weaker


# --- c. bigger invoice, bigger index ---------------------------------------


def test_a_larger_invoice_scores_higher_at_the_same_cate():
    """Same customer, same estimated effect, more money on the table."""
    large = compute_whittle_index(record(cate=0.3, amount=999_00))
    small = compute_whittle_index(record(cate=0.3, amount=149_00))

    assert large > small


def test_the_index_reads_as_value_per_attempt():
    """The denominator is what makes indices comparable across plan sizes: a
    record scoring 1.0 exactly pays for the attempt it consumes."""
    scored = record(cate=0.5)
    index = compute_whittle_index(scored)

    remaining_ltv = scored.plan_amount * DEFAULT_REMAINING_MONTHS
    expected_net = (
        scored.cate * scored.amount
        + scored.cate * remaining_ltv * DEFAULT_RECOVERY_SURVIVAL_RATE
    )
    assert index == pytest.approx(expected_net / DEFAULT_ATTEMPT_COST_PAISE)


def test_a_cheaper_attempt_makes_the_same_record_worth_more():
    expensive = compute_whittle_index(record(cate=0.3), attempt_cost_paise=1000)
    cheap = compute_whittle_index(record(cate=0.3), attempt_cost_paise=100)

    assert cheap > expensive


# --- e. lifetime value at risk ---------------------------------------------


def test_more_lifetime_value_at_risk_lowers_a_negative_index():
    """The churn case. Two customers the model believes contact would drive
    away; the one with the larger subscription behind them has more to lose, so
    chasing them is the worse idea of the two.

    The invoice is held constant so the only thing moving is the lifetime value
    - otherwise this would be re-testing the amount term.
    """
    premium = compute_whittle_index(
        record(cate=-0.3, amount=PLAN, plan_amount=1_999_00)
    )
    budget = compute_whittle_index(record(cate=-0.3, amount=PLAN, plan_amount=149_00))

    assert premium < budget


def test_more_lifetime_value_raises_a_positive_index():
    """The mirror image: when contact helps, a bigger subscription behind the
    invoice is more value retained, not less."""
    premium = compute_whittle_index(record(cate=0.3, amount=PLAN, plan_amount=1_999_00))
    budget = compute_whittle_index(record(cate=0.3, amount=PLAN, plan_amount=149_00))

    assert premium > budget


def test_a_longer_horizon_scales_the_lifetime_term():
    twelve = compute_whittle_index(record(cate=0.3), remaining_months=12)
    twenty_four = compute_whittle_index(record(cate=0.3), remaining_months=24)

    assert twenty_four > twelve


def test_a_zero_horizon_leaves_only_the_invoice():
    """With no future to protect, the index is the invoice alone. This is the
    knob that says how much of the decision is retention rather than cash."""
    scored = record(cate=0.5)
    index = compute_whittle_index(scored, remaining_months=0)

    assert index == pytest.approx(
        scored.cate * scored.amount / DEFAULT_ATTEMPT_COST_PAISE
    )


# --- d. the ordering is the budget -----------------------------------------


def test_compute_indices_returns_records_sorted_descending():
    records = [
        record(cate=-0.4, record_id="churn_risk"),
        record(cate=0.8, record_id="persuadable"),
        record(cate=0.02, record_id="sure_thing"),
    ]

    ordered = compute_indices(records)

    assert [scored.record_id for scored in ordered] == [
        "persuadable",
        "sure_thing",
        "churn_risk",
    ]
    indices = [scored.whittle_index for scored in ordered]
    assert indices == sorted(indices, reverse=True)


def test_compute_indices_attaches_the_index_to_each_record():
    records = [record(cate=0.3, record_id=f"cust_{n}") for n in range(5)]

    compute_indices(records)

    assert all(scored.whittle_index is not None for scored in records)
    assert all(
        scored.whittle_index == compute_whittle_index(scored) for scored in records
    )


def test_compute_indices_does_not_filter_negative_records():
    """Dropping them here would be a decision, and this module does not make
    decisions. The allocator skips them - and writes a skip reason and an audit
    entry when it does, which cannot happen for a record that vanished during
    scoring."""
    records = [
        record(cate=-0.5, record_id="a"),
        record(cate=0.5, record_id="b"),
    ]

    ordered = compute_indices(records)

    assert len(ordered) == 2
    assert ordered[-1].whittle_index < 0


def test_ties_are_broken_deterministically():
    """A batch that reorders itself between runs makes every downstream
    comparison unreproducible for no visible reason."""
    records = [record(cate=0.3, record_id=name) for name in ("c", "a", "b")]

    ordered = compute_indices(records)

    assert [scored.record_id for scored in ordered] == ["a", "b", "c"]


def test_compute_indices_handles_an_empty_batch():
    assert compute_indices([]) == []


# --- refusals --------------------------------------------------------------


def test_a_free_attempt_is_refused():
    """Zero cost makes every index infinite and the ranking meaningless."""
    with pytest.raises(ValueError):
        compute_whittle_index(record(cate=0.3), attempt_cost_paise=0)


def test_a_negative_horizon_is_refused():
    with pytest.raises(ValueError):
        compute_whittle_index(record(cate=0.3), remaining_months=-1)


@pytest.mark.parametrize("bad_rate", [-0.1, 1.5])
def test_a_survival_rate_outside_zero_to_one_is_refused(bad_rate):
    with pytest.raises(ValueError):
        compute_whittle_index(record(cate=0.3), recovery_survival_rate=bad_rate)


def test_the_documented_defaults_are_the_defaults():
    """These are the numbers the architecture document quotes. Changing one
    silently would make every index in the write-up wrong."""
    assert DEFAULT_REMAINING_MONTHS == 12
    assert DEFAULT_RECOVERY_SURVIVAL_RATE == 0.85
    assert DEFAULT_ATTEMPT_COST_PAISE == 200  # 2.00 rupees
