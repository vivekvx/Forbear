"""Generator tests.

Two things are being tested here, and only one of them is code. The first is
ordinary: seeds reproduce, the mix is the mix, dates land where they should.
The second is the property the whole evaluation rests on - that treatment is
assigned independently of the causal segment, and that the observed outcomes
therefore carry real incremental signal rather than an echo of the assignment
rule. A generator that fails those tests produces numbers that look like
evidence and are not.

No database here: the generator touches none. The suite's other files need
PostgreSQL; this one is pure.
"""

from __future__ import annotations

import ast
import math
import pathlib
from collections import Counter
from datetime import date, timedelta

import pytest

from forbear.generator.batch_generator import (
    LIVE_MANDATE_CODE_WEIGHTS,
    generate_batch,
)
from forbear.generator.customer_profiles import (
    SEGMENT_MIX,
    CustomerProfile,
    Segment,
    generate_profiles,
)
from forbear.generator.outcome_simulator import (
    LOST_CAUSE_RECOVERY_RATE,
    SURE_THING_FAILURE_RATE,
    simulate_outcomes,
)
from forbear.generator.treatment_assignment import assign_treatment
from forbear.models.models import FailureClass, MandateStatus
from forbear.services.classifier import classify, known_codes

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent

BILLING_DATE = date(2026, 8, 24)


def run(n: int, seed: int = 7, treatment_rate: float = 0.5):
    """One end-to-end run: profiles, batch, assignment, outcomes."""
    profiles = generate_profiles(n, seed=seed)
    batch = generate_batch(profiles, BILLING_DATE, seed=seed)
    assignments = assign_treatment(batch, treatment_rate, seed=seed)
    outcomes = simulate_outcomes(batch, assignments, seed=seed)
    return profiles, batch, assignments, outcomes


def segment_of(record) -> Segment:
    return Segment(record.ground_truth["segment"])


def chi_square_p_value(table: list[list[int]]) -> float:
    """Pearson chi-square for a 4x2 table, p-value from the df=3 closed form.

    Hand-rolled rather than pulled from scipy: this is the only place in the
    suite that needs a distribution function, and the df here is fixed at three
    (four segments, two arms). For df=3 the survival function is exact in terms
    of erfc, so there is nothing to approximate.
    """
    row_totals = [sum(row) for row in table]
    col_totals = [sum(column) for column in zip(*table)]
    total = sum(row_totals)

    statistic = 0.0
    for row_index, row in enumerate(table):
        for col_index, observed in enumerate(row):
            expected = row_totals[row_index] * col_totals[col_index] / total
            assert expected > 5, "chi-square needs expected counts above 5"
            statistic += (observed - expected) ** 2 / expected

    # P(X > x) for df=3.
    return math.erfc(math.sqrt(statistic / 2)) + math.sqrt(
        2 * statistic / math.pi
    ) * math.exp(-statistic / 2)


def rate(outcomes, records, predicate) -> float:
    selected = [outcomes[record.customer_id] for record in records]
    matching = [outcome for outcome in selected if predicate(outcome)]
    return len(matching) / len(selected) if selected else 0.0


# --- a. reproducibility ----------------------------------------------------


def test_same_seed_produces_identical_batches():
    """Every stage, not just the batch: one unseeded draw anywhere breaks all
    comparison between runs, and it would break silently."""
    first = run(500, seed=11)
    second = run(500, seed=11)

    for left, right in zip(first, second):
        assert left == right


def test_different_seeds_produce_different_batches():
    """The reproducibility test above passes trivially if nothing is random."""
    _, batch_a, _, _ = run(500, seed=11)
    _, batch_b, _, _ = run(500, seed=12)

    assert batch_a != batch_b


def test_ground_truth_survives_regeneration_unchanged():
    """The answer key is part of the reproducible state, not a side effect."""
    _, batch_a, _, _ = run(200, seed=3)
    _, batch_b, _, _ = run(200, seed=3)

    assert [record.ground_truth for record in batch_a] == [
        record.ground_truth for record in batch_b
    ]


# --- b. segment distribution ----------------------------------------------


def test_segment_distribution_matches_the_target_mix():
    profiles = generate_profiles(1000, seed=5)
    counts = Counter(profile.segment for profile in profiles)

    for segment, target in SEGMENT_MIX.items():
        observed = counts[segment] / len(profiles)
        assert abs(observed - target) < 0.05, f"{segment}: {observed:.3f} vs {target}"


def test_profile_fields_stay_inside_their_declared_ranges():
    profiles = generate_profiles(1000, seed=5)

    for profile in profiles:
        assert 1 <= profile.salary_day <= 28
        assert 0 <= profile.salary_variance_days <= 5
        assert profile.monthly_income > 0
        assert 0.0 < profile.monthly_spend_rate <= 1.45
        assert profile.plan_amount > 0
        assert 1 <= profile.subscription_age_months <= 60
        assert isinstance(profile.mandate_status, MandateStatus)


def test_only_do_not_disturbs_are_contact_sensitive():
    """Invariant of the segment definitions: churning on contact is what makes
    a do_not_disturb one, so a non-zero sensitivity anywhere else is a bug in
    the world model, not a tuning choice."""
    for profile in generate_profiles(1000, seed=5):
        if profile.segment is Segment.DO_NOT_DISTURB:
            assert profile.contact_sensitivity > 0
        else:
            assert profile.contact_sensitivity == 0.0


def test_salary_days_cluster_rather_than_spread_uniformly():
    """A uniform salary_day would make time-dependent recovery far easier to
    predict here than it is in production."""
    profiles = generate_profiles(2000, seed=5)
    counts = Counter(profile.salary_day for profile in profiles)

    # 1-3, 13-17 and 23-27 are the month-start, mid-month and month-end runs.
    clustered = sum(
        count
        for day, count in counts.items()
        if day <= 3 or 13 <= day <= 17 or 23 <= day <= 27
    )
    uniform_share = (3 + 5 + 5) / 28
    assert clustered / len(profiles) > uniform_share * 1.5


def test_income_is_right_skewed():
    """Log-normal, so the mean sits above the median. A symmetric income
    distribution would put too much of the book at the same ability to pay."""
    incomes = sorted(profile.monthly_income for profile in generate_profiles(2000, 5))
    median = incomes[len(incomes) // 2]
    mean = sum(incomes) / len(incomes)

    assert mean > median


# --- ground truth encodes the causal structure -----------------------------


@pytest.mark.parametrize(
    "segment,without,with_contact",
    [
        (Segment.SURE_THING, True, True),
        (Segment.PERSUADABLE, False, True),
        (Segment.LOST_CAUSE, False, False),
        (Segment.DO_NOT_DISTURB, True, True),
    ],
)
def test_ground_truth_matches_the_segment_definition(segment, without, with_contact):
    _, batch, _, _ = run(1000, seed=9)
    records = [record for record in batch if segment_of(record) is segment]
    assert records, f"no {segment} records generated"

    for record in records:
        truth = record.ground_truth
        assert truth["would_pay_without_contact"] is without
        assert truth["would_pay_with_contact"] is with_contact
        assert (truth["would_pay_without_date"] is not None) is without
        assert (truth["would_pay_with_date"] is not None) is with_contact


def test_only_do_not_disturbs_can_churn():
    _, batch, _, _ = run(1000, seed=9)

    for record in batch:
        if segment_of(record) is not Segment.DO_NOT_DISTURB:
            assert record.ground_truth["would_churn_if_contacted"] is False


def test_time_dependent_pay_dates_track_salary_day_and_variance():
    """A customer paid on the 7th with two days of variance may pay between the
    5th and the 9th - and never before the debit that failed."""
    profiles = generate_profiles(1000, seed=13)
    batch = generate_batch(profiles, BILLING_DATE, seed=13)
    by_id = {profile.customer_id: profile for profile in profiles}

    checked = 0
    for record in batch:
        if classify(record.failure_code) is not FailureClass.TIME_DEPENDENT:
            continue
        pay_date = record.ground_truth["would_pay_without_date"]
        if pay_date is None:
            continue

        profile = by_id[record.customer_id]
        scheduled = date(BILLING_DATE.year, BILLING_DATE.month, profile.salary_day)
        if scheduled <= BILLING_DATE:
            year, month = (
                (scheduled.year + 1, 1)
                if scheduled.month == 12
                else (scheduled.year, scheduled.month + 1)
            )
            scheduled = date(year, month, profile.salary_day)

        earliest = max(
            scheduled - timedelta(days=profile.salary_variance_days),
            BILLING_DATE + timedelta(days=1),
        )
        latest = scheduled + timedelta(days=profile.salary_variance_days)
        assert earliest <= pay_date <= latest
        checked += 1

    assert checked > 100, "too few time-dependent records to be meaningful"


def test_persuadables_pay_within_three_days_of_contact():
    _, batch, _, _ = run(1000, seed=13)

    for record in batch:
        if segment_of(record) is not Segment.PERSUADABLE:
            continue
        pay_date = record.ground_truth["would_pay_with_date"]
        assert BILLING_DATE < pay_date <= BILLING_DATE + timedelta(days=3)


def test_failure_codes_come_from_the_classifier_table():
    """An invented code would land on the exception list rather than being
    classified, and the batch would be exercising the wrong path."""
    _, batch, _, _ = run(1000, seed=13)
    codes = {record.failure_code for record in batch}

    assert codes <= known_codes()


def test_insufficient_funds_dominates_the_code_mix():
    _, batch, _, _ = run(2000, seed=13)
    codes = Counter(record.failure_code for record in batch)

    assert codes["INSUFFICIENT_FUNDS"] / len(batch) > 0.5
    assert codes["INSUFFICIENT_FUNDS"] == max(codes.values())


def test_dead_mandates_decline_for_the_reason_they_are_dead():
    profiles = generate_profiles(1000, seed=13)
    batch = generate_batch(profiles, BILLING_DATE, seed=13)
    by_id = {profile.customer_id: profile for profile in profiles}

    for record in batch:
        status = by_id[record.customer_id].mandate_status
        if status is MandateStatus.REVOKED:
            assert record.failure_code == "MANDATE_REVOKED"
        elif status is MandateStatus.EXPIRED:
            assert record.failure_code == "MANDATE_EXPIRED"
        else:
            assert record.failure_code in LIVE_MANDATE_CODE_WEIGHTS


# --- c. treatment assignment is independent of segment ---------------------


def test_treatment_assignment_is_independent_of_segment():
    """The whole experiment rests on this. If assignment correlated with
    segment, the treated and control arms would differ in composition and the
    measured uplift would be that difference, not the effect of contact."""
    _, batch, assignments, _ = run(2000, seed=21)

    table = []
    for segment in Segment:
        records = [record for record in batch if segment_of(record) is segment]
        treated = sum(assignments[record.customer_id] for record in records)
        table.append([treated, len(records) - treated])

    p_value = chi_square_p_value(table)
    assert p_value > 0.05, f"assignment looks segment-dependent (p={p_value:.4f})"


def test_treatment_rate_is_honoured():
    _, batch, assignments, _ = run(2000, seed=21, treatment_rate=0.3)
    treated_share = sum(assignments.values()) / len(batch)

    assert abs(treated_share - 0.3) < 0.05


def test_assignment_ignores_everything_observable_too():
    """Not just segment: an assignment correlated with the amount or the
    decline code would confound the estimate exactly as badly."""
    _, batch, assignments, _ = run(2000, seed=21)

    treated = [record for record in batch if assignments[record.customer_id]]
    control = [record for record in batch if not assignments[record.customer_id]]

    mean_treated = sum(record.amount for record in treated) / len(treated)
    mean_control = sum(record.amount for record in control) / len(control)
    assert abs(mean_treated - mean_control) / mean_control < 0.10

    share_treated = sum(
        record.failure_code == "INSUFFICIENT_FUNDS" for record in treated
    ) / len(treated)
    share_control = sum(
        record.failure_code == "INSUFFICIENT_FUNDS" for record in control
    ) / len(control)
    assert abs(share_treated - share_control) < 0.05


@pytest.mark.parametrize("bad_rate", [0.0, 1.0, -0.1, 1.5])
def test_degenerate_treatment_rates_are_refused(bad_rate):
    """At 0 or 1 there is no counterfactual arm left to compare against."""
    _, batch, _, _ = run(50, seed=21)

    with pytest.raises(ValueError):
        assign_treatment(batch, bad_rate, seed=1)


# --- d. uplift exists where it should and nowhere else ---------------------


def test_sure_things_show_no_uplift_and_persuadables_show_a_lot():
    _, batch, assignments, outcomes = run(2000, seed=33)

    def arm(segment: Segment, treated: bool):
        return [
            record
            for record in batch
            if segment_of(record) is segment
            and assignments[record.customer_id] is treated
        ]

    def recovered(outcome):
        return outcome.recovered

    sure_treated = rate(outcomes, arm(Segment.SURE_THING, True), recovered)
    sure_control = rate(outcomes, arm(Segment.SURE_THING, False), recovered)
    # Contact does not put money in an account before payday. Any gap here is
    # sampling noise plus the 5% failure rate, not an effect.
    assert abs(sure_treated - sure_control) < 0.05
    assert sure_treated > 0.85 and sure_control > 0.85

    persuadable_treated = rate(outcomes, arm(Segment.PERSUADABLE, True), recovered)
    persuadable_control = rate(outcomes, arm(Segment.PERSUADABLE, False), recovered)
    # The direction is what matters, not the number: this is the only segment
    # where spending an attempt changes the outcome.
    assert persuadable_treated - persuadable_control > 0.8
    assert persuadable_control == 0.0

    lost_treated = rate(outcomes, arm(Segment.LOST_CAUSE, True), recovered)
    lost_control = rate(outcomes, arm(Segment.LOST_CAUSE, False), recovered)
    assert abs(lost_treated - lost_control) < 0.05
    assert lost_treated < 0.10


def test_noise_prevents_perfect_separation():
    """If every sure_thing recovered and no lost_cause ever did, a model could
    hit perfect separation, and perfect separation on synthetic data means the
    data was built from the model's own assumptions."""
    _, batch, _, outcomes = run(4000, seed=41)

    sure_things = [
        record for record in batch if segment_of(record) is Segment.SURE_THING
    ]
    lost_causes = [
        record for record in batch if segment_of(record) is Segment.LOST_CAUSE
    ]

    sure_failures = 1 - rate(outcomes, sure_things, lambda o: o.recovered)
    lost_recoveries = rate(outcomes, lost_causes, lambda o: o.recovered)

    assert sure_failures > 0.0, "no sure_thing ever failed: noise is missing"
    assert lost_recoveries > 0.0, "no lost_cause ever paid: noise is missing"
    assert abs(sure_failures - SURE_THING_FAILURE_RATE) < 0.03
    assert abs(lost_recoveries - LOST_CAUSE_RECOVERY_RATE) < 0.03


def test_recovered_records_carry_a_date_and_unrecovered_ones_do_not():
    _, _, _, outcomes = run(1000, seed=41)

    for outcome in outcomes.values():
        assert (outcome.recovered_date is not None) is outcome.recovered


def test_missing_treatment_assignment_is_refused():
    """Silently defaulting to control would drop an unassigned record into the
    comparison group and bias the estimate."""
    _, batch, assignments, _ = run(50, seed=41)
    del assignments[batch[0].customer_id]

    with pytest.raises(KeyError):
        simulate_outcomes(batch, assignments, seed=41)


# --- e. do_not_disturb churn ----------------------------------------------


def test_do_not_disturbs_churn_only_when_contacted():
    _, batch, assignments, outcomes = run(2000, seed=53)

    treated_churn = 0
    control_churn = 0
    treated_total = 0
    for record in batch:
        outcome = outcomes[record.customer_id]
        if segment_of(record) is Segment.DO_NOT_DISTURB:
            if assignments[record.customer_id]:
                treated_total += 1
                treated_churn += outcome.churned
            else:
                control_churn += outcome.churned
        else:
            assert not outcome.churned

    assert treated_churn > 0, "contact never cost a do_not_disturb anything"
    assert control_churn == 0, "an uncontacted customer churned"
    # Sensitivity is drawn from U(0.25, 0.75), so roughly half should go.
    assert 0.2 < treated_churn / treated_total < 0.8


def test_a_churned_do_not_disturb_can_still_have_paid():
    """The expensive case: the invoice is recovered and the subscription is
    gone. A recovery-rate metric records this as a win."""
    _, _, _, outcomes = run(2000, seed=53)

    both = [
        outcome for outcome in outcomes.values() if outcome.churned and outcome.recovered
    ]
    assert both, "churn and recovery never co-occurred; the cost case is missing"


# --- the answer key stays in the generator ---------------------------------


def _project_imports(path: pathlib.Path) -> set[str]:
    tree = ast.parse(path.read_text())
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module or "")
    return {name for name in imported if name.startswith("forbear")}


# The measurement harness generates the book it measures, so it holds the
# answer key by definition - it is a simulation driver, not a decision path.
# Naming it here keeps the exception explicit and countable: if a second entry
# ever appears, someone has to justify it in a diff.
SIMULATION_MODULES = frozenset(
    {
        "forbear/services/harness.py",
        # The demo stream endpoint generates the book it streams, which is why
        # it can run without a production database behind it. It is a
        # simulation surface that happens to be mounted on the API, and listing
        # it here rather than letting it pass silently is the point: a route
        # that fabricates customers should be visible in a diff.
        "forbear/api/stream.py",
    }
)

# The path a real decision travels: score, rank, plan, permit, execute. None of
# it may reach the generator, whatever a simulation surface is allowed to do.
DECISION_PATH = (
    "forbear/scoring",
    "forbear/core",
    "forbear/api/webhooks.py",
    "forbear/api/main.py",
    "forbear/services/allocator.py",
    "forbear/services/baseline.py",
    "forbear/services/unconstrained_baseline.py",
    "forbear/services/executor.py",
    "forbear/services/classifier.py",
    "forbear/services/ingestion.py",
)


def test_no_runtime_module_can_reach_the_ground_truth():
    """Structural enforcement, following test_guard's import check.

    ground_truth is the answer key. A scoring or allocation module that could
    import the generator could read the counterfactual it is supposed to be
    predicting, and every number the system reported afterwards would be
    circular. Keeping the dependency one-way is the only check that does not
    rely on someone remembering.

    The measurement harness is the one sanctioned exception: it generates the
    book it measures, the way a test does. It is named rather than skipped, so
    the exception cannot quietly grow.
    """
    offenders = {}
    for path in sorted((REPO_ROOT / "forbear").rglob("*.py")):
        if path.parent.name == "generator":
            continue
        if str(path.relative_to(REPO_ROOT)) in SIMULATION_MODULES:
            continue
        leaking = {
            name for name in _project_imports(path) if name.startswith("forbear.generator")
        }
        if leaking:
            offenders[str(path.relative_to(REPO_ROOT))] = sorted(leaking)

    assert offenders == {}, f"runtime code importing the answer key: {offenders}"


def test_the_decision_path_is_clean_even_of_sanctioned_exceptions():
    """The narrower rule, stated separately so the exception above cannot creep.

    Scoring, ranking, planning, permitting and executing are what a real
    decision passes through. A simulation driver may hold the answer key; none
    of these may, and no future entry in SIMULATION_MODULES can change that.
    """
    for target in DECISION_PATH:
        location = REPO_ROOT / target
        paths = (
            sorted(location.rglob("*.py")) if location.is_dir() else [location]
        )
        for path in paths:
            leaking = sorted(
                name
                for name in _project_imports(path)
                if name.startswith("forbear.generator")
            )
            assert leaking == [], (
                f"{path.relative_to(REPO_ROOT)} is on the decision path and "
                f"imports the answer key: {leaking}"
            )


def test_the_generator_depends_on_runtime_code_and_not_the_reverse():
    """The one-way direction is deliberate: the generator uses the real
    classifier table so its decline codes stay honest."""
    imports = _project_imports(REPO_ROOT / "forbear" / "generator" / "batch_generator.py")

    assert "forbear.services.classifier" in imports


def test_outcome_objects_carry_no_segment_or_counterfactual():
    """What the model sees must be what production could contain."""
    _, _, _, outcomes = run(10, seed=1)
    fields = set(vars(next(iter(outcomes.values()))))

    assert fields == {
        "customer_id",
        "treated",
        "recovered",
        "recovered_date",
        "churned",
    }


def test_profiles_are_immutable():
    """A profile mutated mid-run would make the batch that came from it
    unreproducible without changing the seed."""
    profile = generate_profiles(1, seed=1)[0]

    assert isinstance(profile, CustomerProfile)
    with pytest.raises(Exception):
        profile.salary_day = 4  # type: ignore[misc]
