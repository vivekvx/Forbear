"""Uplift model tests.

This is the one file where the answer key and the predictions are allowed to
meet. The model is fitted on observable features and randomised treatment, with
no segment anywhere in its input; the segments come back out only to check
whether the estimated effects point the way the world actually works.

What is being tested is direction, not magnitude. A T-Learner on 2000 records
shrinks every estimate toward the population mean, so the persuadables score
well below their true +1.0 and the do_not_disturbs well above their true -0.5.
Asserting on exact values would be asserting on the amount of shrinkage, which
is a property of the sample size. Asserting on sign is asserting that the
causal structure was found.

The outcome the model trains on is recovered AND NOT churned. Recovery alone
cannot produce a negative effect for anyone - a do_not_disturb pays when chased
and the churn that follows is simply invisible - so a model trained on recovery
would rate the most expensive segment in the book as harmless.
"""

from __future__ import annotations

import ast
import pathlib
from datetime import date

import numpy as np
import pytest

from forbear.generator.batch_generator import generate_batch
from forbear.generator.customer_profiles import generate_profiles
from forbear.generator.outcome_simulator import simulate_outcomes
from forbear.generator.treatment_assignment import assign_treatment
from forbear.scoring.evaluation import plot_qini_curve, qini_score, segment_accuracy
from forbear.scoring.uplift import (
    FEATURE_NAMES,
    NEVER_CONTACTED,
    FeatureRow,
    NotFittedError,
    UpliftModel,
    build_feature_matrix,
)

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent

# Four billing runs rather than one. A single date would make day_of_month a
# constant column, and the feature would be silently untested.
BILLING_DATES = (
    date(2026, 5, 4),
    date(2026, 6, 12),
    date(2026, 7, 20),
    date(2026, 8, 24),
)

SEED = 1
RECORD_COUNT = 2000


class Dataset:
    """Everything one run produces, with the answer key kept to one side."""

    def __init__(self, X, treatment, outcome, segments, cate, recovery_probability):
        self.X = X
        self.treatment = treatment
        self.outcome = outcome
        self.segments = segments
        self.cate = cate
        self.recovery_probability = recovery_probability

    def mask(self, segment: str) -> np.ndarray:
        return self.segments == segment


def build_dataset(n: int = RECORD_COUNT, seed: int = SEED) -> Dataset:
    """Generate, treat, observe, fit, score.

    Attempt history is drawn as noise here: the generator models a single
    billing failure, not a book with months of dunning behind it. The columns
    are carried anyway because the production feature vector has them, and a
    model that fell over when a feature is uninformative would be a model that
    falls over in its first week.
    """
    profiles = generate_profiles(n, seed=seed)
    by_id = {profile.customer_id: profile for profile in profiles}
    history_rng = np.random.RandomState(seed)

    rows: list[FeatureRow] = []
    treatment: list[bool] = []
    outcome: list[int] = []
    segments: list[str] = []

    per_run = n // len(BILLING_DATES)
    for run_index, billing_date in enumerate(BILLING_DATES):
        slice_start = run_index * per_run
        run_profiles = profiles[slice_start : slice_start + per_run]

        batch = generate_batch(run_profiles, billing_date, seed=seed + run_index)
        assignments = assign_treatment(batch, 0.5, seed=seed + run_index)
        outcomes = simulate_outcomes(batch, assignments, seed=seed + run_index)

        for record in batch:
            profile = by_id[record.customer_id]
            contacted_before = history_rng.random_sample() < 0.4
            rows.append(
                FeatureRow(
                    plan_amount=record.amount,
                    subscription_age_months=profile.subscription_age_months,
                    failure_code=record.failure_code,
                    hour_of_failure=record.timestamp.hour,
                    day_of_month=record.timestamp.day,
                    attempts_so_far=int(history_rng.randint(0, 3)),
                    days_since_last_contact=(
                        int(history_rng.randint(1, 61)) if contacted_before else None
                    ),
                )
            )
            observed = outcomes[record.customer_id]
            treatment.append(assignments[record.customer_id])
            # The net-favourable event: the invoice came in and the customer is
            # still here. Recovery alone hides the entire churn cost.
            outcome.append(int(observed.recovered and not observed.churned))
            segments.append(record.ground_truth["segment"])

    X = build_feature_matrix(rows)
    treatment_array = np.array(treatment)
    outcome_array = np.array(outcome)

    model = UpliftModel(seed=seed).fit(X, treatment_array, outcome_array)

    return Dataset(
        X=X,
        treatment=treatment_array,
        outcome=outcome_array,
        segments=np.array(segments),
        cate=model.predict_cate(X),
        recovery_probability=model.predict_recovery_probability(X),
    )


@pytest.fixture(scope="module")
def dataset() -> Dataset:
    """Fitted once: two gradient-boosted models per run is the slow part."""
    return build_dataset()


# --- a-d. the estimated effects point the right way ------------------------


def test_the_dataset_is_the_size_and_shape_expected(dataset):
    assert len(dataset.cate) == RECORD_COUNT
    assert dataset.X.shape == (RECORD_COUNT, len(FEATURE_NAMES))
    assert set(np.unique(dataset.segments)) == {
        "sure_thing",
        "persuadable",
        "lost_cause",
        "do_not_disturb",
    }


def test_persuadables_have_positive_mean_cate(dataset):
    """The only segment where an attempt changes anything."""
    mean_cate = dataset.cate[dataset.mask("persuadable")].mean()

    print(f"\npersuadable mean CATE: {mean_cate:+.4f}")
    assert mean_cate > 0


def test_do_not_disturbs_have_negative_mean_cate(dataset):
    """The churn cost, visible as a sign.

    This is the assertion the whole design turns on. If it fails, either the
    outcome fed to the model was recovery rather than net-favourable, or the
    features carry nothing that separates a premium long-tenured customer from
    an ordinary one - and the allocator would spend real attempts destroying
    real subscriptions while its recovery rate went up.
    """
    mean_cate = dataset.cate[dataset.mask("do_not_disturb")].mean()

    print(f"do_not_disturb mean CATE: {mean_cate:+.4f}")
    assert mean_cate < 0


def test_sure_things_move_less_than_persuadables(dataset):
    """Near zero, and in any case nearer than the segment that responds."""
    sure_thing = np.abs(dataset.cate[dataset.mask("sure_thing")]).mean()
    persuadable = np.abs(dataset.cate[dataset.mask("persuadable")]).mean()

    print(f"mean |CATE|: sure_thing {sure_thing:.4f}, persuadable {persuadable:.4f}")
    assert sure_thing < persuadable


def test_the_segment_ordering_is_the_expected_one(dataset):
    """Persuadable above sure_thing above do_not_disturb: the ranking the
    allocator will spend its budget along."""
    means = {
        segment: dataset.cate[dataset.mask(segment)].mean()
        for segment in ("persuadable", "sure_thing", "lost_cause", "do_not_disturb")
    }

    assert means["persuadable"] > means["sure_thing"]
    assert means["sure_thing"] > means["do_not_disturb"]


# --- e. Qini ---------------------------------------------------------------


def test_held_out_qini_beats_random_ranking(dataset):
    """The honest number: ranking quality on records the model never saw.

    This replaced an in-sample assertion. That one passed comfortably and
    measured the wrong thing - gradient boosting memorises, so scoring on the
    fitted rows reports discrimination the model does not have. The in-sample
    figure is printed beside the held-out one to keep the gap in view rather
    than quietly dropping it.
    """
    evaluation = UpliftModel(seed=SEED).fit_and_evaluate(
        dataset.X, dataset.treatment, dataset.outcome
    )

    print(
        f"\nQini in-sample {evaluation.in_sample_qini:.4f}  "
        f"held-out {evaluation.held_out_qini:.4f}  "
        f"(in-sample overstates by {evaluation.overstatement:.1f}x, "
        f"train {evaluation.n_train}, test {evaluation.n_test})"
    )

    assert evaluation.held_out_qini > 0


def test_the_in_sample_score_overstates_the_held_out_one(dataset):
    """Records the defect that motivated the split, so it cannot come back
    unnoticed: if these two ever converge, the evaluation changed."""
    evaluation = UpliftModel(seed=SEED).fit_and_evaluate(
        dataset.X, dataset.treatment, dataset.outcome
    )

    assert evaluation.in_sample_qini > evaluation.held_out_qini


def test_a_shuffled_ranking_scores_worse_than_the_model(dataset):
    """Guards the Qini assertion itself: a metric that any ordering passes is
    not evidence the model learned anything."""
    model_score = qini_score(dataset.cate, dataset.treatment, dataset.outcome)

    rng = np.random.RandomState(0)
    shuffled = dataset.cate.copy()
    rng.shuffle(shuffled)
    random_score = qini_score(shuffled, dataset.treatment, dataset.outcome)

    print(f"Qini, shuffled predictions: {random_score:.4f}")
    assert model_score > random_score


def test_qini_curve_renders_to_a_png(dataset, tmp_path):
    destination = plot_qini_curve(
        dataset.cate, dataset.treatment, dataset.outcome, tmp_path / "qini.png"
    )

    assert destination.exists()
    assert destination.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


# --- f. diagnostic, not a gate ---------------------------------------------


def test_segment_sign_accuracy_is_reported(dataset):
    """Printed and inspected, never asserted against a threshold.

    A number that gates the build is a number someone will tune until it
    passes, and tuning against the segments means fitting the generator.
    """
    accuracy = segment_accuracy(dataset.cate, dataset.segments)

    print("\nsegment sign accuracy (tolerance 0.05):")
    for segment, share in accuracy.items():
        count = int(dataset.mask(segment).sum())
        print(f"  {segment:16s} n={count:5d}  {share:6.1%}")

    assert set(accuracy) == {
        "sure_thing",
        "persuadable",
        "lost_cause",
        "do_not_disturb",
    }
    assert all(0.0 <= share <= 1.0 for share in accuracy.values())


def test_segment_accuracy_refuses_a_length_mismatch():
    with pytest.raises(ValueError):
        segment_accuracy(np.array([0.1, 0.2]), ["persuadable"])


def test_segment_accuracy_refuses_an_unknown_segment():
    """An unrecognised segment silently scored as wrong would look like a model
    failure rather than a caller error."""
    with pytest.raises(ValueError):
        segment_accuracy(np.array([0.1]), ["definitely_not_a_segment"])


# --- absolute probability, alongside the effect ----------------------------


def test_recovery_probabilities_are_probabilities(dataset):
    assert np.all(dataset.recovery_probability >= 0.0)
    assert np.all(dataset.recovery_probability <= 1.0)


def test_recovery_probability_is_not_the_same_number_as_cate(dataset):
    """The distinction the allocator exists to act on. Sure things recover at a
    high rate and are worth nothing to contact; if these two arrays agreed, one
    of them would be redundant."""
    sure_thing = dataset.mask("sure_thing")
    lost_cause = dataset.mask("lost_cause")

    assert (
        dataset.recovery_probability[sure_thing].mean()
        > dataset.recovery_probability[lost_cause].mean()
    )
    correlation = np.corrcoef(dataset.recovery_probability, dataset.cate)[0, 1]
    assert abs(correlation) < 0.95


# --- the model refuses to guess -------------------------------------------


def test_scoring_before_fitting_raises():
    """No default-shaped answer: a zero array here would be indistinguishable
    from a model that found no effect anywhere."""
    model = UpliftModel(seed=SEED)

    with pytest.raises(NotFittedError):
        model.predict_cate(np.zeros((3, len(FEATURE_NAMES))))
    with pytest.raises(NotFittedError):
        model.predict_recovery_probability(np.zeros((3, len(FEATURE_NAMES))))


def test_an_empty_arm_is_refused():
    X = np.random.RandomState(0).random_sample((40, len(FEATURE_NAMES)))
    outcome = np.array([0, 1] * 20)

    with pytest.raises(ValueError, match="empty"):
        UpliftModel(seed=SEED).fit(X, np.ones(40, dtype=bool), outcome)


def test_an_arm_with_one_outcome_class_is_refused():
    """A constant model produces a CATE that is an artefact of the split."""
    X = np.random.RandomState(0).random_sample((40, len(FEATURE_NAMES)))
    treatment = np.array([True, False] * 20)
    outcome = np.where(treatment, 1, np.array([0, 1] * 20))

    with pytest.raises(ValueError, match="identical"):
        UpliftModel(seed=SEED).fit(X, treatment, outcome)


def test_mismatched_lengths_are_refused():
    X = np.zeros((10, len(FEATURE_NAMES)))

    with pytest.raises(ValueError, match="line up"):
        UpliftModel(seed=SEED).fit(X, np.ones(9, dtype=bool), np.ones(10, dtype=int))


def test_a_wrong_width_feature_matrix_is_refused():
    """Silently accepting the wrong columns would mis-score every record while
    the model looked perfectly healthy."""
    X = np.zeros((10, len(FEATURE_NAMES) - 1))
    treatment = np.array([True, False] * 5)

    with pytest.raises(ValueError, match="features"):
        UpliftModel(seed=SEED).fit(X, treatment, np.array([0, 1] * 5))


# --- features --------------------------------------------------------------


def test_feature_matrix_encodes_a_never_contacted_customer_out_of_range():
    """Zero would read as "contacted today", which is the opposite of true."""
    row = FeatureRow(
        plan_amount=29900,
        subscription_age_months=7,
        failure_code="INSUFFICIENT_FUNDS",
        hour_of_failure=14,
        day_of_month=24,
        attempts_so_far=1,
        days_since_last_contact=None,
    )

    matrix = build_feature_matrix([row])
    assert matrix[0][FEATURE_NAMES.index("days_since_last_contact")] == NEVER_CONTACTED


def test_an_unknown_failure_code_is_refused():
    """It belongs on the exception list, not silently encoded as a number."""
    row = FeatureRow(
        plan_amount=29900,
        subscription_age_months=7,
        failure_code="NOT_A_RAZORPAY_CODE",
        hour_of_failure=14,
        day_of_month=24,
        attempts_so_far=1,
        days_since_last_contact=3,
    )

    with pytest.raises(ValueError):
        build_feature_matrix([row])


def test_no_segment_label_reaches_the_feature_vector():
    """Seven columns, all observable. The moment an eighth appears, someone
    should have to explain where it comes from in production."""
    assert FEATURE_NAMES == (
        "plan_amount",
        "subscription_age_months",
        "failure_code_encoded",
        "hour_of_failure",
        "day_of_month",
        "attempts_so_far",
        "days_since_last_contact",
    )


# --- the boundary ----------------------------------------------------------


def test_scoring_never_imports_the_generator():
    """Same structural check as test_guard's, for the same reason: at scoring
    time the ground truth does not exist, and in production it never did.

    An import here would let a model read the counterfactual it is meant to be
    estimating, and every number the system reported afterwards would be a
    restatement of the generator's own assumptions.
    """
    offenders = {}
    for path in sorted((REPO_ROOT / "forbear" / "scoring").rglob("*.py")):
        tree = ast.parse(path.read_text())
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported.add(node.module or "")

        leaking = sorted(
            name for name in imported if name.startswith("forbear.generator")
        )
        if leaking:
            offenders[str(path.relative_to(REPO_ROOT))] = leaking

    assert offenders == {}, f"scoring code importing the answer key: {offenders}"
