"""Does the uplift model rank better than chance? Measurement, not scoring.

Uplift models cannot be scored the ordinary way. The label a classifier would
need - what this customer would have done under the other arm - does not exist
for any record, and never will: each customer was either contacted or not.
Accuracy, AUC and log loss all quietly answer a different question, "can you
predict recovery", which a model that ignores treatment entirely will win.

The Qini curve sidesteps that by scoring the ranking instead of the record.
Sort by predicted uplift, walk down the list, and track the difference in
outcomes between the treated and control records seen so far, scaled for the
imbalance between arms. A model that puts the persuadables at the top and the
do_not_disturbs at the bottom pulls that curve above the diagonal. A model
whose ordering is noise traces the diagonal itself, and the coefficient is
zero. Negative means the ranking is actively worse than random - which happens,
and is worth knowing.

The number is a ranking quality, not a business result: it says the ordering is
useful, not that contacting the top decile pays for itself. That question
belongs to the Whittle index and the attempt budget.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Sequence

import matplotlib

# Headless by default: this renders into the architecture document from a test
# run or a CI job, never into a window. Set before pyplot is imported, which is
# when the backend is bound.
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklift.metrics import qini_auc_score, qini_curve  # noqa: E402

# How far from zero a CATE has to be before it counts as a direction rather
# than noise. Used only by segment_accuracy, which is a diagnostic: the
# allocator's threshold is the Whittle index, not this.
DEFAULT_SIGN_TOLERANCE = 0.05

# What each segment's CATE should look like if the model has found the causal
# structure. Not a target to optimise toward - a rubric for reading the result.
EXPECTED_DIRECTION: dict[str, str] = {
    "persuadable": "positive",
    "sure_thing": "near_zero",
    "lost_cause": "near_zero",
    "do_not_disturb": "negative",
}


def qini_score(
    predictions: np.ndarray,
    treatment: np.ndarray,
    outcome: np.ndarray,
) -> float:
    """Qini coefficient. Above zero beats random ranking.

    sklift takes (y_true, uplift, treatment) in that order; the argument order
    here follows the rest of this codebase instead, so the mapping happens in
    one place rather than at every call site.
    """
    return float(
        qini_auc_score(
            y_true=np.asarray(outcome).astype(int),
            uplift=np.asarray(predictions, dtype=float),
            treatment=np.asarray(treatment).astype(int),
        )
    )


def plot_qini_curve(
    predictions: np.ndarray,
    treatment: np.ndarray,
    outcome: np.ndarray,
    path: str | Path,
) -> Path:
    """Save the Qini curve as a PNG and return where it landed.

    The diagonal is drawn alongside it deliberately. The coefficient alone says
    a model beat random; the curve says where it did - a model that is
    excellent in the top decile and useless below it looks identical to a
    mediocre uniform one by area, and they call for entirely different budgets.
    """
    y_true = np.asarray(outcome).astype(int)
    uplift = np.asarray(predictions, dtype=float)
    treatment_flags = np.asarray(treatment).astype(int)

    x_values, y_values = qini_curve(y_true, uplift, treatment_flags)
    coefficient = qini_score(uplift, treatment_flags, y_true)

    figure, axes = plt.subplots(figsize=(7, 5))
    axes.plot(x_values, y_values, label=f"model (Qini = {coefficient:.4f})")
    axes.plot(
        [0, x_values[-1]],
        [0, y_values[-1]],
        linestyle="--",
        color="grey",
        label="random targeting",
    )
    axes.set_xlabel("records targeted, ranked by predicted uplift")
    axes.set_ylabel("incremental favourable outcomes")
    axes.set_title("Qini curve")
    axes.legend()
    figure.tight_layout()

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=150)
    plt.close(figure)
    return destination


def segment_accuracy(
    predictions: np.ndarray,
    ground_truth_segments: Sequence[str],
    tolerance: float = DEFAULT_SIGN_TOLERANCE,
) -> dict[str, float]:
    """Share of each segment whose CATE points the way it should. TEST ONLY.

    This is the one function in scoring/ that needs the answer key, which is
    why it takes the segments as an argument and never goes looking for them.
    Nothing in production can call it usefully: in production there are no
    segments to pass, only outcomes.

    Read it as a diagnostic, not a gate. A low score on do_not_disturb means
    the churn cost is not being detected and the allocator will happily spend
    attempts destroying value - worth investigating. But tuning until this
    number goes up is fitting the generator, and the generator is not the thing
    being predicted.
    """
    predictions = np.asarray(predictions, dtype=float)
    if len(predictions) != len(ground_truth_segments):
        raise ValueError(
            f"got {len(predictions)} predictions for "
            f"{len(ground_truth_segments)} segments"
        )

    correct: dict[str, int] = defaultdict(int)
    total: dict[str, int] = defaultdict(int)

    for prediction, segment in zip(predictions, ground_truth_segments):
        expected = EXPECTED_DIRECTION.get(segment)
        if expected is None:
            raise ValueError(f"no expected direction defined for segment {segment!r}")

        total[segment] += 1
        if expected == "positive":
            correct[segment] += prediction > tolerance
        elif expected == "negative":
            correct[segment] += prediction < -tolerance
        else:
            correct[segment] += abs(prediction) <= tolerance

    return {
        segment: correct[segment] / count for segment, count in sorted(total.items())
    }
