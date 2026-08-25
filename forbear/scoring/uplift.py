"""Incremental effect of contacting a customer. Scores only, decides nothing.

The quantity here is not "will this invoice recover". It is "does contacting
this customer change whether it recovers", which is a different number and
frequently has a different sign. A customer who pays on payday either way has a
high recovery probability and zero uplift; spending an attempt on them buys
nothing. That distinction is the entire reason this module exists rather than a
churn-style propensity model.

A T-Learner: fit one model on the treated arm, one on the control arm, and take
the difference of their predicted probabilities. Two independent models rather
than one model with treatment as a feature, because a single learner is free to
ignore the treatment column when the other features are more predictive, and it
usually does - the effect of contact is small next to the effect of being
broke. The T-Learner cannot make that mistake: the arms never share a tree.

Both learners are gradient boosting at stock settings. The point is not to win
a leaderboard. The point is that someone can be walked through why a record
scored the way it did, and that neither the model nor its hyperparameters are
where the interesting risk lives.

WHAT `outcome` MUST MEAN
------------------------
The outcome passed to fit() has to be the net-favourable event - the invoice
recovered AND the customer stayed - not recovery on its own. Trained on
recovery alone, contact can only ever help or do nothing: a do_not_disturb
customer pays when chased, so their measured uplift is zero and the churn they
took on the way out is invisible to every number downstream. Fold the churn in
and their CATE goes properly negative, which is what lets the allocator skip
them.

Invariant 1: nothing here reaches the payment path. This module returns floats.
And nothing here may import from the generator - at scoring time the ground
truth does not exist, and in production it never existed at all.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier

from forbear.services.classifier import known_codes

FEATURE_NAMES = (
    "plan_amount",
    "subscription_age_months",
    "failure_code_encoded",
    "hour_of_failure",
    "day_of_month",
    "attempts_so_far",
    "days_since_last_contact",
)

# Ordinal encoding over the classifier's table, sorted so the mapping is stable
# across runs and processes. Trees split on thresholds rather than distances, so
# an arbitrary ordering costs nothing and one-hot would only add columns. This
# is tied to the classifier's MAPPING_VERSION: if the table gains a code the
# encoding shifts, and a model fitted under the old one must be refit, not
# reused.
FAILURE_CODE_ENCODING: dict[str, int] = {
    code: index for index, code in enumerate(sorted(known_codes()))
}

# A customer nobody has contacted has no "days since last contact". Zero would
# be a lie in the wrong direction - it reads as contacted today - so the gap is
# encoded further out than any real history reaches.
NEVER_CONTACTED = 9_999


@dataclass(frozen=True)
class FeatureRow:
    """The observable half of a record. No segment, no counterfactual.

    Everything here is something a production database holds on the morning the
    allocator runs. If a field could not be filled in from Razorpay plus
    Forbear's own attempt history, it does not belong in this dataclass.
    """

    plan_amount: int  # paise
    subscription_age_months: int
    failure_code: str
    hour_of_failure: int
    day_of_month: int
    attempts_so_far: int
    days_since_last_contact: Optional[int]  # None: never contacted


def build_feature_matrix(rows: Sequence[FeatureRow]) -> np.ndarray:
    """FeatureRows -> the float matrix both learners see.

    One place does the encoding, so training and scoring cannot drift apart in
    column order. A silent version of that bug would mis-score every record
    while looking entirely healthy.
    """
    matrix = np.empty((len(rows), len(FEATURE_NAMES)), dtype=float)
    for index, row in enumerate(rows):
        encoded = FAILURE_CODE_ENCODING.get(row.failure_code)
        if encoded is None:
            raise ValueError(
                f"failure_code {row.failure_code!r} is not in the classifier's "
                f"table; it belongs on the exception list, not in a feature vector"
            )
        matrix[index] = (
            row.plan_amount,
            row.subscription_age_months,
            encoded,
            row.hour_of_failure,
            row.day_of_month,
            row.attempts_so_far,
            NEVER_CONTACTED
            if row.days_since_last_contact is None
            else row.days_since_last_contact,
        )
    return matrix


class NotFittedError(Exception):
    """Scoring was attempted before fit(). Never a default-shaped answer."""


@dataclass(frozen=True)
class UpliftEvaluation:
    """Both Qini scores, because only one of them means anything.

    held_out_qini is the number to publish. in_sample_qini is carried beside it
    so the gap stays visible: quoting the in-sample figure alone is how a model
    ends up described as four times better than it is.
    """

    in_sample_qini: float
    held_out_qini: float
    n_train: int
    n_test: int

    @property
    def overstatement(self) -> float:
        """How many times larger the in-sample score is. 1.0 means no gap."""
        if self.held_out_qini == 0:
            return float("inf")
        return self.in_sample_qini / self.held_out_qini


class UpliftModel:
    """T-Learner over two gradient-boosted classifiers.

    seed is threaded into both learners so a run can be reproduced exactly.
    n_estimators and max_depth are constructor arguments because they have to
    live somewhere, not because they are meant to be searched: the defaults are
    the documented settings, and moving them to chase a Qini score would be
    fitting the evaluation rather than the problem.
    """

    def __init__(
        self, seed: int = 0, n_estimators: int = 100, max_depth: int = 4
    ) -> None:
        self.seed = seed
        self._treated_model = GradientBoostingClassifier(
            n_estimators=n_estimators, max_depth=max_depth, random_state=seed
        )
        self._control_model = GradientBoostingClassifier(
            n_estimators=n_estimators, max_depth=max_depth, random_state=seed
        )
        self._fitted = False

    def fit(
        self,
        X: np.ndarray,
        treatment: np.ndarray,
        outcome: np.ndarray,
    ) -> "UpliftModel":
        """Fit one learner per arm.

        The checks below all guard the same failure: an arm that cannot support
        an estimate. A missing arm, or an arm where every outcome is identical,
        yields a model that returns a constant, and the CATE built from it is
        an artefact of the split rather than an effect. Worth an exception
        rather than a number nobody can tell is wrong.
        """
        X = np.asarray(X, dtype=float)
        treatment = np.asarray(treatment).astype(bool)
        outcome = np.asarray(outcome).astype(int)

        if not len(X) == len(treatment) == len(outcome):
            raise ValueError(
                f"X, treatment and outcome must line up: got {len(X)}, "
                f"{len(treatment)}, {len(outcome)}"
            )
        if X.shape[1] != len(FEATURE_NAMES):
            raise ValueError(
                f"expected {len(FEATURE_NAMES)} features {FEATURE_NAMES}, "
                f"got {X.shape[1]}"
            )

        for arm_name, mask in (("treated", treatment), ("control", ~treatment)):
            if not mask.any():
                raise ValueError(
                    f"the {arm_name} arm is empty; there is no counterfactual to "
                    f"estimate against"
                )
            if len(np.unique(outcome[mask])) < 2:
                raise ValueError(
                    f"every outcome in the {arm_name} arm is identical; the "
                    f"resulting model would be a constant, not an estimate"
                )

        self._treated_model.fit(X[treatment], outcome[treatment])
        self._control_model.fit(X[~treatment], outcome[~treatment])
        self._fitted = True
        return self

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise NotFittedError("fit() must be called before scoring")

    def predict_cate(self, X: np.ndarray) -> np.ndarray:
        """Estimated change in the favourable outcome caused by contact.

        Positive: contact helps, and the record is worth an attempt. Negative:
        contact costs more than it recovers, and the allocator must skip it.
        Near zero: the customer's decision was never ours to influence.
        """
        self._check_fitted()
        X = np.asarray(X, dtype=float)
        treated = self._treated_model.predict_proba(X)[:, 1]
        control = self._control_model.predict_proba(X)[:, 1]
        return treated - control

    def fit_and_evaluate(
        self,
        X: np.ndarray,
        treatment: np.ndarray,
        outcome: np.ndarray,
        test_size: float = 0.3,
    ) -> "UpliftEvaluation":
        """Fit, and report how well the ranking survives unseen records.

        Scoring a model on the rows it was fitted on measures memorisation, not
        discrimination, and gradient boosting is good at memorisation. On this
        generator the in-sample Qini runs about four times the held-out figure,
        so the gap is not a rounding detail - it is most of the number.

        The split is stratified on the treatment/outcome pair rather than on
        treatment alone. Treatment alone can hand the training half an arm
        whose outcomes are all identical, which fit() refuses outright; the
        pair keeps both arms and both classes proportional in both halves.

        The model is left fitted on ALL the data, not just the training half.
        The held-out score estimates how a model built this way generalises;
        the model worth scoring with is the one that has seen everything.
        Reporting the first while shipping the second is the standard
        arrangement, and saying so here is cheaper than the argument later.
        """
        from sklearn.model_selection import train_test_split

        # Local import: forbear.scoring.evaluation pulls in matplotlib, and the
        # allocator imports this module on the decision path.
        from forbear.scoring.evaluation import qini_score

        X = np.asarray(X, dtype=float)
        treatment = np.asarray(treatment).astype(bool)
        outcome = np.asarray(outcome).astype(int)

        strata = treatment.astype(int) * 2 + outcome
        indices = np.arange(len(X))
        train_index, test_index = train_test_split(
            indices,
            test_size=test_size,
            random_state=self.seed,
            stratify=strata,
        )

        self.fit(X[train_index], treatment[train_index], outcome[train_index])

        in_sample = qini_score(
            self.predict_cate(X[train_index]),
            treatment[train_index],
            outcome[train_index],
        )
        held_out = qini_score(
            self.predict_cate(X[test_index]),
            treatment[test_index],
            outcome[test_index],
        )

        self.fit(X, treatment, outcome)

        return UpliftEvaluation(
            in_sample_qini=in_sample,
            held_out_qini=held_out,
            n_train=len(train_index),
            n_test=len(test_index),
        )

    def predict_recovery_probability(self, X: np.ndarray) -> np.ndarray:
        """Absolute probability of the favourable outcome under contact.

        The allocator needs this alongside the CATE: uplift ranks who benefits
        most from an attempt, this says how much is actually likely to land. A
        large uplift on a record that recovers 4% of the time either way is
        still a thin thing to spend a capped attempt on.
        """
        self._check_fitted()
        X = np.asarray(X, dtype=float)
        return self._treated_model.predict_proba(X)[:, 1]
