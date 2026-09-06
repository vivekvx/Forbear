"""Fit the uplift model once, on synthetic training data, and persist it.

This is the only place the model gets fit. Production loads the artifact this
script writes; it never fits per webhook, per worklist read, or per demo run.
Re-run this script (and redeploy the artifact) to retrain - that is a
deliberate, versioned act, not something ingestion does on your behalf.
"""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import numpy as np

from forbear.generator.batch_generator import generate_batch
from forbear.generator.customer_profiles import generate_profiles
from forbear.generator.outcome_simulator import simulate_outcomes
from forbear.generator.treatment_assignment import assign_treatment
from forbear.scoring.uplift import UpliftModel, build_feature_matrix
from forbear.services.harness import _feature_rows

DEFAULT_MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "uplift_model.pkl"


def train(n_records: int, seed: int) -> tuple[UpliftModel, float, float]:
    profiles = generate_profiles(n_records, seed=seed)
    batch = generate_batch(profiles, billing_date=date.today(), seed=seed)
    assignments = assign_treatment(batch, seed=seed)
    observed = simulate_outcomes(batch, assignments, seed=seed)

    feature_rng = np.random.RandomState(seed)
    X = build_feature_matrix(_feature_rows(profiles, batch, feature_rng))
    treatment = np.array([assignments[record.customer_id] for record in batch])
    outcome = np.array(
        [
            int(
                observed[record.customer_id].recovered
                and not observed[record.customer_id].churned
            )
            for record in batch
        ]
    )

    model = UpliftModel(seed=seed)
    evaluation = model.fit_and_evaluate(X, treatment, outcome)
    return model, evaluation.held_out_qini, evaluation.in_sample_qini


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-records", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=DEFAULT_MODEL_PATH)
    args = parser.parse_args()

    model, held_out_qini, in_sample_qini = train(args.n_records, args.seed)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(args.out)

    print(f"held-out Qini:  {held_out_qini:.4f}")
    print(f"in-sample Qini: {in_sample_qini:.4f}")
    print(f"model written to {args.out}")


if __name__ == "__main__":
    main()
