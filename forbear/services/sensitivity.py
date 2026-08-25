"""Where the answer flips.

Forbear's whole argument rests on one number nobody actually knows: what it
costs when you chase a customer who did not want to be chased. Everything else
in the system is measurable - recovery rates, attempt counts, decline codes -
but churn caused by dunning is the term every merchant estimates and almost
none of them measure.

So rather than assert a value, sweep it. Hold the book, the seeds, the model
and the policies fixed, vary only the probability that a contacted
do_not_disturb cancels, and find the point where selective chasing overtakes
chasing everything. Below that point the aggressive policy is genuinely better
and Forbear is an expensive way to recover less money. Above it, every extra
attempt is buying invoices with subscriptions.

That crossover is the most defensible output of this project, because it does
not require anyone to accept our churn estimate. It hands the merchant the
question in the only form that can be answered from their own data: is your
dunning churn above or below this line?
"""

from __future__ import annotations

import csv
from dataclasses import replace
from pathlib import Path
from typing import Optional, Sequence

from forbear.services.harness import (
    STRATEGY_FORBEAR,
    STRATEGY_UNCONSTRAINED,
    ComparisonResult,
    HarnessConfig,
    format_comparison,
    run_comparison,
)

# Dense where the crossover tends to sit, sparse at the ends where the ordering
# is not in doubt.
DEFAULT_CHURN_RATES = (0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0)

CSV_COLUMNS = (
    "churn_rate",
    "strategy",
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
    "churn_rate_observed",
    "ltv_lost_to_churn",
    "net_value",
)

SweepPoint = tuple[float, ComparisonResult]


async def run_sweep(
    conn,
    seed: int,
    n_records: int,
    churn_rates: Optional[Sequence[float]] = None,
    config: Optional[HarnessConfig] = None,
    csv_path: Optional[str | Path] = None,
) -> list[SweepPoint]:
    """Run the full comparison once per churn rate.

    Every run uses the same seed, so the customers, their segments, their
    salary days and their luck are identical across the sweep. The only thing
    that moves is how a do_not_disturb reacts to being contacted - which is
    what makes the resulting curve a statement about that parameter rather than
    about sampling.
    """
    rates = tuple(churn_rates if churn_rates is not None else DEFAULT_CHURN_RATES)
    for rate in rates:
        if not 0.0 <= rate <= 1.0:
            raise ValueError(f"churn rate must be a probability, got {rate}")

    base = config or HarnessConfig()
    points: list[SweepPoint] = []

    for rate in rates:
        result = await run_comparison(
            conn,
            seed=seed,
            n_records=n_records,
            config=replace(base, contact_sensitivity=rate),
        )
        points.append((rate, result))

    if csv_path is not None:
        write_sweep_csv(points, csv_path)

    return points


def find_crossover(points: Sequence[SweepPoint]) -> Optional[float]:
    """The first churn rate at which Forbear is worth more than chasing all.

    Returns None when the sweep never crosses - which is a real answer, not a
    failure. If Forbear leads at every rate there was no crossover to find, and
    if it trails at every rate the honest report is that on this book, at these
    rates, the selective policy did not pay.

    The value returned is a sampled rate, not an interpolated root: it is the
    first rate tested where the ordering holds, so a coarse sweep reports a
    coarse answer rather than a precise-looking one it cannot support.
    """
    for rate, result in points:
        forbear = result.get(STRATEGY_FORBEAR)
        unconstrained = result.get(STRATEGY_UNCONSTRAINED)
        if forbear is None or unconstrained is None:
            continue
        if forbear.net_value > unconstrained.net_value:
            return rate
    return None


def write_sweep_csv(points: Sequence[SweepPoint], path: str | Path) -> Path:
    """One row per (churn rate, strategy).

    Long format, because the thing anyone plots from this is net value against
    churn rate, split by strategy.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(CSV_COLUMNS)
        for rate, result in points:
            for strategy, metrics in result.items():
                writer.writerow(
                    [
                        rate,
                        strategy,
                        metrics.records_processed,
                        metrics.amount_at_risk,
                        metrics.amount_recovered,
                        round(metrics.recovery_rate, 6),
                        metrics.attempts_consumed,
                        round(metrics.recovered_per_attempt, 6),
                        metrics.records_skipped,
                        metrics.records_blocked,
                        metrics.ltv_at_risk_from_skips,
                        metrics.churned_count,
                        round(metrics.churn_rate, 6),
                        metrics.ltv_lost_to_churn,
                        metrics.net_value,
                    ]
                )

    return destination


def format_sweep(points: Sequence[SweepPoint]) -> str:
    """Net value per strategy at each churn rate, and where the order changes.

    Printed in rupees, because paise in a summary table is a way of looking
    precise about a simulation.
    """
    if not points:
        return "sweep produced no points"

    strategies = list(points[0][1])
    label_width = 12
    column_width = 20

    header = "churn rate".ljust(label_width) + "".join(
        name.rjust(column_width) for name in strategies
    )
    lines = ["net value by churn rate (rupees)", header, "-" * len(header)]

    for rate, result in points:
        lines.append(
            f"{rate:.2f}".ljust(label_width)
            + "".join(
                f"{result[name].net_value / 100:,.0f}".rjust(column_width)
                for name in strategies
            )
        )

    crossover = find_crossover(points)
    lines.append("-" * len(header))
    if crossover is None:
        lines.append("crossover: none in the sampled range - the ordering never changed")
    else:
        lines.append(
            f"crossover: {crossover:.2f} - below this churn rate, chasing "
            f"everything wins; at or above it, selective chasing wins"
        )
    return "\n".join(lines)


def format_point(rate: float, result: ComparisonResult) -> str:
    """One churn rate's full comparison table, labelled with its rate."""
    return f"churn rate = {rate:.2f}\n{format_comparison(result)}"
