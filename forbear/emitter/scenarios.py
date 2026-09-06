"""Pre-built lifecycles exercising Forbear's classification and state paths.

Each scenario is self-contained: given a suffix, it fabricates its own
subscription/customer/invoice ids so scenarios never collide with each other
in the same database. Callers that need a fresh scenario just omit the
suffix.
"""

from __future__ import annotations

import uuid

from forbear.emitter.emitter import Scenario, ScenarioStep
from forbear.emitter.payload_templates import (
    build_payment_captured,
    build_payment_failed,
    build_subscription_charged,
    build_subscription_halted,
    build_subscription_pending,
)

# ₹499, matching the plan amount used throughout the rest of the codebase's
# fixtures (see tests/conftest.py insert_scenario).
DEFAULT_AMOUNT = 49900


def _suffix(given) -> str:
    return given or uuid.uuid4().hex[:10]


def _ids(suffix: str) -> dict:
    return {
        "subscription_id": f"sub_{suffix}",
        "customer_id": f"cust_{suffix}",
        "invoice_id": f"inv_{suffix}",
    }


def insufficient_funds_then_recovers(suffix: str = None) -> Scenario:
    """A retry-eligible decline followed by a successful charge."""
    suffix = _suffix(suffix)
    ids = _ids(suffix)
    pending = build_subscription_pending(
        error_code="INSUFFICIENT_FUNDS", amount=DEFAULT_AMOUNT, **ids
    )
    charged = build_subscription_charged(amount=DEFAULT_AMOUNT, **ids)
    return Scenario(
        name=f"insufficient_funds_then_recovers:{suffix}",
        steps=[
            ScenarioStep(pending, event_id=f"evt_{suffix}_pending"),
            ScenarioStep(charged, event_id=f"evt_{suffix}_charged"),
        ],
    )


def insufficient_funds_then_halts(suffix: str = None) -> Scenario:
    """Retries exhausted: Razorpay hands the subscription over to Forbear."""
    suffix = _suffix(suffix)
    ids = _ids(suffix)
    pending = build_subscription_pending(
        error_code="INSUFFICIENT_FUNDS", amount=DEFAULT_AMOUNT, **ids
    )
    halted = build_subscription_halted(
        error_code="INSUFFICIENT_FUNDS", amount=DEFAULT_AMOUNT, **ids
    )
    return Scenario(
        name=f"insufficient_funds_then_halts:{suffix}",
        steps=[
            ScenarioStep(pending, event_id=f"evt_{suffix}_pending"),
            ScenarioStep(halted, event_id=f"evt_{suffix}_halted"),
        ],
    )


def revoked_mandate(suffix: str = None) -> Scenario:
    """No usable authorisation left: the classifier must route this terminal."""
    suffix = _suffix(suffix)
    ids = _ids(suffix)
    pending = build_subscription_pending(
        error_code="MANDATE_REVOKED", amount=DEFAULT_AMOUNT, **ids
    )
    return Scenario(
        name=f"revoked_mandate:{suffix}",
        steps=[ScenarioStep(pending, event_id=f"evt_{suffix}_revoked")],
    )


def duplicate_delivery(suffix: str = None) -> Scenario:
    """The identical event, redelivered under the identical event id."""
    suffix = _suffix(suffix)
    ids = _ids(suffix)
    pending = build_subscription_pending(
        error_code="INSUFFICIENT_FUNDS", amount=DEFAULT_AMOUNT, **ids
    )
    event_id = f"evt_{suffix}_dup"
    return Scenario(
        name=f"duplicate_delivery:{suffix}",
        steps=[
            ScenarioStep(pending, event_id=event_id),
            ScenarioStep(pending, event_id=event_id),
        ],
    )


def tampered_signature(suffix: str = None) -> Scenario:
    """A correct payload sent with a signature that does not match it."""
    suffix = _suffix(suffix)
    ids = _ids(suffix)
    pending = build_subscription_pending(
        error_code="INSUFFICIENT_FUNDS", amount=DEFAULT_AMOUNT, **ids
    )
    return Scenario(
        name=f"tampered_signature:{suffix}",
        steps=[
            ScenarioStep(
                pending,
                event_id=f"evt_{suffix}_tampered",
                signature_override="0" * 64,
            )
        ],
    )


def payment_failed_standalone(suffix: str = None) -> Scenario:
    """A payment.failed with no preceding subscription.pending."""
    suffix = _suffix(suffix)
    ids = _ids(suffix)
    failed = build_payment_failed(
        error_code="GATEWAY_ERROR", amount=DEFAULT_AMOUNT, **ids
    )
    return Scenario(
        name=f"payment_failed_standalone:{suffix}",
        steps=[ScenarioStep(failed, event_id=f"evt_{suffix}_failed")],
    )


def payment_captured_out_of_band(suffix: str = None) -> Scenario:
    """A pending record recovered by a payment.captured rather than a charge."""
    suffix = _suffix(suffix)
    ids = _ids(suffix)
    pending = build_subscription_pending(
        error_code="INSUFFICIENT_FUNDS", amount=DEFAULT_AMOUNT, **ids
    )
    captured = build_payment_captured(amount=DEFAULT_AMOUNT, **ids)
    return Scenario(
        name=f"payment_captured_out_of_band:{suffix}",
        steps=[
            ScenarioStep(pending, event_id=f"evt_{suffix}_pending"),
            ScenarioStep(captured, event_id=f"evt_{suffix}_captured"),
        ],
    )


_MIXED_BUILDERS = (
    insufficient_funds_then_recovers,
    insufficient_funds_then_halts,
    revoked_mandate,
)


def batch_scenario(n: int) -> Scenario:
    """n mixed lifecycles, round-robined across the recoverable/terminal paths.

    Enough to populate a worklist with a mix of open, recovered, and
    terminal-skipped records in one call.
    """
    steps = []
    for i in range(n):
        builder = _MIXED_BUILDERS[i % len(_MIXED_BUILDERS)]
        steps.extend(builder(suffix=f"batch{i}").steps)
    return Scenario(name=f"batch_scenario:{n}", steps=steps)
