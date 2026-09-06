"""Builders for Razorpay's documented webhook payload shapes.

Structure follows Razorpay's public webhook documentation: a top-level event
envelope (entity, account_id, event, contains, payload, created_at), with
nested payload.<resource>.entity.* blocks carrying that resource's real API
representation (Payment entity, Subscription entity, Invoice entity).

Two events (subscription.pending, subscription.halted) are built here with a
payment entity attached alongside the subscription entity, reporting the
status change in the context of the decline that produced it. This mirrors
how Forbear's own ingestion tests have always shaped these events (see
tests/test_ingestion.py's event() helper) and is what makes
handle_subscription_halted's amount lookup succeed without inventing a
non-Razorpay field.
"""

from __future__ import annotations

import time
from typing import Optional

ACCOUNT_ID = "acc_forbear_demo"

# error_code -> the error_description / error_source / error_step / error_reason
# quartet Razorpay attaches directly on payment.entity for a failed payment.
# Limited to the codes Forbear's classifier table (forbear/services/classifier.py)
# actually maps, since those are the only ones this demo needs to exercise.
_ERROR_DETAILS = {
    "INSUFFICIENT_FUNDS": dict(
        error_description="Payment failed due to insufficient funds in the customer's account.",
        error_source="bank",
        error_step="payment_authorization",
        error_reason="insufficient_funds",
    ),
    "BANK_ACCOUNT_DEBITED_ALREADY": dict(
        error_description="The customer's bank account has already been debited for this invoice.",
        error_source="bank",
        error_step="payment_authorization",
        error_reason="debited_already",
    ),
    "GATEWAY_ERROR": dict(
        error_description="Payment failed due to an error at the bank or gateway.",
        error_source="gateway",
        error_step="payment_authorization",
        error_reason="gateway_error",
    ),
    "MANDATE_EXPIRED": dict(
        error_description="The e-mandate registered for this subscription has expired.",
        error_source="customer",
        error_step="payment_authorization",
        error_reason="mandate_expired",
    ),
    "MANDATE_REVOKED": dict(
        error_description="The customer has revoked the e-mandate for this subscription.",
        error_source="customer",
        error_step="payment_authorization",
        error_reason="mandate_revoked",
    ),
    "BAD_REQUEST_ERROR": dict(
        error_description="The payment request was rejected by the bank.",
        error_source="business",
        error_step="payment_initiation",
        error_reason="input_validation_failed",
    ),
}


def _now(timestamp: Optional[int]) -> int:
    return timestamp if timestamp is not None else int(time.time())


def _subscription_entity(
    *,
    subscription_id: str,
    customer_id: str,
    status: str,
    plan_id: str,
    timestamp: int,
) -> dict:
    return {
        "id": subscription_id,
        "entity": "subscription",
        "plan_id": plan_id,
        "customer_id": customer_id,
        "status": status,
        "current_start": timestamp,
        "current_end": timestamp + 30 * 86400,
        "ended_at": None,
        "quantity": 1,
        "notes": [],
        "charge_at": timestamp,
        "start_at": timestamp,
        "end_at": timestamp + 365 * 86400,
        "auth_attempts": 0,
        "total_count": 12,
        "paid_count": 0,
        "customer_notify": True,
        "created_at": timestamp,
        "expire_by": timestamp + 86400,
        "short_url": f"https://rzp.io/i/{subscription_id}",
        "has_scheduled_changes": False,
        "change_scheduled_at": None,
        "source": "api",
        "payment_method": "card",
        "offer_id": None,
        "remaining_count": 12,
    }


def _payment_entity(
    *,
    payment_id: str,
    subscription_id: str,
    customer_id: str,
    invoice_id: str,
    amount: int,
    status: str,
    timestamp: int,
    error_code: Optional[str] = None,
) -> dict:
    entity = {
        "id": payment_id,
        "entity": "payment",
        "amount": amount,
        "currency": "INR",
        "status": status,
        "order_id": None,
        "invoice_id": invoice_id,
        "international": False,
        "method": "card",
        "amount_refunded": 0,
        "refund_status": None,
        "captured": status == "captured",
        "description": "Subscription charge",
        "card_id": "card_forbear_demo",
        "bank": None,
        "wallet": None,
        "vpa": None,
        "email": f"{customer_id}@example.com",
        "contact": "+919999999999",
        "notes": [],
        "fee": None,
        "tax": None,
        "error_code": None,
        "error_description": None,
        "error_source": None,
        "error_step": None,
        "error_reason": None,
        "created_at": timestamp,
        "subscription_id": subscription_id,
        "customer_id": customer_id,
    }
    if error_code is not None:
        details = _ERROR_DETAILS.get(error_code, {})
        entity.update(
            error_code=error_code,
            error_description=details.get(
                "error_description", "The payment could not be completed."
            ),
            error_source=details.get("error_source", "bank"),
            error_step=details.get("error_step", "payment_authorization"),
            error_reason=details.get("error_reason", error_code.lower()),
        )
    return entity


def _invoice_entity(
    *,
    invoice_id: str,
    subscription_id: str,
    customer_id: str,
    amount: int,
    status: str,
    timestamp: int,
) -> dict:
    return {
        "id": invoice_id,
        "entity": "invoice",
        "receipt": None,
        "invoice_number": None,
        "customer_id": customer_id,
        "subscription_id": subscription_id,
        "payment_id": None,
        "status": status,
        "expire_by": None,
        "issued_at": timestamp,
        "paid_at": timestamp if status == "paid" else None,
        "cancelled_at": None,
        "sms_status": "sent",
        "email_status": "sent",
        "date": timestamp,
        "amount": amount,
        "amount_paid": amount if status == "paid" else 0,
        "amount_due": 0 if status == "paid" else amount,
        "currency": "INR",
        "description": None,
        "short_url": f"https://rzp.io/i/{invoice_id}",
        "billing_start": timestamp,
        "billing_end": timestamp + 30 * 86400,
        "type": "invoice",
        "created_at": timestamp,
    }


def _envelope(*, event: str, contains: list, payload: dict, timestamp: int) -> dict:
    return {
        "entity": "event",
        "account_id": ACCOUNT_ID,
        "event": event,
        "contains": contains,
        "payload": payload,
        "created_at": timestamp,
    }


def build_subscription_pending(
    *,
    subscription_id: str,
    customer_id: str,
    invoice_id: str,
    amount: int,
    error_code: str,
    plan_id: str = "plan_forbear_demo",
    payment_id: Optional[str] = None,
    timestamp: Optional[int] = None,
) -> dict:
    """subscription.pending: Razorpay is mid-retry, has not given up yet."""
    ts = _now(timestamp)
    payment_id = payment_id or f"pay_{invoice_id}"
    return _envelope(
        event="subscription.pending",
        contains=["payment", "subscription"],
        timestamp=ts,
        payload={
            "payment": {
                "entity": _payment_entity(
                    payment_id=payment_id,
                    subscription_id=subscription_id,
                    customer_id=customer_id,
                    invoice_id=invoice_id,
                    amount=amount,
                    status="failed",
                    timestamp=ts,
                    error_code=error_code,
                )
            },
            "subscription": {
                "entity": _subscription_entity(
                    subscription_id=subscription_id,
                    customer_id=customer_id,
                    status="pending",
                    plan_id=plan_id,
                    timestamp=ts,
                )
            },
        },
    )


def build_subscription_halted(
    *,
    subscription_id: str,
    customer_id: str,
    invoice_id: str,
    amount: int,
    error_code: str = "INSUFFICIENT_FUNDS",
    plan_id: str = "plan_forbear_demo",
    payment_id: Optional[str] = None,
    timestamp: Optional[int] = None,
) -> dict:
    """subscription.halted: Razorpay has exhausted its own retry schedule.

    Reports the halt alongside the last failed attempt that caused it, which
    is also what supplies the amount the mandate was for.
    """
    ts = _now(timestamp)
    payment_id = payment_id or f"pay_{invoice_id}"
    return _envelope(
        event="subscription.halted",
        contains=["payment", "subscription"],
        timestamp=ts,
        payload={
            "payment": {
                "entity": _payment_entity(
                    payment_id=payment_id,
                    subscription_id=subscription_id,
                    customer_id=customer_id,
                    invoice_id=invoice_id,
                    amount=amount,
                    status="failed",
                    timestamp=ts,
                    error_code=error_code,
                )
            },
            "subscription": {
                "entity": _subscription_entity(
                    subscription_id=subscription_id,
                    customer_id=customer_id,
                    status="halted",
                    plan_id=plan_id,
                    timestamp=ts,
                )
            },
        },
    )


def build_subscription_charged(
    *,
    subscription_id: str,
    customer_id: str,
    invoice_id: str,
    amount: int,
    plan_id: str = "plan_forbear_demo",
    payment_id: Optional[str] = None,
    timestamp: Optional[int] = None,
) -> dict:
    """subscription.charged: a successful debit against the mandate."""
    ts = _now(timestamp)
    payment_id = payment_id or f"pay_{invoice_id}"
    return _envelope(
        event="subscription.charged",
        contains=["payment", "subscription", "invoice"],
        timestamp=ts,
        payload={
            "payment": {
                "entity": _payment_entity(
                    payment_id=payment_id,
                    subscription_id=subscription_id,
                    customer_id=customer_id,
                    invoice_id=invoice_id,
                    amount=amount,
                    status="captured",
                    timestamp=ts,
                )
            },
            "subscription": {
                "entity": _subscription_entity(
                    subscription_id=subscription_id,
                    customer_id=customer_id,
                    status="active",
                    plan_id=plan_id,
                    timestamp=ts,
                )
            },
            "invoice": {
                "entity": _invoice_entity(
                    invoice_id=invoice_id,
                    subscription_id=subscription_id,
                    customer_id=customer_id,
                    amount=amount,
                    status="paid",
                    timestamp=ts,
                )
            },
        },
    )


def build_payment_failed(
    *,
    subscription_id: str,
    customer_id: str,
    invoice_id: str,
    amount: int,
    error_code: str,
    payment_id: Optional[str] = None,
    timestamp: Optional[int] = None,
) -> dict:
    """payment.failed: a standalone decline notification."""
    ts = _now(timestamp)
    payment_id = payment_id or f"pay_{invoice_id}"
    return _envelope(
        event="payment.failed",
        contains=["payment"],
        timestamp=ts,
        payload={
            "payment": {
                "entity": _payment_entity(
                    payment_id=payment_id,
                    subscription_id=subscription_id,
                    customer_id=customer_id,
                    invoice_id=invoice_id,
                    amount=amount,
                    status="failed",
                    timestamp=ts,
                    error_code=error_code,
                )
            }
        },
    )


def build_payment_captured(
    *,
    subscription_id: str,
    customer_id: str,
    invoice_id: str,
    amount: int,
    payment_id: Optional[str] = None,
    timestamp: Optional[int] = None,
) -> dict:
    """payment.captured: money arrived, possibly out of band from a retry."""
    ts = _now(timestamp)
    payment_id = payment_id or f"pay_{invoice_id}"
    return _envelope(
        event="payment.captured",
        contains=["payment"],
        timestamp=ts,
        payload={
            "payment": {
                "entity": _payment_entity(
                    payment_id=payment_id,
                    subscription_id=subscription_id,
                    customer_id=customer_id,
                    invoice_id=invoice_id,
                    amount=amount,
                    status="captured",
                    timestamp=ts,
                )
            }
        },
    )
