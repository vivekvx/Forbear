"""Plain data objects mirroring the tables in schema.sql.

No ORM. These carry rows between explicit SQL and the rest of the system;
they hold no persistence behaviour of their own.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Optional


class MandateStatus(str, Enum):
    ACTIVE = "active"
    PAUSED = "paused"
    REVOKED = "revoked"
    EXPIRED = "expired"


class FailureClass(str, Enum):
    TIME_DEPENDENT = "time_dependent"
    TRANSIENT = "transient"
    REAUTH_REQUIRED = "reauth_required"
    TERMINAL = "terminal"


class RecordStatus(str, Enum):
    OPEN = "open"
    SCHEDULED = "scheduled"
    IN_FLIGHT = "in_flight"
    RECOVERED = "recovered"
    ABANDONED = "abandoned"
    SKIPPED = "skipped"


class AttemptOutcome(str, Enum):
    PENDING = "pending"
    SUCCESS = "success"
    FAILURE = "failure"
    BLOCKED_BY_GUARD = "blocked_by_guard"


class ContactChannel(str, Enum):
    PAYMENT_LINK = "payment_link"
    SMS = "sms"
    EMAIL = "email"


@dataclass
class Customer:
    id: int
    external_id: str  # Razorpay customer_id
    created_at: datetime


@dataclass
class Subscription:
    id: int
    customer_id: int
    external_id: str  # Razorpay subscription_id
    plan_amount: int  # paise
    billing_cycle_days: int
    mandate_status: MandateStatus
    created_at: datetime


@dataclass
class AtRiskRecord:
    id: int
    subscription_id: int
    customer_id: int
    invoice_id: str  # Razorpay invoice_id
    amount: int  # paise
    failure_code: str
    failure_class: FailureClass
    status: RecordStatus
    created_at: datetime
    updated_at: datetime
    uplift_score: Optional[float] = None
    whittle_index: Optional[float] = None
    skip_reason: Optional[str] = None


@dataclass
class Attempt:
    id: int
    at_risk_record_id: int
    attempt_number: int
    scheduled_at: datetime
    outcome: AttemptOutcome
    created_at: datetime
    executed_at: Optional[datetime] = None
    guard_verdict: Optional[dict[str, Any]] = None


@dataclass
class Contact:
    id: int
    customer_id: int
    channel: ContactChannel
    sent_at: datetime
    created_at: datetime


@dataclass(frozen=True)
class AuditEntry:
    id: int
    entity_type: str
    entity_id: str
    action: str
    details: dict[str, Any]
    prev_hash: str
    hash: str
    created_at: datetime


@dataclass
class WebhookEvent:
    id: int
    event_id: str
    event_type: str
    payload: dict[str, Any]
    received_at: datetime


def at_risk_record_from_row(row) -> AtRiskRecord:
    """Build an AtRiskRecord from an asyncpg Record."""
    return AtRiskRecord(
        id=row["id"],
        subscription_id=row["subscription_id"],
        customer_id=row["customer_id"],
        invoice_id=row["invoice_id"],
        amount=row["amount"],
        failure_code=row["failure_code"],
        failure_class=FailureClass(row["failure_class"]),
        status=RecordStatus(row["status"]),
        uplift_score=row["uplift_score"],
        whittle_index=row["whittle_index"],
        skip_reason=row["skip_reason"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )
