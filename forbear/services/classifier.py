"""Razorpay decline code -> recovery class.

This is a lookup table, not a model. Nothing here scores or predicts, so
nothing here can be wrong in an interesting way: it can only be incomplete.

Incompleteness is handled by raising, never by defaulting. A default class
would be a silent guess, and a wrong guess here makes the allocator spend
attempts on records that cannot recover, or skip records that could. An
unknown code belongs on a human's exception list.

The mapping is versioned. Callers record MAPPING_VERSION in the audit entry
alongside the decision, so a later reader can tell which table was in force
when a record was classified.
"""

from __future__ import annotations

from typing import Optional

from forbear.models.models import FailureClass

# Bump on every change to _FAILURE_CLASSES. Never edit the table in place
# without bumping: the version is what makes an old audit entry legible.
MAPPING_VERSION = "2026-08-24.1"

# Keyed on (error_code, error_reason). A None reason is the catch-all for that
# code; a specific reason takes precedence over it, which is how a single code
# gets split into two classes when Razorpay overloads it.
ANY_REASON: Optional[str] = None

_FAILURE_CLASSES: dict[tuple[str, Optional[str]], FailureClass] = {
    # Money was not there, or was already taken. Time may fix both.
    ("INSUFFICIENT_FUNDS", ANY_REASON): FailureClass.TIME_DEPENDENT,
    ("BANK_ACCOUNT_DEBITED_ALREADY", ANY_REASON): FailureClass.TIME_DEPENDENT,
    # The rails failed, not the customer. Retry cheaply.
    ("GATEWAY_ERROR", ANY_REASON): FailureClass.TRANSIENT,
    ("TIMEOUT", ANY_REASON): FailureClass.TRANSIENT,
    ("BAD_GATEWAY", ANY_REASON): FailureClass.TRANSIENT,
    # No usable authorisation. Retrying the debit cannot work; only the
    # customer re-authorising can.
    ("MANDATE_EXPIRED", ANY_REASON): FailureClass.REAUTH_REQUIRED,
    ("MANDATE_LIMIT_EXCEEDED", ANY_REASON): FailureClass.REAUTH_REQUIRED,
    ("TOKEN_EXPIRED", ANY_REASON): FailureClass.REAUTH_REQUIRED,
    # Over. Chasing these costs money and goodwill and recovers nothing.
    ("MANDATE_REVOKED", ANY_REASON): FailureClass.TERMINAL,
    ("ACCOUNT_CLOSED", ANY_REASON): FailureClass.TERMINAL,
    ("CUSTOMER_DISPUTED", ANY_REASON): FailureClass.TERMINAL,
}


class UnknownFailureCode(Exception):
    """A code the table does not cover.

    Carries the raw, un-normalised values so the exception list shows exactly
    what Razorpay sent, not what this module made of it.
    """

    def __init__(
        self,
        failure_code: Optional[str],
        failure_reason: Optional[str] = None,
        mapping_version: str = MAPPING_VERSION,
    ) -> None:
        self.failure_code = failure_code
        self.failure_reason = failure_reason
        self.mapping_version = mapping_version
        super().__init__(
            f"no mapping in version {mapping_version} for "
            f"failure_code={failure_code!r} failure_reason={failure_reason!r}; "
            f"route to the exception list rather than defaulting"
        )


def _normalise(value: Optional[str]) -> Optional[str]:
    """Razorpay sends uppercase codes; tolerate case and padding, nothing more."""
    if value is None:
        return None
    stripped = value.strip().upper()
    return stripped or None


def classify(
    failure_code: Optional[str], failure_reason: Optional[str] = None
) -> FailureClass:
    """Return the recovery class for a decline, or raise UnknownFailureCode.

    Looks for an exact (code, reason) entry first, then falls back to the
    code-level entry. There is no third fallback, on purpose.
    """
    code = _normalise(failure_code)
    reason = _normalise(failure_reason)

    if code is not None:
        if reason is not None:
            specific = _FAILURE_CLASSES.get((code, reason))
            if specific is not None:
                return specific

        general = _FAILURE_CLASSES.get((code, ANY_REASON))
        if general is not None:
            return general

    raise UnknownFailureCode(failure_code, failure_reason)


def known_codes() -> frozenset[str]:
    """Every error_code the table covers. For monitoring the exception list."""
    return frozenset(code for code, _ in _FAILURE_CLASSES)
