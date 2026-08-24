"""Classifier tests.

A lookup table cannot be subtly wrong, only silently incomplete, so most of
these tests are about what happens at the edge of the table rather than inside
it.
"""

from __future__ import annotations

import pytest

from forbear.models.models import FailureClass
from forbear.services.classifier import (
    MAPPING_VERSION,
    UnknownFailureCode,
    classify,
    known_codes,
)

EXPECTED = {
    "INSUFFICIENT_FUNDS": FailureClass.TIME_DEPENDENT,
    "BANK_ACCOUNT_DEBITED_ALREADY": FailureClass.TIME_DEPENDENT,
    "GATEWAY_ERROR": FailureClass.TRANSIENT,
    "TIMEOUT": FailureClass.TRANSIENT,
    "BAD_GATEWAY": FailureClass.TRANSIENT,
    "MANDATE_EXPIRED": FailureClass.REAUTH_REQUIRED,
    "MANDATE_LIMIT_EXCEEDED": FailureClass.REAUTH_REQUIRED,
    "TOKEN_EXPIRED": FailureClass.REAUTH_REQUIRED,
    "MANDATE_REVOKED": FailureClass.TERMINAL,
    "ACCOUNT_CLOSED": FailureClass.TERMINAL,
    "CUSTOMER_DISPUTED": FailureClass.TERMINAL,
}


@pytest.mark.parametrize(
    ("code", "expected"), sorted(EXPECTED.items()), ids=sorted(EXPECTED)
)
def test_known_code_maps_to_its_class(code, expected):
    assert classify(code) is expected


@pytest.mark.parametrize(
    ("code", "expected"), sorted(EXPECTED.items()), ids=sorted(EXPECTED)
)
def test_unknown_reason_falls_back_to_the_code(code, expected):
    """Razorpay adds reason strings freely; a new one must not unclassify a code."""
    assert classify(code, "some reason we have never seen") is expected


def test_table_covers_exactly_the_documented_codes():
    assert known_codes() == frozenset(EXPECTED)


def test_every_failure_class_is_reachable():
    """A class no code maps to is either dead or a missing row."""
    assert set(EXPECTED.values()) == set(FailureClass)


@pytest.mark.parametrize(
    "code",
    ["insufficient_funds", "  INSUFFICIENT_FUNDS  ", "Insufficient_Funds"],
)
def test_case_and_padding_are_tolerated(code):
    assert classify(code) is FailureClass.TIME_DEPENDENT


def test_unknown_code_raises_with_raw_values_attached():
    with pytest.raises(UnknownFailureCode) as excinfo:
        classify("  card_On_Fire ", "reason as sent")

    error = excinfo.value
    # Raw, not normalised: the exception list must show what Razorpay sent.
    assert error.failure_code == "  card_On_Fire "
    assert error.failure_reason == "reason as sent"
    assert error.mapping_version == MAPPING_VERSION
    assert "card_On_Fire" in str(error)
    assert MAPPING_VERSION in str(error)


@pytest.mark.parametrize("code", [None, "", "   "])
def test_missing_code_raises_rather_than_defaulting(code):
    with pytest.raises(UnknownFailureCode) as excinfo:
        classify(code)

    assert excinfo.value.failure_code == code


@pytest.mark.parametrize(
    "code", ["INSUFFICIENT", "INSUFFICIENT_FUNDS_2", "TIMEOUT_ERROR"]
)
def test_near_miss_code_is_not_matched(code):
    """Prefix and substring matches are not matches."""
    with pytest.raises(UnknownFailureCode):
        classify(code)


def test_mapping_version_is_recorded_and_non_empty():
    assert isinstance(MAPPING_VERSION, str)
    assert MAPPING_VERSION.strip()


def test_classifier_returns_the_enum_not_a_string():
    result = classify("MANDATE_REVOKED")
    assert isinstance(result, FailureClass)
    assert result.value == "terminal"
