"""Hard limits.

Invariant 5: these are hard limits, not soft preferences. Nothing here is
overridable from the environment at runtime, because a regulatory cap that can
be raised by an env var is not a cap.

This module and forbear.models.models are the only things the allocator and the
guard are allowed to share. Keep it to constants; a helper function here would
become shared logic, and shared logic is exactly what the guard must not have
with the code it re-validates.
"""

from __future__ import annotations

from datetime import timedelta, timezone

# Maximum attempts Forbear will ever consume against one at-risk record,
# counting every attempt row regardless of outcome.
MAX_ATTEMPTS = 4

# Minimum gap between the last executed attempt and the next one.
ATTEMPT_COOLDOWN = timedelta(hours=24)

# How long a pre-debit notification remains valid as authorisation for a debit.
NOTIFICATION_VALIDITY = timedelta(hours=24)

# Fixed UTC+05:30. Deliberately not zoneinfo("Asia/Kolkata"): a tz database
# update could silently move these boundaries. India does not observe DST, but
# the code should not be quietly relying on that.
IST = timezone(timedelta(hours=5, minutes=30), name="IST")

# NPCI-permitted debit windows, as minutes since IST midnight, half-open
# [start, end): before 10:00, 13:00-17:00, and after 21:30.
NPCI_DEBIT_WINDOWS_IST = (
    (0, 10 * 60),
    (13 * 60, 17 * 60),
    (21 * 60 + 30, 24 * 60),
)
