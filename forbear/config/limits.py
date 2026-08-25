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

# Pre-debit notification timing. Two bounds, and they point in opposite
# directions, which is the whole reason this was wrong once already.
#
# NPCI's e-mandate rules require the customer to be told BEFORE the debit, with
# at least a day to see it and act on it. So a notification authorises a debit
# only once it has aged past the lead time: one sent ten seconds ago authorises
# nothing, however recent and however genuine.
#
# The comparison against this is strict. "At least 24 hours' notice" is not
# satisfied by a notification sent at exactly the debit instant minus 24 hours,
# and a boundary that admits the exact instant is one an auditor gets to argue
# about.
NOTIFICATION_MIN_LEAD = timedelta(hours=24)

# The other end. Consent goes stale: a notification from last month must not
# authorise a debit today, or the lead-time rule would be satisfied forever by
# a single notification sent once.
NOTIFICATION_MAX_AGE = timedelta(days=7)

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
