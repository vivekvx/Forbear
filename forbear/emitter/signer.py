"""Sender-side HMAC-SHA256 signing.

Mirrors forbear.api.webhooks.signature_is_valid exactly: same key encoding,
same digest, same hex output. A signature produced here is accepted there
without the receiver ever needing to special-case a local sender.
"""

from __future__ import annotations

import hashlib
import hmac


def sign_payload(raw_body: bytes, secret: str) -> str:
    """Hex HMAC-SHA256 of raw_body under secret.

    raw_body must be the exact bytes that will be sent -- signing a
    re-serialization of the same dict is not guaranteed to match.
    """
    return hmac.new(secret.encode("utf-8"), raw_body, hashlib.sha256).hexdigest()
