"""Authentication primitives: password hashing, session tokens, URL redaction.

Dependency-free by design -- `hashlib.scrypt` is in the standard library and is a
memory-hard KDF, so there is no reason to pull in bcrypt or argon2 for this.

Sessions are server-side: a random token goes to the browser in an HttpOnly
cookie, and only its SHA-256 hash is stored. That means a database dump cannot be
replayed as a login, and sessions stay individually revocable -- which matters
because this UI can silence a fire alert.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import re
import secrets
from dataclasses import dataclass
from urllib.parse import quote, urlsplit, urlunsplit

# scrypt parameters. n=2**15 costs roughly 100 ms and 32 MB per verification on a
# modern CPU: slow enough to make offline cracking expensive, fast enough that a
# login is not noticeable.
_SCRYPT_N = 2**15
_SCRYPT_R = 8
_SCRYPT_P = 1
_SALT_BYTES = 16
_DKLEN = 32
#: OpenSSL refuses scrypt above a 32 MiB default, and these parameters need
#: exactly 128 * r * n = 32 MiB, so the ceiling has to be raised explicitly or
#: every hash raises "memory limit exceeded".
_MAXMEM = 128 * _SCRYPT_R * _SCRYPT_N * 2

SESSION_COOKIE = "firemex_session"
CSRF_COOKIE = "firemex_csrf"
CSRF_HEADER = "X-FiremeX-CSRF"

#: The seeded first-run account. Login is allowed but every other action is
#: refused until the password is changed -- see `must_change_password`.
DEFAULT_USERNAME = "admin"
DEFAULT_PASSWORD = "admin"

ROLE_ADMIN = "admin"
ROLE_OPERATOR = "operator"
ROLE_VIEWER = "viewer"
ROLES = (ROLE_ADMIN, ROLE_OPERATOR, ROLE_VIEWER)

#: What each role may do. Viewers watch; operators act on incidents; admins
#: configure. Deliberately coarse -- a fire dashboard with a confusing permission
#: model is worse than one with three obvious tiers.
ROLE_RANK = {ROLE_VIEWER: 0, ROLE_OPERATOR: 1, ROLE_ADMIN: 2}

USERNAME_RE = re.compile(r"^[a-zA-Z0-9._-]{3,32}$")
MIN_PASSWORD_LENGTH = 8


class AuthError(ValueError):
    """Raised for invalid credentials, weak passwords, or bad usernames."""


def hash_password(password: str) -> str:
    """Hash a password with a fresh random salt.

    Returned format is ``scrypt$n$r$p$salt_b64$hash_b64`` so the parameters travel
    with the hash and can be raised later without invalidating existing users.
    """
    if not isinstance(password, str) or not password:
        raise AuthError("password must not be empty")
    salt = secrets.token_bytes(_SALT_BYTES)
    digest = hashlib.scrypt(
        password.encode("utf-8"),
        salt=salt,
        n=_SCRYPT_N,
        r=_SCRYPT_R,
        p=_SCRYPT_P,
        dklen=_DKLEN,
        maxmem=_MAXMEM,
    )
    return "$".join(
        [
            "scrypt",
            str(_SCRYPT_N),
            str(_SCRYPT_R),
            str(_SCRYPT_P),
            base64.b64encode(salt).decode(),
            base64.b64encode(digest).decode(),
        ]
    )


def verify_password(password: str, encoded: str) -> bool:
    """Check a password against a stored hash. Never raises on malformed input."""
    if not password or not encoded:
        return False
    try:
        scheme, n_raw, r_raw, p_raw, salt_b64, hash_b64 = encoded.split("$")
        if scheme != "scrypt":
            return False
        salt = base64.b64decode(salt_b64)
        expected = base64.b64decode(hash_b64)
        actual = hashlib.scrypt(
            password.encode("utf-8"),
            salt=salt,
            n=int(n_raw),
            r=int(r_raw),
            p=int(p_raw),
            dklen=len(expected),
            # Derived from the stored parameters, so raising the cost later stays
            # backward compatible with hashes written today.
            maxmem=128 * int(r_raw) * int(n_raw) * 2,
        )
    except (ValueError, TypeError):
        return False
    return hmac.compare_digest(actual, expected)


def validate_username(username: str) -> str:
    username = (username or "").strip()
    if not USERNAME_RE.match(username):
        raise AuthError(
            "username must be 3-32 characters, letters/digits/dot/underscore/hyphen only"
        )
    return username.lower()


def validate_password(password: str) -> str:
    """Enforce a length floor only.

    Composition rules ("one capital, one symbol") measurably push people toward
    predictable substitutions and sticky notes. Length is the property that
    actually helps, and this is a self-hosted appliance, not a public service.
    """
    if not isinstance(password, str) or len(password) < MIN_PASSWORD_LENGTH:
        raise AuthError(f"password must be at least {MIN_PASSWORD_LENGTH} characters")
    # No explicit check against DEFAULT_PASSWORD: it is shorter than the minimum, so
    # the length floor already rejects it. A separate guard would be dead code that
    # merely looks reassuring.
    return password


def new_session_token() -> tuple[str, str]:
    """Return ``(token, token_hash)``. Only the hash is ever persisted."""
    token = secrets.token_urlsafe(32)
    return token, hash_session_token(token)


def hash_session_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def new_csrf_token() -> str:
    return secrets.token_urlsafe(24)


def csrf_matches(cookie_value: str | None, header_value: str | None) -> bool:
    """Double-submit check, in constant time."""
    if not cookie_value or not header_value:
        return False
    return hmac.compare_digest(cookie_value, header_value)


def role_allows(role: str, required: str) -> bool:
    return ROLE_RANK.get(role, -1) >= ROLE_RANK.get(required, 99)


@dataclass(frozen=True, slots=True)
class Principal:
    """The authenticated caller, as far as the routes are concerned."""

    id: int
    username: str
    role: str
    must_change_password: bool

    @property
    def is_admin(self) -> bool:
        return self.role == ROLE_ADMIN

    def can(self, required: str) -> bool:
        return role_allows(self.role, required)


# ---- RTSP credential handling -------------------------------------------


def build_rtsp_url(url: str, username: str | None, password: str | None) -> str:
    """Inject credentials into an RTSP URL.

    Cameras are configured in the UI as a plain URL plus separate username and
    password fields, so the secret never has to be typed into a URL bar, is stored
    in its own field, and can be redacted independently on the way out.
    """
    if not username and not password:
        return url
    parts = urlsplit(url)
    host = parts.hostname or ""
    if not host:
        return url
    if parts.port:
        host = f"{host}:{parts.port}"
    # Percent-encode so a password containing '@' or ':' cannot break the URL.
    credentials = quote(username or "", safe="")
    if password:
        credentials = f"{credentials}:{quote(password, safe='')}"
    return urlunsplit(
        (parts.scheme, f"{credentials}@{host}", parts.path, parts.query, parts.fragment)
    )


def redact_url(url: str) -> str:
    """Strip credentials from a URL so it is safe to return over the API or log.

    ``GET /api/cameras`` must never hand back a camera password, even to an
    authenticated admin: it would end up in browser history, screenshots and
    support tickets.
    """
    if not url or "@" not in url:
        return url
    scheme, separator, rest = url.partition("://")
    if not separator:
        return url
    _, _, host_part = rest.rpartition("@")
    return f"{scheme}://***@{host_part}"
