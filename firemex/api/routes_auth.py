"""Login, logout, session identity, and password change."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from pydantic import BaseModel, Field

from .. import auth
from ..config import Settings
from ..supervisor import Supervisor
from .deps import (
    _client_ip,
    clear_session_cookies,
    current_user,
    get_settings,
    get_supervisor,
    resolve_principal,
    set_session_cookies,
)

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api/auth", tags=["auth"])

SESSION_TTL_HOURS = 12


class LoginRequest(BaseModel):
    username: str = Field(max_length=64)
    password: str = Field(max_length=256)


class PasswordChangeRequest(BaseModel):
    current_password: str = Field(max_length=256)
    new_password: str = Field(max_length=256)


@router.post("/login")
async def login(
    body: LoginRequest,
    request: Request,
    response: Response,
    supervisor: Supervisor = Depends(get_supervisor),
    settings: Settings = Depends(get_settings),
) -> dict:
    user, reason = await supervisor.store.authenticate(body.username, body.password)
    if user is None:
        # One message for every failure mode except the lockout, which the operator
        # needs to be told about so they stop retrying. Distinguishing "no such
        # user" from "wrong password" would just enumerate accounts.
        log.warning("failed login for %r from %s (%s)", body.username, _client_ip(request), reason)
        if reason == "locked":
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="too many failed attempts; this account is locked for 5 minutes",
            )
        if reason == "disabled":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN, detail="this account is disabled"
            )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid username or password"
        )

    token, token_hash = auth.new_session_token()
    csrf_token = auth.new_csrf_token()
    await supervisor.store.create_session(
        user_id=user.id,
        token_hash=token_hash,
        csrf_token=csrf_token,
        ttl_hours=SESSION_TTL_HOURS,
        user_agent=request.headers.get("User-Agent"),
        ip=_client_ip(request),
    )
    set_session_cookies(response, token, csrf_token, settings, SESSION_TTL_HOURS)
    log.info("login: %s (%s) from %s", user.username, user.role, _client_ip(request))
    return {
        "username": user.username,
        "role": user.role,
        "must_change_password": user.must_change_password,
    }


@router.post("/logout")
async def logout(
    request: Request,
    response: Response,
    supervisor: Supervisor = Depends(get_supervisor),
    principal: auth.Principal = Depends(current_user),
) -> dict:
    token = request.cookies.get(auth.SESSION_COOKIE)
    if token:
        await supervisor.store.delete_session(auth.hash_session_token(token))
    clear_session_cookies(response)
    log.info("logout: %s", principal.username)
    return {"logged_out": True}


@router.get("/session")
async def session_state(request: Request) -> dict:
    """Whether this browser is signed in. Always 200.

    The shell calls this on every page load. Using an endpoint that 401s for the
    normal not-logged-in case would print a red error in the browser console on
    every fresh visit, which trains operators to ignore console errors.
    """
    resolved = await resolve_principal(request)
    if resolved is None:
        return {"authenticated": False}
    principal, _csrf = resolved
    return {
        "authenticated": True,
        "username": principal.username,
        "role": principal.role,
        "must_change_password": principal.must_change_password,
        "is_admin": principal.is_admin,
    }


@router.get("/me")
async def me(principal: auth.Principal = Depends(current_user)) -> dict:
    return {
        "username": principal.username,
        "role": principal.role,
        "must_change_password": principal.must_change_password,
        "is_admin": principal.is_admin,
    }


@router.post("/password")
async def change_password(
    body: PasswordChangeRequest,
    response: Response,
    request: Request,
    supervisor: Supervisor = Depends(get_supervisor),
    settings: Settings = Depends(get_settings),
    principal: auth.Principal = Depends(current_user),
) -> dict:
    """Change your own password.

    Requires the current password even though the session is already
    authenticated, so an unattended browser cannot be used to lock out its owner.
    """
    user, reason = await supervisor.store.authenticate(principal.username, body.current_password)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=(
                "current password is incorrect"
                if reason == "invalid_credentials"
                else f"cannot verify current password ({reason})"
            ),
        )
    try:
        new_password = auth.validate_password(body.new_password)
    except auth.AuthError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    if new_password == body.current_password:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="new password must be different"
        )

    await supervisor.store.set_password(principal.username, new_password)
    # Revoke every session, including this one: a password change should
    # invalidate anything an attacker already holds. Then issue a fresh session so
    # the operator is not bounced back to the login screen mid-incident.
    await supervisor.store.delete_user_sessions(principal.id)
    token, token_hash = auth.new_session_token()
    csrf_token = auth.new_csrf_token()
    await supervisor.store.create_session(
        user_id=principal.id,
        token_hash=token_hash,
        csrf_token=csrf_token,
        ttl_hours=SESSION_TTL_HOURS,
        user_agent=request.headers.get("User-Agent"),
        ip=_client_ip(request),
    )
    set_session_cookies(response, token, csrf_token, settings, SESSION_TTL_HOURS)
    log.info("password changed for %s", principal.username)
    return {"changed": True, "username": principal.username}
