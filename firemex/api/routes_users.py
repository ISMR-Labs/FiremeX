"""User administration. Admin only."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from .. import auth
from ..supervisor import Supervisor
from .deps import get_supervisor, require_admin

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api/users", tags=["users"], dependencies=[Depends(require_admin)])


class CreateUserRequest(BaseModel):
    username: str = Field(max_length=64)
    password: str = Field(max_length=256)
    role: str = Field(default=auth.ROLE_VIEWER)


class UpdateUserRequest(BaseModel):
    role: str | None = None
    disabled: bool | None = None
    #: Set by an admin for someone who has forgotten theirs. The user is required
    #: to change it at next login.
    new_password: str | None = Field(default=None, max_length=256)


@router.get("")
async def list_users(supervisor: Supervisor = Depends(get_supervisor)) -> list[dict]:
    return await supervisor.store.list_users()


@router.get("/roles")
async def list_roles() -> list[dict]:
    return [
        {
            "id": auth.ROLE_VIEWER,
            "name": "Viewer",
            "description": "See cameras, incidents and evidence. Cannot change anything.",
        },
        {
            "id": auth.ROLE_OPERATOR,
            "name": "Operator",
            "description": "Everything a viewer can do, plus cancel and review incidents.",
        },
        {
            "id": auth.ROLE_ADMIN,
            "name": "Administrator",
            "description": "Full access: cameras, contacts, model settings and users.",
        },
    ]


@router.post("", status_code=status.HTTP_201_CREATED)
async def create_user(
    body: CreateUserRequest,
    supervisor: Supervisor = Depends(get_supervisor),
    principal: auth.Principal = Depends(require_admin),
) -> dict:
    try:
        username = auth.validate_username(body.username)
        password = auth.validate_password(body.password)
    except auth.AuthError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    if body.role not in auth.ROLES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"role must be one of {', '.join(auth.ROLES)}",
        )
    try:
        created = await supervisor.store.create_user(
            username, password, body.role, created_by=principal.username
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    log.info("user %s (%s) created by %s", username, body.role, principal.username)
    return created


@router.patch("/{username}")
async def update_user(
    username: str,
    body: UpdateUserRequest,
    supervisor: Supervisor = Depends(get_supervisor),
    principal: auth.Principal = Depends(require_admin),
) -> dict:
    target = await supervisor.store.get_user(username)
    if target is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="user not found")

    fields: dict = {}
    if body.role is not None:
        if body.role not in auth.ROLES:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"role must be one of {', '.join(auth.ROLES)}",
            )
        # Demoting or disabling the last admin would leave an appliance nobody can
        # administer -- exactly the wrong discovery to make during an incident.
        if target.role == auth.ROLE_ADMIN and body.role != auth.ROLE_ADMIN:
            await _assert_another_admin_remains(supervisor, target.username)
        fields["role"] = body.role

    if body.disabled is not None:
        if body.disabled and target.role == auth.ROLE_ADMIN:
            await _assert_another_admin_remains(supervisor, target.username)
        fields["disabled"] = body.disabled

    if body.new_password is not None:
        try:
            password = auth.validate_password(body.new_password)
        except auth.AuthError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
        fields["password_hash"] = auth.hash_password(password)
        # An admin-set password is a temporary credential the admin has seen, so
        # the user must replace it before doing anything else.
        fields["must_change_password"] = True
        fields["failed_logins"] = 0
        fields["locked_until"] = None

    if not fields:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="nothing to update")

    updated = await supervisor.store.update_user(target.username, **fields)
    # Any of these changes should take effect immediately, not at session expiry.
    if "password_hash" in fields or fields.get("disabled") or "role" in fields:
        await supervisor.store.delete_user_sessions(target.id)
    log.info("user %s updated by %s (%s)", target.username, principal.username, sorted(fields))
    return updated


@router.post("/{username}/unlock")
async def unlock_user(
    username: str, supervisor: Supervisor = Depends(get_supervisor)
) -> dict:
    updated = await supervisor.store.update_user(username, failed_logins=0, locked_until=None)
    if updated is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="user not found")
    return updated


@router.delete("/{username}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_user(
    username: str,
    supervisor: Supervisor = Depends(get_supervisor),
    principal: auth.Principal = Depends(require_admin),
) -> None:
    if username.lower() == principal.username:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="you cannot delete your own account"
        )
    try:
        deleted = await supervisor.store.delete_user(username)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    if not deleted:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="user not found")
    log.info("user %s deleted by %s", username, principal.username)


async def _assert_another_admin_remains(supervisor: Supervisor, username: str) -> None:
    if await supervisor.store.count_active_admins(excluding=username) < 1:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="cannot remove the last administrator",
        )
