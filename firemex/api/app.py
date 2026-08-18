"""FastAPI application factory."""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import AsyncIterator
from pathlib import Path

from fastapi import APIRouter, FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from .. import auth
from ..config import Settings, load_site_config
from ..logging_conf import configure_logging
from ..supervisor import Supervisor
from . import (
    routes_auth,
    routes_config,
    routes_incidents,
    routes_stream,
    routes_system,
    routes_twiml,
    routes_users,
)
from .deps import get_supervisor

log = logging.getLogger(__name__)
WEB_DIR = Path(__file__).resolve().parent.parent / "web"

live_router = APIRouter(tags=["live"])


@live_router.websocket("/api/live")
async def live_updates(websocket: WebSocket) -> None:
    """Stream detections, incidents and alert progress to the dashboard.

    Authenticated from the session cookie, which the browser sends with the
    WebSocket handshake. An unauthenticated socket is closed rather than accepted:
    these events name cameras and incidents.
    """
    supervisor: Supervisor | None = getattr(websocket.app.state, "supervisor", None)
    if supervisor is None:  # pragma: no cover - only before startup
        await websocket.close(code=1013)
        return

    token = websocket.cookies.get(auth.SESSION_COOKIE)
    resolved = (
        await supervisor.store.resolve_session(auth.hash_session_token(token)) if token else None
    )
    if resolved is None:
        # 1008 = policy violation. The dashboard treats it as "go to login".
        await websocket.close(code=1008)
        return
    principal, _csrf = resolved
    if principal.must_change_password:
        await websocket.close(code=1008)
        return

    await websocket.accept()
    try:
        await websocket.send_json({"type": "status", **supervisor.status()})
        async for event in supervisor.bus.subscribe():
            await websocket.send_json(event)
    except WebSocketDisconnect:
        pass
    except Exception:  # noqa: BLE001 - a broken socket must not disturb the pipeline
        log.debug("live websocket closed with error", exc_info=True)
    finally:
        with contextlib.suppress(RuntimeError):
            await websocket.close()


def create_app(
    settings: Settings | None = None,
    supervisor: Supervisor | None = None,
    start_supervisor: bool = True,
) -> FastAPI:
    """Build the app.

    ``supervisor`` and ``start_supervisor`` exist so tests can inject a supervisor
    wired to a stub detector and synthetic cameras.
    """
    settings = settings or Settings()

    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        configure_logging(settings.log_level, settings.log_json)
        instance = supervisor or Supervisor(settings, site=load_site_config(settings.config_path))
        app.state.supervisor = instance
        if start_supervisor:
            await instance.start()
        else:
            # Tests and embedded use still need the schema to exist.
            settings.ensure_dirs()
            instance.store.create_all()

        # First run: seed admin/admin, flagged so it cannot be used for anything
        # except changing itself.
        if await instance.store.ensure_default_admin():
            log.warning(
                "no users existed, so the default account %r was created with password %r. "
                "You must change it at first login; nothing else will work until you do.",
                auth.DEFAULT_USERNAME,
                auth.DEFAULT_PASSWORD,
            )
        await instance.store.purge_expired_sessions()

        try:
            yield
        finally:
            if start_supervisor:
                with contextlib.suppress(asyncio.CancelledError):
                    await instance.stop()

    app = FastAPI(
        title="FiremeX",
        version="0.1.0",
        summary="Fire and smoke detection for CCTV, with automated emergency calls",
        description=(
            "FiremeX is a supplementary monitoring aid. It is not a certified fire "
            "alarm system and must not replace code-compliant detectors, sprinklers "
            "or a monitored alarm panel."
        ),
        lifespan=lifespan,
    )
    app.state.settings = settings

    app.include_router(routes_auth.router)
    app.include_router(routes_users.router)
    app.include_router(routes_system.router)
    app.include_router(routes_config.router)
    app.include_router(routes_stream.router)
    app.include_router(routes_incidents.router)
    # Twilio cannot log in: these are protected by signature verification instead.
    app.include_router(routes_twiml.router)
    app.include_router(live_router)

    if WEB_DIR.is_dir():
        app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")

        @app.get("/", include_in_schema=False)
        async def dashboard() -> FileResponse:
            # The shell is served unauthenticated and decides for itself whether to
            # show the login screen; every byte of actual data behind it requires a
            # session.
            return FileResponse(WEB_DIR / "index.html")

    else:  # pragma: no cover - only if the package is installed without web assets

        @app.get("/", include_in_schema=False)
        async def docs_redirect() -> RedirectResponse:
            return RedirectResponse("/docs")

    return app


__all__ = ["create_app", "get_supervisor"]
