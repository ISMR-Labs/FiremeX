"""FastAPI application factory."""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import AsyncIterator
from pathlib import Path

from fastapi import APIRouter, Depends, FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from ..config import Settings, load_site_config
from ..logging_conf import configure_logging
from ..supervisor import Supervisor
from . import routes_config, routes_incidents, routes_system, routes_twiml
from .deps import get_supervisor

log = logging.getLogger(__name__)
WEB_DIR = Path(__file__).resolve().parent.parent / "web"

live_router = APIRouter(tags=["live"])


@live_router.websocket("/api/live")
async def live_updates(websocket: WebSocket) -> None:
    """Stream detections, incidents and alert progress to the dashboard."""
    supervisor: Supervisor | None = getattr(websocket.app.state, "supervisor", None)
    await websocket.accept()
    if supervisor is None:  # pragma: no cover - only before startup
        await websocket.close(code=1013)
        return
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

    app.include_router(routes_system.router)
    app.include_router(routes_config.router)
    app.include_router(routes_incidents.router)
    app.include_router(routes_twiml.router)
    app.include_router(live_router)

    if WEB_DIR.is_dir():
        app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")

        @app.get("/", include_in_schema=False)
        async def dashboard() -> FileResponse:
            return FileResponse(WEB_DIR / "index.html")

    else:  # pragma: no cover - only if the package is installed without web assets

        @app.get("/", include_in_schema=False)
        async def docs_redirect() -> RedirectResponse:
            return RedirectResponse("/docs")

    return app


__all__ = ["create_app", "get_supervisor", "Depends"]
