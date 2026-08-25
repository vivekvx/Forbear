"""The application. One screen, one stream, one webhook endpoint.

Deliberately small. This is the seam between Forbear and the outside world:
Razorpay pushes events in at one end, an operator watches decisions come out at
the other, and everything interesting happens in between, where this module
cannot reach it.

The pool lives on app.state because the stream endpoint holds a connection open
for the length of a run, and a request-scoped dependency would hand it back to
the pool halfway through the cycle it is streaming.
"""

from __future__ import annotations

import os
import pathlib

import asyncpg
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from forbear.api import stream, webhooks

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
SCHEMA_PATH = REPO_ROOT / "schema.sql"
FRONTEND_PATH = pathlib.Path(__file__).resolve().parent / "static" / "index.html"

DSN = os.environ.get("FORBEAR_DSN", "postgres:///forbear")


async def ensure_schema(pool) -> None:
    """Load schema.sql if the database is empty.

    A demo that needs a documented setup step is a demo that fails in front of
    an audience. Existing tables are left alone: this creates, it never
    migrates.
    """
    async with pool.acquire() as conn:
        exists = await conn.fetchval("SELECT to_regclass('public.at_risk_records')")
        if exists is None:
            await conn.execute(SCHEMA_PATH.read_text())


def create_app() -> FastAPI:
    app = FastAPI(title="Forbear", version="0.1.0")

    # Local development only. The stream gets read from a different port often
    # enough that refusing it wastes more time than it saves, and there is
    # nothing behind this API worth stealing.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    app.include_router(webhooks.router)
    app.include_router(stream.router)

    @app.on_event("startup")
    async def _startup() -> None:
        app.state.pool = await asyncpg.create_pool(DSN, min_size=2, max_size=10)
        await ensure_schema(app.state.pool)

    @app.on_event("shutdown")
    async def _shutdown() -> None:
        pool = getattr(app.state, "pool", None)
        if pool is not None:
            await pool.close()

    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(FRONTEND_PATH, media_type="text/html")

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    return app


app = create_app()
