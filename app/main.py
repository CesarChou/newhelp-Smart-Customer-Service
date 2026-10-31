"""FastAPI application entrypoint (Task 6).

Wires the chat and extraction routers into a single ``app`` instance exported
for ``uvicorn`` / ``fastapi.testclient.TestClient``, and serves the chat
frontend (``app/static/index.html``) at ``/``.
"""

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.routes import chat, extraction

app = FastAPI(title="电商客服系统")

app.include_router(chat.router)
app.include_router(extraction.router)

STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
async def index() -> FileResponse:
    """Serve the chat frontend."""
    return FileResponse(STATIC_DIR / "index.html")
