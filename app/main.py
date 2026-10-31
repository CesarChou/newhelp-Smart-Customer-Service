"""FastAPI application entrypoint (Task 6).

Wires the chat and extraction routers into a single ``app`` instance exported
for ``uvicorn`` / ``fastapi.testclient.TestClient``.
"""

from fastapi import FastAPI

from app.routes import chat, extraction

app = FastAPI(title="电商客服系统")

app.include_router(chat.router)
app.include_router(extraction.router)
