"""Chat request schema (Task 6).

``ChatRequest`` is the JSON body of ``POST /chat``. The route layer (Task 6)
consumes it and streams the assistant reply back over SSE.
"""

from pydantic import BaseModel


class ChatRequest(BaseModel):
    """One chat turn: the session to append to and the user's message."""

    session_id: str
    message: str
