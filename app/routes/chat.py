"""``POST /chat`` SSE route (Task 6).

Streams one chat turn as Server-Sent Events and persists the completed turn to
the shared session store so the next turn sees it (R1).

``stream_chat`` already injects the current user turn into the message list
(see :func:`app.services.chat.assemble_messages`), so the user message must NOT
be appended before streaming — that would duplicate it. Instead we accumulate
the assistant tokens during the stream and only afterwards persist BOTH the
user turn and the assistant reply.
"""

import asyncio
import json

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, HumanMessage

from app.schemas.chat import ChatRequest
from app.services.chat import store, stream_chat

router = APIRouter()


@router.post("/chat")
async def chat(req: ChatRequest) -> StreamingResponse:
    """Stream one chat turn as SSE, then persist the completed turn (R1)."""

    async def gen():
        accumulated: list[str] = []
        try:
            async for token in stream_chat(req.session_id, req.message):
                accumulated.append(token)
                yield f"data: {json.dumps({'delta': token}, ensure_ascii=False)}\n\n"
        except asyncio.CancelledError:
            # Client disconnected mid-stream: let the cancellation propagate so
            # the upstream generator is closed and no partial turn is stored.
            raise

        store.append(req.session_id, HumanMessage(req.message))
        store.append(req.session_id, AIMessage("".join(accumulated)))
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )
