"""Customer-service chat service (Task 4).

Orchestrates one streaming chat turn:

* :func:`assemble_messages` builds the full message list: the server-injected
  system prompt + the trimmed session history + the new user turn.
* :func:`stream_chat` streams the assistant reply one token at a time.

Scope boundary: ``stream_chat`` only yields tokens — it never persists anything
to the :class:`~app.services.memory.SessionStore`. Appending the user turn and
the assistant reply is the route layer's job (Task 6).
"""

from collections.abc import AsyncIterator

from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from app.core.llm import get_chat_model
from app.prompts.customer_service import CUSTOMER_SERVICE_SYSTEM_PROMPT
from app.services.memory import SessionStore

# Shared in-memory session store. The route layer (Task 6) appends the user
# turn and the assistant reply here; ``assemble_messages`` reads from it.
store = SessionStore()

_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", CUSTOMER_SERVICE_SYSTEM_PROMPT),
        MessagesPlaceholder("history"),
        ("human", "{input}"),
    ]
)


def assemble_messages(session_id: str, user_input: str) -> list[BaseMessage]:
    """Build the full message list for one chat turn.

    The system prompt is injected server-side (clients cannot override it); the
    middle is the trimmed session history; the last message is the new user turn.
    """
    history = store.get_history(session_id)
    return _PROMPT.invoke({"history": history, "input": user_input}).to_messages()


async def stream_chat(session_id: str, user_input: str) -> AsyncIterator[str]:
    """Stream the assistant reply for one chat turn, token by token.

    Yields one string per upstream chunk. If a chunk carries empty content, an
    empty string is yielded in its place; if the upstream produces no chunks at
    all, a single empty string is yielded so the client never hangs waiting.
    """
    messages = assemble_messages(session_id, user_input)
    model = get_chat_model()

    yielded = False
    async for chunk in model.astream(messages):
        content = chunk.content
        yielded = True
        yield content if content else ""
    if not yielded:
        yield ""
