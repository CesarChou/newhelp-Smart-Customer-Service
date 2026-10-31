"""In-memory session store with token-budget trimming (Task 3).

``SessionStore`` keeps a plain ``dict[str, list[BaseMessage]]`` and returns a
trimmed history on ``get_history``:

1. Tail-trim to the last ``context_max_messages`` messages.
2. Coarse-estimate tokens as ``sum(len(content) / chars_per_token)``
   (list-of-blocks content is flattened to text first).
3. While over ``context_max_tokens``, drop from the OLDEST end, always
   keeping ``SystemMessage`` entries and the latest ``HumanMessage``.
"""

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from app.config import settings


def _content_to_text(content: object) -> str:
    """Flatten a message ``content`` (str or list of blocks) into plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                text = block.get("text")
                if text:
                    parts.append(str(text))
            else:
                parts.append(str(block))
        return "".join(parts)
    return str(content)


def _estimate_tokens(messages: list[BaseMessage]) -> float:
    chars_per_token = settings.chars_per_token if settings.chars_per_token > 0 else 1.0
    return sum(len(_content_to_text(m.content)) / chars_per_token for m in messages)


class SessionStore:
    """In-memory, per-session message history."""

    def __init__(self) -> None:
        self._sessions: dict[str, list[BaseMessage]] = {}

    def append(self, session_id: str, message: BaseMessage) -> None:
        """Append a message to the session's history."""
        self._sessions.setdefault(session_id, []).append(message)

    def get_history(self, session_id: str) -> list[BaseMessage]:
        """Return the trimmed history for ``session_id`` (``[]`` if unknown)."""
        stored = self._sessions.get(session_id, [])
        if not stored:
            return []

        history = list(stored[-settings.context_max_messages :])

        # The latest (most recent) user message must never be trimmed away.
        latest_user: BaseMessage | None = None
        for message in reversed(history):
            if isinstance(message, HumanMessage):
                latest_user = message
                break

        while _estimate_tokens(history) > settings.context_max_tokens:
            drop_index: int | None = None
            for index, message in enumerate(history):
                if isinstance(message, SystemMessage):
                    continue
                if message is latest_user:
                    continue
                drop_index = index
                break

            if drop_index is None:
                break  # nothing left we are allowed to drop
            history.pop(drop_index)

        return history
