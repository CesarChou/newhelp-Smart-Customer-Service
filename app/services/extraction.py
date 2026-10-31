"""Structured extraction service (Task 5).

``extract`` turns a free-text user message into an
:class:`~app.schemas.extraction.AfterSaleExtraction` using the structured LLM
from :func:`app.core.llm.get_structured_llm` (default function-calling).

The function-calling wrapper raises when the model emits an out-of-enum
``request_type`` (pydantic ``ValidationError``) or malformed output
(``OutputParserException``). To keep the HTTP endpoint from 500-ing on a
single bad model output, ``extract`` retries once with an explicit instruction
that ``request_type`` must stay within the enum, then falls back to
``request_type=其他`` with the other fields left as ``None``.
"""

from langchain_core.exceptions import OutputParserException
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from pydantic import ValidationError

from app.core.llm import get_structured_llm
from app.prompts.extraction import EXTRACTION_PROMPT
from app.schemas.extraction import AfterSaleExtraction, RequestType

# Appended to the system prompt on the retry attempt (Review Focus #5).
_ENUM_HINT = "诉求类型必须在给定枚举内"


def _messages(text: str, *, extra_hint: bool = False) -> list[BaseMessage]:
    """Build the [system prompt, user text] pair for one extraction attempt."""
    system = EXTRACTION_PROMPT
    if extra_hint:
        system = f"{system}\n{_ENUM_HINT}"
    return [SystemMessage(content=system), HumanMessage(content=text)]


async def extract(text: str) -> AfterSaleExtraction:
    """Extract the structured after-sale intent from ``text``.

    Never raises for a malformed model output: it tries the base prompt, retries
    once with the enum-constraint hint appended, then falls back to
    ``request_type=其他`` (other fields ``None``).
    """
    llm = get_structured_llm(AfterSaleExtraction)

    for extra_hint in (False, True):
        try:
            result = await llm.ainvoke(_messages(text, extra_hint=extra_hint))
        except (OutputParserException, ValidationError, ValueError):
            # Parse/validation/malformed-output failures (e.g. a plain
            # ValueError from non-dict tool args or invalid JSON) -> retry
            # with the hint, then fall back. Genuine upstream/network errors
            # (not ValueError subclasses) still propagate.
            continue

        if isinstance(result, AfterSaleExtraction):
            return result
        # The wrapper may return None when the model emits no tool call at all;
        # treat that like a parse failure rather than leaking None upstream.

    return AfterSaleExtraction(request_type=RequestType.其他)
