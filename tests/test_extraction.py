"""Annotated-sample verification for structured extraction (Task 5).

The real model is verified in Task 7; here a fake structured model (a
``get_structured_llm`` stand-in returning a fixed ``AfterSaleExtraction``)
verifies ``extract()``'s orchestration and the illegal-enum-value fallback.
"""

import pytest
from langchain_core.exceptions import OutputParserException
from pydantic import ValidationError

from app.prompts.extraction import EXTRACTION_PROMPT
from app.schemas.extraction import AfterSaleExtraction, RequestType
from app.services.extraction import extract

# 标注样例（真实模型校验放到 Task 7 验收）
SAMPLES = [
    (
        "订单号 20231015001 的鞋子开胶了，我要退货退款。",
        {"order_id": "20231015001", "request_type": "退货", "expected_solution": "退款"},
    ),
    (
        "物流三天没动，帮我催一下。",
        {"order_id": None, "request_type": "咨询", "expected_solution": "催物流"},
    ),
    (
        "客服态度太差，我要投诉！",
        {"order_id": None, "request_type": "投诉", "expected_solution": None},
    ),
]


def test_request_type_enum_has_exact_chinese_values():
    assert [m.value for m in RequestType] == [
        "退款",
        "退货",
        "换货",
        "维修",
        "投诉",
        "咨询",
        "其他",
    ]


def test_after_sale_extraction_nulls_default():
    parsed = AfterSaleExtraction(request_type=RequestType.其他)
    assert parsed.order_id is None
    assert parsed.expected_solution is None
    assert parsed.request_type == RequestType.其他


def test_samples_fit_after_sale_schema():
    # 标注样例：每条期望 dict 都能被 schema 原样接受，且诉求类型落在枚举内
    for _text, expected in SAMPLES:
        parsed = AfterSaleExtraction.model_validate(expected)
        assert parsed.model_dump() == expected
        assert isinstance(parsed.request_type, RequestType)


def _invalid_enum_validation_error() -> ValidationError:
    """Produce a real pydantic ValidationError from an out-of-enum value."""
    try:
        AfterSaleExtraction(request_type="退钱")  # type: ignore[arg-type]
    except ValidationError as exc:
        return exc
    raise AssertionError("expected ValidationError")


class _FakeStructuredLLM:
    """Stand-in for ``get_structured_llm(schema)``; records every ``ainvoke``."""

    def __init__(self, result=None, *, always_fail=False):
        self._result = result
        self._always_fail = always_fail
        self.calls: list = []

    async def ainvoke(self, messages):
        self.calls.append(messages)
        if self._always_fail:
            raise OutputParserException("malformed tool call")
        return self._result


@pytest.mark.asyncio
async def test_extract_returns_parsed_result(monkeypatch):
    text, expected = SAMPLES[0]
    fixed = AfterSaleExtraction.model_validate(expected)
    fake = _FakeStructuredLLM(result=fixed)
    monkeypatch.setattr("app.services.extraction.get_structured_llm", lambda _: fake)

    result = await extract(text)

    assert result == fixed
    assert result.request_type == RequestType.退货
    # 编排：系统 prompt 在前、用户原文在后，且只调用一次
    assert len(fake.calls) == 1
    assert fake.calls[0][0].content == EXTRACTION_PROMPT
    assert fake.calls[0][1].content == text


@pytest.mark.asyncio
async def test_extract_retries_once_with_enum_hint(monkeypatch):
    """Review Focus #5: 首次解析失败 -> 追加「诉求类型必须在给定枚举内」重试。"""

    class _FlakyStructuredLLM:
        def __init__(self):
            self.calls = []

        async def ainvoke(self, messages):
            self.calls.append(messages)
            if len(self.calls) == 1:
                raise _invalid_enum_validation_error()
            return AfterSaleExtraction(
                order_id=None, request_type=RequestType.其他, expected_solution=None
            )

    fake = _FlakyStructuredLLM()
    monkeypatch.setattr("app.services.extraction.get_structured_llm", lambda _: fake)

    result = await extract("帮我退钱")

    assert result.request_type == RequestType.其他
    assert len(fake.calls) == 2
    assert "诉求类型必须在给定枚举内" in fake.calls[1][0].content


@pytest.mark.asyncio
async def test_extract_falls_back_to_other_on_persistent_failure(monkeypatch):
    fake = _FakeStructuredLLM(always_fail=True)
    monkeypatch.setattr("app.services.extraction.get_structured_llm", lambda _: fake)

    result = await extract("随便什么")

    assert result.request_type == RequestType.其他
    assert result.order_id is None
    assert result.expected_solution is None
    assert len(fake.calls) == 2  # 一次原样、一次追加提示，仍失败则兜底


@pytest.mark.asyncio
async def test_extract_falls_back_on_plain_value_error(monkeypatch):
    """A malformed model output surfacing as a plain ``ValueError`` (e.g.
    ``json.JSONDecodeError`` for invalid JSON / non-dict tool args) must not
    500: retry once with the enum hint, then fall back to ``其他``."""

    class _ValueErrorLLM:
        def __init__(self):
            self.calls = []

        async def ainvoke(self, messages):
            self.calls.append(messages)
            raise ValueError("tool args must be a dict")

    fake = _ValueErrorLLM()
    monkeypatch.setattr("app.services.extraction.get_structured_llm", lambda _: fake)

    result = await extract("随便什么")

    assert result.request_type == RequestType.其他
    assert result.order_id is None
    assert result.expected_solution is None
    assert len(fake.calls) == 2  # 原样 + 追加提示，仍抛 ValueError 则兜底


@pytest.mark.asyncio
async def test_extract_falls_back_on_none_no_tool_call(monkeypatch):
    """The wrapper returns ``None`` when the model emits no tool call at all;
    treat that like a parse failure and fall back after the retry."""
    fake = _FakeStructuredLLM(result=None)
    monkeypatch.setattr("app.services.extraction.get_structured_llm", lambda _: fake)

    result = await extract("随便什么")

    assert result.request_type == RequestType.其他
    assert result.order_id is None
    assert result.expected_solution is None
    assert len(fake.calls) == 2  # 一次原样、一次追加提示，均无 tool call 则兜底
