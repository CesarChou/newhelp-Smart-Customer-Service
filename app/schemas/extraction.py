"""Structured after-sale extraction schema (Task 5).

Defines the enum of after-sale request types and the Pydantic model returned
by :func:`app.services.extraction.extract`. The route layer (Task 6) consumes
:class:`AfterSaleExtraction` directly.
"""

from enum import Enum

from pydantic import BaseModel, Field


class RequestType(str, Enum):
    """售后诉求类型（枚举值必须为中文，模型只能从这些值中选择）。"""

    退款 = "退款"
    退货 = "退货"
    换货 = "换货"
    维修 = "维修"
    投诉 = "投诉"
    咨询 = "咨询"
    其他 = "其他"


class AfterSaleExtraction(BaseModel):
    """从用户消息中结构化提取的售后诉求。"""

    order_id: str | None = Field(default=None, description="订单号，消息中未出现时为空")
    request_type: RequestType = Field(
        description="诉求类型，只能取给定枚举值之一，无法判断时取「其他」"
    )
    expected_solution: str | None = Field(
        default=None, description="期望方案（自由文本），无法识别时为空"
    )
