"""
현재 시간을 조회하는 LangGraph 도구 모듈
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from langchain_core.tools import tool
from pydantic import BaseModel, Field

DEFAULT_TIME_FORMAT = "%Y-%m-%d %H:%M:%S"


class CurrentTimeInput(BaseModel):
    """현재 시각 조회 도구 입력 스키마"""

    time_format: str = Field(
        DEFAULT_TIME_FORMAT,
        description=(
            "datetime.strftime 포맷 문자열입니다. " "예: %Y-%m-%d %H:%M:%S, %Y/%m/%d."
        ),
    )
    timezone: Optional[str] = Field(
        None,
        description="(선택) IANA 타임존 이름을 입력하세요. 예: Asia/Seoul",
    )


@tool(
    "current_time",
    args_schema=CurrentTimeInput,
)
def current_time(
    time_format: str = DEFAULT_TIME_FORMAT,
    timezone: str | None = None,
) -> str:
    """
    지정된 포맷과 타임존으로 현재 시각을 반환

    Args:
        time_format: datetime.strftime 포맷 문자열
        timezone: IANA 타임존 문자열. 지정하지 않으면 시스템 로컬 시간을 사용

    Returns:
        포맷팅된 현재 시각 문자열
    """
    tz_info = None
    if timezone:
        try:
            tz_info = ZoneInfo(timezone)
        except ZoneInfoNotFoundError as exc:
            raise ValueError("존재하지 않는 타임존입니다") from exc

    now = datetime.now(tz_info)
    return now.strftime(time_format)


__all__ = ["CurrentTimeInput", "current_time"]
