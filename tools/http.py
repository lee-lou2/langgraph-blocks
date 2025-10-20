"""
간단한 HTTP GET 요청을 수행하는 LangGraph 도구
"""

from __future__ import annotations

from typing import Dict, Optional, Union

import httpx
from langchain_core.tools import tool
from pydantic import BaseModel, Field, HttpUrl


class HttpGetInput(BaseModel):
    """HTTP GET 도구 입력 스키마"""

    url: HttpUrl = Field(..., description="요청할 완전한 URL")
    params: Optional[Dict[str, str]] = Field(
        default=None,
        description="선택 사항. 쿼리 스트링 파라미터를 딕셔너리로 전달",
    )
    headers: Optional[Dict[str, str]] = Field(
        default=None,
        description="선택 사항. 추가 요청 헤더",
    )
    timeout: float = Field(
        10.0,
        ge=1.0,
        le=60.0,
        description="요청 타임아웃(초)",
    )


@tool(
    "http_get",
    args_schema=HttpGetInput,
)
def http_get(
    url: Union[str, HttpUrl],
    params: Optional[Dict[str, str]] = None,
    headers: Optional[Dict[str, str]] = None,
    timeout: float = 10.0,
):
    """
    HTTP GET 요청에 대한 결과 조회

    Args:
        url: 호출할 URL 경로
        params: 파라미터 정보
        headers: 헤더 입력 정보
        timeout: 요청 타임아웃

    Returns:
        호출 결과에 대한 상태 및 내용 반환
    """
    client = httpx.Client(timeout=timeout, follow_redirects=True)
    try:
        response = client.get(str(url), params=params, headers=headers)
    finally:
        client.close()
    response.raise_for_status()
    return {
        "status_code": response.status_code,
        "reason": response.reason_phrase,
        "headers": dict(response.headers),
        "body": response.text[:1000],
    }


__all__ = [
    "HttpGetInput",
    "http_get",
]
