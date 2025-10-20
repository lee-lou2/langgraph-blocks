"""
작업 공간 내 파일을 다루기 위한 LangGraph 도구 모음
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

from langchain_core.tools import tool
from pydantic import BaseModel, Field

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent


def _resolve_path(path: str) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = WORKSPACE_ROOT / candidate
    resolved = candidate.resolve()
    try:
        resolved.relative_to(WORKSPACE_ROOT)
    except ValueError as exc:
        raise ValueError("작업 공간 바깥 경로는 접근할 수 없습니다.") from exc
    return resolved


class ReadFileInput(BaseModel):
    path: str = Field(
        ..., description="읽을 파일 경로. 작업 공간 기준 상대 경로를 권장합니다."
    )
    encoding: str = Field("utf-8", description="파일 인코딩.")


class WriteFileInput(BaseModel):
    path: str = Field(..., description="작성할 파일 경로.")
    content: str = Field(..., description="저장할 텍스트 내용.")
    encoding: str = Field("utf-8", description="파일 인코딩.")
    append: bool = Field(False, description="True면 기존 파일 끝에 내용을 추가합니다.")


class ListDirectoryInput(BaseModel):
    path: str | None = Field(
        None,
        description="목록을 확인할 경로. None이면 작업 공간 루트를 사용합니다.",
    )


@tool(
    "read_file",
    args_schema=ReadFileInput,
)
def read_file(path: str, encoding: str = "utf-8") -> str:
    """
    작업 공간 내 텍스트 파일 읽기

    Args:
        path: 파일 경로
        encoding: 인코딩 방식 지정(기본: utf-8)

    Returns:
        조회된 파일 내용 반환
    """
    file_path = _resolve_path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"{file_path} 파일을 찾을 수 없습니다.")
    return file_path.read_text(encoding=encoding)


@tool(
    "write_file",
    args_schema=WriteFileInput,
)
def write_file(
    path: str,
    content: str,
    encoding: str = "utf-8",
    append: bool = False,
) -> str:
    """
    작업 공간 내 텍스트 파일을 생성하거나 수정

    Args:
        path: 파일 경로
        content: 파일에 저장할 내용
        encoding: 인코딩 방식 지정(기본: utf-8)
        append: 기존 파일에 내용 추가 여부

    Returns:
        생성/수정된 파일의 정보 반환
    """
    file_path = _resolve_path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    with file_path.open(mode, encoding=encoding) as file:
        file.write(content)
    return str(file_path.relative_to(WORKSPACE_ROOT))


@tool(
    "list_directory",
    args_schema=ListDirectoryInput,
)
def list_directory(path: str | None = None) -> List[str]:
    """
    작업 공간 디렉터리의 항목을 나열

    Args:
        path: 파일 경로

    Returns:
        조회된 파일 리스트 반환
    """
    directory = _resolve_path(path or ".")
    if not directory.exists() or not directory.is_dir():
        raise NotADirectoryError(f"{directory} 디렉터리를 찾을 수 없습니다.")
    entries: Iterable[Path] = sorted(directory.iterdir())
    return [entry.name + ("/" if entry.is_dir() else "") for entry in entries]


__all__ = [
    "ReadFileInput",
    "WriteFileInput",
    "ListDirectoryInput",
    "read_file",
    "write_file",
    "list_directory",
]
