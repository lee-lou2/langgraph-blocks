"""
안전한 범위에서 파이썬 코드를 실행하는 LangGraph 도구
"""

from __future__ import annotations

import io
import math
from contextlib import redirect_stderr, redirect_stdout
from typing import Any, Dict

from langchain_core.tools import tool
from pydantic import BaseModel, Field

SAFE_BUILTINS: Dict[str, Any] = {
    "abs": abs,
    "all": all,
    "any": any,
    "bool": bool,
    "dict": dict,
    "enumerate": enumerate,
    "float": float,
    "int": int,
    "len": len,
    "list": list,
    "max": max,
    "min": min,
    "pow": pow,
    "print": print,
    "range": range,
    "round": round,
    "sum": sum,
    "tuple": tuple,
    "zip": zip,
}

SAFE_GLOBALS: Dict[str, Any] = {
    "__builtins__": SAFE_BUILTINS,
    "math": math,
}


class PythonREPLInput(BaseModel):
    """파이썬 REPL 도구 입력 스키마"""

    code: str = Field(..., description="실행할 파이썬 코드 블록")


@tool(
    "python_repl",
    args_schema=PythonREPLInput,
)
def python_repl(code: str) -> str:
    """
    제한된 전역/내역 범위 내에서 파이썬 코드를 실행

    Args:
        code: 실행할 코드

    Returns:
        표준 출력이 존재하면 해당 내용을, 그렇지 않으면 표현식의 값을 문자열로 반환
    """
    local_env: Dict[str, Any] = {}
    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()

    compiled = None
    try:
        compiled = compile(code, "<python_repl>", "eval")
        mode = "eval"
    except SyntaxError:
        compiled = compile(code, "<python_repl>", "exec")
        mode = "exec"

    try:
        with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
            if mode == "eval":
                result = eval(compiled, SAFE_GLOBALS.copy(), local_env)
            else:
                exec(compiled, SAFE_GLOBALS.copy(), local_env)
                result = None
    except Exception as exc:
        error_output = stderr_buffer.getvalue().strip()
        message = error_output if error_output else f"{type(exc).__name__}: {exc}"
        raise ValueError(message) from exc

    output = stdout_buffer.getvalue().strip()
    if output:
        return output
    if result is not None:
        return repr(result)
    return "None"


__all__ = [
    "PythonREPLInput",
    "python_repl",
]
