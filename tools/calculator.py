"""
간단한 산술 계산을 위한 LangGraph 도구 모듈
"""

from __future__ import annotations

import ast
import math
import operator
import sys
from typing import Callable, Dict

from langchain_core.tools import tool
from pydantic import BaseModel, Field

Numeric = float | int

_BINARY_OPERATORS: Dict[type[ast.AST], Callable[[Numeric, Numeric], Numeric]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
}

_UNARY_OPERATORS: Dict[type[ast.AST], Callable[[Numeric], Numeric]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

_SUPPORTED_FUNCTIONS: Dict[str, Callable[..., Numeric]] = {
    "abs": abs,
    "round": round,
    "sqrt": math.sqrt,
    "log": math.log,
    "log10": math.log10,
    "exp": math.exp,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
}

_NAMED_CONSTANTS: Dict[str, Numeric] = {
    "pi": math.pi,
    "e": math.e,
}


class CalculatorInput(BaseModel):
    """계산기에 전달되는 입력 스키마"""

    expression: str = Field(
        ...,
        description="계산할 수식을 입력하세요. 사칙연산과 sqrt, log 같은 기본 함수만 지원",
    )


def _evaluate_node(node: ast.AST) -> Numeric:
    """안전한 AST 기반 수식 평가"""
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if sys.version_info < (3, 8) and hasattr(ast, "Num") and isinstance(node, ast.Num):
        return node.n
    if isinstance(node, ast.BinOp):
        if type(node.op) not in _BINARY_OPERATORS:
            raise ValueError("지원하지 않는 연산자가 포함되어 있습니다")
        left = _evaluate_node(node.left)
        right = _evaluate_node(node.right)
        return _BINARY_OPERATORS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp):
        if type(node.op) not in _UNARY_OPERATORS:
            raise ValueError("지원하지 않는 단항 연산자가 포함되어 있습니다")
        operand = _evaluate_node(node.operand)
        return _UNARY_OPERATORS[type(node.op)](operand)
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise ValueError("함수 호출 형식이 올바르지 않습니다")
        func_name = node.func.id
        if func_name not in _SUPPORTED_FUNCTIONS:
            raise ValueError(f"{func_name} 함수는 지원하지 않습니다")
        args = [_evaluate_node(arg) for arg in node.args]
        return _SUPPORTED_FUNCTIONS[func_name](*args)
    if isinstance(node, ast.Name):
        if node.id not in _NAMED_CONSTANTS:
            raise ValueError(f"{node.id} 이름은 지원하지 않습니다")
        return _NAMED_CONSTANTS[node.id]
    raise ValueError("지원하지 않는 표현식입니다")


@tool(
    "calculator",
    args_schema=CalculatorInput,
)
def calculator(expression: str) -> str:
    """
    문자열로 전달된 수식을 계산

    Args:
        expression: 평가할 수식

    Returns:
        계산 결과를 float 형태로 반환
    """
    try:
        parsed = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ValueError("수식 구문이 올바르지 않습니다.") from exc
    except Exception as exc:
        raise ValueError(exc)
    result = _evaluate_node(parsed.body)

    value = float(result)
    if value.is_integer():
        return str(int(value))
    return f"{value:.10g}"


__all__ = ["CalculatorInput", "calculator"]
