"""工具注册表。

一个工具 = 名字 + 描述 + 参数 JSON Schema + 一个可调用对象。
Schema 既喂给模型做决策, 也用来在执行前校验参数 —— 一份定义两处用。
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from agent.types import ToolCall, ToolResult


class ToolError(Exception):
    """工具层错误: 未知工具 / 参数非法。由主循环折叠成 error 结果。"""


class UnknownTool(ToolError):
    pass


class InvalidArgs(ToolError):
    pass


@dataclass
class ToolSpec:
    name: str
    description: str
    params_schema: dict[str, Any]
    fn: Callable[[dict[str, Any]], Awaitable[str] | str]


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> None:
        self._tools[spec.name] = spec

    def get(self, name: str) -> ToolSpec | None:
        return self._tools.get(name)

    def specs(self) -> list[dict[str, Any]]:
        """给模型的工具描述列表。"""
        return [
            {
                "name": t.name,
                "description": t.description,
                "input_schema": t.params_schema,
            }
            for t in self._tools.values()
        ]

    async def run(self, call: ToolCall) -> ToolResult:
        """执行一次工具调用。

        约定: 未知工具 / 参数非法一律抛 ToolError, 工具内部异常原样抛。
        要不要捕获、捕获后怎么告诉模型, 是主循环的决定 —— 不是这里的。
        """
        spec = self._tools.get(call.name)
        if spec is None:
            raise UnknownTool(f"unknown tool: {call.name}")

        self._validate(spec, call.args)

        out = spec.fn(call.args)
        if inspect.isawaitable(out):
            out = await out
        return ToolResult(tool_use_id=call.id, content=str(out))

    @staticmethod
    def _validate(spec: ToolSpec, args: dict[str, Any]) -> None:
        schema = spec.params_schema
        for key in schema.get("required", []):
            if key not in args:
                raise InvalidArgs(f"{spec.name}: missing required field '{key}'")
        for key, prop in schema.get("properties", {}).items():
            if key in args and prop.get("type") == "string" and not isinstance(args[key], str):
                raise InvalidArgs(f"{spec.name}: field '{key}' must be a string")


def echo_tool() -> ToolSpec:
    """最小工具, 只用来跑通主循环。"""

    async def _echo(args: dict[str, Any]) -> str:
        return args["text"]

    return ToolSpec(
        name="Echo",
        description="Echo back the given text. Use it to verify the tool loop works.",
        params_schema={
            "type": "object",
            "properties": {"text": {"type": "string", "description": "Text to echo"}},
            "required": ["text"],
        },
        fn=_echo,
    )
