"""LLM 抽象 + 假模型。

真客户端什么时候写: D2。届时只要实现同一个 stream() 接口,
把 SSE 事件翻译成下面三种 StreamEvent, 主循环一行都不用改。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, AsyncIterator, Protocol

from agent.types import ToolCall


@dataclass
class TextDelta:
    text: str


@dataclass
class ToolCallEvent:
    call: ToolCall


@dataclass
class Done:
    stop_reason: str = "end_turn"


StreamEvent = TextDelta | ToolCallEvent | Done


class LLM(Protocol):
    def stream(
        self, messages: list[Any], tools: list[dict[str, Any]]
    ) -> AsyncIterator[StreamEvent]:
        ...


class FakeLLM:
    """按脚本回放事件的假模型, 让主循环可以离线测试。

    script 的每个元素 = 一次请求要吐出的完整事件序列。
    """

    def __init__(self, script: list[list[StreamEvent]]) -> None:
        self._script = script
        self.turns = 0

    async def stream(
        self, messages: list[Any], tools: list[dict[str, Any]]
    ) -> AsyncIterator[StreamEvent]:
        if self.turns >= len(self._script):
            raise AssertionError(
                f"FakeLLM script exhausted at turn {self.turns + 1}: "
                "主循环请求的次数比预期多(可能陷入了死循环)"
            )
        events = self._script[self.turns]
        self.turns += 1
        for event in events:
            yield event
