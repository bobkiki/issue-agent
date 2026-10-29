"""M1 验收测试 —— 这 5 个测试就是主循环的规格。

跑法: python -m pytest -q
目标: 5 passed

注意这里用 asyncio.run 包一层, 而不是 pytest-asyncio:
少一个依赖, 少一类环境问题。
"""

from __future__ import annotations

import asyncio

import pytest

from agent.llm import Done, FakeLLM, TextDelta, ToolCallEvent
from agent.loop import MaxTurnsExceeded, run
from agent.tools import ToolRegistry, echo_tool
from agent.types import Message, ToolCall


def make_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(echo_tool())
    return registry


def call(call_id: str, name: str = "Echo", args: dict | None = None) -> ToolCallEvent:
    return ToolCallEvent(ToolCall(id=call_id, name=name, args=args or {"text": "x"}))


def test_1_no_tool_call_returns_text() -> None:
    """模型直接回答 → 返回文本, 历史长度 2。"""
    llm = FakeLLM([[TextDelta("hi"), Done("end_turn")]])
    messages = [Message(role="user", content="hello")]

    out = asyncio.run(run(messages, llm, make_registry()))

    assert out == "hi"
    assert len(messages) == 2
    assert messages[1].role == "assistant"


def test_2_one_tool_round_pairs_ids() -> None:
    """一轮工具调用 → 历史 4 条, 且 tool_use_id 与 tool_call.id 配对。"""
    llm = FakeLLM(
        [
            [call("t1"), Done("tool_use")],
            [TextDelta("done"), Done("end_turn")],
        ]
    )
    messages = [Message(role="user", content="echo please")]

    out = asyncio.run(run(messages, llm, make_registry()))

    assert out == "done"
    assert len(messages) == 4, "应为 user / assistant / user(tool_results) / assistant"

    assert messages[1].role == "assistant"
    assert messages[1].tool_calls[0].id == "t1"

    assert messages[2].role == "user"
    result = messages[2].tool_results[0]
    assert result.tool_use_id == "t1"
    assert result.content == "x"
    assert result.is_error is False


def test_3_tool_exception_is_folded() -> None:
    """工具抛异常 → 折叠成 error 结果, 循环继续并返回最终文本。"""
    llm = FakeLLM(
        [
            [call("t1", args={}), Done("tool_use")],  # 缺 text 字段 → 参数非法
            [TextDelta("recovered"), Done("end_turn")],
        ]
    )
    messages = [Message(role="user", content="bad args")]

    out = asyncio.run(run(messages, llm, make_registry()))

    assert out == "recovered"
    assert messages[2].tool_results[0].is_error is True
    assert messages[2].tool_results[0].content


def test_4_unknown_tool_is_folded() -> None:
    """未知工具 → 同样折叠成 error 结果, 不崩。"""
    llm = FakeLLM(
        [
            [call("t1", name="NoSuchTool", args={}), Done("tool_use")],
            [TextDelta("ok"), Done("end_turn")],
        ]
    )
    messages = [Message(role="user", content="call a ghost")]

    out = asyncio.run(run(messages, llm, make_registry()))

    assert out == "ok"
    assert messages[2].tool_results[0].is_error is True


def test_5_max_turns_raises() -> None:
    """模型永远要调工具 → 达到上限抛 MaxTurnsExceeded。"""
    script = [[call(f"t{i}"), Done("tool_use")] for i in range(10)]
    messages = [Message(role="user", content="loop forever")]

    with pytest.raises(MaxTurnsExceeded):
        asyncio.run(run(messages, FakeLLM(script), make_registry(), max_turns=3))
