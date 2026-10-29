"""对话数据结构。

这里只有一个不变量值得记住:
    assistant 消息里的 tool_calls 和 user 消息里的 tool_results
    必须按 tool_use_id 一一对应、且顺序一致。
后面做截断/压缩时, 任何破坏这个不变量的裁剪都会让下一次 API 请求直接 400。
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ToolCall:
    id: str
    name: str
    args: dict


@dataclass
class ToolResult:
    tool_use_id: str
    content: str
    is_error: bool = False


@dataclass
class Message:
    role: str  # "user" | "assistant"
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_results: list[ToolResult] = field(default_factory=list)


@dataclass
class Reply:
    """一轮模型回复的聚合结果。"""

    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    stop_reason: str = "end_turn"


def assistant_message(reply: Reply) -> Message:
    return Message(role="assistant", content=reply.text, tool_calls=reply.tool_calls)


def tool_results_message(results: list[ToolResult]) -> Message:
    # 工具结果按 user 角色回填, 和 Anthropic Messages API 的形状一致
    return Message(role="user", tool_results=results)
