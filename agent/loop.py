"""主循环 —— 这个项目里唯一需要你自己写的地方。

契约
----
输入: messages(就地追加)、llm、registry、max_turns
输出: 模型最后一段正文 (str)
副作用: messages 被追加

必须满足的 5 条 (测试会逐条验)
--------------------------------------------------------------------------
1. 模型这一轮没有 tool_calls  → 立刻返回 reply.text, 不再请求
2. 有 tool_calls           → 先 append 一条 assistant 消息(带 tool_calls),
                             再 append 一条 user 消息(带 tool_results)
3. 每个 tool_result.tool_use_id 必须等于对应 tool_call.id, 且顺序一致
4. 工具抛异常 / 未知工具 / 参数非法 → 折叠成 is_error=True 的结果,
   循环继续; 不能让异常穿透出去
5. 迭代达到 max_turns 仍未收敛 → raise MaxTurnsExceeded

实现提示
--------------------------------------------------------------------------
- 先把流聚合成 Reply: 遍历 llm.stream(...), TextDelta 累加文本,
  ToolCallEvent 收集 ToolCall, Done 记 stop_reason
- 聚合循环结束 = 一轮结束, 只有聚合完才知道要不要调工具
- 工具是串行执行还是并行, 现在不用管; 先串行, 能跑通再说

写完之后: python -m pytest -q  应该看到 5 passed
"""

from __future__ import annotations

from typing import Any

from agent.llm import LLM
from agent.tools import ToolRegistry
from agent.types import Message, Reply


class MaxTurnsExceeded(Exception):
    """达到迭代上限仍未收敛。"""


async def collect(llm: LLM, messages: list[Message], registry: ToolRegistry) -> Reply:
    """把一次 stream() 聚合成 Reply。留给你的第一个小任务。"""
    raise NotImplementedError("TODO: 遍历 llm.stream(...) 累积 text / tool_calls / stop_reason")


async def run(
    messages: list[Message],
    llm: LLM,
    registry: ToolRegistry,
    max_turns: int = 20,
) -> str:
    raise NotImplementedError("TODO: 按上面的 5 条契约实现主循环")
