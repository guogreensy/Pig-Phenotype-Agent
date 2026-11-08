"""agent 层（阶段 C）。

模块划分
--------
* :mod:`src.agent.llm_client`  —— 支持 function calling 的对话客户端（T11）
* :mod:`src.agent.planner`     —— 系统提示词与消息装配（T12）
* :mod:`src.agent.executor`    —— 工具调用循环（T13）
* :mod:`src.agent.report`      —— 结果组装与渲染（T14）

与视觉客户端的区别
------------------
视觉客户端只发单轮图文消息、不涉及函数调用；本包面向**多轮 + 工具调用**，
因此单独实现，但复用同一套重试、退避与限流基础件，避免两套行为不一致。
"""