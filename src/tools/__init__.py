"""工具层（阶段 B）。

模块划分
--------
* :mod:`src.tools.registry`    —— 注册表、参数校验、调用日志（T6）
* :mod:`src.tools.mock_data`   —— 模拟养殖数据（T7）
* :mod:`src.tools.data_tools`  —— 数据查询工具（T8）
* :mod:`src.tools.vision_tool` —— 姿态识别工具（T9）
* :mod:`src.tools.rag_tool`    —— 知识库检索工具（T10）
* :mod:`src.tools.builder`     —— 默认工具集装配（可按需裁剪，供消融实验用）

一律遵守两条约束：
1. **不伪造**：任何失败都返回 ``ok=False`` 与真实原因，不用预设答案顶替；
2. **标明来源**：模拟数据在返回值里带 ``source="simulated"``，
   知识库结果带许可与原文偏移，方便报告与前端如实展示。
"""