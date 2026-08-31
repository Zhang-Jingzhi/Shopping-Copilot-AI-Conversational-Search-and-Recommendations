# 多轮对话状态、记忆与动态上下文

这是 Shopping Copilot 项目的第 3 部分，负责维护用户在多轮购物对话中的
当前需求与长期偏好。本组件不负责商品目录检索；它向检索、排序和 Agent
编排层提供可执行的结构化上下文。

## 负责内容

- 用于信息逐轮累积和需求覆盖的动态状态机。
- 硬约束、软偏好、显式排除项、槽位优先级及衰减。
- 短期会话状态与长期用户画像蒸馏。
- 紧凑的对话摘要及可执行的 `ContextSnapshot` 输出。
- 自适应下一步决策：检索、重路由、澄清、比较或转化。

## 目录结构

```text
src/state_memory/     组件实现
tests/                多轮对话回归测试
docs/                 API 与设计契约
```

## 运行测试

在当前目录执行：

```powershell
& ..\.tools\python312\python.exe -m unittest discover -s tests -v
```

官方 Agent 通过 `StateMemoryManager.update(...)` 调用本组件，并取得供检索与
编排层使用的 `ContextSnapshot`。详细字段和行为请阅读
[模块设计与 API 契约](docs/state_memory_manager.md)。
