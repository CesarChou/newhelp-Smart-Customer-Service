# 电商智能客服系统 · 第 1 章「纯对话」设计文档

日期：2026-10-31
状态：待批准

## 1. 意图（intent / why）

做一个电商智能客服系统。整体会逐步长成「工具调用 + Agent + 记忆 + 向量检索」的完整形态，
本章只做地基：把「模型接入 → 流式多轮对话 → Prompt 管理 → 结构化输出 → 上下文控制」这条最小链路打穿，
后续章节在这条链路上叠加工具与 Agent。

验收对象是接口行为（curl），不是页面。

## 2. 目标 / 非目标 / 反向目标

### 目标（本章交付）
1. **对话接口**：多轮对话，SSE 流式输出、逐 token 推送。
2. **Prompt 管理**：用 LangChain `PromptTemplate` 模板化，System Prompt 写清客服角色设定与行为约束。
3. **结构化输出**：把售后描述提取为固定字段 `{订单号、诉求类型、期望方案}`，用 `with_structured_output` 实现。
4. **最简上下文**：历史消息裁剪 + token 预算控制。

### 非目标（本章明确不做）
- 工具调用、Agent 循环。
- 聊天页面（后续章节用 Vibe Coding 单独做）。
- 持久化、数据库、会话存储（内存即可）。
- Langfuse / Milvus 等后续组件。

### 反向目标（就算技术上跑通也视为失败）
- 为了「通用」引入超出需要的重型多 provider 抽象框架。
- 把上下文管理做成精确 token 计费系统——本章只要「够用且不超预算」的粗粒度控制。
- 未经确认私自更换已锁定的技术选型。

## 3. 约束（constraints）

- **技术栈锁定**：Python + FastAPI + LangChain（`langchain-core` + `langchain-openai` 等）。
- **Python 3.12**：机器上 3.14 默认路径含中文会编码报错，统一用干净路径的 3.12，`pip + venv`（无 uv）。
- **模型接入**：直连上游，应用侧统一 OpenAI 协议，`base_url` / `model` / `api_key` 全部放 `.env`。
- **方案 B（已定）**：薄 provider 工厂，返回统一的 `BaseChatModel`。
- **验收模型**：DeepSeek `deepseek-chat`（支持 function calling）。
- **查文档**：涉及 FastAPI / LangChain 等库的 API 用法，先经 Context7 MCP 查最新官方文档再动手，不凭记忆写。
- 密钥不进 git、不进 dev-notes；提供 `.env.example`，真实 `.env` 由用户填写。

## 4. 方案对比（approaches）

| | 方案 A | 方案 B（已选） |
|---|---|---|
| 做法 | 单一 `ChatOpenAI(base_url, api_key, model)` 直连 | `get_chat_model()` 按 `.env` 的 `LLM_PROVIDER` 路由，返回 `BaseChatModel` |
| 能接的模型 | GPT / DeepSeek / Ollama(OpenAI 兼容) | 上述三家 + Claude（`ChatAnthropic`） |
| Claude | 不支持（协议不同） | 支持 |
| 成本 | 最少代码 | 多一个约 20 行工厂函数 |
| 取舍 | 快，但「都能换着接」的目标不完整 | 一步到位达成换模型只改 `.env` |

**选定 B** 的理由：`PromptTemplate`、`.astream()` 流式、`with_structured_output` 都挂在 `BaseChatModel`
这个统一接口上，B 只增加一个极薄工厂，就把「GPT/Claude/DeepSeek/Ollama 都能换着接」落实了；
第一步验收仍只用 DeepSeek，Claude 分支可先写但标注「未实测」。

## 5. 已定的「怎么做」（由对话确定）

- **目录结构**：分层 `app/routes`、`app/services`、`app/prompts`、`app/schemas`、`app/core`（工厂放 `app/core/llm.py`），为后续工具/Agent/Milvus/Langfuse 留位置。
- **流式**：FastAPI `StreamingResponse` + `media_type="text/event-stream"`，服务端用 `chat_model.astream()` 逐 token 产出，包装成 `data: {...}\n\n`。
- **结构化输出**：`with_structured_output(PydanticSchema)`，走 function calling（`deepseek-chat` 支持）。
- **上下文最简版**：保留最近 N 条消息 + 按「字符数 × 系数」粗估 token（中文约 1 字符 ≈ 1 token），设预算上限，超限从最老消息开始裁剪；系数与预算值通过配置给出。
- **模型工厂**：`.env` 里 `LLM_PROVIDER`（openai/deepseek/ollama/anthropic）+ `LLM_BASE_URL` + `LLM_MODEL` + `LLM_API_KEY`。
- **会话状态**：服务端内存按 `session_id` 持有对话历史；客户端每轮只发 `session_id + 本轮新消息`。System Prompt 由服务端统一注入（客服人设），客户端不可覆盖。
- **结构化输出暴露**：独立非流式接口（`POST /extract`），输入原始售后描述文本，使用专用提取 Prompt（不复用客服人设），返回 JSON。
- **结构化字段语义（已与用户确认）**：`订单号` 字符串、可空；`诉求类型` 枚举 = `退款/退货/换货/维修/投诉/咨询/其他`；`期望方案` 自由文本单值、可空。
- **验证路径**：以真实 DeepSeek key 为主线；另备一个离线 mock `BaseChatModel`（伪造流 + 结构化输出）用于不依赖 key 先验证接口骨架，最终验收仍用真实模型。

## 6. 留给 builder 的（可在实现时自行决定）

- 具体依赖及其版本（实现前用 Context7 确认接口签名）。
- 各路由与 schema 的精确字段命名（会话形态已定：服务端持态 + `session_id`）。
- token 预算的具体数值、裁剪粒度、系数默认值。
- 错误处理与异常映射的细节（上游超时、key 缺失、流中断等）。
- Claude 分支是否在第一步实装（可写但标注未实测，或仅留分支位）。
