# 电商智能客服系统 · 第 1 章「纯对话」

FastAPI + LangChain 实现的最小可运行链路：SSE 流式多轮对话 + 结构化售后提取，模型走 OpenAI 协议直连（默认 DeepSeek，可换 GPT/Ollama/Claude）。

## 环境

- Python 3.12（本机用 `C:\Users\26531\AppData\Local\Programs\Python\Python312\python.exe`，勿用默认 3.14——路径含中文会编码报错）
- pip + venv（无 uv）

## 安装

```bash
python -m venv .venv
source .venv/Scripts/activate      # Windows Git Bash
pip install -r requirements.txt
```

> 本机 pypi.org 不可达时，追加镜像 `-i https://pypi.tuna.tsinghua.edu.cn/simple/`。

## 配置

复制 `.env.example` 为 `.env`，填入 DeepSeek key：

```bash
LLM_PROVIDER=deepseek
LLM_BASE_URL=https://api.deepseek.com
LLM_MODEL=deepseek-chat
LLM_API_KEY=sk-xxxx
```

> `LLM_MODEL` 必须用 `deepseek-chat`（支持 function calling）；`deepseek-reasoner` 不支持结构化输出。

## 启动

```bash
python -m uvicorn app.main:app --reload --port 8000
```

## 演示（验收三条）

**1. SSE 流式对话：**

```bash
curl -N -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id":"s1","message":"你好，我是客户"}'
```

预期逐行输出 `data: {"delta": "..."}` 直到 `data: [DONE]`。

**2. 两轮上下文（同一个 session_id 接着问）：**

```bash
curl -N -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id":"s1","message":"我刚才说的那个问题，你再帮我看看"}'
```

**3. 结构化售后提取：**

```bash
curl -X POST http://localhost:8000/extract \
  -H "Content-Type: application/json" \
  -d '{"text":"订单号 20231015001 的鞋子开胶了，我要退货退款"}'
```

预期 JSON 含 `order_id=20231015001`、`request_type=退货`、`expected_solution=退款`。

## 测试

```bash
python -m pytest
```
