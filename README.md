# AI Proxy Detector — 中转站模型真伪探测器

探测 OpenAI 兼容 API 中转站背后的真实模型。14 维度交叉验证，层层剥开伪装。

```
probe-station/
├── cli/                  ← 终端 CLI 工具（零依赖）
│   └── probe.py
├── web/                  ← Docker Web 可视化版（已部署 :9090）
│   ├── docker-compose.yml
│   ├── Dockerfile
│   ├── requirements.txt
│   └── app/
│       ├── main.py
│       └── static/index.html
├── desktop/              ← pywebview 桌面版（可编译为 .exe）
│   ├── app_desktop.py
│   ├── index.html
│   ├── build.bat
│   └── requirements-desktop.txt
└── README.md
```

---

## 三种运行方式

| 版本 | 启动方式 | 依赖 | 适用场景 |
|------|---------|------|----------|
| **CLI** | `python cli/probe.py` | Python 3 标准库 | 终端执行，可集成脚本 |
| **Web** | `cd web && docker compose up -d` | Docker | 团队共享，浏览器访问 |
| **Desktop** | `python desktop/app_desktop.py` | Python + pywebview | 桌面应用，可编译 .exe 分发 |

---

## 1. CLI 版本

零 pip 依赖，仅需 Python 3 标准库。

```bash
# 交互模式
python cli/probe.py

# 命令行模式
python cli/probe.py --url https://api.example.com --key sk-xxx --model gpt-4o

# 生成 HTML 报告
python cli/probe.py --url https://api.example.com --key sk-xxx --html
```

输出终端彩色报告，14 维度逐项结论，自动计算置信度。

---

## 2. Web 版本（Docker）

```bash
cd web
docker compose up -d --build
```

访问 **http://localhost:9090**，输入 Base URL 和 API Key，点击探测。左侧面板实时更新统计，右侧卡片网格逐步展示 12 个步骤，最终输出置信度判词 + 完整证据链。

修改端口：编辑 `web/docker-compose.yml` 中 `ports` 映射。

---

## 3. Desktop 版本

### 直接运行

```bash
pip install pywebview
python desktop/app_desktop.py
```

打开 1200x800 原生窗口，功能与 Web 版一致。Python 探测引擎直接与前端通信，无需 HTTP 服务。

### 编译为独立 .exe

```bash
cd desktop
build.bat
```

生成 `dist/AI-Proxy-Detector.exe`，单文件，无 Python 环境也可运行。依赖 Edge WebView2（Windows 10/11 已内置）。

---

## 探测维度

| # | 维度 | 说明 |
|---|------|------|
| 1 | /v1/models 扫描 | 列出所有可用模型 ID 和所属组织 |
| 2 | 模型名矩阵嗅探 | 20+ 厂家模型名逐一试探，统计通过/拒绝 |
| 3 | 错误消息分析 | 检查 503 错误是否泄露渠道分组名 |
| 4 | 身份追问 | 直问"你是什么模型" |
| 5 | 越狱 / 角色扮演 | DAN 模式、假装管理员等 |
| 6 | 知识截止日期 | 自述 training cutoff |
| 7 | 安全对齐指纹 | 政治问题 + 越狱指令的对齐风格 |
| 8 | 数学推理陷阱 | 9.11 vs 9.9 经典题 |
| 9 | Prompt Token 注入 | 检测 prompt_tokens 是否异常偏高 |
| 10 | Function Calling | 工具调用是否可用 |
| 11 | HTTP 响应头 | 识别 New-API / One-API 等框架 |
| 12 | 速度基准 | t/s 吞吐量测量 |

---

## 架构要点

- **探测引擎**：`cli/probe.py`、`web/app/main.py`、`desktop/app_desktop.py` 共享同一套探测逻辑
- **Web 通信**：FastAPI SSE (text/event-stream)，前端 fetch + ReadableStream 逐步消费
- **Desktop 通信**：pywebview JS API bridge，Python 通过 `evaluate_js` 直接推送事件到前端 DOM
- **前端 UI**：同款响应式双栏布局，PC + 移动端兼容，暗色主题

---

## 联系方式

QQ: 466656664