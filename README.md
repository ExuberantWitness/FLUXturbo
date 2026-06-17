# FLUXturbo ⚡🧬

*Flux-Insight Claim Chain v2 提取引擎 — 将任意学术文本转化为结构化知识图谱的 MCP Server*

[English](#english) | 中文

> 📰 **v0.1.0** (2026-06-12) — Initial release: CC v2 aligned extraction, 8 atom types + 11 edge types + Rho evidence records, OntologyGatekeeper validation, MCP Server stdio transport.

---

> 🧬 **FLUXturbo 是 Flux-Insight 生态的知识提取引擎。** 它不是一个完整的科研自动化系统，而是将 Claim Chain v2 本体论（8 种 atom 类型、11 种 edge 类型、Rho 证据记录、OntologyGatekeeper 验证规则）封装为一个独立的 MCP Server。任何 MCP 客户端（Claude Code、VS Code、自定义 Agent）都可以通过 4 个标准化工具调用它，从学术文本中提取结构化的 claim chain。

[![License](https://img.shields.io/badge/License-MIT-green?style=flat)](LICENSE) · [![MCP](https://img.shields.io/badge/MCP-4_tools-blue?style=flat)]() · [![CC](https://img.shields.io/badge/Claim_Chain-v2-8A2BE2?style=flat)]() · [![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat)]() · [![Ontology](https://img.shields.io/badge/Atoms-8_types-red?style=flat)]() · [![Edges](https://img.shields.io/badge/Edges-11_types-orange?style=flat)]()

---

## 🔬 核心亮点

### 🧠 Claim Chain v2 本体论 — 8 种 Atom + 11 种 Edge

FLUXturbo 实现了 Flux-Insight Claim Chain v2 的完整类型系统。这不是简单的 NER + 关系抽取，而是一个**有约束的知识本体**：每种 atom 和 edge 都有明确的语义定义和类型兼容性规则。

#### Atom 类型（知识节点）

```
┌──────────────────────────────────────────────────────────────┐
│  method       算法/模型/架构         如 "ReMAC", "MAPPO"       │
│  bottleneck   问题/挑战/局限         如 "credit_assignment"    │
│  paper        论文/技术报告          如 "Yu et al. 2022"       │
│  fact         实验事实/数值结果      如 "achieves 93.2%"       │
│  component    模块/子组件            如 "centralized critic"   │
│  hypothesis   假设/主张              如 "sharing improves..."  │
│  experiment   实验设置/评估协议      如 "ablation on 3 agents" │
│  verification 验证/证明/证据         如 "convergence proof"    │
└──────────────────────────────────────────────────────────────┘
```

#### Edge 类型（关系边）

| 类型 | 强度 | Rho 证据 | 语义 | 典型兼容 |
|------|------|----------|------|----------|
| **EXTENDS** | 强因果 | 必需 | A 扩展 B | method→method |
| **IMPROVES** | 强因果 | 必需 | A 改进 B | method→method, method→bottleneck |
| **REPLACES** | 强因果 | 必需 | A 替代 B | method→method, component→component |
| **ADAPTS** | 强因果 | 必需 | A 适配 B | method→method, method→bottleneck |
| **USES_COMPONENT** | 弱关联 | 否 | A 使用 B | method→component |
| **COMPARES** | 弱关联 | 否 | A 对比 B | method→method, experiment→method |
| **BACKGROUND** | 语义 | 否 | A 是 B 的背景 | paper→method |
| **IMPLEMENTS** | 语义 | 否 | A 实现 B | method→hypothesis, method→paper |
| **VALIDATES** | 语义 | 否 | A 验证 B | experiment→hypothesis, method→method |
| **BOUNDARY_OF** | 语义 | 否 | A 的边界 | fact→method, fact→experiment |
| **RELATED_TO** | 语义 | 否 | 通用关联 | *→* |

### 🧾 Rho 证据记录 — 强因果边的可追溯性

每条强因果边（EXTENDS / IMPROVES / REPLACES / ADAPTS）必须携带 Rho 证据四元组：

```
Rho = {
  bottleneck:  "指向一个 bottleneck 类型 atom（问题是什么）",
  mechanism:   "解决机制描述（怎么解决的）",
  tradeoff:    "代价/权衡（付出了什么）",
  confidence:  0.85  # 0.0–1.0
}
```

这保证了知识图谱中的每一条强声明都有**可追溯的证据链**，而非空口断言。

### 🔬 14 种 Bottleneck 分类

```
overestimation_bias      training_instability       sample_inefficiency
exploration_exploitation credit_assignment          catastrophic_forgetting
scalability              communication_overhead     non_stationarity
partial_observability    multi_objective_conflict   representational_limitation
computational_cost       generalization_gap
```

### 🛡️ OntologyGatekeeper — 4 条验证规则

| 规则 | 名称 | 说明 |
|------|------|------|
| **R1** | 引用完整性 | 所有边的端点必须存在于 atom 集合中 |
| **R2** | 时序一致性 | IMPROVES 关系的源不能早于目标 |
| **R3** | 无矛盾 | 同一对节点不能同时有 EXTENDS 和 REPLACES |
| **R4** | Rho 完整性 | 强因果边必须有完整的 Rho 证据记录 |

---

## 🏗️ 架构

```
┌──────────────────────────────────────────────────┐
│                  MCP 客户端                       │
│     Claude Code · VS Code · 自定义 Agent          │
└────────────────────┬─────────────────────────────┘
                     │  stdio (JSON-RPC)
┌────────────────────┴─────────────────────────────┐
│              cc-blueprint MCP Server              │
│                                                   │
│  ┌─────────────┐  ┌──────────────┐               │
│  │ extract_     │  │ extract_batch│               │
│  │ blueprint    │  │ (并行)       │               │
│  └──────┬───────┘  └──────┬───────┘               │
│         │                 │                       │
│         ▼                 ▼                       │
│  ┌──────────────────────────────────┐            │
│  │       BlueprintExtractor         │            │
│  │  ┌────────────────────────────┐  │            │
│  │  │  LLM 调用 (DeepSeek/GPT)    │  │            │
│  │  │  · System prompt: CC schema │  │            │
│  │  │  · User prompt: 学术文本     │  │            │
│  │  │  · Response: JSON blueprint │  │            │
│  │  └─────────────┬──────────────┘  │            │
│  │                ▼                 │            │
│  │  ┌────────────────────────────┐  │            │
│  │  │  OntologyGatekeeper        │  │            │
│  │  │  · R1 引用完整性           │  │            │
│  │  │  · R4 Rho 完整性           │  │            │
│  │  │  · 类型兼容性检查          │  │            │
│  │  └────────────────────────────┘  │            │
│  └──────────────────────────────────┘            │
│                                                   │
│  ┌──────────────┐  ┌──────────────┐              │
│  │ get_cc_      │  │ validate_    │              │
│  │ schema       │  │ blueprint    │              │
│  └──────────────┘  └──────────────┘              │
└──────────────────────────────────────────────────┘
```

> 💡 **设计原则**: 单次 LLM 调用完成全部提取（atoms + edges + Rho），Python 端只做验证不过滤。LEAP 论文的 "blueprint → DAG → verification-driven refinement" 思想：先规划后验证。

---

## 🎯 4 个 MCP 工具

### 1. `extract_blueprint` — 单段提取

从一段学术文本中提取 CC atoms + edges，返回经过 Gatekeeper 验证的 blueprint JSON。

**输入**: `{"text": "学术文本段落..."}`

**输出**: 包含 `atoms` 和 `edges` 的完整 blueprint，每个强因果边携带 `rho` 证据记录。

### 2. `extract_batch` — 并行批量提取

对多个文本段落并行调用 LLM 提取，显著加速大批量文献处理。

**输入**: `{"segments": ["段落1", "段落2", ...], "max_workers": 4}`

**输出**: 多个 blueprint 的列表，每个 blueprint 独立验证。

### 3. `get_cc_schema` — 获取本体论

返回完整的 Claim Chain v2 类型系统：8 种 atom 类型、11 种 edge 类型、14 种 bottleneck 分类、类型兼容性矩阵、强因果边列表。

**输入**: `{}`

**输出**: schema JSON（可直接用于 prompt 工程或文档生成）。

### 4. `validate_blueprint` — 验证已有图谱

将一组 atoms + edges 提交给 OntologyGatekeeper 进行完整验证（R1/R4 + 类型兼容性）。

**输入**: `{"atoms": [...], "edges": [...]}`

**输出**: `{"valid": true/false, "error_count": N, "errors": [...]}`

---

## ⚙️ 安装与配置

### 前置条件

- [x] Python 3.10+
- [x] DeepSeek API key（或其他 OpenAI 兼容 API）
- [x] MCP 客户端（Claude Code / VS Code / 自定义）

### 安装

```bash
# 1. 克隆项目
git clone https://github.com/ExuberantWitness/FLUXturbo.git
cd FLUXturbo

# 2. 安装依赖
pip install mcp openai httpx
```

### MCP 客户端配置

在项目的 `.mcp.json` 中配置：

```json
{
  "mcpServers": {
    "cc-blueprint": {
      "command": "C:\\Users\\zhang\\.conda\\envs\\FLUX\\python.exe",
      "args": ["-m", "cc_blueprint"],
      "cwd": "E:\\DATA\\vscode\\FLUXturbo",
      "env": {
        "CC_API_KEY": "sk-your-deepseek-key",
        "CC_HTTP_PROXY": "http://127.0.0.1:6789",
        "CC_LLM_MODEL": "deepseek-chat",
        "CC_LLM_BASE_URL": "https://api.deepseek.com"
      }
    }
  }
}
```

### 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `CC_API_KEY` | `$OPENAI_API_KEY` | LLM API 密钥 |
| `CC_LLM_MODEL` | `deepseek-chat` | 模型名称 |
| `CC_LLM_BASE_URL` | `https://api.deepseek.com` | API 端点 |
| `CC_HTTP_PROXY` | `$HTTP_PROXY` | HTTP 代理 |
| `CC_MAX_ATOMS` | `30` | 每段最大 atom 数 |
| `CC_MAX_EDGES` | `20` | 每段最大 edge 数 |

### 命令行直接运行

```bash
python -m cc_blueprint
# Server 启动，等待 stdio 上的 MCP 请求
```

---

## 📊 与 spaCy NER 的对比

传统 HippoRAG 管线使用 spaCy NER 提取实体，再用 LLM 提取关系。FLUXturbo 用一次 LLM 调用同时完成 atom 识别和关系提取，结果质量有质的飞跃：

| 维度 | spaCy NER + RelationExtractor | FLUXturbo (CC Blueprint) |
|------|------------------------------|--------------------------|
| **实体质量** | 单字母 (`g`, `l`)、纯数字 (`1`, `2`) | 可读名称 (`ReMAC`, `CO-PQ`) |
| **实体类型** | 通用 NER 标签 (ORG, PERSON) | 8 种 CC atom 类型，语义明确 |
| **关系类型** | 空字符串 `""` 或无意义词 | 11 种 CC edge 类型，可追溯 |
| **证据记录** | 无 | Rho 四元组 (bottleneck + mechanism + tradeoff + confidence) |
| **验证机制** | 无 | OntologyGatekeeper R1/R4 + 类型兼容性 |
| **图谱可读性** | 节点显示 `entity-xxxxx` 哈希 | 节点显示实际概念名称 + 类型 |
| **LLM 调用** | 2 次 (NER + 关系) | 1 次 (atoms + edges + Rho 同时) |

### 实际案例（3 篇 MARL 论文，24 段）

```
spaCy 路径:  节点 "g", "l", "entity-a3f2c1"  ·  边 predicate="" (空)
CC 路径:     节点 "ReMAC"(method), "credit_assignment"(bottleneck)
             边 "ReMAC" --IMPROVES--> "credit_assignment"
             Rho: {bottleneck: "credit_assignment", mechanism: "centralized critic with attention",
                   tradeoff: "communication overhead", confidence: 0.85}
```

**提取结果**: 243 atoms, 192 edges, 73 条强因果边带 Rho 证据。

---

## 📁 项目结构

```
FLUXturbo/
├── README.md                          # 项目文档（中英双语）
├── LICENSE                            # MIT
├── pyproject.toml                     # Python 项目配置
├── .mcp.json                          # MCP 客户端配置示例
├── .gitignore
│
├── cc_blueprint/                      # MCP Server 核心包
│   ├── __init__.py                    # 包标记
│   ├── __main__.py                    # 入口: python -m cc_blueprint
│   ├── server.py                      # MCP Server — 4 工具注册 + stdio transport
│   ├── extractor.py                   # BlueprintExtractor — LLM 调用 + 解析 + 验证
│   └── ontology.py                    # Claim Chain v2 本体论
│       ├── ATOM_TYPES (8)             #   类型定义
│       ├── EDGE_TYPES (11)            #   关系定义
│       ├── BOTTLENECK_CATEGORIES (14) #   Bottleneck 分类
│       ├── EDGE_COMPAT                #   类型兼容性矩阵
│       ├── STRONG_CAUSAL              #   强因果边集合
│       ├── Rho                        #   证据记录 dataclass
│       ├── Atom / Edge / Blueprint    #   数据模型
│       └── OntologyGatekeeper         #   验证器 (R1/R4 + 类型检查)
│
└── blueprint_output/                  # 实验输出（.gitignore 排除）
    ├── merged_blueprint.json          #   合并后的完整蓝图
    └── blueprint_graph.html           #   vis.js 交互式知识图谱
```

> 💡 **设计哲学**: `ontology.py` 是唯一真相源 — 所有类型定义、兼容性规则、验证逻辑都在这里。`extractor.py` 只负责 LLM 交互，`server.py` 只负责 MCP 协议适配。三者职责清晰，修改本体不影响 MCP 接口。

---

## 🔄 典型工作流

```
PDF 文献
    │
    ▼
┌────────────────┐
│  docreader     │  WeKnora PDF 解析 → Markdown → 2000-char 切分
│  (外部工具)     │
└───────┬────────┘
        │ 文本段落 (N 段)
        ▼
┌────────────────┐
│  FLUXturbo     │  MCP extract_batch → 并行 LLM 调用
│  (本 MCP)      │  · 每段: atom 识别 + edge 提取 + Rho 证据
│                │  · OntologyGatekeeper 验证
└───────┬────────┘
        │ Blueprint JSON (atoms + edges + rho)
        ▼
┌────────────────┐
│  下游应用       │
│  · HippoRAG    │  Embedding → igraph → PPR 检索
│  · Flux-Insight │  Claim Chain SQLite → Dashboard
│  · 自定义分析   │  JSON → Pandas/NetworkX 分析
└────────────────┘
```

---

## 🎨 可视化报告 (ccchain.visualize)

`ccchain.visualize` 是一个**独立的报告工具模块**，将 CoE 审计结果渲染成自包含的交互式 HTML（单一外部依赖：[vis-network](https://visjs.org/) CDN，无需本地 JS 安装）。

> 💡 **定位**：它是报告工具，**不是第 4 个 SDK 方法**（不破坏 `ingest` / `search` / `evaluate` 三接口原则）。任何持有 `CCStore` + 审计报告的场景都可调用。

### 一行调用

```python
from ccchain.visualize import build_audit_html

path = build_audit_html(store, reports, "audit_report.html")
# → 返回写入文件的绝对路径
```

### 参数

| 参数 | 类型 | 说明 |
|------|------|------|
| `store` | `CCStore` | 持有 atoms + 图边的存储；按全部 4 个层级、全部 status 查询 |
| `reports` | `list` | 每项是 `(label, audit_report)` 二元组，**或** `(label, ingest_result, audit_report)` 三元组（内部归一化，`ingest()` 的返回值可直接喂入） |
| `output_path` | `str` | 输出 `.html` 路径（自动创建父目录） |
| `title` | `str` | 可选关键字参数，HTML `<title>` 与页眉 |

`audit_report` 即 CoE 验证器返回的 dict（`cpr` / `atoms_audited` / `atoms_passed` / `atoms_failed` / `atoms_skipped` / `failures_by_check` / `per_atom`）。

### 完整示例：ingest → 审计 → 可视化

```python
import ccchain
from ccchain.core.ontology import TaskSpec
from ccchain.visualize import build_audit_html

# 1. 摄入 + 自动 CoE 审计 (I1/I2/I3/I4)
result, err = ccchain.ingest(
    segments, source_pdf="cop-q.pdf",
    task_spec=TaskSpec("safety-gym", "safety-gym-v1",
                       "return>=0.9 & 0 violations", ["CTDE"]),
)
# result["audit_report"] 含 cpr、各检查失败计数、逐 atom 裁决

# 2. 渲染交互式 HTML 报告（store 是 ingest() 初始化的单例）
build_audit_html(
    ccchain._store,
    [("cop-q.pdf", result["audit_report"])],
    "blueprint_output/audit_report.html",
)
```

**跨进程 / 离线模式**：直接从磁盘加载已有数据库，无需在同一进程跑 ingest：

```python
from ccchain.core.store import CCStore
from ccchain.visualize import build_audit_html

store = CCStore("blueprint_output/cc_base.db", "blueprint_output/")
build_audit_html(store, [("cop-q.pdf", audit_report)], "audit_report.html")
```

### 报告内容

| 元素 | 编码 |
|------|------|
| **节点填充色** | 审计 status（绿 verified / 灰 skipped / 橙 low_confidence / 红 low_reliability / 紫 demoted） |
| **节点边框色** | 规约层级（W2 红 → W3 橙 → W4 蓝 → W5 绿），两个维度同时可见 |
| **节点大小** | 层级金字塔（W2 最大 → W5 最小） |
| **边** | 按关系着色：`decomposes_into`（自上而下规约链，蓝虚线）/ `aggregates_to`（自下而上，橙实线）/ `extends`/`improves`/`compares` 等 |
| **每篇论文卡片** | CPR 数值 + status 分布条 + I1/I2/I3/I4 失败计数 |
| **点击节点** | 右侧详情面板：context、逐项 CoE 检查裁决（passed/failed/skipped + reasoning）、provenance |
| **状态过滤按钮** | 按 status 一键隐藏/显示节点子集 |

层级规约链 `W2 → W3 → W4 → W5`（decomposes_into）在图上是完整连续的纵向链，双向 O(1) 遍历由 `aggregates_to` 反向边支持。

### 运行 demo

```bash
# 用 ARIS/pdf 下的真实论文跑完整管线并生成 audit_report.html
python scripts/audit_demo_real_pdfs.py
```

> 注：demo 在无本地 LLM 服务时用确定性 mock 替换 LLM/embedding/引用 API 层，但真实 PDF 文本走完整 ingest → refine → store → reduce → audit 管线。

---

## 📋 Roadmap

### Done / 已完成

- [x] **CC v2 完整类型系统** — 8 atom types + 11 edge types + Rho 证据记录 + 14 bottleneck categories
- [x] **OntologyGatekeeper** — R1/R4 验证规则 + 类型兼容性矩阵 + 错误报告
- [x] **MCP Server 4 工具** — extract_blueprint / extract_batch / get_cc_schema / validate_blueprint
- [x] **LLM 单次调用提取** — 一次 LLM 调用同时完成 atom 识别 + edge 提取 + Rho 证据
- [x] **并行批量提取** — ThreadPoolExecutor 支持，可配置并发数
- [x] **vis.js 图谱可视化** — 8 色 CC atom 类型着色 + 搜索 + 标签切换
- [x] **LEAP 论文对齐** — blueprint → verification 两阶段设计

### Planned / 计划中

- [ ] **R2/R3 完整验证** — 时序一致性检查 + 矛盾检测
- [ ] **BGE-M3 语义嵌入** — atom 写入时自动计算 1024 维向量，支持语义去重
- [ ] **Claim Chain SQLite 存储** — 直接将 blueprint 写入 cc.db，与 Flux-Insight 互通
- [ ] **流式提取** — 大文档流式处理，逐段输出 blueprint（避免 OOM）
- [ ] **LLM 容错重试** — 指数退避重试 + 降级到弱模型
- [ ] **Grounding 精炼** — 多轮验证驱动精炼：Gatekeeper 反馈 → LLM 修正 → 再验证
- [ ] **MCP Resources** — 将 schema 暴露为 MCP resource（`cc-schema://ontology`），支持客户端轮询
- [ ] **Docker 镜像** — 一键部署，零依赖启动

---

## 🙏 Acknowledgements / 致谢

**基础设施：**
- [MCP](https://modelcontextprotocol.io/) — Model Context Protocol，AI 工具集成标准
- [DeepSeek](https://api.deepseek.com) — 高性能低成本 LLM API
- [vis.js](https://visjs.org/) — 浏览器端交互式网络可视化

**理论来源：**
- [LEAP](https://arxiv.org/abs/2606.03303) — Blueprint-driven DAG 验证精炼范式
- [Flux-Insight](https://github.com/ExuberantWitness/Flux-Insight) — Claim Chain v2 本体论 + 完整科研自动化系统
- [HippoRAG](https://github.com/OSU-NLP-Group/HippoRAG) — 海马体启发的检索增强生成

**参考项目：**
- [ARIS](https://github.com/wanshuiyin/Auto-claude-code-research-in-sleep) — MCP Server 架构范式
- [WeKnora](https://github.com/ArtisanCloud/WeKnora) — docreader PDF 解析

---

## 📖 Citation

```bibtex
@software{fluxturbo_2026,
  title  = {FLUXturbo: Flux-Insight Claim Chain v2 Extraction Engine},
  author = {FLUXturbo Contributors},
  year   = {2026},
  url    = {https://github.com/ExuberantWitness/FLUXturbo}
}
```

---

## License

MIT

---

<a name="english"></a>

## English Summary

**FLUXturbo** is the extraction engine for the Flux-Insight Claim Chain v2 ecosystem. It wraps the complete CC v2 ontology — 8 atom types, 11 edge types, 14 bottleneck categories, Rho evidence records, and OntologyGatekeeper validation — into a standalone MCP Server with 4 tools:

1. **extract_blueprint** — Single-pass LLM extraction of CC atoms + edges + Rho evidence from academic text
2. **extract_batch** — Parallel multi-segment extraction with configurable worker count
3. **get_cc_schema** — Return the full CC v2 type system (atoms, edges, compatibility matrix, bottleneck categories)
4. **validate_blueprint** — Validate existing atoms + edges against OntologyGatekeeper rules (R1/R4 + type compatibility)

**Key design decisions:**
- **Single LLM call per segment** — atoms, edges, and Rho evidence extracted simultaneously (not pipelined NER → relation)
- **Python-side validation only** — LLM outputs pass through OntologyGatekeeper; no post-hoc filtering or rewriting
- **LEAP-aligned** — "blueprint first, then verify": LLM generates, Gatekeeper checks, errors reported but not silently fixed

**Why it matters:** Traditional spaCy NER produces garbage entities (`g`, `l`, `entity-xxxxx`). FLUXturbo produces readable, typed atoms (`ReMAC`[method], `credit_assignment`[bottleneck]) with traceable evidence chains. The knowledge graph becomes a genuine reasoning substrate rather than a visualization toy.

Quick start: `pip install mcp openai httpx && python -m cc_blueprint`

**ccchain (v0.2/v0.3) layer** — the `ccchain` package adds a three-method Python SDK (`ingest` / `search` / `evaluate`) over the W2→W3→W4→W5 knowledge pyramid, a 12-type unified atom system, and a Chain-of-Evidence (CoE) integrity audit (I1 score / I2 spec / I3 reference / I4 method-code) computing a CPR (Claim Provenance Rate). Visualization is a standalone reporting module:

```python
from ccchain.visualize import build_audit_html
build_audit_html(store, reports, "audit_report.html")
# nodes: fill = CoE audit status, border = W2/W3/W4/W5 spec level
```

See the 🎨 可视化报告 section above for full usage.

See the Chinese sections above for full documentation.
