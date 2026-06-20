# cc_wm_demo — CC⇌WM 互逆转化的可运行实证（le-wm reacher × FLUXturbo）

把"物理知觉世界模型（LeWorldModel, JEPA）⇌ 认知性 claim chain（FLUXturbo）"的**互逆往返**在一个真实 JEPA 世界模型上跑通：

```
reacher pixels → WM.encode → 潜空间 z
   WM→CC 抽取（知觉→认知）：线性探针 z→物理量 → numerical/concept/bottleneck/method/component atoms
   CC→WM 编译回（认知→知觉）：把"z 编码不了 qvel"的瓶颈编成辅助损失微调 → WM'
   三层往返保真度（信息/行为/逻辑）+ CC vs CC' 瓶颈消解
```

**实证结论（本地 reacher，诚实结果）**：
1. **WM→CC 忠实捕获知觉结构**：LeWM 单帧潜空间**线性编码关节位置 qpos（R²≈0.99）却不编码速度 qvel（R²≈0.09）**；CC 将后者记为 `representational_limitation` 瓶颈（验证 0 error）。
2. **CC→WM 编译回 + 保真度无损**：注入 `z→qpos`(保真)+`z→qvel`(修复) 辅助损失微调 → WM' 的 **JEPA 预测 MSE 完全保持（0.21606→0.21606）、qpos 编码保持并微升（0.990→0.994）**。
3. **核心机理发现**：速度瓶颈经单帧编译回**未消解**（qvel R²≈0.08 持平），且**双帧线性探针亦失败（R²=−0.64）** → 确证这是**单帧部分可观测性本质极限**（速度不在单帧图像中），而非容量不足——**正是申请书"非定常记忆/时序世界模型"主线的实验动机**。

> 即：本 demo 证明了 CC⇌WM 互逆**机制在真实 JEPA 世界模型上端到端可跑通**（抽取/编译回/三层保真度全部工作），并实证暴露了"单帧知觉无法编码速度"这一关键局限——为申请书从单帧表征走向**时序记忆世界模型**提供了直接的实验依据。

## 目录结构
- `wm.py` — 轻量加载 LeWM（HuggingFace ViT-Tiny + le-wm 的 jepa.py/module.py），`encode()`，读 reacher.h5。
- `gen_data.py` — 本地生成 DMC reacher 数据（避免 23GB 的 HF 全量下载）。
- `extract.py` — WM→CC 分析：线性探针（R² + selectivity 控制任务）、PCA 组件、潜动力学拟合。
- `cc.py` — 构造 ccchain `Atom/Edge/Rho` → `gatekeeper.validate` → `CCStore` → `build_audit_html`。
- `compile.py` — CC→WM 编译回：辅助损失（`z→qpos` 保真 + `z→qvel` 修复瓶颈）短微调 → WM'。
- `fidelity.py` — 三层往返保真度（信息 cosine(R²) / 行为 pred-MSE / 逻辑 约束满足）+ 单向留出 + CC↔CC' diff。
- `run_demo.py` — 编排全流程。

## 安装

```bash
# 1) venv（Python 3.10）
C:/Users/zhang/.conda/envs/FLUX/python.exe -m venv .venv
VPY=.venv/Scripts/python.exe

# 2) 依赖（cu121，匹配 RTX 2060 / driver 561）
$VPY -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
$VPY -m pip install "transformers==4.53.2" h5py scikit-learn scipy igraph einops pillow huggingface_hub openai
$VPY -m pip install dm_control "mujoco==3.8.1"   # 本地生成 reacher 数据

# 3) LeWM reacher checkpoint（weights.pt + config.json）
export HTTP_PROXY=http://127.0.0.1:6789 HTTPS_PROXY=http://127.0.0.1:6789   # 国内代理
$VPY -c "from huggingface_hub import snapshot_download as s; s('quentinll/lewm-reacher', local_dir='data/lewm-reacher')"

# 4) 生成数据（~3000 帧，DMC reacher，本地渲染）
$VPY gen_data.py
```

> 注：HF 全量 `reacher.tar.zst` 为 **23.7GB**，故用 `gen_data.py` 本地生成等价的小样本（DMC reacher easy，与 LeWM 训练环境一致）。
> 注：`transformers` 必须 4.x（5.x 重命名了 ViT 键，与 checkpoint 不兼容）；`mujoco` 必须 3.8.1（dm_control 1.0.41 的字段匹配版本）。

## 运行

```bash
export PYTHONIOENCODING=utf-8   # Windows GBK 控制台打印 R²/→ 需要
$VPY run_demo.py --frames 3000 --epochs 3
# 仅单方向（WM→CC 出图）：$VPY run_demo.py --skip-roundtrip
```

## 产物（`output/`）
- `extracted_cc.html` — WM→CC 的 claim chain 图（reacher WM 知道/不知道什么；点节点看 provenance/R²）。
- `extracted_cc_prime.html` — WM'→CC（编译回之后，瓶颈消解）。
- `roundtrip_report.md` — 三层保真度（cosine(R²)、pred-MSE、约束满足）+ CC↔CC' 瓶颈消解。
- `compiled_wm.pt` — WM' checkpoint。
- `*.json` — 原子/边/报告的机读版。

## 与申请书 CC⇌WM 互逆方案的对应
| 方案环节 | 本 demo 实现 |
|---|---|
| WM→CC 抽取（五段式） | 线性探针→numerical/concept；PCA→component；潜动力学拟合→method；surprise(可扩展)→boundary |
| CC→WM 编译回 | 瓶颈 atom → 辅助 `z→qvel` 损失；保真 fact → 辅助 `z→qpos` 损失 |
| 互逆保真性（三层） | 信息 cosine(R²)/行为 pred-MSE/逻辑 约束满足 + 单向留出 |
| 物理实验腿 | DMC reacher 即证伪证据源（编译回前后探测对比） |
