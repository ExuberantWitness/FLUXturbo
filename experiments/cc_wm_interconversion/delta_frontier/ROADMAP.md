# CC⇌WM 普适对齐律（δ-frontier）— 代码状态 + 修改路线图

> 同步到 `ExuberantWitness/FLUXturbo` 的 `experiment/cc-wm-interconversion` 分支。
> 状态：Phase A+B 完成（δ 仪表 + 三 regime + 前沿律，合成与真实 WM 均成立）。下一步：Phase C 普适扫掠 / Phase E 理论推导 / 论文初稿。

---

## 1. 当前代码状态

### 1.1 三个包（本地 `E:/DATA/vscode/`）
- **`cc_wm_demo/`** — v0 机制 pilot：轻量 LeWM 加载、probe→CC、aux-loss 往返。
- **`cc_wm_research/`** — SOTA 真双向：`cc_wm.py`(extract/compile/roundtrip 正式接口，compile 读 CC)、`das.py`(DAS 因果)、`semantic_compile.py`(LoRA)、`fidelity_sota.py`(MI-gap+Wasserstein)。**真双向闭环跑通**。
- **`delta_frontier/`** — Nature 计划工程化（本轮重点）：
  ```
  wms/{base,lewmdm,synthetic,noisy}.py   # WorldModel protocol + le-wm/合成/噪声包装适配器
  delta/{probe,extract,intervene,delta}.py   # δ 测量（probe←extract.py；pysindy-on-PCA；IIA+null集+dose-response；统一δ+反平凡+一致性）
  frontier/fit.py                         # (δ,g,c) 前沿拟合 + 硬界检测
  run/{run_delta_calib,run_exactness,run_three_regimes,run_frontier,run_frontier_lewm}.py
  ```
  - 90% 复用 cc_wm_demo/cc_wm_research/ccchain；已装 `pyvene`+`pysindy`。
  - 路径参数化（`paths.py`，env-overridable，无硬编码）。

### 1.2 Phase A+B 实测结果（全部跑通）
| 阶段 | 结果 |
|---|---|
| **A** δ 仪表（le-wm reacher） | δ 因果分离：qpos δ≈0.01、qvel δ≈0.71；null=−0.88（强反平凡）；一致性 ρ(δ,1−R²)=0.87。gate PASS。 |
| **B** 三 regime | C1 exact δ=0（合成 identity）/ C2 lossy δ≈0.06（state+noise）/ C3 unobservable δ_min=0.74（le-wm 速度，架构性不可观测）。exactness 三段闭合。 |
| **B.2** 前沿律 | **合成** G(δ)=1.02·exp(−1.49δ) + **真实 le-wm** G(δ)=1.42·exp(−2.18δ)；两者 ρ(g,δ)=ρ(c,δ)=−1.0（g≤G(δ) 单调 + 低δ高g必高代价硬界）。 |

**产出文件**：`output/{delta_calib_lewm_reacher, exactness_c1, three_regimes, frontier_fit, frontier_lewm}.json`。

---

## 2. 修改路线图（三方向）

### 2.1 Phase C — 普适扫掠（最重，决定 Nature 成色）🔥主战场
**目标**：证明同一 G(δ)=a·exp(−bδ) 形式跨 **≥3 模型族 + ≥2 模态**成立。
**电池**（需 clone+适配，复用 `wms/base.py` protocol）：
- **TD-MPC2**（`nicklashansen/tdmpc2`，HF 有预训练 ckpt，decoder-free，控制域）。
- **DreamerV3**（`danijar/dreamerv3`，RSSM，生成式，控制域另一族）。
- **Othello-GPT**（`likenneth/othello_world`，**序列/语言模态**——跨模态最有价值，board-state 可探针）。
- （可选 Newton 灰箱 δ=0 锚点，warp env）。

**步骤**：每个族写一个 `wms/<name>.py` 适配器（load/encode/probe_targets/cost）→ `run/run_universality.py` 批跑 δ–frontier → 跨系统报 G(δ) 的 (a,b) + Spearman ρ + 系数一致性。
**go/no-go 门**：跨族/跨模态 G(δ) 形式一致（指数）+ ρ 显著 + 系数同量级。le-wm 一家付不起"普适"这账。
**风险**：3 族装/适配踩坑；单 session 不一定走完。Othello-GPT 跨模态是最高价值但最难。

### 2.2 Phase E — 理论推导（可行 now，提升成色）📐性价比最高
**目标**：从**信息论/率失真**推导出 G(δ)=a·exp(−bδ) 的形式，把"经验指数拟合"升级为"导出律"。
**思路**：
- 可信度 g ∝ I(z;q)（z 对任务量 q 的互信息，"保证度"上界）。
- δ（对齐缺陷）↔ 编码噪声/SNR 衰减。
- 在 Gaussian/IB 框架下：I(z;q) ∝ −log(1−…) ∝ SNR，SNR 随噪声指数衰减 ⇒ g ∝ exp(−b·δ)。
- 推出 a（上界，≤1）、b（系统/任务相关常数）的物理意义；给 DPI/RD 下界。
**产出**：`frontier/theory.py`（形式化命题 + 证明骨架）+ 论文 Method 节。
**价值**：理论+实证闭环 = 会议级→Nature-family 的关键。无新算力。

### 2.3 论文初稿 ✍️
**目标**：把 Phase A+B（+E）写成 Intro+Method+Results 正文 + 图。
**结构**：
1. Intro（vibe research 需仿真+写作 → 缺共性基底 → δ-frontier 律）。
2. Related（bisimulation/identifiability/neuro-symbolic + 区分：均单向，本工作双向往返+δ 可判定）。
3. Method（δ=IIA 定义 + exactness 定理三 regime + 前沿律 G(δ) + 防平凡）。
4. Results（δ 表 / 三 regime / 双前沿律合成+真实 / 防平凡对照）。
5. Phase C 普适确认（占位，待数据）+ Limitations。
**图**：① δ 因果验证（IIA vs null）；② 三 regime；③ 前沿律 G(δ)（合成+真实叠加）；④ 防平凡。

---

## 3. 诚实评估（Nature 成色）
- **当前（合成+le-wm）**：前沿律经验成立——是"律存在的证据"，**但不是"普适律"**（le-wm 一族 + 合成不足以叫普适）。
- **Phase C 后**：若 G(δ) 指数形式跨 ≥3 族+2 模态一致 → **Nature MI / Communications 可冲**；主刊 Nature 需 δ 既普适又能预言 arresting 外部现象（尾部）。
- **Phase E 后**：理论推导把指数形式从拟合升为原理 → 显著抬升。
- **最大风险**：被判"bisimulation/identifiability 换皮" → 必须用 Rho 证据不变量 + 双向往返保真 + 可判定 δ + 理论推导作差异化。
- **推荐顺序**：E（理论，now）→ C（普适，主战场）→ 论文。E 为 C 提供理论预言（"指数形式应跨族不变"）。

---

## 4. 运行（复现）
```powershell
$env:PYTHONIOENCODING="utf-8"
# venv: cc_wm_demo/.venv (cu121 torch2.5.1 + dm_control + pyvene + pysindy)
python delta_frontier/run/run_delta_calib.py       # Phase A gate
python delta_frontier/run/run_three_regimes.py     # Phase B 三 regime
python delta_frontier/run/run_frontier.py          # 合成前沿律
python delta_frontier/run/run_frontier_lewm.py     # 真实 le-wm 前沿律
```
