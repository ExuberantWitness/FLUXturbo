# CC⇌WM 普适对齐律（δ-frontier）— 进展总结报告

> 状态：Phase A 完成（δ 测量管线校准通过，gate PASS）｜ 下一步：Phase B（exactness 锚点 + 三 regime）
> 日期：2026-06-20

---

## 0. 项目脉络

从"直驱串列翼扑翼飞行器 + vibe research"出发，经多轮定位收敛到一条 Nature 级研究主线：

**CC⇌WM 互逆**（claim chain ⇌ world model）不是被审计的对象，也非某个 agent，而是连接"仿真(WM)"与"符号认知/写作(CC)"的**共性关键技术**。围绕它，发现可追求的 Nature 级贡献是：

> **可信度–保真度前沿（δ-frontier）**：对任意 world model，沿其 claim-chain 层级展开到不同保真度档位，得到一个普适的 `(对齐缺陷 δ, 可信度/保证 g, 代价 c)` 关系；δ 由因果干预误差(IIA/DAS)度量；前沿跨模型族/模态/尺度普适；δ=0 档位=构造性 100% 对齐子类。

4 个核心决策已锁定（研究计划 v0.2 §12）：① δ=IIA/DAS(pyvene)；② P=可信度–保真度前沿；③ 普适电池=跨模态多族多尺度；④ Newton=构造性 δ=0 锚点。

---

## 1. 已建成的代码资产（三层）

### 1.1 `cc_wm_demo/`（v0 机制 pilot）
轻量加载 LeWorldModel（HF ViT-Tiny + le-wm 的 jepa.py/module.py，strict load 成功），probe→CC，aux-loss 往返。证明机制可跑通；发现 reacher 单帧编码位置(R²≈0.99)不编码速度(R²≈0.09)。

### 1.2 `cc_wm_research/`（SOTA 真双向）
- `cc_wm.py` — 正式接口 `extract(wm)→cc` / `compile(cc,wm)→wm'`（**读取 CC atoms→约束**）/ `roundtrip`。
- `das.py` — DAS 交换干预（因果验证）。
- `semantic_compile.py` — LoRA(projector) + 多目标解码头 compile-back。
- `fidelity_sota.py` — MI-gap(log-det) + 潜 Wasserstein。
- **真双向闭环跑通**：CC 驱动编译（preserve 类 aux-R²≈0.9、fix 类≈0→架构性不可编译）；往返保真 cosine(R²)=1.0、pred-MSE Δ=0。

### 1.3 `delta_frontier/`（Nature 计划工程化，Phase A）
```
wms/{base,lewmdm}.py     # WorldModel protocol + le-wm 适配器（路径参数化）
delta/{probe,extract,intervene,delta}.py   # δ 测量管线
run/run_delta_calib.py   # gate runner
```
- **90% 复用** cc_wm_demo/cc_wm_research/ccchain；4 个真实缺口正在补（pyvene 级 δ、WM-protocol、前沿拟合、Newton δ=0 锚点）。
- 已装 `pyvene` + `pysindy`。

---

## 2. Phase A 结果（δ 测量 gate PASS ✅）

在 LeWorldModel reacher 上校准 δ（因果干预度量）。**干净的因果分离**：

| claim 类别 | δ_iia | IIA | null | R²_probe |
|---|---|---|---|---|
| 已编码（qpos[0,1], observation[0-3]） | **0.009–0.021** | 0.98–0.99 | −0.9~-1.7 | 0.97–0.99 |
| 未编码（qvel[0,1], observation[4,5]） | **0.71–0.91** | 0.09–0.29 | −0.25~-0.46 | ~0–0.20 |

**4 项 gate 全过**：
- δ_iia(qpos[0])=0.013 < 0.3 ✅（忠实编码）
- δ_iia(qvel[0])=0.707 > 0.5 ✅（未编码）
- null IIA=−0.885 < 0.2 ✅（**强反平凡**——随机方向无因果效应）
- 一致性 ρ(δ, 1−R²)=**0.867** ✅（δ 与编码缺口单调一致）
- 潜动力学线性 R²=0.967；复合 δ=0.33

**关键修正**：δ 公式定为 **`δ = 1 − IIA`**（研究计划 §2.2 原定义）；null 是单独的反平凡对照（极负=强反平凡），不进 δ。

---

## 3. 定位与诚实评估（Nature 成色）

- **CC⇌WM 互逆技术本身**：会议级方法贡献（ICLR/NeurIPS/TMLR），非 Nature 主刊——神经符号空间拥挤，无新原理。
- **δ-frontier 普适律**：若证出（前沿存在 + δ exactness 定理 + 硬界 + 跨系统普适）→ **Nature MI / Communications 可冲**；Nature 主刊需 δ 既普适又能预言一个 arresting 外部现象（尾部事件）。
- **最大风险**：被判"bisimulation/identifiability 换皮" → 必须用 **Rho 证据不变量 + 双向往返保真 + 可判定 δ** 作差异化。
- **关键判据**：普适性是演示出来的（无标度网络/代谢标度律先例）—— Phase C 跨系统扫掠是真门槛，le-wm 一家付不起。

---

## 4. 路线图（Phase A-E）

| Phase | 内容 | 状态 | 门 |
|---|---|---|---|
| **A** δ 装仪表 | δ 管线 + le-wm 校准 | **✅ 完成** | δ 因果分离 + 反平凡 + 一致性 |
| **B** exactness 锚点 | Newton δ=0 / C1-C3 三 regime | 进行中 | C1 δ=0、C3 δ≥δ_min>0 |
| **C** 普适扫掠 | ≥3 族+2 模态 δ–P | 待 | 跨系统 Spearman ρ 显著+系数一致 |
| **D** 组合/层级律 | ccchain W-levels + 灰箱组合体 | 待 | δ_composite≤f(δ_i) |
| **E** 写作 | 理论 headline + 普适确认 | 待 | — |

---

## 5. 下一步（Phase B 即将启动）
1. **构造性 δ=0 锚点**（C1）：合成有限/线性系统，WM=符号系统 → δ=0 by construction（快速，无需 Newton）。
2. **Newton 锚点**（warp env，cu128）：灰箱组合体 + exactness 上端 + C2（已知 ODE，δ 小）。
3. **C3 下界**：混沌系统（双摆）→ δ≥δ_min>0（可证）。
4. 三 regime 闭合 → 独立理论+受控实证论文（地基）。

---

## 6. Phase B 实测结果（三 regime 闭合 + 前沿律）— 已完成

### 6.1 三 regime 闭合（exactness 定理的三段）
| regime | 系统 | δ |
|---|---|---|
| **C1 exact (δ=0)** | 合成 identity (z≡state) | max δ = **0.000** |
| **C2 lossy (δ small)** | 合成 z=state+noise σ=0.25 | mean δ = **0.056** |
| **C3 unobservable (δ≥δ_min)** | le-wm 速度（单帧不可观测） | δ_min = **0.741**（可观测量 δ≈0.013） |

δ 跨 **[0, ~0.82]**：WM≡claim 时 δ=0；编码不完美时 δ 小；架构性不可观测时 δ≥δ_min>0（编译回无法修复）。
→ `run/run_exactness.py` + `run/run_three_regimes.py`

### 6.2 可信度–保真度前沿（HEADLINE 律）— 拟合并成立 ✅
多保真度扫描（合成 WM，σ∈[0.02..1.5]）测 (δ, g, c)，拟合：
```
G(δ) = 1.018·exp(−1.49·δ)         # 可信度随缺陷指数衰减
ρ(g, δ) = −1.0   (g ≤ G(δ) 严格单调递减)
ρ(c, δ) = −1.0   (硬界：低δ高g 必高代价 c)
```
σ=0.02→δ=0/g=1.0/c=47.6  …  σ=1.5→δ=0.68/g=0.32/c=0.67。
→ **FRONTIER LAW HOLDS**（前沿律成立）：g 单调依赖 δ，低缺陷高可信必高代价。
→ `run/run_frontier.py` + `frontier/fit.py`

### 6.3 诚实边界
- 前沿律目前在**合成受控保真度扫描**上成立——**普适性**（同一 G(δ) 形式跨 DreamerV3/TD-MPC2/Othello-GPT 等真实 WM 族）是 Phase C 的主战场，尚未证。
- G(δ) 的指数形式目前是**经验拟合**，理论推导（连到信息论/率失真）是 Phase E。

### 6.4 前沿律在真实 WM 上也成立 ✅
le-wm reacher 潜空间（NoisyWrapper 扫描 noise∈[0..3.5]）拟合：
```
le-wm:   G(δ) = 1.42·exp(−2.18·δ)    ρ(g,δ)=−1.0  ρ(c,δ)=−1.0   → REAL-WM FRONTIER LAW HOLDS
synth:   G(δ) = 1.018·exp(−1.49·δ)   ρ(g,δ)=−1.0  ρ(c,δ)=−1.0
```
**同一指数形式 g=a·exp(−b·δ)** 在合成与真实 WM 上都成立，严格单调 + 硬界。
→ `run/run_frontier_lewm.py`

---

## 7. 当前状态总览
- **Phase A**（δ 仪表）✅；**Phase B**（三 regime + 前沿律）✅。
- 已建成：δ 测量、三 regime、前沿拟合，全部在 le-wm + 合成上跑通。
- **下一步 = Phase C（普适扫掠）**：clone+适配 DreamerV3 / TD-MPC2 / Othello-GPT，跨 ≥3 族+2 模态测同一前沿形状（Spearman ρ + 系数一致性）—— 决定 Nature 成色的主战场，工作量最大。
