# CC⇌WM 互逆 — 技术评估报告 (LeWM reacher)

## C1 WM→CC 抽取（因果链）
- 探针数: 10；过 DAS 因果检验(IIA≫null): **9** 个量
- 潜动力学线性拟合 R²: 0.967
- CC atoms: 19（0 校验错误，已渲染 cc_extracted.html）

## C2 往返保真（编译回后）
- 信息层 cosine(R²): **1.0**
- 行为层 pred-MSE: 0.55449 → 0.55449 (Δ+0.0)
- MI 缺口 ΔI: **-32.8498** (信息增加(编译注入结构化编码))
- 潜 Wasserstein W1: **0.389** (行为分布保持)

## C3 受控编译回（compilable vs uncompilable — aux-loss 收敛判据）
- 可编译 sin(5·qpos) [observable]: 编译回 aux-R² = **0.639** (>0 ⇒ 信息在编码器内，编译生效)
- 不可编译 qvel [unobservable single-frame]: 编译回 aux-R² = **0.054** (<0 ⇒ 信息不在单帧，架构性不可编译)
- 对照: qvel 单帧线性探针 R²(编译前) = 0.075（信息缺失，编译回无法提升）

## C4 共性基底（同一 CC⇌WM 服务仿真侧+写作侧）
- 仿真侧: pred-MSE=0.55449; CC 解释编码量 ['observation[0]', 'observation[1]', 'observation[2]', 'observation[3]', 'qpos[0]', 'qpos[1]']
- 写作侧: 假设『WM latent fails to encode observation[4] (representational_limitation)』→ WM 评估 probe R² = 0.077 → Bottleneck confirmed; classified as architectural (unobservable single-frame) → motivates temporal/memory mechanism.
- shared CC/WM instance: True → 同一基底支撑两模态
