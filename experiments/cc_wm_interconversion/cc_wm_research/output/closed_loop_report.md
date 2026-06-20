# TRUE bidirectional CC⇌WM (closed loop, LeWM reacher)

## WM→CC (extraction produced a real claim chain)
- CC atoms: 19
- preserve (encoded, from numerical atoms): ['observation[0]', 'observation[1]', 'observation[2]', 'observation[3]', 'qpos[0]', 'qpos[1]']
- fix (bottleneck, from bottleneck atoms): ['observation[4]', 'observation[5]', 'qvel[0]', 'qvel[1]']

## CC→WM (compiled FROM the CC — reads atoms → decode heads)
- per-target aux-R² after compile (high=kept/compiled, ≈0=architecturally uncompilable):
    observation[0]: 0.91
    observation[1]: 0.921
    observation[2]: 0.531
    observation[3]: 0.705
    observation[4]: 0.196
    observation[5]: -0.213
    qpos[0]: 0.932
    qpos[1]: 0.76
    qvel[0]: 0.107
    qvel[1]: -0.225

## Fix-target outcome (does compile-back resolve the CC-identified bottleneck?)
    observation[4]: R² 0.077 → 0.08  [still bottlenecked (architectural — unobservable)]
    observation[5]: R² 0.005 → 0.004  [still bottlenecked (architectural — unobservable)]
    qvel[0]: R² 0.204 → 0.205  [still bottlenecked (architectural — unobservable)]
    qvel[1]: R² -0.029 → -0.024  [still bottlenecked (architectural — unobservable)]

## Round-trip fidelity
- cosine(R²): 1.0; pred-MSE Δ: -0.22185
- MI-gap ΔI: -32.2217; latent W₁: 0.3905

→ The compile step CONSUMED the extracted CC (its atoms drove the constraints): genuine bidirectional interconversion.