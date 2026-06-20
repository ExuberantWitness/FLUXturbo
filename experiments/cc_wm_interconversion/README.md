# Experiment Feature: CC ⇌ WM Interconversion

**Claim-Chain ⇌ World-Model bidirectional interconversion** — connecting FLUXturbo's
symbolic claim chain (CC, the `ccchain` package) to a JEPA world model (WM, e.g.
LeWorldModel) as the common substrate bridging *simulation* (WM) and *symbolic cognition /
writing* (CC).

> Status: **research prototype / experiment_feature**. Demonstrated end-to-end on
> LeWorldModel (DMC reacher). Not part of the core `cc_blueprint` MCP API.

## What it does

A formal interconversion interface ([`cc_wm_research/cc_wm.py`](cc_wm_research/cc_wm.py)):

```
extract(wm, data)      → cc     # WM→CC: causal probing (R²→selectivity→DAS/IIA→SINDy) → claim chain
compile(cc, wm, data)  → wm'    # CC→WM: READS the cc graph (numerical=preserve, bottleneck=fix)
                                #         → decode-head constraints → LoRA finetune
roundtrip(wm, data)    → report # WM→CC→WM'(from CC)→CC' + 3-layer + MI-gap + Wasserstein fidelity
```

The compile step **genuinely consumes the extracted CC** (parses its atoms into constraints),
closing a true bidirectional loop.

## Key result (LeWM reacher)

- **WM→CC** (causal): 9/10 quantities pass DAS interchange-intervention (IIA ≫ null);
  latent dynamics linear R²=0.97; 19-atom CC, 0 ontology-validation errors.
- **CC→WM** (CC-driven): preserve targets (qpos/position) → aux-R²≈0.9 (encoding kept);
  fix targets (qvel/velocity) → aux-R²≈0 → **architecturally uncompilable** (velocity is
  unobservable from a single frame — a partial-observability limit, not a capacity gap).
- **Round-trip fidelity**: cosine(R²)=1.0, pred-MSE Δ=0, MI-gap ΔI<0 (info gained),
  latent Wasserstein W₁≈0.39.

See [`cc_wm_research/output/`](cc_wm_research/output/) for reports/figures and
[`cc_wm_research/paper/`](cc_wm_research/paper/) for the paper skeleton + figures.

## Layout

- `cc_wm_demo/` — v0 mechanism: light-path LeWM loader, probe→CC, aux-loss round-trip
  (the pilot that proved the mechanism; see its README).
- `cc_wm_research/` — SOTA bidirectional upgrade: DAS causal extraction, LoRA-fidelity
  compile-from-CC, MI-gap/bisimulation fidelity, closed-loop runner, paper skeleton.
  - `cc_wm.py` — the formal `extract`/`compile`/`roundtrip` interface.
  - `das.py` — DAS interchange-intervention (causal verification).
  - `semantic_compile.py` — LoRA projector + multi-target decode-head compile.
  - `fidelity_sota.py` — MI-gap (log-det) + latent Wasserstein.

## Run (prototype; paths assume the sibling repos on this machine)

```powershell
$env:PYTHONIOENCODING="utf-8"
# venv with torch(cu121)+transformers(4.x)+dm_control+mujoco==3.8.1+ccchain(openai)
python cc_wm_research/run_closed_loop.py        # TRUE bidirectional closed loop
python cc_wm_research/run_tech_eval.py          # 4 make-or-break experiments + figures
```

Data/weights are **not** committed (large): generate reacher data locally via
`cc_wm_demo/gen_data.py` (dm_control) and download the LeWM checkpoint from
`quentinll/lewm-reacher` (HuggingFace). See `cc_wm_demo/README.md`.

## Relation to FLUXturbo

FLUXturbo's `ccchain` provides the claim-chain ontology (atoms/edges/Rho/CoE) used as the
**CC side**. This experiment adds the **WM side** and the bidirectional bridge — positioning
CC⇌WM as the common key technology for future vibe-research (autonomous modeling /
control design) that needs both simulation and claim-chain-supported writing.
