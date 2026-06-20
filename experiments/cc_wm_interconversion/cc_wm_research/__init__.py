"""cc_wm_research — CC⇌WM interconversion as the common key technology (SOTA upgrade over cc_wm_demo).

Reuses the validated demo modules (wm/extract/cc/compile/fidelity) and adds:
  - WM→CC: causal extraction chain (completeness + DAS interchange intervention) + 7 faithfulness metrics
  - CC→WM: semantic-loss compile + KG-as-prior + Rho.confidence routing + fidelity stack (LoRA/EWC)
  - round-trip fidelity: + MI-gap ΔI + latent bisimulation + DPI bound
  - the "common substrate" demonstration (same CC⇌WM runs both a sim-side and a write-side task)
"""
