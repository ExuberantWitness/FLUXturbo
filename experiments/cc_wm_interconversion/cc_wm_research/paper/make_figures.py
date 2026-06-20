"""make_figures.py — generate the paper figures from tech_eval_results.json."""
from __future__ import annotations
import json, sys
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "output"
FIG = Path(__file__).resolve().parent
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = json.loads((OUT / "tech_eval_results.json").read_text(encoding="utf-8"))


def fig_das():
    das = R["extraction"]["das"]
    probes = {p["name"]: p["r2"] for p in R["extraction"]["probes"]}
    names = [n for n in das if n in probes]
    r2 = [probes[n] for n in names]
    iia = [das[n].get("iia", 0) for n in names]
    null = [das[n].get("iia_null", 0) for n in names]
    x = range(len(names))
    w = 0.27
    plt.figure(figsize=(9, 4))
    plt.bar([i - w for i in x], r2, w, label="probe R²")
    plt.bar(x, iia, w, label="DAS IIA (causal)")
    plt.bar([i + w for i in x], null, w, label="DAS null control")
    plt.xticks(list(x), names, rotation=45, ha="right", fontsize=8)
    plt.ylabel("score"); plt.ylim(-0.3, 1.05); plt.axhline(0, color="k", lw=0.5)
    plt.title("WM→CC causal verification: IIA ≫ null ⇒ causally encoded")
    plt.legend(fontsize=8); plt.tight_layout()
    plt.savefig(FIG / "fig_das.png", dpi=130); plt.close()


def fig_fidelity():
    rt = R["roundtrip"]; bp = rt["behavioral_pred_mse"]
    labels = ["cosine(R²)\n[info]", "pred-MSE Δ\n[behavior]", "MI-gap ΔI", "W₁ latent"]
    vals = [rt["info_layer_cosine_R2"], bp["delta"], rt["mi_gap_dI"], rt["latent_wasserstein_W1"]]
    cols = ["#2ca02c", "#1f77b4", "#ff7f0e", "#9467bd"]
    plt.figure(figsize=(6, 4))
    plt.bar(labels, vals, color=cols)
    plt.axhline(0, color="k", lw=0.5)
    plt.title("Round-trip fidelity (WM→CC→WM')  —  small ΔI & W₁ ⇒ preserved")
    plt.ylabel("metric value"); plt.tight_layout()
    plt.savefig(FIG / "fig_fidelity.png", dpi=130); plt.close()


def fig_compile():
    ct = R["controlled"]
    plt.figure(figsize=(6.5, 4))
    grp = [f"compilable:\n{ct['compilable_target']}", f"uncompilable:\n{ct['uncompilable_target']}"]
    vals = [ct["compilable_auxR2"], ct["uncompilable_auxR2"]]
    cols = ["#2ca02c", "#d62728"]
    plt.bar(grp, vals, color=cols)
    plt.axhline(0, color="k", lw=0.6)
    plt.ylabel("compile-back aux-R²  (1 − aux MSE)")
    plt.title("Controlled compile-back: compilable decodable (R²>0), uncompilable not (R²<0)")
    plt.tight_layout()
    plt.savefig(FIG / "fig_compile.png", dpi=130); plt.close()


if __name__ == "__main__":
    for f in (fig_das, fig_fidelity, fig_compile):
        try:
            f(); print("wrote", f.__name__)
        except Exception as e:
            print("skip", f.__name__, e)
