"""fidelity.py — 3-layer round-trip fidelity (WM vs WM') + single-direction guard + report."""
from __future__ import annotations
import numpy as np
import torch
import config as C
from wm import prepare_pixels


def _r2vec(probes):
    return np.array([p["r2"] if p["r2"] == p["r2"] else 0.0 for p in probes])


def _by_name(probes):
    return {p["name"]: p["r2"] for p in probes if p["r2"] == p["r2"]}


def _col_mean(probes, col):
    """Mean R² over all components of a column (e.g. 'qpos' → qpos[0], qpos[1])."""
    vals = [p["r2"] for p in probes if p["r2"] == p["r2"] and p["name"].startswith(col + "[")]
    return float(np.mean(vals)) if vals else float("nan")


@torch.no_grad()
def pred_mse(model, data, device: str = C.DEVICE, n_eval: int = 200, seed: int = 0) -> float:
    """JEPA next-embedding prediction MSE on held-out consecutive windows (behavioral layer)."""
    rng = np.random.default_rng(seed)
    pixels = data["pixels"]
    N = len(pixels)
    has_action = "action" in data
    idx = rng.choice(N - 4, size=min(n_eval, N - 4), replace=False)
    wins = np.stack([np.arange(i, i + 4) for i in idx])
    pix = prepare_pixels(pixels[wins].reshape(-1, *pixels.shape[1:]), device)
    info = {"pixels": pix.view(len(idx), 4, *pix.shape[1:])}
    if has_action:
        act = torch.tensor(np.nan_to_num(data["action"][wins], 0.0), dtype=torch.float32, device=device)
        info["action"] = act
    info = model.encode(info)
    emb = info["emb"]
    D = emb.shape[-1]
    act_emb = info.get("act_emb")
    pred = model.predict(emb[:, :3], act_emb[:, :3] if act_emb is not None else None)
    pred = model.pred_proj(pred.reshape(-1, D)).reshape(len(idx), 3, D)
    return float(((pred[:, -1, :] - emb[:, 3, :]) ** 2).mean().item())


def roundtrip_report(probes_WM, probes_WMp, mse_WM, mse_WMp,
                     preserve_col, fix_col, temporal_probes, out_md):
    """Assemble the 3-layer report + partial-observability finding. Returns report dict."""
    r, rp = _r2vec(probes_WM), _r2vec(probes_WMp)
    cos = float(np.dot(r, rp) / (np.linalg.norm(r) * np.linalg.norm(rp) + 1e-12))
    by_WM, by_WMp = _by_name(probes_WM), _by_name(probes_WMp)

    # logical layer — aggregate by COLUMN (mean over components)
    def _col_pair(col):
        return round(_col_mean(probes_WM, col), 3), round(_col_mean(probes_WMp, col), 3)
    p_a, p_b = _col_pair(preserve_col)
    f_a, f_b = _col_pair(fix_col)
    logical = [("preserve", preserve_col, p_a, p_b, p_b >= 0.5),
               ("fix", fix_col, f_a, f_b, f_b >= 0.3)]

    # partial-observability finding: single-frame vs temporal R² for the fix target
    sf_fix = _col_mean(probes_WM, fix_col)
    tp_names = [p["name"] for p in temporal_probes]
    tp_r2 = [p["r2"] for p in temporal_probes if p["r2"] == p["r2"]]
    temporal_fix = float(np.mean(tp_r2)) if tp_r2 else float("nan")
    po_resolvable = (not np.isnan(temporal_fix)) and temporal_fix > sf_fix + 0.2

    # CC vs CC' bottleneck diff
    low_WM = [p["name"] for p in probes_WM if p["r2"] == p["r2"] and p["r2"] < C.R2_LOW]
    low_WMp = [p["name"] for p in probes_WMp if p["r2"] == p["r2"] and p["r2"] < C.R2_LOW]
    resolved = sorted(set(low_WM) - set(low_WMp))

    rep = {
        "info_layer": {"cosine_R2_WM_WMprime": round(cos, 4),
                       "per_quantity": {k: {"WM": round(by_WM.get(k, 0), 3),
                                           "WM'": round(by_WMp.get(k, 0), 3),
                                           "delta": round(by_WMp.get(k, 0) - by_WM.get(k, 0), 3)}
                                        for k in sorted(set(by_WM) | set(by_WMp))}},
        "behavioral_layer": {"pred_mse_WM": round(mse_WM, 5), "pred_mse_WMprime": round(mse_WMp, 5),
                             "delta": round(mse_WMp - mse_WM, 5)},
        "logical_layer": [{"role": ro, "target": t, "R2_WM": a, "R2_WMprime": b, "satisfied": s}
                          for ro, t, a, b, s in logical],
        "partial_observability": {"fix_target": fix_col,
                                  "single_frame_R2": round(sf_fix, 3),
                                  "temporal_2frame_R2": round(temporal_fix, 3),
                                  "resolvable_via_temporal_context": bool(po_resolvable)},
        "cc_diff": {"bottlenecks_WM": low_WM, "bottlenecks_WMprime": low_WMp,
                    "resolved_by_compileback": resolved},
        "single_direction_note": "Held-out pred MSE reported per model (guards against round-trip masking).",
    }

    lines = ["# CC⇌WM 往返保真度报告 (reacher)", "",
             "## 信息层（probing R² 向量）", f"- cosine(WM, WM') = **{cos:.4f}**",
             "- 逐量 R² 变化（WM → WM'）：", ""]
    for k in sorted(set(by_WM) | set(by_WMp)):
        d = by_WMp.get(k, 0) - by_WM.get(k, 0)
        flag = " ↑fix" if k.startswith(fix_col + "[") and d > 0.1 else \
               (" ↓WARN" if k.startswith(preserve_col + "[") and d < -0.1 else "")
        lines.append(f"  - `{k}`: {by_WM.get(k,0):.3f} → {by_WMp.get(k,0):.3f} (Δ{d:+.3f}){flag}")
    lines += ["", "## 行为层（JEPA next-emb pred MSE，留出）",
              f"- WM = **{mse_WM:.5f}**, WM' = **{mse_WMp:.5f}** (Δ{mse_WMp-mse_WM:+.5f})",
              "- 单向留出验证（防往返掩盖）：见上行各自绝对值。", "",
              "## 逻辑层（编译约束，按列聚合）"]
    for ro, t, a, b, s in logical:
        lines.append(f"- [{ro}] `{t}`: R² {a:.3f} → {b:.3f}  {'✓' if s else '✗'}")
    lines += ["", "## 部分可观测性发现（核心机理）",
              f"- 单帧潜空间对 `{fix_col}` 的线性编码 R² = **{sf_fix:.3f}**（瓶颈）",
              f"- 双帧时序上下文 [z(t-1),z(t)] 对 `{fix_col}` 的 R² = **{temporal_fix:.3f}**",
              f"- → {'速度信息**存在于时序上下文而非单帧** → 部分可观测性瓶颈，需时序/记忆机制（呼应「非定常记忆」主线）' if po_resolvable else '时序上下文亦不足'}", "",
              "## CC vs CC'（瓶颈消解）",
              f"- WM 瓶颈(R²<{C.R2_LOW}): {low_WM}",
              f"- WM' 瓶颈: {low_WMp}",
              f"- 经单帧编译回消解: **{resolved}**（单帧不可观测 → 需时序机制，非容量问题）", ""]
    from pathlib import Path
    Path(out_md).write_text("\n".join(lines), encoding="utf-8")
    print(f"[fidelity] report → {out_md}")
    return rep
