"""gen_data.py — generate a small DMC reacher dataset locally (avoids the 23GB HF download).

LeWM was trained on DMC reacher, so we roll out `suite.load("reacher","easy")` and record
pixels + qpos + qvel + observation + action. Saves to data/reacher_local.h5.
"""
from __future__ import annotations
import argparse
import numpy as np
import h5py
from pathlib import Path
from dm_control import suite
import config as C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--render", type=int, default=100)
    args = ap.parse_args()

    env = suite.load(domain_name="reacher", task_name="easy")
    spec = env.action_spec()
    pix, qpos_l, qvel_l, obs_l, act_l, ep_idx, st_idx = [], [], [], [], [], [], []
    frame = 0
    for ep in range(args.episodes):
        ts = env.reset()
        for st in range(args.steps):
            img = env.physics.render(height=args.render, width=args.render, camera_id=0)
            qpos = np.array(env.physics.data.qpos, dtype=np.float32)
            qvel = np.array(env.physics.data.qvel, dtype=np.float32)
            obs = np.concatenate([np.asarray(ts.observation[k], dtype=np.float32)
                                  for k in ("position", "to_target", "velocity")])
            # random action in spec range
            a = np.random.uniform(spec.minimum, spec.maximum, size=spec.shape).astype(np.float32)
            pix.append(img); qpos_l.append(qpos); qvel_l.append(qvel)
            obs_l.append(obs); act_l.append(a); ep_idx.append(ep); st_idx.append(st)
            ts = env.step(a)
            frame += 1
        if (ep + 1) % 10 == 0:
            print(f"  episode {ep+1}/{args.episodes}  frames={frame}")

    out = C.DATA_DIR / "reacher_local.h5"
    out.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out, "w") as f:
        f["pixels"] = np.stack(pix)              # (N,H,W,3) uint8
        f["qpos"] = np.stack(qpos_l)
        f["qvel"] = np.stack(qvel_l)
        f["observation"] = np.stack(obs_l)       # (N,6): position,to_target,velocity
        f["action"] = np.stack(act_l)
        f["episode_idx"] = np.array(ep_idx)
        f["step_idx"] = np.array(st_idx)
    print(f"[gen] wrote {out}  frames={frame}  pixels={np.stack(pix).shape}")


if __name__ == "__main__":
    main()
