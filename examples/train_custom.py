"""A training loop written by hand (PPO in plain PyTorch, no RL library) on HM3D/Svg-v0.

    python examples/train_custom.py                                  # demo data, ~2 min
    python examples/train_custom.py --dataset isb-svg-v1 --robot pal_tiago --steps 2000000

A template for your own algorithm, not a tuned agent: replace the PPO part, keep the
pieces every algorithm needs, each marked with a numbered comment:

  1. parallel environments with SAME_STEP autoreset (the finished episode's last
     observation is in info["final_obs"]),
  2. the Dict observation turned into one normalised vector,
  3. bootstrapping on truncation (time limit) but not on termination (goal, collision),
  4. episode metrics from info["final_info"],
  5. evaluation on the unseen test maps with hm3denv.evaluate.

Needs PyTorch (installed by .\\train.bat / ./train.sh, or pip install torch).
Guide: docs/features/own-algorithm.md
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from gymnasium.vector import AutoresetMode

import hm3denv
from hm3denv.evaluate import evaluate


def preprocess(obs: dict, space) -> np.ndarray:
    """2. Dict observation -> float32 vector in about [-1, 1] (works on one observation or a
    batch). Fixed scales from the observation space, so evaluation needs no saved
    statistics: LiDAR / its range, goal distance squashed with tanh, velocity / its limit."""
    lidar = obs["lidar"] / space["lidar"].high
    d, bearing = obs["goal"][..., :1], obs["goal"][..., 1:]       # bearing = (sin, cos)
    vel = obs["velocity"] / space["velocity"].high
    return np.concatenate([lidar, np.tanh(d / 5.0), bearing, vel], -1).astype(np.float32)


class ActorCritic(nn.Module):
    def __init__(self, n_obs: int, n_act: int):
        super().__init__()
        def mlp(n_out):
            return nn.Sequential(nn.Linear(n_obs, 128), nn.Tanh(), nn.Linear(128, 128), nn.Tanh(),
                                 nn.Linear(128, n_out))
        self.actor, self.critic = mlp(n_act), mlp(1)
        self.log_std = nn.Parameter(torch.full((n_act,), -0.5))

    def value(self, x):
        return self.critic(x).squeeze(-1)

    def dist(self, x):
        return torch.distributions.Normal(self.actor(x), self.log_std.exp())


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="demo-svg", help="dataset name or path")
    p.add_argument("--robot", default="turtlebot4", help="robot preset id (docs/robots.md)")
    p.add_argument("--steps", type=int, default=50_000, help="environment steps (all envs together)")
    p.add_argument("--n-envs", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", help="run folder (default: runs/custom_<dataset>_<robot>)")
    p.add_argument("--eval-episodes", type=int, default=3, help="test episodes per map (0: every task)")
    a = p.parse_args()
    torch.manual_seed(a.seed)
    # gamma 0.99 = a horizon of ~100 steps; slow robots have long episodes, try 0.995-0.999
    gamma, lam, n_steps, epochs, n_minibatches, clip = 0.99, 0.95, 256, 4, 8, 0.2

    # 1. parallel environments; SAME_STEP: a finished env is reset inside the same step()
    envs = hm3denv.make_vec("HM3D/Svg-v0", a.n_envs, dataset=a.dataset, robot=a.robot,
                            split="train", autoreset_mode=AutoresetMode.SAME_STEP)
    space, n_act = envs.single_observation_space, envs.single_action_space.shape[0]
    obs, _ = envs.reset(seed=a.seed)
    x = preprocess(obs, space)
    net = ActorCritic(x.shape[1], n_act)
    opt = torch.optim.Adam(net.parameters(), lr=3e-4)

    N = a.n_envs
    buf_x = np.zeros((n_steps, N, x.shape[1]), np.float32)
    buf_a = np.zeros((n_steps, N, n_act), np.float32)
    buf_logp, buf_v, buf_r, buf_done = (np.zeros((n_steps, N), np.float32) for _ in range(4))
    ended = []                                                    # final infos of finished episodes

    for update in range(max(1, a.steps // (n_steps * N))):
        for t in range(n_steps):
            with torch.no_grad():
                xt = torch.as_tensor(x)
                d = net.dist(xt)
                act = d.sample()
                buf_logp[t], buf_v[t] = d.log_prob(act).sum(-1).numpy(), net.value(xt).numpy()
            buf_x[t], buf_a[t] = x, act.numpy()
            obs, rew, term, trunc, info = envs.step(np.clip(act.numpy(), -1, 1))
            rew = rew.astype(np.float32)
            # 3. time limit: the episode was cut, not finished -> bootstrap from its last obs
            for i in np.flatnonzero(trunc):
                xf = torch.as_tensor(preprocess(info["final_obs"][i], space))
                with torch.no_grad():
                    rew[i] += gamma * net.value(xf).item()
            # 4. metrics of the episodes that ended in this step
            if "final_info" in info:
                fi = info["final_info"]
                for i in np.flatnonzero(fi["_success"]):            # progress: share of the
                    covered = 1 - fi["goal_geodesic_m"][i] / fi["geodesic_m"][i]   # shortest path
                    ended.append({"success": fi["success"][i], "spl": fi["spl"][i],
                                  "progress": 1.0 if fi["success"][i] else covered})
            buf_r[t], buf_done[t] = rew, term | trunc
            x = preprocess(obs, space)

        # generalised advantage estimation; done = no bootstrap across the episode boundary
        with torch.no_grad():
            next_v = net.value(torch.as_tensor(x)).numpy()
        adv, last = np.zeros_like(buf_r), np.zeros(N, np.float32)
        for t in reversed(range(n_steps)):
            nv = next_v if t == n_steps - 1 else buf_v[t + 1]
            nonterminal = 1.0 - buf_done[t]
            delta = buf_r[t] + gamma * nv * nonterminal - buf_v[t]
            last = delta + gamma * lam * nonterminal * last
            adv[t] = last
        ret = adv + buf_v

        # PPO update (clipped objective) -- replace this block with your algorithm
        bx, ba = torch.as_tensor(buf_x.reshape(-1, x.shape[1])), torch.as_tensor(buf_a.reshape(-1, n_act))
        blogp, badv, bret = (torch.as_tensor(z.reshape(-1)) for z in (buf_logp, adv, ret))
        for _ in range(epochs):
            for idx in torch.randperm(len(bx)).chunk(n_minibatches):
                d = net.dist(bx[idx])
                ratio = (d.log_prob(ba[idx]).sum(-1) - blogp[idx]).exp()
                A = (badv[idx] - badv[idx].mean()) / (badv[idx].std() + 1e-8)
                loss_pi = -torch.min(ratio * A, ratio.clamp(1 - clip, 1 + clip) * A).mean()
                loss_v = 0.5 * (net.value(bx[idx]) - bret[idx]).pow(2).mean()
                opt.zero_grad()
                (loss_pi + loss_v).backward()
                nn.utils.clip_grad_norm_(net.parameters(), 0.5)
                opt.step()

        recent = ended[-100:]
        if recent:
            print(f"update {update + 1}: {(update + 1) * n_steps * N} steps, last {len(recent)} episodes: "
                  f"success {np.mean([e['success'] for e in recent]):.1%}, "
                  f"progress {np.mean([e['progress'] for e in recent]):.1%}, "
                  f"SPL {np.mean([e['spl'] for e in recent]):.3f}")
    envs.close()

    run = Path(a.out or f"runs/custom_{Path(a.dataset).name}_{a.robot}")
    run.mkdir(parents=True, exist_ok=True)
    torch.save(net.state_dict(), run / "policy.pt")
    print(f"saved {run / 'policy.pt'}")

    # 5. evaluation on the test split: make(env) is called at the start of every episode
    #    (reset the state of a recurrent policy there) and returns policy(obs) -> action
    def make(env):
        def policy(o):
            with torch.no_grad():
                mean = net.actor(torch.as_tensor(preprocess(o, space)))
            return np.clip(mean.numpy(), -1, 1)                    # deterministic: the mean
        return policy

    for name, agent in (("custom", make), ("random", "random"), ("oracle", "oracle")):
        s = evaluate(a.dataset, robot=a.robot, agent=agent, split="test",
                     per_map=a.eval_episodes or None, seed=a.seed)
        print(f"  {name:<7} success {s['success']:6.1%}   progress {s['mean_progress']:6.1%}   "
              f"SPL {s['spl']:.3f}   ({s['episodes']} episodes)")


if __name__ == "__main__":                 # required with worker processes on Windows / macOS
    main()
