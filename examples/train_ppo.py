"""Train a PPO agent (Stable-Baselines3) on an hm3denv dataset, then evaluate it on the test split.

    .\\train.bat                                        # (installs torch + SB3 once, then runs this)
    .\\train.bat --dataset isb-svg-v1 --robot pal_tiago --steps 2000000
    .\\train.bat --config my_sim.yaml                   # the session config of .\\sim.bat: maps, fixed
                                                     # episodes, task filter, env arguments
    .\\train.bat --eval-only runs/demo-svg_turtlebot4
    (Linux/macOS: ./train.sh ...; or python examples/train_ppo.py ... in the activated .venv)

Works with any dataset name (bundled demo, downloaded with ./get-data.sh, or built yourself):
continuous (Svg) datasets use a MultiInputPolicy, grid datasets an MlpPolicy. The run folder
gets model.zip (the policy) and vecnormalize.pkl (observation statistics, needed to run it).

What to expect: tasks are sampled so that the straight line to the goal is blocked, so
the robot has to learn avoidance from its LiDAR. The default run on the small demo dataset
only checks that everything works: expect 0 % success, and look at "progress" (share of the
shortest path covered) against the random agent. Reaching goals takes a full dataset and
millions of steps (--steps 2000000 or more).
"""

import argparse
import os
import pickle
from pathlib import Path

from hm3denv.evaluate import evaluate
from hm3denv.sim.config import load_sim_config
from hm3denv.sim.session import env_fns, env_setup


def load_policy(run: Path):
    """`make(env) -> policy(obs)` for hm3denv.evaluate, from a saved run folder."""
    from stable_baselines3 import PPO
    model = PPO.load(run / "model.zip", device="cpu")
    with open(run / "vecnormalize.pkl", "rb") as f:
        norm = pickle.load(f)                      # VecNormalize without its venv: normalize_obs() only

    def make(_env):
        return lambda obs: model.predict(norm.normalize_obs(obs), deterministic=True)[0]
    return make


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", help="session config (YAML file or packaged name, see .\\sim.bat): "
                                    "dataset, robot, maps, episodes, task_filter, env arguments")
    p.add_argument("--dataset", help="dataset name or path (default: the config's, else demo-svg)")
    p.add_argument("--robot", help="robot preset (default: turtlebot4 if present, else the first robot)")
    p.add_argument("--steps", type=int, default=200_000, help="training steps (default: 200k)")
    p.add_argument("--n-envs", type=int,
                   help="parallel environments (default: the config's num_envs if > 1, else min(8, CPUs))")
    p.add_argument("--out", help="run folder (default: runs/<dataset>_<robot>)")
    p.add_argument("--eval-episodes", type=int, help="test episodes per map (default: every task)")
    p.add_argument("--eval-only", metavar="RUN", help="skip training, evaluate a saved run folder")
    p.add_argument("--seed", type=int, help="default: the config's seed, else 0")
    a = p.parse_args()
    # imported after parsing, so --help works before PyTorch / Stable-Baselines3 are installed
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecNormalize

    try:
        cfg = load_sim_config(a.config, dataset=a.dataset, robot=a.robot, seed=a.seed)
        ds, robot, _, _ = env_setup(cfg)
    except (ValueError, FileNotFoundError) as e:
        raise SystemExit(f"error: {e}")
    n_envs = a.n_envs or (cfg["num_envs"] if cfg["num_envs"] > 1 else min(8, os.cpu_count() or 1))
    run = Path(a.eval_only or a.out or f"runs/{ds.name}_{robot}")
    split = cfg["split"] or "train"           # test maps stay unseen for the evaluation below
    chosen = {e.get("map") for e in cfg["episodes"] or []} | set(cfg["maps"] or [])
    seen_test = sorted(chosen & set(ds.maps(robot, "test")))
    if seen_test and not a.eval_only:
        print(f"warning: the config trains on test maps {seen_test}: the evaluation at the end is "
              "then not on unseen maps")

    if not a.eval_only:
        extra = (f", maps {cfg['maps']}" if cfg["maps"] else "") + (
            f", {len(cfg['episodes'])} fixed episode(s)" if cfg["episodes"] else "")
        print(f"training PPO on {ds.name} ({ds.env_type}, split {split}{extra}), robot {robot}, "
              f"{n_envs} envs, {a.steps} steps -> {run}")
        venv = SubprocVecEnv(env_fns(cfg, n_envs, split))
        venv = VecNormalize(VecMonitor(venv), norm_obs=True, norm_reward=True)
        policy = "MultiInputPolicy" if ds.env_type == "svg" else "MlpPolicy"
        model = PPO(policy, venv, n_steps=512, batch_size=256, seed=cfg["seed"], device="cpu", verbose=1)
        model.learn(total_timesteps=a.steps)
        run.mkdir(parents=True, exist_ok=True)
        model.save(run / "model.zip")
        venv.save(str(run / "vecnormalize.pkl"))
        venv.close()
        print(f"saved {run / 'model.zip'}")

    print("\nevaluating on the test split (maps never seen in training) ...")
    for name, agent in (("ppo", load_policy(run)), ("random", "random"), ("oracle", "oracle")):
        s = evaluate(str(ds.root), robot=robot, agent=agent, split="test", per_map=a.eval_episodes,
                     seed=cfg["seed"], **cfg["env"])
        progress = "   -  " if s["mean_progress"] is None else f"{s['mean_progress']:6.1%}"
        print(f"  {name:<7} success {s['success']:6.1%}   progress {progress}   SPL {s['spl']:.3f}   "
              f"collision episodes {s['collision_episodes']:6.1%}   ({s['episodes']} episodes)")
    print("\nprogress = share of the shortest path covered (oracle 100 %, random about 0 %).")
    if ds.name.startswith("demo"):
        print("The demo run only checks the pipeline: 0 % success is expected here. Train on a full\n"
              "dataset with more steps to reach goals, e.g. .\\train.bat --dataset isb-svg-v1 --steps 2000000")


if __name__ == "__main__":
    main()
