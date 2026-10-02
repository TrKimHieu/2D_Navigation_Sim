"""Run a few episodes on the bundled demo datasets and print the results.

    python examples/quickstart.py                      # continuous env, oracle agent
    python examples/quickstart.py --agent random       # random actions
    python examples/quickstart.py --env grid           # grid env
    python examples/quickstart.py --dataset svg-v1 --robot pal_tiago --split test

Replace `policy` below with your own model to try it.
"""

import argparse

import gymnasium as gym

import hm3denv  # noqa: F401  registers HM3D/Svg-v0 and HM3D/Grid-v0
from hm3denv.envs import GridOracle, OracleFollower

DEFAULTS = {"svg": ("HM3D/Svg-v0", "demo-svg", "turtlebot4"),
            "grid": ("HM3D/Grid-v0", "demo-grid", "jetauto_pro")}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--env", choices=["svg", "grid"], default="svg")
    p.add_argument("--dataset", help="dataset name or path (default: the bundled demo)")
    p.add_argument("--robot", help="robot preset (default: one of the demo's robots)")
    p.add_argument("--split", choices=["train", "val", "test"], default="train")
    p.add_argument("--agent", choices=["oracle", "random"], default="oracle")
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    env_id, dataset, robot = DEFAULTS[a.env]
    env = gym.make(env_id, dataset=a.dataset or dataset, robot=a.robot or robot, split=a.split)
    env.action_space.seed(a.seed)
    print(f"{env_id}  dataset={a.dataset or dataset}  robot={a.robot or robot}  agent={a.agent}")
    print(f"  observation: {env.observation_space}\n  action:      {env.action_space}\n")

    n_success = 0
    for ep in range(a.episodes):
        obs, info = env.reset(seed=a.seed + ep)        # a random map of the split, then a random task
        if a.agent == "oracle":                          # shortest-path follower, rebuilt every episode
            oracle = GridOracle(env) if a.env == "grid" else OracleFollower(env)
            policy = lambda obs: oracle.act()            # noqa: E731
        else:
            policy = lambda obs: env.action_space.sample()  # noqa: E731

        done, steps = False, 0
        while not done:
            obs, reward, terminated, truncated, info = env.step(policy(obs))
            done, steps = terminated or truncated, steps + 1
        n_success += bool(info["success"])
        print(f"episode {ep}: map {info['map_id']:<14} {info['termination']:<10} "
              f"steps {steps:<4} SPL {info['spl']:.2f}")
    print(f"\nsuccess {n_success}/{a.episodes}")
    env.close()


if __name__ == "__main__":
    main()
