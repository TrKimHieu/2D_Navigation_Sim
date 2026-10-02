# Train an agent — `train`

[← README](../../README.md)

`train` trains a PPO agent (Stable-Baselines3) on parallel environments, saves it, and
tests it on maps it has never seen next to a random agent and the shortest-path oracle.

```bash
.\train.bat                                                    # demo data: a 5 minute check
.\train.bat --dataset isb-svg-v1 --robot pal_tiago --steps 2000000
.\train.bat --config my_sim.yaml --steps 2000000               # the session file of sim
.\train.bat --eval-only runs\isb-svg-v1_pal_tiago              # test a saved run again
```

(Linux / macOS: `./train.sh ...`.) The first run installs PyTorch (CPU) and
Stable-Baselines3 into `.venv`.

| Option | Meaning |
|---|---|
| `--config FILE` | a [session file](simulate.md#choosing-what-to-simulate): dataset, robot, maps, fixed episodes, task filter, environment arguments, `num_envs` |
| `--dataset`, `--robot` | override the file (default: `demo-svg`, `turtlebot4`) |
| `--steps N` | training steps (default 200 000) |
| `--n-envs N` | parallel environments (default: the file's `num_envs` if > 1, else up to 8) |
| `--out DIR` | run folder (default `runs/<dataset>_<robot>`) |
| `--eval-episodes N` | test episodes per map (default: every task) |
| `--seed N` | seed of the environments and of PPO |

Training uses the `train` split (or the file's `split`); the test at the end uses the `test`
split, i.e. other buildings. If the session file puts test maps in `maps` / `episodes`,
`train` warns that the test is then not on unseen maps.

The run folder gets `model.zip` (the policy) and `vecnormalize.pkl` (observation statistics,
needed to run the policy).

## What to expect

Tasks are chosen so that the straight line to the goal is blocked: the robot must learn to
avoid obstacles from its LiDAR. On the small demo data a few minutes of training reach **no
goal (0 % success) — that is expected**; look at `progress` (share of the shortest path
covered): in one run PPO reached 19 % against −2 % for the random agent. Reaching goals
takes a full dataset and millions of steps.

## Your own algorithm

[`examples/train_ppo.py`](../../examples/train_ppo.py) is short: copy it as a starting point.
The pieces it uses:

- `hm3denv.sim.session.env_fns(cfg, n)` — picklable environment factories for any vector-env
  implementation, applying the session file's maps / fixed episodes at every reset;
- `hm3denv.envs.CustomReward`, `set_task_filter` (curriculum) — see
  [Environments & API](../environments.md);
- `hm3denv.evaluate.evaluate(dataset, robot, agent=make_policy, split="test")`.
