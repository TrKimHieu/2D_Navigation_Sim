# Use your own algorithm

[← README](../../README.md)

Both environments are standard [Gymnasium](https://gymnasium.farama.org/) 1.x environments
(they pass `gymnasium.utils.env_checker.check_env`), so any algorithm or library that trains on
a Gymnasium environment works. This page says what your code receives and has to send, the
points where navigation differs from the usual benchmarks, and how to report results that
others can compare with.

Two working starting points:

- [`examples/train_custom.py`](../../examples/train_custom.py): a training loop written by hand
  (PPO in plain PyTorch, no RL library). Every piece an algorithm needs is marked with a
  numbered comment; replace the PPO block with your method.
- [`examples/train_ppo.py`](../../examples/train_ppo.py): the same with Stable-Baselines3, used by
  [`train`](train.md).

```bat
.\setup.bat --train                                   :: PyTorch (CPU) + Stable-Baselines3, once
.\.venv\Scripts\python.exe examples\train_custom.py   :: demo data, ~2 min
```

Contents: [Checklist](#checklist) · [Create the environment](#1-create-the-environment) ·
[Observation](#2-observation) · [Action](#3-action) · [Reward and episode end](#4-reward-and-episode-end) ·
[Parallel environments](#5-parallel-environments) · [Libraries](#6-libraries) ·
[Evaluate and report](#7-evaluate-and-report) · [Watch your agent](#watch-your-agent) ·
[Other languages](#other-languages-or-processes)

## Checklist

| | Do | Why |
|---|---|---|
| 1 | Train on `split="train"`, tune on `val`, report on `test` | splits are by building: test maps are never seen in training |
| 2 | Turn the `Dict` observation into what your network takes, with the **same** function at training and evaluation | many libraries need one flat vector; scales differ (metres, sin / cos, m/s) |
| 3 | Bootstrap the value on `truncated`, not on `terminated` | `truncated` = time limit (the episode was cut), `terminated` = goal reached or collision |
| 4 | Use `autoreset_mode=SAME_STEP` in a hand-written loop over a vector env | with Gymnasium's default (`NEXT_STEP`) the step after the end of an episode only resets it and ignores the action |
| 5 | Feed the policy only `obs`; use `info` for rewards and metrics | `info` holds environment information the robot cannot sense (geodesic distance, pose, ...) |
| 6 | Evaluate with `hm3denv.evaluate.evaluate` and report random and oracle next to your agent | same episodes and metrics as everyone else |
| 7 | On Windows / macOS start vector environments under `if __name__ == "__main__":` | worker processes are spawned, they re-import your script |

## 1. Create the environment

```python
import gymnasium as gym
import hm3denv                                   # registers HM3D/Svg-v0 and HM3D/Grid-v0

env = gym.make("HM3D/Svg-v0", dataset="isb-svg-v1", robot="pal_tiago", split="train")
```

- `dataset`: a name from [Datasets](../datasets.md) (`demo-svg` / `demo-grid` work right after
  installation) or a path.
- `robot`: a preset id from [Robots](../robots.md); it must be one of the dataset's robots
  (`hm3d info NAME` lists them).
- Every other argument (`dt`, `n_beams`, `lidar_noise`, `reward`, `on_collision`,
  `time_factor`, `task_filter`, ...): [Environments & API](../environments.md#hm3dsvg-v0-continuous).
  Write down the ones you change: they are part of your results.

`env.reset(seed=s)` draws a random map of the split, then a random task (start and goal) on it.

## 2. Observation

`HM3D/Svg-v0` returns a `Dict` (`n` = `n_beams`, default 72):

| Key | Shape | Unit | Range |
|---|---|---|---|
| `lidar` | `(n,)` | m | the robot's LiDAR `range_min` … `range_max` (e.g. 0.15 … 12 for TurtleBot 4) |
| `goal` | `(3,)` | m, –, – | distance to the goal (0 … ∞), sin and cos of its bearing in the robot frame |
| `velocity` | `(2,)` or `(3,)` | m/s, rad/s | current (v, [v_y], ω), within the robot's limits |

Two ways to feed it to a network:

- **Keep the `Dict`**: Stable-Baselines3 `MultiInputPolicy`, or your own network with one input
  per key.
- **One vector**: `gymnasium.wrappers.FlattenObservation(env)` concatenates the keys in
  alphabetical order (`goal`, `lidar`, `velocity`): 77 values for 72 beams and a differential
  robot.

Normalise before the network. The goal distance has no upper bound, so divide-by-the-maximum
does not work for it. Two common choices:

- fixed scales from the observation space, as `preprocess()` in
  [`train_custom.py`](../../examples/train_custom.py) does: LiDAR / `range_max`, `tanh(distance / 5)`,
  velocity / its limit. Nothing has to be saved for evaluation;
- running statistics (`VecNormalize`, `gymnasium.wrappers.NormalizeObservation`): save them with
  the model and apply them, frozen, at evaluation (`train_ppo.py` saves `vecnormalize.pkl`).

`HM3D/Grid-v0` returns a `Box` of 22 LiDAR ranges **in cells** and **does not contain the goal**;
add it with a wrapper if your agent should know it
([example](../environments.md#hm3dgrid-v0-grid)).

## 3. Action

`HM3D/Svg-v0`: `Box(-1, 1, (k,), float32)`, k = 2 `(v, ω)` for differential robots and 3
`(v, v_y, ω)` for omnidirectional ones (`jetauto_pro`). Values are clipped to [-1, 1] and
multiplied by the robot's speed limits ([details](../environments.md#hm3dsvg-v0-continuous)), so
a `tanh` output or a Gaussian sample clipped to [-1, 1] fits directly.

For discrete-action algorithms (DQN, ...):

- `hm3denv.envs.DiscreteActions(env)` gives `Discrete(5)` (stay, forward / backward 0.25 m,
  turn ±15°) or `Discrete(7)` with strafing for omnidirectional robots;
- `HM3D/Grid-v0` is `Discrete(4)`: up, down, left, right.

## 4. Reward and episode end

Default reward (`reward="dense"`) per step:

```
progress + 10 · success − 0.1 · collision − 0.01
```

*progress* is the decrease of the geodesic distance to the goal in metres: a few centimetres per
step (`v_max · dt`), small next to the success bonus of 10. Reward normalisation helps most
algorithms (`train_ppo.py` uses `VecNormalize(norm_reward=True)`). Change the weights with
`reward_weights={...}`, or compute your own reward from the raw terms:

```python
from hm3denv.envs import CustomReward

env = CustomReward(env, lambda i: i["reward_terms"]["progress"] + 10 * i["reward_terms"]["success"]
                   - 0.05 * (i["wall_distance_m"] < 0.3))       # original reward in info["env_reward"]
```

An episode ends with

- `terminated=True`: goal reached (`info["termination"] == "success"`), or a collision with
  `on_collision="terminate"`. No future reward: do not bootstrap.
- `truncated=True`: time limit, `time_factor` (3) × the minimum time to drive the shortest
  path, so it differs per task: `info["time_limit"]` (s) / `dt` steps. The episode was cut:
  bootstrap from the last observation.

Episodes can be long: TurtleBot 4 (0.31 m/s) on `demo-svg` gets a median limit of ~1 200 steps
(up to ~2 700), JetAuto Pro (0.6 m/s) on `isb-svg-v1` ~170 (up to ~1 300). With `gamma=0.99` the effective horizon is ~100 steps
(~10 s at `dt=0.1`); consider a higher `gamma` (0.995–0.999) or a larger `dt` for slow robots.

With the default `on_collision="stop"` a collision does not end the episode: the robot stops at
its last free pose and `info["collided"]` is set.

## 5. Parallel environments

```python
import hm3denv
from gymnasium.vector import AutoresetMode

if __name__ == "__main__":
    envs = hm3denv.make_vec("HM3D/Svg-v0", 8, dataset="isb-svg-v1", robot="pal_tiago",
                            split="train", autoreset_mode=AutoresetMode.SAME_STEP)
    obs, info = envs.reset(seed=0)                 # env i is seeded with 0 + i
    obs, rew, term, trunc, info = envs.step(envs.action_space.sample())
    # an env whose episode ended was reset in this step: obs is the new episode's first
    # observation, info["final_obs"][i] / info["final_info"] belong to the finished one
```

- One worker process per environment (CPU-bound, one core each); `envs_per_worker=k` when you
  run many more environments than cores. More in [Vector environments](../environments.md#vector-environments).
- Infos of a vector env are batched: `info["key"][i]`, valid where `info["_key"][i]` is true.
  `hm3denv.sim.session.unbatch_info(info, i)` gives the plain dict of environment i.
- `map_repeat=K` keeps each drawn map for K episodes (faster, but consecutive episodes are
  correlated: report it).
- Libraries with their own vector env (SB3 `SubprocVecEnv`, ...): pass them picklable
  factories, `hm3denv.env_fn("HM3D/Svg-v0", dataset=..., robot=..., split="train")`, or
  `hm3denv.sim.session.env_fns(cfg, n)` to apply the maps / fixed episodes of a
  [session file](simulate.md#choosing-what-to-simulate).
- Curriculum: `envs.call("set_task_filter", geodesic_m=(1.0, 5.0))` changes the tasks drawn by
  every environment ([Task filters](../environments.md#task-filters)).

## 6. Libraries

Stable-Baselines3, checked with PPO, SAC and DQN:

```python
import gymnasium as gym
import hm3denv
from stable_baselines3 import DQN, SAC
from hm3denv.envs import DiscreteActions

env = gym.make("HM3D/Svg-v0", dataset="demo-svg", robot="turtlebot4", split="train")
SAC("MultiInputPolicy", env).learn(100_000)                             # continuous, Dict obs
SAC("MlpPolicy", gym.wrappers.FlattenObservation(env)).learn(100_000)   # one vector
DQN("MultiInputPolicy", DiscreteActions(env)).learn(100_000)            # discrete actions
```

Any other library (CleanRL-style scripts, TorchRL, RLlib, ...) takes either a Gymnasium
environment (`gym.make(...)`) or a factory (`hm3denv.env_fn(...)`); the points above
(observation, truncation, autoreset mode) are what to check in its configuration.

## 7. Evaluate and report

`evaluate` runs **every task of every map** of a split once, in a fixed order, so two agents
evaluated on the same dataset, robot and split see exactly the same episodes:

```python
from hm3denv.evaluate import evaluate

def make(env):                       # called at the start of every episode (reset RNN state here)
    return lambda obs: my_model.act(preprocess(obs))     # policy(obs) -> action, deterministic

for agent in (make, "random", "oracle"):
    print(evaluate("isb-svg-v1", robot="pal_tiago", agent=agent, split="test"))
```

From the command line, with `make` in a module (e.g. `mypkg/policies.py`). `hm3d` looks for it in
the current folder and on `PYTHONPATH`; the `.bat` / `.sh` launchers (`sim --agent`) start in the
repository folder, so put the module there or add its folder to `PYTHONPATH`:

```bat
hm3d eval isb-svg-v1 --robot pal_tiago --split test --agent mypkg.policies:make --out episodes.jsonl
```

The summary has `success`, `spl` (success weighted by path length), `mean_progress` (share of
the shortest path covered at the end, 1 on success), `collision_episodes`, `mean_time` and
`mean_path_m`; `--out` / `out=` also writes one JSON line per episode.

Report with your numbers:

- dataset name and version, robot, split (`test`), number of episodes;
- every environment argument you changed (`dt`, `n_beams`, `lidar_noise`, `reward`,
  `reward_weights`, `on_collision`, `time_factor`, `task_filter`) and `map_repeat`;
- training steps and several seeds (mean ± std);
- the random agent and the oracle on the same split (the oracle reaches 100 %: every task is
  solvable for the robot's footprint).

## Watch your agent

- In the browser, with the same `make` as for `evaluate`:
  `.\sim.bat --dataset isb-svg-v1 --robot pal_tiago --split test --agent mypkg.policies:make --view`
  ([sim](simulate.md#watching---view)).
- As frames: `gym.make(..., render_mode="rgb_array")`, `env.render()` returns an RGB `uint8`
  image; `gymnasium.wrappers.RecordVideo(env, "videos")` writes MP4 files (needs
  `pip install moviepy`).
- One episode as an SVG drawing: `hm3d render NAME --robot ID --map MAP --out episode.svg`.

## Other languages or processes

To train from another language, or from a program that must not import hm3denv, run the
simulator as a server (`.\sim.bat my_sim.yaml --serve`, ZeroMQ, one JSON request per step):
[Driving it from your program](simulate.md#driving-it-from-your-program---serve).
`hm3denv.sim.RemoteEnv(address)` turns such a server back into a Gymnasium environment.
