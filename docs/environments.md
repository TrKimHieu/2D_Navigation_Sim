# Environments & API

[← README](../README.md)

Contents: [Overview](#overview) · [Python quick start](#quick-start) · [`HM3D/Svg-v0`](#hm3dsvg-v0-continuous) · [`HM3D/Grid-v0`](#hm3dgrid-v0-grid) · [Wrappers](#wrappers-hm3denvenvs) · [Vector environments](#vector-environments) · [Task filters](#task-filters) · [Custom rewards and coverage](#custom-rewards-and-coverage) · [Training with PPO](#training-with-ppo) · [Evaluation](#evaluation)

## Overview

hm3denv turns each storey of a 3D building into a 2D navigation map with two Gymnasium
environments on top of it:

| | `HM3D/Svg-v0` — continuous | `HM3D/Grid-v0` — grid |
|---|---|---|
| Map | polygons in metres (SVG) | square cells, one robot body each |
| Robot | real footprint and speed limits of 9 real robots | one cell, moves in 4 directions |
| Action | `Box(-1, 1, (k,), float32)`: k = 2 (v, ω) or 3 (v, v_y, ω), multiplied by the robot's speed limits | `Discrete(4)`: up / down / left / right |
| Observation | LiDAR (real FOV and range) + goal + velocity | LiDAR in cells (no goal) |
| Speed | ~0.3–0.6 ms per step | ~0.05 ms per step |

Every task is collision-free for the robot's real footprint and solvable (an oracle agent
reaches 100 %).

- **Two environments, one API** — continuous (`Svg`) and grid, standard Gymnasium 1.x.
- **Real robots** — TurtleBot3/4, Kobuki, Jackal, LIMO, TIAGo, JetAuto Pro: footprints from
  the manufacturers' URDF, speed and LiDAR limits with their sources.
- **Exact physics for a 2D robot** — motion integrated exactly over each step, sub-steps of at
  most 2 cm / 2° (no tunnelling through thin walls), polygon–polygon collision.
- **Environment information for research** — every step reports the geodesic distance to the
  goal, progress, distance to the nearest wall, collisions and the raw reward terms, so you can
  shape rewards and measure coverage without touching the simulator.
- **Task control** — filter tasks by geodesic distance, difficulty labels, … and change the
  filter at any time, also inside vector environments (e.g. for curriculum learning).
- **Fast** — optional numba kernels with a spatial grid (cost independent of map size), an
  on-disk map cache, vector environments with several environments per process.
- **Deterministic and reproducible** — seeded episodes, versioned datasets with sha256 of every
  file, the fast paths give bit-identical results to the reference implementation.
- **Your own scenes** — build datasets from [Isaac-Scene-Builder](building-datasets.md#from-isaac-scene-builder)
  scenes; maps use the same coordinates as the USD scene in Isaac Sim.
- **Ready for servers** — headless (no display, no GPU needed) and a Docker image.

## Quick start

```python
import gymnasium as gym
import hm3denv                     # registers HM3D/Svg-v0 and HM3D/Grid-v0

env = gym.make("HM3D/Svg-v0", dataset="demo-svg", robot="turtlebot4", split="train")
obs, info = env.reset(seed=0)      # a random map of the split, then a random task on it
print(obs["lidar"].shape, obs["goal"], info["map_id"], info["geodesic_m"])

done = False
while not done:
    action = env.action_space.sample()                 # your policy here
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

print(info["success"], info["spl"], info["termination"])   # end-of-episode metrics
```

Pick an episode explicitly with `env.reset(options={"map_id": "demo-S001_s0", "task_idx": 3})`,
or fix `map_id` / `task_idx` in `gym.make`. Your own start and goal on a map:
`env.reset(options={"map_id": "demo-S001_s0", "start": [2.48, 12.36, 0.0], "goal": [4.84, 10.72]})`
(metres and radians; theta optional, default facing the goal; grid env: `[row, col]` cells).
They are checked — a start in a wall or an unreachable goal raises `ValueError` saying why — and
the episode has `task_id` -1. To open a simulation from the command line, watch it or drive it
from another program: [sim](features/simulate.md). The grid environment works the same way:

```python
env = gym.make("HM3D/Grid-v0", dataset="demo-grid", robot="jetauto_pro")
obs, info = env.reset(seed=0)
obs, reward, terminated, truncated, info = env.step(0)      # 0..3: up / down / left / right
```

Replace `demo-svg` / `demo-grid` with the name of a dataset you built (e.g. `svg-v1`) for
real experiments.

## `HM3D/Svg-v0` (continuous)

**Observation** — `Dict`:

| Key | Shape | Meaning |
|---|---|---|
| `lidar` | `(n_beams,)` | ranges in metres over the robot's real LiDAR FOV, clipped to its range |
| `goal` | `(3,)` | distance to the goal and sin / cos of its bearing in the robot frame |
| `velocity` | `(2,)` or `(3,)` | current (v, [v_y], ω) |

**Action** — `gymnasium.spaces.Box(low=-1, high=1, shape=(k,), dtype=float32)`, printed as
`Box(-1.0, 1.0, (2,), float32)`. k depends on the robot's drive ([Robots](robots.md)):

| Drive | k | Action | Robots |
|---|---|---|---|
| differential | 2 | (v, ω) | the other 8 presets |
| omnidirectional | 3 | (v, v_y, ω) | `jetauto_pro` |

The action is normalised; the environment clips it to [-1, 1] and multiplies each component by
the robot's limit: v = a₀ · v_max (forward, m/s; a negative a₀ uses |v_min|), v_y = a₁ · vy_max
(sideways, positive to the left, m/s), ω = a₋₁ · ω_max (rad/s, positive counter-clockwise).
`hm3denv.envs.DiscreteActions` gives Habitat-style discrete actions (stay, forward / backward
0.25 m, turn ±15°, strafe).

**Reward** — `reward="dense"` (default):
`progress + 10·success − 0.1·collision − 0.01` per step, where *progress* is the decrease of the
geodesic distance to the goal. `reward="sparse"`: 1 on success. Change the weights with
`reward_weights={"progress": 1.0, "success": 10.0, "collision": 0.1, "step": 0.01}` or replace
the reward entirely with the `CustomReward` wrapper (below).

**Episode end** — success when the robot centre is within `success_radius` of the goal
(checked at every sub-step); truncation after `time_factor ×` the minimum time needed to drive
the shortest path. A collision stops the robot (`on_collision="stop"`) or ends the episode
(`"terminate"`).

**Step `info`** (environment information, not robot senses):

| Key | Meaning |
|---|---|
| `goal_geodesic_m` | current shortest-path distance to the goal (m) |
| `progress_m` | its decrease during this step |
| `goal_distance_m` | straight-line distance to the goal |
| `wall_distance_m` | distance from the robot centre to the nearest wall |
| `collided`, `n_collisions` | collision this step / so far |
| `path_length`, `time`, `pose` | distance driven (m), time (s), (x, y, θ) |
| `reward_terms` | raw reward components (`progress`, `success`, `collision`, `step`) |

At the end of an episode `info` also has `success`, `spl`, `path_length`, `time`,
`n_collisions`, `termination` (`success` / `collision` / `time_limit`) and `geodesic_m`.
`nav_info=False` skips the four distance keys (slightly faster).

**Main parameters** — `dataset`, `robot`, `split` (`train` / `val` / `test`, split by *scene*
so a house never appears in two splits), `map_id`, `task_idx`, `dt=0.1`, `n_beams=72`,
`lidar_noise=0.0` (Gaussian σ in metres; it has its own random stream so turning it on does not
change which episodes are drawn), `reward`, `reward_weights`, `on_collision`, `time_limit`,
`time_factor=3.0`, `success_radius`, `task_filter`, `render_mode` (`"rgb_array"` or `"svg"`),
`nav_info=True`, `map_repeat=1`; caches: `map_cache=8` maps and `field_cache=4` goal distance
fields in memory, `disk_cache=True` ([Performance](performance.md)).

## `HM3D/Grid-v0` (grid)

Each cell is one robot body; blocked cells are obstacles, floorless and roofless (outdoor)
cells. Observation: `lidar_rays=22` ranges **in cells** (up to `lidar_range=10`). Action:
`Discrete(4)` = up, down, left, right; a blocked move leaves the agent in place. Reward 1 on
reaching the goal; truncation after `budget=200` steps. Step `info` has `goal_geodesic`
(BFS steps), `progress`, `collided` and `reward_terms`. Tasks carry difficulty labels
(`level`: easy / medium / hard / starved, `d_bfs`, `p0`, …); `difficulty="easy"` restricts
random tasks to one level.

**The grid observation does not contain the goal**: the agent only sees its LiDAR and has to
find the goal. If your agent should know where the goal is, add it with a wrapper, for example
the offset to the goal in cells:

```python
import gymnasium as gym
import numpy as np

class GridGoal(gym.ObservationWrapper):
    """LiDAR + (rows, cols) from the agent to the goal."""
    def __init__(self, env):
        super().__init__(env)
        n = env.observation_space.shape[0]
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (n + 2,), np.float32)

    def observation(self, obs):
        u = self.env.unwrapped
        return np.concatenate([obs, np.subtract(u.goal, u.agent_pos, dtype=np.float32)])

env = GridGoal(gym.make("HM3D/Grid-v0", dataset="demo-grid"))
```

## Wrappers (`hm3denv.envs`)

| Wrapper | Purpose |
|---|---|
| `DiscreteActions(env)` | discrete macro actions for the continuous env |
| `CustomReward(env, fn)` | `reward = fn(info)`; the original reward stays in `info["env_reward"]` |
| `Coverage(env, res=0.1)` | `info["coverage"]`: share of the free area seen by the LiDAR so far |
| `EpisodeRecorder(env, path)` | one JSON line per finished episode (optionally the trajectory) |
| `OracleFollower`, `GridOracle` | shortest-path agents (baselines, sanity checks) |

## Vector environments

```python
import hm3denv

if __name__ == "__main__":                   # required with worker processes on Windows / macOS
    venv = hm3denv.make_vec("HM3D/Svg-v0", 16, dataset="svg-v1", robot="turtlebot4",
                            split="train", map_repeat=4)
    obs, info = venv.reset(seed=0)           # environment i is seeded with 0 + i
```

- One worker process per environment by default (Gymnasium `AsyncVectorEnv`); OMP / MKL /
  OpenBLAS are limited to one thread per worker.
- `envs_per_worker=k` runs k environments per process. It only helps when you run **many more
  environments than physical CPU cores**; choose k ≈ environments / cores.
- `map_repeat=K` keeps each drawn map for K consecutive episodes (fewer map loads). It makes
  consecutive episodes of one environment correlated — report it with your results.
- `autoreset_mode=AutoresetMode.SAME_STEP` (from `gymnasium.vector`) resets a finished
  environment inside the same `step()` and puts the last observation / info of its episode in
  `info["final_obs"]` / `info["final_info"]`, as most hand-written training loops expect; the
  default `NEXT_STEP` (Gymnasium's) resets it at the next `step()`, ignoring that action.
- `hm3denv.env_fn(env_id, **kwargs)` is a picklable factory for vector-env implementations
  of other libraries that start worker processes.

## Task filters

```python
env.unwrapped.set_task_filter(geodesic_m=(1.0, 5.0))            # ranges, both ends included
env.unwrapped.set_task_filter(geodesic_m={"max": 8.0}, gdr=(None, 1.5))
env.unwrapped.set_task_filter(level=["easy", "medium"])         # grid: allowed labels
env.unwrapped.set_task_filter()                                 # no filter
venv.call("set_task_filter", geodesic_m=(1.0, 12.0))            # every env of a vector env
```

Keys are looked up in the task (`geodesic_m`, `euclidean_m`) and then in its labels (`gdr` =
geodesic / straight-line ratio, `level`, `d_bfs`, …). Only maps with at least one matching task
are drawn; the call returns how many maps and tasks are left. Changing the filter over time
gives a curriculum. `task_filter={...}` in `gym.make` sets it from the start.

## Custom rewards and coverage

```python
from hm3denv.envs import CustomReward, Coverage

env = CustomReward(env, lambda i: i["reward_terms"]["progress"]
                   + 10 * i["reward_terms"]["success"]
                   - 0.05 * (i["wall_distance_m"] < 0.3))       # keep away from walls
env = Coverage(env, res=0.1)                                    # info["coverage"] in [0, 1]
```

## Training with PPO

`.\train.bat` (Linux/macOS `./train.sh`, [guide](features/train.md)) installs PyTorch (CPU) and
Stable-Baselines3 into `.venv` once, then runs [`examples/train_ppo.py`](../examples/train_ppo.py):
PPO on `--n-envs` parallel environments (`SubprocVecEnv` with `hm3denv.sim.session.env_fns`,
which also applies the maps / fixed episodes of a `--config` session file), observations and rewards normalised
with `VecNormalize`, `MultiInputPolicy` for `HM3D/Svg-v0` and `MlpPolicy` for `HM3D/Grid-v0`.

```bash
./train.sh --dataset isb-svg-v1 --robot pal_tiago --steps 2000000 --n-envs 16
./train.sh --dataset isb-svg-v1 --robot pal_tiago --eval-only runs/isb-svg-v1_pal_tiago
```

The run folder gets `model.zip` and `vecnormalize.pkl`; at the end the policy is evaluated on
every task of the test split next to a random agent and the oracle. Besides success and SPL
the summary has `mean_progress`: the share of the shortest path covered at the end of an
episode (1 on success, negative when the agent ended farther away than it started).

Tasks are sampled with a detour (`min_gdr: 1.1`: geodesic ≥ 1.1 × straight line, the build
default and the demo setting), so a policy has to learn to avoid obstacles from its LiDAR;
turning to the goal and driving straight at it fails on every demo task. The default run
on the demo dataset (200 000 steps, ~5 min on 4 CPU cores) is a check of the pipeline: expect
0 % success and a `progress` clearly above the random agent's (one run: PPO 19 %, random
−2 %). For agents that reach goals, train on a full dataset for millions of steps. The script is short — copy it as a starting point for your
own algorithm, reward (`CustomReward`) or curriculum (`set_task_filter`). Other algorithms and
libraries, or a training loop of your own: [Use your own algorithm](features/own-algorithm.md).

## Evaluation

```bash
hm3d eval svg-v1 --robot turtlebot4 --split test --agent oracle
hm3d eval svg-v1 --robot turtlebot4 --split test --agent mypkg.policies:make --out episodes.jsonl
hm3d render svg-v1 --robot pal_tiago --map 00800-TEEsavR23oF_s1 --out episode.svg
```

`--agent module:function` calls `function(env)` at the start of every episode; it must
return `policy(obs) -> action`. Every task of every map of the split is run once, in a fixed
order, and the summary reports success rate, SPL (success weighted by path length), the
share of episodes with a collision, mean time and mean path length. From Python:
`hm3denv.evaluate.evaluate(dataset, robot, agent=..., split="test")`.
