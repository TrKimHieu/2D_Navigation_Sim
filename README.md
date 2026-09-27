<div align="center">

# hm3denv

**A physically valid indoor robot-navigation simulator, built from real 3D houses
([HM3D](https://aihabitat.org/datasets/hm3d/)) and from your own
[Isaac Sim](https://developer.nvidia.com/isaac/sim) scenes, with a standard Gymnasium API.**

![Python](https://img.shields.io/badge/python-3.10%20|%203.11%20|%203.12-3776AB?logo=python&logoColor=white)
![Gymnasium](https://img.shields.io/badge/API-Gymnasium%201.x-0081A5)
![License](https://img.shields.io/badge/license-MIT-green)
[![CI](https://github.com/TrKimHieu/2D_Navigation_Sim/actions/workflows/ci.yml/badge.svg)](https://github.com/TrKimHieu/2D_Navigation_Sim/actions/workflows/ci.yml)

</div>

hm3denv turns each storey of a 3D building into a 2D navigation map and gives you two
[Gymnasium](https://gymnasium.farama.org/) environments on top of it:

| | `HM3D/Svg-v0` — continuous | `HM3D/Grid-v0` — grid |
|---|---|---|
| Map | polygons in metres (SVG) | square cells, one robot body each |
| Robot | real footprint (URDF section) and real speed limits of 9 robots | one cell, moves in 4 directions |
| Action | `Box[-1, 1]^k`: v, (v_y), ω scaled to the robot | `Discrete(4)`: up / down / left / right |
| Observation | LiDAR with the robot's real FOV and range + goal (distance, sin, cos) + velocity | LiDAR in cells |
| Speed | ~0.3–0.6 ms per step (numba) | ~0.05 ms per step |

Every start and goal is collision-free for the robot's real footprint and connected to the
goal (for HM3D datasets it was also checked on the real 3D mesh: floor under the robot, nothing
touching its body, indoors). An oracle agent reaches 100 % of the tasks, so every task is
solvable.

**Contents** —
[Features](#features) ·
[Installation](#installation) ·
[Getting data](#getting-data) ·
[Quick start](#quick-start) ·
[Environments](#environments) ·
[Robots](#robots) ·
[Parallel simulation & task control](#parallel-simulation--task-control) ·
[Evaluation](#evaluation) ·
[Performance](#performance) ·
[Building datasets](#building-datasets) ·
[Docker](#docker) ·
[Dataset format](#dataset-format) ·
[Development](#development) ·
[Citation & license](#citation--license)

---

## Features

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
- **Your own scenes** — build datasets from [Isaac-Scene-Builder](#from-isaac-scene-builder)
  scenes; maps use the same coordinates as the USD scene in Isaac Sim.
- **Ready for servers** — headless (no display, no GPU needed) and a Docker image.

## Installation

Python 3.10–3.12. The environments run on the CPU; a GPU is only useful for your policy.

```bash
git clone https://github.com/TrKimHieu/2D_Navigation_Sim && cd 2D_Navigation_Sim
python -m venv .venv
source .venv/bin/activate                 # Windows (PowerShell): .venv\Scripts\Activate.ps1
pip install ".[fast]"
hm3d eval demo-svg --robot turtlebot4 --agent oracle     # works right away: 100 % success
```

That is all: two small demo datasets ship with the package, so the environments run without
downloading anything (see [Getting data](#getting-data)). Commands such as `hm3d` are available
while the virtual environment is active; without one, if your shell says `hm3d` is not
recognized, use `python -m hm3denv` instead. If PowerShell refuses to run `Activate.ps1`
("running scripts is disabled"), run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once,
or use `.venv\Scripts\activate.bat` from `cmd`. Without cloning:

```bash
pip install "hm3denv[fast] @ git+https://github.com/TrKimHieu/2D_Navigation_Sim"
```

Optional extras (combine them, e.g. `hm3denv[fast,build]`):

| Extra | Adds | Needed for |
|---|---|---|
| `fast` | numba | compiled collision / LiDAR kernels (steps ~3.5× faster on HM3D maps, up to ~45× on very dense maps) — recommended |
| `hub` | huggingface_hub | downloading the pre-built datasets (`hm3d download`) |
| `build` | trimesh, pillow, xacro, pycollada, shapely | building datasets (`hm3d build`, `hm3d review`, …) |
| `test` | pytest | running the tests |

From a clone (for development):

```bash
git clone https://github.com/TrKimHieu/2D_Navigation_Sim && cd 2D_Navigation_Sim
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\Activate.ps1
pip install -e ".[fast,build,test]"
pytest -m "not workspace"
```

## Getting data

Environments read *datasets* (maps + tasks). Two small demo datasets are **bundled with the
package**, so everything works right after installation:

| Dataset | Environment | Robots | Maps (train / val / test) | Tasks |
|---|---|---|---|---|
| `demo-svg` | `HM3D/Svg-v0` | `turtlebot4`, `clearpath_jackal` | 6 scenes (3 / 2 / 1) | 10 per map and robot |
| `demo-grid` | `HM3D/Grid-v0` | `jetauto_pro` | 6 scenes (3 / 2 / 1) | 10 per map |

They were built from Isaac-Scene-Builder scenes with the same pipeline as full datasets and are
meant for trying the API, tests and debugging.

### Pre-built datasets (Hugging Face)

Full datasets are hosted on Hugging Face:
[**TranKimHieu/2D_Navigation_Sim**](https://huggingface.co/datasets/TranKimHieu/2D_Navigation_Sim).

| Dataset | Environment | Source | Maps | Robots | Tasks |
|---|---|---|---|---|---|
| `isb-svg-v1` | `HM3D/Svg-v0` | Isaac-Scene-Builder | 204 | all 9 | 36 720 |
| `isb-grid-v1` | `HM3D/Grid-v0` | Isaac-Scene-Builder | 202 | `jetauto_pro` | 4 040 |
| `svg-v1` | `HM3D/Svg-v0` | HM3D | 177 | all 9 | 30 835 |
| `grid-jetauto-v1` | `HM3D/Grid-v0` | HM3D | 157 | `jetauto_pro` | 3 140 |
| `grid-s15-v1` | `HM3D/Grid-v0` (15 cm cells) | HM3D | 166 | `s15_h63` | 3 320 |

Access is gated and reviewed by hand; the HM3D-derived datasets are only shared with people who
have been granted access to HM3D by Matterport.

1. With a (free) Hugging Face account, request access with the form on the
   [dataset page](https://huggingface.co/datasets/TranKimHieu/2D_Navigation_Sim) and wait for
   the approval.
2. Install the download support, log in once, and download by name:

```bash
pip install ".[fast,hub]"        # adds huggingface_hub and its `hf` command
hf auth login                    # once per machine
hm3d download                    # list the datasets in the repository
hm3d download isb-svg-v1 svg-v1  # -> ~/.hm3denv/datasets/ (or $HM3D_HOME/datasets)
```

Downloaded datasets are validated against their manifest and then found by name:
`gym.make("HM3D/Svg-v0", dataset="isb-svg-v1", robot="pal_tiago")`. Without access approval,
`hm3d download NAME` stops with an explanation and downloads nothing.

### Building your own

- **From HM3D** — real houses. HM3D is distributed by Matterport under its own terms of use
  (see the [official HM3D page](https://aihabitat.org/datasets/hm3d/)); request access and
  download it yourself, then [build the datasets](#from-hm3d).
- **From Isaac-Scene-Builder** — scenes you designed yourself, see
  [below](#from-isaac-scene-builder). No license restriction.
- **Shared dataset directories** — put them anywhere and point `HM3D_DATASETS` to their parent
  directory.

### Where datasets are found

By name, in this order: the directories of `HM3D_DATASETS` (separated by `;` on Windows and `:`
on Linux/macOS), `<workspace>/datasets/`, the download directory (`$HM3D_HOME/datasets`, default
`~/.hm3denv/datasets`), then the bundled demos. You can also pass a path.

```bash
hm3d datasets                      # list the datasets found
hm3d info demo-svg                 # robots, splits, map and task counts, flagged maps
hm3d datasets validate demo-svg    # check every file against the sha256 in the manifest
```

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
or fix `map_id` / `task_idx` in `gym.make`. The grid environment works the same way:

```python
env = gym.make("HM3D/Grid-v0", dataset="demo-grid", robot="jetauto_pro")
obs, info = env.reset(seed=0)
obs, reward, terminated, truncated, info = env.step(0)      # 0..3: up / down / left / right
```

Replace `demo-svg` / `demo-grid` with the name of a dataset you built (e.g. `svg-v1`) for
real experiments.

## Environments

### `HM3D/Svg-v0` (continuous)

**Observation** — `Dict`:

| Key | Shape | Meaning |
|---|---|---|
| `lidar` | `(n_beams,)` | ranges in metres over the robot's real LiDAR FOV, clipped to its range |
| `goal` | `(3,)` | distance to the goal and sin / cos of its bearing in the robot frame |
| `velocity` | `(2,)` or `(3,)` | current (v, [v_y], ω) |

**Action** — `Box[-1, 1]^k`, scaled by the robot limits: k = 2 (v, ω) for differential
drives, 3 (v, v_y, ω) for omnidirectional robots. `hm3denv.envs.DiscreteActions` gives
Habitat-style discrete actions (stay, forward / backward 0.25 m, turn ±15°, strafe).

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
`time_factor=3.0`, `success_radius`, `task_filter`, `render_mode` (`"rgb_array"` or `"svg"`).

### `HM3D/Grid-v0` (grid)

Each cell is one robot body; blocked cells are obstacles, floorless and roofless (outdoor)
cells. Observation: `lidar_rays=22` ranges **in cells** (up to `lidar_range=10`). Action:
`Discrete(4)` = up, down, left, right; a blocked move leaves the agent in place. Reward 1 on
reaching the goal; truncation after `budget=200` steps. Step `info` has `goal_geodesic`
(BFS steps), `progress`, `collided` and `reward_terms`. Tasks carry difficulty labels
(`level`: easy / medium / hard / starved, `d_bfs`, `p0`, …); `difficulty="easy"` restricts
random tasks to one level.

### Wrappers (`hm3denv.envs`)

| Wrapper | Purpose |
|---|---|
| `DiscreteActions(env)` | discrete macro actions for the continuous env |
| `CustomReward(env, fn)` | `reward = fn(info)`; the original reward stays in `info["env_reward"]` |
| `Coverage(env, res=0.1)` | `info["coverage"]`: share of the free area seen by the LiDAR so far |
| `EpisodeRecorder(env, path)` | one JSON line per finished episode (optionally the trajectory) |
| `OracleFollower`, `GridOracle` | shortest-path agents (baselines, sanity checks) |

## Robots

| Preset | Drive | v max (m/s) | ω max (rad/s) | Height (m) | LiDAR FOV / range |
|---|---|---|---|---|---|
| `agilex_limo` | differential | 1.0 | 1.0 | 0.25 | 360° / 8 m |
| `clearpath_jackal` | differential | 2.0 | 1.0 | 0.25 | 360° / 12 m |
| `jetauto_pro` | omnidirectional | 0.6 | 1.0 | 0.63 | 360° / 12 m |
| `kobuki` | differential | 0.7 | 3.14 | 0.12 | 360° / 12 m |
| `pal_tiago` | differential | 1.0 | 1.0 | 1.45 | 270° / 10 m |
| `turtlebot3_burger` | differential | 0.22 | 2.84 | 0.19 | 360° / 8 m |
| `turtlebot3_waffle_pi` | differential | 0.26 | 1.82 | 0.14 | 360° / 8 m |
| `turtlebot4` | differential | 0.31 | 1.9 | 0.35 | 360° / 12 m |
| `turtlebot4_lite` | differential | 0.31 | 1.9 | 0.19 | 360° / 12 m |

Every number in a preset (JSON) has a source: `official` with a URL, or `estimated` with the
method. Footprints of 8 robots are real sections of the manufacturers' URDF; JetAuto Pro is a
32.4 × 26 cm box. Robots are grouped into height classes (`h36`, `h63`, `h145`) that decide
which obstacles block them (a TurtleBot passes under a table, TIAGo does not).

```bash
hm3d robots                                  # list the presets
hm3d robots check my_robot.json              # validate your own preset
hm3d robots render --out robots.svg          # draw every footprint to scale
hm3d robots import-urdf turtlebot4 --dry-run # section a manufacturer URDF
```

Your own presets go into `<workspace>/robots/`; they override the packaged ones.

## Parallel simulation & task control

### Vector environments

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
- `hm3denv.env_fn(env_id, **kwargs)` is a picklable factory for vector-env implementations
  of other libraries that start worker processes.

### Task filters

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

### Custom rewards and coverage

```python
from hm3denv.envs import CustomReward, Coverage

env = CustomReward(env, lambda i: i["reward_terms"]["progress"]
                   + 10 * i["reward_terms"]["success"]
                   - 0.05 * (i["wall_distance_m"] < 0.3))       # keep away from walls
env = Coverage(env, res=0.1)                                    # info["coverage"] in [0, 1]
```

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

## Performance

| Knob | Effect |
|---|---|
| `pip install hm3denv[fast]` | numba kernels with a spatial segment grid — the cost of a step no longer depends on the map size (0.33 ms on a 53 000-segment map). Results are bit-identical to the pure-numpy path (`HM3D_ACCEL=0`). |
| disk cache (on by default) | maps load in ~15 ms instead of ~100 ms after the first time. Location: `HM3D_CACHE` (default `%LOCALAPPDATA%\hm3denv\cache` / `~/.cache/hm3denv`); `HM3D_CACHE=off` disables it. |
| `hm3d cache build svg-v1 --robot turtlebot4 --fields` | pre-builds the cache, including the distance field of every task (~3.5 MB per map). |
| `map_repeat=K` | K× fewer map loads. |
| `envs_per_worker=k` | helps with many more environments than cores. |

Measure on your machine before choosing the number of environments:

```bash
hm3d bench svg-v1 --robot turtlebot4 --num-envs 1 8 16 32
hm3d bench svg-v1 --robot turtlebot4 --num-envs 64 --envs-per-worker 4 --map-repeat 4
```

## Building datasets

Install the `build` extra first: `pip install "hm3denv[fast,build]"`.

### Workspace

```bash
hm3d init ~/hm3d-ws          # creates raw/glb, cache, stages, robots, datasets, configs
export HM3D_WORKSPACE=~/hm3d-ws   # Windows: set HM3D_WORKSPACE=... (or run hm3d inside it)
```

```
<workspace>/
  workspace.yaml
  raw/glb/<scene>.glb          HM3D scenes, e.g. raw/glb/00800-TEEsavR23oF.glb
  cache/                       mesh cache (.npz), downloaded robot models
  stages/slice/<scene>/        storey maps (+ review.json after a manual review)
  robots/                      your robot presets
  configs/                     build configurations (examples copied by `hm3d init`)
  datasets/<name>/             built datasets
```

### From HM3D

1. Download HM3D (any split) after accepting Matterport's terms, following the instructions
   on the [official HM3D page](https://aihabitat.org/datasets/hm3d/) (for example with
   habitat-sim's `datasets_download` utility).
2. Copy one GLB per scene into `raw/glb/`, named after the scene directory. Each scene
   directory `<id>-<hash>/` contains `<hash>.glb` (or `<hash>.basis.glb`); geometry is all that
   is read, textures are skipped:

   ```bash
   for d in /path/to/hm3d/val/*/; do
     id=$(basename "$d"); f=$(ls "$d"*.glb | grep -v semantic | head -1)
     cp "$f" "$HM3D_WORKSPACE/raw/glb/$id.glb"
   done
   ```

3. Build:

   ```bash
   hm3d build configs/svg_9robots.yaml        # continuous dataset, 9 robots
   hm3d build configs/grid_jetauto.yaml       # grid dataset, JetAuto Pro cells
   ```

   The pipeline slices every storey of every GLB into 2 cm maps (floor, obstacles in the
   robot's height band, roof), vectorises them (SVG) or cuts them into robot-sized cells
   (grid), samples tasks and checks every start and goal on the 3D mesh, writes previews and a
   manifest with sha256 hashes.

   ```
   GLB ──slice──► 2 cm storey maps ──(review)──► vectorize ──► SVG tasks  ──► SVG dataset
                                                 gridify   ──► grid tasks ──► grid dataset
   ```

4. Optional manual review of suspicious scenes: `hm3d review` opens a web UI at
   <http://127.0.0.1:8765> to fix the up axis, merge or split storeys, or drop a scene. The next
   `hm3d build` re-slices the reviewed scenes.

Every stage writes a stamp (hash of its parameters and inputs) into `<dataset>/.stages/`;
running `hm3d build` again only redoes what changed, and `--force tasks` re-runs a stage and
everything after it.

### From Isaac-Scene-Builder

Isaac-Scene-Builder (a grid-based scene editor for Isaac Sim) exports each scene as a USD
file plus a bird's-eye view (`output/bev/<id>/<id>.pgm|.yaml|.csv`). hm3denv builds datasets from
that output:

```yaml
# configs/isb_svg.yaml
name: isb-svg-v1
env: svg                       # or grid
source:
  type: isaac_scene_builder
  path: /path/to/Isaac-Scene-Builder/output
  prefix: isb                  # scene names become isb-<id>, maps isb-<id>_s0
scenes: all                    # the scenes of output/manifest.jsonl, or a list of ids
robots: all
```

- SVG maps are built exactly from the scene description (walls, doors, boxes, discs and the
  real footprint image of every piece of furniture, `geometry: vector`, the default);
  `geometry: raster` vectorises the BEV image instead. Grid datasets use the BEV image.
- **Map coordinates are the USD world coordinates of the scene** (Z up), so starts, goals and
  poses can be used directly in Isaac Sim. (The BEV `.yaml` follows the ROS convention, with y
  flipped; hm3denv handles it.)
- The BEV has no heights: every piece of furniture blocks every robot. There is no 3D mesh to
  check against, so the manifest records `source.verified_3d: false`.

### Configuration reference

```yaml
name: my-dataset               # dataset directory name
env: svg                       # svg | grid
source: {type: hm3d}           # or isaac_scene_builder (above)
scenes: all                    # or a list of scene ids
exclude: []                    # scenes to leave out
split: {train: 0.7, val: 0.15, test: 0.15, seed: 0}   # by scene
slice: {res: 0.02, step: 0.1, floor_margin: 0.05, ceil_margin: 0.1, min_storey_m: 1.8}
# env: svg
robots: all                    # or a list of presets
height_classes: {h36: 0.36, h63: 0.626, h145: 1.45}
map: {epsilon_px: 0.5, min_iou: 0.98, min_free_m2: 2.0}
tasks: {k: 20, min_geo: 1.0, max_geo: 30.0, min_gdr: 1.1, clearance: 0.05, success_radius: 0.2, seed: 0}
# env: grid
grid: {robot: jetauto_pro, robot_size: null, margin: 0.04, floor_cover: 0.9, roof: true}
tasks: {k: 20, d_min: 8, budget: 200, slack: 20, seed: 0}
```

Unknown keys are errors, so a typo never silently falls back to a default.

### Checks

```bash
hm3d datasets validate my-dataset             # sha256 of every file
hm3d verify my-dataset --robot turtlebot4     # re-check every task on the 3D mesh (HM3D only)
hm3d eval my-dataset --robot turtlebot4       # the oracle must reach 100 %
```

## Docker

```bash
docker build -t hm3denv .                                   # environments + CLI (numba)
docker build -t hm3denv:build --build-arg EXTRAS=fast,build .   # + dataset building tools
docker build --target test .                                # runs the test suite

docker run --rm -v /data/hm3d/datasets:/data/datasets:ro -v hm3d-cache:/cache \
    hm3denv bench svg-v1 --robot turtlebot4 --num-envs 1 8 32
docker run --rm -v /data/hm3d/datasets:/data/datasets:ro -v hm3d-cache:/cache \
    hm3denv eval svg-v1 --robot turtlebot4 --split test --agent oracle
```

Datasets and caches are mounted, never baked into the image.

## Dataset format

```
<dataset>/
  manifest.json            configuration, splits, statistics, flags, stage stamps, sha256 of every file
  robots/<id>.json         snapshot of each robot preset used
  maps/<h36|h63|h145>/     SVG maps (SVG datasets)
  grids/<map>.npz|.json    grids and metadata (grid datasets)
  tasks/<robot>/<map>.json tasks: start (x, y, θ), goal (x, y), geodesic_m, labels
  verify/<robot>.csv       3D check of every sampled point
  preview/<robot>/         preview images (sheets/page_NN.png)
```

SVG maps are plain SVG files in metres (1 m = 1 cm when printed), y up, with the free region as
an even-odd path and metadata as JSON: open them in any browser or vector editor. Coordinates in
task files are metres in the map frame; grid tasks also have `cell: [row, col]`.

## Development

```
src/hm3denv/
  core/       geometry, kinematics, planning, SVG maps, grid core, numba kernels
  envs/       SvgEnv, GridEnv, wrappers, oracles
  dataset/    schema, reading and writing datasets
  robots/     presets and their checks
  build/      slice, review UI, gridify, vectorize, tasks, verify, preview, URDF,
              Isaac-Scene-Builder import, pipeline
  cache.py    on-disk map cache       vector.py  vector environments
  bench.py    throughput              evaluate.py evaluation
  cli.py      the `hm3d` command      download.py  `hm3d download` (Hugging Face)
  demo/       bundled demo datasets (demo-svg, demo-grid)
tests/        pytest suite (synthetic data, no download needed)
```

```bash
pytest -m "not workspace"         # all tests that need no downloaded data (~2 min)
HM3D_ACCEL=0 pytest               # the pure-numpy reference path
pytest                            # + tests on a real workspace (HM3D_WORKSPACE)
```

CI (`.github/workflows/ci.yml`) runs the tests on Linux and Windows, Python 3.10 and 3.12, with
and without numba, plus the Docker test stage.

## Citation & license

If you use hm3denv in your research, please cite it (see [`CITATION.cff`](CITATION.cff)):

```bibtex
@software{hm3denv,
  author  = {Tran, Kim Hieu},
  title   = {hm3denv: physically valid indoor robot-navigation environments from HM3D},
  year    = {2026},
  url     = {https://github.com/TrKimHieu/2D_Navigation_Sim},
  version = {0.8.0}
}
```

The code is released under the [MIT License](LICENSE). **Datasets built from HM3D are subject
to Matterport's HM3D terms of use** — do not redistribute them unless those terms allow it.
Please also cite HM3D (Ramakrishnan et al., *Habitat-Matterport 3D Dataset*, NeurIPS Datasets
and Benchmarks 2021) when you use datasets built from it.
