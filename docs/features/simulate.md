# Open a simulation — `sim`

[← README](../../README.md)

`sim` opens a simulation: a dataset, a robot, one or more environments side by side, the
maps and the start / goal pairs you choose. Then either

- an agent drives it and you **watch it in the browser** (`--view`), or
- **your own program drives it** step by step from another process, over ZeroMQ (`--serve`),
  or from Python in the same process.

```bash
.\sim.bat --dataset demo-svg --view                          # oracle agent, watch it
.\sim.bat --dataset isb-svg-v1 --robot pal_tiago --num-envs 4 --view
.\sim.bat --dataset demo-svg --map demo-S001_s0 --start 2.48,12.36,0 --goal 4.84,10.72 --view
.\sim.bat my_sim.yaml --serve                                # your program sends actions
```

(Linux / macOS: `./sim.sh ...`.) The first run installs what is needed into `.venv`.

## Choosing what to simulate

On the command line, or in a session file (YAML) for anything longer. Command-line options
override the file.

| Option | Session file key | Meaning |
|---|---|---|
| `--dataset NAME` | `dataset` | dataset name (`data/datasets`, demos `demo-svg` / `demo-grid`) or path; [list of datasets](../datasets.md) |
| `--robot ID` | `robot` | robot preset id, see the [list of robots](../robots.md); it must be one of the dataset's robots ([which dataset has which robots](../datasets.md), or `hm3d info NAME`) |
| `--split S` | `split` | `train`, `val` or `test` maps |
| `--map M ...` | `maps` | only these maps (ids like `demo-S001_s0`) |
| `--task I` | `episodes: [{map: M, task_idx: I}]` | a task of the dataset (a prepared start / goal pair) |
| `--start X,Y[,TH] --goal X,Y` | `episodes: [{map: M, start: [...], goal: [...]}]` | your own start and goal |
| `--num-envs N` | `num_envs` | environments side by side (one process each by default) |
| | `task_filter` | random tasks only within limits, e.g. `{geodesic_m: {min: 2, max: 8}}` |
| | `env` | any environment argument: `dt`, `n_beams`, `lidar_noise`, `reward`, `on_collision`, ... |
| | `auto_reset` | start the next episode as soon as one ends (default true) |

Start and goal are in metres in the map frame (theta in radians, default: facing the goal);
for grid datasets they are `row,col` cells. They are checked: a start in a wall, a goal too
close to a wall or unreachable is refused with a message saying why.

```yaml
# my_sim.yaml
dataset: isb-svg-v1
robot: turtlebot4
split: train
num_envs: 4
episodes:                      # played in turn; without it: random tasks of the maps
  - {map: isb-S001_s0, task_idx: 3}
  - {map: isb-S001_s0, start: [1.2, 3.4, 0.0], goal: [5.0, 2.0]}
env: {dt: 0.1, n_beams: 72}
server: {bind: "tcp://127.0.0.1:5555"}
viewer: {port: 8770}
```

Packaged examples (use them by name): `.\sim.bat demo`, `.\sim.bat custom_pairs` — see
[`src/hm3denv/configs/sim/`](../../src/hm3denv/configs/sim/).

## Watching (`--view`)

`--view` opens <http://127.0.0.1:8770/> in the browser: the map, the robot, its path, the
goal, and the episode's numbers, for any environment of the session. `--agent oracle |
random | mypkg.module:make` chooses who drives (`make(env)` returns `policy(obs) -> action`,
as for `hm3d eval`); `--fps` the speed. The agent plays **until Ctrl+C** (or `--episodes N`:
N episodes of all environments together; fixed episodes are played once). The header of the
page shows the run: running / finished / stopped, episodes played, success rate and SPL. When
a run finishes, environments still on an episode stay where they are.

## Driving it from your program (`--serve`)

`.\sim.bat my_sim.yaml --serve` waits for requests on `tcp://127.0.0.1:5555` (ZeroMQ REQ/REP,
one JSON request → one JSON reply), so any language with ZeroMQ can drive it. Add `--view`
to watch at the same time.

```python
from hm3denv.sim import SimClient

with SimClient("tcp://127.0.0.1:5555") as sim:
    info = sim.info()                         # spaces, robot, maps, number of environments
    obs, infos = sim.reset()                  # one entry per environment
    while True:
        rep = sim.step([my_policy(o) for o in obs])
        obs = rep["obs"]                      # rep["reward"], rep["terminated"], rep["info"], ...
```

A complete example: [`examples/remote_client.py`](../../examples/remote_client.py).
`hm3denv.sim.RemoteEnv(address)` wraps a one-environment server as a gymnasium `Env`, so
existing training code runs against it unchanged (start the server with `--no-auto-reset`).

| Request | Reply |
|---|---|
| `{"cmd": "info"}` | `env_id`, `dataset`, `robot`, `num_envs`, `maps`, `observation_space`, `action_space` |
| `{"cmd": "reset", "env": 0, "options": {...}}` | `obs`, `info` (lists); `env` omitted = every env; `options` = `{"map_id", "task_idx"}` or `{"map_id", "start", "goal"}` |
| `{"cmd": "step", "actions": [[v, w], ...]}` | `obs`, `reward`, `terminated`, `truncated`, `info`, `reset_info`: one entry per env |
| `{"cmd": "render", "env": 0}` | `svg` (continuous) or `png_base64` (grid) |
| `{"cmd": "stats"}` | per env: episodes, steps, return, last episode |
| `{"cmd": "close"}` | stops the server |

Every reply has `"ok": true`, or `"ok": false` and `"error"`. With `auto_reset` an environment
whose episode ended has already started the next one: `obs` is the new first observation,
`info` the finished episode's metrics (`success`, `spl`, `termination`, ...), `reset_info` the
new episode's. Infinite numbers are sent as `null`.

## From Python in the same process

```python
from hm3denv.sim import Session, load_sim_config

with Session(load_sim_config("my_sim.yaml", num_envs=2)) as s:
    obs, infos = s.reset()
    out = s.step(actions)        # list of {obs, reward, terminated, truncated, info}
```

Or the plain Gymnasium API with your own episode:

```python
env = gym.make("HM3D/Svg-v0", dataset="demo-svg", robot="turtlebot4")
obs, info = env.reset(options={"map_id": "demo-S001_s0", "start": [2.48, 12.36], "goal": [4.84, 10.72]})
```

Observations, actions, rewards and the step `info`: [Environments & API](../environments.md).
