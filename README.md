# hm3denv

Simulator for training robots to navigate indoors: real house floor plans, real robot sizes
and sensors, standard [Gymnasium](https://gymnasium.farama.org/) API.
[Tiếng Việt](README.vi.md)

![Python](https://img.shields.io/badge/python-3.10%20|%203.11%20|%203.12-3776AB?logo=python&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-green)
[![CI](https://github.com/TrKimHieu/2D_Navigation_Sim/actions/workflows/ci.yml/badge.svg)](https://github.com/TrKimHieu/2D_Navigation_Sim/actions/workflows/ci.yml)

<p align="center"><img src="docs/images/episode.gif" width="420" alt="A TurtleBot 4 driving to the red goal in a demo map; orange lines are its LiDAR rays, blue is its path"></p>

## Features

Every feature is **one command**. On Windows run the `.bat` file (in cmd or PowerShell, or
double-click it); on Linux / macOS the `.sh` file. The first run of a feature installs what it
needs into `.venv`. Not sure where to start? Run **`.\app.bat`** (`./app.sh`): a menu of all of them.

| Feature | Windows | Linux / macOS | Guide |
|---|---|---|---|
| **Download a dataset** from Hugging Face into `data/datasets`, ready to use | `.\get-data.bat isb-svg-v1` | `./get-data.sh isb-svg-v1` | [get-data](docs/features/get-data.md) |
| **Open a simulation**: dataset, map, robot, number of envs, start / goal or prepared pairs; watch it, or drive it from your program (ZeroMQ) | `.\sim.bat --dataset demo-svg --view` | `./sim.sh --dataset demo-svg --view` | [sim](docs/features/simulate.md) |
| **Train an agent** (PPO) and test it on unseen maps | `.\train.bat --dataset isb-svg-v1` | `./train.sh --dataset isb-svg-v1` | [train](docs/features/train.md) |
| **Build maps from GLB files** (your scenes, HM3D) | `.\build-map.bat D:\scenes\house.glb` | `./build-map.sh ~/scenes/house.glb` | [build-map](docs/features/build-map.md) |
| **Continue a dataset**: new scenes, more tasks, new robots | `.\extend-data.bat svg-v1 --status` | `./extend-data.sh svg-v1 --status` | [extend-data](docs/features/extend-data.md) |

![get-data, build-map and extend-data fill data/datasets; sim and train use it](docs/images/workflow.svg)

## Quick start

You need git and Python 3.10–3.12 ([python.org](https://www.python.org/downloads/); on Windows tick
*Add python.exe to PATH*).

```bash
git clone https://github.com/TrKimHieu/2D_Navigation_Sim
cd 2D_Navigation_Sim
```

Then, on Windows:

```bat
.\sim.bat --dataset demo-svg --view
```

or on Linux / macOS:

```bash
./sim.sh --dataset demo-svg --view
```

The first run creates `.venv` (a few minutes), then a browser tab shows a robot driving to its
goal in a demo map, one episode after the other until you press Ctrl+C. Small demo datasets (`demo-svg`, `demo-grid`) are included; get the full
ones with `get-data` (needs a free Hugging Face account and an access request, see
[get-data](docs/features/get-data.md)).

## Where things are

```
get-data / sim / train / build-map / extend-data / app (.bat, .sh)   the features
setup.bat, setup.sh          install or update .venv yourself (.\setup.bat --all: everything)
data/                        everything downloaded or built: data/datasets/<name>, data/raw/glb, ...
src/hm3denv/                 the Python package (environments, simulator, dataset tools)
src/hm3denv/configs/sim/     example session files for sim / train (.\sim.bat demo)
examples/                    quickstart.py, train_ppo.py, train_custom.py, remote_client.py
docs/                        guides
```

## Use it in your own code

```python
import gymnasium as gym
import hm3denv

env = gym.make("HM3D/Svg-v0", dataset="isb-svg-v1", robot="turtlebot4")   # or dataset="demo-svg"
obs, info = env.reset(seed=0)                                              # a random task
obs, info = env.reset(options={"map_id": "isb-S001_s0", "task_idx": 3})    # a prepared pair
obs, info = env.reset(options={"map_id": "isb-S001_s0", "start": [1.2, 3.4], "goal": [5.0, 2.0]})
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
```

Activate the environment first (`.venv\Scripts\activate.bat`, PowerShell
`.venv\Scripts\Activate.ps1`, Linux / macOS `source .venv/bin/activate`). Evaluate any policy:
`hm3d eval isb-svg-v1 --robot turtlebot4 --agent mypkg.policies:make` (`make(env)` returns
`policy(obs) -> action`). All commands: `hm3d --help`.

**Your own algorithm** (another RL method, another library, your own training loop): the
observation, action, truncation and evaluation details, and a plain-PyTorch template:
[Use your own algorithm](docs/features/own-algorithm.md).

## Troubleshooting

| Problem | Fix |
|---|---|
| `python` not found / wrong version | install Python 3.10–3.12 and tick *Add python.exe to PATH*; then run the command again |
| `hm3d` / `python` can't find hm3denv | activate `.venv` first (see above), or use the `.bat` / `.sh` commands, which need no activation |
| `get-data`: "you do not have access" | request access on the [dataset page](https://huggingface.co/datasets/TranKimHieu/2D_Navigation_Sim) and wait for the approval e-mail |
| `get-data`: token invalid or expired | `hf auth login` with a new *Read* token from huggingface.co/settings/tokens |
| `sim --serve`: "cannot listen" | another simulator uses the port: `--serve tcp://127.0.0.1:5556` |
| a map from `build-map` looks wrong | check `data/datasets/<name>/preview`, fix the storey with `hm3d review`, run `build-map` again |
| you had a `workspace/` folder from an older version | move its content into `data/` (or set `HM3D_WORKSPACE` to it) |

## More

[Environments & API](docs/environments.md) ·
[Use your own algorithm](docs/features/own-algorithm.md) ·
[Datasets](docs/datasets.md) ·
[Robots](docs/robots.md) ·
[Building datasets in detail](docs/building-datasets.md) ·
[Performance & Docker](docs/performance.md) ·
[Development & citation](docs/development.md)

MIT License. Datasets built from HM3D are subject to Matterport's HM3D terms of use.
