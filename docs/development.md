# Development

[← README](../README.md)

## Manual installation

`.\setup.bat` / `setup.sh` (and every feature command on first use) do exactly this; use it if
you prefer to manage the environment yourself (conda, uv, an existing venv, …). Python 3.10–3.12.

```bash
git clone https://github.com/TrKimHieu/2D_Navigation_Sim && cd 2D_Navigation_Sim
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\Activate.ps1
pip install -e ".[fast,build,test]"
pytest -m "not workspace"
```

Without cloning (as a dependency of your own project):

```bash
pip install "hm3denv[fast] @ git+https://github.com/TrKimHieu/2D_Navigation_Sim"
```

Optional extras (combine them, e.g. `hm3denv[fast,build]`; `.\setup.bat` / `setup.sh` flags in brackets):

| Extra | Adds | Needed for |
|---|---|---|
| `fast` | numba | compiled collision / LiDAR kernels (steps ~3.5× faster on HM3D maps, up to ~45× on very dense maps) — recommended (always installed by the setup scripts) |
| `hub` | huggingface_hub | downloading the pre-built datasets (`get-data`, `hm3d download`) — `--hub` |
| `build` | trimesh, pillow, xacro, pycollada, shapely | building datasets (`build-map`, `extend-data`, `hm3d build`, `hm3d review`, …) — `--build` |
| `sim` | pyzmq | the simulator server (`sim --serve`, `hm3denv.sim.SimClient`) — `--sim` |
| `test` | pytest | running the tests — `--all` (every extra except `train`) |
| `train` | stable-baselines3 | `examples/train_ppo.py` — `--train` (also installs CPU PyTorch; `train` does it on first use) |

The feature commands (`get-data`, `sim`, ...) run `scripts/win/run.ps1` (Windows) or
`scripts/run.sh`, which install the extras a feature needs before starting it.

## Source layout and tests

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
and without numba, the README's scripts (`setup`, `quickstart.py`, a tiny `train` run) on Linux
and Windows, plus the Docker test stage.

## Citation & license

If you use hm3denv in your research, please cite it (see [`CITATION.cff`](../CITATION.cff)):

```bibtex
@software{hm3denv,
  author  = {Tran, Kim Hieu},
  title   = {hm3denv: physically valid indoor robot-navigation environments from HM3D},
  year    = {2026},
  url     = {https://github.com/TrKimHieu/2D_Navigation_Sim},
  version = {0.8.0}
}
```

The code is released under the [MIT License](../LICENSE). **Datasets built from HM3D are subject
to Matterport's HM3D terms of use** — do not redistribute them unless those terms allow it.
Please also cite HM3D (Ramakrishnan et al., *Habitat-Matterport 3D Dataset*, NeurIPS Datasets
and Benchmarks 2021) when you use datasets built from it.
