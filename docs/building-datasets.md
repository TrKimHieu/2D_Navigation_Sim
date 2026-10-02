# Building datasets

[← README](../README.md)

Contents: [Workspace](#workspace) · [From HM3D](#from-hm3d) · [From Isaac-Scene-Builder](#from-isaac-scene-builder) · [Configuration reference](#configuration-reference) · [Checks](#checks)

**The quick way:** `.\build-map.bat scene.glb` ([guide](features/build-map.md)) does
everything on this page with the default configuration — copies the GLBs into the workspace,
builds, and adds to an existing dataset; `extend-data` ([guide](features/extend-data.md))
continues a dataset. This page explains the pipeline and how to drive it with your own
configuration.

The build tools are installed by `build-map` / `extend-data` on first use, or with
`.\setup.bat --build` (`./setup.sh --build`), or `pip install -e ".[fast,build]"` in your own
environment.

## Workspace

The workspace is `data/` of the repository, created on first use. Another folder:

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

## From HM3D

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

## From Isaac-Scene-Builder

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

## Configuration reference

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

## Checks

```bash
hm3d datasets validate my-dataset             # sha256 of every file
hm3d verify my-dataset --robot turtlebot4     # re-check every task on the 3D mesh (HM3D only)
hm3d eval my-dataset --robot turtlebot4       # the oracle must reach 100 %
```
