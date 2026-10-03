# Build maps from GLB files — `build-map`

[← README](../../README.md)

`build-map` turns 3D scenes (`.glb`) into navigation maps and tasks: a dataset in
`data/datasets/<name>`, ready for `sim`, `train` and `gym.make`.

```bash
.\build-map.bat D:\scenes\house.glb                       # one file
.\build-map.bat D:\scenes                                 # every .glb of a folder (and below)
.\build-map.bat D:\hm3d\val --name hm3d-val               # an HM3D split (<id>-<hash>\<hash>.glb folders)
.\build-map.bat house.glb --env grid                      # grid maps instead of continuous ones
.\build-map.bat house.glb --robots turtlebot4 kobuki      # only these robots (faster)
```

(Linux / macOS: `./build-map.sh ...`.) The first run installs the build tools into `.venv`.

What it does:

1. copies the GLBs into `data/raw/glb/` under a scene name: the HM3D folder name
   (`00800-TEEsavR23oF`) for HM3D scenes, else the file name cleaned up (`My Room.glb` →
   `My-Room`); HM3D semantic meshes are skipped;
2. slices every storey of every scene into a 2 cm map (floor, obstacles in each robot's height
   band, ceiling), finds the up axis by itself;
3. vectorises the maps (SVG, continuous env) or cuts them into robot-sized cells (grid env);
4. samples tasks (start / goal pairs) and checks every start and goal on the 3D mesh;
5. writes previews (`data/datasets/<name>/preview/`) and a manifest with sha256 hashes.

These are the same steps and defaults as the published datasets: a scene of `svg-v1` built
with `build-map` gives exactly its tasks.

Running `build-map` again with the same `--name` **adds** the new scenes to the dataset (its
maps, tasks and splits are kept, see [extend-data](extend-data.md)); scenes already in it are
skipped.

| Option | Meaning |
|---|---|
| `--name NAME` | dataset name (default `my-maps`) |
| `--env svg \| grid` | continuous maps for every robot (default), or grid cells |
| `--robots ID ...` | SVG: these robots (default all 9, [list](../robots.md)); grid: the robot whose size sets the cells (default `jetauto_pro`) |
| `--up x \| y \| z` | up axis of the meshes, if the detection gets it wrong |
| `--replace` | a different GLB with the same scene name is already there: replace it |
| `--no-preview`, `--no-verify` | skip the preview images / the 3D checks of the tasks (faster) |
| `--review` | open the storey review tool at the end |

Meshes are read in metres.

## When a storey looks wrong

Check the previews. A scene upside down, two storeys merged, a mezzanine split: run
`hm3d review` (web tool at <http://127.0.0.1:8765>) to fix the up axis, merge or split storeys
or drop a scene, then run `build-map` again (it re-slices the reviewed scenes).

The pipeline in detail, the configuration file and Isaac-Scene-Builder scenes:
[Building datasets](../building-datasets.md).
