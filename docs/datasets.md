# Datasets

[← README](../README.md)

Contents: [Bundled demos](#bundled-demo-datasets) · [Pre-built datasets (Hugging Face)](#pre-built-datasets-hugging-face) · [Building your own](#building-your-own) · [Where datasets are found](#where-datasets-are-found) · [Dataset format](#dataset-format)

## Bundled demo datasets

Environments read *datasets* (maps + tasks). Two small demo datasets are **bundled with the
package**, so everything works right after installation:

| Dataset | Environment | Robots | Maps (train / val / test) | Tasks |
|---|---|---|---|---|
| `demo-svg` | `HM3D/Svg-v0` | `turtlebot4`, `clearpath_jackal` | 6 scenes (3 / 2 / 1) | 10 per map and robot |
| `demo-grid` | `HM3D/Grid-v0` | `jetauto_pro` | 6 scenes (3 / 2 / 1) | 10 per map |

They were built from Isaac-Scene-Builder scenes with the same pipeline as full datasets and are
meant for trying the API, tests and debugging.

## Pre-built datasets (Hugging Face)

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
2. Run `get-data` ([guide](features/get-data.md)). It installs the download support, asks
   for a Hugging Face token once (create one with *Read* access at
   <https://huggingface.co/settings/tokens>), lists the datasets and downloads the ones you
   pick into `data/datasets/` of the repository:

```bash
.\get-data.bat                     # choose from a list      (Linux/macOS: ./get-data.sh)
.\get-data.bat isb-svg-v1 svg-v1   # by name
.\get-data.bat --all               # everything you have access to
```

The same with the CLI, if you prefer:

```bash
.\setup.bat --hub                  # adds huggingface_hub and its `hf` command (./setup.sh --hub)
hf auth login                    # once per machine
hm3d download                    # list the datasets in the repository
hm3d download isb-svg-v1 svg-v1  # -> data/datasets/ (or $HM3D_HOME/datasets)
```

Downloaded datasets are validated against their manifest and then found by name:
`gym.make("HM3D/Svg-v0", dataset="isb-svg-v1", robot="pal_tiago")`. Without access approval,
`hm3d download NAME` stops with an explanation and downloads nothing.

## Building your own

- **From HM3D** — real houses. HM3D is distributed by Matterport under its own terms of use
  (see the [official HM3D page](https://aihabitat.org/datasets/hm3d/)); request access and
  download it yourself, then [build the datasets](building-datasets.md#from-hm3d).
- **From Isaac-Scene-Builder** — scenes you designed yourself, see
  [Building datasets](building-datasets.md#from-isaac-scene-builder). No license restriction.
- **Shared dataset directories** — put them anywhere and point `HM3D_DATASETS` to their parent
  directory.

## Where datasets are found

By name, in this order: the directories of `HM3D_DATASETS` (separated by `;` on Windows and `:`
on Linux/macOS), `<workspace>/datasets/`, the data home's `datasets/` (`data/datasets` of the
repository; `$HM3D_HOME/datasets` if set; `~/.hm3denv/datasets` when hm3denv is installed
without the repository), then the bundled demos. You can also pass a path.
By default the workspace *is* the data home `data/`, so downloaded, built (`build-map`) and
extended (`extend-data`) datasets all land in `data/datasets/`.

```bash
hm3d datasets                      # list the datasets found
hm3d info demo-svg                 # robots, splits, map and task counts, flagged maps
hm3d datasets validate demo-svg    # check every file against the sha256 in the manifest
```

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
