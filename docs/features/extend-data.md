# Continue a dataset — `extend-data`

[← README](../../README.md)

`extend-data` adds to a dataset — one downloaded from Hugging Face or one you built — instead
of rebuilding it:

- **new scenes** (more buildings → more maps and tasks),
- **more tasks** on the maps it already has,
- **new robots** (continuous datasets).

```bash
.\extend-data.bat svg-v1 --status                       # what is in it, what can be added
.\extend-data.bat svg-v1 --new-scenes                   # every GLB of data\raw\glb not in it yet
.\extend-data.bat svg-v1 --scenes 00900-abcdefghijk     # these scenes
.\extend-data.bat svg-v1 --add-tasks 10                 # 10 more tasks on every map
.\extend-data.bat svg-v1 --robots kobuki                # tasks for another robot
.\extend-data.bat svg-v1 --from-hf --new-scenes         # download it first if it is not here
```

(Linux / macOS: `./extend-data.sh ...`.)

## What is guaranteed

- What is already in the dataset does not change: its maps and task files stay byte for
  byte, new tasks get the next ids.
- Its scenes keep their split. Tasks someone trained on never move to the test split; new
  scenes are split with the dataset's fractions (70 / 15 / 15 by default).
- New maps and tasks are made with the configuration stored in the dataset's manifest, i.e.
  exactly like the old ones.
- The result goes to a copy, `data/datasets/<name>-ext` (or `--out NAME`); `--in-place`
  changes the dataset itself. The manifest gets a `history` entry (what was added, when, with
  which version) and new sha256 hashes; `hm3d datasets validate NAME` checks it.

## What it needs

| To add | HM3D datasets | Isaac-Scene-Builder datasets |
|---|---|---|
| new scenes | their GLB files in `data/raw/glb/` (`build-map` copies them there; or copy `<scene>.glb` yourself) | the scene builder's output folder (`--source PATH` if it moved) |
| more tasks | the GLBs of the dataset's scenes (3D checks), or `--no-verify` | nothing |
| a robot of an existing height class | the GLBs (3D checks), or `--no-verify` | nothing |
| a robot of a new height class | the GLBs of every scene (slicing at that height) | the output folder |

`--status` lists which scenes can be added and which GLBs are missing. HM3D GLBs come from
Matterport's HM3D release (see [Building datasets](../building-datasets.md#from-hm3d)); the
published datasets do not contain them. With `--no-verify` the new tasks are checked in 2D only;
the manifest records it (`source.verified_3d: "partial"`).

Scenes that give no usable map (too small, no free floor) are reported and remembered, so
`--status` does not offer them again.

## Sharing an extended dataset

An extended dataset is a normal dataset folder: copy it, or upload it to your own Hugging Face
dataset repository with `hf upload`.
