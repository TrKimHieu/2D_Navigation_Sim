# Download a dataset — `get-data`

[← README](../../README.md)

One command downloads a full dataset from Hugging Face into `data/datasets/<name>` of this
repository, checks every file against its sha256 and makes it usable by name everywhere
(`sim`, `train`, `hm3d eval`, `gym.make(..., dataset="<name>")`). Nothing to move or configure.

```bash
.\get-data.bat isb-svg-v1          # Windows (cmd, PowerShell, or double-click get-data.bat for the list)
./get-data.sh isb-svg-v1         # Linux / macOS
```

| Command | What it does |
|---|---|
| `.\get-data.bat` | asks for your token if needed, lists the datasets you can download, you pick |
| `.\get-data.bat isb-svg-v1 svg-v1` | downloads these datasets |
| `.\get-data.bat --all` | everything you have access to |
| `.\get-data.bat NAME --force` | downloads again (e.g. after a new release) |
| `.\get-data.bat NAME --dir D:\data` | another folder (then set `HM3D_DATASETS=D:\data`) |

The first run creates `.venv` and installs what is needed (`huggingface_hub`). Datasets
already present are kept.

## Before the first download (once per account)

1. Create a free account on <https://huggingface.co>.
2. Request access on the [dataset page](https://huggingface.co/datasets/TranKimHieu/2D_Navigation_Sim)
   and wait for the approval e-mail. The HM3D-based datasets are only shared with people who
   have Matterport's HM3D licence.
3. Create a token of type *Read* at <https://huggingface.co/settings/tokens>. `get-data` asks for
   it the first time and remembers it (or run `hf auth login`, or set `HF_TOKEN`).

## Errors

Every problem stops the command with an explanation and a distinct exit code (for scripts):

| Exit code | Meaning | What to do |
|---|---|---|
| 2 | unknown dataset name, or files that do not match the manifest | check the name (`.\get-data.bat` lists them); `--force` to download again |
| 3 | `huggingface_hub` could not be installed | check the internet connection, run `.\setup.bat --hub` |
| 4 | not logged in (no terminal to ask), or the token is invalid / expired | `hf auth login` with a new *Read* token, or set `HF_TOKEN` |
| 5 | your account has no access to the datasets yet | request access (above) and wait for the approval |
| 6 | Hugging Face unreachable (offline, proxy, timeout) | check the connection and run it again |

## Available datasets

| Dataset | Environment | Source | Maps | Robots | Tasks |
|---|---|---|---|---|---|
| `isb-svg-v1` | `HM3D/Svg-v0` | Isaac-Scene-Builder | 204 | all 9 | 36 720 |
| `isb-grid-v1` | `HM3D/Grid-v0` | Isaac-Scene-Builder | 202 | `jetauto_pro` | 4 040 |
| `svg-v1` | `HM3D/Svg-v0` | HM3D | 177 | all 9 | 30 835 |
| `grid-jetauto-v1` | `HM3D/Grid-v0` | HM3D | 157 | `jetauto_pro` | 3 140 |
| `grid-s15-v1` | `HM3D/Grid-v0` (15 cm cells) | HM3D | 166 | `s15_h63` | 3 320 |

More about the datasets, their format and where they are searched: [Datasets](../datasets.md).
To add scenes, tasks or robots to a downloaded dataset: [extend-data](extend-data.md).
