# data/

Everything hm3denv downloads or builds lands here (nothing in this folder is committed
except this file). The commands create what they need:

```
data/
  datasets/<name>/     datasets: downloaded (get-data) and built (build-map, extend-data)
  raw/glb/             your .glb scenes (build-map copies them here)
  cache/               mesh cache, downloaded robot models
  stages/slice/        intermediate storey maps (+ review.json after `hm3d review`)
  robots/              your robot presets (override the packaged ones)
  configs/             build configurations
  workspace.yaml       marks this folder as the hm3denv workspace
```

A dataset in `data/datasets/` is found by name everywhere: `sim`, `train`, `hm3d eval`,
`gym.make("HM3D/Svg-v0", dataset="<name>")`.

Use another folder by setting `HM3D_HOME` (data home) or `HM3D_WORKSPACE` (workspace).
