# Performance & Docker

[← README](../README.md)

## Performance

| Knob | Effect |
|---|---|
| `fast` extra (installed by `setup.sh`) | numba kernels with a spatial segment grid — the cost of a step no longer depends on the map size (0.33 ms on a 53 000-segment map). Results are bit-identical to the pure-numpy path (`HM3D_ACCEL=0`). |
| disk cache (on by default) | maps load in ~15 ms instead of ~100 ms after the first time. Location: `HM3D_CACHE` (default `%LOCALAPPDATA%\hm3denv\cache` / `~/.cache/hm3denv`); `HM3D_CACHE=off` disables it. |
| `hm3d cache build svg-v1 --robot turtlebot4 --fields` | pre-builds the cache, including the distance field of every task (~3.5 MB per map). |
| `map_repeat=K` | K× fewer map loads. |
| `envs_per_worker=k` | helps with many more environments than cores. |

Measure on your machine before choosing the number of environments:

```bash
hm3d bench svg-v1 --robot turtlebot4 --num-envs 1 8 16 32
hm3d bench svg-v1 --robot turtlebot4 --num-envs 64 --envs-per-worker 4 --map-repeat 4
```

**Small maps: more processes can be slower.** On the demo maps a step takes ~0.2 ms, less
than sending it to a worker process and back. Measured on a 4-core cloud VM with `demo-svg`:

| Setup | steps/s (total) |
|---|---|
| 1 environment, same process (`hm3d bench --num-envs 1`) | 5 000 |
| 4 environments, `make_vec(..., vectorization="sync")` (same process) | 2 500 |
| 4 environments, one worker process each (default) | 1 650–1 900 |
| 4 environments, `envs_per_worker=2` or `4` | 1 550–1 600 |

Worker processes pay off when a step costs more (large or dense maps, many LiDAR beams) and
you have many cores. When you train a neural network, the network update is often the
bottleneck rather than the simulator: in `examples/train_ppo.py` on the demo data the
environment workers used under 10 % of a core each.

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
