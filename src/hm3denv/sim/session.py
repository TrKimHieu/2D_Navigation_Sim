"""A simulation session: ``num_envs`` environments of one dataset and robot, driven step
by step, with the episodes of the config played in turn (see sim.config).

    s = Session(load_sim_config("my_sim.yaml"))
    obs, infos = s.reset()                       # one entry per environment
    out = s.step([action_0, action_1, ...])      # dicts: obs, reward, terminated, truncated, info
    s.close()

The environments run in a gymnasium vector env with autoreset disabled; when an episode
ends the session starts the next one of the schedule right away (``auto_reset``) and
returns its first observation, with the finished episode's metrics in ``info`` and the
new episode's reset info in ``reset_info``. ``to_jsonable`` turns any result into plain
JSON (lists, None for inf / NaN), which is what the ZeroMQ server sends.
"""

from __future__ import annotations

import base64
import math
import threading
from functools import partial

import gymnasium as gym
import numpy as np
from gymnasium.vector import AutoresetMode

from ..dataset import load_dataset
from ..vector import GroupedVectorEnv, limit_threads
from .config import episode_options


class SimError(ValueError):
    """A request the session cannot carry out (bad episode, finished env, ...)."""


class _ResetGuard(gym.Wrapper):
    """An invalid reset (custom start in a wall, unknown task, ...) must not leave a worker
    process without an episode: reset a random task of the same map instead (or of the
    split) and report the error in info["reset_error"]."""

    def reset(self, *, seed=None, options=None):
        try:
            return self.env.reset(seed=seed, options=options)
        except (ValueError, KeyError, IndexError, FileNotFoundError) as e:
            err = f"{type(e).__name__}: {e}"
        try:
            keep = {"map_id": options["map_id"]} if options and "map_id" in options else None
            obs, info = self.env.reset(seed=seed, options=keep)
        except (ValueError, KeyError, IndexError, FileNotFoundError):
            obs, info = self.env.reset(seed=seed)
        return obs, {**info, "reset_error": err}


def _make_guarded(env_id, kwargs):
    # module-level so worker processes can unpickle it (and import hm3denv there)
    import hm3denv  # noqa: F401  registers HM3D/*
    return _ResetGuard(gym.make(env_id, **kwargs))


def _make_scheduled(env_id, kwargs, maps, episodes, offset, seed):
    import hm3denv  # noqa: F401  registers HM3D/*
    from ..envs.wrappers import EpisodeSchedule
    env = gym.make(env_id, **kwargs)
    return EpisodeSchedule(env, maps, episodes, offset, seed) if maps or episodes else env


def env_setup(cfg: dict, split: str | None = None):
    """(dataset, robot, env_id, env kwargs) of a session config; `split` replaces the
    config's split when given."""
    ds = load_dataset(cfg["dataset"])
    robot = cfg["robot"] or (ds.robots[0] if len(ds.robots) == 1 else
                             "turtlebot4" if "turtlebot4" in ds.robots else ds.robots[0])
    if robot not in ds.robots:
        raise SimError(f"robot {robot!r} is not in dataset {ds.name}: {ds.robots}")
    env_id = "HM3D/Grid-v0" if ds.env_type == "grid" else "HM3D/Svg-v0"
    kwargs = {"dataset": str(ds.root), "robot": robot, "split": split or cfg["split"], **cfg["env"]}
    if cfg["task_filter"]:
        kwargs["task_filter"] = cfg["task_filter"]
    return ds, robot, env_id, kwargs


def env_fns(cfg: dict, n: int, split: str | None = None) -> list:
    """`n` picklable environment factories for a training library's vector env (e.g.
    SB3 SubprocVecEnv): the config's maps / fixed episodes are applied at every reset."""
    _, _, env_id, kwargs = env_setup(cfg, split)
    eps = [episode_options(e) for e in cfg["episodes"]] if cfg["episodes"] else None
    return [partial(_make_scheduled, env_id, kwargs, cfg["maps"], eps, i, cfg["seed"])
            for i in range(n)]


def unbatch_info(info: dict, i: int) -> dict:
    """The info of environment i from a gymnasium vector-env info dict."""
    out = {}
    for k, v in info.items():
        if k.startswith("_"):
            continue
        mask = info.get("_" + k)
        if mask is not None and not mask[i]:
            continue
        out[k] = unbatch_info(v, i) if isinstance(v, dict) else v[i]
    return out


def to_jsonable(x):
    """numpy -> lists / numbers, inf / NaN -> None, tuples -> lists (plain JSON)."""
    if isinstance(x, dict):
        return {str(k): to_jsonable(v) for k, v in x.items()}
    if isinstance(x, np.ndarray):
        return to_jsonable(x.tolist())
    if isinstance(x, (list, tuple)):
        return [to_jsonable(v) for v in x]
    if isinstance(x, np.generic):
        x = x.item()
    if isinstance(x, float):
        return x if math.isfinite(x) else None
    if x is None or isinstance(x, (bool, int, str)):
        return x
    return str(x)


def describe_space(space) -> dict:
    """A JSON description of a gymnasium space (for clients in any language)."""
    if isinstance(space, gym.spaces.Dict):
        return {"type": "Dict", "spaces": {k: describe_space(s) for k, s in space.spaces.items()}}
    if isinstance(space, gym.spaces.Discrete):
        return {"type": "Discrete", "n": int(space.n)}
    if isinstance(space, gym.spaces.Box):
        return {"type": "Box", "shape": list(space.shape), "dtype": str(space.dtype),
                "low": to_jsonable(space.low), "high": to_jsonable(space.high)}
    return {"type": type(space).__name__}


class Session:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        ds, robot, self.env_id, kwargs = env_setup(cfg)
        self.dataset, self.robot = ds, robot
        self.maps = ds.maps(robot, cfg["split"])
        if not self.maps:
            raise SimError(f"{ds.name} has no maps for {robot} in split {cfg['split']}")
        every = set(ds.maps(robot))
        for m in (cfg["maps"] or []) + [e["map"] for e in cfg["episodes"] or [] if e.get("map")]:
            if m not in every:
                raise SimError(f"map {m!r} is not in {ds.name} for {robot} "
                               f"(e.g. {sorted(every)[:3]}; list them with `hm3d info {ds.name}`)")
        self.num_envs = n = cfg["num_envs"]
        fns = [partial(_make_guarded, self.env_id, kwargs)] * n
        mode = AutoresetMode.DISABLED
        if cfg["vectorization"] == "sync" or n == 1:
            self.vec = gym.vector.SyncVectorEnv(fns, autoreset_mode=mode)
        else:
            limit_threads()
            if cfg["envs_per_worker"] > 1:
                self.vec = GroupedVectorEnv(fns, envs_per_worker=cfg["envs_per_worker"],
                                            autoreset_mode=mode)
            else:
                self.vec = gym.vector.AsyncVectorEnv(fns, autoreset_mode=mode)
        self.observation_space = self.vec.single_observation_space
        self.action_space = self.vec.single_action_space
        self._rng = np.random.default_rng(cfg["seed"])
        self._lock = threading.RLock()
        self._next_episode = 0
        self._obs = None
        self._fresh: set[int] = set()
        self._done = [True] * n
        self.stats = [{"episodes": 0, "steps": 0, "return": 0.0, "last": None, "current": None}
                      for _ in range(n)]

    # ------------------------------------------------------------ episodes

    @property
    def local_envs(self):
        """The environment objects (same process only: vectorization 'sync' or 1 env)."""
        return getattr(self.vec, "envs", None)

    def next_options(self) -> dict:
        """Reset options of the next episode of the schedule."""
        c = self.cfg
        if c["episodes"]:
            e = c["episodes"][self._next_episode % len(c["episodes"])]
            self._next_episode += 1
            opt = episode_options(e)
            if "map_id" not in opt and c["maps"]:
                opt["map_id"] = c["maps"][0]
            return opt
        if c["maps"]:
            return {"map_id": c["maps"][int(self._rng.integers(len(c["maps"])))]}
        return {}

    def _obs_of(self, obs, i):
        if isinstance(obs, dict):
            return {k: v[i] for k, v in obs.items()}
        return obs[i]

    def _start(self, skip=()):
        """First call: one plain reset of every env (masked resets need an observation of
        every env to batch; it also seeds env i with seed + i). Then every env not yet
        started (except `skip`) gets the next episode of the schedule."""
        if self._obs is None:
            self._obs, _ = self.vec.reset(seed=self.cfg["seed"])
            self._fresh = set(range(self.num_envs))
        for j in sorted(self._fresh - set(skip)):
            self._reset_one(j)

    def _reset_one(self, i, options=None):
        self._fresh.discard(i)
        mask = np.zeros(self.num_envs, np.bool_)
        mask[i] = True
        opt = dict(self.next_options() if options is None else options)
        obs, info = self.vec.reset(options={**opt, "reset_mask": mask})
        self._obs = obs
        info = unbatch_info(info, i)
        self._done[i] = False
        st = self.stats[i]
        st.update(steps=0, current={k: info.get(k) for k in ("map_id", "task_id", "geodesic_m")})
        st["return"] = 0.0
        return self._obs_of(obs, i), info

    def reset(self, env: int | None = None, options: dict | None = None):
        """Start the next episode of every environment (or of `env`); `options` replaces
        the schedule (map_id, task_idx or start + goal). Returns (obs list, info list)."""
        with self._lock:
            idx = range(self.num_envs) if env is None else [self._check_env(env)]
            self._start(skip=idx)
            obs, infos, errors = [], [], []
            for i in idx:
                o, info = self._reset_one(i, options)
                obs.append(o)
                infos.append(info)
                if "reset_error" in info:
                    errors.append(f"env {i}: {info['reset_error']}")
            if errors:
                raise SimError("; ".join(errors) + " (a random task of the map was started instead)")
            return obs, infos

    def step(self, actions) -> list[dict]:
        """One step of every environment. `actions`: one action per environment (a single
        action is accepted when there is one environment)."""
        with self._lock:
            if self._obs is None:
                self.reset()
            done = [i for i, d in enumerate(self._done) if d]
            if done:
                raise SimError(f"environment(s) {done} finished their episode: reset them first "
                               "(or set auto_reset: true)")
            acts = self._actions(actions)
            obs, rew, term, trunc, info = self.vec.step(acts)
            self._obs = obs
            out = []
            for i in range(self.num_envs):
                r = {"obs": self._obs_of(obs, i), "reward": float(rew[i]),
                     "terminated": bool(term[i]), "truncated": bool(trunc[i]),
                     "info": unbatch_info(info, i)}
                st = self.stats[i]
                st["steps"] += 1
                st["return"] += r["reward"]
                if term[i] or trunc[i]:
                    st["episodes"] += 1
                    st["last"] = {**{k: r["info"].get(k) for k in
                                     ("map_id", "task_id", "success", "spl", "termination",
                                      "path_length", "time")},
                                  "return": st["return"], "steps": st["steps"]}
                    self._done[i] = True
                    if self.cfg["auto_reset"]:
                        r["obs"], r["reset_info"] = self._reset_one(i)
                out.append(r)
            return out

    def _check_env(self, i) -> int:
        if not isinstance(i, int) or not 0 <= i < self.num_envs:
            raise SimError(f"env must be an integer in [0, {self.num_envs - 1}], got {i!r}")
        return i

    def _actions(self, actions):
        sp = self.action_space
        a = np.asarray(actions, dtype=sp.dtype)
        one = (self.num_envs,) + sp.shape
        if a.shape == sp.shape and self.num_envs == 1:
            a = a.reshape(one)
        if a.shape != one:
            raise SimError(f"expected {self.num_envs} action(s) of shape {list(sp.shape)} "
                           f"({sp}), got shape {list(a.shape)}")
        if isinstance(sp, gym.spaces.Discrete) and not ((a >= sp.start) & (a < sp.start + sp.n)).all():
            raise SimError(f"discrete actions must be in [0, {sp.n - 1}]")
        return a

    # ------------------------------------------------------------ rendering

    def render(self, env: int = 0) -> tuple[str, bytes | str]:
        """("svg", text) for the continuous env, ("png", bytes) for the grid env."""
        with self._lock:
            i = self._check_env(env)
            if self._obs is None:
                self.reset()
            if self.env_id == "HM3D/Svg-v0":
                return "svg", self.vec.call("render_svg")[i]
            import cv2
            img = self.vec.call("_frame")[i]
            return "png", cv2.imencode(".png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))[1].tobytes()

    def render_json(self, env: int = 0) -> dict:
        kind, data = self.render(env)
        return {"svg": data} if kind == "svg" else {"png_base64": base64.b64encode(data).decode()}

    # ------------------------------------------------------------ description

    def describe(self) -> dict:
        return {"env_id": self.env_id, "dataset": self.dataset.name, "robot": self.robot,
                "split": self.cfg["split"], "num_envs": self.num_envs,
                "auto_reset": self.cfg["auto_reset"], "maps": self.maps,
                "episodes": self.cfg["episodes"],
                "observation_space": describe_space(self.observation_space),
                "action_space": describe_space(self.action_space)}

    def close(self):
        with self._lock:
            self.vec.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
