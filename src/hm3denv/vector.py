"""Vectorised environments: many simulations in parallel on the CPU cores.

    venv = hm3denv.make_vec("HM3D/Svg-v0", 16, dataset="svg-v1", robot="turtlebot4",
                            split="train")
    obs, info = venv.reset(seed=0)             # env i is seeded with 0 + i

The environments are CPU-bound and single-threaded, so the right layout is one worker
process per core. On Windows (spawn start method) call this under
``if __name__ == "__main__":``.

``envs_per_worker = k > 1`` runs k environments in each worker process
(GroupedVectorEnv): one message per worker per step instead of one per environment,
which matters when a step costs about as much as the inter-process round trip.
Observations go through shared memory (workers write them in place, as AsyncVectorEnv
does); only rewards, flags and infos travel through the pipes. This pays off when there
are many more environments than physical CPU cores (choose k ~ envs / cores); with up
to one environment per core, envs_per_worker=1 is fastest.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import traceback
from copy import deepcopy
from functools import partial

import gymnasium as gym
import numpy as np
from gymnasium.vector import AutoresetMode, VectorEnv
from gymnasium.vector.utils import (CloudpickleWrapper, batch_space, concatenate,
                                    create_empty_array, create_shared_memory, iterate,
                                    read_from_shared_memory, write_to_shared_memory)

THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")


def _make(env_id, kwargs):
    # module-level so that unpickling it in a worker imports hm3denv (registers HM3D/*)
    return gym.make(env_id, **kwargs)


def env_fn(env_id: str, **kwargs):
    """A picklable ``() -> env`` factory for any vector env implementation that starts
    worker processes: they import hm3denv when they unpickle it, so the HM3D/* ids are
    registered there."""
    return partial(_make, env_id, kwargs)


def limit_threads() -> None:
    """One BLAS/OpenMP thread per worker process (only where not set already)."""
    for var in THREAD_VARS:
        os.environ.setdefault(var, "1")


def make_vec(env_id: str, num_envs: int, *, vectorization: str = "async",
             envs_per_worker: int = 1, **kwargs):
    """``num_envs`` copies of ``gym.make(env_id, **kwargs)`` in a gymnasium vector env.

    vectorization: "async" (worker processes) or "sync" (same process, for debugging).
    envs_per_worker: environments per worker process (async only); 1 uses gymnasium's
    AsyncVectorEnv, more uses GroupedVectorEnv (same results, fewer messages).
    Worker processes inherit OMP/MKL/OpenBLAS limited to one thread, so the workers do
    not oversubscribe the cores; these variables are only set where still unset.
    """
    if vectorization not in ("async", "sync"):
        raise ValueError("vectorization must be 'async' or 'sync'")
    if num_envs < 1 or envs_per_worker < 1:
        raise ValueError("num_envs and envs_per_worker must be >= 1")
    fns = [env_fn(env_id, **kwargs)] * num_envs
    if vectorization == "sync":
        return gym.vector.SyncVectorEnv(fns)
    limit_threads()
    if envs_per_worker == 1:
        return gym.vector.AsyncVectorEnv(fns)
    return GroupedVectorEnv(fns, envs_per_worker=envs_per_worker)


# ------------------------------------------------------------------ worker processes

def _worker(remote, parent_remote, fns, autoreset, shm=None, space=None, gidx=None):
    """Runs a group of environments with the per-environment loop of gymnasium's
    SyncVectorEnv (autoreset included). Every reply is ("ok", payload) or ("error",
    traceback). With shared memory (shm, the observation space, and the global index of
    each env) observations are written there and replaced by None in the replies."""
    parent_remote.close()

    def emit(i, obs):
        if shm is None:
            return obs
        write_to_shared_memory(space, gidx[i], obs, shm)
        return None

    try:
        envs = [fn() for fn in fns.fn]
        remote.send(("ok", [(e.observation_space, e.action_space) for e in envs]
                     + [(envs[0].metadata, envs[0].render_mode)]))
    except Exception:
        remote.send(("error", traceback.format_exc()))
        return
    auto = [False] * len(envs)                 # NEXT_STEP: reset at the next step
    while True:
        try:
            cmd, data = remote.recv()
        except (EOFError, KeyboardInterrupt):
            break
        try:
            if cmd == "step":
                out = []
                for i, (env, action) in enumerate(zip(envs, data)):
                    adds = []
                    if autoreset == AutoresetMode.NEXT_STEP.value and auto[i]:
                        obs, info = env.reset()
                        rew, term, trunc = 0.0, False, False
                    else:
                        if autoreset == AutoresetMode.DISABLED.value:
                            assert not auto[i], "step on a finished episode (autoreset disabled)"
                        obs, rew, term, trunc, info = env.step(action)
                        if autoreset == AutoresetMode.SAME_STEP.value and (term or trunc):
                            adds.append({"final_obs": obs, "final_info": info})
                            obs, info = env.reset()
                    adds.append(info)
                    auto[i] = bool(term or trunc)
                    out.append((emit(i, obs), rew, term, trunc, adds))
                remote.send(("ok", out))
            elif cmd == "reset":
                seeds, options, mask = data
                out = []
                for i, env in enumerate(envs):
                    if mask is None or mask[i]:
                        auto[i] = False
                        obs, info = env.reset(seed=seeds[i], options=options)
                        out.append((i, emit(i, obs), info))
                remote.send(("ok", out))
            elif cmd == "call":
                idx, name, args, kwargs = data
                res = []
                for i in idx:
                    f = envs[i].get_wrapper_attr(name)
                    res.append(f(*args, **kwargs) if callable(f) else f)
                remote.send(("ok", res))
            elif cmd == "get_attr":
                idx, name = data
                remote.send(("ok", [envs[i].get_wrapper_attr(name) for i in idx]))
            elif cmd == "set_attr":
                idx, name, values = data
                for i, v in zip(idx, values):
                    envs[i].set_wrapper_attr(name, v)
                remote.send(("ok", [None] * len(idx)))
            elif cmd == "close":
                for env in envs:
                    env.close()
                remote.send(("ok", None))
                break
            else:
                raise NotImplementedError(f"unknown command {cmd!r}")
        except Exception:
            remote.send(("error", traceback.format_exc()))
    remote.close()


class Workers:
    """Worker processes holding k consecutive environments each (the last group may be
    smaller); results always come back in environment order."""

    def __init__(self, env_fns, envs_per_worker: int, autoreset=None,
                 context: str | None = None, shared_space=None):
        env_fns = list(env_fns)
        if not env_fns or envs_per_worker < 1:
            raise ValueError("need at least one env and envs_per_worker >= 1")
        self.n, self.k = len(env_fns), envs_per_worker
        self.groups = [list(range(s, min(s + self.k, self.n))) for s in range(0, self.n, self.k)]
        ctx = mp.get_context(context)
        self.remotes, self.processes = [], []
        self.closed = False
        # observations of all envs in shared memory (None: sent through the pipes)
        self.shm = None if shared_space is None else create_shared_memory(shared_space, self.n, ctx)
        for g in self.groups:
            remote, work = ctx.Pipe()
            fns = CloudpickleWrapper([env_fns[i] for i in g])
            p = ctx.Process(target=_worker, daemon=True,
                            args=(work, remote, fns, autoreset, self.shm, shared_space, g))
            p.start()
            work.close()
            self.remotes.append(remote)
            self.processes.append(p)
        try:
            first = self.recv_all()
        except Exception:
            self.close()
            raise
        self.spaces = [s for reply in first for s in reply[:-1]]
        self.metadata, self.render_mode = first[0][-1]

    def recv_all(self, remotes=None) -> list:
        """One reply per remote; raises after reading all of them if any worker failed."""
        out, err = [], None
        for r in (self.remotes if remotes is None else remotes):
            status, payload = r.recv()
            if status == "error" and err is None:
                err = payload
            out.append(payload)
        if err is not None:
            raise RuntimeError(f"error in a worker process:\n{err}")
        return out

    def send_each(self, cmd, payloads) -> None:
        for r, p in zip(self.remotes, payloads):
            r.send((cmd, p))

    def per_env(self, cmd, indices, make_payload) -> list:
        """A command on some environments; results in the order of `indices`.
        make_payload(worker, local_indices) builds each worker's message."""
        indices = list(indices)
        where: dict = {}
        for gi in indices:
            where.setdefault(gi // self.k, []).append(gi % self.k)
        order = list(where)
        for w in order:
            self.remotes[w].send((cmd, make_payload(w, where[w])))
        res = dict(zip(order, self.recv_all([self.remotes[w] for w in order])))
        pos = dict.fromkeys(order, 0)
        out = []
        for gi in indices:
            w = gi // self.k
            out.append(res[w][pos[w]])
            pos[w] += 1
        return out

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        for r in self.remotes:
            try:
                r.send(("close", None))
            except (BrokenPipeError, OSError):
                pass
        for r in self.remotes:
            try:
                r.recv()
            except (EOFError, OSError):
                pass
        for p in self.processes:
            p.join(timeout=10)
            if p.is_alive():
                p.terminate()


class GroupedVectorEnv(VectorEnv):
    """gymnasium VectorEnv running ``envs_per_worker`` environments in each worker
    process. Observations, rewards, flags and infos equal those of
    ``SyncVectorEnv(env_fns)`` for the same seeds and actions: each worker runs
    SyncVectorEnv's per-environment loop and the batching uses gymnasium's own
    concatenate / _add_info (or shared memory for the observations).

    shared_memory: workers write observations into shared memory (default) instead of
    sending them; they must then have the dtype / shape of the observation space.
    copy: return a copy of the observations (else the shared buffer itself).
    """

    def __init__(self, env_fns, envs_per_worker: int = 2, copy: bool = True,
                 autoreset_mode: str | AutoresetMode = AutoresetMode.NEXT_STEP,
                 context: str | None = None, shared_memory: bool = True):
        super().__init__()
        self.autoreset_mode = (autoreset_mode if isinstance(autoreset_mode, AutoresetMode)
                               else AutoresetMode(autoreset_mode))
        self.copy = copy
        env_fns = list(env_fns)
        space = None
        if shared_memory:                         # the space is needed before the workers
            dummy = env_fns[0]()
            space = dummy.observation_space
            dummy.close()
        self._w = Workers(env_fns, envs_per_worker, self.autoreset_mode.value, context,
                          shared_space=space)
        self._view = None if space is None else read_from_shared_memory(space, self._w.shm,
                                                                        self._w.n)
        self.num_envs = self._w.n
        self.single_observation_space, self.single_action_space = self._w.spaces[0]
        for obs_space, act_space in self._w.spaces:
            assert obs_space == self.single_observation_space, "sub-environment observation spaces differ"
            assert act_space == self.single_action_space, "sub-environment action spaces differ"
        self.metadata = dict(self._w.metadata)
        self.metadata["autoreset_mode"] = self.autoreset_mode
        self.render_mode = self._w.render_mode
        self.observation_space = batch_space(self.single_observation_space, self.num_envs)
        self.action_space = batch_space(self.single_action_space, self.num_envs)
        self._env_obs = [None] * self.num_envs
        self._observations = create_empty_array(self.single_observation_space, n=self.num_envs,
                                                fn=np.zeros)
        self._rewards = np.zeros((self.num_envs,), dtype=np.float64)
        self._terminations = np.zeros((self.num_envs,), dtype=np.bool_)
        self._truncations = np.zeros((self.num_envs,), dtype=np.bool_)

    def reset(self, *, seed=None, options=None):
        if seed is None:
            seed = [None] * self.num_envs
        elif isinstance(seed, int):
            seed = [seed + i for i in range(self.num_envs)]
        assert len(seed) == self.num_envs, f"seed list must have length num_envs={self.num_envs}"
        mask = None
        if options is not None and "reset_mask" in options:
            options = dict(options)
            mask = options.pop("reset_mask")
            assert isinstance(mask, np.ndarray) and mask.shape == (self.num_envs,) \
                and mask.dtype == np.bool_ and mask.any(), "invalid options['reset_mask']"
            self._terminations[mask] = False
            self._truncations[mask] = False
        else:
            self._terminations[:] = False
            self._truncations[:] = False
        self._w.send_each("reset", [([seed[i] for i in g], options,
                                     None if mask is None else [bool(mask[i]) for i in g])
                                    for g in self._w.groups])
        infos = {}
        for g, res in zip(self._w.groups, self._w.recv_all()):
            for local, obs, info in res:
                self._env_obs[g[local]] = obs
                infos = self._add_info(infos, info, g[local])
        return self._batch_obs(), infos

    def _batch_obs(self):
        if self._view is not None:
            return deepcopy(self._view) if self.copy else self._view
        self._observations = concatenate(self.single_observation_space, self._env_obs,
                                         self._observations)
        return deepcopy(self._observations) if self.copy else self._observations

    def step(self, actions):
        actions = list(iterate(self.action_space, actions))
        self._w.send_each("step", [[actions[i] for i in g] for g in self._w.groups])
        infos = {}
        for g, res in zip(self._w.groups, self._w.recv_all()):
            for i, (obs, rew, term, trunc, adds) in zip(g, res):
                self._env_obs[i] = obs
                self._rewards[i], self._terminations[i], self._truncations[i] = rew, term, trunc
                for info in adds:
                    infos = self._add_info(infos, info, i)
        return (self._batch_obs(), np.copy(self._rewards), np.copy(self._terminations),
                np.copy(self._truncations), infos)

    def call(self, name: str, *args, **kwargs) -> tuple:
        return tuple(self._w.per_env("call", range(self.num_envs),
                                     lambda w, idx: (idx, name, args, kwargs)))

    def get_attr(self, name: str) -> tuple:
        return self.call(name)

    def set_attr(self, name: str, values) -> None:
        if not isinstance(values, (list, tuple)):
            values = [values] * self.num_envs
        if len(values) != self.num_envs:
            raise ValueError(f"need {self.num_envs} values, got {len(values)}")
        k = self._w.k
        self._w.per_env("set_attr", range(self.num_envs),
                        lambda w, idx: (idx, name, [values[w * k + i] for i in idx]))

    def render(self):
        return self.call("render")

    def close_extras(self, **kwargs) -> None:
        if hasattr(self, "_w"):
            self._w.close()
