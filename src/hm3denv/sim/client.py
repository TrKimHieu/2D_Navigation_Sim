"""Python client of the simulator server (see sim.server for the protocol).

    from hm3denv.sim import SimClient
    with SimClient("tcp://127.0.0.1:5555") as sim:
        print(sim.info()["action_space"])
        obs, info = sim.reset()
        out = sim.step([[0.5, 0.0]])            # one action per environment

``RemoteEnv`` wraps a one-environment server as a gymnasium Env, so any training code
written for gymnasium runs against a simulator in another process or on another machine.
"""

from __future__ import annotations

import json

import gymnasium as gym
import numpy as np


class SimRemoteError(RuntimeError):
    """The server answered with ok: false."""


class SimClient:
    def __init__(self, address: str = "tcp://127.0.0.1:5555", timeout_s: float = 30.0):
        from .server import _zmq
        self._zmq = zmq = _zmq()
        self.address, self.timeout_ms = address, int(timeout_s * 1000)
        self.ctx = zmq.Context.instance()
        self._connect()

    def _connect(self):
        zmq = self._zmq
        self.sock = self.ctx.socket(zmq.REQ)
        self.sock.setsockopt(zmq.LINGER, 0)
        self.sock.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        self.sock.connect(self.address)

    def request(self, **req) -> dict:
        self.sock.send(json.dumps(req).encode())
        try:
            rep = json.loads(self.sock.recv())
        except self._zmq.Again:
            self.sock.close()                   # a REQ socket is stuck after a lost reply
            self._connect()
            raise TimeoutError(f"no answer from the simulator at {self.address} "
                               f"within {self.timeout_ms / 1000:g} s; is `hm3d sim --serve` running?") from None
        if not rep.get("ok"):
            raise SimRemoteError(rep.get("error", "unknown error"))
        return rep

    def info(self) -> dict:
        return self.request(cmd="info")

    def reset(self, env: int | None = None, options: dict | None = None):
        rep = self.request(cmd="reset", env=env, options=options)
        return rep["obs"], rep["info"]

    def step(self, actions) -> dict:
        return self.request(cmd="step", actions=np.asarray(actions).tolist())

    def render(self, env: int = 0) -> dict:
        return self.request(cmd="render", env=env)

    def stats(self) -> list:
        return self.request(cmd="stats")["stats"]

    def close(self, stop_server: bool = False):
        if stop_server:
            self.request(cmd="close")
        self.sock.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def space_from(d: dict) -> gym.Space:
    """The gymnasium space of a server description (sim.session.describe_space)."""
    t = d["type"]
    if t == "Dict":
        return gym.spaces.Dict({k: space_from(v) for k, v in d["spaces"].items()})
    if t == "Discrete":
        return gym.spaces.Discrete(d["n"])
    if t == "Box":
        def bound(v, inf):
            a = np.array(v, dtype=object)
            return np.where(a == None, inf, a).astype(d["dtype"])  # noqa: E711  null = inf
        return gym.spaces.Box(bound(d["low"], -np.inf), bound(d["high"], np.inf),
                              tuple(d["shape"]), np.dtype(d["dtype"]))
    raise ValueError(f"unsupported space {t}")


class RemoteEnv(gym.Env):
    """A gymnasium Env backed by a simulator server with one environment.

    The server should run with ``auto_reset: false`` (``hm3d sim --serve --no-auto-reset``)
    so that the episode ends exactly as the training loop expects; reset(options=...)
    passes map_id / task_idx / start + goal through."""

    def __init__(self, address: str = "tcp://127.0.0.1:5555", timeout_s: float = 30.0):
        self.client = SimClient(address, timeout_s)
        info = self.client.info()
        if info["num_envs"] != 1:
            raise ValueError(f"RemoteEnv needs a server with one environment, this one has {info['num_envs']}")
        self.server_info = info
        self.observation_space = space_from(info["observation_space"])
        self.action_space = space_from(info["action_space"])

    def _obs(self, o):
        sp = self.observation_space
        if isinstance(sp, gym.spaces.Dict):
            return {k: np.asarray(o[k], dtype=sp[k].dtype) for k in sp.spaces}
        return np.asarray(o, dtype=sp.dtype)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        obs, info = self.client.reset(0, options)
        return self._obs(obs[0]), info[0]

    def step(self, action):
        a = np.asarray(action)
        rep = self.client.step([a.tolist()])
        info = rep["info"][0]
        if rep["reset_info"][0] is not None:
            info = {**info, "reset_info": rep["reset_info"][0]}
        return (self._obs(rep["obs"][0]), float(rep["reward"][0]), rep["terminated"][0],
                rep["truncated"][0], info)

    def close(self):
        self.client.close()
