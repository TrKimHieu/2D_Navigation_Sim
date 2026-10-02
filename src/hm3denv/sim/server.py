"""ZeroMQ server of a simulation session (``hm3d sim --serve``): a REP socket, one JSON
request -> one JSON reply. Any language with ZeroMQ can drive the simulator.

Requests (``env`` is optional: all environments when absent):

    {"cmd": "info"}                                   -> spaces, robot, maps, num_envs, ...
    {"cmd": "reset", "env": 0, "options": {...}}      -> {"obs": [...], "info": [...]}
        options (optional, replaces the episode schedule): {"map_id": M, "task_idx": I}
        or {"map_id": M, "start": [x, y, theta], "goal": [x, y]}  (grid: [row, col] cells)
    {"cmd": "step", "actions": [a_0, a_1, ...]}       -> {"obs", "reward", "terminated",
                                                          "truncated", "info"}: one entry per env
        (a single action is accepted with one environment). With auto_reset an env whose
        episode ended has already started the next one: its obs is the new first
        observation, info holds the finished episode's metrics, reset_info the new one's.
    {"cmd": "render", "env": 0}                       -> {"svg": "..."} or {"png_base64": "..."}
    {"cmd": "stats"}                                  -> episodes, steps, return, last episode
    {"cmd": "close"}                                  -> stops the server

Every reply has "ok": true, or "ok": false and "error". Numbers that are infinite or NaN
are sent as null.
"""

from __future__ import annotations

import json
import logging
import time

from .session import Session, SimError, to_jsonable

log = logging.getLogger(__name__)


def _zmq():
    try:
        import zmq
    except ImportError:
        raise ImportError('the simulator server needs pyzmq: pip install "hm3denv[sim]" '
                          "(sim.bat / ./sim.sh install it)") from None
    return zmq


class SimServer:
    def __init__(self, session: Session, bind: str = "tcp://127.0.0.1:5555"):
        zmq = _zmq()
        self.session, self.bind = session, bind
        self.ctx = zmq.Context.instance()
        self.sock = self.ctx.socket(zmq.REP)
        self.sock.setsockopt(zmq.LINGER, 0)
        try:
            self.sock.bind(bind)
        except zmq.ZMQError as e:
            self.sock.close()
            raise OSError(f"cannot listen on {bind} ({e}); is another simulator running? "
                          "Choose another address with --serve tcp://127.0.0.1:5556") from None
        self.requests = 0
        self._stop = False

    def handle(self, req) -> dict:
        """The reply to one request (also usable without a socket, e.g. in tests)."""
        s = self.session
        try:
            if not isinstance(req, dict) or "cmd" not in req:
                raise SimError('a request is a JSON object with "cmd"')
            cmd, env = req["cmd"], req.get("env")
            if cmd == "info":
                return {"ok": True, **to_jsonable(s.describe())}
            if cmd == "reset":
                obs, info = s.reset(env, req.get("options"))
                return {"ok": True, "obs": to_jsonable(obs), "info": to_jsonable(info)}
            if cmd == "step":
                if "actions" not in req:
                    raise SimError('step needs "actions"')
                out = s.step(req["actions"])
                return {"ok": True, **{k: to_jsonable([r[k] for r in out])
                                       for k in ("obs", "reward", "terminated", "truncated", "info")},
                        "reset_info": to_jsonable([r.get("reset_info") for r in out])}
            if cmd == "render":
                return {"ok": True, **s.render_json(0 if env is None else env)}
            if cmd == "stats":
                return {"ok": True, "stats": to_jsonable(s.stats)}
            if cmd == "close":
                self._stop = True
                return {"ok": True}
            raise SimError(f"unknown cmd {cmd!r}; use info, reset, step, render, stats, close")
        except (SimError, ValueError, KeyError, TypeError) as e:
            return {"ok": False, "error": f"{type(e).__name__}: {e}" if not isinstance(e, SimError) else str(e)}

    def serve_forever(self, poll_ms: int = 200):
        """Answer requests until a "close" request (or Ctrl+C)."""
        zmq = _zmq()
        poller = zmq.Poller()
        poller.register(self.sock, zmq.POLLIN)
        t0 = time.monotonic()
        try:
            while not self._stop:
                if not poller.poll(poll_ms):      # short polls keep Ctrl+C working on Windows
                    continue
                raw = self.sock.recv()
                try:
                    req = json.loads(raw)
                except ValueError:
                    rep = {"ok": False, "error": "the request is not valid JSON"}
                else:
                    rep = self.handle(req)
                self.sock.send(json.dumps(rep, allow_nan=False).encode())
                self.requests += 1
        except KeyboardInterrupt:
            pass
        finally:
            self.sock.close()
            log.info("server stopped after %d requests in %.0f s", self.requests, time.monotonic() - t0)
