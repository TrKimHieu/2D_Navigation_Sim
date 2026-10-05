"""Run an agent in a session (``hm3d sim`` without --serve): oracle, random or your own
``module:function`` (``function(env)`` returns ``policy(obs) -> action``, as for hm3d eval)."""

from __future__ import annotations

import math
import time

import numpy as np

from ..evaluate import _agent_factory
from .session import Session


def _line(i, last) -> str:
    spl = last.get("spl")
    return (f"env {i}  {last.get('map_id')}  task {last.get('task_id')}: {last.get('termination')}"
            f"  steps {last.get('steps')}  return {last.get('return', 0):.2f}"
            + (f"  SPL {spl:.3f}" if isinstance(spl, float) else ""))


def run_agent(session: Session, agent: str = "oracle", episodes: int | None = None,
              max_steps: int | None = None, fps: float | None = None, log=print) -> dict:
    """Play until `episodes` episodes have finished (0 / None: until Ctrl+C) or
    `max_steps` steps; `fps` limits the speed (for watching). Returns a summary.
    ``session.run_status`` follows the run (the viewer shows it): state "running",
    then "finished" (target reached), "stopped" (Ctrl+C) or "error"."""
    envs = session.local_envs
    if envs is None:
        raise ValueError("running an agent needs the environments in this process "
                         "(vectorization: sync)")
    make_policy = _agent_factory(agent)
    obs, _ = session.reset()
    policies = [make_policy(e) for e in envs]
    done_eps, steps, results = 0, 0, []
    status = session.run_status = {"state": "running", "agent": agent if isinstance(agent, str)
                                   else "custom", "episodes": 0, "target": episodes or None,
                                   "max_steps": max_steps, "steps": 0, "success": None, "spl": None}
    period = 1.0 / fps if fps else 0.0
    t_next = time.perf_counter()
    try:
        while (not episodes or done_eps < episodes) and (not max_steps or steps < max_steps):
            actions = np.stack([np.asarray(policies[i](obs[i])) for i in range(session.num_envs)])
            out = session.step(actions)
            steps += 1
            status["steps"] = steps
            for i, r in enumerate(out):
                obs[i] = r["obs"]
                if r["terminated"] or r["truncated"]:
                    done_eps += 1
                    last = session.stats[i]["last"]
                    results.append(last)
                    status.update(episodes=done_eps, **_summary(results))
                    log(_line(i, last))
                    if not session.cfg["auto_reset"]:
                        o, _ = session.reset(i)
                        obs[i] = o[0]
                    policies[i] = make_policy(envs[i])
            if period:
                t_next += period
                time.sleep(max(0.0, t_next - time.perf_counter()))
    except KeyboardInterrupt:
        status["state"] = "stopped"
        log("stopped")
    except Exception:
        status["state"] = "error"
        raise
    else:
        status["state"] = "finished"
    return {"episodes": len(results), "steps": steps, **_summary(results)}


def _summary(results) -> dict:
    ok = [bool(r.get("success")) for r in results]
    spls = [r["spl"] for r in results if isinstance(r.get("spl"), float) and math.isfinite(r["spl"])]
    return {"success": float(np.mean(ok)) if ok else None,
            "spl": float(np.mean(spls)) if spls else None}
