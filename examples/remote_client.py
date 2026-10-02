"""Drive a simulation that runs in another process (or on another machine) over ZeroMQ.

1. Start the simulator (it waits for requests):

       .\\sim.bat demo --serve                 (Linux/macOS: ./sim.sh demo --serve)

2. In a second terminal, run this client:

       .venv\\Scripts\\python examples\\remote_client.py       (Linux/macOS: .venv/bin/python ...)

It sends actions and receives observations, rewards and episode results in a loop.
Replace `policy` with your controller. The protocol is plain JSON (see
hm3denv/sim/server.py), so a client in C++, MATLAB, ROS, ... works the same way.
"""

import argparse
import math

from hm3denv.sim import SimClient


def policy(obs, action_dim):
    """Turn towards the goal and drive (a naive controller: it does not avoid walls)."""
    dist, sin_b, cos_b = obs["goal"]
    bearing = math.atan2(sin_b, cos_b)
    v = 0.5 if abs(bearing) < 0.5 else 0.0
    w = max(-1.0, min(1.0, 1.5 * bearing))
    return [v, w] if action_dim == 2 else [v, 0.0, w]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--address", default="tcp://127.0.0.1:5555")
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--stop-server", action="store_true", help="stop the simulator at the end")
    a = p.parse_args()

    with SimClient(a.address) as sim:
        info = sim.info()
        n, act = info["num_envs"], info["action_space"]
        print(f"connected: {info['dataset']} / {info['robot']} / {info['env_id']}, {n} env(s), "
              f"action space {act}")
        if act["type"] != "Box":
            raise SystemExit("this example drives the continuous environment (HM3D/Svg-v0)")
        obs, _ = sim.reset()
        for t in range(a.steps):
            rep = sim.step([policy(o, act["shape"][0]) for o in obs])
            obs = rep["obs"]                          # with auto_reset: already the next episode
            for i in range(n):
                if rep["terminated"][i] or rep["truncated"][i]:
                    e = rep["info"][i]
                    print(f"step {t}: env {i} {e['map_id']} task {e['task_id']}: {e['termination']}, "
                          f"SPL {e['spl']:.2f}")
        print("stats:", [(s["episodes"], round(s["return"], 2)) for s in sim.stats()])
        if a.stop_server:
            sim.close(stop_server=True)


if __name__ == "__main__":
    main()
