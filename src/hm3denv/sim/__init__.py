"""Simulation sessions: open a simulation from a config file or command-line options, drive
it step by step from Python, from another process over ZeroMQ, or watch it in the browser.

    hm3d sim my_sim.yaml --view               # run an agent and watch it
    hm3d sim my_sim.yaml --serve              # let another program drive it (see sim.server)

    from hm3denv.sim import Session, load_sim_config
    with Session(load_sim_config("my_sim.yaml", num_envs=4)) as s:
        obs, infos = s.reset()
        out = s.step(actions)
"""

from .client import RemoteEnv, SimClient, SimRemoteError  # noqa: F401
from .config import SimConfigError, load_sim_config  # noqa: F401
from .session import Session, SimError, to_jsonable  # noqa: F401
