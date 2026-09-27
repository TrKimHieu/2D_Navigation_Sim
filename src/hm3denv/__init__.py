"""hm3denv: physically valid indoor navigation environments built from HM3D floors.

    import gymnasium as gym, hm3denv
    env = gym.make("HM3D/Svg-v0", dataset="svg-v1", robot="turtlebot4", split="train")
    env = gym.make("HM3D/Grid-v0", dataset="grid-jetauto-v1", split="train")

    ds = hm3denv.load_dataset("svg-v1")
    venv = hm3denv.make_vec("HM3D/Svg-v0", 16, dataset="svg-v1", robot="turtlebot4")
"""

from gymnasium.envs.registration import register

__version__ = "0.8.0"

register(id="HM3D/Svg-v0", entry_point="hm3denv.envs.svg:SvgEnv")
register(id="HM3D/Grid-v0", entry_point="hm3denv.envs.grid:GridEnv")

from . import robots  # noqa: E402,F401
from .dataset import Dataset, list_datasets, load_dataset  # noqa: E402,F401
from .vector import env_fn, make_vec  # noqa: E402,F401
