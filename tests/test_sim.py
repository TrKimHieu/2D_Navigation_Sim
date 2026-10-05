"""Simulation sessions (hm3denv.sim): config, scheduling, auto-reset, ZeroMQ server and
client, browser viewer, `hm3d sim`."""

import json
import shutil
import subprocess
import threading
import urllib.request

import numpy as np
import pytest

from hm3denv.cli import main
from hm3denv.sim import Session, SimConfigError, SimError, load_sim_config
from hm3denv.sim.server import SimServer

M0 = "demo-S001_s0"


def session(**kw):
    kw.setdefault("vectorization", "sync")
    return Session(load_sim_config(kw))


# ------------------------------------------------------------------ config

@pytest.mark.parametrize("cfg, msg", [
    ({"dataset": "demo-svg", "nope": 1}, "unknown key"),
    ({"num_envs": 0}, "num_envs"),
    ({"split": "dev"}, "split"),
    ({"episodes": [{"map": M0, "start": [1, 2]}]}, "both start and goal"),
    ({"episodes": [{"map": M0, "task_idx": 1, "start": [1, 2], "goal": [3, 4]}]}, "either task_idx"),
    ({"episodes": [{"mapp": M0}]}, "unknown key"),
    ({"server": {"port": 1}}, "'server' takes"),
    ({"env": {"robot": "x"}}, "top level"),
])
def test_config_errors(cfg, msg):
    with pytest.raises(SimConfigError, match=msg):
        load_sim_config(cfg)


def test_config_file_and_overrides(tmp_path):
    p = tmp_path / "s.yaml"
    p.write_text("dataset: demo-grid\nnum_envs: 3\nepisodes:\n  - {map: demo-S001_s0, task: 2}\n")
    cfg = load_sim_config(p, num_envs=2, robot=None)
    assert cfg["dataset"] == "demo-grid" and cfg["num_envs"] == 2
    assert cfg["episodes"] == [{"map": M0, "task_idx": 2}]
    assert cfg["server"]["bind"].startswith("tcp://")
    with pytest.raises(SimConfigError, match="not found"):
        load_sim_config(tmp_path / "missing.yaml")


def test_packaged_session_configs_load_by_name():
    from hm3denv.sim.config import packaged_configs
    assert {"demo", "custom_pairs"} <= set(packaged_configs())
    for name in packaged_configs():
        Session(load_sim_config(name, vectorization="sync")).close()
    assert load_sim_config("demo.yaml")["num_envs"] == 2
    with pytest.raises(SimConfigError, match="not found"):
        load_sim_config("dmeo")


# ------------------------------------------------------------------ session

def test_session_steps_all_envs_and_auto_resets():
    with session(dataset="demo-svg", num_envs=2, env={"time_limit": 0.3}) as s:
        obs, infos = s.reset()
        assert len(obs) == 2 and obs[0]["lidar"].shape == s.observation_space["lidar"].shape
        for _ in range(3):
            out = s.step(np.zeros((2, 2)))
        assert all(r["truncated"] for r in out)                    # 0.3 s = 3 steps
        assert all(r["reset_info"]["map_id"] for r in out)          # next episode started
        assert s.stats[0]["episodes"] == 1 and s.stats[0]["last"]["termination"] == "time_limit"
        assert s.step(np.zeros((2, 2)))[0]["info"]["time"] == pytest.approx(0.1)


def test_session_plays_fixed_episodes_in_turn():
    eps = [{"map": M0, "task_idx": 3}, {"map": M0, "start": [2.48, 12.36], "goal": [4.84, 10.72]}]
    with session(dataset="demo-svg", episodes=eps) as s:
        _, i0 = s.reset()
        _, i1 = s.reset()
        _, i2 = s.reset()
    assert (i0[0]["task_id"], i1[0]["task_id"], i2[0]["task_id"]) == (3, -1, 3)


def test_session_invalid_custom_episode_keeps_env_usable():
    with session(dataset="demo-svg") as s:
        with pytest.raises(SimError, match="collides"):
            s.reset(options={"map_id": M0, "start": [0, 0], "goal": [4.84, 10.72]})
        assert s.stats[0]["current"]["map_id"] == M0                 # a task of the map instead
        s.step([0.0, 0.0])


def test_session_errors():
    with pytest.raises(SimError, match="not in"):
        session(dataset="demo-svg", maps=["nope_s0"])
    with pytest.raises(SimError, match="robot"):
        session(dataset="demo-svg", robot="kobuki")
    with session(dataset="demo-svg", num_envs=2) as s:
        with pytest.raises(SimError, match="expected 2 action"):
            s.step([0.0, 0.0])
        with pytest.raises(SimError, match="env must be"):
            s.reset(env=5)
    with session(dataset="demo-svg", auto_reset=False, env={"time_limit": 0.1}) as s:
        s.step([0.0, 0.0])
        with pytest.raises(SimError, match="reset them first"):
            s.step([0.0, 0.0])


def test_session_render_and_maps_filter():
    with session(dataset="demo-grid", maps=[M0]) as s:
        _, infos = s.reset()
        assert infos[0]["map_id"] == M0
        kind, png = s.render(0)
        assert kind == "png" and png[:4] == b"\x89PNG"
    with session(dataset="demo-svg") as s:
        kind, svg = s.render(0)
        assert kind == "svg" and svg.lstrip().startswith("<")


def test_session_async_workers():
    with Session(load_sim_config({"dataset": "demo-svg", "num_envs": 2})) as s:
        s.reset(env=1, options={"map_id": M0, "task_idx": 2})
        out = s.step(np.zeros((2, 2)))
        assert len(out) == 2 and s.stats[1]["current"]["task_id"] == 2
        assert s.render(1)[0] == "svg"


# ------------------------------------------------------------------ server, client, viewer

def test_server_handle_without_socket():
    pytest.importorskip("zmq")
    with session(dataset="demo-svg", num_envs=2) as s:
        srv = SimServer(s, "tcp://127.0.0.1:0")
        try:
            info = srv.handle({"cmd": "info"})
            assert info["ok"] and info["action_space"]["shape"] == [2] and info["num_envs"] == 2
            json.dumps(info, allow_nan=False)                         # plain JSON, no inf
            r = srv.handle({"cmd": "reset", "env": 1, "options": {"map_id": M0, "task_idx": 0}})
            assert r["ok"] and r["info"][0]["task_id"] == 0
            r = srv.handle({"cmd": "step", "actions": [[0.5, 0.0], [0.5, 0.0]]})
            assert r["ok"] and len(r["obs"]) == 2 and isinstance(r["reward"][0], float)
            assert srv.handle({"cmd": "step", "actions": [[1, 2, 3]]})["ok"] is False
            assert "unknown cmd" in srv.handle({"cmd": "fly"})["error"]
            assert "svg" in srv.handle({"cmd": "render", "env": 1})
            assert srv.handle({"cmd": "close"})["ok"] and srv._stop
        finally:
            srv.sock.close()


def test_zeromq_round_trip_and_remote_env():
    pytest.importorskip("zmq")
    from hm3denv.sim import RemoteEnv, SimClient
    s = session(dataset="demo-svg", auto_reset=False, env={"time_limit": 0.5})
    srv = SimServer(s, "tcp://127.0.0.1:*")
    addr = srv.sock.getsockopt_string(__import__("zmq").LAST_ENDPOINT)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    try:
        env = RemoteEnv(addr, timeout_s=20)
        obs, info = env.reset(options={"map_id": M0, "task_idx": 1})
        assert env.observation_space.contains(obs) and info["task_id"] == 1
        done = False
        while not done:
            obs, r, term, trunc, info = env.step(env.action_space.sample())
            done = term or trunc
        assert info["termination"] in ("time_limit", "collision", "success")
        with SimClient(addr) as c:
            with pytest.raises(Exception, match="reset them first"):
                c.step([[0, 0]])
            assert c.stats()[0]["episodes"] == 1
            c.close(stop_server=True)
        t.join(10)
        assert not t.is_alive()
    finally:
        s.close()


def test_viewer_serves_page_and_frames():
    from hm3denv.sim.viewer import Viewer
    with session(dataset="demo-svg") as s:
        v = Viewer(s, port=0).start()
        try:
            get = lambda p: urllib.request.urlopen(v.url + p, timeout=20)  # noqa: E731
            assert b"hm3denv" in get("").read()
            assert json.loads(get("info").read())["robot"] == "turtlebot4"
            r = get("frame?env=0")
            assert r.headers["Content-Type"] == "image/svg+xml"
            assert json.loads(get("stats").read())[0]["current"]["map_id"]
        finally:
            v.stop()


def test_viewer_reports_the_run_status():
    from hm3denv.sim.run import run_agent
    from hm3denv.sim.viewer import Viewer
    with session(dataset="demo-svg") as s:
        v = Viewer(s, port=0).start()
        try:
            status = lambda: json.loads(urllib.request.urlopen(v.url + "status", timeout=20).read())  # noqa: E731
            assert status() is None                              # nothing runs an agent yet
            run_agent(s, "oracle", episodes=1, log=lambda *_: None)
            st = status()
            assert st["state"] == "finished" and st["episodes"] == 1 and st["target"] == 1
            assert st["success"] == 1.0
        finally:
            v.stop()


@pytest.mark.skipif(shutil.which("node") is None, reason="needs Node.js")
def test_viewer_page_script_parses(tmp_path):
    """The page's JavaScript is not run by the other tests: a syntax error there leaves the
    viewer on "loading..." with no other sign."""
    from hm3denv.sim.viewer import PAGE
    js = tmp_path / "viewer.js"
    js.write_text(PAGE.split("<script>")[1].split("</script>")[0], encoding="utf-8")
    r = subprocess.run(["node", "--check", str(js)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


# ------------------------------------------------------------------ CLI

def test_cli_sim_runs_fixed_episodes(capsys):
    assert main(["sim", "--dataset", "demo-svg", "--map", M0, "--start", "2.48,12.36,0",
                 "--goal", "4.84,10.72"]) == 0
    out = capsys.readouterr().out
    assert "task -1: success" in out and '"success": 1.0' in out


def test_cli_sim_view_plays_until_stopped(capsys):
    """With --view and no fixed episodes the agent keeps playing (default --episodes 0):
    here it is stopped by --steps, past the 5 episodes (794 steps) of the default without
    --view."""
    assert main(["sim", "--dataset", "demo-svg", "--view", "0", "--no-browser", "--no-wait",
                 "--steps", "1200", "--fps", "100000"]) == 0
    summary = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert summary["steps"] == 1200 and summary["episodes"] > 5


def test_cli_sim_reports_bad_start(capsys):
    assert main(["sim", "--dataset", "demo-svg", "--map", M0, "--start", "0,0", "--goal", "4.84,10.72"]) == 2
    assert "collides" in capsys.readouterr().err


# ------------------------------------------------------------------ training factories

def test_env_fns_apply_the_schedule_on_plain_resets():
    from hm3denv.sim.session import env_fns
    eps = [{"map": M0, "task_idx": 4}, {"map": M0, "start": [2.48, 12.36], "goal": [4.84, 10.72]}]
    cfg = load_sim_config({"dataset": "demo-svg", "episodes": eps, "env": {"n_beams": 36}})
    e0, e1 = (f() for f in env_fns(cfg, 2, split="train"))
    assert e0.observation_space["lidar"].shape == (36,)
    assert [e0.reset()[1]["task_id"] for _ in range(3)] == [4, -1, 4]
    assert e1.reset()[1]["task_id"] == -1                  # env i starts at entry i
    assert e0.reset(options={"map_id": M0, "task_idx": 0})[1]["task_id"] == 0   # explicit wins
    cfg = load_sim_config({"dataset": "demo-svg", "maps": [M0]})
    env = env_fns(cfg, 1)[0]()
    assert {env.reset()[1]["map_id"] for _ in range(4)} == {M0}
