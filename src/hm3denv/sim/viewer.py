"""Watch a simulation session in the browser (``hm3d sim --view``): the map, the robot,
its path, start and goal of every environment, refreshed several times per second. Runs
in a background thread next to the simulation or the ZeroMQ server; standard library
only, so it also works on a server (open the page through an SSH tunnel).
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from .session import Session, to_jsonable

PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>hm3denv simulation</title>
<style>
 :root { color-scheme: light dark; --bg:#fafafa; --fg:#222; --muted:#666; --card:#fff; --line:#ddd; }
 @media (prefers-color-scheme: dark) { :root { --bg:#16181c; --fg:#e8e8e8; --muted:#9aa; --card:#1f2228; --line:#333; } }
 body { margin:0; font:14px/1.4 system-ui, sans-serif; background:var(--bg); color:var(--fg); }
 header { padding:10px 16px; border-bottom:1px solid var(--line); display:flex; gap:16px; flex-wrap:wrap; align-items:center; }
 header b { font-size:16px; } header span { color:var(--muted); }
 main { padding:12px 16px; display:grid; gap:12px; grid-template-columns:minmax(0,1fr) 280px; }
 @media (max-width: 800px) { main { grid-template-columns:1fr; } }
 #view { background:#fff; border:1px solid var(--line); border-radius:6px; min-height:300px; display:flex; align-items:center; justify-content:center; overflow:hidden; }
 #view svg, #view img { width:100%; height:auto; max-height:80vh; image-rendering:pixelated; }
 aside { background:var(--card); border:1px solid var(--line); border-radius:6px; padding:10px 12px; }
 table { border-collapse:collapse; width:100%; } td { padding:2px 4px; vertical-align:top; } td:first-child { color:var(--muted); }
 select { font:inherit; }
 #run { padding:2px 10px; border-radius:12px; background:var(--line); }
 #run:empty { display:none; }
 #run.finished { background:#2e7d32; color:#fff; } #run.stopped, #run.error { background:#b3261e; color:#fff; }
</style></head><body>
<header><b>hm3denv</b><span id="title"></span>
 <label>environment <select id="env"></select></label><span id="run"></span><span id="age"></span></header>
<main><div id="view">loading...</div><aside><table id="stats"></table></aside></main>
<script>
const $ = (id) => document.getElementById(id);
let env = 0, busy = false;
async function info() {
  const d = await (await fetch('/info')).json();
  $('title').textContent = `${d.dataset} · ${d.robot} · ${d.env_id} · ${d.num_envs} env(s)`;
  $('env').innerHTML = [...Array(d.num_envs).keys()].map(i => `<option value="${i}">${i}</option>`).join('');
  $('env').onchange = (e) => { env = +e.target.value; };
}
function row(k, v) { return `<tr><td>${k}</td><td>${v ?? '–'}</td></tr>`; }
function fmt(v) { return typeof v === 'number' ? (Number.isInteger(v) ? v : v.toFixed(3)) : v; }
function pct(v) { return v == null ? '–' : `${Math.round(100 * v)} %`; }
function runText(r) {
  const n = r.target ? `${r.episodes}/${r.target} episodes` : `${r.episodes} episode(s)`;
  const res = r.episodes ? ` · success ${pct(r.success)} · SPL ${(r.spl ?? 0).toFixed(3)}` : '';
  if (r.state === 'running') return `running · ${n}${res}` + (r.target || r.max_steps ? '' : ' · Ctrl+C in the terminal to stop');
  if (r.state === 'finished') return `finished · ${n}${res} · environments still on an episode stay where they are; Ctrl+C in the terminal to quit`;
  if (r.state === 'stopped') return `stopped · ${n}${res}`;
  return 'error: see the terminal';
}
async function tick() {
  if (busy) return; busy = true;
  try {
    const r = await fetch(`/frame?env=${env}`);
    const type = r.headers.get('content-type') || '';
    if (type.includes('svg')) { $('view').innerHTML = await r.text(); }
    else { const b = await r.blob(); $('view').innerHTML = `<img alt="grid" src="${URL.createObjectURL(b)}">`; }
    const s = (await (await fetch('/stats')).json())[env];
    const c = s.current || {}, l = s.last || {};
    $('stats').innerHTML = row('map', c.map_id) + row('task', c.task_id) + row('geodesic (m)', fmt(c.geodesic_m))
      + row('steps', s.steps) + row('return', fmt(s.return)) + row('episodes done', s.episodes)
      + row('last episode', l.termination ? `${l.termination}, SPL ${fmt(l.spl)}` : null);
    const run = await (await fetch('/status')).json();
    $('run').textContent = run ? runText(run) : '';
    $('run').className = run ? run.state : '';
    $('age').textContent = '';
  } catch (e) { $('age').textContent = 'simulation stopped'; }
  busy = false;
}
info(); setInterval(tick, 250);
</script></body></html>
"""


class _Handler(BaseHTTPRequestHandler):
    session: Session = None

    def log_message(self, *args):             # quiet
        pass

    def _send(self, body: bytes, ctype: str, code: int = 200):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        try:
            if u.path == "/":
                self._send(PAGE.encode(), "text/html; charset=utf-8")
            elif u.path == "/info":
                self._send(json.dumps(to_jsonable(self.session.describe())).encode(), "application/json")
            elif u.path == "/status":
                self._send(json.dumps(to_jsonable(self.session.run_status)).encode(), "application/json")
            elif u.path == "/stats":
                self._send(json.dumps(to_jsonable(self.session.stats)).encode(), "application/json")
            elif u.path == "/frame":
                kind, data = self.session.render(int(q.get("env", ["0"])[0]))
                if kind == "svg":
                    self._send(data.encode(), "image/svg+xml")
                else:
                    self._send(data, "image/png")
            else:
                self._send(b"not found", "text/plain", 404)
        except Exception as e:                   # noqa: BLE001  report, keep serving
            self._send(f"{type(e).__name__}: {e}".encode(), "text/plain", 500)


class Viewer:
    """``Viewer(session, port=8770).start()``; ``.url`` is the page to open."""

    def __init__(self, session: Session, host: str = "127.0.0.1", port: int = 8770):
        handler = type("Handler", (_Handler,), {"session": session})
        try:
            self.httpd = ThreadingHTTPServer((host, port), handler)
        except OSError as e:
            raise OSError(f"cannot open the viewer on {host}:{port} ({e}); choose another port "
                          "with --view PORT") from None
        self.httpd.daemon_threads = True
        self.url = f"http://{host}:{self.httpd.server_address[1]}/"
        self._thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    def start(self, open_browser: bool = False) -> "Viewer":
        self._thread.start()
        if open_browser:
            import webbrowser
            webbrowser.open(self.url)
        return self

    def stop(self):
        self.httpd.shutdown()
        self.httpd.server_close()
