#!/usr/bin/env python3
"""mk_top: live view of the agents a --parallel mk_server is serving (tok/s, context, task, latest output).

  python3 tools/mk_top.py [--url http://127.0.0.1:18080] [--title "4 agents, one RTX 3090"]
  python3 tools/mk_top.py --run 4    (also starts the server unless one answers, and 4 open-jet agents on Slipstream
                                     eval tasks; stops what it started on exit)
  python3 tools/mk_top.py --demo     (simulated agents, labelled SIMULATED: for trying the layout)

Polls GET /slots of the server and nvidia-smi. Tasks come from each request's metadata.task, else the first line
of its first user message. Ctrl-C quits.
"""
import argparse
import json
import os
import random
import re
import signal
import subprocess
import sys
import tempfile
import threading
import time
import urllib.parse
import urllib.request

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

HERE = os.path.dirname(os.path.abspath(__file__))
COLORS = ["#4FC3F7", "#FF8A65", "#AED581", "#BA68C8", "#FFD54F", "#4DB6AC", "#F06292", "#90A4AE"]
HIST = 48  # samples of tok/s history per sparkline
TICK = 0.5  # seconds per history sample

DIGITS = {  # 3-row block font
    "0": ("█▀█", "█ █", "▀▀▀"), "1": ("▀█ ", " █ ", "▀▀▀"), "2": ("▀▀█", "█▀▀", "▀▀▀"), "3": ("▀▀█", " ▀█", "▀▀▀"),
    "4": ("█ █", "▀▀█", "  ▀"), "5": ("█▀▀", "▀▀█", "▀▀▀"), "6": ("█▀▀", "█▀█", "▀▀▀"), "7": ("▀▀█", "  █", "  ▀"),
    "8": ("█▀█", "█▀█", "▀▀▀"), "9": ("█▀█", "▀▀█", "▀▀▀"), " ": ("   ", "   ", "   "),
}
BARS = " ▁▂▃▄▅▆▇█"


def big(n, width=3):
    s = str(int(round(n))).rjust(width)
    return [" ".join(DIGITS[c][r] for c in s) for r in range(3)]


def spark(values, top, rows=3):
    """rows-high sparkline of values scaled to top"""
    lines = []
    for r in range(rows - 1, -1, -1):
        line = ""
        for v in values:
            lvl = max(0.0, min(1.0, v / top)) * rows * 8 if top > 0 else 0
            line += BARS[int(max(0, min(8, lvl - r * 8)))]
        lines.append(line)
    return lines


def kfmt(n):
    return f"{n / 1000:.1f}k" if n >= 1000 else str(n)


def ctx_bar(used, total, width):
    f = 0 if total <= 0 else min(1.0, used / total)
    full = int(f * width)
    return "█" * full + "░" * (width - full)


class Source:
    """polls the server's /slots and nvidia-smi in background threads"""

    def __init__(self, url):
        self.url = url.rstrip("/") + "/slots"
        self.data, self.gpu, self.error = None, None, None
        threading.Thread(target=self._poll, daemon=True).start()
        threading.Thread(target=self._gpu, daemon=True).start()

    def _poll(self):
        while True:
            try:
                with urllib.request.urlopen(self.url, timeout=2) as r:
                    self.data, self.error = json.loads(r.read()), None
            except Exception as e:
                self.error = str(e)
            time.sleep(0.25)

    def _gpu(self):
        q = "name,utilization.gpu,power.draw,memory.used,memory.total,temperature.gpu"
        while True:
            try:
                out = subprocess.run(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader,nounits"],
                                     capture_output=True, text=True, timeout=5).stdout.splitlines()[0]
                name, util, power, used, total, temp = [x.strip() for x in out.split(",")]
                self.gpu = {"name": name.replace("NVIDIA GeForce ", ""), "util": float(util), "power": float(power),
                            "mem": float(used) / 1024, "mem_total": float(total) / 1024, "temp": float(temp)}
            except Exception:
                self.gpu = None
            time.sleep(1.0)


def server_up(url):
    try:
        with urllib.request.urlopen(url.rstrip("/") + "/health", timeout=2):
            return True
    except Exception:
        return False


class Runner:
    """--run N: starts the server (unless one answers at url) and N agents (tools/run_agents.py) in the background,
    collects the agents' task results; stop() ends what it started"""

    def __init__(self, url, n, model, tasks_each):
        self.status, self.results, self.procs = "", [], []
        self.logdir = tempfile.mkdtemp(prefix="mk_top_")
        threading.Thread(target=self._run, args=(url, n, model, tasks_each), daemon=True).start()

    def _spawn(self, cmd, log, **kw):
        p = subprocess.Popen(cmd, stderr=log, start_new_session=True, **kw)
        self.procs.append(p)
        return p

    def _run(self, url, n, model, tasks_each):
        if not server_up(url):
            log = open(os.path.join(self.logdir, "server.log"), "w")
            self.status = f"loading the model (1-4 min) · log {log.name}"
            srv = self._spawn([sys.executable, os.path.join(HERE, "..", "server", "mk_server.py"), "-m", model,
                               "--agents", str(n), "--port", str(urllib.parse.urlparse(url).port or 18080)], log,
                              stdout=log)
            while not server_up(url):
                if srv.poll() is not None:
                    self.status = f"the server exited · see {log.name}"
                    return
                time.sleep(1)
        log = open(os.path.join(self.logdir, "agents.log"), "w")
        self.status = f"agents running · log {log.name}"
        p = self._spawn([sys.executable, os.path.join(HERE, "run_agents.py"), "-n", str(n), "--tasks-each", str(tasks_each),
                         "--url", url], log, stdout=subprocess.PIPE, text=True)
        logs = log.name
        for line in p.stdout:
            if line.startswith("agent ") and "{" in line:
                try:
                    res = json.loads(line[line.index("{"):])
                except ValueError:  # (other output of an agent)
                    continue
                if isinstance(res, dict) and "task" in res:
                    self.results.append(res)
            elif "workspaces and logs in" in line:
                logs = line.rsplit(" in ", 1)[1].strip()
        self.status = "agents finished" if self.results else f"the agents exited without finishing a task · logs in {logs}"

    def stop(self):
        for p in reversed(self.procs):
            if p.poll() is None:
                os.killpg(p.pid, signal.SIGTERM)
                p.wait()


class Demo:
    """simulated agents (rates follow the 4-slot server's measured scaling), for the layout only"""

    TASKS = ["Add rate limiting to the /login endpoint", "Fix the race in the job queue's retry path",
             "Write property tests for the CSV parser", "Refactor auth middleware to async",
             "Port the config loader to pydantic v2", "Find why safe_join accepts '..' segments"]
    TEXT = ("Let me look at how the handler builds the path first. def safe_join(root, *parts): resolved = "
            "os.path.realpath(os.path.join(root, *parts)) if not resolved.startswith(root + os.sep): raise "
            "ValueError('path escapes root') return resolved The tests pass now; next I'll cover the symlink case. ")
    PER_SLOT = {1: 110, 2: 98, 3: 87, 4: 78}

    def __init__(self, n):
        self.error, self.t0 = None, time.time()
        self.slots = [{"id": i, "state": "idle", "ctx": 0, "tok_s": 0.0, "prompt": 0, "reused": 0, "prefill_left": 0,
                       "generated": 0, "task": self.TASKS[i % len(self.TASKS)], "agent": "", "tail": "",
                       "idle_s": 0.0, "_until": random.uniform(0.5, 3), "_pos": random.randrange(len(self.TEXT))}
                      for i in range(n)]
        self.total, self.last = 0, time.time()
        self.gpu = {"name": "RTX 3090", "util": 0, "power": 0, "mem": 23.1, "mem_total": 24.0, "temp": 61}

    @property
    def data(self):
        now = time.time()
        dt, self.last = now - self.last, now
        active = sum(s["state"] == "decode" for s in self.slots)
        for s in self.slots:
            s["_until"] -= dt
            if s["state"] == "decode":
                s["tok_s"] = self.PER_SLOT.get(active, 78) * random.uniform(0.85, 1.12)
                k = int(s["tok_s"] * dt)
                s["generated"] += k
                s["ctx"] += k
                self.total += k
                s["tail"] = (s["tail"] + (self.TEXT * 2)[s["_pos"]:s["_pos"] + k])[-400:]
                s["_pos"] = (s["_pos"] + k) % len(self.TEXT)
            elif s["state"] == "prefill":
                s["prefill_left"] = max(0, s["prefill_left"] - int(2500 * dt))
                s["ctx"] = s["prompt"] - s["prefill_left"]
            else:
                s["idle_s"] += dt
            if s["_until"] <= 0:
                nxt = {"idle": "prefill", "prefill": "decode", "decode": "idle"}[s["state"]]
                s["state"], s["tok_s"] = nxt, 0.0
                if nxt == "prefill":
                    s["prompt"] = s["ctx"] + random.randint(300, 2500)
                    s["prefill_left"] = s["prompt"] - s["ctx"]
                    s["generated"], s["_until"] = 0, s["prefill_left"] / 2500
                    if s["ctx"] > 60000:
                        s["task"], s["ctx"] = random.choice(self.TASKS), 0
                elif nxt == "decode":
                    s["_until"] = random.uniform(3, 9)
                else:
                    s["idle_s"], s["_until"] = 0.0, random.uniform(0.5, 3)
        self.gpu.update(util=min(100, 35 + 16 * active + random.uniform(-3, 3)), power=110 + 58 * active)
        return {"model": "Qwen3.8-27B-Q4_K_M", "slots": [{k: v for k, v in s.items() if not k.startswith("_")}
                                                        for s in self.slots],
                "pending": 0, "ctx": 123648, "total_generated": self.total}


def tidy(tail):
    """the output tail with Qwen's XML tool-call markup shown as ⚙ name args"""
    tail = re.sub(r"^[^<>]*>", "", tail)  # (a tag cut by the tail's start)
    tail = re.sub(r"<function=([\w.-]+)>?", r" ⚙ \1 ", tail)
    tail = re.sub(r"</?(tool_call|function|parameter|think)(=[\w.-]+)?>", " ", tail)
    tail = re.sub(r"<[^>]*$", "", tail)  # (a tag the output has not finished yet)
    return " ".join(tail.split())


class Agents:
    """when requests name their agent (metadata.agent), one entry per agent instead of per slot: the slot holding
    its latest request; an agent between requests (or whose slot another agent took) shows idle"""

    def __init__(self):
        self.seen = {}  # agent -> (its latest slot entry, time seen)

    def view(self, d):
        now = time.time()
        named = [s for s in d["slots"] if s["agent"]]
        if not named and not self.seen:
            return d["slots"]
        cur = {}
        for s in named:
            o = cur.get(s["agent"])
            if o is None or (s["state"] != "idle", -s["idle_s"]) > (o["state"] != "idle", -o["idle_s"]):
                cur[s["agent"]] = s
        for a, s in cur.items():
            self.seen[a] = (dict(s), now)
        out = []
        for i, a in enumerate(sorted(self.seen, key=lambda a: [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", a)])):
            s, t = self.seen[a]
            num = re.search(r"(\d+)$", a)  # ("agent 3" keeps the third colour whichever agents have shown up)
            s = dict(s, id=int(num.group(1)) - 1 if num else i)
            if a not in cur:  # (its slot now serves another agent)
                s.update(state="idle", tok_s=0.0, idle_s=s["idle_s"] + now - t)
            out.append(s)
        return out


def agent_panel(s, hist, top, color, ctx_total, width):
    st = s["state"]
    if st == "decode":
        badge = Text("● generating", style=f"bold {color}")
    elif st == "prefill":
        done = 1 - s["prefill_left"] / max(1, s["prompt"] - s["reused"])
        badge = Text(f"◐ reading prompt {done:4.0%}", style="bold yellow")
    else:
        badge = Text(f"○ idle {s['idle_s']:.0f}s" if s["task"] else "○ idle", style="dim")
    inner = width - 4
    rate = s["tok_s"]
    digits = big(rate)
    sp = spark(hist, top, 3)
    head = Text()
    for r in range(3):
        head.append(digits[r], style=f"bold {color}" if st == "decode" else "grey35")
        head.append("  tok/s " if r == 1 else "        ", style="dim")
        head.append(sp[r][-(inner - len(digits[r]) - 8):], style=color)
        if r < 2:
            head.append("\n")
    bar_w = max(8, inner - 26)
    ctx = Text()
    ctx.append("context ", style="dim")
    ctx.append(ctx_bar(s["ctx"], ctx_total, bar_w), style=color)
    ctx.append(f" {kfmt(s['ctx']):>6} / {kfmt(ctx_total)}", style="bold")
    task = Text()
    task.append("task    ", style="dim")
    task.append((s["task"] or "—")[: inner - 8], style="bold white")
    tail = tidy(s["tail"])[-(inner - 2):]
    out = Text("▸ " + tail, style="italic grey62", no_wrap=True, overflow="ellipsis")
    name = s["agent"].upper() if s["agent"] else f"AGENT {s['id'] + 1}"
    title = Text.assemble((f" {name} ", f"bold black on {color}"), " ", badge, " ")
    return Panel(Group(head, ctx, task, out), title=title, title_align="left", border_style=color if st != "idle" else "grey35",
                 width=width, padding=(0, 1))


def render(src, d, slots, hists, total_hist, title, demo, width, t0, runner=None):
    if d is None:
        msg = runner.status if runner and runner.status else f"waiting for {getattr(src, 'url', 'server')} … {src.error or ''}"
        return Panel(Text(msg, style="yellow"))
    total = sum(s["tok_s"] for s in slots)
    active = sum(s["state"] == "decode" for s in slots)
    cols = 2 if len(slots) > 1 else 1
    pw = min(70, (width - 1) // cols)
    top = max([max(h) for h in hists.values() if h] + [1]) * 1.05

    hdr = Table.grid(expand=False)
    hdr.add_column(width=pw * cols - 4 - 18)
    hdr.add_column(width=18, justify="right")
    left = Text(no_wrap=True, overflow="ellipsis")
    left.append(title or f"{len(slots)} agents · one GPU", style="bold white")
    if demo:
        left.append("  SIMULATED", style="bold black on yellow")
    left.append("\n")
    left.append(d.get("model", "").removesuffix(".gguf"), style="cyan")
    g = src.gpu
    if g:
        left.append(f" · {g['name']}\n", style="cyan")
        left.append(f"{g['util']:.0f}% util · {g['power']:.0f} W · {g['mem']:.1f}/{g['mem_total']:.0f} GB · "
                    f"{g['temp']:.0f}°C", style="dim")
    left.append("\n")
    left.append(f"{active}/{len(slots)} generating", style="bold green" if active else "dim")
    left.append(f" · {d['total_generated']:,} tokens · up {int(time.time() - t0) // 60}m"
                f"{int(time.time() - t0) % 60:02d}s", style="dim")
    if d.get("pending"):
        left.append(f" · {d['pending']} queued", style="yellow")
    if runner and runner.status.startswith("the agents exited"):
        left.append("\n" + runner.status, style="bold red")
    if runner and runner.results:
        ok = sum(bool(r.get("hidden")) for r in runner.results)
        left.append(f"\n{len(runner.results)} tasks done · {ok} pass hidden tests  ", style="bold")
        for r in runner.results[-4:]:
            left.append(f"{'✓' if r.get('hidden') else '✗'} {r['task']} {r['seconds']:.0f}s  ",
                        style="green" if r.get("hidden") else "red")
    # (digits and label in columns of their own: right-justifying padded lines strips the padding and skews the rows)
    right = Table.grid(padding=(0, 0, 0, 1))
    right.add_column(no_wrap=True)
    right.add_column(no_wrap=True)
    right.add_row(Text("\n".join(big(total, 3)), style="bold #00E676"), Text("\ntok/s", style="bold"))
    hdr.add_row(left, right)
    tsp = spark(total_hist, max(total_hist + [1]) * 1.05, 2)
    spk = Text("\n".join(s[-(pw * cols - 4):] for s in tsp), style="#00E676", no_wrap=True)

    grid = Table.grid(padding=(0, 0))
    for _ in range(cols):
        grid.add_column()
    # (an agent first seen between history ticks has none yet)
    panels = [agent_panel(s, hists.get(s["id"], [0.0] * HIST), top, COLORS[s["id"] % len(COLORS)], d["ctx"], pw) for s in slots]
    for i in range(0, len(panels), cols):
        grid.add_row(*panels[i:i + cols])
    return Group(Panel(Group(hdr, spk), border_style="#00E676", width=pw * cols, padding=(0, 1)), grid)


def main():
    ap = argparse.ArgumentParser(description="live view of the agents an mk_server --parallel is serving")
    ap.add_argument("--url", default="http://127.0.0.1:18080")
    ap.add_argument("--title", default=None, help="headline (default: 'N agents · one GPU')")
    ap.add_argument("--demo", action="store_true", help="simulated agents (shown as SIMULATED)")
    ap.add_argument("--agents", type=int, default=4, help="--demo: number of agents")
    ap.add_argument("--run", type=int, default=0, metavar="N",
                    help="start the server (unless one answers) and N agents on Slipstream eval tasks")
    ap.add_argument("--tasks-each", type=int, default=4, help="--run: tasks per agent")
    ap.add_argument("--model", default=os.path.expanduser("~/open-jet/models/Qwen3.8-27B-Q4_K_M.gguf"),
                    help="--run: GGUF the server loads when it has to start one")
    args = ap.parse_args()
    src = Demo(args.agents) if args.demo else Source(args.url)
    runner = Runner(args.url, args.run, args.model, args.tasks_each) if args.run and not args.demo else None
    console = Console()
    hists, total_hist, agents = {}, [], Agents()
    t0, last = time.time(), 0.0
    with Live(console=console, screen=True, auto_refresh=False) as live:
        try:
            while True:
                now = time.time()
                d = src.data
                slots = agents.view(d) if d else []
                if d and now - last >= TICK:
                    last = now
                    for s in slots:
                        h = hists.setdefault(s["id"], [0.0] * HIST)
                        h.append(s["tok_s"])
                        del h[:-HIST]
                    total_hist.append(sum(s["tok_s"] for s in slots))
                    del total_hist[:-240]
                live.update(render(src, d, slots, hists, total_hist, args.title, args.demo, console.width, t0, runner),
                            refresh=True)
                time.sleep(0.25)
        except KeyboardInterrupt:
            pass
    if runner:
        runner.stop()
        print(f"logs in {runner.logdir}")
    elif src.data is None:
        print(f"no server answered at {args.url}/slots ({src.error}); start it with: "
              "python3 megakernel/server/mk_server.py -m MODEL.gguf --agents 4")


if __name__ == "__main__":
    main()
