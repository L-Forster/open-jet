#!/usr/bin/env python3
"""Runs N open-jet coding agents (HybridWorker, the Slipstream local worker) at once against an already running
`mk_server.py --agents N` (default http://127.0.0.1:18080), each on its own Slipstream eval tasks in its own temp
workspace. Watch them with tools/mk_top.py.

  python3 tools/run_agents.py [-n 4] [--tasks-each 4] [--open-jet ~/open-jet] [--url http://127.0.0.1:18080]
"""
import argparse
import asyncio
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import urllib.parse
import urllib.request
from pathlib import Path


def wait_for_server(url):
    """(a restarted server takes minutes to load: tasks wait for it rather than fail)"""
    while True:
        try:
            with urllib.request.urlopen(url + "/health", timeout=2):
                return
        except Exception:
            time.sleep(2)


async def worker(open_jet, url, ws, agent, idxs):
    sys.path.insert(0, open_jet)
    from evals.slipstream_sabotage.run_eval import changed_files, run_tests, write_repo
    from evals.slipstream_sabotage.tasks import TASKS
    from src import llama_server, provisioning
    from src.config import load_config
    from src.hybrid import HybridWorker
    from src.model_profiles import get_model_profile

    async def shared_server(self):  # (the server is started once, outside, for all agents)
        self.active_backend = "megakernel"
        u = urllib.parse.urlparse(url)
        self.host, self.port, self.base_url = u.hostname, u.port or 80, url

    async def as_is(cfg, **_):
        return cfg

    llama_server.LlamaServerClient.start = shared_server
    current = {}
    stream = llama_server.stream_openai_chat

    def tagged(*a, extra_body=None, **kw):  # (names the agent and its task for GET /slots, i.e. mk_top)
        return stream(*a, extra_body={**(extra_body or {}), "metadata": {"agent": agent, **current}}, **kw)

    llama_server.stream_openai_chat = tagged
    provisioning.provision_setup_artifacts = as_is
    os.chdir(open_jet)  # (load_config reads ./config.yaml first: the one of the checkout the agents come from)
    cfg = load_config()
    prof = get_model_profile(cfg, str(cfg.get("agent_local_profile") or cfg["active_model_profile"]))
    ws = Path(ws)
    write_repo(ws, TASKS[idxs[0]])
    os.chdir(ws)
    hw = await HybridWorker.start(base_cfg=cfg, local_profile=prof, root=ws, approval_handler=lambda _tc: True)
    try:
        for i in idxs:
            task = TASKS[i]
            current["task"] = task["task"].split(". ")[0].rstrip(".")
            for attempt in range(2):  # (again once when the server went away mid-task)
                await asyncio.to_thread(wait_for_server, url)
                write_repo(ws, task)
                os.chdir(ws)
                hw.session.agent.reset_conversation()
                t0 = time.monotonic()
                try:
                    res = await asyncio.wait_for(hw.delegate({"task": task["task"], "acceptance_criteria": task["criteria"]}),
                                                 1800)
                    status = (res.meta or {}).get("status")
                    error = res.output if status == "failed" else None
                except asyncio.TimeoutError:
                    status, error = "timeout", None
                    hw.session.agent.reset_conversation()
                    break
                if status != "failed" or "connect" not in (error or "") and "peer closed" not in (error or ""):
                    break
            print(json.dumps({"task": task["name"], "seconds": round(time.monotonic() - t0, 1), "status": status,
                              "files": sorted(changed_files(ws)), "visible": run_tests(ws)["passed"],
                              "hidden": run_tests(ws, task["hidden"])["passed"], "error": error}), flush=True)
    finally:
        await hw.close()


def main():
    ap = argparse.ArgumentParser(description="run N open-jet coding agents against a running mk_server --agents N")
    ap.add_argument("-n", type=int, default=4, help="agents")
    ap.add_argument("--tasks-each", type=int, default=4)
    ap.add_argument("--open-jet", default=os.path.expanduser("~/open-jet"), help="open-jet checkout with evals/")
    ap.add_argument("--url", default="http://127.0.0.1:18080", help="the mk_server")
    ap.add_argument("--worker", nargs="+", help=argparse.SUPPRESS)  # WORKSPACE AGENT TASK_INDEX...
    args = ap.parse_args()
    args.url = args.url.rstrip("/")
    if args.worker:
        asyncio.run(worker(args.open_jet, args.url, args.worker[0], args.worker[1], [int(a) for a in args.worker[2:]]))
        return
    sys.path.insert(0, args.open_jet)
    from evals.slipstream_sabotage.tasks import TASKS
    root = Path(tempfile.mkdtemp(prefix="mk_agents_"))
    procs = []
    for a in range(args.n):
        ws = root / f"agent{a + 1}"
        ws.mkdir()
        idxs = [str((a + k * args.n) % len(TASKS)) for k in range(args.tasks_each)]
        log = open(root / f"agent{a + 1}.log", "w")
        procs.append((a, subprocess.Popen([sys.executable, os.path.abspath(__file__), "--open-jet", args.open_jet,
                                           "--url", args.url, "--worker", str(ws), f"agent {a + 1}", *idxs], stdout=subprocess.PIPE, stderr=log, text=True)))
    print(f"{args.n} agents running, workspaces and logs in {root}", flush=True)

    def relay(a, p):
        for line in p.stdout:
            print(f"agent {a + 1}: {line.strip()}", flush=True)

    threads = [threading.Thread(target=relay, args=ap) for ap in procs]
    for t in threads:
        t.start()
    try:
        for t in threads:
            t.join()
    except KeyboardInterrupt:
        for _, p in procs:
            p.terminate()


if __name__ == "__main__":
    main()
