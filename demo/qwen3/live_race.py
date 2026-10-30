"""Side-by-side race: the same questions go at the same moment to servers that
differ only in their prefix cache -- none, the CPU prefix cache, the GPU
prefix cache -- and their answers stream next to each other.

Start the servers first, each on its own idle GPU (a persistent kernel must
not share its GPU), identical except for the cache flag:

    CUDA_VISIBLE_DEVICES=0 python -m mirage.engine.launch_server \\
        --model Qwen/Qwen3-0.6B --max-num-batched-requests 4 --max-num-batched-tokens 8 \\
        --page-size 64 --max-seq-length 4096 --max-num-pages 512 --port 8001
    CUDA_VISIBLE_DEVICES=1 python -m mirage.engine.launch_server \\
        --model Qwen/Qwen3-0.6B --max-num-batched-requests 4 --max-num-batched-tokens 8 \\
        --page-size 64 --max-seq-length 4096 --max-num-pages 512 --port 8002 \\
        --gpu-prefix-cache
    CUDA_VISIBLE_DEVICES=2 python -m mirage.engine.launch_server \\
        --model Qwen/Qwen3-0.6B --max-num-batched-requests 4 --max-num-batched-tokens 8 \\
        --page-size 64 --max-seq-length 4096 --max-num-pages 512 --port 8003 \\
        --enable-prefix-cache

Then run, with no further input (servers that are not up are left out):

    python demo/qwen3/live_race.py                 # 3 questions one by one, then 16 at once
    python demo/qwen3/live_race.py "Q1?" "Q2?"      # your own questions instead of the 3
    python demo/qwen3/live_race.py --only cpu      # one server only (one free GPU)
    python demo/qwen3/live_race.py --cold-burst    # the 16 arrive before anything is cached

Every request carries the whole handbook (``live_race_doc.md``) as its system
prompt.  The prompt starts with a run number that changes on every run, so
the first question of a run is cold on every side; from the second one on,
the cached sides reuse the handbook.  Servers started with
``--max-seq-length 2048`` need ``--doc short``.
"""

from __future__ import annotations

import argparse
import http.client
import json
import os
import random
import re
import sys
import threading
import time
import urllib.request
from dataclasses import dataclass
from urllib.parse import urlparse

try:
    from rich.console import Console, Group
    from rich.live import Live
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
except ImportError:  # pragma: no cover
    sys.exit("live_race.py needs rich:  pip install rich")

DOC_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "live_race_doc.md")
CUT = "<!-- short-cut -->"
# One KV page of Qwen3-0.6B: 28 layers x (K, V) x 64 tokens x 8 heads x 128 dims x bf16.
MB_PER_PAGE = 7
# Column title and colour for each cache mode a server reports.
MODES = {"off": ("No prefix cache", "yellow"),
         "cpu": ("CPU prefix cache", "magenta"),
         "gpu": ("GPU prefix cache", "cyan")}

# All answered in sections 1-13, so they work with the short handbook too.
QUESTIONS = [
    "How much extra does it cost to enter or exit at Airport station?",  # 5 yuan
    "How long are lost items kept?",                                     # 30 days
    "When is the last train on Chinese New Year's Eve?",                 # 21:00
    "When did Line 6 open?",                                             # December 28, 2025
    "Which station has the longest transfer, and how long is it?",       # Xinghe Avenue, 420 m
    "Where do I collect something I lost on Line 1?",                    # People's Square, level B1
    "When can I bring a regular bicycle on the metro?",                  # weekends 9:00-16:00, Line 3, last car
    "How large a power bank may I bring?",                               # up to 100 Wh
    "How many points do I need for 1 yuan off a fare?",                  # 100
    "What is the maximum weight of one piece of luggage?",               # 20 kg
    "From what age do seniors ride for free?",                           # 65
    "How many free children can one adult bring?",                       # 2
    "What discount does the Xinghe Pass give?",                          # 5% off
    "What is the highest single-trip fare?",                             # 12 yuan
    "What is the top speed on the Line 4 express section?",              # 120 km/h
    "How often do Line 1 trains run at peak hours?",                     # every 2 min 30 s
    "How far ahead must I book accessibility assistance?",               # 2 hours
    "How soon will I get a reply to a complaint?",                       # within 3 working days
    "What is the name of the free Wi-Fi network?",                       # XingheMetro
]


@dataclass
class Side:
    label: str
    url: str
    style: str
    model: str = ""
    kv_now: int | None = None
    kv_peak: int | None = None

    def get(self, path, timeout=30.0):
        with urllib.request.urlopen(self.url + path, timeout=timeout) as r:
            return json.loads(r.read())

    def cache_mode(self):
        rep = self.get("/debug/prefix_cache?settle=0", timeout=10)
        if rep.get("gpu_mode") == "owner":
            return "gpu"
        return "cpu" if (rep.get("cpu") or {}).get("cache_enabled") else "off"

    def pages_hit(self):
        """Prefix pages served from the cache so far (any cache mode)."""
        try:
            rep = self.get("/debug/prefix_cache?settle=2", timeout=40)
        except Exception:
            return 0
        if rep.get("gpu_mode") == "owner":
            return int(rep["gpu"]["gpu_stats"].get("match_pages", 0))
        return int((rep.get("cpu") or {}).get("cache_blocks_matched", 0))

    def pages_in_use(self):
        """KV pages held right now: by running requests and, with a cache, by
        the cache for later requests."""
        try:
            pages = self.get("/debug/prefix_cache?settle=0", timeout=5)["pages"]
        except Exception:
            return None
        return pages["total"] - pages["gpu_free"] - pages["pending_returns"]


def watch_kv(side: Side, stop: threading.Event):
    side.kv_now = side.kv_peak = None
    while not stop.is_set():
        used = side.pages_in_use()
        if used is not None:
            side.kv_now = used
            side.kv_peak = max(side.kv_peak or 0, used)
        stop.wait(0.05)


@dataclass
class Run:
    question: str
    t0: float = 0.0
    t_first: float | None = None
    t_end: float | None = None
    text: str = ""
    error: str | None = None
    tokens: int = 0  # prompt + completion, as the server counts them

    @property
    def ttft(self):
        return None if self.t_first is None else self.t_first - self.t0

    @property
    def total(self):
        return None if self.t_end is None else self.t_end - self.t0


THINK = re.compile(r"<think>.*?</think>\s*", re.S)


def visible(text):
    """Hide Qwen3's (empty, under /no_think) reasoning block."""
    shown = THINK.sub("", text)
    return "" if "<think>" in shown else shown.strip()


def stream(side: Side, run: Run, system: str, max_tokens: int, start: threading.Event):
    body = json.dumps({
        "model": side.model, "stream": True, "temperature": 0, "max_tokens": max_tokens,
        "stream_options": {"include_usage": True},
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": run.question + " /no_think"}],
    })
    url = urlparse(side.url)
    conn = http.client.HTTPConnection(url.hostname, url.port or 80, timeout=900)
    start.wait()  # both sides leave at the same instant
    try:
        conn.request("POST", "/v1/chat/completions", body=body,
                     headers={"Content-Type": "application/json"})
        resp = conn.getresponse()
        if resp.status != 200:
            run.error = f"HTTP {resp.status}"
            return
        for line in resp:
            line = line.decode("utf-8", "replace").strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            event = json.loads(line[5:])
            if event.get("usage"):
                run.tokens = event["usage"].get("total_tokens", 0)
            for choice in event.get("choices", []):
                piece = (choice.get("delta") or {}).get("content") or ""
                if piece:
                    if run.t_first is None:
                        run.t_first = time.perf_counter()
                    run.text += piece
    except Exception as exc:
        run.error = f"{type(exc).__name__}: {exc}"
    finally:
        run.t_end = time.perf_counter()
        conn.close()


def fmt_s(seconds):
    return f"{seconds * 1000:.0f} ms" if seconds < 1 else f"{seconds:.2f} s"


def fmt_kv(pages):
    mb = pages * MB_PER_PAGE
    return f"{mb:.0f} MB" if mb < 1000 else f"{mb / 1024:.1f} GB"


def side_by_side(*panels):
    grid = Table.grid(expand=True, padding=(0, 1))
    for _ in panels:
        grid.add_column(ratio=1)
    grid.add_row(*panels)
    return grid


def race(console, sides, questions, system, max_tokens, render, kv=False):
    """Send *questions* to both sides at once, redrawing with *render*; with
    *kv*, also follow each side's KV page use."""
    runs = [[Run(q) for q in questions] for _ in sides]
    start, stop = threading.Event(), threading.Event()
    threads = [threading.Thread(target=stream, args=(s, r, system, max_tokens, start), daemon=True)
               for s, rs in zip(sides, runs) for r in rs]
    watchers = [threading.Thread(target=watch_kv, args=(s, stop), daemon=True)
                for s in sides if kv]
    for t in watchers + threads:
        t.start()
    t0 = time.perf_counter()
    for r in (r for rs in runs for r in rs):
        r.t0 = t0
    start.set()
    with Live(console=console, refresh_per_second=15) as live:
        while any(t.is_alive() for t in threads):
            live.update(render(sides, runs, time.perf_counter()))
            time.sleep(1 / 15)
        live.update(render(sides, runs, time.perf_counter()))
    stop.set()
    for t in watchers:
        t.join()
    return runs


def render_one(sides, runs, now):
    panels = []
    for side, (run,) in zip(sides, runs):
        if run.error:
            head = Text(run.error, style="bold red")
        elif run.t_first is None:
            head = Text(f"waiting {fmt_s(now - run.t0)}", style="dim")
        else:
            head = Text(f"first token {fmt_s(run.ttft)}", style=f"bold {side.style}")
        foot = Text("" if run.t_end is None else f"done {fmt_s(run.total)}", style="dim")
        panels.append(Panel(Group(head, Text(""), Text(visible(run.text)), Text(""), foot),
                            title=side.label, border_style=side.style, height=10))
    return side_by_side(*panels)


def render_many(sides, runs, now):
    panels = []
    for side, rs in zip(sides, runs):
        done = sum(r.t_end is not None for r in rs)
        end = max(r.t_end for r in rs) if done == len(rs) else now
        lines = [Text(f"answered {done}/{len(rs)}   elapsed {end - rs[0].t0:.1f} s",
                      style=f"bold {side.style}" if done == len(rs) else "")]
        if side.kv_now is not None:
            lines.append(Text(f"KV memory in use {fmt_kv(side.kv_now)}   peak {fmt_kv(side.kv_peak)}"))
        panels.append(Panel(Group(*lines), title=side.label, border_style=side.style))
    return side_by_side(*panels)


def main():
    p = argparse.ArgumentParser(description="Side-by-side race: no prefix cache vs GPU prefix cache")
    p.add_argument("questions", nargs="*", help="asked one by one (default: three built-in ones)")
    p.add_argument("--left", default="http://127.0.0.1:8001", help="server without a prefix cache")
    p.add_argument("--right", default="http://127.0.0.1:8002", help="server with --gpu-prefix-cache")
    p.add_argument("--cpu", default="http://127.0.0.1:8003", help="server with --enable-prefix-cache")
    p.add_argument("--doc", choices=["long", "short"], default="long",
                   help="long (~2600 tokens) for --max-seq-length 4096, short (~1750) for 2048")
    p.add_argument("--n", type=int, default=16, help="questions sent at once at the end (0: skip)")
    p.add_argument("--max-tokens", type=int, default=80)
    p.add_argument("--cold-burst", action="store_true",
                   help="send the batch with a new run number, before the handbook is cached")
    p.add_argument("--only", nargs="+", choices=["left", "cpu", "right"],
                   help="race only these servers (default: every one that is up)")
    args = p.parse_args()

    console = Console()
    sides = []
    for name, url in (("left", args.left), ("cpu", args.cpu), ("right", args.right)):
        if args.only and name not in args.only:
            continue
        side = Side("", url.rstrip("/"), "")
        try:
            side.model = side.get("/v1/models", timeout=5)["data"][0]["id"]
            # Titled by what the server says it runs, not by its port.
            side.label, side.style = MODES[side.cache_mode()]
        except Exception as exc:
            if args.only:
                sys.exit(f"cannot reach {side.url}: {exc}")
            continue
        sides.append(side)
    if not sides:
        sys.exit("No server is up yet; start them first (see the top of this file).")
    console.print("Racing: " + " | ".join(f"[{s.style}]{s.label}[/] ({s.url})" for s in sides))
    cached = [s for s in sides if s.label != MODES["off"][0]]

    doc = open(DOC_PATH, encoding="utf-8").read()
    doc = doc.split(CUT)[0].rstrip() if args.doc == "short" else doc.replace(CUT + "\n\n", "")

    def system_prompt():
        # The run number comes first, so nothing cached under another number can match.
        return (f"Demo run {random.randint(1000, 9999)}. You are the customer service assistant "
                f"of Xinghe City Metro. Answer the passenger's question using only the handbook "
                f"below, in at most two sentences.\n\n{doc}")

    system = system_prompt()

    for i, q in enumerate(args.questions or QUESTIONS[:3], 1):
        console.print(f"\n[bold]Question {i}: {q}")
        before = [s.pages_hit() for s in cached]
        firsts = [rs[0] for rs in race(console, sides, [q], system, args.max_tokens, render_one)]
        if all(r.ttft is not None for r in firsts):
            hits = [s.pages_hit() - b for s, b in zip(cached, before)]
            note = "  (reused from cache: " + ", ".join(
                f"{s.label} {h} pages = {fmt_kv(h)}" for s, h in zip(cached, hits)) + ")"
            note = note if cached else ""
            console.print("first token " + " vs ".join(fmt_s(r.ttft) for r in firsts) + note, style="dim")
        time.sleep(1)

    if args.n:
        qs = [QUESTIONS[(3 + i) % len(QUESTIONS)] for i in range(args.n)]
        if args.cold_burst:
            system = system_prompt()
        console.print(f"\n[bold]{args.n} questions at once"
                      + (", handbook not cached yet" if args.cold_burst else ""))
        before = [s.pages_hit() for s in cached]
        runs = race(console, sides, qs, system, args.max_tokens, render_many, kv=True)
        hits = {s.url: s.pages_hit() - b for s, b in zip(cached, before)}
        console.print("all answered in " + " vs ".join(
            f"{max(r.t_end for r in rs) - rs[0].t0:.1f} s" for rs in runs), style="dim")
        # KV the side had to compute: every prompt and answer token, minus
        # what the cache handed over ready-made.
        fresh = [sum(r.tokens for r in rs) / 64 - hits.get(s.url, 0)
                 for s, rs in zip(sides, runs)]
        note = "  (taken from cache: " + ", ".join(
            f"{s.label} {fmt_kv(hits[s.url])}" for s in cached) + ")" if cached else ""
        if all(r.tokens for rs in runs for r in rs):
            console.print("KV computed " + " vs ".join(fmt_kv(f) for f in fresh) + note, style="dim")
        if all(s.kv_peak is not None for s in sides):
            console.print("peak KV memory " + " vs ".join(fmt_kv(s.kv_peak) for s in sides), style="dim")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
