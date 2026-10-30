"""Side-by-side race: the same questions go to two servers at the same moment
-- one without a prefix cache, one with the GPU prefix cache -- and both
answers stream next to each other.

Start the two servers first, each on its own idle GPU (a persistent kernel must
not share its GPU), identical except for the cache flag:

    CUDA_VISIBLE_DEVICES=0 python -m mirage.engine.launch_server \\
        --model Qwen/Qwen3-0.6B --max-num-batched-requests 4 --max-num-batched-tokens 8 \\
        --page-size 64 --max-seq-length 4096 --max-num-pages 512 --port 8001
    CUDA_VISIBLE_DEVICES=1 python -m mirage.engine.launch_server \\
        --model Qwen/Qwen3-0.6B --max-num-batched-requests 4 --max-num-batched-tokens 8 \\
        --page-size 64 --max-seq-length 4096 --max-num-pages 512 --port 8002 \\
        --gpu-prefix-cache

Then run, with no further input:

    python demo/qwen3/live_race.py                 # 3 questions one by one, then 16 at once
    python demo/qwen3/live_race.py "问题一" "问题二"  # your own questions instead of the 3
    python demo/qwen3/live_race.py --only right    # one server only (one free GPU)

Every request carries the whole handbook (``live_race_doc.md``) as its system
prompt.  The prompt starts with a run number that changes on every run, so
the first question of a run is cold on both sides; from the second one on,
the cached side reuses the handbook.  Servers started with
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

# All answered in sections 1-13, so they work with the short handbook too.
QUESTIONS = [
    "机场站进出站要另外加收多少钱？",          # 5 元
    "失物一般保管多少天？",                    # 30 天
    "除夕当天末班车几点？",                    # 21:00
    "6 号线是哪一天开通的？",                  # 2025 年 12 月 28 日
    "最长的换乘通道在哪个站？有多长？",        # 星河大道站，420 米
    "1 号线丢了东西应该去哪里领？",            # 人民广场站 B1 层
    "普通自行车什么时候可以带上地铁？",        # 周末 9:00-16:00，3 号线最后一节车厢
    "可以带多大的充电宝？",                    # 不超过 100 瓦时
    "多少积分可以抵扣 1 元车费？",             # 100 个
    "单件行李最重可以多少公斤？",              # 20 公斤
    "老年人多少岁以上可以免费乘车？",          # 65 岁
    "一名成人最多可以带几名免费儿童？",        # 2 名
    "星河通卡打几折？",                        # 9.5 折
    "单程票最高多少钱？",                      # 12 元
    "4 号线快线区段最高时速是多少？",          # 120 公里
    "高峰期 1 号线的发车间隔是多少？",         # 2 分 30 秒
    "无障碍服务要提前多久预约？",              # 2 小时
    "投诉多久会得到答复？",                    # 3 个工作日内
    "免费 Wi-Fi 的网络名称是什么？",           # XingheMetro
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
            head = Text(f"等待中 {fmt_s(now - run.t0)}", style="dim")
        else:
            head = Text(f"首字 {fmt_s(run.ttft)}", style=f"bold {side.style}")
        foot = Text("" if run.t_end is None else f"完成 {fmt_s(run.total)}", style="dim")
        panels.append(Panel(Group(head, Text(""), Text(visible(run.text)), Text(""), foot),
                            title=side.label, border_style=side.style, height=9))
    return side_by_side(*panels)


def render_many(sides, runs, now):
    panels = []
    for side, rs in zip(sides, runs):
        done = sum(r.t_end is not None for r in rs)
        end = max(r.t_end for r in rs) if done == len(rs) else now
        lines = [Text(f"已完成 {done}/{len(rs)}   用时 {end - rs[0].t0:.1f} s",
                      style=f"bold {side.style}" if done == len(rs) else "")]
        if side.kv_now is not None:
            lines.append(Text(f"KV 显存 正在用 {fmt_kv(side.kv_now)}   峰值 {fmt_kv(side.kv_peak)}"))
        panels.append(Panel(Group(*lines), title=side.label, border_style=side.style))
    return side_by_side(*panels)


def main():
    p = argparse.ArgumentParser(description="Side-by-side race: no prefix cache vs GPU prefix cache")
    p.add_argument("questions", nargs="*", help="asked one by one (default: three built-in ones)")
    p.add_argument("--left", default="http://127.0.0.1:8001", help="server without a prefix cache")
    p.add_argument("--right", default="http://127.0.0.1:8002", help="server with --gpu-prefix-cache")
    p.add_argument("--doc", choices=["long", "short"], default="long",
                   help="long (~2750 tokens) for --max-seq-length 4096, short (~1900) for 2048")
    p.add_argument("--n", type=int, default=16, help="questions sent at once at the end (0: skip)")
    p.add_argument("--max-tokens", type=int, default=80)
    p.add_argument("--only", choices=["left", "right"],
                   help="run the same flow against one server (when only one GPU is free)")
    args = p.parse_args()

    console = Console()
    left = Side("不开 prefix cache", args.left.rstrip("/"), "yellow")
    right = Side("GPU prefix cache", args.right.rstrip("/"), "cyan")
    sides = {"left": [left], "right": [right]}.get(args.only, [left, right])
    cached = right if right in sides else None
    for s in sides:
        try:
            s.model = s.get("/v1/models", timeout=5)["data"][0]["id"]
        except Exception as exc:
            sys.exit(f"{s.label} {s.url} 连不上：{exc}")

    doc = open(DOC_PATH, encoding="utf-8").read()
    doc = doc.split(CUT)[0].rstrip() if args.doc == "short" else doc.replace(CUT + "\n\n", "")
    # The run number comes first, so nothing cached by an earlier run can match.
    system = (f"演示编号 {random.randint(1000, 9999)}。你是星河市地铁的客服助手。"
              f"请只根据下面的手册回答乘客的问题，回答不超过两句话。\n\n{doc}")

    for i, q in enumerate(args.questions or QUESTIONS[:3], 1):
        console.print(f"\n[bold]问题 {i}：{q}")
        before = cached.pages_hit() if cached else 0
        firsts = [rs[0] for rs in race(console, sides, [q], system, args.max_tokens, render_one)]
        if all(r.ttft is not None for r in firsts):
            note = ""
            if cached:
                hit = cached.pages_hit() - before
                note = f"（{cached.label} 从 cache 复用了 {hit} 页 KV，{fmt_kv(hit)}）"
            console.print("首字 " + " vs ".join(fmt_s(r.ttft) for r in firsts) + note, style="dim")
        time.sleep(1)

    if args.n:
        qs = [QUESTIONS[(3 + i) % len(QUESTIONS)] for i in range(args.n)]
        console.print(f"\n[bold]{args.n} 个问题同时发出")
        before = cached.pages_hit() if cached else 0
        runs = race(console, sides, qs, system, args.max_tokens, render_many, kv=True)
        hit = cached.pages_hit() - before if cached else 0
        console.print("全部答完 " + " vs ".join(
            f"{max(r.t_end for r in rs) - rs[0].t0:.1f} s" for rs in runs), style="dim")
        # KV the side had to compute: every prompt and answer token, minus
        # what the cache handed over ready-made.
        fresh = [sum(r.tokens for r in rs) / 64 - (hit if s is cached else 0)
                 for s, rs in zip(sides, runs)]
        note = f"（{cached.label} 从 cache 复用了 {fmt_kv(hit)}）" if cached else ""
        if all(r.tokens for rs in runs for r in rs):
            console.print("新算的 KV " + " vs ".join(fmt_kv(f) for f in fresh) + note, style="dim")
        if all(s.kv_peak is not None for s in sides):
            console.print("KV 显存峰值 " + " vs ".join(fmt_kv(s.kv_peak) for s in sides), style="dim")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
