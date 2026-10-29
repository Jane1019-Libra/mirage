"""Live side-by-side race: the same questions go to two servers at the same
moment -- one without a prefix cache, one with the GPU prefix cache -- and the
answers stream next to each other, so an audience sees the difference instead
of reading it off a table.

Start the two servers first, each on its own idle GPU (a persistent kernel must
not share its GPU), identical except for the cache flag:

    CUDA_VISIBLE_DEVICES=0 python -m mirage.engine.launch_server \\
        --model Qwen/Qwen3-0.6B --max-num-batched-requests 4 --max-num-batched-tokens 8 \\
        --page-size 64 --max-seq-length 4096 --max-num-pages 512 --port 8001
    CUDA_VISIBLE_DEVICES=1 python -m mirage.engine.launch_server \\
        --model Qwen/Qwen3-0.6B --max-num-batched-requests 4 --max-num-batched-tokens 8 \\
        --page-size 64 --max-seq-length 4096 --max-num-pages 512 --port 8002 \\
        --gpu-prefix-cache

Then, in a third terminal:

    python demo/qwen3/live_race.py check          # both servers up? which cache?
    python demo/qwen3/live_race.py show           # the three acts, Enter between them
    python demo/qwen3/live_race.py reset          # new document number: cache cold again
    python demo/qwen3/live_race.py ask            # next question from the bank
    python demo/qwen3/live_race.py ask "末班车几点？" # an audience question
    python demo/qwen3/live_race.py burst --n 16   # 16 questions at once to each side

Every request carries the whole handbook (``live_race_doc.md``) as its system
prompt.  The prompt starts with a document number; ``reset`` changes it, so
nothing cached earlier can match and the next question is cold on both sides.
Servers started with ``--max-seq-length 2048`` need ``--doc short``.
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
from dataclasses import dataclass, field
from urllib.parse import urlparse

try:
    from rich.align import Align
    from rich.console import Console, Group
    from rich.live import Live
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
except ImportError:  # pragma: no cover - a hint beats a traceback on stage
    sys.exit("live_race.py needs rich:  pip install rich")

HERE = os.path.dirname(os.path.abspath(__file__))
DOC_PATH = os.path.join(HERE, "live_race_doc.md")
STATE_PATH = os.path.expanduser("~/.cache/mirage_demo/live_race_state.json")
CUT = "<!-- short-cut -->"

# (question, answer in the handbook, section number). Sections 1-13 are in
# the short handbook as well; 14-20 only in the long one.
QUESTIONS = [
    ("6 号线是哪一天开通的？", "2025 年 12 月 28 日", 1),
    ("机场站进出站要另外加收多少钱？", "5 元", 3),
    ("最长的换乘通道在哪个站？有多长？", "星河大道站，420 米", 4),
    ("失物一般保管多少天？", "30 天", 7),
    ("1 号线丢了东西应该去哪里领？", "人民广场站 B1 层", 7),
    ("普通自行车什么时候可以带上地铁？", "周末 9:00-16:00，3 号线最后一节车厢", 6),
    ("可以带多大的充电宝？", "额定能量不超过 100 瓦时", 9),
    ("除夕当天末班车几点？", "21:00", 11),
    ("多少积分可以抵扣 1 元车费？", "100 个积分", 12),
    ("单件行李最重可以多少公斤？", "20 公斤", 5),
    ("老年人多少岁以上可以免费乘车？", "65 岁", 3),
    ("一名成人最多可以带几名免费儿童？", "2 名", 3),
    ("星河通卡打几折？", "9.5 折", 3),
    ("单程票最高多少钱？", "12 元", 3),
    ("4 号线快线区段最高时速是多少？", "120 公里", 1),
    ("高峰期 1 号线的发车间隔是多少？", "2 分 30 秒", 2),
    ("周五和周六哪几条线路延长运营？", "1、2、4 号线，到次日 0:30", 2),
    ("无障碍服务要提前多久预约？", "2 小时", 8),
    ("投诉多久会得到答复？", "3 个工作日内", 10),
    ("积分的有效期是多久？", "2 年", 12),
    ("哪些车站有母婴室？一共多少座？", "32 座", 13),
    ("免费 Wi-Fi 的网络名称是什么？", "XingheMetro", 13),
    ("宠物笼子的尺寸上限是多少？", "50×40×40 厘米", 6),
    ("2 号线跑完一整圈要多久？", "约 78 分钟", 1),
    ("学生卡每天最多享受几次优惠？", "4 次", 15),
    ("进站后超过多久不出站要补票？", "4 小时，补 12 元", 16),
    ("在车站吸烟罚款多少？", "200 元", 18),
    ("纪念车票在哪里买？多少钱？", "西山站和东湖站客服中心，每套 50 元", 19),
    ("风力几级时 3 号线高架区段会暂停运营？", "8 级", 17),
    ("补办星河通卡的工本费是多少？", "20 元", 16),
]


# ── servers ──────────────────────────────────────────────────────────────────


@dataclass
class Side:
    label: str
    url: str
    style: str
    model: str = ""

    @property
    def host(self):
        return urlparse(self.url).hostname

    @property
    def port(self):
        return urlparse(self.url).port or 80

    def get(self, path, timeout=30.0):
        with urllib.request.urlopen(self.url + path, timeout=timeout) as r:
            return json.loads(r.read())

    def cache_report(self, settle=2.0):
        try:
            return self.get(f"/debug/prefix_cache?settle={settle}", timeout=settle + 30)
        except Exception:
            return None


def cache_mode(rep):
    if not rep:
        return "unknown"
    if rep.get("gpu_mode") == "owner":
        return "GPU prefix cache"
    if rep.get("gpu_mode") == "shadow":
        return "CPU cache（GPU 影子）"
    if (rep.get("cpu") or {}).get("cache_enabled"):
        return "CPU prefix cache"
    return "无 prefix cache"


def pages_hit(rep):
    """Cumulative prefix pages served from the cache so far, any mode."""
    if not rep:
        return 0
    if rep.get("gpu_mode") == "owner":
        return int(((rep.get("gpu") or {}).get("gpu_stats") or {}).get("match_pages", 0))
    return int((rep.get("cpu") or {}).get("cache_blocks_matched", 0))


# ── one streamed request ─────────────────────────────────────────────────────


@dataclass
class Run:
    question: str
    t0: float = 0.0
    t_first: float | None = None
    t_end: float | None = None
    text: str = ""
    completion_tokens: int | None = None
    error: str | None = None

    def elapsed(self, now):
        end = self.t_end if self.t_end is not None else now
        return end - self.t0 if self.t0 else 0.0

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
    conn = http.client.HTTPConnection(side.host, side.port, timeout=900)
    start.wait()  # both sides leave at the same instant
    run.t0 = time.perf_counter()
    try:
        conn.request("POST", "/v1/chat/completions", body=body,
                     headers={"Content-Type": "application/json"})
        resp = conn.getresponse()
        if resp.status != 200:
            run.error = f"HTTP {resp.status}: {resp.read()[:200]!r}"
            return
        while True:
            line = resp.readline()
            if not line:
                break
            line = line.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            event = json.loads(data)
            if event.get("usage"):
                run.completion_tokens = event["usage"].get("completion_tokens")
            for choice in event.get("choices", []):
                piece = (choice.get("delta") or {}).get("content") or ""
                if piece:
                    if run.t_first is None:
                        run.t_first = time.perf_counter()
                    run.text += piece
    except Exception as exc:  # shown in the panel; the other side keeps going
        run.error = f"{type(exc).__name__}: {exc}"
    finally:
        run.t_end = time.perf_counter()
        conn.close()


# ── rendering ────────────────────────────────────────────────────────────────


def fmt_s(seconds):
    if seconds is None:
        return "—"
    return f"{seconds * 1000:.0f} ms" if seconds < 1 else f"{seconds:.2f} s"


def ask_panel(side: Side, run: Run, now: float):
    head = Text()
    if run.error:
        head.append(f"出错：{run.error}", style="bold red")
    elif run.t_first is None:
        head.append("等待第一个字…  ", style="dim")
        head.append(fmt_s(run.elapsed(now)), style="bold")
    else:
        head.append("首字 ", style="dim")
        head.append(fmt_s(run.ttft), style=f"bold {side.style}")
    body = Text(visible(run.text) or " ", style="default")
    lines = [head, Text(""), body]
    if run.t_end is not None and not run.error:
        foot = Text(f"完成 {fmt_s(run.total)}", style="dim")
        if run.completion_tokens and run.t_first and run.completion_tokens > 1:
            rate = (run.completion_tokens - 1) / max(run.t_end - run.t_first, 1e-6)
            foot.append(f" · {run.completion_tokens} token · 解码 {rate:.0f} token/s", style="dim")
        lines += [Text(""), foot]
    return Panel(Group(*lines), title=f"[bold {side.style}]{side.label}", border_style=side.style,
                 padding=(1, 2), height=11)


def two_columns(left, right):
    grid = Table.grid(expand=True, padding=(0, 1))
    grid.add_column(ratio=1)
    grid.add_column(ratio=1)
    grid.add_row(left, right)
    return grid


def burst_panel(side: Side, runs: list[Run], now: float):
    done = sum(1 for r in runs if r.t_end is not None)
    started = min((r.t0 for r in runs if r.t0), default=now)
    ends = [r.t_end for r in runs if r.t_end is not None]
    wall = (max(ends) if done == len(runs) else now) - started
    table = Table(expand=True, show_edge=False, box=None, pad_edge=False)
    table.add_column("#", justify="right", style="dim", width=3)
    table.add_column("问题", ratio=1, no_wrap=True, overflow="ellipsis")
    table.add_column("首字", justify="right", width=8)
    table.add_column("完成", justify="right", width=8)
    for i, r in enumerate(runs, 1):
        if r.error:
            mark, first, total = "✗", "出错", ""
        elif r.t_end is not None:
            mark, first, total = "✓", fmt_s(r.ttft), fmt_s(r.total)
        elif r.t_first is not None:
            mark, first, total = "…", fmt_s(r.ttft), ""
        else:
            mark, first, total = "·", "", ""
        style = side.style if r.t_end is not None else ("default" if r.t_first else "dim")
        table.add_row(str(i), Text(f"{mark} {r.question}", style=style), first, total)
    title = f"[bold {side.style}]{side.label}[/]  已完成 {done}/{len(runs)} · 用时 {wall:.1f} s"
    return Panel(table, title=title, border_style=side.style, padding=(0, 1))


# ── actions ──────────────────────────────────────────────────────────────────


def load_state():
    try:
        with open(STATE_PATH) as f:
            return json.load(f)
    except Exception:
        return {"nonce": "0001", "next": 0}


def save_state(state):
    os.makedirs(os.path.dirname(STATE_PATH), exist_ok=True)
    with open(STATE_PATH, "w") as f:
        json.dump(state, f)


def handbook(size):
    doc = open(DOC_PATH, encoding="utf-8").read()
    return doc.split(CUT)[0].rstrip() if size == "short" else doc.replace(CUT + "\n\n", "")


def system_prompt(state, size):
    # The number comes first: change it and no cached page can match any more.
    return (f"演示编号 {state['nonce']}。你是星河市地铁的客服助手。"
            f"请只根据下面的手册回答乘客的问题，回答不超过两句话。\n\n{handbook(size)}")


def bank(size):
    return [q for q in QUESTIONS if size == "long" or q[2] <= 13]


def next_questions(state, size, n):
    qs = bank(size)
    picked = [qs[(state["next"] + i) % len(qs)] for i in range(n)]
    state["next"] = (state["next"] + n) % len(qs)
    save_state(state)
    return picked


def race(console, sides, questions, system, max_tokens, burst):
    """Send *questions* to both sides at once and draw them live."""
    runs = {s.label: [Run(q) for q in questions] for s in sides}
    start = threading.Event()
    threads = [threading.Thread(target=stream, args=(s, r, system, max_tokens, start), daemon=True)
               for s in sides for r in runs[s.label]]
    for t in threads:
        t.start()
    before = [s.cache_report(settle=0.5) for s in sides]

    def frame():
        now = time.perf_counter()
        if burst:
            return two_columns(*(burst_panel(s, runs[s.label], now) for s in sides))
        return two_columns(*(ask_panel(s, runs[s.label][0], now) for s in sides))

    start.set()
    with Live(frame(), console=console, refresh_per_second=15, transient=False) as live:
        while any(t.is_alive() for t in threads):
            live.update(frame())
            time.sleep(1 / 15)
        live.update(frame())
    console.print()
    after = [s.cache_report(settle=2.0) for s in sides]
    return runs, before, after


def summary_ask(console, sides, runs, before, after):
    a, b = (runs[s.label][0] for s in sides)
    line = Text()
    if a.ttft and b.ttft:
        line.append("首字：", style="bold")
        line.append(f"{sides[0].label} {fmt_s(a.ttft)}", style=sides[0].style)
        line.append("  vs  ")
        line.append(f"{sides[1].label} {fmt_s(b.ttft)}", style=f"bold {sides[1].style}")
        if a.ttft > 1.2 * b.ttft:
            line.append(f"   快 {a.ttft / b.ttft:.1f} 倍", style="bold")
    hit = pages_hit(after[1]) - pages_hit(before[1])
    note = Text(f"{sides[1].label} 这一题从 cache 复用了 {hit} 页（{hit * 64} 个 token 不用重算）"
                if hit else f"{sides[1].label} 这一题没有命中 cache（这篇手册还没算过）",
                style="dim")
    console.print(Align.center(line))
    console.print(Align.center(note))


def summary_burst(console, sides, runs, before, after):
    def wall(rs):
        return max(r.t_end for r in rs) - min(r.t0 for r in rs)

    def median_ttft(rs):
        xs = sorted(r.ttft for r in rs if r.ttft is not None)
        return xs[len(xs) // 2] if xs else None

    ra, rb = runs[sides[0].label], runs[sides[1].label]
    wa, wb = wall(ra), wall(rb)
    line = Text()
    line.append(f"{len(ra)} 个问题全部答完：", style="bold")
    line.append(f"{sides[0].label} {wa:.1f} s", style=sides[0].style)
    line.append("  vs  ")
    line.append(f"{sides[1].label} {wb:.1f} s", style=f"bold {sides[1].style}")
    if wa > 1.2 * wb:
        line.append(f"   快 {wa / wb:.1f} 倍", style="bold")
    ma, mb = median_ttft(ra), median_ttft(rb)
    hit = pages_hit(after[1]) - pages_hit(before[1])
    note = Text(f"首字中位数 {fmt_s(ma)} vs {fmt_s(mb)} · {sides[1].label} 本轮从 cache 复用了 "
                f"{hit} 页（{hit * 64} 个 token）", style="dim")
    console.print(Align.center(line))
    console.print(Align.center(note))


def act_title(console, text):
    console.print()
    console.rule(f"[bold]{text}")


def do_reset(console, state):
    state["nonce"] = f"{random.randint(1000, 9999)}"
    state["next"] = 0
    save_state(state)
    console.print(f"已换成新的演示编号 [bold]{state['nonce']}[/]：下一题两边都从零开始算。")


def do_ask(console, sides, state, args, question=None):
    q = question or next_questions(state, args.doc, 1)[0][0]
    console.print(Text(f"问题：{q}", style="bold"))
    runs, before, after = race(console, sides, [q], system_prompt(state, args.doc),
                               args.max_tokens, burst=False)
    summary_ask(console, sides, runs, before, after)


def do_burst(console, sides, state, args):
    qs = [q for q, _a, _s in next_questions(state, args.doc, args.n)]
    console.print(Text(f"同时向两边各发 {len(qs)} 个问题", style="bold"))
    runs, before, after = race(console, sides, qs, system_prompt(state, args.doc),
                               args.max_tokens, burst=True)
    summary_burst(console, sides, runs, before, after)


def do_check(console, sides, args):
    ok = True
    for s in sides:
        try:
            s.get("/health", timeout=5)
            rep = s.cache_report(settle=0.2)
            console.print(f"[{s.style}]{s.label}[/]  {s.url}  模型 {s.model}  ·  {cache_mode(rep)}")
        except Exception as exc:
            ok = False
            console.print(f"[red]{s.label}  {s.url}  连不上：{exc}[/]")
    doc = handbook(args.doc)
    console.print(f"手册：{args.doc}（{len(doc)} 字）· 题库 {len(bank(args.doc))} 题 · "
                  f"演示编号 {load_state()['nonce']}")
    return ok


def do_show(console, sides, state, args):
    act_title(console, "准备：换一个新的演示编号，两边的 cache 都是冷的")
    do_reset(console, state)
    act_title(console, "第一幕：第一个问题，两边都要读完整本手册")
    do_ask(console, sides, state, args)
    input("\n按回车进入第二幕…")
    act_title(console, "第二幕：换一个问题，右边已经读过这本手册")
    typed = input("输入观众的问题（直接回车用题库里的下一题）：").strip()
    do_ask(console, sides, state, args, typed or None)
    input("\n按回车进入第三幕…")
    act_title(console, f"第三幕：{args.n} 个问题同时涌进来")
    do_burst(console, sides, state, args)


def main():
    p = argparse.ArgumentParser(description="Side-by-side live race: no prefix cache vs GPU prefix cache")
    p.add_argument("action", choices=["check", "show", "reset", "ask", "burst"])
    p.add_argument("question", nargs="?", default=None, help="for ask: the question to send")
    p.add_argument("--left", default="http://127.0.0.1:8001", help="server without a prefix cache")
    p.add_argument("--right", default="http://127.0.0.1:8002", help="server with --gpu-prefix-cache")
    p.add_argument("--left-label", default="不开 prefix cache")
    p.add_argument("--right-label", default="GPU prefix cache")
    p.add_argument("--doc", choices=["long", "short"], default="long",
                   help="long (~2750 tokens) for --max-seq-length 4096, short (~1900) for 2048")
    p.add_argument("--n", type=int, default=16, help="burst size per side")
    p.add_argument("--max-tokens", type=int, default=80)
    args = p.parse_args()

    console = Console()
    sides = [Side(args.left_label, args.left.rstrip("/"), "yellow"),
             Side(args.right_label, args.right.rstrip("/"), "cyan")]
    state = load_state()
    if args.action == "reset":
        do_reset(console, state)
        return 0
    for s in sides:
        try:
            s.model = s.get("/v1/models", timeout=5)["data"][0]["id"]
        except Exception as exc:
            console.print(f"[red]{s.label} {s.url} 连不上：{exc}。先按文件开头的说明启动两个服务。[/]")
            return 1
    if args.action == "check":
        return 0 if do_check(console, sides, args) else 1
    if args.action == "ask":
        do_ask(console, sides, state, args, args.question)
    elif args.action == "burst":
        do_burst(console, sides, state, args)
    elif args.action == "show":
        do_show(console, sides, state, args)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (KeyboardInterrupt, EOFError):
        sys.exit(130)
