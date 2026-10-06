"""Showcase: OpenAI-compatible online serving on the MPK persistent megakernel.

Drives a live server through eleven short scenes, each exercising one feature end
to end and printing what it measured:

   1 server      /health, /v1/models, engine geometry
   2 api         chat + raw text completions, usage, finish_reason
   3 stream      SSE streaming: time to first token, inter-token gaps, usage
   4 sampling    greedy, seeded sampling, top-k / top-p, max_tokens,
                 logit_bias, repetition / frequency penalties (in-kernel)
   5 prefix      shared system prompt: cold vs warm, pages reused, TTFT
   6 multiturn   a conversation that grows; each turn reuses the last one's KV
   7 concurrent  8 simultaneous streams with private markers (isolation)
   8 cancel      a client hangs up mid-stream; the server recovers cleanly
   9 errors      OpenAI-style 400 / 404 answers for bad requests
  10 load        24 system prompts at once: the cache fills, evicts, keeps serving
  11 internals   GPU prefix-cache health: page conservation, structure, cost

Features come from three branches: the OpenAI layer and in-kernel per-request
sampling (mirage-project/mirage#784), the batched streaming monitor with
disconnect cancellation (zhendonghua/mirage#2), and the OIPL page lifecycle
with the GPU-resident prefix cache (oipl-base + oipl-gpu-cache).

Run it either way:

    # start a server on GPU 0 for the demo, stop it afterwards
    python demo/qwen3/demo_serving_showcase.py --launch --gpu 0

    # attach to a server you started yourself
    python -m mirage.engine.launch_server --model Qwen/Qwen3-0.6B \\
        --page-size 64 --max-seq-length 2048 --max-num-pages 256 \\
        --gpu-prefix-cache --port 8000
    python demo/qwen3/demo_serving_showcase.py --port 8000

``--scenes 5,6`` runs a subset; ``--report out.json`` saves every measurement.
Scenes 5, 6, 10 and 11 read ``GET /debug/prefix_cache``; against a server without a
prefix cache they still run and say so.
"""

from __future__ import annotations

import argparse
import http.client
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

# ── terminal output ──────────────────────────────────────────────────────────

USE_COLOR = sys.stdout.isatty()


def _c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if USE_COLOR else text


def bold(t): return _c("1", t)
def dim(t): return _c("2", t)
def green(t): return _c("32", t)
def red(t): return _c("31", t)
def cyan(t): return _c("36", t)
def yellow(t): return _c("33", t)


def scene_header(number: int, title: str, what: str) -> None:
    print()
    print(bold(cyan(f"━━ {number:>2}. {title} " + "━" * max(0, 64 - len(title)))))
    print(dim("   " + what))


def ok(msg: str) -> None:
    print("   " + green("✓ ") + msg)


def bad(msg: str) -> None:
    print("   " + red("✗ ") + msg)


def info(msg: str) -> None:
    print("   " + msg)


def table(rows: list[list], header: list[str]) -> None:
    cells = [[str(c) for c in header]] + [[str(c) for c in r] for r in rows]
    widths = [max(len(r[i]) for r in cells) for i in range(len(header))]
    line = lambda r: "   " + "  ".join(c.rjust(w) if i else c.ljust(w)
                                       for i, (c, w) in enumerate(zip(r, widths)))
    print(bold(line(cells[0])))
    for r in cells[1:]:
        print(line(r))


THINK = re.compile(r"<think>.*?</think>\s*", re.S)


def visible(text: str) -> str:
    """Drop Qwen3's (empty, under /no_think) reasoning block for display."""
    return THINK.sub("", text).strip()


def one_line(text: str, width: int = 90) -> str:
    text = " ".join(visible(text).split())
    return text if len(text) <= width else text[: width - 1] + "…"


# ── HTTP client ──────────────────────────────────────────────────────────────


class Client:
    def __init__(self, host: str, port: int, timeout: float):
        self.host, self.port, self.timeout = host, port, timeout
        self.model = None  # filled from /v1/models

    def get(self, path: str, timeout: float | None = None):
        url = f"http://{self.host}:{self.port}{path}"
        with urllib.request.urlopen(url, timeout=timeout or self.timeout) as r:
            return json.loads(r.read())

    def post(self, path: str, payload, raw: bytes | None = None):
        """POST JSON; returns (status, parsed body) for 2xx and 4xx alike."""
        body = raw if raw is not None else json.dumps(payload).encode()
        req = urllib.request.Request(
            f"http://{self.host}:{self.port}{path}", data=body,
            headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as exc:
            try:
                return exc.code, json.loads(exc.read())
            except Exception:
                return exc.code, None

    def chat(self, messages, **params):
        payload = {"model": self.model, "messages": messages, **params}
        t0 = time.perf_counter()
        status, body = self.post("/v1/chat/completions", payload)
        body = body or {}
        body["_ms"] = (time.perf_counter() - t0) * 1000
        body["_status"] = status
        return body

    def stream(self, path: str, payload: dict, stop_after: int | None = None,
               on_token=None) -> dict:
        """Stream one request.  Returns timing, text, finish reason and usage.

        *stop_after* hangs up after that many content chunks, as a client
        that loses interest would.
        """
        conn = http.client.HTTPConnection(self.host, self.port, timeout=self.timeout)
        body = json.dumps({"model": self.model, "stream": True, **payload})
        t0 = time.perf_counter()
        conn.request("POST", path, body=body,
                     headers={"Content-Type": "application/json"})
        resp = conn.getresponse()
        out = {"status": resp.status, "text": "", "times": [], "finish": None,
               "usage": None, "role_chunk": False, "done": False,
               "hung_up": False}
        if resp.status != 200:
            out["error"] = resp.read().decode(errors="replace")
            conn.close()
            return out
        try:
            while True:
                line = resp.readline()
                if not line:
                    break
                line = line.decode().strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    out["done"] = True
                    break
                event = json.loads(data)
                if event.get("usage"):
                    out["usage"] = event["usage"]
                for choice in event.get("choices", []):
                    delta = choice.get("delta") or {}
                    if delta.get("role") == "assistant":
                        out["role_chunk"] = True
                    text = delta.get("content", choice.get("text", "")) or ""
                    if text:
                        out["times"].append((time.perf_counter() - t0) * 1000)
                        out["text"] += text
                        if on_token:
                            on_token(text)
                    if choice.get("finish_reason"):
                        out["finish"] = choice["finish_reason"]
                if stop_after is not None and len(out["times"]) >= stop_after:
                    out["hung_up"] = True
                    break
        finally:
            conn.close()
        out["e2e_ms"] = (time.perf_counter() - t0) * 1000
        out["ttft_ms"] = out["times"][0] if out["times"] else None
        return out

    def cache(self, settle: float = 2.0) -> dict | None:
        try:
            return self.get(f"/debug/prefix_cache?settle={settle}",
                            timeout=settle + self.timeout)
        except Exception:
            return None


def gpu_stats(report: dict | None) -> dict:
    if not report:
        return {}
    return (report.get("gpu") or {}).get("gpu_stats") or {}


def cached_pages(report: dict | None) -> int:
    """Pages the prefix cache holds right now, in either mode."""
    if not report:
        return 0
    stats = gpu_stats(report)
    if stats.get("enabled") and report.get("gpu_mode") == "owner":
        return int(stats.get("resident", 0))
    return int((report.get("cpu") or {}).get("cache_resident_pages", 0))


def matched_pages(report: dict | None) -> int:
    """Cumulative prefix pages imported so far, in either mode."""
    if not report:
        return 0
    stats = gpu_stats(report)
    if report.get("gpu_mode") == "owner":
        return int(stats.get("match_pages", 0))
    return int((report.get("cpu") or {}).get("imported_pages", 0))


NO_THINK = " /no_think"


# ── scenes ───────────────────────────────────────────────────────────────────


def scene_server(cl: Client, ctx: dict) -> dict:
    scene_header(1, "Server", "Liveness, model listing, and the engine's KV / cache geometry.")
    health = cl.get("/health")
    models = cl.get("/v1/models")
    cl.model = models["data"][0]["id"]
    ok(f"/health → {health}")
    ok(f"/v1/models → {cl.model}")
    rep = cl.cache(settle=0.5)
    result = {"model": cl.model, "health": health}
    if rep:
        pages = rep["pages"]
        mode = rep.get("gpu_mode", "off")
        cpu_on = bool((rep.get("cpu") or {}).get("cache_enabled"))
        where = {"owner": "on the GPU (kernel-owned)",
                 "shadow": "on the CPU, shadowed on the GPU"}.get(
                     mode, "on the CPU" if cpu_on else "off")
        warp = gpu_stats(rep).get("warp")
        info(f"KV pool: {pages['total']} pages · prefix cache {where}"
             + (" · helper lanes on" if warp else ""))
        result.update(pages=pages, mode=mode, cpu_cache=cpu_on)
        if pages.get("page_size"):
            ctx["page_size"] = pages["page_size"]
    else:
        info(yellow("no /debug/prefix_cache endpoint: cache scenes will only time"))
    return result


def scene_api(cl: Client, ctx: dict) -> dict:
    scene_header(2, "OpenAI API", "Chat completion and raw text completion, with usage and finish_reason.")
    r = cl.chat([{"role": "user", "content": "What is the capital of France? One word." + NO_THINK}],
                temperature=0, max_tokens=24)
    ans = r["choices"][0]["message"]["content"]
    ok(f"chat → {one_line(ans)!r}   finish_reason={r['choices'][0]['finish_reason']}"
       f"   usage={r['usage']}")
    status, c = cl.post("/v1/completions", {
        "model": cl.model, "prompt": "The three primary colors of light are",
        "temperature": 0, "max_tokens": 12})
    text = c["choices"][0]["text"]
    shown = " ".join(text.split())
    ok(f"completion → 'The three primary colors of light are"
       f"{' ' if text[:1].isspace() else ''}{shown[:60]}'")
    info(dim(f"   finish_reason={c['choices'][0]['finish_reason']} (hit max_tokens=12)"
             f"   usage={c['usage']}"))
    return {"chat": visible(ans), "chat_usage": r["usage"],
            "completion": text, "completion_finish": c["choices"][0]["finish_reason"]}


def scene_stream(cl: Client, ctx: dict) -> dict:
    scene_header(3, "Streaming", "Server-sent events, printed as they arrive.")
    print("   " + dim("› "), end="", flush=True)
    printed = {"think": True, "buf": ""}

    def show(t):
        printed["buf"] += t
        if printed["think"]:
            # Hold output until the (empty) reasoning block and the blank
            # lines after it are past, then print from the first real text.
            if "</think>" in printed["buf"]:
                rest = printed["buf"].split("</think>", 1)[1].lstrip()
                if rest:
                    printed["think"] = False
                    print(rest, end="", flush=True)
            return
        print(t, end="", flush=True)

    s = cl.stream("/v1/chat/completions", {
        "messages": [{"role": "user", "content":
                      "In two sentences, what does a GPU do?" + NO_THINK}],
        "temperature": 0, "max_tokens": 96,
        "stream_options": {"include_usage": True}}, on_token=show)
    print()
    gaps = [b - a for a, b in zip(s["times"], s["times"][1:])]
    gaps.sort()
    n = len(s["times"])
    rate = (n - 1) / ((s["times"][-1] - s["times"][0]) / 1000) if n > 1 else 0
    ok(f"{n} tokens · TTFT {s['ttft_ms']:.1f} ms · median gap "
       f"{gaps[len(gaps) // 2]:.2f} ms · {rate:.0f} tok/s · finish_reason={s['finish']}")
    ok(f"role chunk first: {s['role_chunk']} · usage chunk: {s['usage']} · [DONE]: {s['done']}")
    return {"tokens": n, "ttft_ms": s["ttft_ms"], "tok_per_s": rate,
            "finish": s["finish"], "usage": s["usage"]}


def _load_tokenizer(model: str):
    try:
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(model)
    except Exception:
        return None


def scene_sampling(cl: Client, ctx: dict) -> dict:
    scene_header(4, "Sampling (in the kernel, per request)",
                 "Every request carries its own settings; the sampler runs inside the megakernel.")
    q = [{"role": "user", "content": "Invent a name for a coffee shop. Reply with the name only." + NO_THINK}]
    out = {}

    def run(max_tokens=24, **p):
        r = cl.chat(q, max_tokens=max_tokens, **p)
        return visible(r["choices"][0]["message"]["content"]), r

    g1, _ = run(temperature=0)
    g2, _ = run(temperature=0)
    (ok if g1 == g2 else bad)(f"greedy twice → identical: {g1!r}")
    seeded = [run(temperature=1.0, seed=7)[0] for _ in range(3)]
    s7a, s7b = seeded[0], seeded[1]
    same = sum(x == seeded[0] for x in seeded)
    s8, _ = run(temperature=1.0, seed=8)
    s9, _ = run(temperature=1.0, seed=9)
    info(green("≈ ") + f"temperature 1, seed 7, three runs → {same}/3 identical: {seeded[0]!r}")
    info(dim("    a seed fixes the random stream, not the logits: SM100 split-K sums in"))
    info(dim("    arrival order, so near-ties can flip run to run (oipl_repro.md §6)"))
    (ok if len({seeded[0], s8, s9}) > 1 else bad)(
        f"other seeds sample differently: seed 8 → {s8!r}, seed 9 → {s9!r}")
    k1, _ = run(temperature=1.0, top_k=1, seed=99)
    (ok if k1 == g1 else bad)(f"top_k=1 at temperature 1 → same as greedy: {k1!r}")
    variety = [run(temperature=0.9, top_p=0.9, top_k=40, seed=s)[0] for s in range(4)]
    info(f"  top_p 0.9 · top_k 40 · seeds 0-3 → " + " | ".join(repr(v) for v in variety))
    out.update(greedy=g1, seed7_runs=seeded, seed7_identical=same, seed8=s8, seed9=s9,
               top_k1=k1, variety=variety)

    _, r = run(temperature=0, max_tokens=5)
    fr, ct = r["choices"][0]["finish_reason"], r["usage"]["completion_tokens"]
    (ok if fr == "length" and ct == 5 else bad)(
        f"max_tokens=5 → finish_reason={fr!r}, completion_tokens={ct}")
    out["max_tokens_5"] = {"finish": fr, "completion_tokens": ct}

    tok = ctx.get("tokenizer")
    yn = [{"role": "user", "content": "Is the sky blue on a clear day? Answer Yes or No." + NO_THINK}]
    plain = visible(cl.chat(yn, temperature=0, max_tokens=12)["choices"][0]["message"]["content"])
    if tok is not None:
        ban = {str(t): -100 for t in {tok.encode(w, add_special_tokens=False)[0]
                                      for w in ("Yes", " Yes", "yes")}}
        biased = visible(cl.chat(yn, temperature=0, max_tokens=12, logit_bias=ban)
                         ["choices"][0]["message"]["content"])
        (ok if "Yes" not in biased.split()[:1] else bad)(
            f"logit_bias bans 'Yes': {plain!r} → {biased!r}")
        out["logit_bias"] = {"plain": plain, "biased": biased}
    else:
        info(dim("  logit_bias: skipped (no local tokenizer to look up token ids)"))

    rep = [{"role": "user", "content": "Write the word ping twelve times, separated by spaces." + NO_THINK}]
    base = visible(cl.chat(rep, temperature=0, max_tokens=40)["choices"][0]["message"]["content"])
    pen = visible(cl.chat(rep, temperature=0, max_tokens=40, frequency_penalty=1.5,
                          presence_penalty=0.5)["choices"][0]["message"]["content"])
    count = lambda s: sum(w.strip(".,").lower() == "ping" for w in s.split())
    info(f"  no penalty      → {count(base):>2}× 'ping': {one_line(base, 60)!r}")
    info(f"  frequency 1.5   → {count(pen):>2}× 'ping': {one_line(pen, 60)!r}")
    (ok if count(pen) < count(base) else bad)("frequency/presence penalties curb repetition")
    out["penalty"] = {"plain_pings": count(base), "penalised_pings": count(pen)}
    return out


SYSTEM_SHOP = (
    "You are the support assistant of Lantern Books, an online bookstore. "
    "Store hours are 9am to 10pm every day. Returns are accepted within seven "
    "days if the book is undamaged; the buyer pays return shipping unless the "
    "book has a printing defect, in which case we pay. Members come in three "
    "tiers: standard, silver and gold. Silver members get five percent off, "
    "gold members ten percent, and every member gets an extra twenty percent "
    "off during their birthday month. Orders over ninety-nine dollars ship "
    "free except to remote regions. Orders ship within twenty-four hours, "
    "later on public holidays. For stock questions, point customers to the "
    "product page. Gift wrapping costs three dollars per item. E-books are "
    "delivered by email within five minutes and cannot be returned once "
    "downloaded. Pre-orders are charged when the book ships. We accept credit "
    "cards, debit cards and store gift cards; gift cards never expire. ")


def scene_prefix(cl: Client, ctx: dict) -> dict:
    scene_header(5, "Prefix cache",
                 "Four customers share one long system prompt; a fifth has a different one.")
    system = SYSTEM_SHOP * 3
    other = SYSTEM_SHOP.replace("Lantern Books", "Harbor Novels").replace(
        "nine", "eight") * 3
    questions = [("A", system, "Can a gold member stack the birthday discount?"),
                 ("B", system, "Who pays shipping for a misprinted book?"),
                 ("C", system, "How fast do e-books arrive?"),
                 ("D", system, "Do gift cards expire?"),
                 ("E", other, "Who pays shipping for a misprinted book?")]
    rows, out = [], []
    before = cl.cache(settle=1.0)
    for who, sysmsg, q in questions:
        s = cl.stream("/v1/chat/completions", {
            "messages": [{"role": "system", "content": sysmsg},
                         {"role": "user", "content": q + NO_THINK}],
            "temperature": 0, "max_tokens": 48,
            "stream_options": {"include_usage": True}})
        after = cl.cache(settle=1.0)
        reused = matched_pages(after) - matched_pages(before)
        page = (ctx.get("page_size") or 64)
        prompt = s["usage"]["prompt_tokens"] if s["usage"] else None
        computed = prompt - reused * page if prompt is not None else None
        label = "shared" if sysmsg is system else "different"
        rows.append([who, label, prompt, reused, computed, f"{s['ttft_ms']:.1f}",
                     one_line(s["text"], 38)])
        out.append({"who": who, "system": label, "prompt_tokens": prompt,
                    "pages_reused": reused, "tokens_computed": computed,
                    "ttft_ms": s["ttft_ms"], "answer": visible(s["text"])})
        before = after
    table(rows, ["req", "system", "prompt tok", "pages reused", "tok computed",
                 "TTFT ms", "answer"])
    warm = [o["ttft_ms"] for o in out[1:4]]
    cold = out[0]["ttft_ms"]
    if before is None:
        info(yellow("no cache counters on this server"))
    ok(f"cold {cold:.1f} ms → warm {sum(warm) / len(warm):.1f} ms on average "
       f"({cold / (sum(warm) / len(warm)):.1f}× faster first token); "
       f"the different-prompt request E is cold again: {out[4]['ttft_ms']:.1f} ms")
    return {"requests": out}


def scene_multiturn(cl: Client, ctx: dict) -> dict:
    scene_header(6, "Multi-turn",
                 "Each turn resends the whole conversation; the previous turn's KV is reused.")
    system = ("You are a meticulous lab assistant. Keep every answer to one short "
              "sentence. Track what the user tells you exactly. ") * 12
    turns = ["My sample code is ZX-4417. Please remember it.",
             "The sample was stored at minus eighty degrees. Note that too.",
             "What is my sample code, and at what temperature is it stored?"]
    messages = [{"role": "system", "content": system}]
    rows, out = [], []
    before = cl.cache(settle=1.0)
    for i, user in enumerate(turns, 1):
        messages.append({"role": "user", "content": user + NO_THINK})
        s = cl.stream("/v1/chat/completions", {
            "messages": messages, "temperature": 0, "max_tokens": 48,
            "stream_options": {"include_usage": True}})
        after = cl.cache(settle=1.0)
        reused = matched_pages(after) - matched_pages(before)
        before = after
        answer = visible(s["text"])
        messages.append({"role": "assistant", "content": answer})
        prompt = s["usage"]["prompt_tokens"] if s["usage"] else None
        rows.append([i, prompt, reused, f"{s['ttft_ms']:.1f}", one_line(answer, 52)])
        out.append({"turn": i, "prompt_tokens": prompt, "pages_reused": reused,
                    "ttft_ms": s["ttft_ms"], "answer": answer})
    table(rows, ["turn", "prompt tok", "pages reused", "TTFT ms", "answer"])
    final = out[-1]["answer"]
    recalled = "4417" in final
    (ok if recalled else bad)(f"turn 3 recalls the code from turn 1: {recalled}")
    return {"turns": out, "recalled": recalled}


def scene_concurrent(cl: Client, ctx: dict) -> dict:
    scene_header(7, "Concurrency and isolation",
                 "Eight streams at once on a 4-row batch; each must get its own marker back.")
    n = 8
    results: list[dict | None] = [None] * n
    markers = [f"QX-{1000 + 37 * i}" for i in range(n)]

    def worker(i):
        results[i] = cl.stream("/v1/chat/completions", {
            "messages": [{"role": "user", "content":
                          f"Repeat this code exactly and nothing else: {markers[i]}" + NO_THINK}],
            "temperature": 0, "max_tokens": 32})

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
    t0 = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = (time.perf_counter() - t0) * 1000
    own = sum(markers[i] in (results[i] or {}).get("text", "") for i in range(n))
    foreign = sum(any(m in (results[i] or {}).get("text", "")
                      for j, m in enumerate(markers) if j != i) for i in range(n))
    ttfts = sorted(r["ttft_ms"] for r in results if r and r["ttft_ms"])
    (ok if own == n and foreign == 0 else bad)(
        f"{own}/{n} streams returned their own marker, {foreign} saw another's")
    ok(f"all {n} done in {wall:.0f} ms · TTFT min {ttfts[0]:.0f} / max {ttfts[-1]:.0f} ms "
       f"(two waves on 4 rows)")
    return {"own": own, "foreign": foreign, "wall_ms": wall, "ttft_ms": ttfts}


def scene_cancel(cl: Client, ctx: dict) -> dict:
    scene_header(8, "Client disconnect",
                 "A client hangs up after 8 tokens of a long answer; nothing may leak.")
    abandoned = lambda rep: int(((rep or {}).get("cpu") or {}).get("abandoned_requests", 0))
    before = abandoned(cl.cache(settle=0.5))
    s = cl.stream("/v1/chat/completions", {
        "messages": [{"role": "user", "content":
                      "Write a long story about a lighthouse keeper." + NO_THINK}],
        "temperature": 0.7, "seed": 3, "max_tokens": 400}, stop_after=8)
    ok(f"hung up after {len(s['times'])} tokens ({s['e2e_ms']:.0f} ms in)")
    t0 = time.perf_counter()
    quiet = None
    for _ in range(60):
        rep = cl.cache(settle=0.2)
        if rep is None:
            break
        g = rep.get("gpu") or {}
        if g.get("quiescent", True) and rep["pages"]["pending_returns"] == 0:
            quiet = rep
            break
        time.sleep(0.25)
    settle_ms = (time.perf_counter() - t0) * 1000
    out = {"tokens_before_hangup": len(s["times"]), "settle_ms": settle_ms}
    if quiet is not None:
        n = abandoned(quiet) - before
        (ok if n == 1 else bad)(
            f"the disconnect reached the engine: {n} request abandoned "
            f"(its tokens are no longer decoded or queued)")
        info(dim("    the kernel has no mid-flight cancel: it finishes the request, bounded"))
        info(dim("    by max_tokens, and the row comes back when it does"))
        out["abandoned"] = n
    if quiet is not None:
        pg = quiet["pages"]
        held = cached_pages(quiet)
        conserved = pg["gpu_free"] + pg["pending_returns"] + held == pg["total"]
        (ok if conserved else bad)(
            f"server idle again after {settle_ms:.0f} ms: {pg['gpu_free']} free + "
            f"{held} cached = {pg['total']} pages")
        out["conserved"] = conserved
    # Every row must be usable: four concurrent requests fill the whole batch.
    t1 = time.perf_counter()
    res = [None] * 4

    def w(i):
        res[i] = cl.chat([{"role": "user", "content": f"Say the number {i}." + NO_THINK}],
                         temperature=0, max_tokens=16)

    th = [threading.Thread(target=w, args=(i,)) for i in range(4)]
    for t in th:
        t.start()
    for t in th:
        t.join()
    good = sum(1 for r in res if r and r.get("_status") == 200)
    (ok if good == 4 else bad)(
        f"4 concurrent follow-up requests served in {(time.perf_counter() - t1) * 1000:.0f} ms "
        f"({good}/4 OK): no row was left behind")
    out["follow_up_ok"] = good
    return out


def scene_errors(cl: Client, ctx: dict) -> dict:
    scene_header(9, "Validation", "Bad requests get OpenAI-shaped errors, never a crash.")
    msg = [{"role": "user", "content": "hi"}]
    cases = [
        ("unknown model", {"model": "gpt-4", "messages": msg}, None),
        ("temperature = -1", {"model": cl.model, "messages": msg, "temperature": -1}, None),
        ("top_p = 0", {"model": cl.model, "messages": msg, "top_p": 0}, None),
        ("logit_bias 200", {"model": cl.model, "messages": msg, "logit_bias": {"1": 200}}, None),
        ("empty messages", {"model": cl.model, "messages": []}, None),
        ("not JSON", None, b"{oops"),
    ]
    rows, out = [], []
    for name, payload, raw in cases:
        status, body = cl.post("/v1/chat/completions", payload, raw=raw)
        err = (body or {}).get("error", {})
        rows.append([name, status, err.get("param") or "", one_line(err.get("message", ""), 46)])
        out.append({"case": name, "status": status, "param": err.get("param")})
    table(rows, ["request", "HTTP", "param", "message"])
    still = cl.chat(msg, temperature=0, max_tokens=8)
    (ok if still.get("_status") == 200 else bad)("server still answers normally afterwards")
    return {"cases": out}


def scene_load(cl: Client, ctx: dict) -> dict:
    scene_header(10, "Load and eviction",
                 "24 different system prompts overflow the cache; it evicts on demand while serving.")
    groups, followers, width = 24, 2, 8
    names = [f"Store-{chr(65 + g // 6)}{g % 6}" for g in range(groups)]
    system = lambda g: SYSTEM_SHOP.replace("Lantern Books", names[g]) * 3
    qs = ["Who pays shipping for a misprinted book?", "Do gift cards expire?",
          "How fast do e-books arrive?"]

    def burst(jobs):
        res: list[dict | None] = [None] * len(jobs)
        nxt = {"i": 0}
        lock = threading.Lock()

        def worker():
            while True:
                with lock:
                    i = nxt["i"]
                    nxt["i"] += 1
                if i >= len(jobs):
                    return
                g, q = jobs[i]
                res[i] = cl.stream("/v1/chat/completions", {
                    "messages": [{"role": "system", "content": system(g)},
                                 {"role": "user", "content": q + NO_THINK}],
                    "temperature": 0, "max_tokens": 16})
        th = [threading.Thread(target=worker) for _ in range(width)]
        t0 = time.perf_counter()
        for t in th:
            t.start()
        for t in th:
            t.join()
        return res, time.perf_counter() - t0

    start = cl.cache(settle=1.0)
    leaders, t_lead = burst([(g, qs[0]) for g in range(groups)])
    mid = cl.cache(settle=2.0)
    jobs = [(g, qs[1 + k]) for k in range(followers) for g in range(groups)]
    follow, t_follow = burst(jobs)
    end = cl.cache(settle=3.0)
    ttft = lambda rs: sorted(r["ttft_ms"] for r in rs if r and r["ttft_ms"])
    p50 = lambda xs: xs[len(xs) // 2] if xs else float("nan")
    ok_count = sum(1 for r in leaders + follow if r and r["status"] == 200 and r["done"])
    total = len(leaders) + len(follow)
    (ok if ok_count == total else bad)(f"{ok_count}/{total} requests completed")
    info(f"  leaders   ({len(leaders)}, all cold) in {t_lead:.2f} s · TTFT p50 {p50(ttft(leaders)):.0f} ms")
    info(f"  followers ({len(follow)}, same prompts) in {t_follow:.2f} s · "
         f"TTFT p50 {p50(ttft(follow)):.0f} ms")
    out = {"requests": total, "completed": ok_count,
           "leader_ttft_p50": p50(ttft(leaders)), "follower_ttft_p50": p50(ttft(follow)),
           "leader_s": t_lead, "follower_s": t_follow}
    st0, st1, st2 = gpu_stats(start), gpu_stats(mid), gpu_stats(end)
    if st2:
        hits = st2.get("match_pages", 0) - st1.get("match_pages", 0)
        best = len(follow) * 8
        ev = st2.get("evicted", 0) - st0.get("evicted", 0)
        ok(f"followers reused {hits} of at most {best} prompt pages; "
           f"{ev} blocks evicted to make room, {st2.get('admit_hold', 0)} admissions held")
        out.update(follower_pages_hit=hits, follower_pages_possible=best, evicted=ev)
    if end:
        pg, held = end["pages"], cached_pages(end)
        conserved = pg["gpu_free"] + pg["pending_returns"] + held == pg["total"]
        (ok if conserved else bad)(
            f"after the burst: {pg['gpu_free']} free + {held} cached = {pg['total']} pages")
        out["conserved"] = conserved
    return out


def scene_internals(cl: Client, ctx: dict) -> dict:
    scene_header(11, "GPU prefix cache internals",
                 "Read back from the kernel once the engine is idle (GET /debug/prefix_cache).")
    rep = cl.cache(settle=3.0)
    if not rep or "gpu" not in rep:
        info(yellow("this server has no GPU prefix cache (start it with --gpu-prefix-cache)"))
        return {}
    g, st, pg = rep["gpu"], rep["gpu"]["gpu_stats"], rep["pages"]
    held = cached_pages(rep)
    conserved = pg["gpu_free"] + pg["pending_returns"] + held == pg["total"]
    (ok if conserved else bad)(
        f"page conservation: {pg['gpu_free']} free + {held} cached = {pg['total']} pages")
    if rep.get("gpu_mode") == "owner":
        (ok if g.get("structure_errors") == 0 else bad)(
            f"index structure: {g.get('resident')} blocks, {g.get('leaves')} leaves, "
            f"{g.get('pinned')} pinned, {g.get('structure_errors')} errors")
    else:
        (ok if g.get("mismatches") == 0 else bad)(
            f"GPU index vs CPU cache: {g.get('mismatches')} mismatches")
    table([[st.get("completions"), st.get("match_pages"), st.get("inserted"),
            st.get("deduped"), st.get("evicted"), st.get("freed"), st.get("admit_hold")]],
          ["completions", "pages hit", "inserted", "deduped", "evicted", "freed", "holds"])
    cost = [f"match {st.get('match_us_per_call', '–')} µs",
            f"insert {st.get('insert_us_per_completion', '–')} µs"]
    if st.get("evict_us_per_block"):
        cost.append(f"evict {st['evict_us_per_block']} µs/block")
    info("cost on the scheduler lane: " + " · ".join(cost)
         + ("  (31 helper lanes)" if st.get("warp") else "  (serial)"))
    return {"pages": pg, "cached": held, "conserved": conserved,
            "structure_errors": g.get("structure_errors"),
            "mismatches": g.get("mismatches"), "stats": st}


SCENES = [scene_server, scene_api, scene_stream, scene_sampling, scene_prefix,
          scene_multiturn, scene_concurrent, scene_cancel, scene_errors,
          scene_load, scene_internals]


# ── server launch ────────────────────────────────────────────────────────────


def launch_server(args) -> subprocess.Popen:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    try:  # mirage's runtime links libz3 from the z3 wheel
        import z3
        lib = os.path.join(os.path.dirname(z3.__file__), "lib")
        env["LD_LIBRARY_PATH"] = lib + os.pathsep + env.get("LD_LIBRARY_PATH", "")
    except ImportError:
        pass
    cmd = [sys.executable, "-m", "mirage.engine.launch_server",
           "--model", args.model, "--port", str(args.port),
           "--page-size", str(args.page_size),
           "--max-seq-length", str(args.max_seq_length),
           "--max-num-pages", str(args.max_num_pages),
           "--max-num-batched-requests", "4", "--max-num-batched-tokens", "8",
           "--output-dir", args.output_dir]
    cmd += {"gpu": ["--gpu-prefix-cache"], "cpu": ["--enable-prefix-cache"],
            "shadow": ["--enable-prefix-cache", "--gpu-prefix-cache"],
            "off": []}[args.cache]
    os.makedirs(args.output_dir, exist_ok=True)
    log = open(args.server_log, "w")
    print(dim(f"launching: {' '.join(cmd)}"))
    print(dim(f"server log: {args.server_log}"))
    proc = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT,
                            start_new_session=True)
    t0 = time.time()
    while time.time() - t0 < args.ready_timeout:
        if proc.poll() is not None:
            raise SystemExit(f"server exited during startup; see {args.server_log}")
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{args.port}/health", timeout=2)
            print(dim(f"server ready after {time.time() - t0:.0f} s (compiles the megakernel)"))
            return proc
        except Exception:
            time.sleep(2)
    proc.kill()
    raise SystemExit(f"server not ready after {args.ready_timeout:.0f} s; see {args.server_log}")


def stop_server(proc: subprocess.Popen) -> None:
    try:
        os.killpg(proc.pid, signal.SIGINT)
        proc.wait(timeout=30)
    except Exception:
        os.killpg(proc.pid, signal.SIGKILL)


# ── main ─────────────────────────────────────────────────────────────────────


def main() -> int:
    global USE_COLOR
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--scenes", default="all",
                   help="comma-separated scene numbers, e.g. 1,5,6 (default: all)")
    p.add_argument("--timeout", type=float, default=300.0)
    p.add_argument("--report", default=None, help="write all measurements as JSON")
    p.add_argument("--no-color", action="store_true")
    g = p.add_argument_group("--launch: start a server for the demo")
    g.add_argument("--launch", action="store_true")
    g.add_argument("--gpu", default="0")
    g.add_argument("--model", default="Qwen/Qwen3-0.6B")
    g.add_argument("--cache", choices=["gpu", "cpu", "shadow", "off"], default="gpu",
                   help="prefix cache mode of the launched server (default: gpu)")
    g.add_argument("--page-size", type=int, default=64)
    g.add_argument("--max-seq-length", type=int, default=2048)
    g.add_argument("--max-num-pages", type=int, default=256)
    g.add_argument("--output-dir", default=os.path.expanduser("~/.cache/mirage_demo/mpk"))
    g.add_argument("--server-log", default=os.path.expanduser("~/.cache/mirage_demo/server.log"))
    g.add_argument("--ready-timeout", type=float, default=900.0)
    args = p.parse_args()
    if args.no_color:
        USE_COLOR = False

    picked = (range(1, len(SCENES) + 1) if args.scenes == "all"
              else [int(x) for x in args.scenes.split(",")])
    proc = launch_server(args) if args.launch else None
    cl = Client(args.host, args.port, args.timeout)
    ctx = {"page_size": args.page_size, "tokenizer": None}
    report = {"started": time.strftime("%Y-%m-%d %H:%M:%S"), "scenes": {}}
    failed = []
    try:
        cl.model = cl.get("/v1/models")["data"][0]["id"]
        if 4 in picked:
            ctx["tokenizer"] = _load_tokenizer(cl.model)
        print(bold(f"\nMPK online serving showcase · {cl.model} · {args.host}:{args.port}"))
        for n in picked:
            fn = SCENES[n - 1]
            try:
                report["scenes"][fn.__name__[6:]] = fn(cl, ctx)
            except Exception as exc:
                failed.append(n)
                bad(f"scene {n} raised {type(exc).__name__}: {exc}")
    finally:
        if proc is not None:
            stop_server(proc)
    print()
    print(bold("done") + (red(f" · scenes with errors: {failed}") if failed
                          else green(" · every scene ran")))
    if args.report:
        with open(args.report, "w") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(dim(f"report: {args.report}"))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
