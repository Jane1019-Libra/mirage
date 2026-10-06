# Online serving showcase

Branch `oipl-demo`: one serving stack that combines everything in flight for MPK online serving.

| Source | What it brings |
|---|---|
| `zhendonghua/mirage:oipl-base` (jackeyhua) | OIPL page lifecycle, CPU prefix cache, multi-turn chat rendering |
| PR #1 (`oipl-pinned-write`) | batched pinned writes, lazy-heap eviction, batch-sized shared-memory snapshot |
| `oipl-gpu-cache` | GPU-resident, kernel-owned prefix cache with helper-lane work (`gpu_prefix_cache.md`) |
| `zhendonghua/mirage#2` (JingfengChen) | streaming monitor: one batched GPU→CPU copy per poll, disconnect cancellation |
| `mirage-project/mirage#784` (GiantPrince) | pydantic OpenAI layer, per-request sampling inside the megakernel, finish reasons |

`demo/qwen3/demo_serving_showcase.py` exercises all of it against a live server in eleven scenes.

---

## 1. Running it

```bash
# environment as in oipl_repro.md §2 (venv, CUDA_HOME, z3 on LD_LIBRARY_PATH, HF_HOME)

# one command: launch a server on GPU 0, run every scene, stop the server
python demo/qwen3/demo_serving_showcase.py --launch --gpu 0

# or attach to a server you started
python -m mirage.engine.launch_server --model Qwen/Qwen3-0.6B \
  --page-size 64 --max-seq-length 2048 --max-num-pages 256 \
  --max-num-batched-requests 4 --max-num-batched-tokens 8 \
  --gpu-prefix-cache --port 8000
python demo/qwen3/demo_serving_showcase.py --port 8000
```

Useful flags:

| Flag | Effect |
|---|---|
| `--scenes 1,5,6` | run a subset (scene 1 should come first: it reads the server's page size) |
| `--cache cpu` / `off` / `shadow` | with `--launch`, start the CPU cache, no cache, or both caches instead of the GPU cache, for a live A/B |
| `--report out.json` | save every measurement |
| `--model …` | any Qwen3 checkpoint (#784's sampler is registered for Qwen3 only) |

The first launch compiles the megakernel (about 50 s on a B200). A full run takes about four minutes. Keep the GPU to yourself: a persistent kernel sharing a GPU with another job can stall.

---

## 2. Scenes

Numbers are from the recorded run on one B200 with Qwen3-0.6B, page size 64, 256 pages, 4 rows, 8 tokens per iteration. Full transcript: `~/oipl/bench/demo_showcase_transcript.txt`; JSON: `~/oipl/bench/demo_showcase_report.json`.

| # | Scene | What it shows | Recorded result | From |
|---|---|---|---|---|
| 1 | Server | `/health`, `/v1/models`, KV pool and cache mode | 256 pages, cache on the GPU, helper lanes on | #784, GPU cache |
| 2 | OpenAI API | chat and raw `/v1/completions`; `usage`; `finish_reason` | `stop` for a natural end, `length` at `max_tokens` | #784 |
| 3 | Streaming | SSE: role chunk, deltas, finish chunk, usage chunk, `[DONE]` | TTFT 11 ms, median gap 1.37 ms, about 740 tok/s | #784, #2 |
| 4 | Sampling | greedy, seeds, top-k/top-p, `max_tokens`, `logit_bias`, penalties, all per request, in the kernel | `logit_bias` turns "Yes." into "No."; frequency penalty 9→6 repeats | #784 |
| 5 | Prefix cache | four requests share a 570-token system prompt; a fifth does not | 8 pages reused, 61 of 573 tokens computed; TTFT 136 → 22 ms (6.3×); the outsider is cold again | GPU cache |
| 6 | Multi-turn | the whole conversation is resent each turn | turn 2 reuses 4 pages, turn 3 reuses 5; turn 3 recalls turn 1's code | oipl-base, GPU cache |
| 7 | Concurrency | 8 simultaneous streams on 4 rows, each with a private marker | 8/8 own marker, 0 foreign; done in about 100 ms | oipl-base, #2 |
| 8 | Client disconnect | client hangs up after 8 tokens of a 400-token answer | 1 request abandoned; server idle in about 0.6 s; all pages back; 4 follow-ups served | #2 + integration fix |
| 9 | Validation | unknown model, out-of-range params, empty messages, bad JSON | 404 / 400 with the offending `param`; server unaffected | #784 |
| 10 | Load and eviction | 24 distinct system prompts (24 cold leaders, then 48 followers), 8 at a time | 72/72 done; followers reuse 334 of 384 possible pages while 104 blocks are evicted; 0 admission holds | GPU cache |
| 11 | Cache internals | read back from the kernel when idle | page conservation holds; 185 blocks, 0 structural errors; match 12 µs, insert 21 µs, evict 5 µs per block | GPU cache |

What to point at while presenting:

- **Scene 5** is the headline: the same request is 6× faster to its first token once its prefix is cached, and a request that shares nothing pays the cold price.
- **Scene 10** is the GPU cache's design in action. Cached pages are pool slack: the kernel evicts them at the admission gate when a new request needs room, so no request ever waited (0 holds) even with the pool full.
- **Scene 4**: every request carries its own sampling settings into the kernel; there is no server-wide sampling flag any more.

---

## 3. Performance on the merged stack

Same geometry and workload as `gpu_prefix_cache.md` §4.3 (1024 tokens, 128 pages, 4 rows, 8 tokens; `bench_serving.py --workload both --groups 4 --prompts-per-group 8 --prefix-tokens 320 --question-tokens 24 --seed 0`, greedy). "Before" is the `oipl-gpu-cache` matrix; "after" is this branch. Both are single runs on one idle B200.

| rate | workload | metric | OFF before | OFF after | CPU cache before | CPU cache after | **GPU cache before** | **GPU cache after** |
|---|---|---|---|---|---|---|---|---|
| 1 | shared prefix | TTFT p50 | 103.7 ms | 100.9 ms | 35.5 ms | 33.0 ms | 35.2 ms | **32.8 ms** |
| 24 | shared prefix | throughput | 9.40 rps | 10.19 rps | 13.98 rps | 15.29 rps | 18.93 rps | **19.02 rps** |
| 24 | shared prefix | TTFT p50 | 1111 ms | 965 ms | 753 ms | 584 ms | 167 ms | **62 ms** |
| 24 | random | throughput | 9.33 rps | 10.14 rps | 8.12 rps | 8.96 rps | 9.16 rps | **10.25 rps** |

- Nothing regressed. The in-kernel sampler from #784 now replaces the argmax in every online decode step, and decode time per token (TPOT) moved by at most 6% either way (largest: +5.6% on the uncached random control at 1 rps).
- At 24 rps the GPU cache serves +87% over no cache and +24% over the CPU cache, with TTFT p50 9× lower than the CPU cache's.
- The baselines moved too (OFF +8%, CPU cache +9% at 24 rps). That comes from the code merged underneath, most likely upstream's scheduler packing (#775) and #2's faster monitor. These runs do not isolate the cause, so the GPU-vs-CPU gap is quoted against the CPU cache measured on this branch (+24%), not against the earlier +35%.

---

## 4. Validation on the merged stack

| Layer | Result |
|---|---|
| `pytest tests/serving_python/` | 101 passed (includes #2's monitor tests) |
| `pytest tests/runtime_python/test_openai_server.py` (#784) | 13 passed |
| `pytest tests/runtime_python/test_serving_sampling_layer.py` (#784, CUDA) | 7 passed: greedy, seed, the sampled distributions for temperature, top-k and top-p, and penalties |
| `test_argmax_vocab_bound_testmode.py` (#2) | passed |
| `oipl_protocol_sim.py --sweep` | all green |
| `test_gpu_prefix_cache.py` | 38/38 serial, 38/38 warp-cooperative |
| L1 paging kernels | 3/3, same numerics as before (`max diff 7.812e-03`) |
| L2 `online_harness.py` | 6 pass / 1 skip in each of GPU-owner, CPU-cache, shadow (0 mismatches), and page size 4096 |
| Demo | 11/11 scenes, 26 checks passed, 0 failed |

---

## 5. What combining the branches needed

Each item is a separate commit on `oipl-demo`. The first three are worth upstreaming to the PR they touch.

1. **HTTP disconnects never reached #2's cancellation.** #784's `_stream_bridge` thread iterated the engine stream to the end whether or not anyone was listening. It now closes the engine stream when the SSE consumer goes away. Measured on one hang-up: 0 requests abandoned with #784's bridge, 1 with the fix.
2. **#784 added a second `TokenizerManager.tokenize_messages`** that silently replaced OIPL's, dropping its message normalization. The duplicate is removed; chat requests still render the whole conversation, which is what multi-turn prefix reuse depends on.
3. **#2 allocated a pinned host buffer on every 0.2 ms poll.** An allocation issued while the persistent kernel runs is the pattern behind the earlier `.clone()` readback hang. Reads now stage through one buffer allocated before launch.
4. **Runtime:** OIPL replaced upstream's ring lock and waiting queue with an owner thread, so #784's per-request `ServingConfig` was ported onto the admission path. #784 also rejected `initial_step != 0` because upstream has no cache; that check is dropped here.
5. **Kernel:** Step 1 stops on the per-request finish reason and still runs the GPU-cache insert. Step 4 copies the request's config next to the OIPL ledger. Meta tensors 33..35 hold the new pointers.
6. **Helper lanes** are sized by upstream's `SCHEDULERS_PER_BLOCK` (#775).
7. **Clients and tests:** the harness and benchmark send `model` and `temperature: 0`, because #784 requires the model name and defaults to the OpenAI temperature of 1.0. The L1 extension defines the macros the SM100 sampler now needs. #784's sampling test fixture carries the OIPL geometry fields.
8. **Observability:** `/debug/prefix_cache` takes its page counters at the same idle moment as the index snapshot and reports the page size. `page_stats` counts abandoned requests.

---

## 6. Limits to state when presenting

- **Seeded sampling is reproducible only as far as the logits are.** SM100's split-K sums partials in arrival order (`oipl_repro.md` §6), so near-ties can flip from run to run. Short answers are usually stable; the recorded run shows 2/3 identical for one seed. Long greedy answers fork the same way: 5 distinct texts in 6 runs of a 120-token answer.
- **Disconnect does not stop the GPU mid-request.** The engine abandons the stream, but the kernel finishes the request, bounded by `max_tokens`, before its row comes back. A kernel-side cancel channel would be the next step.
- **OpenAI defaults apply.** Without a `temperature`, a request is sampled at 1.0. Clients that compare text must send `temperature: 0`.
- **Single GPU and Qwen3 only.** Tensor parallelism is not supported by any of these branches, and #784's `ModelRunner` refuses non-Qwen3 models.
