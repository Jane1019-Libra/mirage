# GPU-resident prefix cache (experimental)

Branch `oipl-gpu-cache`, on top of `oipl-base` + PR #1 (the three polish changes in
`oipl_polish.md`). This moves the prefix-cache index from the CPU owner thread into the
persistent kernel's scheduler lane, in six verified increments. It is a design experiment
answering one question from the team meeting: *can the prefix cache live on the GPU, and
is it worth it?*

The short answer from the measurements in §4: yes, and at saturation it is clearly better than
the CPU cache on the same geometry, because the GPU matches at admission (no publish-to-admit
gap) and treats every cached page as pool slack (no pin budget, no staging holds). The serial
cost on the scheduler lane was the risk; the helper-lane version (§3.6) removes most of it.

---

## 1. What was wrong with keeping it on the CPU

The OIPL design (`oipl_design.md` §6.4) keeps the index on the CPU and moves pages through
three pinned channels. That was the right call for a first version, but it has three structural
costs the GPU version does not pay:

1. **The match is stale by construction.** The CPU matches at *publish*; the GPU admits the
   request one or more iterations later. A block another request finished in between is not
   seen. The GPU matches at *admission*, on the prompt already sitting in device memory.
2. **The pin budget and the staging watermark hold requests back.** A published request pins
   its prefix until it is admitted, so the CPU caps pins (`publish_pin_budget`) and refuses to
   publish when the free-page estimate is low (`_staging_room`, counted as `publish_holds`).
   With the GPU owning the pool, a matched block is pinned only from admission to completion,
   and cached pages are evicted on demand at the admission gate, so nothing is ever held back
   for a page the cache could give up.
3. **Capacity is `total − max_pages_per_req`.** The CPU reserves one worst-case request out of
   the cache's reach because it cannot evict synchronously with admission. The GPU can, so
   the cache may use the whole pool.

The doc's own rejection of "Option 1" (GPU-side index) cited scheduler-lane cost and the lack of
a place to put the work. Both are answered below: the work is measured (§4.3), and the 31 idle
lanes of the scheduler warp are that place (§3.6).

---

## 2. Semantics kept exactly

The device index is not an approximation of the CPU `PrefixCache`; increment 2 proved they are
the same function by running both on the same traffic and diffing them (§4.1).

| Rule | Where |
|---|---|
| Chained hash `h_j = H(h_{j-1}, ids_j)`; hit verified against every stored id **and** the parent slot (§6.4 v2.1-b) | `pc_find` |
| Full-hit clamp: match at most `(prompt_len − 1) / PS` blocks so one token is left to prefill | `pc_match` |
| Cacheable range `min(prompt_len, final_kv_len) − gen_tail_len`; only full pages | Step 1 |
| Chain continues from the imported pages; if an imported block is gone the tail is dropped, never re-parented | `pc_insert_sequence` |
| Dedup against a block another request cached first; the redundant page is freed | `pc_insert_sequence` |
| Leaf-first eviction, never a pinned block | `pc_evict_one` |
| `initial_step = npp · PS`, page-aligned, page table = imported pages then fresh ones | Step 4 |

---

## 3. The six increments

Each increment was built, then verified, before the next (`git log` on the branch has one
commit per increment with its numbers).

### 3.1 Data structure (`gpu_prefix_cache.cuh`)

Open-addressing table of `PrefixEntry {hash, page_id, parent_slot, refcount, children, lru,
ref, ids[PS]}` with linear probing; `PC_TABLE_SIZE` is a power of two ≥ 2× the page pool, so a
full pool of cached pages fills at most half the table. Size at PS = 64: 544 B per entry,
8.9 MB for an 8192-page pool. Unit test in `tests/runtime_python/blackwell/gpu_prefix_cache/`
builds it with 4-token pages and a 64-slot table via `torch.utils.cpp_extension.load_inline`:
**38 checks, run twice (serial and warp-cooperative)**.

### 3.2 Shadow insert/erase (mirror of the CPU cache)

Two device tensors ride `meta_tensors[31..32]`: the table and a 32-int control block
(parameters in, counters out). In shadow mode the kernel indexes each completion in Step 1 and
erases the block a returned page owned in Step 0, so the GPU table should equal the CPU cache
whenever the engine is idle. `GET /debug/prefix_cache` waits for quiescence and diffs them.

### 3.3 Shadow match

Step 4 also matches the prompt and compares against what the CPU imported for the same
request. Both operations are timed with `%globaltimer` into 64-bit accumulators so their cost is
a number.

### 3.4 Owner mode (`--gpu-prefix-cache` alone)

Step 4 pops the row first (the match needs the prompt in device memory; the inbox is pinned host
memory and would be hashed across PCIe), matches, pins the chain, evicts on demand until the
worst-case budget is uncommitted, and hands the row back if every leaf is pinned. Step 1 unpins
the imported blocks, indexes the cacheable pages and pushes every page that owns no block
(deduped, partial, generated) straight back to the free queue. The CPU neither matches nor
returns pages; the export ring is read-only. Eviction is a CLOCK sweep (reference bit set on
insert and hit, cleared on first visit, evicted on second) over leaves with `refcount == 0`; a
fruitless full sweep sets a stall flag that the next completion clears.

### 3.5 Measurement

§4.

### 3.6 Helper lanes

`prepare_next_batch` runs on lane 0 of a scheduler warp; lanes 1..31 of that warp did nothing
for the life of the kernel. They now wait on a per-warp shared-memory mailbox and run the match,
the completion insert and the eviction sweep 31-wide (`pc_w_*`): each lane mixes two of the
64 ids and the block hash is an XOR-reduction (the serial hash computes the identical value, the
unit test checks it bit for bit), the id compare is one `__all_sync`, the sweep inspects 31
slots per step with a ballot. Lane 0's code path is untouched: it posts, spins on the reply,
reads the result. `MPK_GPU_PREFIX_SERIAL=1` keeps the serial path for an A/B on the same build.

---

## 4. Measurements

Everything `oipl_repro.md` §4 and `oipl_benchmarks.md` run for the CPU cache, re-run with the
kernel owning the cache (`--gpu-prefix-cache`, helper lanes on), OFF and CPU cache alongside as
controls. Qwen3-0.6B, B200, one uncontended GPU per run, same seed across configurations. Raw
JSON, server logs and `/debug/prefix_cache` snapshots: `~/oipl/bench/gpc_matrix/`.

### 4.1 Test pyramid

| Layer | Command | Result |
|---|---|---|
| L0 | `pytest tests/serving_python/` | **106 passed** |
| L0 | `oipl_protocol_sim.py --sweep --events 300000 --seeds 2` | all green |
| L0 | `test_gpu_prefix_cache.py` (device index, 4-token pages) | **38/38 serial, 38/38 warp-cooperative** |
| L1 | `test_page_geometry` / `test_page_boundary_testmode` / `test_window_skip` (rebuilt: `runtime_header.h` changed) | **3/3**, same numerics (`max diff 7.812e-03`) |
| L2 | `online_harness.py`, GPU owner mode, `--page-size 64 --max-num-pages 64` | **6 pass / 1 skip**, pages conserved (63 free + 1 cached = 64) |
| L2 | same, `--max-num-pages 8192` (16 384-slot table) | **6 pass / 1 skip**, 8191 + 1 = 8192 |
| L2 | same, `--page-size 4096 --max-num-pages 16` | **6 pass / 1 skip**, 16 + 0 = 16 |

Harness detail strings (cold/warm identical, `MK7Q2-31 recalled`, 10/10 isolation) are the ones
`oipl_repro.md` lists. Page conservation `free + cached (+ pending returns) == total` held at
every quiesce point of every run below; the owner-mode structural self-check (parent links,
children counts, no pins while idle) reported 0 errors after every run; 0 admission holds and
0 fruitless eviction sweeps.

### 4.2 Shadow-mode equivalence (increments 2 and 3)

With both caches on, the GPU table was diffed against the CPU cache after the harness and after
a 32-request load with eviction: 56 blocks each, **0 mismatches** in page ids, token ids and
parent links; counters equal (93 completions, 245 inserted, 5 deduped, 189 evicted = erased).
The GPU's admission-time match agreed with the CPU's import on all 93 requests (136 pages =
136, 0 shorter, 0 longer, 0 page differences).

### 4.3 W1 shared prefix (4 groups × 8 prompts, ~320-word prefix, ~24-word questions)

Geometry `--page-size 64 --max-seq-length 1024 --max-num-pages 128`, 4 rows, 8 tokens/batch.

| rate | metric | cache OFF | CPU cache | **GPU cache** | GPU vs OFF | GPU vs CPU |
|---|---|---|---|---|---|---|
| 1 | TTFT p50 | 103.7 ms | 35.5 ms | **35.2 ms** | −66 % | −1 % |
| 1 | TTFT p95 | 105.9 ms | 103.4 ms | 105.0 ms | (cold leaders) | |
| 1 | throughput | 0.814 rps | 0.816 rps | 0.816 rps | | |
| 3 | TTFT p50 | 111.8 ms | 38.4 ms | **37.6 ms** | −66 % | −2 % |
| 3 | TTFT p95 | 116.5 ms | 117.9 ms | **77.8 ms** | −33 % | −34 % |
| 12 | TTFT p50 | 305.2 ms | 55.1 ms | **38.4 ms** | −87 % | −30 % |
| 12 | TTFT p95 | 541.6 ms | 286.9 ms | **139.5 ms** | −74 % | −51 % |
| 12 | throughput | 9.24 rps | 9.64 rps | 9.65 rps | | |
| 24 | TTFT p50 | 1111.4 ms | 753.4 ms | **167.0 ms** | −85 % | −78 % |
| 24 | TTFT p95 | 1711.9 ms | 960.9 ms | **432.9 ms** | −75 % | −55 % |
| 24 | served throughput | 9.40 rps | 13.98 rps | **18.93 rps** | **+101 %** | **+35 %** |

`oipl_benchmarks.md` reported −65.7 % TTFT p50 at 1 rps and +78 % throughput at 24 rps for the
CPU cache; the CPU-cache column here reproduces the first (−66 %) and lands lower on the second
(+49 %) — two runs in this session agree to 0.4 %, so this is a day-to-day difference of the
box, not noise within the grid. The GPU column is measured against the same OFF baseline in the
same session.

Why the GPU cache pulls ahead as load rises: at 12 rps the CPU cache hit 103 of 256 publishes
and imported 515 pages; the GPU matched at admission and imported 578 (the leader's pages are
already indexed by the time a follower is admitted), and cached pages are pool slack (118 of
128 resident vs the CPU's fixed 112), so no request waited for a page.

### 4.4 W2 random control (32 distinct ~360-word prompts, nothing shared)

| rate | metric | cache OFF | CPU cache | **GPU cache** |
|---|---|---|---|---|
| 1 | TTFT p50 | 107.5 ms | 115.0 ms | 115.7 ms |
| 3 | TTFT p50 | 112.8 ms | 115.9 ms | 117.0 ms |
| 12 | throughput | 9.18 rps | 8.07 rps (−12 %) | **9.09 rps (−1 %)** |
| 12 | TTFT p50 | 325.7 ms | 608.0 ms | **338.8 ms** |
| 24 | throughput | 9.33 rps | 8.12 rps (−13 %) | **9.16 rps (−2 %)** |
| 24 | TTFT p50 | 1133.3 ms | 1432.2 ms | **1160.3 ms** |

At low rate both caches cost the control ~7 % TTFT (the chat-template header is a one-page hit
that changes `initial_step`; `oipl_benchmarks.md` saw ±2 % on a different day). Under load the
CPU cache's staging/eviction work costs 12–13 % throughput on traffic that cannot benefit; the
GPU cache costs 1–2 %.

### 4.5 Long prompts (geometry 2048 / 256 pages, ~1200-word shared prefix ≈ 25 pages)

The case the serial-lane cost argument was about.

| rate | workload | metric | CPU cache | **GPU cache** |
|---|---|---|---|---|
| 1 | shared_prefix | TTFT p50 | 48.2 ms | 47.6 ms |
| 1 | random | TTFT p50 | 413.4 ms | 410.5 ms |
| 24 | shared_prefix | served throughput | 4.90 rps | **8.93 rps (+82 %)** |
| 24 | shared_prefix | TTFT p50 | 3926 ms | **2077 ms** |
| 24 | random | served throughput | 2.41 rps | 2.47 rps |

GPU-side cost per request at this prompt length (31 helper lanes): match 30 µs, completion
insert 86 µs, eviction 5 µs per block; 1058 pages imported over 128 requests against the
CPU's 904.

### 4.6 Cost on the scheduler lane, serial vs helper lanes

128-page geometry, prompts of ~6 pages, `%globaltimer` around each operation:

| operation | serial lane 0 | 31 helper lanes | speed-up |
|---|---|---|---|
| match (Step 4, per admission) | 37.6 µs | 12.1 µs | 3.1× |
| completion insert (Step 1) | 87.7 µs | 25.7 µs | 3.4× |
| eviction (per block) | 27.9 µs | 4.5 µs | 6.2× |

The remaining cost is the dependent chain of one probe per block (a global-memory round trip
each); it grows linearly with pages per request (§4.5: 30 / 86 µs at ~25 pages) and is paid on
the barrier between iterations. The 64-id hash and compare, the serial version's inner loop, are
one reduction and one ballot each.

## 5. What is not done

- Multi-GPU (tensor parallel): the index is per kernel; a TP deployment needs the same table on
  every rank or a rank-0 owner broadcasting decisions. Not addressed.
- `--enable-prefix-cache --gpu-prefix-cache` (shadow mode) keeps both caches; it is a debugging
  configuration, not a deployment one.
- The CLOCK sweep is over table slots, so at low occupancy (a 64-page pool in a 1024-slot table)
  it visits many empty slots per eviction. At the intended occupancy (≈50 %) and 31-wide it is
  cheap; a dense resident list would make it cheap at any occupancy.
- Match and insert are still a dependent chain of one probe per block. For 2K-token prompts
  (32 pages) that is ~5× the per-request cost above on the barrier. Prefetching the next
  block's bucket, or hashing all blocks of a prompt up front and probing them in one pass,
  would cut it; not done.
- Only a single scheduler warp's helpers ever do work (the one whose lane 0 reaches the
  end-of-graph event); the helpers of the other scheduler warps idle at `__nanosleep`.
- A persistent kernel sharing its GPU with another tenant's job can stall (all blocks must be
  co-resident). One L2 run hung and one OOM'd while a 152 GB job sat on the same GPU; both
  passed on a free GPU. Not a cache issue, but the reason every number above names one GPU.
- The protocol simulator (`oipl_protocol_sim.py`) models the CPU-owned lifecycle; owner mode has
  no equivalent model yet, only the structural self-check at `/debug/prefix_cache`.

---

## 6. How to run

```bash
# owner mode (the configuration §4 measured)
python -m mirage.engine.launch_server --model Qwen/Qwen3-0.6B \
  --max-num-batched-requests 4 --max-num-batched-tokens 8 \
  --page-size 64 --max-seq-length 1024 --max-num-pages 128 \
  --gpu-prefix-cache --port 18700
curl -s "http://127.0.0.1:18700/debug/prefix_cache?settle=3" | python -m json.tool

# shadow mode (diff against the CPU cache)
python -m mirage.engine.launch_server ... --enable-prefix-cache --gpu-prefix-cache

# unit test (serial + warp)
cd tests/runtime_python/blackwell/gpu_prefix_cache && python test_gpu_prefix_cache.py

# serial A/B on the same build
MPK_GPU_PREFIX_SERIAL=1 python -m mirage.engine.launch_server ... --gpu-prefix-cache
```
