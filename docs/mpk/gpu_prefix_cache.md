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

Qwen3-0.6B, B200, one GPU for every run. Harness: `online_harness.py` (6 cases). Load:
`bench_serving.py --workload both --groups 4 --prompts-per-group 8 --prefix-tokens 320
--question-tokens 24 --seed 0`, server `--page-size 64 --max-seq-length 1024 --max-num-pages
128 --max-num-batched-requests 4 --max-num-batched-tokens 8`.

### 4.1 Correctness (shadow mode, 64-page pool, harness then a 32-request load with eviction)

| Check | Result |
|---|---|
| GPU table vs CPU cache after load | 56 blocks each, **0 mismatches** (page ids, token ids, parent links) |
| Counters | completions 93 = 93, inserted 245 = 245, deduped 5 = 5, evicted 189 = erased 189 |
| Shadow match vs CPU import, 93 admissions | 136 pages = 136, 0 shorter, 0 longer, 0 page differences |
| Harness | 6/6 in every mode (cold/warm byte-identical, multi-turn recall) |

### 4.2 Owner mode, structural self-check after load

0 dangling parents, 0 wrong children counts, 0 negative refcounts, 0 pins while idle;
0 admission holds, 0 fruitless sweeps. 136 prefix pages hit on the 64-page run, the same count
the CPU cache reached on the same workload.

### 4.3 Load benchmark

Five server configurations, run back to back on one GPU with the same seed. "CPU cache" was
run twice (`cpu`, `cpu2`) to see the run-to-run spread; "GPU serial" is increment 4 on lane 0
alone, "GPU warp" is increment 6 with the helper lanes.

| rate | workload | metric | cache OFF | CPU cache | CPU cache (rerun) | GPU serial | **GPU warp** |
|---|---|---|---|---|---|---|---|
| 1.0 | shared_prefix | TTFT p50 | 103.4 ms | 34.5 ms | 35.1 ms | 35.0 ms | **34.2 ms** |
| 1.0 | shared_prefix | throughput | 0.814 rps | 0.816 rps | 0.816 rps | 0.816 rps | 0.816 rps |
| 1.0 | random | TTFT p50 | 105.4 ms | 114.7 ms | 115.4 ms | 114.8 ms | 110.9 ms |
| 24 | shared_prefix | TTFT p50 | 1072.7 ms | 767.8 ms | 782.1 ms | 101.3 ms | **68.2 ms** |
| 24 | shared_prefix | served throughput | 9.60 rps | 13.81 rps | 13.76 rps | 18.94 rps | **18.98 rps** |
| 24 | random | TTFT p50 | 1091.3 ms | 1431.1 ms | 1438.9 ms | 1170.8 ms | 1118.8 ms |
| 24 | random | served throughput | 9.53 rps | 8.17 rps | 8.12 rps | 9.16 rps | 9.37 rps |

At low load the three caches are indistinguishable: the same −67 % TTFT on shared prefixes.
At saturation the GPU cache serves **+38 % over the CPU cache** and **+98 % over no cache**,
with TTFT p50 **11× lower than the CPU cache's** (68 vs 780 ms); 288 prefix pages were hit
against the CPU's 245 on the same 128 requests, from the admission-time match (§1.1) and the
whole pool being usable (§1.3). The random control, which shares nothing, is the honest cost
line: the CPU cache takes 15 % off it at saturation, the GPU cache 1.6 %.

The two CPU-cache runs agree to 0.4 %, so the spread within this session is small. The same
configuration measured 18.2 rps in `oipl_polish.md` on an earlier day; treat cross-session
absolute values with care and same-session ordering as reliable.

### 4.4 Cost on the scheduler lane

Measured by `%globaltimer` around each operation, 128-page pool, prompts of ~6 pages
(`/debug/prefix_cache` after the rate-24 run):

| operation | serial lane 0 | 31 helper lanes | speed-up |
|---|---|---|---|
| match (Step 4, per admission) | 37.6 µs | 10.8 µs | 3.5× |
| completion insert (Step 1) | 87.7 µs | 23.2 µs | 3.8× |
| eviction (per block) | 27.9 µs | 4.5 µs | 6.2× |

The 64-page harness run (sparser table) gave 22 / 57 / 44 µs serial and 7.6 / 16 / 5.1 µs
with helpers. The remaining cost is dominated by the dependent chain of block lookups (one
probe per block, each a global-memory round trip); it grows linearly with pages per request
and is paid on the barrier between iterations, so for long prompts it is still worth watching
(§5). The 64-id hash and compare, which were the serial version's inner loop, are now one
reduction and one ballot each.

---

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
