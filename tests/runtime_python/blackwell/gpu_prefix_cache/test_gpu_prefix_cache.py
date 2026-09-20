"""Unit test for the GPU-resident prefix-cache index (gpu_prefix_cache.cuh).

Builds the header with 4-token pages and a 64-slot table so every case reads
by eye, drives the device functions from single-thread kernels (the scheduler
lane is single-thread too), and checks the CPU PrefixCache's semantics hold:
full-hit clamp, per-id verification, parent-link verification, dedup,
chain continuation from an imported prefix, broken-chain guard, table-full,
and (increment 2) erase-by-page with tombstones that keep probe runs intact,
and (increment 4) leaf-first CLOCK eviction that never takes a pinned block.

Run:  python test_gpu_prefix_cache.py      (or pytest)
"""

import os
import sys

import torch
from torch.utils.cpp_extension import load_inline

PS = 4
TABLE = 64
HERE = os.path.dirname(os.path.abspath(__file__))
INCLUDE = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", "include"))

CUDA_SRC = r"""
#include "mirage/persistent_kernel/gpu_prefix_cache.cuh"
using namespace mirage::runtime;

__global__ void k_reset(PrefixEntry *t) {
  for (int i = threadIdx.x; i < PC_TABLE_SIZE; i += blockDim.x) t[i].page_id = PC_EMPTY;
}
__global__ void k_insert(PrefixEntry *t, long long const *seq, int seq_len,
                         int const *pages, int num_prefix, unsigned lru,
                         int *page_slot, int *out) {
  if (threadIdx.x == 0) {
    PcInsertResult r = pc_insert_sequence(t, seq, seq_len, pages, num_prefix, lru, page_slot);
    out[0] = r.inserted; out[1] = r.deduped; out[2] = r.chain_dropped; out[3] = r.full;
  }
}
__global__ void k_erase(PrefixEntry *t, int *page_slot, int page, int *out) {
  if (threadIdx.x == 0) *out = pc_erase(t, page_slot, page);
}
__global__ void k_evict(PrefixEntry *t, int *page_slot, int *ctl, int *out) {
  if (threadIdx.x == 0) *out = pc_evict_one(t, page_slot, ctl);
}
__global__ void k_pin(PrefixEntry *t, int const *slots, int n, int delta) {
  if (threadIdx.x == 0) {
    if (delta > 0) pc_pin_slots(t, slots, n, 1u); else pc_unpin_slots(t, slots, n);
  }
}
__global__ void k_match(PrefixEntry const *t, long long const *prompt, int len,
                        int max_pages, int *pages, int *slots, int *out) {
  if (threadIdx.x == 0) *out = pc_match(t, prompt, len, max_pages, pages, slots);
}

torch::Tensor make_table() {
  auto t = torch::empty({(long)(PC_TABLE_SIZE * sizeof(PrefixEntry))},
                        torch::dtype(torch::kUInt8).device(torch::kCUDA));
  k_reset<<<1, 128>>>(reinterpret_cast<PrefixEntry *>(t.data_ptr()));
  return t;
}
torch::Tensor insert_seq(torch::Tensor t, torch::Tensor seq, torch::Tensor pages, int num_prefix,
                         int lru, torch::Tensor page_slot) {
  auto out = torch::zeros({4}, torch::dtype(torch::kInt32).device(torch::kCUDA));
  k_insert<<<1, 32>>>(reinterpret_cast<PrefixEntry *>(t.data_ptr()),
                      reinterpret_cast<long long const *>(seq.data_ptr<int64_t>()), (int)seq.numel(),
                      pages.data_ptr<int>(), num_prefix, (unsigned)lru,
                      page_slot.data_ptr<int>(), out.data_ptr<int>());
  return out.cpu();
}
int evict_one(torch::Tensor t, torch::Tensor page_slot, torch::Tensor ctl) {
  auto out = torch::zeros({1}, torch::dtype(torch::kInt32).device(torch::kCUDA));
  k_evict<<<1, 32>>>(reinterpret_cast<PrefixEntry *>(t.data_ptr()), page_slot.data_ptr<int>(),
                     ctl.data_ptr<int>(), out.data_ptr<int>());
  return out.item<int>();
}
void pin(torch::Tensor t, torch::Tensor slots, int n, int delta) {
  k_pin<<<1, 32>>>(reinterpret_cast<PrefixEntry *>(t.data_ptr()), slots.data_ptr<int>(), n, delta);
}
int erase_page(torch::Tensor t, torch::Tensor page_slot, int page) {
  auto out = torch::zeros({1}, torch::dtype(torch::kInt32).device(torch::kCUDA));
  k_erase<<<1, 32>>>(reinterpret_cast<PrefixEntry *>(t.data_ptr()), page_slot.data_ptr<int>(),
                     page, out.data_ptr<int>());
  return out.item<int>();
}
int match(torch::Tensor t, torch::Tensor prompt, int max_pages,
          torch::Tensor pages, torch::Tensor slots) {
  auto out = torch::zeros({1}, torch::dtype(torch::kInt32).device(torch::kCUDA));
  k_match<<<1, 32>>>(reinterpret_cast<PrefixEntry const *>(t.data_ptr()),
                     reinterpret_cast<long long const *>(prompt.data_ptr<int64_t>()), (int)prompt.numel(), max_pages,
                     pages.data_ptr<int>(), slots.data_ptr<int>(), out.data_ptr<int>());
  return out.item<int>();
}
"""

CPP_SRC = r"""
torch::Tensor make_table();
torch::Tensor insert_seq(torch::Tensor, torch::Tensor, torch::Tensor, int, int, torch::Tensor);
int erase_page(torch::Tensor, torch::Tensor, int);
int evict_one(torch::Tensor, torch::Tensor, torch::Tensor);
void pin(torch::Tensor, torch::Tensor, int, int);
int match(torch::Tensor, torch::Tensor, int, torch::Tensor, torch::Tensor);
"""

ext = load_inline(
    name="gpu_prefix_cache_test",
    cpp_sources=CPP_SRC,
    cuda_sources=CUDA_SRC,
    functions=["make_table", "insert_seq", "erase_page", "evict_one", "pin", "match"],
    extra_include_paths=[INCLUDE],
    extra_cuda_cflags=[f"-DPC_PAGE_SIZE={PS}", f"-DPC_TABLE_SIZE={TABLE}", "-O2"],
    verbose=False,
)

dev = "cuda"
I64 = lambda xs: torch.tensor(xs, dtype=torch.int64, device=dev)
I32 = lambda xs: torch.tensor(xs, dtype=torch.int32, device=dev)


NUM_PAGES = 4096  # page-id space of the owner map


class Cache:
    def __init__(self):
        self.t = ext.make_table()
        self.page_slot = torch.full((NUM_PAGES,), -1, dtype=torch.int32, device=dev)
        self.ctl = torch.zeros(32, dtype=torch.int32, device=dev)

    def insert(self, seq, pages, num_prefix=0, lru=0):
        """(inserted, kept) where kept[j] says page j now owns a block.

        An imported page (j < num_prefix) owns the block from its own insert,
        a deduped page owns nothing, so kept reads the owner map.
        """
        r = ext.insert_seq(self.t, I64(seq), I32(pages), num_prefix, lru, self.page_slot)
        self.last = dict(zip(("inserted", "deduped", "chain_dropped", "full"), r.tolist()))
        owned = self.page_slot[I32(pages).long()].tolist() if pages else []
        return self.last["inserted"], [int(s >= 0) for s in owned]

    def erase(self, page):
        return ext.erase_page(self.t, self.page_slot, page)

    def evict(self):
        return ext.evict_one(self.t, self.page_slot, self.ctl)

    def clear_stall(self):
        self.ctl[4] = 0  # what Step 1 does after a completion

    def pin(self, prompt, delta):
        """Pin (+1) or unpin (-1) the chain a match of *prompt* finds."""
        slots = torch.full((16,), -1, dtype=torch.int32, device=dev)
        pages = torch.full((16,), -1, dtype=torch.int32, device=dev)
        n = ext.match(self.t, I64(prompt), 16, pages, slots)
        ext.pin(self.t, slots, n, delta)
        return n

    def match(self, prompt, max_pages=16):
        pages = torch.full((16,), -1, dtype=torch.int32, device=dev)
        slots = torch.full((16,), -1, dtype=torch.int32, device=dev)
        n = ext.match(self.t, I64(prompt), max_pages, pages, slots)
        return n, pages.tolist()[:n]


def check(name, got, want):
    ok = got == want
    print(f"  {'✓' if ok else '✗'} {name:44s} got={got}  want={want}")
    return ok


def run():
    results = []
    A = list(range(1, 13))  # 3 pages: [1-4][5-8][9-12]

    print("insert / match basics")
    c = Cache()
    results.append(check("insert A (3 pages)", c.insert(A, [10, 11, 12]), (3, [1, 1, 1])))
    results.append(check("match A+1 token -> 3 pages", c.match(A + [13]), (3, [10, 11, 12])))
    results.append(check("FULL-HIT CLAMP: match A exactly -> 2", c.match(A), (2, [10, 11])))
    results.append(check("id mismatch in block 1 -> 1", c.match([1, 2, 3, 4, 5, 6, 7, 99] + A[8:] + [13]), (1, [10])))
    results.append(check("no match at block 0 -> 0", c.match([50, 51, 52, 53, 54]), (0, [])))
    results.append(check("max_pages cap -> 2", c.match(A + [13], max_pages=2), (2, [10, 11])))

    print("dedup")
    results.append(check("re-insert A with new pages -> 0 kept", c.insert(A, [40, 41, 42]), (0, [0, 0, 0])))

    print("PARENT-LINK VERIFICATION (same ids, different context)")
    # B shares block 0 with A (dedups to page 10), then [9,9,9,9] under it -> page 21
    results.append(check("insert B: block0 dedups, block1 new", c.insert([1, 2, 3, 4, 9, 9, 9, 9], [20, 21]), (1, [0, 1])))
    # C: different block 0, then the SAME [9,9,9,9] under a different parent -> must be a NEW entry
    results.append(check("insert C: both new", c.insert([7, 7, 7, 7, 9, 9, 9, 9], [30, 31]), (2, [1, 1])))
    results.append(check("match C-prompt -> C's pages, not B's", c.match([7, 7, 7, 7, 9, 9, 9, 9, 0]), (2, [30, 31])))
    results.append(check("match B-prompt -> A.p0 + B.p1", c.match([1, 2, 3, 4, 9, 9, 9, 9, 0]), (2, [10, 21])))

    print("chain continuation from an imported prefix (num_prefix)")
    D = A + [13, 14, 15, 16]
    results.append(check("insert D, num_prefix=3 -> 1 new", c.insert(D, [10, 11, 12, 50], num_prefix=3), (1, [1, 1, 1, 1])))
    results.append(check("match D+1 -> 4 pages", c.match(D + [17]), (4, [10, 11, 12, 50])))

    print("broken-chain guard")
    results.append(check("imported block missing -> insert nothing", c.insert([99, 99, 99, 99, 5, 6, 7, 8], [60, 61], num_prefix=1), (0, [0, 0])))
    results.append(check("  ...and reports chain_dropped", c.last["chain_dropped"], 1))

    print("erase (tombstones)")
    # Erase D's leaf page 50: match must stop at 3, the page owns nothing.
    results.append(check("erase leaf page 50 -> 1", c.erase(50), 1))
    results.append(check("erase again -> 0 (no owner)", c.erase(50), 0))
    results.append(check("match D+1 after erase -> 3 pages", c.match(D + [17]), (3, [10, 11, 12])))
    results.append(check("re-insert D tail -> 1 new, slot reused", c.insert(D, [10, 11, 12, 55], num_prefix=3), (1, [1, 1, 1, 1])))
    results.append(check("match D+1 -> new page 55", c.match(D + [17]), (4, [10, 11, 12, 55])))
    # Probe-run integrity: fill a small table so runs collide, erase entries in
    # the middle of runs, and confirm every survivor is still reachable.
    c3 = Cache()
    seqs = {k: [2000 + k] * PS for k in range(TABLE - 4)}
    for k, s in seqs.items():
        c3.insert(s, [k])
    for k in range(0, TABLE - 4, 3):
        c3.erase(k)
    survivors_ok = all(c3.match(seqs[k] + [0]) == (1, [k]) for k in range(TABLE - 4) if k % 3 != 0)
    erased_ok = all(c3.match(seqs[k] + [0]) == (0, []) for k in range(0, TABLE - 4, 3))
    results.append(check("survivors reachable through tombstones", survivors_ok, True))
    results.append(check("erased blocks miss", erased_ok, True))
    # Tombstones are reusable capacity: the erased third fits again.
    refill = sum(c3.insert([3000 + k] * PS, [k])[0] for k in range(0, TABLE - 4, 3))
    results.append(check("erased slots accept new blocks", refill, len(range(0, TABLE - 4, 3))))

    print("eviction (leaf-first CLOCK, pins respected)")
    e = Cache()
    e.insert(A, [10, 11, 12])                          # chain 10 -> 11 -> 12
    e.insert([1, 2, 3, 4, 9, 9, 9, 9], [20, 21])       # 10 -> 21 (20 deduped)
    first_two = {e.evict(), e.evict()}
    results.append(check("leaves go first: {12, 21}", first_two, {12, 21}))
    results.append(check("then 11 (now a leaf)", e.evict(), 11))
    results.append(check("then the root 10", e.evict(), 10))
    results.append(check("empty -> -1 and stall flag set", (e.evict(), int(e.ctl[4].item())), (-1, 1)))
    results.append(check("nothing left to match", e.match(A + [13]), (0, [])))
    e2 = Cache()
    e2.insert(A, [10, 11, 12])
    results.append(check("pin the chain (3 blocks)", e2.pin(A + [13], +1), 3))
    results.append(check("pinned leaf: nothing evictable", e2.evict(), -1))
    results.append(check("stall flag blocks further sweeps", e2.evict(), -1))
    e2.pin(A + [13], -1)
    results.append(check("unpinned but still stalled -> -1", e2.evict(), -1))
    e2.clear_stall()
    results.append(check("after Step-1 clear: leaf 12 evicted", e2.evict(), 12))
    results.append(check("match now stops at 2 pages", e2.match(A + [13]), (2, [10, 11])))
    results.append(check("re-insert tail reuses chain", e2.insert(A, [10, 11, 70], num_prefix=2), (1, [1, 1, 1])))

    print("table full")
    c2 = Cache()
    total = 0
    for k in range(TABLE + 8):
        n, _ = c2.insert([1000 + k] * PS, [k])
        total += n
    results.append(check(f"inserted exactly TABLE={TABLE} then refused", total, TABLE))
    n_ok, _ = c2.match([1000] * PS + [0])
    results.append(check("first block still matchable when full", n_ok, 1))

    passed = sum(results)
    print(f"\n{passed}/{len(results)} passed")
    return passed == len(results)


if __name__ == "__main__":
    sys.exit(0 if run() else 1)


def test_gpu_prefix_cache():
    assert run()
