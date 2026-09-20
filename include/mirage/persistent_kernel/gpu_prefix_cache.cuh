/* Copyright 2025 CMU
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

// GPU-resident prefix-cache index. Experimental.
//
// Increment 1: the data structure, driven by a single thread.
// Increment 2: deletion (tombstones) and a page -> slot map, so the scheduler
//   lane can mirror the CPU PrefixCache: insert on completion, erase when the
//   CPU returns a cached page. The kernel does not act on the table yet.
// Increment 3: a shadow match at admission, compared against the prefix the
//   CPU imported for the same request, and nanosecond accounting of both the
//   match and the insert so their cost on the scheduler lane is measurable.
// Increment 4: the kernel owns the cache (ctl mode 2). Step 4 matches, pins
//   and imports; Step 1 unpins, indexes and frees the rest of the page table
//   itself; admission evicts on demand, leaf-first, by a CLOCK sweep over the
//   table. The CPU cache and the page-return channel are idle in this mode.
//
// Semantics are the CPU PrefixCache's (§6.4 v2.1-b), not an approximation of
// them: a block is keyed by the chained hash h_j = H(h_{j-1}, ids_j), and a
// hit is confirmed by comparing every stored token id AND the parent slot. A
// bucket can hold a block with identical ids under a different preceding
// context; importing that would be a silently wrong hit, so it must miss.
//
// Open addressing with linear probing. Erase leaves a tombstone so the probe
// runs through it stay intact; insert reuses the first tombstone of its run.
// Eviction is leaf-first (children == 0) and never touches a pinned block
// (refcount > 0): an inner block's KV is only reachable through its
// descendants, and a pinned block is being read by a running request.
//
// PC_PAGE_SIZE / PC_TABLE_SIZE are macros so a unit test can build a tiny
// table with 4-token pages and read the results by eye.

#ifndef PC_PAGE_SIZE
#define PC_PAGE_SIZE MPK_PAGE_SIZE
#endif
#ifndef PC_TABLE_SIZE
#define PC_TABLE_SIZE 4096
#endif

namespace mirage {
namespace runtime {

constexpr int PC_TABLE_MASK = PC_TABLE_SIZE - 1;
constexpr int PC_EMPTY = -1;     // slot never used: ends a probe run
constexpr int PC_TOMBSTONE = -2; // slot erased: probing continues through it
static_assert((PC_TABLE_SIZE & (PC_TABLE_SIZE - 1)) == 0,
              "PC_TABLE_SIZE must be a power of two");

struct PrefixEntry {
  unsigned long long hash; // chained hash of this block
  int page_id;             // PC_EMPTY / PC_TOMBSTONE: slot unused
  int parent_slot;         // slot of the preceding block, -1 for a root
  int refcount;            // running requests reading this block (pin)
  int children;            // live blocks whose parent this is
  unsigned lru;            // tick of the last insert or hit (observability)
  int ref;                 // CLOCK reference bit: set on insert/hit
  long long ids[PC_PAGE_SIZE]; // the tokens, for verification on hit
};
// The Python side decodes the table with a structured dtype of this layout.
static_assert(sizeof(PrefixEntry) == 32 + 8 * PC_PAGE_SIZE,
              "PrefixEntry layout drifted from the Python decoder");

// Control block shared with Python: parameters below PC_CTL_STATS, counters
// from it. Both live in one int32 device array so the meta-tensor plumbing
// grows by two entries (table + ctl), not a dozen.
constexpr int PC_CTL_SIZE = 32;
constexpr int PC_CTL_ENABLED = 0;  // Python-written: 0 inert, 1 shadow, 2 owner
constexpr int PC_CTL_GEN_TAIL = 1; // Python-written: prompt tail never cached
constexpr int PC_CTL_TICK = 2;     // GPU-written: iterations since launch (lru)
constexpr int PC_CTL_CLOCK = 3;    // GPU-written: CLOCK sweep cursor
constexpr int PC_CTL_EVICT_STALL = 4; // GPU-written: a full sweep found nothing
constexpr int PC_MODE_SHADOW = 1;
constexpr int PC_MODE_OWNER = 2;
constexpr int PC_CTL_STATS = 8;
constexpr int PC_STAT_COMPLETIONS = 8;   // completions the table saw
constexpr int PC_STAT_INSERTED = 9;      // blocks added
constexpr int PC_STAT_DEDUPED = 10;      // fresh pages that duplicated a block
constexpr int PC_STAT_CHAIN_DROPPED = 11; // imported parent gone; tail dropped
constexpr int PC_STAT_FULL = 12;         // inserts refused by a full table
constexpr int PC_STAT_ERASED = 13;       // blocks removed on page return
constexpr int PC_STAT_ERASE_MISS = 14;   // returned pages that owned no block
// Shadow match (increment 3): the GPU matches the prompt at admission and
// compares with the prefix the CPU imported for the same request. The CPU
// matched earlier (at publish) and under a pin budget, so the GPU may find
// more; finding less, or different pages, would be a bug.
constexpr int PC_STAT_MATCH_CALLS = 15;
constexpr int PC_STAT_MATCH_PAGES = 16;     // pages the GPU match found
constexpr int PC_STAT_MATCH_CPU_PAGES = 17; // pages the CPU imported
constexpr int PC_STAT_MATCH_SHORTER = 18;   // requests where GPU < CPU
constexpr int PC_STAT_MATCH_LONGER = 19;    // requests where GPU > CPU
constexpr int PC_STAT_MATCH_PAGE_DIFF = 20; // common prefix names other pages
constexpr int PC_STAT_EVICTED = 21;         // owner mode: blocks evicted
// Nanoseconds on the scheduler lane, 64-bit across two consecutive ints.
constexpr int PC_STAT_MATCH_NS = 22;  // [22..23]
constexpr int PC_STAT_INSERT_NS = 24; // [24..25]
constexpr int PC_STAT_FREED = 26;       // owner mode: pages freed at completion
constexpr int PC_STAT_RESIDENT = 27;    // live blocks (== pages the cache holds)
constexpr int PC_STAT_ADMIT_HOLD = 28;  // owner mode: gate failed after evicting
constexpr int PC_STAT_EVICT_SWEEPS = 29; // owner mode: fruitless full sweeps
constexpr int PC_STAT_EVICT_NS = 30;    // [30..31]

__device__ __forceinline__ unsigned long long pc_now_ns() {
  unsigned long long t;
  asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t));
  return t;
}

__device__ __forceinline__ void pc_add_ns(int *ctl, int idx,
                                          unsigned long long ns) {
  unsigned long long *acc =
      reinterpret_cast<unsigned long long *>(ctl + idx);
  *acc += ns;
}

// 64-bit multiply-xorshift mix. Not cryptographic and does not need to be:
// every hit is verified against the stored ids, so a collision costs a probe,
// never a wrong page. Cheap enough for one scheduler lane to run per block.
__device__ __forceinline__ unsigned long long
    pc_hash(unsigned long long parent, long long const *ids) {
  unsigned long long h = parent ^ 0x9E3779B97F4A7C15ull;
#pragma unroll 8
  for (int i = 0; i < PC_PAGE_SIZE; i++) {
    h ^= static_cast<unsigned long long>(ids[i]);
    h *= 0xff51afd7ed558ccdull;
    h ^= h >> 33;
  }
  return h;
}

__device__ __forceinline__ bool pc_ids_equal(long long const *a,
                                             long long const *b) {
  for (int i = 0; i < PC_PAGE_SIZE; i++) {
    if (a[i] != b[i]) {
      return false;
    }
  }
  return true;
}

// The slot holding *ids* directly after *parent_slot*, or -1.
__device__ __forceinline__ int pc_find(PrefixEntry const *t,
                                       unsigned long long h,
                                       int parent_slot,
                                       long long const *ids) {
  int s = static_cast<int>(h & PC_TABLE_MASK);
  for (int probe = 0; probe < PC_TABLE_SIZE; probe++) {
    PrefixEntry const &e = t[s];
    if (e.page_id == PC_EMPTY) {
      return -1; // linear probing: an empty slot ends the run
    }
    if (e.page_id != PC_TOMBSTONE && e.hash == h &&
        e.parent_slot == parent_slot && pc_ids_equal(e.ids, ids)) {
      return s;
    }
    s = (s + 1) & PC_TABLE_MASK;
  }
  return -1;
}

// Insert into the first tombstone of the probe run, else its empty end.
// Returns the slot, or -1 when the table is full. Callers pc_find first, so
// an identical block cached by an earlier request is reused, not duplicated.
__device__ __forceinline__ int pc_insert(PrefixEntry *t,
                                         unsigned long long h,
                                         int parent_slot,
                                         long long const *ids,
                                         int page_id,
                                         unsigned lru) {
  int s = static_cast<int>(h & PC_TABLE_MASK);
  int target = -1;
  for (int probe = 0; probe < PC_TABLE_SIZE; probe++) {
    int pid = t[s].page_id;
    if (pid == PC_TOMBSTONE && target < 0) {
      target = s;
    }
    if (pid == PC_EMPTY) {
      if (target < 0) {
        target = s;
      }
      break;
    }
    s = (s + 1) & PC_TABLE_MASK;
  }
  if (target < 0) {
    return -1;
  }
  PrefixEntry &e = t[target];
  e.hash = h;
  e.parent_slot = parent_slot;
  e.page_id = page_id;
  e.refcount = 0;
  e.children = 0;
  e.lru = lru;
  e.ref = 1;
  for (int i = 0; i < PC_PAGE_SIZE; i++) {
    e.ids[i] = ids[i];
  }
  if (parent_slot >= 0) {
    t[parent_slot].children++;
  }
  return target;
}

// Remove the block that *page* owns, if any. page_slot is the owner map the
// insert path maintains: page id -> table slot, -1 for a page that owns no
// block. Returns 1 if a block was removed.
__device__ __forceinline__ int pc_erase(PrefixEntry *t, int *page_slot,
                                        int page) {
  int s = page_slot[page];
  if (s < 0) {
    return 0;
  }
  int parent = t[s].parent_slot;
  if (parent >= 0 && t[parent].page_id >= 0) {
    t[parent].children--;
  }
  t[s].page_id = PC_TOMBSTONE;
  page_slot[page] = -1;
  return 1;
}

// Owner mode. Evict one block and return its page, or -1 when nothing can
// go. CLOCK over the table: a slot is skipped if unused, pinned or not a
// leaf; a leaf with its reference bit set gets a second chance; the first
// leaf without one is evicted. Once a full sweep finds nothing, every leaf is
// pinned -- the pool is committed to running requests -- and sweeping again
// before a completion would only burn the scheduler lane, so a stall flag
// short-circuits until Step 1 clears it.
__device__ __forceinline__ int pc_evict_one(PrefixEntry *t, int *page_slot,
                                            int *ctl) {
  if (ctl[PC_CTL_EVICT_STALL]) {
    return -1;
  }
  int cursor = ctl[PC_CTL_CLOCK];
  for (int visited = 0; visited < 2 * PC_TABLE_SIZE; visited++) {
    PrefixEntry &e = t[cursor];
    cursor = (cursor + 1) & PC_TABLE_MASK;
    if (e.page_id < 0 || e.refcount > 0 || e.children > 0) {
      continue;
    }
    if (e.ref) {
      e.ref = 0;
      continue;
    }
    int page = e.page_id;
    pc_erase(t, page_slot, page);
    ctl[PC_CTL_CLOCK] = cursor;
    return page;
  }
  ctl[PC_CTL_CLOCK] = cursor;
  ctl[PC_CTL_EVICT_STALL] = 1;
  ctl[PC_STAT_EVICT_SWEEPS]++;
  return -1;
}

// Owner mode: pin the matched chain (Step 4) or release it (Step 1 / a
// refused admission). A pinned block is touched for CLOCK too.
__device__ __forceinline__ void pc_pin_slots(PrefixEntry *t, int const *slots,
                                             int n, unsigned tick) {
  for (int j = 0; j < n; j++) {
    PrefixEntry &e = t[slots[j]];
    e.refcount++;
    e.ref = 1;
    e.lru = tick;
  }
}

__device__ __forceinline__ void pc_unpin_slots(PrefixEntry *t,
                                               int const *slots, int n) {
  for (int j = 0; j < n; j++) {
    t[slots[j]].refcount--;
  }
}

// Longest verified cached prefix of prompt[0, prompt_len). Writes the page ids
// and table slots of the matched chain, returns its length. Capped at
// (prompt_len - 1) / PS so at least one prompt token is left for the kernel to
// forward -- a full hit would leave nothing to compute -- and at max_pages.
__device__ __forceinline__ int pc_match(PrefixEntry const *t,
                                        long long const *prompt,
                                        int prompt_len,
                                        int max_pages,
                                        int *out_pages,
                                        int *out_slots) {
  int limit = (prompt_len - 1) / PC_PAGE_SIZE;
  if (limit > max_pages) {
    limit = max_pages;
  }
  int parent = -1;
  unsigned long long ph = 0;
  int n = 0;
  for (int j = 0; j < limit; j++) {
    long long const *ids = prompt + j * PC_PAGE_SIZE;
    unsigned long long h = pc_hash(ph, ids);
    int s = pc_find(t, h, parent, ids);
    if (s < 0) {
      break;
    }
    out_pages[n] = t[s].page_id;
    out_slots[n] = s;
    n++;
    parent = s;
    ph = h;
  }
  return n;
}

// Outcome of pc_insert_sequence, for the caller's counters.
struct PcInsertResult {
  int inserted;      // blocks added; their pages now own a slot
  int deduped;       // fresh pages that duplicated an existing block
  int chain_dropped; // 1 if an imported parent was gone and the tail dropped
  int full;          // 1 if the table refused an insert
};

// Index every full page of a completed sequence. The first num_prefix pages
// were imported and are already resident -- the walk just recovers the chain
// position from them -- and the rest are fresh. A fresh block that an earlier
// request already cached is deduped: the request's own page stays unmapped in
// page_slot, so the CPU's return of it is a no-op here. Every inserted page is
// recorded in page_slot, which pc_erase consults later.
//
// If an imported block is no longer in the table the chain cannot be extended
// under a verified parent, and inserting the tail under the wrong one would
// be a silently wrong future hit (§6.4 v2.1-b); the tail is dropped instead.
__device__ __forceinline__ PcInsertResult
    pc_insert_sequence(PrefixEntry *t,
                       long long const *seq,
                       int seq_len,
                       int const *page_ids,
                       int num_prefix,
                       unsigned lru,
                       int *page_slot) {
  PcInsertResult r = {0, 0, 0, 0};
  int full = seq_len > 0 ? seq_len / PC_PAGE_SIZE : 0;
  int parent = -1;
  unsigned long long ph = 0;
  for (int j = 0; j < num_prefix && j < full; j++) {
    long long const *ids = seq + j * PC_PAGE_SIZE;
    unsigned long long h = pc_hash(ph, ids);
    int s = pc_find(t, h, parent, ids);
    if (s < 0) {
      r.chain_dropped = 1;
      return r; // imported block gone: cannot extend under it
    }
    parent = s;
    ph = h;
  }
  for (int j = num_prefix; j < full; j++) {
    long long const *ids = seq + j * PC_PAGE_SIZE;
    unsigned long long h = pc_hash(ph, ids);
    int s = pc_find(t, h, parent, ids);
    if (s < 0) {
      s = pc_insert(t, h, parent, ids, page_ids[j], lru);
      if (s < 0) {
        r.full = 1;
        return r; // table full
      }
      page_slot[page_ids[j]] = s;
      r.inserted++;
    } else {
      r.deduped++;
    }
    parent = s;
    ph = h;
  }
  return r;
}

} // namespace runtime
} // namespace mirage
