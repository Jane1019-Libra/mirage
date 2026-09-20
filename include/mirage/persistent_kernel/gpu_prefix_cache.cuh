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

// GPU-resident prefix-cache index. Experimental; increment 1 is the data
// structure alone, driven by a single thread, with no eviction.
//
// Semantics are the CPU PrefixCache's (§6.4 v2.1-b), not an approximation of
// them: a block is keyed by the chained hash h_j = H(h_{j-1}, ids_j), and a
// hit is confirmed by comparing every stored token id AND the parent slot. A
// bucket can hold a block with identical ids under a different preceding
// context; importing that would be a silently wrong hit, so it must miss.
//
// Open addressing with linear probing. Deletion arrives with eviction
// (tombstones); until then a full table simply stops accepting inserts.
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
constexpr int PC_EMPTY = -1;
static_assert((PC_TABLE_SIZE & (PC_TABLE_SIZE - 1)) == 0,
              "PC_TABLE_SIZE must be a power of two");

struct PrefixEntry {
  unsigned long long hash; // chained hash of this block
  int page_id;             // PC_EMPTY: slot unused
  int parent_slot;         // slot of the preceding block, -1 for a root
  int refcount;            // in-flight importers (eviction guard, later)
  unsigned lru;            // last-touched tick (eviction order, later)
  long long ids[PC_PAGE_SIZE]; // the tokens, for verification on hit
};

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
    if (e.hash == h && e.parent_slot == parent_slot &&
        pc_ids_equal(e.ids, ids)) {
      return s;
    }
    s = (s + 1) & PC_TABLE_MASK;
  }
  return -1;
}

// Insert into the first empty slot of the probe run. Returns the slot, or -1
// when the table is full. Callers pc_find first, so an identical block cached
// by an earlier request is reused rather than duplicated.
__device__ __forceinline__ int pc_insert(PrefixEntry *t,
                                         unsigned long long h,
                                         int parent_slot,
                                         long long const *ids,
                                         int page_id,
                                         unsigned lru) {
  int s = static_cast<int>(h & PC_TABLE_MASK);
  for (int probe = 0; probe < PC_TABLE_SIZE; probe++) {
    PrefixEntry &e = t[s];
    if (e.page_id == PC_EMPTY) {
      e.hash = h;
      e.parent_slot = parent_slot;
      e.page_id = page_id;
      e.refcount = 0;
      e.lru = lru;
      for (int i = 0; i < PC_PAGE_SIZE; i++) {
        e.ids[i] = ids[i];
      }
      return s;
    }
    s = (s + 1) & PC_TABLE_MASK;
  }
  return -1;
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

// Index every full page of a completed sequence. The first num_prefix pages
// were imported and are already resident -- the walk just recovers the chain
// position from them -- and the rest are fresh. A fresh block that an earlier
// request already cached is deduped: kept[j] = 0 says the request's own page
// j is redundant and should go back to the pool. Returns pages inserted.
//
// If an imported block is no longer in the table the chain cannot be extended
// under a verified parent, and inserting the tail under the wrong one would
// be a silently wrong future hit (§6.4 v2.1-b); the tail is dropped instead.
__device__ __forceinline__ int pc_insert_sequence(PrefixEntry *t,
                                                  long long const *seq,
                                                  int seq_len,
                                                  int const *page_ids,
                                                  int num_prefix,
                                                  unsigned lru,
                                                  int *kept) {
  int full = seq_len / PC_PAGE_SIZE;
  int parent = -1;
  unsigned long long ph = 0;
  int inserted = 0;
  for (int j = 0; j < full; j++) {
    kept[j] = 0;
  }
  for (int j = 0; j < num_prefix && j < full; j++) {
    long long const *ids = seq + j * PC_PAGE_SIZE;
    unsigned long long h = pc_hash(ph, ids);
    int s = pc_find(t, h, parent, ids);
    if (s < 0) {
      return inserted; // imported block gone: cannot extend under it
    }
    kept[j] = 1; // imported pages stay with the block that owns them
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
        return inserted; // table full
      }
      kept[j] = 1;
      inserted++;
    }
    parent = s;
    ph = h;
  }
  return inserted;
}

} // namespace runtime
} // namespace mirage
