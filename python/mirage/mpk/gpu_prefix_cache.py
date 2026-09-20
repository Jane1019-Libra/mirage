"""Host-side view of the GPU-resident prefix index (``gpu_prefix_cache.cuh``).

The kernel keeps its index in two Python-owned device tensors -- the entry
table and a small int32 control block -- so the CPU can size them, switch the
feature on, and read them back.  This module holds the layout that both sides
agree on and the decode/compare helpers the shadow-mode check is built from.
Nothing here touches the GPU; the runtime does the copies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

# Control-block layout (mirrors the PC_CTL_* constants in the header).
PC_CTL_SIZE = 32
PC_CTL_ENABLED = 0  # 0 inert, 1 shadow (mirror the CPU cache), 2 owner
PC_CTL_GEN_TAIL = 1
PC_CTL_TICK = 2
PC_MODE_OFF = 0
PC_MODE_SHADOW = 1
PC_MODE_OWNER = 2
PC_STAT_NAMES = {
    8: "completions",
    9: "inserted",
    10: "deduped",
    11: "chain_dropped",
    12: "full",
    13: "erased",
    14: "erase_miss",
    15: "match_calls",
    16: "match_pages",
    17: "match_cpu_pages",
    18: "match_shorter",
    19: "match_longer",
    20: "match_page_diff",
    21: "evicted",
    26: "freed",
    27: "resident",
    28: "admit_hold",
    29: "evict_sweeps",
}
# 64-bit nanosecond accumulators, two consecutive ints each.
PC_STAT_NS = {22: "match_ns", 24: "insert_ns", 30: "evict_ns"}

PC_EMPTY = -1
PC_TOMBSTONE = -2


def entry_dtype(page_size: int) -> np.dtype:
    """Structured dtype of one ``PrefixEntry`` (the header static_asserts it)."""
    return np.dtype([
        ("hash", "<u8"),
        ("page_id", "<i4"),
        ("parent_slot", "<i4"),
        ("refcount", "<i4"),
        ("children", "<i4"),
        ("lru", "<u4"),
        ("ref", "<i4"),
        ("ids", "<i8", (page_size,)),
    ])


def table_bytes(table_size: int, page_size: int) -> int:
    return table_size * entry_dtype(page_size).itemsize


@dataclass(frozen=True)
class GpuBlock:
    """One live entry, keyed the way the CPU cache keys its blocks."""

    slot: int
    page_id: int
    token_ids: Tuple[int, ...]
    parent_page_id: Optional[int]  # None for a root
    parent_slot: int = -1
    refcount: int = 0
    children: int = 0


def decode_table(raw: np.ndarray, page_size: int) -> Dict[int, GpuBlock]:
    """Decode a byte snapshot of the table into ``page_id -> GpuBlock``.

    The parent link is stored as a slot; it is resolved to the parent's page id
    here so the result compares directly against ``PrefixCache._by_page``.
    """
    entries = raw.view(entry_dtype(page_size))
    page_ids = entries["page_id"]
    live = np.nonzero(page_ids >= 0)[0]
    out: Dict[int, GpuBlock] = {}
    for slot in live.tolist():
        e = entries[slot]
        parent_slot = int(e["parent_slot"])
        parent_page: Optional[int] = None
        if parent_slot >= 0:
            parent_page = int(page_ids[parent_slot])
        out[int(e["page_id"])] = GpuBlock(
            slot=slot,
            page_id=int(e["page_id"]),
            token_ids=tuple(int(x) for x in e["ids"]),
            parent_page_id=parent_page,
            parent_slot=parent_slot,
            refcount=int(e["refcount"]),
            children=int(e["children"]),
        )
    return out


def check_structure(gpu: Dict[int, GpuBlock], idle: bool) -> Dict[str, object]:
    """Owner-mode self-consistency of a table snapshot.

    Every parent link must point at a live block, every ``children`` count
    must equal the number of live blocks naming that slot as parent, and when
    the engine is idle no block may still be pinned.  Returns counts plus the
    first few offenders of each kind.
    """
    by_slot = {b.slot: b for b in gpu.values()}
    dangling = [b.page_id for b in gpu.values()
                if b.parent_slot >= 0 and b.parent_slot not in by_slot]
    counted: Dict[int, int] = {}
    for b in gpu.values():
        if b.parent_slot >= 0:
            counted[b.parent_slot] = counted.get(b.parent_slot, 0) + 1
    bad_children = [(b.page_id, b.children, counted.get(b.slot, 0))
                    for b in gpu.values()
                    if b.children != counted.get(b.slot, 0)]
    pinned = [b.page_id for b in gpu.values() if b.refcount > 0]
    negative = [b.page_id for b in gpu.values() if b.refcount < 0]
    leaves = sum(1 for b in gpu.values() if b.children == 0)
    errors = len(dangling) + len(bad_children) + len(negative)
    if idle:
        errors += len(pinned)
    return {
        "resident": len(gpu),
        "leaves": leaves,
        "pinned": len(pinned),
        "structure_errors": errors,
        "examples": {
            "dangling_parent": dangling[:8],
            "children_count": bad_children[:8],
            "negative_refcount": negative[:8],
            "pinned_while_idle": pinned[:8] if idle else [],
        },
    }


def decode_stats(ctl: np.ndarray) -> Dict[str, float]:
    stats: Dict[str, float] = {
        name: int(ctl[idx]) for idx, name in PC_STAT_NAMES.items()}
    ctl64 = np.ascontiguousarray(ctl).view("<i8")
    for idx, name in PC_STAT_NS.items():
        stats[name] = int(ctl64[idx // 2])
    stats["enabled"] = int(ctl[PC_CTL_ENABLED])
    stats["gen_tail_len"] = int(ctl[PC_CTL_GEN_TAIL])
    stats["tick"] = int(ctl[PC_CTL_TICK])
    # Per-call cost on the scheduler lane, in microseconds.
    if stats["match_calls"]:
        stats["match_us_per_call"] = round(
            stats["match_ns"] / stats["match_calls"] / 1e3, 2)
    if stats["completions"]:
        stats["insert_us_per_completion"] = round(
            stats["insert_ns"] / stats["completions"] / 1e3, 2)
    if stats["evicted"]:
        stats["evict_us_per_block"] = round(
            stats["evict_ns"] / stats["evicted"] / 1e3, 2)
    return stats


def compare_with_cpu(gpu: Dict[int, GpuBlock], cpu_blocks) -> Dict[str, object]:
    """Diff the GPU index against the CPU cache's live blocks.

    *cpu_blocks* is ``PrefixCache._by_page`` (page id -> Block).  The two agree
    when they hold the same page ids, with the same token ids under the same
    parent page.  Counts plus a few examples of each kind of disagreement.
    """
    only_gpu = sorted(set(gpu) - set(cpu_blocks))
    only_cpu = sorted(set(cpu_blocks) - set(gpu))
    ids_mismatch: List[int] = []
    parent_mismatch: List[Tuple[int, Optional[int], Optional[int]]] = []
    for page in sorted(set(gpu) & set(cpu_blocks)):
        g = gpu[page]
        c = cpu_blocks[page]
        if g.token_ids != tuple(c.token_ids):
            ids_mismatch.append(page)
        c_parent = None if c.parent is None else c.parent.page_id
        if g.parent_page_id != c_parent:
            parent_mismatch.append((page, g.parent_page_id, c_parent))
    mismatches = (len(only_gpu) + len(only_cpu)
                  + len(ids_mismatch) + len(parent_mismatch))
    return {
        "gpu_blocks": len(gpu),
        "cpu_blocks": len(cpu_blocks),
        "mismatches": mismatches,
        "only_gpu": len(only_gpu),
        "only_cpu": len(only_cpu),
        "ids_mismatch": len(ids_mismatch),
        "parent_mismatch": len(parent_mismatch),
        "examples": {
            "only_gpu": only_gpu[:8],
            "only_cpu": only_cpu[:8],
            "ids_mismatch": ids_mismatch[:8],
            "parent_mismatch": parent_mismatch[:8],
        },
    }
