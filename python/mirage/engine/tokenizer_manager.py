"""Thread-safe tokenizer/detokenizer wrapper."""

from __future__ import annotations

import collections
import hashlib
import threading
from typing import Any, Dict, List, Optional, Sequence, Tuple

DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."

# Conversations whose canonical id sequence is kept for continuation.  Each
# entry is two small id lists, so the budget is about memory-free; it only
# bounds how many interleaved conversations can stay on the fast path.
DEFAULT_CONTINUATIONS = 64


class ChatTemplateError(ValueError):
    """A messages list the chat template cannot be asked to render.

    Raised instead of letting a missing key surface as a ``KeyError``, so the
    HTTP layer can answer 400 for a malformed request body rather than 500.
    """


# Needle lengths tried, longest first, when locating the seam in a fresh
# rendering.  A long needle cannot collide by accident; a short one is what a
# brief reply leaves room for once the ``<think>`` block is excluded.  Nothing
# below 8 is attempted -- the risk of a coincidental match stops being
# negligible, and a reply that short is not worth a splice.
_SEAM_NEEDLES = (24, 16, 8)


def _splice(
    head: Sequence[int], plen: int, ids: Sequence[int]
) -> Optional[List[int]]:
    """Rebuild *ids* on *head*, or ``None`` if the seam cannot be found.

    *head* is the whole canonical sequence through the previous reply; *ids* is
    this turn rendered afresh, which carries the same conversation without the
    ``<think>`` blocks and then continues into the new turn.  Everything *head*
    already covers ends, in *ids*, at the last occurrence of the reply's tail --
    so finding that tail gives the seam, and the remainder of *ids* is the new
    turn verbatim.

    Matching on the reply's *tail* rather than the prompt's head is what lets a
    conversation splice more than once: after the first splice *head* carries
    think blocks that no fresh rendering of a later turn will ever contain, so
    any comparison anchored at the front is guaranteed to diverge.  A reply that
    did not re-tokenize to the ids the model emitted simply fails to match here
    and falls back, which is the intended safe outcome.

    Needles are tried longest first and never reach past the generated region,
    because the ``<think>`` block sits at its *start* and appears in no fresh
    rendering -- a needle long enough to touch it can never match.
    """
    head = list(head)
    ids = list(ids)
    if not 0 < plen < len(head):
        return None
    generated = len(head) - plen
    for span in _SEAM_NEEDLES:
        if span > generated:
            continue
        needle = head[len(head) - span:]
        # Search from the end: the reply is the last assistant content in *ids*,
        # and a repetitive one could also match earlier.
        for at in range(len(ids) - span, -1, -1):
            if ids[at:at + span] == needle:
                return head + ids[at + span:]
    return None


def _state_key(messages: Sequence[dict]) -> str:
    """Identify one conversation state — role/content pairs, in order."""
    digest = hashlib.blake2b(digest_size=16)
    for message in messages:
        digest.update(message["role"].encode("utf-8"))
        digest.update(b"\x00")
        digest.update(message["content"].encode("utf-8"))
        digest.update(b"\x00")
    return digest.hexdigest()


def _flatten_content(content: Any, where: str) -> str:
    """Collapse OpenAI content parts into the plain text a template renders."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
                continue
            if isinstance(part, dict) and part.get("type", "text") == "text":
                text = part.get("text")
                if not isinstance(text, str):
                    raise ChatTemplateError(
                        f"{where}: text content part has no 'text' string")
                parts.append(text)
                continue
            kind = part.get("type") if isinstance(part, dict) else type(part).__name__
            raise ChatTemplateError(
                f"{where}: unsupported content part of type {kind!r}")
        return "".join(parts)
    raise ChatTemplateError(
        f"{where}: 'content' must be a string or a list of content parts")


def normalize_messages(messages: Any) -> list[dict]:
    """Validate an OpenAI messages list and render it template-ready.

    Content parts are flattened to text and the default system message is
    prepended only when the client sent none, so a client that supplies its own
    system turn keeps a prompt that is byte-identical to what it asked for.
    Idempotent: a list this returned normalizes to itself.
    """
    if not isinstance(messages, list) or not messages:
        raise ChatTemplateError("'messages' must be a non-empty list")
    out: list[dict] = []
    has_system = False
    for index, message in enumerate(messages):
        where = f"messages[{index}]"
        if not isinstance(message, dict):
            raise ChatTemplateError(f"{where} is not an object")
        role = message.get("role")
        if not isinstance(role, str) or not role:
            raise ChatTemplateError(f"{where} has no 'role' string")
        if "content" not in message:
            raise ChatTemplateError(f"{where} ({role}) has no 'content'")
        out.append({
            "role": role,
            "content": _flatten_content(message["content"], where),
        })
        has_system = has_system or role == "system"
    if not has_system:
        out.insert(0, {"role": "system", "content": DEFAULT_SYSTEM_PROMPT})
    return out


class TokenizerManager:
    """Wraps a HuggingFace tokenizer for thread-safe tokenization/detokenization."""

    def __init__(
        self, tokenizer, continuations: int = DEFAULT_CONTINUATIONS
    ) -> None:
        self._tokenizer = tokenizer
        self._lock = threading.Lock()
        # Remembered conversations: key -> (canonical ids, prompt length).
        # Guarded by the same ``_lock`` as the tokenizer.  Every use is a short
        # dict operation taken on its own, never nested inside a render or an
        # encode, so a plain lock is enough.
        self._conts: "collections.OrderedDict[str, Tuple[List[int], int]]" = (
            collections.OrderedDict())
        self._cont_cap = max(0, continuations)
        self._cont_stats: Dict[str, int] = collections.Counter()

    def tokenize_messages(
        self, messages: Sequence[dict], add_generation_prompt: bool = True
    ) -> list[int]:
        """Render a whole conversation through the chat template and tokenize.

        Rendering the full list — not just its last user turn — is what makes
        turn *k*'s prompt a token prefix of turn *k+1*'s, which is the shape the
        OIPL prefix cache matches on (§11-1).

        When this conversation's previous turn is remembered, the head of the
        result is that turn's *canonical* id sequence rather than a fresh
        rendering.  See :meth:`_continue` for why that is what makes the
        generated KV pages reusable.
        """
        rendered = normalize_messages(messages)
        ids = self._encode(self._render(rendered, add_generation_prompt))
        spliced = self._continue(rendered, ids)
        return spliced if spliced is not None else ids

    # ── conversation continuation ─────────────────────────────────────────

    def remember_continuation(
        self,
        messages: Sequence[dict],
        reply: str,
        canonical_ids: Sequence[int],
        prompt_len: int,
    ) -> None:
        """Record what this conversation's tokens *actually* were.

        *canonical_ids* is the sequence the kernel ran on -- the prompt plus the
        tokens it generated, exactly as they sit behind the KV pages -- and
        *prompt_len* is where the generated part starts.  The next turn splices
        this back in instead of re-rendering, which is what keeps the cached
        pages at the absolute positions their KV was computed for.
        """
        if self._cont_cap <= 0 or not canonical_ids:
            return
        if not 0 < prompt_len < len(canonical_ids):
            return
        state = list(normalize_messages(messages))
        state.append({"role": "assistant", "content": reply})
        key = _state_key(state)
        with self._lock:
            self._conts[key] = (list(canonical_ids), prompt_len)
            self._conts.move_to_end(key)
            while len(self._conts) > self._cont_cap:
                self._conts.popitem(last=False)

    def _continue(
        self, rendered: Sequence[dict], ids: List[int]
    ) -> Optional[List[int]]:
        """Rebuild *ids* on the remembered head, or return ``None``.

        Qwen3's template gives the *last* assistant turn an empty ``<think>``
        block and a historical one none, so a freshly rendered turn *k+1* is
        turn *k*'s sequence with that block missing -- every token after it
        shifts.  K is stored post-RoPE, so a shifted page is invalid however
        well its ids match, and the generated pages can never be imported.
        Splicing the canonical head back in restores the positions the KV was
        computed at.

        Alignment, not comparison, is what finds the seam.  Both sequences
        share turn *k*'s prompt, then diverge::

            head :  [prompt_k][think][answer]
            ids  :  [prompt_k][answer][closer][turn k+1 ...]

        so locating where *head*'s answer begins says how much of *ids* the
        head already covers, and the rest of *ids* is the new turn.  The match
        is verified before it is used, which is also the guard against a reply
        that did not re-tokenize to the ids the model emitted: an answer that
        cannot be aligned simply falls back to the fresh rendering.
        """
        if self._cont_cap <= 0:
            return None
        for cut in range(len(rendered), 1, -1):
            if rendered[cut - 1].get("role") != "assistant":
                continue
            key = _state_key(rendered[:cut])
            with self._lock:
                entry = self._conts.get(key)
                if entry is not None:
                    self._conts.move_to_end(key)
            if entry is None:
                continue
            head, plen = entry
            spliced = _splice(head, plen, ids)
            if spliced is None:
                self._cont_stats["unalignable"] += 1
                return None
            self._cont_stats["hits"] += 1
            self._cont_stats["spliced_tokens"] += len(spliced) - len(ids)
            return spliced
        self._cont_stats["misses"] += 1
        return None

    @property
    def continuation_stats(self) -> Dict[str, int]:
        """Counter snapshot; ``resident`` is the live conversation count."""
        snapshot = {"hits": 0, "misses": 0, "unalignable": 0,
                    "spliced_tokens": 0}
        snapshot.update(self._cont_stats)
        with self._lock:
            snapshot["resident"] = len(self._conts)
        return snapshot

    def _render(
        self, messages: Sequence[dict], add_generation_prompt: bool
    ) -> str:
        with self._lock:
            return self._tokenizer.apply_chat_template(
                list(messages), tokenize=False,
                add_generation_prompt=add_generation_prompt,
            )

    def tokenize(self, prompt: str, use_template: bool = True) -> list[int]:
        """Apply chat template (if requested) and return token IDs."""
        if use_template:
            return self.tokenize_messages([{"role": "user", "content": prompt}])
        return self._encode(prompt)

    def decode(self, token_ids: list[int]) -> str:
        """Decode token IDs to text, skipping special tokens."""
        with self._lock:
            return self._tokenizer.decode(token_ids, skip_special_tokens=True)

    def decode_single(self, token_id: int) -> str:
        """Decode a single token ID to text."""
        with self._lock:
            return self._tokenizer.decode([token_id], skip_special_tokens=True)

    def _encode(self, text: str) -> list[int]:
        with self._lock:
            return self._tokenizer([text], return_tensors="pt").input_ids[0].tolist()
