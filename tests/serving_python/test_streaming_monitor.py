"""CPU-only tests for the engine's shared token-streaming monitor."""

from __future__ import annotations

import importlib.util
import pathlib
import queue
import sys
import threading
import time
import types


def _load_llm_engine():
    """Load llm_engine without importing Mirage's compiled dependencies."""
    root = pathlib.Path(__file__).resolve().parents[2]
    package = "streaming_monitor_test_package"
    injected_torch = "torch" not in sys.modules
    if injected_torch:
        sys.modules["torch"] = types.ModuleType("torch")
    for name in (package, f"{package}.engine", f"{package}.mpk"):
        module = types.ModuleType(name)
        module.__path__ = []
        sys.modules[name] = module

    model_runner = types.ModuleType(f"{package}.engine.model_runner")
    model_runner.ModelRunner = object
    sys.modules[model_runner.__name__] = model_runner

    tokenizer_manager = types.ModuleType(f"{package}.engine.tokenizer_manager")
    tokenizer_manager.TokenizerManager = object
    sys.modules[tokenizer_manager.__name__] = tokenizer_manager

    runtime = types.ModuleType(f"{package}.mpk.online_pinned_runtime")
    runtime.OnlinePinnedRuntime = object
    sys.modules[runtime.__name__] = runtime

    name = f"{package}.engine.llm_engine"
    spec = importlib.util.spec_from_file_location(
        name, root / "python" / "mirage" / "engine" / "llm_engine.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        if injected_torch:
            del sys.modules["torch"]
    return module


llm_engine = _load_llm_engine()


class _Tokens:
    def __init__(self, values):
        self._values = values

    def tolist(self):
        return list(self._values)


class _FakeTokenizerManager:
    def decode_each(self, token_ids):
        return [f"<{token_id}>" for token_id in token_ids]


class _FakeRuntime:
    def __init__(self):
        self.rows = {1: 0, 2: 1}
        self.steps = {0: 2, 1: 2}
        self.tokens = {0: [100, 11, 12], 1: [200, 21, 22]}
        self.completions = {}
        self.released = []
        self.abandoned = []
        self.read_calls = []
        self.allow_rows = threading.Event()

    def get_completion(self, rid):
        return self.completions.get(rid)

    def find_row_for_rid(self, rid):
        return self.rows[rid] if self.allow_rows.is_set() else -1

    def get_current_step_at_row(self, row):
        return self.steps[row]

    def read_token_ranges(self, ranges):
        self.read_calls.append(list(ranges))
        return [_Tokens(self.tokens[row][start:end + 1])
                for row, start, end in ranges]

    def release_request(self, rid):
        if rid not in self.completions:
            return False
        self.released.append(rid)
        del self.completions[rid]
        return True

    def abandon_request(self, rid):
        self.abandoned.append(rid)


def _get(q, timeout=1.0):
    try:
        return q.get(timeout=timeout)
    except queue.Empty as exc:  # pragma: no cover - clearer test failure
        raise AssertionError("streaming monitor produced no item") from exc


def test_active_streams_share_one_batched_read_and_complete_cleanly():
    runtime = _FakeRuntime()
    monitor = llm_engine._StreamingMonitor(runtime, _FakeTokenizerManager())
    try:
        first = monitor.register(1, prompt_len=1, timeout=1.0)
        second = monitor.register(2, prompt_len=1, timeout=1.0)
        runtime.allow_rows.set()

        assert _get(first) == ("<11>", False)
        assert _get(first) == ("<12>", False)
        assert _get(second) == ("<21>", False)
        assert _get(second) == ("<22>", False)

        assert any(len(call) == 2 for call in runtime.read_calls)

        runtime.completions.update({1: (0, 2), 2: (1, 2)})
        assert _get(first) == ("", True)
        assert _get(second) == ("", True)

        deadline = time.monotonic() + 1.0
        while len(runtime.released) < 2 and time.monotonic() < deadline:
            time.sleep(0.001)
        assert sorted(runtime.released) == [1, 2]
    finally:
        monitor.shutdown()


def test_closing_a_stream_generator_cancels_its_runtime_session():
    class _Monitor:
        def __init__(self):
            self.cancelled = []

        def cancel(self, rid):
            self.cancelled.append(rid)

    engine = object.__new__(llm_engine.LLMEngine)
    engine._monitor = _Monitor()
    token_queue = queue.Queue()
    token_queue.put(("hello", False))

    stream = engine._submit_stream(7, token_queue)
    assert next(stream) == ("hello", False)
    stream.close()

    assert engine._monitor.cancelled == [7]
