"""Tests for the threaded streaming helpers (queue handler + worker)."""

import queue
import threading
import time

import pytest

from streaming import GenerationStopped, LLMWorker, QueueStreamHandler


def _wait_done(worker, timeout=2.0):
    deadline = time.time() + timeout
    while not worker.done and time.time() < deadline:
        time.sleep(0.005)
    assert worker.done, "worker did not finish within timeout"


# ── QueueStreamHandler ────────────────────────────────────────────────────────

def test_handler_enqueues_tokens():
    q: queue.Queue = queue.Queue()
    handler = QueueStreamHandler(q, threading.Event())
    handler.on_llm_new_token("a")
    handler.on_llm_new_token("b")
    assert q.get_nowait() == "a"
    assert q.get_nowait() == "b"


def test_handler_raises_when_stop_set():
    ev = threading.Event()
    ev.set()
    handler = QueueStreamHandler(queue.Queue(), ev)
    with pytest.raises(GenerationStopped):
        handler.on_llm_new_token("x")


# ── LLMWorker.drain ───────────────────────────────────────────────────────────

def test_drain_returns_all_then_empties():
    worker = LLMWorker()
    worker.queue.put("a")
    worker.queue.put("b")
    assert worker.drain() == "ab"
    assert worker.drain() == ""


# ── LLMWorker end-to-end ──────────────────────────────────────────────────────

class _FakeLLM:
    def __init__(self, handler, tokens):
        self._handler = handler
        self._tokens = tokens

    def invoke(self, prompt):
        out = ""
        for token in self._tokens:
            self._handler.on_llm_new_token(token)  # may raise GenerationStopped
            out += token
        return out


def test_worker_streams_and_completes():
    worker = LLMWorker()
    handler = QueueStreamHandler(worker.queue, worker.stop_event)
    worker.start(_FakeLLM(handler, ["Hel", "lo"]), "hi")
    _wait_done(worker)
    assert worker.result == "Hello"
    assert worker.error is None
    assert worker.stopped is False


def test_worker_captures_error():
    class _BoomLLM:
        def invoke(self, prompt):
            raise RuntimeError("boom")

    worker = LLMWorker()
    QueueStreamHandler(worker.queue, worker.stop_event)
    worker.start(_BoomLLM(), "x")
    _wait_done(worker)
    assert isinstance(worker.error, RuntimeError)
    assert worker.stopped is False


def test_worker_stop_marks_stopped():
    class _SlowLLM:
        def __init__(self, handler):
            self._handler = handler

        def invoke(self, prompt):
            for _ in range(100000):
                self._handler.on_llm_new_token("t ")
                time.sleep(0.001)
            return "finished"

    worker = LLMWorker()
    handler = QueueStreamHandler(worker.queue, worker.stop_event)
    worker.start(_SlowLLM(handler), "x")
    time.sleep(0.03)
    worker.stop()
    _wait_done(worker)
    assert worker.stopped is True
    assert worker.error is None
