"""Threaded LLM streaming so the UI can interrupt a generation.

Streamlit runs each session's script single-threaded, top-to-bottom, so a
synchronous ``llm.invoke`` blocks the script and no "Stop" click can be
processed until it returns. Running the invoke in a background thread that
streams tokens into a queue lets the main thread poll the queue (via a
self-rerunning ``st.fragment``) and stay responsive to the Stop button.

The worker thread touches only a ``queue.Queue`` and ``threading.Event`` —
never any ``st.*`` API — so it needs no Streamlit script-run context and this
module is unit-testable without Streamlit installed.
"""
import queue
import threading

from langchain_core.callbacks.base import BaseCallbackHandler


class GenerationStopped(Exception):
    """Raised inside the stream callback to abort a generation on user request."""


class QueueStreamHandler(BaseCallbackHandler):
    """LangChain callback that pushes tokens onto a queue and honours a stop flag.

    Raising :class:`GenerationStopped` when the stop event is set aborts the
    underlying ``llm.invoke`` for backends that propagate callback exceptions.
    The UI does not rely on that propagation for responsiveness — see
    :class:`LLMWorker` — but it lets cooperating backends stop promptly.
    """

    def __init__(self, token_queue: "queue.Queue", stop_event: threading.Event):
        self.queue = token_queue
        self.stop_event = stop_event

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        if self.stop_event.is_set():
            raise GenerationStopped("Generation stopped by user")
        self.queue.put(token)


class LLMWorker:
    """Run ``llm.invoke(prompt)`` in a daemon thread, streaming tokens to a queue.

    Usage::

        worker = LLMWorker()
        llm = build_llm(QueueStreamHandler(worker.queue, worker.stop_event))
        worker.start(llm, prompt)
        ...                       # poll worker.drain() / worker.done
        worker.stop()             # request abort

    The queue and stop/done events are created in ``__init__`` so the caller can
    wire a :class:`QueueStreamHandler` to the same objects before the LLM exists.
    """

    def __init__(self):
        self.queue: "queue.Queue[str]" = queue.Queue()
        self.stop_event = threading.Event()
        self.done_event = threading.Event()
        self.result: str = ""
        self.error: Exception | None = None
        self.stopped: bool = False
        self._thread: threading.Thread | None = None

    def start(self, llm, prompt: str) -> None:
        """Launch the background invoke. Returns immediately."""
        self._thread = threading.Thread(
            target=self._run, args=(llm, prompt), daemon=True
        )
        self._thread.start()

    def _run(self, llm, prompt: str) -> None:
        try:
            result = llm.invoke(prompt)
            self.result = result if isinstance(result, str) else str(result)
        except GenerationStopped:
            self.stopped = True
        except Exception as e:  # captured for the main thread to render/redact
            self.error = e
        finally:
            self.done_event.set()

    def stop(self) -> None:
        """Signal the generation to abort. The main thread need not wait for it."""
        self.stop_event.set()

    def drain(self) -> str:
        """Return all tokens queued since the last drain (possibly empty)."""
        tokens = []
        while True:
            try:
                tokens.append(self.queue.get_nowait())
            except queue.Empty:
                break
        return "".join(tokens)

    @property
    def done(self) -> bool:
        return self.done_event.is_set()
