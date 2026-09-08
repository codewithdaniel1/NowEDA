"""Shared loading UI helpers for notebooks and the CLI."""

from contextlib import contextmanager
import html
import sys
import threading
import time


def _in_notebook():
    try:
        from IPython import get_ipython
    except Exception:
        return False

    shell = get_ipython()
    return bool(shell and shell.__class__.__name__ == "ZMQInteractiveShell")


def _render_line(message, elapsed, state="running"):
    if state == "running":
        return f"[...] Working ({elapsed}s) {message}"
    if state == "done":
        return f"[{'#' * 24}] 100% {message}"
    if state == "stopped":
        return f"[stopped] Stopped before completion: {message}"
    return f"[{'!' * 24}] ERR  {message}"


def _notebook_html(message, elapsed, state="running"):
    return f"<pre>{html.escape(_render_line(message, elapsed, state))}</pre>"


class _ProgressDisplay:
    def __init__(self, message):
        self.message = message
        self.elapsed = 0
        self.state = "running"
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._start_time = time.monotonic()
        self._thread = None
        self._handle = None
        self._prev_len = 0

    def start(self):
        if _in_notebook():
            try:
                from IPython.display import HTML, display

                self._handle = display(
                    HTML(_notebook_html(self.message, 0, "running")),
                    display_id=True,
                )
            except Exception:
                self._handle = None

        if self._handle is not None or sys.stderr.isatty():
            self._thread = threading.Thread(target=self._run_progress, daemon=True)
            self._thread.start()
        elif self._handle is None:
            print(_render_line(self.message, 0, "running"), file=sys.stderr)

    def _update_terminal(self):
        line = _render_line(self.message, self.elapsed, self.state)
        padded = line.ljust(self._prev_len)
        sys.stderr.write("\r" + padded)
        sys.stderr.flush()
        self._prev_len = len(line)

    def _update_notebook(self):
        if self._handle is None:
            return
        try:
            from IPython.display import HTML

            self._handle.update(HTML(_notebook_html(self.message, self.elapsed, self.state)))
        except Exception:
            pass

    def _tick(self):
        with self._lock:
            self.elapsed = int(time.monotonic() - self._start_time)
            if self.state == "running":
                if self._handle is not None:
                    self._update_notebook()
                elif sys.stderr.isatty():
                    self._update_terminal()

    def _run_progress(self):
        while not self._stop.wait(0.12):
            self._tick()

    def finish(self, state="done"):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=0.3)

        with self._lock:
            self.state = state
            if self._handle is not None:
                self._update_notebook()
            elif sys.stderr.isatty():
                self._update_terminal()
            else:
                print(_render_line(self.message, self.elapsed, state), file=sys.stderr)

        if sys.stderr.isatty():
            sys.stderr.write("\n")
            sys.stderr.flush()


@contextmanager
def loading(message):
    """Show a temporary loading indicator while work is running."""
    progress = _ProgressDisplay(message)
    progress.start()
    try:
        yield
    except (GeneratorExit, KeyboardInterrupt):
        progress.finish("stopped")
        raise
    except BaseException:
        progress.finish("error")
        raise
    else:
        progress.finish("done")
