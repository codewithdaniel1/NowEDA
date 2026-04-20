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


def _progress_bar(percent, width=24):
    filled = int(round(width * max(0, min(percent, 100)) / 100))
    filled = max(0, min(filled, width))
    return "[" + ("#" * filled) + ("-" * (width - filled)) + "]"


def _render_line(message, percent, state="running"):
    bar = _progress_bar(100 if state == "done" else percent)
    if state == "running":
        return f"{bar} {percent:3d}% {message}"
    if state == "done":
        return f"{bar} 100% {message}"
    return f"[{'!' * 24}] ERR  {message}"


def _notebook_html(message, percent, state="running"):
    return f"<pre>{html.escape(_render_line(message, percent, state))}</pre>"


class _ProgressDisplay:
    def __init__(self, message):
        self.message = message
        self.percent = 0
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

    def _elapsed_percent(self):
        elapsed = time.monotonic() - self._start_time
        # A small, traditional-feeling ramp: 0 -> 99 over ~8 seconds.
        # The last 1% is reserved for the explicit completion state.
        return min(99, int((elapsed / 8.0) * 99))

    def _update_terminal(self):
        line = _render_line(self.message, self.percent, self.state)
        padded = line.ljust(self._prev_len)
        sys.stderr.write("\r" + padded)
        sys.stderr.flush()
        self._prev_len = len(line)

    def _update_notebook(self):
        if self._handle is None:
            return
        try:
            from IPython.display import HTML

            self._handle.update(HTML(_notebook_html(self.message, self.percent, self.state)))
        except Exception:
            pass

    def _tick(self):
        new_percent = self._elapsed_percent()
        with self._lock:
            if new_percent > self.percent:
                self.percent = new_percent
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
            if state == "done":
                self.percent = 100

            if self._handle is not None:
                self._update_notebook()
            elif sys.stderr.isatty():
                self._update_terminal()
            else:
                print(_render_line(self.message, self.percent, state), file=sys.stderr)

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
    except Exception:
        progress.finish("error")
        raise
    else:
        progress.finish("done")
