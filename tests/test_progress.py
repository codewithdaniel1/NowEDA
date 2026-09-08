"""Loading indicators and readers must finish on exhaustion, closure and errors."""
from contextlib import closing
import sys
from types import SimpleNamespace

import pandas as pd
import pytest

import noweda
import noweda.ui as ui


@pytest.fixture
def source_and_readers(tmp_path, monkeypatch):
    path = tmp_path / "rows.csv"
    path.write_text("x\n1\n2\n3\n")
    original = pd.read_csv
    readers = []
    def track(*args, **kwargs):
        reader = original(*args, **kwargs)
        readers.append(reader)
        return reader
    monkeypatch.setattr(pd, "read_csv", track)
    monkeypatch.setattr(ui, "_in_notebook", lambda: False)
    return path, readers


@pytest.mark.parametrize("concat", [True, False])
def test_chunk_completion_closes_reader_and_reports_100(source_and_readers, capsys, concat):
    path, readers = source_and_readers
    result = noweda.read_chunked(path, chunksize=1, concat=concat)
    if not concat:
        result = pd.concat(list(result))
    assert result.x.tolist() == [1, 2, 3]
    assert readers[0].handles.handle.closed
    output = capsys.readouterr().err
    assert "100%" in output
    assert "99%" not in output
    assert "Stopped" not in output


def test_early_break_with_closing_stops_instead_of_claiming_completion(source_and_readers, capsys):
    path, readers = source_and_readers
    with closing(noweda.read_chunked(path, chunksize=1, concat=False)) as chunks:
        for chunk in chunks:
            assert chunk.x.tolist() == [1]
            break
    assert readers[0].handles.handle.closed
    output = capsys.readouterr().err
    assert "Stopped before completion" in output
    assert "100%" not in output


def test_unstarted_generator_never_opens_reader(source_and_readers):
    path, readers = source_and_readers
    chunks = noweda.read_chunked(path, concat=False)
    chunks.close()
    assert readers == []


def test_processing_failure_closes_retained_generator(source_and_readers, capsys):
    path, readers = source_and_readers
    chunks = noweda.read_chunked(path, chunksize=1, concat=False)
    with pytest.raises(RuntimeError, match="processing failed"):
        with closing(chunks):
            for chunk in chunks:
                raise RuntimeError("processing failed")
    assert readers[0].handles.handle.closed
    assert "Stopped before completion" in capsys.readouterr().err


@pytest.mark.parametrize("concat", [True, False])
def test_reader_failure_closes_reader_and_reports_error(tmp_path, monkeypatch, capsys, concat):
    class FailingReader:
        closed = False
        def __iter__(self):
            return self
        def __next__(self):
            raise ValueError("bad data")
        def close(self):
            self.closed = True
    reader = FailingReader()
    path = tmp_path / "bad.csv"
    path.touch()
    monkeypatch.setattr(ui, "_in_notebook", lambda: False)
    monkeypatch.setattr(pd, "read_csv", lambda *a, **kw: reader)
    with pytest.raises(ValueError, match="bad data"):
        result = noweda.read_chunked(path, concat=concat)
        if not concat:
            list(result)
    assert reader.closed
    output = capsys.readouterr().err
    assert "ERR" in output
    assert "100%" not in output


def test_keyboard_interrupt_stops_indicator(monkeypatch, capsys):
    monkeypatch.setattr(ui, "_in_notebook", lambda: False)
    with pytest.raises(KeyboardInterrupt):
        with ui.loading("interrupted task"):
            raise KeyboardInterrupt
    assert "Stopped before completion" in capsys.readouterr().err


def test_notebook_display_closes_thread_and_shows_final_state(monkeypatch):
    updates = []
    handle = SimpleNamespace(update=lambda value: updates.append(value.data))
    def display(value, **kwargs):
        updates.append(value.data)
        return handle
    monkeypatch.setitem(sys.modules, "IPython.display", SimpleNamespace(
        HTML=lambda value: SimpleNamespace(data=value), display=display))
    monkeypatch.setattr(ui, "_in_notebook", lambda: True)
    for state, expected in [("done", "100%"), ("stopped", "Stopped before completion"), ("error", "ERR")]:
        progress = ui._ProgressDisplay("read <data>")
        progress.start()
        progress._start_time -= 100  # Long jobs must not acquire a fake 99%.
        progress._tick()
        assert "Working" in updates[-1]
        assert "%" not in updates[-1]
        progress.finish(state)
        assert expected in updates[-1]
        assert "&lt;data&gt;" in updates[-1]
        assert not progress._thread.is_alive()
