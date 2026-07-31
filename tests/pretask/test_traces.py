"""Stage A unit tests for Module 2 (stack-trace parser)."""

from __future__ import annotations

import os

from groundtruth.pretask.traces import parse_stack_trace_hints, parse_stack_traces


PYTHON_TRACE = '''Traceback (most recent call last):
  File "patroni/postmaster.py", line 89, in start
    self.watchdog.activate()
  File "patroni/watchdog.py", line 142, in activate
    self._fd.write(b"\\x56")
OSError: [Errno 9] Bad file descriptor
'''


def test_traces_python(tmp_path) -> None:
    """Python traceback frames are extracted with line + func."""
    repo = tmp_path  # any prefix works since paths are relative
    for relative in ("patroni/postmaster.py", "patroni/watchdog.py"):
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")
    frames = parse_stack_traces(PYTHON_TRACE, str(repo))
    # Both frames name files proven to exist under the repository.
    assert len(frames) == 2
    # Deepest first → activate at line 142 should come first.
    assert frames[0].func == "activate"
    assert frames[0].line == 142
    assert frames[0].lang == "python"
    assert frames[1].func == "start"


def test_traces_in_repo_filter(tmp_path) -> None:
    """Frames pointing at site-packages / stdlib are dropped."""
    handler = tmp_path / "myrepo" / "handlers.py"
    handler.parent.mkdir(parents=True)
    handler.write_text("", encoding="utf-8")
    text = (
        'Traceback (most recent call last):\n'
        '  File "/usr/lib/python3.11/threading.py", line 980, in run\n'
        '    self._target(*self._args, **self._kwargs)\n'
        '  File "myrepo/handlers.py", line 12, in handler\n'
        '    raise ValueError\n'
    )
    frames = parse_stack_traces(text, str(tmp_path))
    files = [fr.file for fr in frames]
    assert any("handlers.py" in f for f in files)
    assert not any("threading.py" in f for f in files)


def test_traces_javascript(tmp_path) -> None:
    """V8-style ``at fn (path:line:col)`` frames parse."""
    source = tmp_path / "src" / "foo.ts"
    source.parent.mkdir(parents=True)
    source.write_text("", encoding="utf-8")
    text = (
        "TypeError: Cannot read property 'x' of undefined\n"
        "    at Foo.bar (src/foo.ts:42:15)\n"
        "    at processTicksAndRejections (node:internal/process/task_queues:96:5)\n"
    )
    frames = parse_stack_traces(text, str(tmp_path))
    files = [fr.file for fr in frames]
    # The internal frame is filtered by the in-repo check (absolute-ish
    # form is treated as not-in-repo). The user frame stays.
    assert "src/foo.ts" in files


def test_traces_javascript_order_and_vendor_filter(tmp_path) -> None:
    """V8 frames keep deepest-at-top order and drop dependency/runtime paths."""
    source = tmp_path / "src" / "App.tsx"
    source.parent.mkdir(parents=True)
    source.write_text("", encoding="utf-8")
    text = (
        "TypeError: Cannot read property 'x' of undefined\n"
        "    at handleClick (src/App.tsx:42:10)\n"
        "    at processTicksAndRejections (node:internal/process/task_queues:96:5)\n"
        "    at lodashMap (node_modules/lodash/map.js:10:1)\n"
    )
    frames = parse_stack_traces(text, str(tmp_path))
    assert [(fr.file, fr.func) for fr in frames] == [
        ("src/App.tsx", "handleClick")
    ]


def test_traces_no_frames(tmp_path) -> None:
    """Issue without any traceback returns []."""
    out = parse_stack_traces("just plain text, no errors here", str(tmp_path))
    assert out == []


def test_traces_empty_inputs(tmp_path) -> None:
    """Empty text or empty repo_root → []."""
    assert parse_stack_traces("", str(tmp_path)) == []
    assert parse_stack_traces(PYTHON_TRACE, "") == []


def test_syntax_hints_keep_repo_lib_directory() -> None:
    frames = parse_stack_trace_hints(
        'File "src/lib/parser.py", line 17, in parse_config'
    )
    assert [frame.file for frame in frames] == ["src/lib/parser.py"]
