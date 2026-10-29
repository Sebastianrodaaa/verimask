"""REAL tests: diagnostics parsing on captured tracebacks."""

from verimask.diagnostics import parse_lean_diagnostics, parse_python_traceback

REAL_TRACEBACK = """
Traceback (most recent call last):
  File "/tmp/candidate.py", line 2, in __verimask_tests
    assert add(2, 3) == 5
  File "/tmp/candidate.py", line 2, in add
    return a - b
AssertionError
"""


def test_parse_python_traceback_real():
    bundle = parse_python_traceback(
        REAL_TRACEBACK,
        source_basename="candidate.py",
        default_margin=1,
    )
    assert bundle.source_kind == "python_test"
    assert bundle.failing_assert is not None
    lines = bundle.line_hints()
    assert 2 in lines or 3 in lines


def test_parse_lean_diagnostics_real():
    lean_out = """
/tmp/Foo.lean:12:4: error: unsolved goals
⊢ 1 + 1 = 2
"""
    bundle = parse_lean_diagnostics(lean_out)
    assert bundle.source_kind == "lean"
    assert any(s.start_line == 12 for s in bundle.spans)
    assert bundle.lean_goal is not None
