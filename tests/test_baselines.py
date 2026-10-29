from baselines.error_line import ErrorLineRemaskBaseline
from verimask.diagnostics import naive_tokenize, parse_python_traceback


def test_error_line_baseline_selects_near_traceback():
    code = "line1\nline2 bug\nline3\n"
    tb = 'File "candidate.py", line 2, in f\nValueError: x'
    diag = parse_python_traceback(tb, source_basename="candidate.py")
    _, offsets = naive_tokenize(code)
    bl = ErrorLineRemaskBaseline(margin=0)
    decision = bl.select_spans(code, diag, offsets, max_tokens=5, seed=0)
    assert decision.token_indices
