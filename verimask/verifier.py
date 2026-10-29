"""Execute Python programs against inline tests (sandbox-lite for research harness)."""

from __future__ import annotations

import subprocess
import sys
import tempfile
import textwrap
from dataclasses import dataclass
from pathlib import Path


@dataclass
class VerifyResult:
    passed: bool
    stderr: str
    stdout: str
    returncode: int


def run_python_tests(
    program: str,
    test_code: str,
    *,
    timeout_s: float = 5.0,
) -> VerifyResult:
    """
    Write program + tests to a temp module and run with ``python``.

    ``test_code`` should invoke assertions (e.g. ``assert foo(1) == 2``).
    """
    wrapper = (
        program.rstrip()
        + "\n\n"
        + "def __verimask_tests():\n"
        + textwrap.indent(test_code.strip(), "    ")
        + "\n\n"
        + 'if __name__ == "__main__":\n'
        + "    __verimask_tests()\n"
    )
    with tempfile.TemporaryDirectory(prefix="verimask_") as tmp:
        path = Path(tmp) / "candidate.py"
        path.write_text(wrapper, encoding="utf-8")
        proc = subprocess.run(
            [sys.executable, str(path)],
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    return VerifyResult(
        passed=proc.returncode == 0,
        stderr=proc.stderr,
        stdout=proc.stdout,
        returncode=proc.returncode,
    )
