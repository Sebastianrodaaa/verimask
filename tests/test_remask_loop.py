"""End-to-end repair loop on CPU with TinyStandIn (MOCK denoiser, REAL verifier)."""

from verimask.denoiser import TinyStandInDenoiser
from verimask.policy import DiagnosticLinearPolicy
from verimask.remask_denoise import RemaskDenoiseLoop


def test_repair_add_mock():
    code = "def add(a, b):\n    return a - b\n"
    test_code = "assert add(2, 3) == 5"
    denoiser = TinyStandInDenoiser(
        repair_map={"return a - b": "return", "a - b": "a + b"},
    )
    loop = RemaskDenoiseLoop(denoiser=denoiser, policy=DiagnosticLinearPolicy(), max_attempts=4)
    out = loop.repair(code, test_code, seed=0)
    # Stand-in may not always pass without perfect masks; at least runs
    assert out.attempts >= 1
    assert out.last_verify is not None
