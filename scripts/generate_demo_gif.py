#!/usr/bin/env python3
"""Generate MOCK demo GIF: failing program → remask → repair (terminal-style)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.animation as animation
import matplotlib.pyplot as plt

FRAMES = [
    ("$ verimask run --task mock_mbpp_001", "neutral"),
    ("def add(a, b):\n    return a - b  # FAIL", "fail"),
    ("Traceback → line 2: wrong operator", "diag"),
    ("Remask spans: [return, a - b]", "mask"),
    ("def add(a, b):\n    return a + b  # PASS", "pass"),
]


def main() -> None:
    out = Path("docs/assets/demo_verimask.gif")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    colors = {
        "neutral": "#cccccc",
        "fail": "#ff6b6b",
        "diag": "#ffd93d",
        "mask": "#6bcbff",
        "pass": "#51cf66",
    }
    text = ax.text(0.02, 0.5, "", fontsize=11, family="monospace", va="center")

    def update(i: int):
        content, kind = FRAMES[i % len(FRAMES)]
        text.set_text(content)
        fig.patch.set_facecolor(colors.get(kind, "#222"))
        return (text,)

    ani = animation.FuncAnimation(fig, update, frames=len(FRAMES) * 3, interval=900, repeat=True)
    ani.save(out, writer="pillow", fps=2)
    plt.close()
    banner = out.with_suffix(".txt")
    banner.write_text(
        "MOCK — animation uses scripted text, not Dream-Coder output.\n",
        encoding="utf-8",
    )
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
