#!/usr/bin/env python3
"""Regenerate README results table marker from results/summary.json."""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean


def main() -> None:
    summary_path = Path("results/summary.json")
    readme = Path("README.md")
    if not summary_path.exists():
        print("No results/summary.json; run make repro first")
        return
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    rows = summary.get("rows", [])
    by_arm: dict[str, list[float]] = {}
    labels: dict[str, str] = {}
    for r in rows:
        by_arm.setdefault(r["arm"], []).append(r["repair_rate"])
        labels[r["arm"]] = r.get("label", "MOCK")
    lines = [
        "| Arm | Mean repair rate | Label |",
        "| --- | ---: | --- |",
    ]
    for arm, rates in sorted(by_arm.items()):
        lines.append(f"| `{arm}` | {mean(rates):.3f} | {labels.get(arm, 'MOCK')} |")
    table = "\n".join(lines)
    content = readme.read_text(encoding="utf-8")
    start = "<!-- RESULTS_TABLE_START -->"
    end = "<!-- RESULTS_TABLE_END -->"
    if start not in content:
        print("README missing results markers")
        return
    new_block = f"{start}\n{table}\n{end}"
    import re

    content = re.sub(rf"{start}.*?{end}", new_block, content, flags=re.DOTALL)
    readme.write_text(content, encoding="utf-8")
    print("Updated README results table")


if __name__ == "__main__":
    main()
