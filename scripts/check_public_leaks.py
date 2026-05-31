"""Scan public-facing docs and artifacts for obvious private path leaks."""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path


DEFAULT_ROOTS = (Path("README.md"), Path("docs"), Path("paper"), Path("outputs/public"))
TEXT_SUFFIXES = {".cff", ".csv", ".json", ".md", ".tex", ".txt", ".yaml", ".yml"}


@dataclass(frozen=True)
class LeakPattern:
    """Named pattern for public/private leak scanning."""

    name: str
    pattern: re.Pattern[str]


LEAK_PATTERNS = (
    LeakPattern("private output path", re.compile(r"outputs/private")),
    LeakPattern("local user path", re.compile(r"/Users/[A-Za-z0-9._-]+")),
    LeakPattern("temporary data path", re.compile(r"/mnt/data")),
    LeakPattern("private checkpoint path", re.compile(r"(?<![A-Za-z0-9_.-])checkpoints/[^\s\"'<>]+\.pt")),
)


@dataclass(frozen=True)
class LeakFinding:
    """One public/private leak finding."""

    path: Path
    line_number: int
    pattern_name: str
    line: str


def find_public_leaks(roots: list[Path] | tuple[Path, ...] = DEFAULT_ROOTS) -> list[LeakFinding]:
    """Return public/private path findings under public-facing roots."""

    findings: list[LeakFinding] = []
    for path in iter_text_files(roots):
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except UnicodeDecodeError:
            continue
        for line_number, line in enumerate(lines, start=1):
            for leak_pattern in LEAK_PATTERNS:
                if leak_pattern.pattern.search(line):
                    findings.append(
                        LeakFinding(
                            path=path,
                            line_number=line_number,
                            pattern_name=leak_pattern.name,
                            line=line.strip(),
                        )
                    )
    return findings


def iter_text_files(roots: list[Path] | tuple[Path, ...]) -> list[Path]:
    """Return text-like files from the requested roots."""

    files: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        if root.is_file():
            if root.suffix in TEXT_SUFFIXES:
                files.append(root)
            continue
        for path in root.rglob("*"):
            if path.is_file() and path.suffix in TEXT_SUFFIXES:
                files.append(path)
    return sorted(files)


def main() -> None:
    parser = argparse.ArgumentParser(description="Scan public docs/artifacts for private path leaks.")
    parser.add_argument("roots", nargs="*", type=Path, default=list(DEFAULT_ROOTS))
    args = parser.parse_args()

    findings = find_public_leaks(tuple(args.roots))
    if findings:
        for finding in findings:
            print(
                f"{finding.path}:{finding.line_number}: {finding.pattern_name}: {finding.line}",
                file=sys.stderr,
            )
        raise SystemExit(1)
    print("No public/private path leaks found.")


if __name__ == "__main__":
    main()
