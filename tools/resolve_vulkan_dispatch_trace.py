#!/usr/bin/env python3
"""Resolve legacy Hierarchos Vulkan dependency-trace shader hashes to names.

Current runtimes emit both stable hashes and compile-time-resolved kernel names.
This helper remains useful for older trace logs: it derives the same FNV-1a
signature from the checked-in SPIR-V files and annotates trace lines from stdin
(or a supplied log file) without changing runtime shader ABIs.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SHADER_DIR = ROOT / "hierarchos-vulkan" / "shaders"
EDGE_RE = re.compile(
    r"hierarchos_vulkan_dependency_edge "
    r"producer=0x(?P<producer>[0-9a-fA-F]{16}) "
    r"consumer=0x(?P<consumer>[0-9a-fA-F]{16}) "
    r"count=(?P<count>\d+)"
    r"(?: hazard=(?P<hazard>raw|war|waw))?"
)


def fnv1a64(data: bytes) -> int:
    value = 0xCBF29CE484222325
    for byte in data:
        value ^= byte
        value = (value * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return value


def shader_names() -> dict[int, str]:
    names: dict[int, str] = {}
    for path in sorted(SHADER_DIR.glob("*.spv")):
        signature = fnv1a64(path.read_bytes())
        previous = names.get(signature)
        if previous is not None and previous != path.name:
            raise RuntimeError(
                f"SPIR-V signature collision 0x{signature:016x}: "
                f"{previous!r} and {path.name!r}"
            )
        names[signature] = path.name
    return names


def resolve_line(line: str, names: dict[int, str]) -> str:
    match = EDGE_RE.search(line)
    if match is None:
        return line.rstrip("\n")
    producer = int(match.group("producer"), 16)
    consumer = int(match.group("consumer"), 16)
    producer_name = names.get(producer, f"0x{producer:016x}")
    consumer_name = names.get(consumer, f"0x{consumer:016x}")
    hazard = match.group("hazard")
    hazard_suffix = f" hazard={hazard}" if hazard is not None else ""
    return (
        f"hierarchos_vulkan_dependency_edge "
        f"producer={producer_name} consumer={consumer_name} "
        f"count={match.group('count')}{hazard_suffix}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "log",
        nargs="?",
        type=Path,
        help="Trace log to resolve; reads stdin when omitted",
    )
    args = parser.parse_args()
    names = shader_names()
    if args.log is None:
        lines = sys.stdin
        for line in lines:
            print(resolve_line(line, names))
        return
    with args.log.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            print(resolve_line(line, names))


if __name__ == "__main__":
    main()
