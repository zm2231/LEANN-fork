#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCES_ROOT = ROOT / "packages" / "leann-sources" / "sources"
sys.path.insert(0, str(ROOT / "packages" / "leann-sources" / "src"))
sys.path.insert(0, str(ROOT / "packages" / "leann-core" / "src"))

AGENT_SESSIONS_ROOT = SOURCES_ROOT / "agent-sessions"
from leann_sources.manifest import SourceManifest  # noqa: E402

READER_SPECS = [
    ("claude-code", "ClaudeCodeSourceReader"),
    ("codex", "CodexSourceReader"),
    ("pi-agent", "PiAgentSourceReader"),
]

PREFIXES = ("/Volumes/4/CF", "/Volumes/4/GitHub/cf-lab")
BUCKETS = [500, 1000, 2000, 4000, 8000, 16000, 32000, 10**9]


def _load_reader(name: str, cls_name: str):
    reader_path = AGENT_SESSIONS_ROOT / name / "reader.py"
    spec = importlib.util.spec_from_file_location(f"_count_reader_{name}", reader_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    manifest = SourceManifest.load(AGENT_SESSIONS_ROOT / name / "manifest.yaml")
    return getattr(module, cls_name)(manifest)


def _matches(cwd: str | None) -> bool:
    if not cwd:
        return False
    return any(cwd == p or cwd.startswith(p + "/") for p in PREFIXES)


def main() -> None:
    grand = defaultdict(
        lambda: {"chunks": 0, "chars": 0, "sessions": set(), "hist": defaultdict(int)}
    )
    for name, cls_name in READER_SPECS:
        reader = _load_reader(name, cls_name)
        if not reader.validate().ok:
            print(f"[{name}] invalid, skipped", file=sys.stderr)
            continue
        for chunk in reader.iter_chunks():
            extra = chunk.metadata.get("extra", {})
            if not _matches(extra.get("cwd")):
                continue
            g = grand[name]
            n = len(chunk.text)
            g["chunks"] += 1
            g["chars"] += n
            g["sessions"].add(extra.get("session_id"))
            for b in BUCKETS:
                if n <= b:
                    g["hist"][b] += 1
                    break
        s = grand[name]
        print(
            f"[{name}] sessions={len(s['sessions'])} chunks={s['chunks']} chars={s['chars']:,}",
            file=sys.stderr,
        )

    tc = sum(s["chunks"] for s in grand.values())
    tch = sum(s["chars"] for s in grand.values())
    print("\n=== TOTALS ===", file=sys.stderr)
    print(f"chunks={tc:,}  chars={tch:,}  (~{tch // max(tc, 1)} avg chars/chunk)", file=sys.stderr)
    print("\n=== LENGTH HISTOGRAM (chars) ===", file=sys.stderr)
    labels = {
        500: "<=500",
        1000: "<=1k",
        2000: "<=2k",
        4000: "<=4k",
        8000: "<=8k",
        16000: "<=16k",
        32000: "<=32k",
        10**9: ">32k",
    }
    for b in BUCKETS:
        total = sum(s["hist"][b] for s in grand.values())
        print(f"  {labels[b]:>7}: {total:,}", file=sys.stderr)


if __name__ == "__main__":
    main()
