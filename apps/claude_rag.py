"""Compatibility shim for the migrated Claude export source."""

from __future__ import annotations


def main() -> None:
    raise SystemExit(
        "apps.claude_rag has moved to the source registry. "
        "Use `leann index --source claude-export <index-name>` or the deprecated "
        "`leann index-claude` alias."
    )


if __name__ == "__main__":
    main()
