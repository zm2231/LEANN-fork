"""Compatibility shim for the migrated browser source."""

from __future__ import annotations


def main() -> None:
    raise SystemExit(
        "apps.browser_rag has moved to the source registry. "
        "Use `leann index --source chrome <index-name>` or the deprecated "
        "`leann index-browser` alias."
    )


if __name__ == "__main__":
    main()
