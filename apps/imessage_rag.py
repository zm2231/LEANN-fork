"""Compatibility shim for the migrated iMessage source."""

from __future__ import annotations


def main() -> None:
    raise SystemExit(
        "apps.imessage_rag has moved to the source registry. "
        "Use `leann index --source imessage <index-name>` or the deprecated "
        "`leann index-imessage` alias."
    )


if __name__ == "__main__":
    main()
