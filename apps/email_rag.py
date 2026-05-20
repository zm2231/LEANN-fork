"""Compatibility shim for the migrated Apple Mail source."""

from __future__ import annotations


def main() -> None:
    raise SystemExit(
        "apps.email_rag has moved to the source registry. "
        "Use `leann index --source apple-mail <index-name>` or the deprecated "
        "`leann index-email` alias."
    )


if __name__ == "__main__":
    main()
