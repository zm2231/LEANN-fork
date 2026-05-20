"""Compatibility shim for the migrated ChatGPT export source."""

from __future__ import annotations


def main() -> None:
    raise SystemExit(
        "apps.chatgpt_rag has moved to the source registry. "
        "Use `leann index --source chatgpt-export <index-name>` or the deprecated "
        "`leann index-chatgpt` alias."
    )


if __name__ == "__main__":
    main()
