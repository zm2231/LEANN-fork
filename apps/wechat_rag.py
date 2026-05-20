"""Compatibility shim for the migrated WeChat source."""

from __future__ import annotations


def main() -> None:
    raise SystemExit(
        "apps.wechat_rag has moved to the source registry. "
        "Use `leann index --source wechat <index-name>` or the deprecated "
        "`leann index-wechat` alias."
    )


if __name__ == "__main__":
    main()
