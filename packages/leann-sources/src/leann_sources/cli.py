"""CLI namespace for registered LEANN sources."""

from __future__ import annotations

import argparse
import importlib
import importlib.util
from pathlib import Path
from typing import Any

from leann_sources.base import SourceReader
from leann_sources.manifest import SourceManifest
from leann_sources.readers import (
    APISourceReader,
    ExportZipSourceReader,
    FilesystemSourceReader,
    SQLiteSourceReader,
)
from leann_sources.registry import SourceRegistryEntry, build_registry


class SourceCLI:
    def __init__(self, sources_root: str | Path):
        self.sources_root = Path(sources_root)

    def register(self, subparsers: Any) -> None:
        sources_parser = subparsers.add_parser("sources", help="Manage registered ingest sources")
        sources_subparsers = sources_parser.add_subparsers(dest="sources_command")

        list_parser = sources_subparsers.add_parser("list", help="List registered sources")
        list_parser.add_argument("--category", type=str, default=None)

        info_parser = sources_subparsers.add_parser("info", help="Show source manifest details")
        info_parser.add_argument("name")

        install_parser = sources_subparsers.add_parser("install", help="Install source config")
        install_parser.add_argument("name")
        install_parser.add_argument("--config-dir", type=str, default=None)

        connect_parser = sources_subparsers.add_parser("connect", help="Connect source auth")
        connect_parser.add_argument("name")

        validate_parser = sources_subparsers.add_parser("validate", help="Validate source access")
        validate_parser.add_argument("name")

        index_parser = sources_subparsers.add_parser("index", help="Index a registered source")
        index_parser.add_argument("name")
        index_parser.add_argument("index_name", nargs="?")
        index_parser.add_argument("--dry-run", action="store_true")
        self._add_index_options(index_parser)

        unified_index = subparsers.add_parser("index", help="Index a registered source")
        unified_index.add_argument("--source", required=True)
        unified_index.add_argument("index_name", nargs="?")
        unified_index.add_argument("--dry-run", action="store_true")
        self._add_index_options(unified_index)

    def _add_index_options(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--embedding-model", type=str, default="facebook/contriever", help="Embedding model"
        )
        parser.add_argument(
            "--embedding-mode",
            type=str,
            default="sentence-transformers",
            choices=["sentence-transformers", "openai", "mlx", "ollama"],
            help="Embedding backend",
        )
        parser.add_argument(
            "--embedding-host", type=str, default=None, help="Ollama embedding host"
        )
        parser.add_argument(
            "--embedding-api-base",
            type=str,
            default=None,
            help="OpenAI-compatible embedding base URL",
        )
        parser.add_argument("--embedding-api-key", type=str, default=None, help="Embedding API key")
        parser.add_argument(
            "--max-count",
            type=int,
            default=-1,
            help="Max items to index (-1 = unlimited, default)",
        )
        parser.add_argument(
            "--no-recompute",
            action="store_true",
            help="Disable embedding recomputation (stores full embeddings)",
        )

    def handle(self, args: argparse.Namespace, core_cli: Any = None) -> bool | Any:
        if args.command == "sources":
            command = getattr(args, "sources_command", None)
            if command == "list":
                self.list_sources(category=args.category)
            elif command == "info":
                self.info(args.name)
            elif command == "install":
                self.install(args.name, config_dir=args.config_dir)
            elif command == "connect":
                self.connect(args.name)
            elif command == "validate":
                self.validate(args.name)
            elif command == "index":
                return self.index(
                    args.name,
                    index_name=args.index_name,
                    dry_run=getattr(args, "dry_run", False),
                    core_cli=core_cli,
                    args=args,
                )
            else:
                raise SystemExit("missing sources subcommand")
            return True
        if args.command == "index" and getattr(args, "source", None):
            return self.index(
                args.source,
                index_name=args.index_name,
                dry_run=getattr(args, "dry_run", False),
                core_cli=core_cli,
                args=args,
            )
        return False

    def list_sources(self, category: str | None = None) -> None:
        registry = build_registry(self.sources_root)
        entries = [
            entry for entry in registry.entries if category is None or entry.category == category
        ]
        grouped: dict[str, list[SourceRegistryEntry]] = {}
        for entry in entries:
            grouped.setdefault(entry.category, []).append(entry)
        for category_name in sorted(grouped):
            print(f"[{category_name}]")
            for entry in grouped[category_name]:
                print(f"{entry.category}/{entry.name}\t{entry.display_name}\t{entry.data_type}")

    def info(self, name: str) -> None:
        manifest = self.load_manifest(name)
        print(f"name: {manifest.name}")
        print(f"category: {manifest.category}")
        print(f"display_name: {manifest.display_name}")
        print(f"data_type: {manifest.data['type']}")
        print(f"privacy_tier: {manifest.privacy['tier']}")
        try:
            stats = self.reader_for(manifest).stats()
        except Exception:
            stats = None
        if stats is not None:
            print(f"count: {stats.count}")

    def install(self, name: str, config_dir: str | None = None) -> None:
        manifest = self.load_manifest(name)
        config_root = (
            Path(config_dir).expanduser() if config_dir else Path.home() / ".leann" / "sources"
        )
        target = config_root / manifest.name
        target.mkdir(parents=True, exist_ok=True)
        print(f"installed {manifest.name} config at {target}")

    def connect(self, name: str) -> None:
        manifest = self.load_manifest(name)
        reader = self.reader_for(manifest)
        auth_setup = getattr(reader, "auth_setup", None)
        if callable(auth_setup):
            auth_setup()
            print(f"connected {manifest.name}")
        elif (manifest.auth or {}).get("type", "none") == "none":
            print(f"{manifest.name}: no auth required")
        else:
            print(f"{manifest.name}: no interactive auth setup available")

    def validate(self, name: str) -> None:
        manifest = self.load_manifest(name)
        report = self.reader_for(manifest).validate()
        status = "ok" if report.ok else report.status
        print(f"{manifest.name}: {status}")
        for error in report.errors:
            print(f"error: {error}")

    def index(
        self,
        name: str,
        index_name: str | None = None,
        *,
        dry_run: bool = False,
        core_cli: Any = None,
        args: argparse.Namespace | None = None,
    ) -> bool | Any:
        manifest = self.load_manifest(name)
        if args is not None:
            self._apply_arg_overrides(manifest, args)
        reader = self.reader_for(manifest)
        target = index_name or manifest.name
        if dry_run:
            chunks = self._limited_chunks(reader, args)
            print(f"{manifest.name}: dry-run emitted {len(chunks)} chunks for {target}")
            return True
        if core_cli is None or args is None:
            print(f"{manifest.name}: source indexing target {target} registered")
            return True
        return self._build_source_index(core_cli, args, reader, manifest.name, target)

    async def _build_source_index(
        self,
        core_cli: Any,
        args: argparse.Namespace,
        reader: SourceReader,
        source_name: str,
        target: str,
    ) -> bool:
        from llama_index.core import Document

        chunks = self._limited_chunks(reader, args)
        documents = [Document(text=chunk.text, metadata=chunk.metadata) for chunk in chunks]
        args.index_name = target
        print(f"Loaded {len(documents)} {source_name} source chunks")
        await core_cli._build_index_from_documents(args, documents)
        return True

    def _limited_chunks(
        self, reader: SourceReader, args: argparse.Namespace | None = None
    ) -> list[Any]:
        chunks = list(reader.iter_chunks())
        max_count = getattr(args, "max_count", None)
        if max_count is None or max_count < 0:
            return chunks
        return chunks[:max_count]

    def _apply_arg_overrides(self, manifest: SourceManifest, args: argparse.Namespace) -> None:
        if getattr(args, "export_dir", None):
            manifest.data["default_path"] = args.export_dir
        if getattr(args, "export_path", None):
            manifest.data["default_path"] = args.export_path
        browser = getattr(args, "browser", None)
        if manifest.name == "chrome" and browser == "brave":
            manifest.data["default_path"] = (
                "~/Library/Application Support/BraveSoftware/Brave-Browser/Default"
            )

    def load_manifest(self, name: str) -> SourceManifest:
        entry = self.entry_for(name)
        return SourceManifest.load(self.sources_root / entry.manifest_path)

    def entry_for(self, name: str) -> SourceRegistryEntry:
        registry = build_registry(self.sources_root)
        for entry in registry.entries:
            if entry.name == name:
                return entry
        raise SystemExit(f"unknown source: {name}")

    def reader_for(self, manifest: SourceManifest) -> SourceReader:
        if manifest.reader:
            return self._custom_reader_for(manifest)
        reader_type = manifest.data["type"]
        if reader_type == "sqlite":
            return SQLiteSourceReader(manifest)
        if reader_type == "filesystem":
            return FilesystemSourceReader(manifest)
        if reader_type == "api":
            return APISourceReader(manifest)
        if reader_type == "export_zip":
            return ExportZipSourceReader(manifest)
        raise SystemExit(f"unsupported source data type: {reader_type}")

    def _custom_reader_for(self, manifest: SourceManifest) -> SourceReader:
        reader_ref = manifest.reader or ""
        module_name, class_name = reader_ref.rsplit(".", 1)
        if module_name == "reader" and manifest.path is not None:
            reader_path = manifest.path.parent / "reader.py"
            module_key = f"_leann_source_{manifest.name}_reader"
            spec = importlib.util.spec_from_file_location(module_key, reader_path)
            if spec is None or spec.loader is None:
                raise SystemExit(f"cannot load source reader: {reader_path}")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        else:
            module = importlib.import_module(module_name)
        reader_cls = getattr(module, class_name)
        return reader_cls(manifest)
