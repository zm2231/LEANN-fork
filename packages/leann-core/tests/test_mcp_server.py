from __future__ import annotations

import json
import subprocess
from types import SimpleNamespace

import leann.api
import leann.mcp as mcp


class _FakeSearcher:
    seen: dict[str, object] = {}

    def __init__(self, *, index_path: str, enable_warmup: bool):
        self.index_path = index_path
        self.enable_warmup = enable_warmup

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def search(self, query: str, **kwargs):
        type(self).seen = {
            "index_path": self.index_path,
            "enable_warmup": self.enable_warmup,
            "query": query,
            "kwargs": kwargs,
        }
        return [
            SimpleNamespace(
                id="p1",
                score=0.75,
                text="direct mcp result",
                metadata={"source": "doc.md"},
            )
        ]

    def facets(self, fields, max_values_per_field=20):
        type(self).seen = {
            "index_path": self.index_path,
            "enable_warmup": self.enable_warmup,
            "fields": fields,
            "max_values_per_field": max_values_per_field,
        }
        return {field: {"value": 2} for field in fields}


def test_leann_search_uses_python_api_not_cli_subprocess(monkeypatch):
    monkeypatch.setattr(mcp, "_resolve_index_path", lambda name: f"/tmp/{name}/documents.leann")
    monkeypatch.setattr(leann.api, "LeannSearcher", _FakeSearcher)

    def fail_subprocess_run(*args, **kwargs):
        raise AssertionError("leann_search must not shell out through the CLI")

    monkeypatch.setattr(subprocess, "run", fail_subprocess_run)

    response = mcp.handle_request(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "leann_search",
                "arguments": {
                    "index_name": "docs",
                    "query": "routing bug",
                    "top_k": 3,
                    "complexity": 16,
                    "prefilter": "always",
                    "metadata_filters": {"source_type": {"==": "slack"}},
                },
            },
        }
    )

    assert response["id"] == 1
    text = response["result"]["content"][0]["text"]
    payload = json.loads(text)
    assert payload == [
        {
            "id": "p1",
            "score": 0.75,
            "text": "direct mcp result",
            "metadata": {"source": "doc.md"},
        }
    ]
    assert _FakeSearcher.seen == {
        "index_path": "/tmp/docs/documents.leann",
        "enable_warmup": False,
        "query": "routing bug",
        "kwargs": {
            "top_k": 3,
            "complexity": 16,
            "metadata_filters": {"source_type": {"==": "slack"}},
            "prefilter": "always",
            "explain_filters": False,
        },
    }


def test_leann_inspect_reports_resolved_index_and_base_dir(monkeypatch, tmp_path):
    base_dir = tmp_path / "base"
    base_dir.mkdir()
    index_prefix = tmp_path / "documents.leann"
    meta_path = tmp_path / "documents.leann.meta.json"
    meta_path.write_text(
        json.dumps(
            {
                "backend_name": "flat",
                "embedding_model": "test-model",
                "embedding_options": {"api_key": "secret-key"},
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(mcp, "_base_dir", str(base_dir))
    monkeypatch.setattr(mcp, "_resolve_index_path", lambda name: str(index_prefix))

    response = mcp.handle_request(
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {
                "name": "leann_inspect",
                "arguments": {"index_name": "docs"},
            },
        }
    )

    payload = json.loads(response["result"]["content"][0]["text"])
    assert payload["index_name"] == "docs"
    assert payload["index_path"] == str(index_prefix)
    assert payload["base_dir"] == str(base_dir)
    assert payload["metadata"]["backend_name"] == "flat"
    assert payload["metadata"]["embedding_options"]["api_key"] == "[redacted]"


def test_leann_inspect_rejects_missing_explicit_index_path(tmp_path):
    response = mcp.handle_request(
        {
            "jsonrpc": "2.0",
            "id": 22,
            "method": "tools/call",
            "params": {
                "name": "leann_inspect",
                "arguments": {"index_path": str(tmp_path / "missing")},
            },
        }
    )

    text = response["result"]["content"][0]["text"]
    assert "Error: index_path metadata file not found" in text


def test_leann_inspect_unknown_index_name_returns_fast_error(monkeypatch):
    class FakeCLI:
        def _find_all_matching_indexes(self, index_name):
            return []

    monkeypatch.setattr("leann.cli.LeannCLI", FakeCLI)

    response = mcp.handle_request(
        {
            "jsonrpc": "2.0",
            "id": 23,
            "method": "tools/call",
            "params": {
                "name": "leann_inspect",
                "arguments": {"index_name": "missing"},
            },
        }
    )

    assert response["result"]["content"][0]["text"] == "Error: Index not found: missing"


def test_leann_inspect_rejects_ambiguous_index_name(monkeypatch, tmp_path):
    first = tmp_path / "first" / ".leann" / "indexes" / "shared"
    second = tmp_path / "second" / ".leann" / "indexes" / "shared"

    class FakeCLI:
        def _find_all_matching_indexes(self, index_name):
            return [
                {"index_dir": first},
                {"index_dir": second},
            ]

    monkeypatch.setattr("leann.cli.LeannCLI", FakeCLI)

    response = mcp.handle_request(
        {
            "jsonrpc": "2.0",
            "id": 26,
            "method": "tools/call",
            "params": {
                "name": "leann_inspect",
                "arguments": {"index_name": "shared"},
            },
        }
    )

    text = response["result"]["content"][0]["text"]
    assert "Error: Index name is ambiguous: shared" in text
    assert str(first / "documents.leann") in text
    assert str(second / "documents.leann") in text


def test_leann_inspect_resolves_app_format_index_name(monkeypatch, tmp_path):
    project = tmp_path / "project"
    app_dir = project / "app-index"
    app_dir.mkdir(parents=True)
    (app_dir / "custom.leann.meta.json").write_text(
        json.dumps({"backend_name": "flat"}),
        encoding="utf-8",
    )

    monkeypatch.chdir(project)

    response = mcp.handle_request(
        {
            "jsonrpc": "2.0",
            "id": 27,
            "method": "tools/call",
            "params": {
                "name": "leann_inspect",
                "arguments": {"index_name": "custom"},
            },
        }
    )

    payload = json.loads(response["result"]["content"][0]["text"])
    assert payload["index_path"] == str(app_dir / "custom.leann")
    assert payload["metadata"]["backend_name"] == "flat"


def test_leann_inspect_base_dir_bounds_index_discovery(monkeypatch, tmp_path):
    project = tmp_path / "project"
    app_dir = project / "app-index"
    app_dir.mkdir(parents=True)
    (app_dir / "custom.leann.meta.json").write_text(
        json.dumps({"backend_name": "flat"}),
        encoding="utf-8",
    )

    class FakeCLI:
        def _find_all_matching_indexes(self, index_name):
            raise AssertionError("base-dir MCP resolution must not scan all projects")

        def _iter_app_meta_files(self, project_path):
            return [app_dir / "custom.leann.meta.json"]

    monkeypatch.setattr(mcp, "_base_dir", str(project))
    monkeypatch.setattr("leann.cli.LeannCLI", FakeCLI)

    response = mcp.handle_request(
        {
            "jsonrpc": "2.0",
            "id": 29,
            "method": "tools/call",
            "params": {
                "name": "leann_inspect",
                "arguments": {"index_name": "custom"},
            },
        }
    )

    payload = json.loads(response["result"]["content"][0]["text"])
    assert payload["index_path"] == str(app_dir / "custom.leann")


def test_leann_search_accepts_explicit_index_path_directory(monkeypatch, tmp_path):
    index_dir = tmp_path / "explicit-index"
    index_dir.mkdir()
    (index_dir / "documents.leann.meta.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(leann.api, "LeannSearcher", _FakeSearcher)

    response = mcp.handle_request(
        {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "leann_search",
                "arguments": {
                    "index_path": str(index_dir),
                    "query": "direct path",
                },
            },
        }
    )

    assert response["id"] == 3
    assert _FakeSearcher.seen["index_path"] == str(index_dir / "documents.leann")


def test_leann_list_returns_structured_index_records(monkeypatch, tmp_path):
    project = tmp_path / "project"
    index_dir = project / ".leann" / "indexes" / "docs"
    index_dir.mkdir(parents=True)
    (index_dir / "documents.leann.meta.json").write_text(
        json.dumps({"backend_name": "flat", "embedding_model": "test-model"}),
        encoding="utf-8",
    )
    (index_dir / "documents.leann.passages.jsonl").write_text("a\nb\n", encoding="utf-8")

    class FakeCLI:
        def _registered_project_paths(self):
            return [project]

    monkeypatch.chdir(project)
    monkeypatch.setattr("leann.cli.LeannCLI", FakeCLI)

    response = mcp.handle_request(
        {
            "jsonrpc": "2.0",
            "id": 24,
            "method": "tools/call",
            "params": {"name": "leann_list", "arguments": {}},
        }
    )

    payload = json.loads(response["result"]["content"][0]["text"])
    assert payload["indexes"] == [
        {
            "name": "docs",
            "index_path": str(index_dir / "documents.leann"),
            "project_path": str(project),
            "backend": "flat",
            "embedding_model": "test-model",
            "passage_count": 2,
        }
    ]


def test_leann_list_includes_app_format_indexes(monkeypatch, tmp_path):
    project = tmp_path / "project"
    app_dir = project / "app-index"
    app_dir.mkdir(parents=True)
    (app_dir / "custom.leann.meta.json").write_text(
        json.dumps({"backend_name": "flat", "embedding_model": "test-model"}),
        encoding="utf-8",
    )
    (app_dir / "custom.leann.passages.jsonl").write_text("a\nb\n", encoding="utf-8")

    class FakeCLI:
        def _registered_project_paths(self):
            return [project]

        def _iter_app_meta_files(self, project_path):
            return [app_dir / "custom.leann.meta.json"]

    monkeypatch.chdir(project)
    monkeypatch.setattr("leann.cli.LeannCLI", FakeCLI)

    response = mcp.handle_request(
        {
            "jsonrpc": "2.0",
            "id": 28,
            "method": "tools/call",
            "params": {"name": "leann_list", "arguments": {}},
        }
    )

    payload = json.loads(response["result"]["content"][0]["text"])
    assert payload["indexes"] == [
        {
            "name": "custom",
            "index_path": str(app_dir / "custom.leann"),
            "project_path": str(project),
            "backend": "flat",
            "embedding_model": "test-model",
            "passage_count": 2,
        }
    ]


def test_leann_list_honors_mcp_base_dir(monkeypatch, tmp_path):
    base_dir = tmp_path / "base"
    index_dir = base_dir / ".leann" / "indexes" / "base-docs"
    index_dir.mkdir(parents=True)
    (index_dir / "documents.leann.meta.json").write_text(
        json.dumps({"backend_name": "flat"}),
        encoding="utf-8",
    )

    monkeypatch.setattr(mcp, "_base_dir", str(base_dir))
    monkeypatch.chdir(tmp_path)

    response = mcp.handle_request(
        {
            "jsonrpc": "2.0",
            "id": 25,
            "method": "tools/call",
            "params": {"name": "leann_list", "arguments": {}},
        }
    )

    payload = json.loads(response["result"]["content"][0]["text"])
    assert payload["base_dir"] == str(base_dir)
    assert any(record["name"] == "base-docs" for record in payload["indexes"])


def test_leann_facets_uses_python_api(monkeypatch):
    monkeypatch.setattr(mcp, "_resolve_index_path", lambda name: f"/tmp/{name}/documents.leann")
    monkeypatch.setattr(leann.api, "LeannSearcher", _FakeSearcher)

    response = mcp.handle_request(
        {
            "jsonrpc": "2.0",
            "id": 4,
            "method": "tools/call",
            "params": {
                "name": "leann_facets",
                "arguments": {
                    "index_name": "docs",
                    "fields": ["source_type", "project_id"],
                    "max_values_per_field": 5,
                },
            },
        }
    )

    payload = json.loads(response["result"]["content"][0]["text"])
    assert payload["index_path"] == "/tmp/docs/documents.leann"
    assert payload["facets"] == {"source_type": {"value": 2}, "project_id": {"value": 2}}
    assert _FakeSearcher.seen == {
        "index_path": "/tmp/docs/documents.leann",
        "enable_warmup": False,
        "fields": ["source_type", "project_id"],
        "max_values_per_field": 5,
    }
