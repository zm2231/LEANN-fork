"""Test the LEANN MCP server JSON-RPC protocol handling."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "packages" / "leann-core" / "src"))
from leann import mcp
from leann.mcp import handle_request


def test_initialize():
    """MCP initialize should return server info and capabilities."""
    req = {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}
    resp = handle_request(req)
    assert resp["id"] == 1
    result = resp["result"]
    assert result["protocolVersion"] == "2024-11-05"
    assert result["serverInfo"]["name"] == "leann-mcp"
    assert "tools" in result["capabilities"]


def test_tools_list():
    """MCP tools/list should expose leann_search and leann_list."""
    req = {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}}
    resp = handle_request(req)
    tools = resp["result"]["tools"]
    names = {t["name"] for t in tools}
    assert "leann_search" in names
    assert "leann_multi_search" in names
    assert "leann_list" in names


def test_tools_list_search_schema():
    """leann_search tool must declare index_name and query as required params."""
    req = {"jsonrpc": "2.0", "id": 3, "method": "tools/list", "params": {}}
    resp = handle_request(req)
    search_tool = next(t for t in resp["result"]["tools"] if t["name"] == "leann_search")
    schema = search_tool["inputSchema"]
    assert "index_name" in schema["properties"]
    assert "query" in schema["properties"]
    assert "index_name" in schema["required"]
    assert "query" in schema["required"]
    assert "metadata_filters" in schema["properties"]
    for prop in (
        "vector_weight",
        "prefilter",
        "prefilter_threshold",
        "explain_filters",
        "diversify_by",
        "max_per_group",
    ):
        assert prop in schema["properties"]
        assert prop not in schema["required"]


def test_tools_list_multi_search_schema():
    """leann_multi_search exposes prose-recall controls."""
    req = {"jsonrpc": "2.0", "id": 31, "method": "tools/list", "params": {}}
    resp = handle_request(req)
    tool = next(t for t in resp["result"]["tools"] if t["name"] == "leann_multi_search")
    schema = tool["inputSchema"]
    assert "index_name" in schema["required"]
    assert "query" in schema["required"]
    for prop in ("extra_queries", "search_mode", "vector_weight", "metadata_filters", "fetch"):
        assert prop in schema["properties"]


def test_search_missing_params():
    """leann_search should return an error when required params are missing."""
    req = {
        "jsonrpc": "2.0",
        "id": 4,
        "method": "tools/call",
        "params": {"name": "leann_search", "arguments": {}},
    }
    resp = handle_request(req)
    text = resp["result"]["content"][0]["text"]
    assert "Error" in text or "error" in text


def test_search_missing_query():
    """leann_search should error when query is empty."""
    req = {
        "jsonrpc": "2.0",
        "id": 5,
        "method": "tools/call",
        "params": {"name": "leann_search", "arguments": {"index_name": "test", "query": ""}},
    }
    resp = handle_request(req)
    text = resp["result"]["content"][0]["text"]
    assert "Error" in text or "error" in text


def test_unknown_tool_returns_content_error():
    req = {
        "jsonrpc": "2.0",
        "id": 51,
        "method": "tools/call",
        "params": {"name": "missing_tool", "arguments": {}},
    }
    resp = handle_request(req)
    text = resp["result"]["content"][0]["text"]
    assert "unknown tool" in text


def test_jsonrpc_envelope():
    """All responses must follow JSON-RPC 2.0 format."""
    req = {"jsonrpc": "2.0", "id": 42, "method": "initialize", "params": {}}
    resp = handle_request(req)
    assert resp["jsonrpc"] == "2.0"
    assert resp["id"] == 42
    serialized = json.dumps(resp)
    parsed = json.loads(serialized)
    assert parsed == resp


def test_leann_search_uses_plural_metadata_filters(monkeypatch):
    """MCP leann_search must pass the real CLI flag name."""
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        return SimpleNamespace(returncode=0, stdout="[]", stderr="")

    monkeypatch.setattr(mcp.subprocess, "run", fake_run)
    req = {
        "jsonrpc": "2.0",
        "id": 6,
        "method": "tools/call",
        "params": {
            "name": "leann_search",
            "arguments": {
                "index_name": "idx",
                "query": "q",
                "metadata_filters": {"source_type": {"==": "slack"}},
            },
        },
    }
    resp = handle_request(req)
    assert resp["result"]["content"][0]["text"] == "[]"
    cmd = calls[0][0]
    assert any(part.startswith("--metadata-filters=") for part in cmd)
    assert not any(part.startswith("--metadata-filter=") for part in cmd)


def test_leann_search_forwards_advanced_search_controls(monkeypatch):
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        return SimpleNamespace(returncode=0, stdout="[]", stderr="")

    monkeypatch.setattr(mcp.subprocess, "run", fake_run)
    req = {
        "jsonrpc": "2.0",
        "id": 61,
        "method": "tools/call",
        "params": {
            "name": "leann_search",
            "arguments": {
                "index_name": "idx",
                "query": "q",
                "vector_weight": 0.4,
                "prefilter": "always",
                "prefilter_threshold": 0.2,
                "explain_filters": True,
                "diversify_by": "source",
                "max_per_group": 3,
            },
        },
    }
    resp = handle_request(req)
    assert resp["result"]["content"][0]["text"] == "[]"
    cmd = calls[0][0]
    assert "--vector-weight=0.4" in cmd
    assert "--prefilter=always" in cmd
    assert "--prefilter-threshold=0.2" in cmd
    assert "--explain-filters" in cmd
    assert "--diversify-by=source" in cmd
    assert "--max-per-group=3" in cmd


def test_search_sessions_uses_plural_metadata_filters(monkeypatch):
    """MCP search_sessions project/agent filters must use the real CLI flag name."""
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        return SimpleNamespace(returncode=0, stdout="[]", stderr="")

    monkeypatch.setattr(mcp.subprocess, "run", fake_run)
    req = {
        "jsonrpc": "2.0",
        "id": 7,
        "method": "tools/call",
        "params": {
            "name": "search_sessions",
            "arguments": {"query": "debug auth", "agent": "codex", "project": "cadence"},
        },
    }
    resp = handle_request(req)
    assert resp["result"]["content"][0]["text"] == "[]"
    cmd = calls[0][0]
    assert any(part.startswith("--metadata-filters=") for part in cmd)
    assert not any(part.startswith("--metadata-filter=") for part in cmd)


def test_leann_multi_search_fuses_results(monkeypatch):
    """leann_multi_search batches queries and RRF-fuses duplicate hits."""
    calls = {}

    class FakeSearcher:
        def __init__(self, index_path, enable_warmup=True):
            calls["index_path"] = index_path
            calls["enable_warmup"] = enable_warmup

        def __enter__(self):
            calls["entered"] = True
            return self

        def __exit__(self, exc_type, exc, tb):
            calls["exited"] = True

        def multi_search(self, queries, top_k=5, **kwargs):
            calls["queries"] = queries
            calls["top_k"] = top_k
            calls["kwargs"] = kwargs
            return [
                [
                    SimpleNamespace(id="a", score=0.9, text="alpha", metadata={"m": 1}),
                    SimpleNamespace(id="b", score=0.8, text="beta", metadata={}),
                ],
                [
                    SimpleNamespace(id="b", score=0.95, text="beta", metadata={}),
                    SimpleNamespace(id="c", score=0.7, text="gamma", metadata={}),
                ],
            ]

    import leann.api

    monkeypatch.setattr(mcp, "_resolve_index_path", lambda name: "/tmp/index/documents.leann")
    monkeypatch.setattr(leann.api, "LeannSearcher", FakeSearcher)

    req = {
        "jsonrpc": "2.0",
        "id": 8,
        "method": "tools/call",
        "params": {
            "name": "leann_multi_search",
            "arguments": {
                "index_name": "idx",
                "query": "primary",
                "extra_queries": ["angle"],
                "top_k": 2,
                "fetch": 2,
                "metadata_filters": {"source_type": {"==": "meeting"}},
            },
        },
    }
    resp = handle_request(req)
    payload = json.loads(resp["result"]["content"][0]["text"])

    assert calls["index_path"] == "/tmp/index/documents.leann"
    assert calls["enable_warmup"] is False
    assert calls["entered"] is True
    assert calls["exited"] is True
    assert calls["queries"] == ["primary", "angle"]
    assert calls["top_k"] == 2
    assert calls["kwargs"]["vector_weight"] == 0.3
    assert calls["kwargs"]["metadata_filters"] == {"source_type": {"==": "meeting"}}
    assert [r["id"] for r in payload["results"]] == ["b", "a"]
    assert payload["results"][0]["matched_queries"] == ["primary", "angle"]


def test_leann_multi_search_filtered_mode_uses_wide_default_fetch(monkeypatch):
    calls = {}

    class FakeSearcher:
        def __init__(self, index_path, enable_warmup=True):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            pass

        def multi_search(self, queries, top_k=5, **kwargs):
            calls["top_k"] = top_k
            calls["kwargs"] = kwargs
            return [[]]

    import leann.api

    monkeypatch.setattr(mcp, "_resolve_index_path", lambda name: "/tmp/index/documents.leann")
    monkeypatch.setattr(leann.api, "LeannSearcher", FakeSearcher)

    req = {
        "jsonrpc": "2.0",
        "id": 9,
        "method": "tools/call",
        "params": {
            "name": "leann_multi_search",
            "arguments": {
                "index_name": "idx",
                "query": "speaker-filtered query",
                "search_mode": "filtered",
            },
        },
    }
    resp = handle_request(req)
    payload = json.loads(resp["result"]["content"][0]["text"])

    assert calls["top_k"] == 50
    assert payload["limit"] == 12
    assert calls["kwargs"]["vector_weight"] == 0.3


def test_leann_multi_search_suppresses_search_stdout(monkeypatch, capfd):
    class FakeSearcher:
        def __init__(self, index_path, enable_warmup=True):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            pass

        def multi_search(self, queries, top_k=5, **kwargs):
            print("native backend noise that would corrupt stdio")
            return [[SimpleNamespace(id="a", score=1.0, text="alpha", metadata={})]]

    import leann.api

    monkeypatch.setattr(mcp, "_resolve_index_path", lambda name: "/tmp/index/documents.leann")
    monkeypatch.setattr(leann.api, "LeannSearcher", FakeSearcher)

    req = {
        "jsonrpc": "2.0",
        "id": 10,
        "method": "tools/call",
        "params": {
            "name": "leann_multi_search",
            "arguments": {"index_name": "idx", "query": "primary"},
        },
    }
    resp = handle_request(req)
    captured = capfd.readouterr()

    assert captured.out == ""
    payload = json.loads(resp["result"]["content"][0]["text"])
    assert payload["results"][0]["id"] == "a"
