import asyncio
import json

from leann.cli import LeannCLI


def test_cli_ask_accepts_positional_query(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    cli = LeannCLI()
    parser = cli.create_parser()

    args = parser.parse_args(["ask", "my-docs", "Where are prompts configured?"])

    assert args.command == "ask"
    assert args.index_name == "my-docs"
    assert args.query == "Where are prompts configured?"


def test_cli_ask_parses_metadata_filters_flag():
    cli = LeannCLI()
    parser = cli.create_parser()

    filters_json = '{"chapter": {"<=": 5}, "genre": {"==": "fiction"}}'
    args = parser.parse_args(
        ["ask", "my-docs", "Summarize early chapters", "--metadata-filters", filters_json]
    )

    assert args.command == "ask"
    assert args.metadata_filters == filters_json
    # The raw string parses to the expected dict so downstream consumers can rely on it.
    assert json.loads(args.metadata_filters) == {
        "chapter": {"<=": 5},
        "genre": {"==": "fiction"},
    }


def test_cli_ask_metadata_filters_default_is_none():
    cli = LeannCLI()
    parser = cli.create_parser()

    args = parser.parse_args(["ask", "my-docs", "any query"])

    assert args.metadata_filters is None


def test_cli_resolve_index_path_accepts_explicit_prefix(tmp_path):
    index_prefix = tmp_path / "documents.leann"
    (tmp_path / "documents.leann.meta.json").write_text("{}", encoding="utf-8")

    cli = LeannCLI()

    assert cli._resolve_index_path(str(index_prefix), quiet=True) == str(index_prefix)


def test_cli_resolve_index_path_accepts_explicit_meta_file(tmp_path):
    meta_path = tmp_path / "documents.leann.meta.json"
    meta_path.write_text("{}", encoding="utf-8")

    cli = LeannCLI()

    assert cli._resolve_index_path(str(meta_path), quiet=True) == str(tmp_path / "documents.leann")


def test_cli_resolve_index_path_accepts_explicit_index_directory(tmp_path):
    (tmp_path / "documents.leann.meta.json").write_text("{}", encoding="utf-8")

    cli = LeannCLI()

    assert cli._resolve_index_path(str(tmp_path), quiet=True) == str(tmp_path / "documents.leann")


def test_cli_resolve_index_path_skips_incomplete_current_index(tmp_path, monkeypatch, capsys):
    current = tmp_path / "current"
    registered = tmp_path / "registered"
    current_index = current / ".leann" / "indexes" / "sessions"
    registered_index = registered / ".leann" / "indexes" / "sessions"
    current_index.mkdir(parents=True)
    registered_index.mkdir(parents=True)
    (registered_index / "documents.leann.meta.json").write_text("{}", encoding="utf-8")

    monkeypatch.chdir(current)
    cli = LeannCLI()
    monkeypatch.setattr(cli, "_registered_project_paths", lambda: [current, registered])

    resolved = cli._resolve_index_path("sessions", quiet=True)

    assert resolved == str(registered_index / "documents.leann")
    assert "registered" in capsys.readouterr().err


def test_cli_ask_rejects_invalid_metadata_filters_json(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)

    cli = LeannCLI()
    parser = cli.create_parser()
    # Set up an empty index dir so the existence check passes and we reach the JSON parser.
    index_dir = tmp_path / ".leann" / "indexes" / "my-docs"
    index_dir.mkdir(parents=True)
    (index_dir / "documents.leann.meta.json").write_text("{}")

    args = parser.parse_args(["ask", "my-docs", "any query", "--metadata-filters", "not-json"])

    asyncio.run(cli.ask_questions(args))

    captured = capsys.readouterr()
    assert "--metadata-filters is not valid JSON" in captured.out


def test_cli_ask_rejects_non_object_metadata_filters(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)

    cli = LeannCLI()
    parser = cli.create_parser()
    index_dir = tmp_path / ".leann" / "indexes" / "my-docs"
    index_dir.mkdir(parents=True)
    (index_dir / "documents.leann.meta.json").write_text("{}")

    # A valid JSON value that is not an object (dict) — must be rejected.
    args = parser.parse_args(["ask", "my-docs", "any query", "--metadata-filters", "[1, 2, 3]"])

    asyncio.run(cli.ask_questions(args))

    captured = capsys.readouterr()
    assert "--metadata-filters must be a JSON object" in captured.out
