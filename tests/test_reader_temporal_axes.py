import json
import sqlite3
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from apps.chatgpt_data.chatgpt_reader import ChatGPTReader  # noqa: E402
from apps.claude_data.claude_reader import ClaudeReader  # noqa: E402
from apps.email_data.LEANN_email_reader import EmlxReader  # noqa: E402
from apps.history_data.history import ChromeHistoryReader  # noqa: E402
from apps.history_data.wechat_history import WeChatHistoryReader  # noqa: E402
from apps.imessage_data.imessage_reader import IMessageReader  # noqa: E402
from scripts import build_eval_corpus  # noqa: E402


def _chrome_time(value: datetime) -> int:
    return int((value.timestamp() + 11644473600) * 1_000_000)


def _cocoa_ns(value: datetime) -> int:
    cocoa_epoch = datetime(2001, 1, 1, tzinfo=timezone.utc)
    return int((value.timestamp() - cocoa_epoch.timestamp()) * 1_000_000_000)


def test_email_reader_emits_temporal_axes(tmp_path):
    emlx = tmp_path / "1.emlx"
    emlx.write_text(
        "1\n"
        "From: alice@example.com\n"
        "To: bob@example.com\n"
        "Subject: Temporal test\n"
        "Date: Tue, 12 May 2026 10:00:00 +0000\n"
        "X-Last-Modified: Tue, 12 May 2026 11:00:00 +0000\n"
        "\n"
        "Body text\n",
        encoding="utf-8",
    )

    docs = EmlxReader().load_data(str(tmp_path), max_count=1)

    metadata = docs[0].metadata
    assert metadata["source_type"] == "email"
    assert metadata["created_at"] == "2026-05-12T10:00:00+00:00"
    assert metadata["event_time"] == "2026-05-12T10:00:00+00:00"
    assert metadata["modified_at"] == "2026-05-12T11:00:00+00:00"


def test_imessage_reader_emits_temporal_axes(tmp_path):
    conn = sqlite3.connect(tmp_path / "chat.db")
    conn.executescript(
        """
        CREATE TABLE message (
            ROWID INTEGER PRIMARY KEY,
            text TEXT,
            date INTEGER,
            date_edited INTEGER,
            is_from_me INTEGER,
            service TEXT,
            handle_id INTEGER
        );
        CREATE TABLE chat (ROWID INTEGER PRIMARY KEY, chat_identifier TEXT, display_name TEXT);
        CREATE TABLE handle (ROWID INTEGER PRIMARY KEY, id TEXT);
        CREATE TABLE chat_message_join (chat_id INTEGER, message_id INTEGER);
        """
    )
    created = datetime(2026, 5, 12, 10, tzinfo=timezone.utc)
    edited = datetime(2026, 5, 12, 11, tzinfo=timezone.utc)
    conn.execute("INSERT INTO chat VALUES (1, 'chat-1', 'Chat One')")
    conn.execute("INSERT INTO handle VALUES (1, '+15555550123')")
    conn.execute(
        "INSERT INTO message VALUES (1, 'hello temporal', ?, ?, 0, 'iMessage', 1)",
        (_cocoa_ns(created), _cocoa_ns(edited)),
    )
    conn.execute("INSERT INTO chat_message_join VALUES (1, 1)")
    conn.commit()
    conn.close()

    docs = IMessageReader(concatenate_conversations=False).load_data(str(tmp_path))

    metadata = docs[0].metadata
    assert metadata["source_type"] == "imessage"
    assert metadata["created_at"] == "2026-05-12T10:00:00+00:00"
    assert metadata["event_time"] == "2026-05-12T10:00:00+00:00"
    assert metadata["modified_at"] == "2026-05-12T11:00:00+00:00"


def test_browser_reader_emits_temporal_axes(tmp_path):
    profile = tmp_path / "profile"
    profile.mkdir()
    conn = sqlite3.connect(profile / "History")
    conn.executescript(
        """
        CREATE TABLE urls (
            id INTEGER PRIMARY KEY,
            url TEXT,
            title TEXT,
            visit_count INTEGER,
            typed_count INTEGER,
            hidden INTEGER,
            last_visit_time INTEGER
        );
        CREATE TABLE visits (url INTEGER, visit_time INTEGER);
        """
    )
    first = datetime(2026, 5, 10, 10, tzinfo=timezone.utc)
    last = datetime(2026, 5, 12, 10, tzinfo=timezone.utc)
    conn.execute(
        "INSERT INTO urls VALUES (1, 'https://example.com/a', 'Example', 2, 0, 0, ?)",
        (_chrome_time(last),),
    )
    conn.execute("INSERT INTO visits VALUES (1, ?)", (_chrome_time(first),))
    conn.commit()
    conn.close()

    docs = ChromeHistoryReader().load_data(chrome_profile_path=str(profile), max_count=1)

    metadata = docs[0].metadata
    assert metadata["source_type"] == "browser_history"
    assert metadata["created_at"] == "2026-05-10T10:00:00+00:00"
    assert metadata["modified_at"] == "2026-05-12T10:00:00+00:00"
    assert metadata["event_time"] == "2026-05-12T10:00:00+00:00"


def test_wechat_reader_emits_temporal_axes(tmp_path):
    export_dir = tmp_path / "wechat"
    export_dir.mkdir()
    (export_dir / "alice.json").write_text(
        json.dumps(
            [
                {
                    "content": "hello temporal",
                    "message": "hello temporal",
                    "createTime": 1778580000,
                    "isSentFromSelf": True,
                }
            ]
        ),
        encoding="utf-8",
    )

    docs = WeChatHistoryReader().load_data(
        wechat_export_dir=str(export_dir), max_count=1, concatenate_messages=False
    )

    metadata = docs[0].metadata
    assert metadata["source_type"] == "wechat"
    assert metadata["created_at"] == "2026-05-12T10:00:00+00:00"
    assert metadata["event_time"] == "2026-05-12T10:00:00+00:00"


def test_chatgpt_reader_emits_temporal_axes(tmp_path):
    html = tmp_path / "chat.html"
    html.write_text("<html>placeholder</html>", encoding="utf-8")
    reader = ChatGPTReader()
    reader._parse_chatgpt_html = lambda _html: [
        {
            "title": "Temporal Chat",
            "timestamp": "2026-05-12T10:00:00Z",
            "messages": [{"role": "user", "content": "hello", "timestamp": "2026-05-12T10:00:00Z"}],
        }
    ]

    docs = reader.load_data(str(html))

    metadata = docs[0].metadata
    assert metadata["source_type"] == "chatgpt"
    assert metadata["created_at"] == "2026-05-12T10:00:00+00:00"
    assert metadata["event_time"] == "2026-05-12T10:00:00+00:00"


def test_claude_reader_emits_temporal_axes(tmp_path):
    data = [
        {
            "title": "Temporal Claude",
            "created_at": "2026-05-12T10:00:00Z",
            "messages": [{"role": "user", "content": "hello", "created_at": "2026-05-12T10:00:00Z"}],
        }
    ]
    json_path = tmp_path / "claude.json"
    json_path.write_text(json.dumps(data), encoding="utf-8")

    docs = ClaudeReader().load_data(str(json_path))

    metadata = docs[0].metadata
    assert metadata["source_type"] == "claude"
    assert metadata["created_at"] == "2026-05-12T10:00:00+00:00"
    assert metadata["event_time"] == "2026-05-12T10:00:00+00:00"


def test_eval_git_commit_reader_emits_created_modified_and_event_axes(monkeypatch):
    def fake_check_output(command, **kwargs):
        if command[:2] == ["git", "log"]:
            return (
                "abc123\x1f2026-05-12T10:00:00+00:00\x1f"
                "2026-05-12T11:00:00+00:00\x1fAlice\x1fSubject\x1fBody\x1e"
            )
        if command[:2] == ["git", "show"]:
            return "file.py\n"
        raise AssertionError(command)

    monkeypatch.setattr(subprocess, "check_output", fake_check_output)

    docs = build_eval_corpus.ingest_commits()

    metadata = docs[0].metadata
    assert metadata["source_type"] == "git_commit"
    assert metadata["created_at"] == "2026-05-12T10:00:00+00:00"
    assert metadata["event_time"] == "2026-05-12T10:00:00+00:00"
    assert metadata["modified_at"] == "2026-05-12T11:00:00+00:00"


def test_eval_slack_reader_emits_created_and_event_axes(monkeypatch):
    class FakeConnection:
        def execute(self, _query):
            return self

        def fetchall(self):
            return [
                (
                    "C1",
                    "general",
                    "1778580000.000000",
                    "U1",
                    "",
                    "hello temporal https://example.com",
                )
            ]

        def close(self):
            return None

    monkeypatch.setattr(build_eval_corpus.sqlite3, "connect", lambda _path: FakeConnection())

    docs = build_eval_corpus.ingest_slack()

    metadata = docs[0].metadata
    assert metadata["source_type"] == "slack"
    assert metadata["created_at"] == "2026-05-12T10:00:00+00:00"
    assert metadata["event_time"] == "2026-05-12T10:00:00+00:00"
