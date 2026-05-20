from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from leann_sources import Chunk, SourceManifest, SourceReader, get_plugin
from leann_sources.manifest import ManifestValidationError, validate_manifest_dict
from leann_sources.transforms import (
    core_data_epoch_to_utc_iso,
    format,
    list_of_handles,
    regex_extract,
    unix_to_utc_iso,
    webkit_epoch_to_utc_iso,
)


def sample_manifest() -> dict:
    return {
        "name": "sample-source",
        "category": "messaging",
        "display_name": "Sample Source",
        "version": "0.1.0",
        "manifest_version": "1.0",
        "data": {
            "type": "sqlite",
            "default_path": "~/Library/Sample/source.db",
            "permissions": ["Full Disk Access"],
        },
        "auth": {"type": "none"},
        "fields": {
            "event_time": {
                "source": "messages.sent_at",
                "transform": "unix_to_utc_iso",
                "required": True,
            },
            "created_at": {
                "source": "messages.sent_at",
                "transform": "unix_to_utc_iso",
                "synthesized_from": "event_time",
            },
            "modified_at": {
                "source": "messages.edited_at",
                "transform": "core_data_epoch_to_utc_iso",
                "required": False,
            },
            "author": {"source": "messages.sender", "transform": "lookup(handles.id)"},
            "participant_ids": {
                "source": "threads.participants",
                "transform": "list_of_handles",
            },
            "source_type": {"value": "sample-source"},
            "source_id": {"source": "messages.id", "transform": "format('sample_{value}')"},
            "source_document_id": {
                "source": "threads.id",
                "transform": "format('thread_{value}')",
            },
            "chunk_seq": {"auto": "row_index_within_source_document_id"},
            "mentioned_urls": {
                "extraction": "regex",
                "pattern": r"https?://[^\s>]+",
            },
        },
        "importance": {
            "recency_decay": "default",
            "participant_count_weight": 0.8,
        },
        "chunking": {
            "granularity": "message",
            "max_chunk_tokens": 400,
            "overlap_tokens": 0,
        },
        "privacy": {
            "tier": "tier_3",
            "participant_consent_required": True,
        },
        "connectors": {"cross_source_match_by": ["mentioned_urls"]},
        "reader": "sample.reader:SampleReader",
    }


def test_manifest_round_trip_load_and_validate(tmp_path: Path):
    path = tmp_path / "manifest.yaml"
    path.write_text(yaml.safe_dump(sample_manifest(), sort_keys=False), encoding="utf-8")

    manifest = SourceManifest.load(path)

    assert manifest.name == "sample-source"
    assert manifest.category == "messaging"
    assert manifest.path == path
    assert manifest.fields["event_time"]["transform"] == "unix_to_utc_iso"
    assert manifest.to_dict()["fields"]["source_id"]["transform"] == "format('sample_{value}')"


def test_manifest_rejects_missing_required_fields():
    data = sample_manifest()
    data.pop("fields")

    with pytest.raises(ManifestValidationError, match="'fields' is a required property"):
        validate_manifest_dict(data)


def test_manifest_rejects_unknown_signals_field():
    data = sample_manifest()
    data["fields"]["published_at"] = {"source": "messages.published_at"}

    with pytest.raises(ManifestValidationError, match="unknown SIGNALS field 'published_at'"):
        validate_manifest_dict(data)


def test_manifest_rejects_malformed_transform():
    data = sample_manifest()
    data["fields"]["source_id"]["transform"] = "format('sample')"

    with pytest.raises(ManifestValidationError, match="format transform must include"):
        validate_manifest_dict(data)


def test_manifest_rejects_unknown_transform():
    data = sample_manifest()
    data["fields"]["event_time"]["transform"] = "local_time"

    with pytest.raises(ManifestValidationError, match="unknown transform 'local_time'"):
        validate_manifest_dict(data)


def test_public_exports_and_transforms():
    assert get_plugin().name == "sources"
    assert SourceReader is not None
    assert Chunk(text="hello", metadata={"source_type": "document"}).text == "hello"
    assert unix_to_utc_iso(0) == "1970-01-01T00:00:00+00:00"
    assert core_data_epoch_to_utc_iso(0) == "2001-01-01T00:00:00+00:00"
    assert webkit_epoch_to_utc_iso(0) == "1601-01-01T00:00:00+00:00"
    assert regex_extract("see https://example.com", r"https?://\S+") == ["https://example.com"]
    assert format("id_{value}", 7) == "id_7"
    assert list_of_handles("alice, bob;carol") == ["alice", "bob", "carol"]
